"""
appel_ollama.py

Appelle le modele de langage local UNIQUEMENT sur les colonnes qui en ont
besoin (voir classification_regles.py), fusionne avec les colonnes deja
classees automatiquement, puis fusionne le risque de fuite et les
doublons potentiels calcules objectivement (detection_fuite.py,
detection_doublons.py). Sauvegarde le resultat complet dans
resultats/classification_finale.json.

Points de robustesse integres, valables pour tout deploiement de ce
pipeline (pas propres a un fichier de donnees particulier) :
- client Ollama explicitement pointe sur 127.0.0.1 plutot que "localhost"
  (evite un probleme de resolution IPv6 frequent sous Windows) ;
- num_ctx augmente au-dela du defaut Ollama, qui tronque silencieusement
  les reponses des qu'un lot de colonnes avec profil complet depasse un
  petit contexte ;
- squelette JSON complet dans le prompt (voir prompt.py) pour eviter que
  le modele ne s'arrete apres la premiere colonne ;
- parsing tolerant (extraction du bloc JSON, deballage d'enveloppe,
  comparaison de noms insensible a la casse), avec affichage de la
  reponse brute en cas d'echec pour faciliter le diagnostic ;
- normalisation des variantes de role_principal et des cles de champ hors
  schema (un modele local peut recopier un type de stockage brut au lieu
  de traduire vers la valeur de sortie attendue, ou legerement mal
  orthographier un nom de champ).
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODEL_NAME, OLLAMA_BASE_URL, DATA_GENERATED_DIR, RESULTATS_DIR, CLASSIFICATION_PATH  # noqa: E402
from prompt import SYSTEM_PROMPT, build_batch_prompt, make_batches, BATCH_SIZE  # noqa: E402
from classification_regles import split_columns  # noqa: E402
from detection_fuite import detect_leakage  # noqa: E402
from detection_doublons import detect_duplicates, DUPLICATE_THRESHOLD  # noqa: E402

NUM_CTX = 8192
MAX_RETRIES = 2

INPUT_PATH = DATA_GENERATED_DIR / "dictionnaire_enrichi.json"
OUTPUT_PATH = CLASSIFICATION_PATH

VALID_ROLES = {"categoriel", "numerique", "date"}
ROLE_ALIASES = {
    "categoriel_ou_texte": "categoriel",
    "texte": "categoriel",
    "identifiant": "categoriel",  # role retire du schema - fusionne avec categoriel
    "taux": "numerique",
    "montant": "numerique",
    "pourcentage": "numerique",
}

# Le modele produit parfois des variantes legerement mal orthographiees des
# NOMS DE CHAMPS eux-memes (pas seulement des valeurs) - ex.
# "peut_etre_denominate_ratio" au lieu de "peut_etre_denominateur_ratio".
# Sans ce rattrapage, la vraie reponse du modele finit dans une cle
# fantome et le champ officiel reste a null (perte de donnees silencieuse).
FIELD_KEY_ALIASES = {
    "peut_etre_denominate_ratio": "peut_etre_denominateur_ratio",
    "peut_etre_denominateur": "peut_etre_denominateur_ratio",
    "denominateur_ratio": "peut_etre_denominateur_ratio",
    "deja_taux": "deja_un_taux",
    "risque_fuite": "risque_de_fuite",
    "dimension_agregation": "dimension_agregation_pertinente",
}


def normalize_field_keys(entry: dict) -> dict:
    """
    Corrige les noms de champs mal orthographies par le modele. Si le champ
    canonique est deja rempli (non null), on le garde plutot que d'ecraser
    avec la variante - le champ canonique est priorise en cas de doublon.
    """
    cleaned = dict(entry)
    for alias, canonical in FIELD_KEY_ALIASES.items():
        if alias in cleaned:
            alias_value = cleaned.pop(alias)
            if cleaned.get(canonical) is None and alias_value is not None:
                cleaned[canonical] = alias_value
    return cleaned


def call_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    import ollama
    client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={
            "temperature": 0.1,
            "num_ctx": NUM_CTX,
        },
    )
    return response["message"]["content"]


def extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise json.JSONDecodeError("Aucun bloc JSON trouve", raw, 0)


def normalize(name: str) -> str:
    return name.strip().upper().replace(" ", "").replace("-", "_")


def unwrap_if_needed(parsed: dict, expected_names: set) -> dict:
    expected_norm = {normalize(n) for n in expected_names}
    keys_norm = {normalize(k) for k in parsed.keys()}
    if keys_norm & expected_norm:
        return parsed
    if len(parsed) == 1:
        inner = list(parsed.values())[0]
        if isinstance(inner, dict):
            return inner
    return parsed


def normalize_entry(entry: dict) -> dict:
    """Corrige les variantes connues de role_principal hors schema, et
    force a null les champs incoherents avec le role_principal renvoye."""
    entry = normalize_field_keys(entry)

    role = entry.get("role_principal")
    if role in ROLE_ALIASES:
        role = ROLE_ALIASES[role]
    elif role not in VALID_ROLES and role is not None:
        role_lower = role.lower()
        if "categor" in role_lower or "identif" in role_lower or "texte" in role_lower:
            role = "categoriel"
        elif "numer" in role_lower or "taux" in role_lower or "montant" in role_lower:
            role = "numerique"


    cleaned = dict(entry)
    cleaned["role_principal"] = role
    if role != "numerique":
        cleaned["peut_etre_denominateur_ratio"] = None
        cleaned["deja_un_taux"] = None
        cleaned["risque_de_fuite"] = None
    if role != "categoriel":
        cleaned["dimension_agregation_pertinente"] = None
    return cleaned


def classify_batch(batch: list[dict]) -> dict:
    expected_names = {r["nom"] for r in batch}
    prompt = build_batch_prompt(batch)

    last_raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        raw_output = call_ollama(prompt)
        last_raw = raw_output
        try:
            parsed = extract_json(raw_output)
            parsed = unwrap_if_needed(parsed, expected_names)

            parsed_norm = {normalize(k): (k, v) for k, v in parsed.items()}
            result, missing = {}, []
            for name in expected_names:
                key = normalize(name)
                if key in parsed_norm:
                    result[name] = normalize_entry(parsed_norm[key][1])
                else:
                    missing.append(name)

            if not missing:
                return result
            print(f"  [tentative {attempt}/{MAX_RETRIES}] colonnes manquantes : {missing}")

        except json.JSONDecodeError as e:
            print(f"  [tentative {attempt}/{MAX_RETRIES}] JSON invalide : {e}")

        print("  --- reponse brute du modele (500 premiers caracteres) ---")
        print(f"  {last_raw[:500]}")
        print("  ----------------------------------------------------------")
        time.sleep(1)

    raise RuntimeError(f"Echec apres {MAX_RETRIES} tentatives sur ce lot.")


def run_full_classification() -> dict:
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    deja_classees, a_envoyer_au_llm = split_columns(records)
    print(f"{len(records)} colonnes au total")
    print(f"{len(deja_classees)} classees automatiquement (dates), sans modele de langage")
    print(f"{len(a_envoyer_au_llm)} a envoyer au modele de langage\n")

    full_classification = {r["nom"]: {k: v for k, v in r.items() if k != "nom"} for r in deja_classees}

    batches = make_batches(a_envoyer_au_llm)
    print(f"{len(batches)} lots de {BATCH_SIZE} a traiter")

    for i, batch in enumerate(batches, start=1):
        noms = [r["nom"] for r in batch]
        print(f"\nLot {i}/{len(batches)} : {noms}")
        try:
            result = classify_batch(batch)
            full_classification.update(result)
            print(f"  -> OK, {len(result)} colonnes classees")
        except RuntimeError as e:
            print(f"  -> ECHEC DEFINITIF sur ce lot : {e}")
            continue

    return full_classification


def merge_leakage_detection(classification: dict) -> dict:
    """Remplace risque_de_fuite (et ajoute correlation_avec_resultat a
    titre informatif) par un calcul statistique objectif, pour toutes les
    colonnes de role_principal == "numerique"."""
    leakage = detect_leakage()

    for nom, entry in classification.items():
        if entry.get("role_principal") == "numerique":
            info = leakage.get(nom)
            if info is not None:
                entry["risque_de_fuite"] = info["risque_de_fuite"]
                entry["correlation_avec_resultat"] = info["correlation_avec_resultat"]
            else:
                entry["risque_de_fuite"] = None
                entry["correlation_avec_resultat"] = None
        else:
            entry["risque_de_fuite"] = None
            entry["correlation_avec_resultat"] = None

    return classification


def merge_duplicate_detection(classification: dict) -> dict:
    """Ajoute une note signalant les doublons potentiels detectes par
    correlation entre colonnes. Complete la note du modele, ne la remplace pas."""
    duplicates = detect_duplicates()

    for nom, entry in classification.items():
        pairs = duplicates.get(nom)
        if not pairs:
            continue
        noms_correles = ", ".join(f"{autre} (r={corr})" for autre, corr in pairs)
        message = (
            f"Doublon potentiel (correlation >= {DUPLICATE_THRESHOLD}, a verifier) "
            f"avec : {noms_correles}."
        )
        existing_note = entry.get("note") or ""
        entry["note"] = f"{existing_note} {message}".strip()

    return classification


if __name__ == "__main__":
    classification = run_full_classification()

    print("\nCalcul du risque de fuite par correlation avec la cible...")
    classification = merge_leakage_detection(classification)

    print("Detection des doublons potentiels par correlation entre colonnes...")
    classification = merge_duplicate_detection(classification)

    RESULTATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(classification, f, ensure_ascii=False, indent=2)

    print(f"\n{len(classification)} colonnes classees au total")
    print(f"Sauvegarde dans : {OUTPUT_PATH}")
