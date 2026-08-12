"""
proposer_kpi_metier.py (Etape 3b - orchestrateur)

Propose des KPI par RAISONNEMENT METIER A PRIORI (modele de langage base
uniquement sur les definitions du dictionnaire, aucune donnee empirique)
- avant meme de generer quoi que ce soit a partir des donnees reelles
(etape 4).

Objectif : eviter que des KPI reconnus comme structurants dans le metier
concerne soient noyes ou absents du classement final simplement parce
qu'ils ne sont pas les mieux notes statistiquement sur l'echantillon
disponible.

Complementaire, pas remplacant, de la generation exhaustive (etape 4) :
les deux approches cohabitent et sont notees par la MEME grille
statistique en aval (etape 5), avec une colonne "origine" qui les
distingue - les KPI "metier" ne sont jamais silencieusement ecartes par
le seul filtre statistique (voir notation_kpi.py, parametre origine_metier).

Ne rappelle PAS le modele a chaque lancement du pipeline : ce script est
lance manuellement, comme la classification (etape 3), et son resultat
est ensuite simplement relu par generation_candidats_guides.py.

DECOUPAGE EN LOTS : envoyer un dictionnaire volumineux en un seul prompt
peut faire tourner certains modeles en boucle sans jamais renvoyer de
reponse exploitable. Meme principe que la classification (etape 3) :
traiter par petits lots (BATCH_SIZE colonnes, voir prompt_raisonnement.py)
et fusionner les propositions - au prix de ne plus pouvoir croiser deux
colonnes tombees dans des lots differents, mais avec une bien meilleure
fiabilite.

INPUT  : dictionnaire_enrichi.json (etape 2) - SEULEMENT nom+definition
OUTPUT : resultats/kpi_proposes_metier.json
"""
import json
import re
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import MODEL_NAME, OLLAMA_BASE_URL, TARGET_COL, DATA_GENERATED_DIR, KPI_METIER_PATH, RESULTATS_DIR  # noqa: E402
from prompt_raisonnement import SYSTEM_PROMPT, build_prompt, make_batches, BATCH_SIZE  # noqa: E402

DICT_ENRICHI_PATH = DATA_GENERATED_DIR / "dictionnaire_enrichi.json"
OUTPUT_PATH = KPI_METIER_PATH

NUM_CTX = 8192
MAX_RETRIES = 3

OPERATIONS_VALIDES = {"ratio", "brute", "complement", "somme_ratios"}


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
        options={"temperature": 0.2, "num_ctx": NUM_CTX},
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


def valider_colonnes(kpi: dict, noms_valides: set) -> tuple[bool, str]:
    """Garde-fou SANS modele de langage : rejette toute proposition qui
    reference une colonne inexistante (hallucination), la variable cible
    elle-meme (circulaire - c'est ce que ces KPI doivent aider a evaluer,
    pas une entree), ou une operation hors schema - jamais de confiance
    aveugle dans la reponse du modele."""
    operation = kpi.get("operation")
    if operation not in OPERATIONS_VALIDES:
        return False, f"operation inconnue ou manquante : {operation!r}"

    if operation == "ratio":
        colonnes = (kpi.get("numerateur") or []) + (kpi.get("denominateur") or [])
    elif operation in ("brute", "complement"):
        colonnes = [kpi["colonne"]] if kpi.get("colonne") else []
    else:  # somme_ratios
        colonnes = []
        for r in (kpi.get("ratios") or []):
            colonnes += (r.get("numerateur") or []) + (r.get("denominateur") or [])

    if not colonnes:
        return False, "aucune colonne renseignee pour cette operation"

    if TARGET_COL in colonnes:
        return False, f"utilise la variable cible {TARGET_COL} comme composante - circulaire, rejete"

    inconnues = [c for c in colonnes if c not in noms_valides]
    if inconnues:
        return False, f"colonne(s) inexistante(s) dans le dictionnaire : {inconnues}"

    return True, ""


def signature_kpi(kpi: dict) -> tuple:
    """Signature STRUCTURELLE (pas le nom) pour deduppliquer les
    propositions - un modele local peut boucler en fin de generation et
    repeter le meme KPI sous un nom identique ou legerement different."""
    operation = kpi.get("operation")
    if operation == "ratio":
        return (operation, tuple(sorted(kpi.get("numerateur") or [])), tuple(sorted(kpi.get("denominateur") or [])))
    if operation in ("brute", "complement"):
        return (operation, kpi.get("colonne"))
    if operation == "somme_ratios":
        parts = tuple(sorted(
            (tuple(sorted(r.get("numerateur") or [])), tuple(sorted(r.get("denominateur") or [])))
            for r in (kpi.get("ratios") or [])
        ))
        return (operation, parts)
    return (operation, kpi.get("nom"))


def dedupliquer(kpis: list[dict]) -> tuple[list[dict], int]:
    vues = set()
    uniques = []
    for kpi in kpis:
        sig = signature_kpi(kpi)
        if sig in vues:
            continue
        vues.add(sig)
        uniques.append(kpi)
    return uniques, len(kpis) - len(uniques)


def proposer_kpi_pour_lot(batch: list[dict]) -> list[dict]:
    """Traite UN lot de colonnes, avec retries - echec d'un lot n'interrompt
    pas les autres (message d'avertissement, lot ignore)."""
    prompt = build_prompt(batch)

    last_raw = None
    for attempt in range(1, MAX_RETRIES + 1):
        raw = call_ollama(prompt)
        last_raw = raw
        try:
            parsed = extract_json(raw)
            return parsed.get("kpis", [])
        except json.JSONDecodeError as e:
            print(f"    [tentative {attempt}/{MAX_RETRIES}] JSON invalide : {e}")
        time.sleep(1)

    print(f"    ECHEC sur ce lot apres {MAX_RETRIES} tentatives - ignore. "
          f"Reponse brute (300 premiers caracteres) : {last_raw[:300] if last_raw else None}")
    return []


def proposer_kpi() -> list[dict]:
    with open(DICT_ENRICHI_PATH, encoding="utf-8") as f:
        records = json.load(f)
    noms_valides = {r["nom"] for r in records}

    batches = make_batches(records)
    print(f"{len(records)} colonnes, {len(batches)} lot(s) de {BATCH_SIZE} colonnes")

    propositions = []
    for i, batch in enumerate(batches, start=1):
        noms_lot = [r["nom"] for r in batch]
        print(f"\nLot {i}/{len(batches)} : {noms_lot}")
        kpis_lot = proposer_kpi_pour_lot(batch)
        print(f"  -> {len(kpis_lot)} KPI propose(s) sur ce lot")
        propositions.extend(kpis_lot)

    if not propositions:
        raise RuntimeError("Aucune proposition exploitable sur l'ensemble des lots.")

    propositions, n_doublons = dedupliquer(propositions)

    valides, rejetees = [], []
    for kpi in propositions:
        ok, raison = valider_colonnes(kpi, noms_valides)
        if ok:
            valides.append(kpi)
        else:
            rejetees.append((kpi.get("nom", "?"), raison))

    print(f"\n{len(propositions) + n_doublons} KPI proposes ({n_doublons} doublon(s) "
          f"structurel(s) ecarte(s) - le modele repete parfois le meme KPI en fin de generation)")
    print(f"{len(valides)} valide(s) (colonnes verifiees contre le dictionnaire)")
    if rejetees:
        print(f"{len(rejetees)} rejete(s) (hallucination de colonne, cible utilisee, ou operation invalide) :")
        for nom, raison in rejetees:
            print(f"  - {nom} : {raison}")

    return valides


if __name__ == "__main__":
    kpis = proposer_kpi()

    RESULTATS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    print(f"\nSauvegarde : {OUTPUT_PATH}")
    for k in kpis:
        print(f"\n- {k['nom']} ({k['operation']})")
        print(f"  {k.get('justification_metier', '')}")
