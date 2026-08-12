"""
classification_regles.py

Classification DETERMINISTE (sans modele de langage) reduite au strict
minimum statistiquement fiable : la detection des colonnes de DATES, a
partir du dtype reellement observe dans les vraies donnees (pas une
interpretation de texte). Generique : ne depend d'aucun nom de colonne ni
categorie metier particuliere.

Tout le reste - y compris distinguer un identifiant d'une colonne
numerique ou categorielle - necessite de LIRE la definition (ex: un
numero sequentiel peut etre stocke comme un nombre mais designer un
identifiant technique d'apres sa definition). C'est un jugement
semantique, donc le travail du modele de langage (etape 3), pas de ce
script.

INPUT  : dictionnaire_enrichi.json (etape 2)
OUTPUT : rien en fichier - fonctions importees par appel_ollama.py
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DATA_GENERATED_DIR  # noqa: E402

INPUT_PATH = DATA_GENERATED_DIR / "dictionnaire_enrichi.json"


def rule_based_role(record: dict) -> str | None:
    """Retourne "date" si le profil empirique detecte un dtype datetime,
    sinon None (la colonne doit etre tranchee par le modele de langage)."""
    if record["profil_empirique"].get("type_empirique") == "date":
        return "date"
    return None


def split_columns(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Separe les colonnes en :
    - deja_classees : role_principal == "date" -> aucun sous-champ a
      completer, donc rien a envoyer au modele
    - a_envoyer_au_llm : tout le reste (identifiant/categoriel/numerique
      a trancher par le modele)
    """
    deja_classees, a_envoyer_au_llm = [], []

    for r in records:
        role = rule_based_role(r)
        if role == "date":
            deja_classees.append({
                "nom": r["nom"],
                "role_principal": "date",
                "peut_etre_denominateur_ratio": None,
                "deja_un_taux": None,
                "risque_de_fuite": None,
                "dimension_agregation_pertinente": None,
                "note": "Classe automatiquement (dtype datetime detecte empiriquement), non envoye au modele.",
            })
        else:
            a_envoyer_au_llm.append(r)

    return deja_classees, a_envoyer_au_llm


if __name__ == "__main__":
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    deja, a_llm = split_columns(records)

    print(f"{len(records)} colonnes au total")
    print(f"{len(deja)} classees automatiquement (dates uniquement) :")
    for r in deja:
        print(f"  {r['nom']:30s} -> {r['role_principal']}")

    print(f"\n{len(a_llm)} a envoyer au modele de langage :")
    for r in a_llm:
        print(f"  {r['nom']}")
