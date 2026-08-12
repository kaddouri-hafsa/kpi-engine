"""
load_dictionnaire.py (Etape 1)

Charge le dictionnaire de donnees fourni par l'utilisateur (fichier Excel
ou CSV) et le convertit en une liste de records {nom, definition},
consommee par les etapes suivantes.

Deux formats acceptes :
- Excel/CSV (tabulaire) : une ligne par colonne du fichier de donnees, avec
  au minimum un nom technique et une definition en langage naturel. La
  colonne nom est detectee par son en-tete (contient "nom", "champ" ou
  "colonne"), la colonne definition de meme (contient "defini") ; a
  defaut de detection, repli sur la 1ere colonne (nom) et la 2eme
  (definition).
- Texte brut (.txt) : une ligne par colonne, au format
  "NOM_TECHNIQUE, -- definition en texte libre" (commentaire de style
  SQL) - format tel que fourni directement par certaines equipes metier,
  sans mise en forme tabulaire prealable.

Aucune interpretation du texte des definitions ici - c'est le role du
modele de langage (etape 3).

INPUT  : config.DICTIONNAIRE_PATH
OUTPUT : load_dictionnaire() -> liste de {"nom": str, "definition": str}
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DICTIONNAIRE_PATH, DATA_GENERATED_DIR  # noqa: E402

OUTPUT_PATH = DATA_GENERATED_DIR / "dictionnaire_brut.json"

LIGNE_TEXTE_PATTERN = re.compile(r"^([A-Za-z0-9_]+)\s*,?\s*(--\s*(.*))?$")


def _lire_dictionnaire_texte(path: Path) -> pd.DataFrame:
    """Parse le format texte brut 'NOM, -- definition' (une colonne par
    ligne, definition en commentaire de style SQL)."""
    lignes_non_reconnues = []
    records = []

    for raw_line in path.read_text(encoding="utf-8").split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        m = LIGNE_TEXTE_PATTERN.match(line)
        if not m:
            lignes_non_reconnues.append(raw_line)
            continue
        nom = m.group(1)
        definition = re.sub(r"\s+", " ", (m.group(3) or "").strip())
        records.append({"nom": nom, "definition": definition})

    if lignes_non_reconnues:
        print(f"ATTENTION : {len(lignes_non_reconnues)} ligne(s) du dictionnaire texte non "
              f"reconnue(s) (format inattendu) :")
        for line in lignes_non_reconnues:
            print(f"  {line!r}")

    return pd.DataFrame(records)


def lire_fichier_tabulaire(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        return _lire_dictionnaire_texte(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def detecter_colonnes(df: pd.DataFrame) -> tuple[str, str]:
    """Detecte les colonnes nom/definition par leur en-tete plutot que par
    position - un dictionnaire peut avoir une colonne d'index ("N°") en
    premiere position, ce qui rendrait un simple df.columns[0] incorrect."""
    if df.shape[1] < 2:
        raise ValueError("le dictionnaire doit contenir au moins deux colonnes (nom technique, definition).")

    colonnes_nom = [c for c in df.columns if any(m in str(c).lower() for m in ("nom", "champ", "colonne"))]
    col_nom = colonnes_nom[0] if colonnes_nom else df.columns[0]

    colonnes_definition = [c for c in df.columns if "defini" in str(c).lower()]
    col_definition = colonnes_definition[0] if colonnes_definition else df.columns[1]

    return col_nom, col_definition


def load_dictionnaire(dict_path: Path = DICTIONNAIRE_PATH) -> list[dict]:
    df = lire_fichier_tabulaire(dict_path)
    col_nom, col_definition = detecter_colonnes(df)

    records = []
    for _, row in df.iterrows():
        nom = str(row[col_nom]).strip()
        if not nom or nom.lower() == "nan":
            continue
        definition = str(row[col_definition]).strip() if pd.notna(row[col_definition]) else ""
        records.append({"nom": nom, "definition": definition})

    noms = [r["nom"] for r in records]
    doublons = {n for n in noms if noms.count(n) > 1}
    if doublons:
        print(f"ATTENTION : {len(doublons)} nom(s) de colonne apparaissant plusieurs "
              f"fois dans le dictionnaire : {sorted(doublons)}")

    return records


if __name__ == "__main__":
    records = load_dictionnaire()
    print(f"{len(records)} colonnes chargees depuis {DICTIONNAIRE_PATH.name}")

    sans_def = [r["nom"] for r in records if not r["definition"]]
    print(f"{len(sans_def)} colonne(s) sans definition (transmise(s) telle(s) quelle(s) au modele de "
          f"langage, qui devra se baser uniquement sur le nom et le profil empirique) :")
    for n in sans_def:
        print(f"  {n}")

    DATA_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegarde : {OUTPUT_PATH}")
