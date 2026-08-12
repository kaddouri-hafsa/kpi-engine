"""
valider_entree.py (Etape 0)

Point d'entree du pipeline : verifie que le fichier de donnees et son
dictionnaire sont exploitables avant de lancer les etapes suivantes, pour
echouer tot avec un message clair plutot qu'au milieu d'une etape.

Verifications effectuees :
- le fichier de donnees et sa feuille existent et sont lisibles ;
- la colonne cible (config.TARGET_COL) existe dans le fichier ;
- le dictionnaire de donnees existe ; les colonnes du fichier sans
  definition correspondante sont signalees (jamais bloquant : la
  classification semantique les ignorera simplement) ;
- nettoyage mecanique des donnees et rapport de qualite pour revue
  humaine (colonnes ambigues ou avec valeurs aberrantes).

INPUT  : config.DATA_PATH, config.DATA_SHEET, config.DICTIONNAIRE_PATH, config.TARGET_COL
OUTPUT : valider_et_charger() -> DataFrame nettoye, pret pour la suite du pipeline
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape4_candidats"))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape1_dictionnaire"))

from config import DATA_PATH, DATA_SHEET, DICTIONNAIRE_PATH, TARGET_COL, QUALITY_REPORT_PATH  # noqa: E402
from nettoyage_donnees import nettoyer, generate_quality_report  # noqa: E402
from load_dictionnaire import lire_fichier_tabulaire, detecter_colonnes  # noqa: E402


class ErreurValidation(Exception):
    """Erreur de configuration ou de donnees d'entree, a corriger avant de relancer le pipeline."""


def valider_et_charger(data_path: Path = DATA_PATH, data_sheet: str = DATA_SHEET,
                        dict_path: Path = DICTIONNAIRE_PATH, target_col: str = TARGET_COL) -> pd.DataFrame:
    if not data_path.exists():
        raise ErreurValidation(
            f"Fichier de donnees introuvable : {data_path}\n"
            f"Renseigne la variable d'environnement KPI_DATA_PATH ou place ton fichier a cet emplacement."
        )

    try:
        df = pd.read_excel(data_path, sheet_name=data_sheet)
    except ValueError as e:
        raise ErreurValidation(
            f"Impossible de lire la feuille '{data_sheet}' dans {data_path.name} : {e}\n"
            f"Verifie le nom de la feuille (KPI_DATA_SHEET) ou le format du fichier."
        ) from e

    if df.empty:
        raise ErreurValidation(f"{data_path.name} ne contient aucune ligne exploitable.")

    if target_col not in df.columns:
        raise ErreurValidation(
            f"La colonne cible '{target_col}' (KPI_TARGET_COL) est absente de {data_path.name}.\n"
            f"Colonnes disponibles : {', '.join(df.columns)}"
        )

    if not dict_path.exists():
        raise ErreurValidation(
            f"Dictionnaire de donnees introuvable : {dict_path}\n"
            f"Renseigne la variable d'environnement KPI_DICT_PATH ou place ton dictionnaire a cet emplacement."
        )

    dictionnaire = lire_fichier_tabulaire(dict_path)
    col_nom, _ = detecter_colonnes(dictionnaire)
    colonnes_dict = set(dictionnaire[col_nom].astype(str))
    colonnes_sans_definition = [c for c in df.columns if c not in colonnes_dict]
    if colonnes_sans_definition:
        apercu = ", ".join(colonnes_sans_definition[:10])
        suite = ", ..." if len(colonnes_sans_definition) > 10 else ""
        print(f"ATTENTION : {len(colonnes_sans_definition)} colonne(s) du fichier de donnees n'ont "
              f"pas de definition dans le dictionnaire, elles seront ignorees par la classification "
              f"semantique : {apercu}{suite}")

    df_propre = nettoyer(df)

    rapport = generate_quality_report(df_propre)
    n_signalements = len(rapport["colonnes_zero_ambigu"]) + len(rapport["colonnes_avec_aberrantes"])
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2, default=str)
    if n_signalements:
        print(f"{n_signalements} colonne(s) signalee(s) dans le rapport de qualite "
              f"({QUALITY_REPORT_PATH.name}) - a revoir avant d'interpreter les KPI produits.")

    print(f"Donnees validees : {df_propre.shape[0]} lignes, {df_propre.shape[1]} colonnes.")
    return df_propre


if __name__ == "__main__":
    valider_et_charger()
