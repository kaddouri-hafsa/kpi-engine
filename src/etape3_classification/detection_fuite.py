"""
detection_fuite.py

Detection du risque de fuite de donnees par un calcul STATISTIQUE OBJECTIF
(correlation de rangs de Spearman avec la variable cible), calcule
directement sur les vraies donnees - plutot que par un jugement du modele
de langage sur la seule base des definitions textuelles.

CHOIX DE SPEARMAN (PAS PEARSON) : si la variable cible est asymetrique
(quelques valeurs extremes dominent la moyenne), une correlation de
Pearson est tres sensible a ces points extremes et peut sous-estimer
fortement une relation reelle et monotone. Spearman (base sur les rangs)
est robuste a ce type d'asymetrie et donne une image plus fiable de la
relation reelle - a preferer par defaut pour des variables financieres ou
operationnelles, naturellement asymetriques.

Generalisation : TARGET_COL est le seul parametre propre au cas d'usage
(le nom de la variable cible que le pipeline cherche a expliquer), fourni
par la configuration centrale - le reste du script ne fait aucune
hypothese sur les noms de colonnes.

LIMITE A GARDER EN TETE : la fiabilite de ce calcul depend directement du
nombre de lignes disponibles. Sur un petit echantillon, une correlation
estimee est statistiquement instable - a recalculer des que le jeu de
donnees complet est disponible.

INPUT  : fichier de donnees (config.DATA_PATH)
OUTPUT : fonction detect_leakage(), importee par appel_ollama.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DATA_PATH, DATA_SHEET, TARGET_COL  # noqa: E402

# Seuil au-dela duquel on considere qu'il y a un risque de fuite. 0.9 est
# volontairement strict : au-dela, une colonne est quasi-deterministe de
# la cible et ne doit pas etre utilisee comme candidat KPI ni comme
# variable d'entree du score predictif.
CORRELATION_THRESHOLD = 0.9


def detect_leakage(threshold: float = CORRELATION_THRESHOLD,
                    data_path: Path = DATA_PATH,
                    data_sheet: str = DATA_SHEET,
                    target_col: str = TARGET_COL) -> dict:
    """
    Retourne, pour chaque colonne numerique du fichier (sauf la cible
    elle-meme), sa correlation avec la cible et un booleen risque_de_fuite.
    """
    df = pd.read_excel(data_path, sheet_name=data_sheet)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if target_col not in df.columns:
        raise ValueError(f"Colonne cible '{target_col}' introuvable dans {data_path}")

    correlations = df[numeric_cols].corr(method="spearman")[target_col]

    result = {
        target_col: {"correlation_avec_resultat": 1.0, "risque_de_fuite": True},
    }

    for col in numeric_cols:
        if col == target_col:
            continue
        corr = correlations.get(col)
        if pd.isna(corr):
            result[col] = {"correlation_avec_resultat": None, "risque_de_fuite": None}
        else:
            result[col] = {
                "correlation_avec_resultat": round(float(corr), 4),
                "risque_de_fuite": bool(abs(corr) >= threshold),
            }

    return result


if __name__ == "__main__":
    leakage = detect_leakage()

    print(f"Correlation avec {TARGET_COL}, seuil de risque de fuite = {CORRELATION_THRESHOLD}\n")

    a_risque = {c: v for c, v in leakage.items() if v["risque_de_fuite"]}
    print(f"{len(a_risque)} colonne(s) a risque de fuite :")
    for col, v in sorted(a_risque.items(), key=lambda x: -(abs(x[1]["correlation_avec_resultat"] or 0))):
        print(f"  {col:30s} | correlation = {v['correlation_avec_resultat']}")

    non_calculable = [c for c, v in leakage.items() if v["correlation_avec_resultat"] is None]
    if non_calculable:
        print(f"\n{len(non_calculable)} colonne(s) sans correlation calculable : {non_calculable}")
