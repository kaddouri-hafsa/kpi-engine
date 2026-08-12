"""
detection_doublons.py

Detection de colonnes numeriques potentiellement DUPLIQUEES entre elles
(pas avec la cible comme dans detection_fuite.py), par correlation directe
sur les vraies donnees. Complement objectif au jugement du modele de
langage, qui traite les colonnes par petits lots et ne les compare pas
systematiquement entre elles pour reperer des doublons.

Seuil volontairement TRES strict (0.98) : on ne veut signaler que des
quasi-certitudes de doublon, pas de simples correlations normales (deux
colonnes peuvent etre correlees a 0.7-0.9 sans etre des doublons).

LIMITE A GARDER EN TETE : sur un petit echantillon, plusieurs paires
signalees peuvent etre du bruit statistique plutot que de vrais doublons.
Ce script fournit des CANDIDATS a verifier, pas des certitudes - a
recouper avec le sens des noms/definitions avant de conclure.

INPUT  : fichier de donnees (config.DATA_PATH)
OUTPUT : fonction detect_duplicates(), importee par appel_ollama.py
"""
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DATA_PATH, DATA_SHEET  # noqa: E402

DUPLICATE_THRESHOLD = 0.98


def detect_duplicates(threshold: float = DUPLICATE_THRESHOLD,
                       data_path: Path = DATA_PATH,
                       data_sheet: str = DATA_SHEET) -> dict:
    """
    Retourne, pour chaque colonne numerique, la liste des autres colonnes
    numeriques avec lesquelles sa correlation depasse le seuil.
    """
    df = pd.read_excel(data_path, sheet_name=data_sheet)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    corr_matrix = df[numeric_cols].corr().abs()

    doublons = {col: [] for col in numeric_cols}
    for col_a, col_b in combinations(numeric_cols, 2):
        corr = corr_matrix.loc[col_a, col_b]
        if pd.notna(corr) and corr >= threshold:
            doublons[col_a].append((col_b, round(float(corr), 4)))
            doublons[col_b].append((col_a, round(float(corr), 4)))

    return {col: pairs for col, pairs in doublons.items() if pairs}


if __name__ == "__main__":
    doublons = detect_duplicates()
    print(f"Seuil de detection de doublon : correlation >= {DUPLICATE_THRESHOLD}\n")

    if not doublons:
        print("Aucun doublon potentiel detecte a ce seuil.")
    else:
        vus = set()
        for col, pairs in doublons.items():
            for autre, corr in pairs:
                paire = tuple(sorted([col, autre]))
                if paire not in vus:
                    vus.add(paire)
                    print(f"  {paire[0]:30s} <-> {paire[1]:30s} | correlation = {corr}")

    print(f"\n{len(doublons)} colonne(s) impliquee(s) dans au moins un doublon potentiel")
    print("RAPPEL : fiabilite dependante de la taille de l'echantillon disponible.")
