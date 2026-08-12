"""
nettoyage_donnees.py

Nettoyage mecanique du jeu de donnees, applique avant toute analyse. Deux
categories d'action, deliberement separees :

1. TRANSFORMATIONS SURES (appliquees automatiquement) : aucune ambiguite
   possible, aucun jugement metier requis.
   - suppression des espaces superflus en debut/fin de chaine et
     normalisation des espaces internes (deux valeurs qui ne different que
     par un espace ne doivent pas etre comptees comme deux categories)
   - suppression des lignes entierement dupliquees
   - parsing des colonnes de date (texte JJ/MM/AAAA -> datetime)

   Ce que ce nettoyage NE FAIT PAS : corriger la casse (peut etre
   volontaire), fusionner des categories qui se ressemblent sans etre
   identiques, ou deviner des valeurs manquantes - cela releve d'un
   jugement metier, pas d'un nettoyage mecanique.

2. DETECTION SANS CORRECTION (rapport, pas de modification) : les cas ou
   trancher automatiquement serait un jugement metier deguise.
   - colonnes numeriques ou une forte proportion de zeros exacts peut
     signifier soit une vraie valeur nulle, soit une absence de saisie -
     signalee, jamais convertie
   - valeurs aberrantes par colonne (methode IQR elargie) - signalees pour
     revue humaine, jamais supprimees automatiquement

INPUT  : DataFrame brut
OUTPUT : nettoyer() -> DataFrame nettoye
         generate_quality_report() -> dict de signalements pour revue humaine
"""
from pathlib import Path

import numpy as np
import pandas as pd

ZERO_HEAVY_THRESHOLD = 0.5    # part de zeros exacts au-dela de laquelle on signale l'ambiguite
IQR_MULTIPLIER = 3.0          # elargi (vs 1.5 standard) : adapte a des grandeurs naturellement asymetriques


# ---------------------------------------------------------------------------
# 1. TRANSFORMATIONS
# ---------------------------------------------------------------------------
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parsing generique des colonnes dont le nom contient 'DATE', stockees
    en texte JJ/MM/AAAA - ne s'applique que si le parsing reussit sur une
    part significative des valeurs non nulles."""
    df = df.copy()
    date_like = [c for c in df.columns if "DATE" in c.upper()]
    for c in date_like:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        parsed = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
        non_null_before = df[c].notna().sum()
        if non_null_before > 0 and parsed.notna().sum() / non_null_before > 0.8:
            df[c] = parsed
    return df


def nettoyer_espaces(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Trim + normalisation des espaces internes sur toutes les colonnes
    texte, et chaines vides/blanches -> NaN. Retourne le DataFrame nettoye
    et un rapport des colonnes affectees (audit avant/apres)."""
    df = df.copy()
    rapport = {}

    for col in df.columns:
        if df[col].dtype != object and not pd.api.types.is_string_dtype(df[col]):
            continue

        masque_non_null = df[col].notna()
        if masque_non_null.sum() == 0:
            continue

        valeurs_avant = df.loc[masque_non_null, col].astype(str)
        valeurs_apres = valeurs_avant.str.strip().str.replace(r"\s+", " ", regex=True)

        n_modifies = int((valeurs_avant != valeurs_apres).sum())
        if n_modifies > 0:
            df.loc[masque_non_null, col] = valeurs_apres
            rapport[col] = n_modifies

        # Chaine vide ou uniquement blanche = absence de valeur, pas une valeur
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    return df, rapport


def supprimer_doublons(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Supprime les lignes entierement dupliquees (toutes colonnes
    identiques) - un doublon exact n'apporte aucune information
    supplementaire et fausse les statistiques (comptages, moyennes)."""
    n_avant = len(df)
    df_dedupliquee = df.drop_duplicates()
    n_supprimees = n_avant - len(df_dedupliquee)
    return df_dedupliquee, n_supprimees


def nettoyer(df: pd.DataFrame) -> pd.DataFrame:
    df = parse_dates(df)

    df, rapport_espaces = nettoyer_espaces(df)
    if rapport_espaces:
        print(f"Espaces superflus corriges sur {len(rapport_espaces)} colonne(s) :")
        for col, n in rapport_espaces.items():
            print(f"  {col} : {n} valeur(s) modifiee(s)")

    df, n_doublons = supprimer_doublons(df)
    if n_doublons > 0:
        print(f"{n_doublons} ligne(s) dupliquee(s) supprimee(s)")

    return df


# ---------------------------------------------------------------------------
# 2. DETECTION SANS CORRECTION (rapport de qualite pour revue humaine)
# ---------------------------------------------------------------------------
def detect_zero_heavy_columns(df: pd.DataFrame, threshold: float = ZERO_HEAVY_THRESHOLD) -> dict:
    """Signale les colonnes numeriques ou une forte part de zeros exacts
    rend ambigu "vraie valeur nulle" vs "non renseigne" - a faire trancher
    par le metier, pas par ce script."""
    resultat = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for c in numeric_cols:
        non_null = df[c].dropna()
        if len(non_null) == 0:
            continue
        part_zero = (non_null == 0).mean()
        if part_zero >= threshold:
            resultat[c] = {
                "part_zero_exact": round(float(part_zero), 3),
                "note": "Ambiguite possible : zero peut signifier une vraie valeur nulle OU un champ non renseigne. A verifier avec le metier avant d'exclure ou d'imputer ces valeurs.",
            }
    return resultat


def detect_outliers(df: pd.DataFrame, iqr_multiplier: float = IQR_MULTIPLIER) -> dict:
    """Detecte les valeurs aberrantes par colonne numerique (methode IQR
    elargie). Rapporte le nombre et les bornes, ne supprime ni ne modifie
    aucune valeur - decision laissee a une revue humaine."""
    resultat = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for c in numeric_cols:
        non_null = df[c].dropna()
        if len(non_null) < 10:
            continue
        q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        borne_basse = q1 - iqr_multiplier * iqr
        borne_haute = q3 + iqr_multiplier * iqr
        aberrantes = non_null[(non_null < borne_basse) | (non_null > borne_haute)]
        if len(aberrantes) > 0:
            resultat[c] = {
                "nb_valeurs_aberrantes": int(len(aberrantes)),
                "part": round(len(aberrantes) / len(non_null), 3),
                "bornes_attendues": [round(float(borne_basse), 2), round(float(borne_haute), 2)],
                "min_observe": round(float(non_null.min()), 2),
                "max_observe": round(float(non_null.max()), 2),
            }
    return resultat


def generate_quality_report(df: pd.DataFrame) -> dict:
    return {
        "colonnes_zero_ambigu": detect_zero_heavy_columns(df),
        "colonnes_avec_aberrantes": detect_outliers(df),
    }


if __name__ == "__main__":
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import DATA_PATH, DATA_SHEET, QUALITY_REPORT_PATH, DATA_GENERATED_DIR  # noqa: E402

    df_brut = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
    print(f"Donnees brutes : {df_brut.shape[0]} lignes, {df_brut.shape[1]} colonnes\n")

    df_propre = nettoyer(df_brut)

    rapport = generate_quality_report(df_propre)
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QUALITY_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(rapport, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{len(rapport['colonnes_zero_ambigu'])} colonne(s) avec ambiguite zero/manquant :")
    for col, info in rapport["colonnes_zero_ambigu"].items():
        print(f"  {col:30s} | {info['part_zero_exact']*100:.0f}% de zeros exacts")

    print(f"\n{len(rapport['colonnes_avec_aberrantes'])} colonne(s) avec valeurs aberrantes :")
    for col, info in rapport["colonnes_avec_aberrantes"].items():
        print(f"  {col:30s} | {info['nb_valeurs_aberrantes']} valeur(s) hors bornes attendues {info['bornes_attendues']}")

    DATA_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_GENERATED_DIR / f"{DATA_PATH.stem}_nettoye{DATA_PATH.suffix}"
    df_propre.to_excel(out_path, sheet_name="Resultat", index=False)
    print(f"\nDonnees nettoyees sauvegardees : {out_path}")
    print(f"Rapport de qualite sauvegarde : {QUALITY_REPORT_PATH}")
