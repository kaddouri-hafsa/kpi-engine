"""
profiler_donnees.py (Etape 2)

Enrichit le dictionnaire brut avec un profilage EMPIRIQUE et STATISTIQUE
des vraies donnees (type reellement stocke, % de manquants, cardinalite,
min/max, exemples de valeurs). Uniquement des faits objectifs et
calculables - aucune interpretation du sens metier des colonnes, c'est le
role du modele de langage (etape 3).

La coherence entre le dictionnaire et le fichier de donnees est verifiee
de facon generique (toute colonne du dictionnaire absente des donnees, ou
l'inverse, est signalee) sans tentative de rapprochement automatique -
ces ecarts se corrigent a la source, pas par une regle codee en dur ici.

INPUT  : dictionnaire_brut.json (etape 1) + fichier de donnees (config.DATA_PATH)
OUTPUT : dictionnaire_enrichi.json + Dictionnaire_Donnees_Enrichi.xlsx
"""
import json
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import DATA_PATH, DATA_SHEET, DATA_GENERATED_DIR  # noqa: E402

INPUT_PATH = DATA_GENERATED_DIR / "dictionnaire_brut.json"
OUTPUT_JSON = DATA_GENERATED_DIR / "dictionnaire_enrichi.json"
OUTPUT_XLSX = DATA_GENERATED_DIR / "Dictionnaire_Donnees_Enrichi.xlsx"


def verify_coverage(dict_records: list[dict], df: pd.DataFrame) -> None:
    """Signale les ecarts entre dictionnaire et donnees reelles, sans
    tenter de les corriger : toute correction reste une decision humaine,
    propre a chaque cas."""
    dict_cols = {r["nom"] for r in dict_records}
    real_cols = set(df.columns)

    manquantes = dict_cols - real_cols
    en_trop = real_cols - dict_cols

    if manquantes:
        print(f"ATTENTION : {len(manquantes)} colonne(s) du dictionnaire absente(s) "
              f"du fichier de donnees reel : {sorted(manquantes)}")
    if en_trop:
        print(f"ATTENTION : {len(en_trop)} colonne(s) du fichier de donnees reel "
              f"absente(s) du dictionnaire : {sorted(en_trop)}")
    if not manquantes and not en_trop:
        print("Couverture verifiee : dictionnaire et donnees reelles correspondent exactement.")


def profile_column(df: pd.DataFrame, col: str) -> dict:
    s = df[col]
    n = len(s)
    profile = {
        "pct_manquant": round(s.isna().mean() * 100, 1),
        "nunique": int(s.nunique(dropna=True)),
        "nunique_ratio": round(s.nunique(dropna=True) / n, 4) if n else None,
    }

    non_null = s.dropna()

    if pd.api.types.is_datetime64_any_dtype(s):
        # Ce test doit passer avant is_numeric_dtype, sinon les dates sont
        # mal classees en "categoriel_ou_texte".
        profile["type_empirique"] = "date"
        if len(non_null) > 0:
            profile["min"] = str(non_null.min().date())
            profile["max"] = str(non_null.max().date())
    elif pd.api.types.is_numeric_dtype(s) and len(non_null) > 0:
        profile["type_empirique"] = "numerique"
        profile["min"] = round(float(non_null.min()), 4)
        profile["max"] = round(float(non_null.max()), 4)
        profile["moyenne"] = round(float(non_null.mean()), 4)
        profile["borne_0_1"] = bool(non_null.between(0, 1).all())
    elif len(non_null) > 0:
        profile["type_empirique"] = "categoriel_ou_texte"
        profile["exemples_valeurs"] = non_null.astype(str).unique()[:5].tolist()
    else:
        profile["type_empirique"] = "entierement_vide"

    return profile


def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Tentative generique de parsing des colonnes dont le nom contient
    "DATE" et qui sont stockees en texte (format JJ/MM/AAAA courant). Ne
    s'applique que si le parsing reussit sur une part significative des
    valeurs non nulles - sinon la colonne reste inchangee."""
    date_like = [c for c in df.columns if "DATE" in c.upper()]
    for c in date_like:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        parsed = pd.to_datetime(df[c], format="%d/%m/%Y", errors="coerce")
        non_null_before = df[c].notna().sum()
        if non_null_before > 0 and parsed.notna().sum() / non_null_before > 0.8:
            df[c] = parsed
    return df


def enrich_dictionary() -> list[dict]:
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = json.load(f)

    df = pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)
    df = try_parse_dates(df)

    verify_coverage(records, df)

    enriched = []
    for record in records:
        col = record["nom"]
        record = dict(record)
        record["profil_empirique"] = (
            profile_column(df, col) if col in df.columns
            else {"erreur": "colonne absente du fichier de donnees reel"}
        )
        enriched.append(record)

    return enriched


def build_excel(enriched: list[dict]) -> None:
    rows = []
    for i, r in enumerate(enriched, start=1):
        p = r["profil_empirique"]
        rows.append({
            "N°": i,
            "Champ (nom technique)": r["nom"],
            "Définition proposée": r["definition"],
            "Type empirique détecté": p.get("type_empirique", ""),
            "% manquant": p.get("pct_manquant"),
            "Valeurs distinctes": p.get("nunique"),
            "Min (si numérique/date)": p.get("min"),
            "Max (si numérique/date)": p.get("max"),
            "Borné [0,1] (si numérique)": p.get("borne_0_1"),
            "Exemples de valeurs (si catégoriel)": ", ".join(p.get("exemples_valeurs", [])) if p.get("exemples_valeurs") else "",
        })
    df_out = pd.DataFrame(rows)

    wb = Workbook()
    ws = wb.active
    ws.title = "Dictionnaire enrichi"

    header_font = Font(name="Arial", bold=True, color="FFFFFF", size=10)
    header_fill = PatternFill(start_color="1F3864", end_color="1F3864", fill_type="solid")
    normal_font = Font(name="Arial", size=10)
    wrap = Alignment(wrap_text=True, vertical="top")

    for col_idx, col_name in enumerate(df_out.columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font, cell.fill, cell.alignment = header_font, header_fill, wrap

    for row_idx, row in enumerate(df_out.itertuples(index=False), start=2):
        for col_idx, value in enumerate(row, start=1):
            cell = ws.cell(row=row_idx, column=col_idx, value=value)
            cell.font, cell.alignment = normal_font, wrap

    widths = [5, 26, 45, 16, 12, 12, 14, 14, 12, 35]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w
    ws.freeze_panes = "A2"

    wb.save(OUTPUT_XLSX)


if __name__ == "__main__":
    enriched = enrich_dictionary()

    DATA_GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n{len(enriched)} colonnes enrichies -> {OUTPUT_JSON}")

    build_excel(enriched)
    print(f"Dictionnaire enrichi (Excel) -> {OUTPUT_XLSX}")

    n_dates = sum(1 for r in enriched if r["profil_empirique"].get("type_empirique") == "date")
    print(f"{n_dates} colonnes dates detectees")
