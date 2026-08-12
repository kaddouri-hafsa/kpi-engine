"""
db_explorer.py

Lecture seule de la structure et d'un echantillon de la base locale, pour
que l'utilisateur puisse explorer ce qui est disponible AVANT de poser une
question - independant de agent_text_to_sql.py : ici aucun modele de
langage, uniquement des requetes SQL fixes (PRAGMA / SELECT), donc aucun
risque de generation incorrecte.

Meme connexion en lecture seule au niveau du fichier (mode=ro) que
l'agent - defense en profondeur identique.
"""
import sqlite3
from pathlib import Path

import pandas as pd


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def get_row_count(db_path: Path, table_name: str) -> int:
    with _connect(db_path) as conn:
        cursor = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        return cursor.fetchone()[0]


def get_columns(db_path: Path, table_name: str) -> pd.DataFrame:
    """Nom + type SQLite declare de chaque colonne, via PRAGMA table_info
    (metadonnee de schema, pas une requete sur les donnees elles-memes)."""
    with _connect(db_path) as conn:
        cursor = conn.execute(f'PRAGMA table_info("{table_name}")')
        rows = cursor.fetchall()
    return pd.DataFrame(
        [{"Colonne": r[1], "Type": r[2] or "indetermine"} for r in rows]
    )


def get_preview(db_path: Path, table_name: str, limit: int = 10) -> pd.DataFrame:
    with _connect(db_path) as conn:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}" LIMIT {int(limit)}', conn)
