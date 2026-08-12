"""
creer_base_locale.py

Recharge le MEME jeu de donnees que le pipeline KPI (via load_real_data(),
reutilise telle quelle - meme nettoyage, memes types) dans une base
SQLite locale, pour un usage sans dependre d'une base de production
distante. Basculer sur une base de production plus tard ne demande qu'un
changement de chaine de connexion dans agent_text_to_sql.py
(SQLDatabase.from_uri), pas une reecriture.

A relancer a chaque fois que le fichier de donnees source change.

OUTPUT : text_to_sql/donnees_locales.db (config.TEXT_TO_SQL_DB_PATH)
"""
import sqlite3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape4_candidats"))

from config import TEXT_TO_SQL_DB_PATH, TEXT_TO_SQL_TABLE_NAME  # noqa: E402
from generation_candidats import load_real_data  # noqa: E402


def main():
    df = load_real_data()

    TEXT_TO_SQL_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(TEXT_TO_SQL_DB_PATH) as conn:
        df.to_sql(TEXT_TO_SQL_TABLE_NAME, conn, if_exists="replace", index=False)

    print(f"\nBase SQLite ecrite : {TEXT_TO_SQL_DB_PATH}")
    print(f"Table '{TEXT_TO_SQL_TABLE_NAME}' : {df.shape[0]} lignes, {df.shape[1]} colonnes")


if __name__ == "__main__":
    main()
