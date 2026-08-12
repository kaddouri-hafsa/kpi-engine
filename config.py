"""
config.py

Point de configuration unique du pipeline. Chaque etape lit ses parametres
ici plutot que de coder en dur un chemin de fichier ou un nom de colonne,
ce qui permet de faire tourner le meme pipeline sur n'importe quel jeu de
donnees tabulaire sans modifier le code.

Chaque parametre peut etre surcharge par une variable d'environnement (par
exemple via un fichier .env charge au demarrage, ou directement dans le
shell) ; les valeurs par defaut ci-dessous pointent vers un exemple neutre
que l'utilisateur remplace par son propre fichier.
"""
import os
import sys
from pathlib import Path

try:
    # Sinon plante sur les caracteres accentues des qu'un script les
    # affiche - la console Windows par defaut est en cp1252, pas utf-8.
    # Sans effet (et sans erreur) si la sortie standard ne le permet pas
    # (ex. redirection vers un flux qui ne supporte pas reconfigure()).
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

ROOT_DIR = Path(__file__).resolve().parent


def _env_path(var_name: str, default: Path) -> Path:
    value = os.environ.get(var_name)
    return Path(value) if value else default


def _default_dict_path() -> Path:
    """Si KPI_DICT_PATH n'est pas defini, cherche un dictionnaire existant
    sous n'importe lequel des formats acceptes (Excel, CSV, texte brut)
    plutot que de supposer une seule extension - evite qu'un dictionnaire
    .txt pose sans configuration soit silencieusement ignore."""
    dossier = ROOT_DIR / "data" / "sources"
    for nom in ("dictionnaire.xlsx", "dictionnaire.csv", "dictionnaire.txt"):
        candidat = dossier / nom
        if candidat.exists():
            return candidat
    return dossier / "dictionnaire.xlsx"  # valeur d'exemple si rien n'existe encore


# Fichier de donnees a analyser (Excel).
DATA_PATH = _env_path("KPI_DATA_PATH", ROOT_DIR / "data" / "sources" / "donnees.xlsx")
DATA_SHEET = os.environ.get("KPI_DATA_SHEET", "Feuil1")

# Dictionnaire de donnees : Excel/CSV (colonnes nom + definition) ou texte brut .txt (format "NOM, -- definition") 
DICTIONNAIRE_PATH = _env_path("KPI_DICT_PATH", _default_dict_path())

# Variable de performance que les KPI generes doivent expliquer (colonne numerique )
TARGET_COL = os.environ.get("KPI_TARGET_COL", "RESULTAT")

# Modele de langage local (Ollama) utilise pour la classification, le raisonnement metier et les explications en langage naturel.
MODEL_NAME = os.environ.get("KPI_LLM_MODEL", "gpt-oss:20b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# Colonne utilisee pour regrouper les candidats par annee lors du calcul du score de stabilite temporelle (optionel).
STABILITY_YEAR_COL = os.environ.get("KPI_YEAR_COL", "")

# Contexte metier optionnel, injecte dans le prompt de raisonnement a
# priori (etape 2) pour orienter le modele vers les dimensions d'analyse
# pertinentes pour CE secteur d'activite 
DOMAIN_CONTEXT = os.environ.get("KPI_DOMAIN_CONTEXT", "")

# Fichiers intermediaires et resultats, regeneres a chaque execution.
DATA_GENERATED_DIR = ROOT_DIR / "data" / "generated"
RESULTATS_DIR = ROOT_DIR / "resultats"
CLASSIFICATION_PATH = RESULTATS_DIR / "classification_finale.json"
DICT_ENRICHI_PATH = DATA_GENERATED_DIR / "dictionnaire_enrichi.json"
KPI_METIER_PATH = RESULTATS_DIR / "kpi_proposes_metier.json"
QUALITY_REPORT_PATH = DATA_GENERATED_DIR / "rapport_qualite_donnees.json"
TOP_KPI_PATH = RESULTATS_DIR / "top_kpi.csv"
TOP_KPI_EXPLIQUE_PATH = RESULTATS_DIR / "top_kpi_explique.xlsx"
DAX_EXPORT_PATH = RESULTATS_DIR / "mesures_powerbi.csx"

# --- Module text-to-SQL --------------------------------------------------
TEXT_TO_SQL_DIR = ROOT_DIR / "text_to_sql"
TEXT_TO_SQL_DB_PATH = TEXT_TO_SQL_DIR / "donnees_locales.db"
# Nom de la table SQL - par defaut, le nom de la feuille source.
TEXT_TO_SQL_TABLE_NAME = os.environ.get("KPI_SQL_TABLE_NAME", DATA_SHEET)

 _num_gpu_env = os.environ.get("KPI_SQL_NUM_GPU")
SQL_NUM_GPU = int(_num_gpu_env) if _num_gpu_env else None

