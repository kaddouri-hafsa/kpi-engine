"""
agent_text_to_sql.py

Module text-to-SQL : traduit une question en francais en requete SQL,
l'execute sur la base (locale par defaut, cf. creer_base_locale.py - a
brancher sur une base de production en changeant CONNECTION_STRING), et
retourne la reponse en langage naturel.

Chaine DIRECTE (2 appels au modele de langage : SQL puis reponse, + un 3e
optionnel si correction necessaire) plutot qu'un agent autonome
multi-etapes de type LangChain "tool-calling" : ce dernier demande au
modele de respecter un format d'appel d'outil structure de facon stricte
et de decider lui-meme, etape par etape, quand executer reellement la
requete - deux exigences peu fiables avec un modele local plus leger
qu'un modele proprietaire heberge (format d'appel non respecte, ou
requete jamais executee, remplacee par une reponse fabriquee). La chaine
fixe garantit une execution deterministe a chaque fois et permet
d'appliquer des garde-fous de securite a des points precis du code.

GARDE-FOUS DE SECURITE (objectif : utilisable par quelqu'un sans aucune
connaissance SQL, donc incapable de reperer une requete dangereuse) :
1. Connexion SQLite ouverte en LECTURE SEULE au niveau du fichier lui-meme
   (mode=ro) - un DELETE/UPDATE genere par erreur echoue au niveau de la
   base elle-meme, pas seulement au niveau d'une consigne dans le prompt
   (une consigne au modele n'est PAS une vraie protection - il peut
   halluciner malgre elle). Une fois branche sur une base de production,
   utiliser un compte dedie en lecture seule (meme principe, au niveau
   serveur).
2. Validation deterministe (`valider_sql`) : rejette toute requete qui
   n'est pas un SELECT (ou un WITH...SELECT), avant meme de l'executer -
   defense en profondeur, independante de la protection SQLite ci-dessus.
3. LIMIT automatique si absent, pour eviter qu'une question mal ciblee ne
   renvoie des dizaines de milliers de lignes.
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # sinon plante sur certains caracteres
# que le modele peut inserer (ex. espace fine insecable dans un nombre
# formate) - la console Windows par defaut est en cp1252, pas utf-8.

import pandas as pd
from langchain_community.utilities import SQLDatabase 
from langchain_ollama import ChatOllama

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import (  # noqa: E402
    MODEL_NAME, OLLAMA_BASE_URL, TEXT_TO_SQL_DB_PATH, SQL_NUM_GPU,
)

# uri=true + file:...?mode=ro : ouvre le fichier SQLite en lecture seule au
# niveau du systeme de fichiers - meme un DELETE genere par le modele
# echoue ("attempt to write a readonly database").
CONNECTION_STRING = f"sqlite:///file:{TEXT_TO_SQL_DB_PATH}?mode=ro&uri=true"

MAX_LIGNES = 200
MAX_TENTATIVES_CORRECTION = 2

MOTS_INTERDITS = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "attach", "detach", "pragma", "vacuum", "replace",
]

# Exemples "few-shot" optionnels : ajoute ici des paires (question, SQL)
# pour orienter le modele sur du vocabulaire metier ambigu propre a ton
# jeu de donnees (ex. un terme courant qui ne correspond pas au nom de
# colonne le plus evident). Vide par defaut - fonctionne sans, mais
# quelques exemples bien choisis ameliorent nettement la fiabilite sur des
# questions formulees en langage courant.
EXEMPLES_SQL: list[tuple[str, str]] = [
    # ("Quel est le taux moyen de X ?", "SELECT AVG(COLONNE) FROM table"),
]

SQL_PROMPT = """Tu es un assistant qui traduit une question en francais en une requete SQL (dialecte SQLite), en lecture seule (SELECT uniquement).

Schema de la base de donnees :
{schema}
{exemples_section}
Question : {question}

Reponds UNIQUEMENT avec la requete SQL, sans explication, sans balises markdown, sans point-virgule final.
"""

CORRECTION_PROMPT = """La requete SQL suivante a echoue :
{sql}

Erreur retournee : {erreur}

Schema de la base de donnees :
{schema}

Question d'origine : {question}

Corrige la requete. Reponds UNIQUEMENT avec la requete SQL corrigee, sans explication, sans balises markdown, sans point-virgule final.
"""

ANSWER_PROMPT = """Question posee : {question}
Requete SQL executee : {sql}
Resultat brut de la requete : {result}

Reponds a la question en une phrase claire en francais, en te basant uniquement sur ce resultat.
"""


def clean_sql(texte: str) -> str:
    texte = re.sub(r"^```sql\s*|^```\s*|```$", "", texte.strip(), flags=re.MULTILINE)
    return texte.strip().rstrip(";")


def valider_sql(sql: str) -> None:
    """Leve ValueError si la requete n'est pas un SELECT en lecture seule.
    Complement de la connexion SQLite en lecture seule (defense en
    profondeur), pas une garantie a elle seule."""
    if not re.match(r"^\s*(with\b|select\b)", sql, flags=re.IGNORECASE):
        raise ValueError("la requete generee n'est pas un SELECT")

    for mot in MOTS_INTERDITS:
        if re.search(rf"\b{mot}\b", sql, flags=re.IGNORECASE):
            raise ValueError(f"mot-cle interdit detecte dans la requete : {mot}")


def appliquer_limite(sql: str, max_lignes: int = MAX_LIGNES) -> str:
    if re.search(r"\blimit\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {max_lignes}"


def formatter_exemples() -> str:
    if not EXEMPLES_SQL:
        return ""
    corps = "\n".join(f'- "{q}" -> {sql}' for q, sql in EXEMPLES_SQL)
    return (
        "\nExemples de traductions correctes (le vocabulaire metier ne correspond pas "
        f"toujours au nom de colonne le plus evident - inspire-toi de ces exemples) :\n{corps}\n"
    )


def build_llm() -> ChatOllama:
    kwargs = {"model": MODEL_NAME, "base_url": OLLAMA_BASE_URL, "temperature": 0}
    if SQL_NUM_GPU is not None:
        kwargs["num_gpu"] = SQL_NUM_GPU
    return ChatOllama(**kwargs)


def generer_et_executer(db: SQLDatabase, llm: ChatOllama, question: str, schema: str, verbose: bool):
    """Genere la requete, la valide, l'execute - et si une erreur survient
    (validation OU execution), redonne l'erreur au modele pour qu'il se
    corrige, jusqu'a MAX_TENTATIVES_CORRECTION essais supplementaires."""
    sql = clean_sql(llm.invoke(SQL_PROMPT.format(
        schema=schema, exemples_section=formatter_exemples(), question=question,
    )).content)

    for tentative in range(1 + MAX_TENTATIVES_CORRECTION):
        try:
            valider_sql(sql)
            sql_final = appliquer_limite(sql)
            resultat = db.run(sql_final)
            return sql_final, resultat
        except Exception as e:
            if verbose:
                print(f"Tentative {tentative + 1} echouee ({e}) - {'nouvelle tentative' if tentative < MAX_TENTATIVES_CORRECTION else 'abandon'}\n")
            if tentative == MAX_TENTATIVES_CORRECTION:
                raise
            sql = clean_sql(llm.invoke(CORRECTION_PROMPT.format(
                sql=sql, erreur=str(e), schema=schema, question=question,
            )).content)


def executer_sql_dataframe(sql: str) -> pd.DataFrame:
    """Reexecute la requete deja validee (SELECT, LIMIT applique) sur une
    connexion lecture seule dediee, pour obtenir le resultat COMPLET sous
    forme de tableau - independant du resultat texte tronque (troncature
    des chaines longues via truncate_word de langchain) que db.run() utilise
    pour construire le prompt de reponse en langage naturel. Necessaire car
    cette reponse en langage naturel resume/n'illustre le resultat que par
    quelques exemples quand il contient beaucoup de lignes - le tableau
    complet reste consultable independamment de ce resume."""
    with sqlite3.connect(f"file:{TEXT_TO_SQL_DB_PATH}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(sql, conn)


def ask_detaille(question: str, verbose: bool = False) -> dict:
    """Comme ask(), mais retourne aussi la requete SQL executee, le
    resultat brut (texte, utilise pour le prompt de reponse) et le resultat
    complet sous forme de DataFrame - utile pour les interfaces (ex.
    Streamlit) qui veulent afficher la requete et le tableau de resultat
    par transparence, pas seulement le resume en langage naturel."""
    db = SQLDatabase.from_uri(CONNECTION_STRING)
    llm = build_llm()
    schema = db.get_table_info()

    sql, resultat_brut = generer_et_executer(db, llm, question, schema, verbose)
    resultat_df = executer_sql_dataframe(sql)
    if verbose:
        print(f"SQL execute :\n  {sql}\n")
        print(f"Resultat brut : {resultat_brut}\n")

    reponse = llm.invoke(ANSWER_PROMPT.format(question=question, sql=sql, result=resultat_brut))
    return {
        "sql": sql,
        "resultat_brut": resultat_brut,
        "resultat_df": resultat_df,
        "reponse": reponse.content,
    }


def ask(question: str, verbose: bool = True) -> str:
    return ask_detaille(question, verbose=verbose)["reponse"]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Combien de lignes contient la base ?"

    print(f"Question : {question}\n")
    print(ask(question))
