"""
streamlit_app.py

Interface de chat pour le module text-to-SQL : pose une question en
langage naturel, recois une reponse en francais. La requete SQL generee
est toujours affichee (repliee par defaut) par transparence - meme si
l'utilisateur n'a pas besoin de la lire pour comprendre la reponse, elle
reste consultable pour verification.

Un second onglet "Explorer les donnees" permet de voir les colonnes
disponibles et un echantillon des donnees AVANT de poser une question -
requetes SQL fixes (PRAGMA / SELECT ... LIMIT), sans passer par le modele
de langage (voir db_explorer.py).

Lancement (depuis la racine du projet) :
    streamlit run text_to_sql/streamlit_app.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import TEXT_TO_SQL_DB_PATH, TEXT_TO_SQL_TABLE_NAME, MODEL_NAME  # noqa: E402
from agent_text_to_sql import ask_detaille  # noqa: E402
from db_explorer import get_row_count, get_columns, get_preview  # noqa: E402

st.set_page_config(
    page_title="KPI Engine | Assistant Données",
    layout="wide",
)


st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 3.5rem;
}
.artk-topbar {
    display: flex;
    align-items: baseline;
    gap: 0.6rem;
    padding-bottom: 0.9rem;
    margin-bottom: 1.3rem;
    border-bottom: 1px solid #DDE1D6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
}
.artk-topbar * {
    font-variant: normal;
    font-feature-settings: normal;
}
.artk-wordmark {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #658D26;
    white-space: nowrap;
}
.artk-sep {
    color: #C7CDBF;
}
.artk-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #20262C;
}
.artk-meta {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.78rem;
    color: #7C8574;
    margin-top: 0.3rem;
}
.artk-kv {
    display: flex;
    justify-content: space-between;
    padding: 0.32rem 0;
    font-size: 0.83rem;
    border-bottom: 1px solid #EDF0E7;
}
.artk-kv-label {
    color: #7C8574;
}
.artk-kv-value {
    color: #20262C;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.artk-sidebar-heading {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #7C8574;
    margin: 1.1rem 0 0.4rem 0;
}
.artk-stat-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.3rem;
}
.artk-stat-card {
    flex: 1;
    background: #FAFBF8;
    border: 1px solid #E2E6DA;
    border-radius: 6px;
    padding: 0.7rem 0.95rem;
}
.artk-stat-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8A9280;
    margin: 0;
}
.artk-stat-value {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 1.25rem;
    font-weight: 600;
    color: #20262C;
    margin: 0.15rem 0 0 0;
}
section[data-testid="stSidebar"] {
    border-right: 1px solid #E2E6DA;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1.4rem;
    border-bottom: 1px solid #DDE1D6;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    padding: 0.4rem 0.05rem;
    font-weight: 500;
    color: #7C8574;
    border-radius: 0;
}
.stTabs [aria-selected="true"] {
    color: #20262C !important;
    background: transparent !important;
    border-bottom: 2px solid #658D26;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="artk-topbar">
  <span class="artk-wordmark">KPI Engine</span>
  <span class="artk-sep">/</span>
  <span class="artk-title">Assistant Données</span>
</div>
""", unsafe_allow_html=True)

if not TEXT_TO_SQL_DB_PATH.exists():
    st.error(
        f"Base de données introuvable : `{TEXT_TO_SQL_DB_PATH}`\n\n"
        f"Lance d'abord, depuis la racine du projet :\n\n"
        f"```\npython text_to_sql/creer_base_locale.py\n```"
    )
    st.stop()


@st.cache_data(show_spinner=False)
def _charger_apercu_base(db_mtime: float, table_name: str):
    """db_mtime en cle de cache : invalide automatiquement si la base est
    regeneree (creer_base_locale.py) pendant que l'appli tourne."""
    return {
        "n_lignes": get_row_count(TEXT_TO_SQL_DB_PATH, table_name),
        "colonnes": get_columns(TEXT_TO_SQL_DB_PATH, table_name),
        "apercu": get_preview(TEXT_TO_SQL_DB_PATH, table_name, limit=20),
    }


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


infos_base = _charger_apercu_base(TEXT_TO_SQL_DB_PATH.stat().st_mtime, TEXT_TO_SQL_TABLE_NAME)

# ---------------------------------------------------------------------------
# Barre laterale : panneau de proprietes + requetes suggerees
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<p class="artk-sidebar-heading">Connexion</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="artk-kv"><span class="artk-kv-label">Table</span><span class="artk-kv-value">{TEXT_TO_SQL_TABLE_NAME}</span></div>
    <div class="artk-kv"><span class="artk-kv-label">Modèle</span><span class="artk-kv-value">{MODEL_NAME}</span></div>
    <div class="artk-kv"><span class="artk-kv-label">Accès</span><span class="artk-kv-value">lecture seule</span></div>
    """, unsafe_allow_html=True)
    st.caption(
        "La requête SQL est générée par le modèle puis exécutée par la base ; "
        "le modèle ne calcule rien lui-même."
    )

    st.markdown('<p class="artk-sidebar-heading">Requêtes suggérées</p>', unsafe_allow_html=True)
    exemples = [
        "Combien de lignes contient la base ?",
        "Quelles sont les colonnes disponibles ?",
        "Donne-moi un aperçu des 5 premières lignes.",
    ]
    for exemple in exemples:
        if st.button(exemple, width='stretch', key=f"ex_{exemple}"):
            st.session_state["question_en_attente"] = exemple

    st.markdown("---")
    if st.button("Effacer la conversation", width='stretch', type="secondary"):
        st.session_state["messages"] = []
        st.rerun()

# ---------------------------------------------------------------------------
# Onglets : conversation vs exploration de la base
# ---------------------------------------------------------------------------
tab_chat, tab_explorer = st.tabs(["Assistant", "Explorer les données"])

with tab_explorer:
    st.markdown(f"""
    <div class="artk-stat-row">
      <div class="artk-stat-card">
        <p class="artk-stat-label">Lignes</p>
        <p class="artk-stat-value">{_fmt(infos_base['n_lignes'])}</p>
      </div>
      <div class="artk-stat-card">
        <p class="artk-stat-label">Colonnes</p>
        <p class="artk-stat-value">{_fmt(len(infos_base['colonnes']))}</p>
      </div>
      <div class="artk-stat-card">
        <p class="artk-stat-label">Table</p>
        <p class="artk-stat-value" style="font-size:1.05rem;">{TEXT_TO_SQL_TABLE_NAME}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_gauche, col_droite = st.columns([1, 2], gap="large")

    with col_gauche:
        st.markdown("**Colonnes**")
        recherche = st.text_input(
            "Rechercher une colonne", placeholder="Rechercher une colonne",
            label_visibility="collapsed",
        )
        colonnes_affichees = infos_base["colonnes"]
        if recherche:
            colonnes_affichees = colonnes_affichees[
                colonnes_affichees["Colonne"].str.contains(recherche, case=False, na=False)
            ]
        st.dataframe(colonnes_affichees, width='stretch', hide_index=True, height=420)

    with col_droite:
        st.markdown("**Aperçu**")
        st.caption("20 premières lignes de la table.")
        st.dataframe(infos_base["apercu"], width='stretch', hide_index=True, height=420)

    st.caption("Nommer explicitement une colonne dans la question lève toute ambiguïté sur un terme métier.")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with tab_chat:
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("Requête SQL générée"):
                    st.code(msg["sql"], language="sql")
            if msg.get("df") is not None:
                with st.expander(f"Résultat ({len(msg['df'])} ligne{'s' if len(msg['df']) != 1 else ''})"):
                    st.dataframe(msg["df"], width='stretch', hide_index=True)

    # Le dernier message est une question pas encore repondue (juste ajoutee
    # ci-dessous puis st.rerun() declenche) - on genere la reponse ICI, a
    # l'interieur de l'onglet, donc TOUJOURS au-dessus du chat_input qui est
    # declare hors des onglets plus bas dans le script.
    derniers = st.session_state["messages"]
    if derniers and derniers[-1]["role"] == "user":
        with st.chat_message("assistant"):
            with st.spinner("Génération en cours…"):
                question_en_cours = derniers[-1]["content"]
                try:
                    resultat = ask_detaille(question_en_cours, verbose=False)
                    df = resultat["resultat_df"]
                    st.markdown(resultat["reponse"])
                    with st.expander("Requête SQL générée"):
                        st.code(resultat["sql"], language="sql")
                    with st.expander(f"Résultat ({len(df)} ligne{'s' if len(df) != 1 else ''})"):
                        st.dataframe(df, width='stretch', hide_index=True)
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": resultat["reponse"],
                        "sql": resultat["sql"],
                        "df": df,
                    })
                except Exception as e:
                    message_erreur = (
                        "La requête a été rejetée ou a échoué "
                        f"({e}). Reformule la question, par exemple en nommant "
                        "explicitement une colonne si le terme employé est ambigu."
                    )
                    st.error(message_erreur)
                    st.session_state["messages"].append({"role": "assistant", "content": message_erreur})

# Chat input declare HORS des onglets (a la racine de la page) : c'est la
# seule condition sous laquelle Streamlit l'epingle automatiquement en bas
# de l'ecran - a l'interieur d'un conteneur (onglet, colonne...) il ne
# ferait que s'inserer au fil du code, pas rester fixe en bas.
question = st.chat_input("Poser une question…")
if not question and "question_en_attente" in st.session_state:
    question = st.session_state.pop("question_en_attente")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    st.rerun()
