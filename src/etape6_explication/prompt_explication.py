"""
prompt_explication.py (Etape 6b)

Seul et unique endroit ou le prompt d'explication metier des KPI est
defini. Importe par generer_explications.py.

Le modele de langage ne recoit JAMAIS le nom technique seul : il recoit
les VRAIES definitions metier des colonnes qui composent le candidat
(fournies par l'equipe metier via le dictionnaire, etape 1) + les 4
scores objectifs + l'effet XAI par quartile (etape 6a). Objectif : qu'il
traduise en langage metier sans jamais inventer une signification qui ne
decoule pas de ces faits fournis.
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import TARGET_COL, DOMAIN_CONTEXT  # noqa: E402

DEFAULT_DOMAIN_CONTEXT = "de l'activité concernée"

SYSTEM_PROMPT_TEMPLATE = """Tu es un expert métier qui explique à des décideurs non-techniques \
la signification d'indicateurs de performance (KPI) générés automatiquement à partir d'un \
jeu de données, {contexte_metier}.

Pour chaque KPI, on te donne :
- son nom lisible et sa formule de construction (ratio, moyenne par groupe, comptage \
d'observations, ou délai entre deux dates)
- les définitions métier RÉELLES (fournies par l'équipe métier) des colonnes brutes qui \
le composent
- 4 scores statistiques objectifs : complétude (part de valeurs renseignées), information \
(variabilité), pouvoir prédictif (importance validée par validation croisée sur la variable \
"{target_col}"), stabilité (régularité dans le temps)
- un signal indiquant si le pouvoir prédictif est réellement significatif en valeur absolue, \
ou seulement "le moins faible du lot"
- une "origine" : "raisonnement_metier" si ce KPI est un ratio/indicateur reconnu du secteur \
d'activité (proposé par raisonnement métier avant même de regarder les données), ou \
"generation_exhaustive" s'il vient d'une combinaison générée automatiquement puis validée \
statistiquement
- un effet XAI : comment "{target_col}" se comporte (médiane) quand ce KPI est dans son \
quartile haut vs son quartile bas

RÈGLES STRICTES :
1. Base-toi UNIQUEMENT sur les définitions métier fournies pour expliquer ce que représente \
le KPI. N'invente JAMAIS une signification, un contexte ou un usage qui ne découle pas \
directement de ces définitions.
2. Si "signal_predictif_significatif" est false, tu DOIS le dire explicitement : ce KPI est \
descriptif (stable, informatif) mais SON POUVOIR PRÉDICTIF SUR LA PERFORMANCE N'EST PAS \
DÉMONTRÉ - ne le présente jamais comme un facteur explicatif de "{target_col}". Ceci s'applique \
MÊME si "origine" est "raisonnement_metier" : être un ratio reconnu du secteur ne garantit \
pas un pouvoir prédictif démontré sur CES données précises - ne confonds jamais les deux.
3. Si "origine" est "raisonnement_metier", precise que ce KPI est un indicateur standard du \
secteur d'activité concerné (sans survendre son pouvoir prédictif si le point 2 s'applique).
4. Interprète l'effet XAI en une phrase simple ("quand ce KPI est élevé, {target_col} tend à \
être plus/moins élevé"), sans jargon statistique (pas de "quartile", "médiane", "corrélation" \
dans ta réponse - reformule en langage courant).
5. Reste factuel et concis : 3 à 4 phrases maximum.
6. Si les définitions fournies sont insuffisantes ou ambiguës pour juger du sens métier, \
dis-le explicitement plutôt que de combler le vide par une supposition.

Réponds UNIQUEMENT avec un objet JSON valide de la forme :
{{"explication": "..."}}
Sans balises markdown, sans texte avant ou après."""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    contexte_metier=(f"dans le contexte suivant : {DOMAIN_CONTEXT}" if DOMAIN_CONTEXT else DEFAULT_DOMAIN_CONTEXT),
    target_col=TARGET_COL,
)


def build_kpi_prompt(nom_lisible: str, formule: str, definitions: dict,
                      scores: dict, xai: dict) -> str:
    definitions_block = json.dumps(definitions, ensure_ascii=False, indent=2)
    scores_block = json.dumps(scores, ensure_ascii=False, indent=2)
    xai_block = json.dumps(xai, ensure_ascii=False, indent=2)

    return f"""KPI à expliquer : {nom_lisible}
Formule : {formule}

Définitions métier des colonnes brutes impliquées :
{definitions_block}

Scores statistiques objectifs :
{scores_block}

Effet XAI (comportement de {TARGET_COL} selon ce KPI) :
{xai_block}

Rédige l'explication metier en respectant strictement les règles de la consigne système."""
