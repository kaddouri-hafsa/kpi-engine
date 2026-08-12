"""
prompt_raisonnement.py (Etape 3b : raisonnement metier a priori)

Seul et unique endroit ou le prompt de proposition de KPI par raisonnement
metier a priori est defini. Importe par proposer_kpi_metier.py.

Difference fondamentale avec le prompt de classification (etape 3) : ce
prompt ne recoit JAMAIS de profil empirique (min/max/cardinalite/exemples
de valeurs reelles) - uniquement le nom et la definition metier de chaque
colonne. Objectif : forcer un raisonnement TOP-DOWN, base sur la
connaissance metier, avant meme de savoir comment les donnees se
comportent empiriquement - a l'oppose de la generation exhaustive
(etape 4) qui part des donnees et note tout apres coup.

Le contexte metier (config.DOMAIN_CONTEXT) est injecte dans le prompt
pour orienter le modele vers les dimensions d'analyse pertinentes pour le
secteur d'activite concerne ; un contexte generique est utilise par
defaut si aucun n'est fourni.

DECOUPAGE EN LOTS : envoyer un dictionnaire volumineux en un seul prompt
peut faire tourner certains modeles en boucle sans jamais renvoyer de
reponse exploitable - meme principe que la classification (etape 3), qui
traite deja par petits lots pour cette raison. Le raisonnement
inter-colonnes reste possible AU SEIN d'un lot, mais une formule qui
necessiterait de combiner deux colonnes tombees dans des lots differents
ne sera pas trouvee - limite acceptee en echange de la fiabilite.
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import TARGET_COL, DOMAIN_CONTEXT  # noqa: E402

BATCH_SIZE = 20

DEFAULT_DOMAIN_CONTEXT = """Appuie-toi sur les dimensions classiques d'analyse de la performance \
d'une activité (rentabilité, coûts et charges rapportés au volume d'activité, taux d'utilisation \
ou d'exposition d'une ressource, adéquation entre un tarif ou un prix et la valeur réellement \
délivrée), en les adaptant au secteur d'activité que révèlent les définitions fournies."""

SYSTEM_PROMPT_TEMPLATE = """Tu es un expert en analyse de la performance métier, chargé de proposer les \
indicateurs de performance (KPI) les plus pertinents pour évaluer la performance et la rentabilité \
des observations d'un jeu de données.

RÈGLE FONDAMENTALE : tu ne reçois AUCUNE donnée empirique (pas de valeurs réelles, pas de \
statistiques, pas de cardinalité) - uniquement le nom technique et la définition métier de \
chaque colonne. Tu dois raisonner UNIQUEMENT à partir de ta connaissance du métier concerné et \
des définitions fournies, PAS à partir de ce qui "marche bien" sur un jeu de données que tu ne vois pas.

{contexte_metier}

Pour CHAQUE colonne fournie, une définition existe déjà - NE PROPOSE JAMAIS un KPI utilisant une \
colonne dont le nom ne figure pas EXACTEMENT (même orthographe) dans la liste fournie. N'invente \
aucune colonne, aucun nom approximatif.

IMPORTANT : la liste fournie est un LOT PARTIEL du dictionnaire complet (traité par lots \
successifs), pas la totalité des colonnes du fichier. S'il te manque une colonne pour former une \
formule pertinente (ex. le dénominateur d'un ratio que tu identifies), NE L'INVENTE PAS et NE \
PROPOSE PAS ce KPI plutôt que d'halluciner une colonne absente - certaines formules ne seront \
tout simplement pas trouvables dans ce lot, c'est normal et acceptable.

INTERDICTION ABSOLUE : la colonne "{target_col}" est la variable que CES KPI DOIVENT AIDER À ÉVALUER \
- ne l'utilise JAMAIS comme numérateur, dénominateur, ou composante d'un KPI. Un KPI qui inclut \
{target_col} dans sa propre formule est circulaire et sera rejeté.

IMPORTANT, À NE PAS OUBLIER : certaines colonnes du dictionnaire peuvent déjà être, par leur \
définition, un indicateur directement exploitable (leur définition dit explicitement qu'il \
s'agit d'un "ratio", d'un "taux", ou d'un "indicateur" déjà calculé) - repère-les et propose-les \
TELLES QUELLES avec operation "brute", sans chercher à les recombiner avec d'autres colonnes. \
Ne les ignore pas.

NE RÉPÈTE JAMAIS le même KPI (mêmes colonnes, même opération) sous un nom différent - chaque \
proposition doit être structurellement distincte des précédentes.

Pour chaque KPI proposé, structure la formule avec le champ "operation" parmi exactement ces \
4 valeurs :
- "ratio" : numerateur (liste de colonnes, sommées entre elles si plusieurs) / denominateur \
(liste de colonnes, sommées entre elles si plusieurs)
- "brute" : une colonne déjà exploitable telle quelle (champ "colonne")
- "complement" : 1 moins la colonne, pour un taux de rétention ou de couverture par exemple \
(champ "colonne")
- "somme_ratios" : somme de plusieurs ratios déjà définis (ex: un ratio composite standard du \
secteur), champ "ratios" = liste de {{"numerateur": [...], "denominateur": [...]}}

Réponds UNIQUEMENT avec un objet JSON valide de la forme :
{{"kpis": [
  {{
    "nom": "...",
    "operation": "ratio",
    "numerateur": ["COL_A", "COL_B"],
    "denominateur": ["COL_C"],
    "colonne": null,
    "ratios": null,
    "justification_metier": "..."
  }},
  {{
    "nom": "...",
    "operation": "complement",
    "numerateur": null,
    "denominateur": null,
    "colonne": "COL_X",
    "ratios": null,
    "justification_metier": "..."
  }}
]}}
Remplis TOUJOURS les 6 clés pour chaque KPI (mets null pour celles qui ne s'appliquent pas à \
l'operation choisie). Sans balises markdown, sans texte avant ou après. Propose entre 2 et 6 \
KPI DIFFÉRENTS pour CE LOT, uniquement ceux que tu juges réellement structurants pour évaluer \
la performance - pas un KPI par colonne, pas un remplissage mécanique, jamais deux fois le \
même. Si aucune colonne de ce lot ne permet de construire un KPI pertinent, réponds \
{{"kpis": []}} plutôt que d'en forcer un."""

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    contexte_metier=DOMAIN_CONTEXT or DEFAULT_DOMAIN_CONTEXT,
    target_col=TARGET_COL,
)


def slim_dictionary(records: list[dict]) -> list[dict]:
    """Ne garde QUE nom + definition - aucune donnee empirique (voir docstring
    du module : le raisonnement doit rester purement top-down)."""
    return [{"nom": r["nom"], "definition": r["definition"]} for r in records]


def make_batches(records: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    return [records[i:i + batch_size] for i in range(0, len(records), batch_size)]


def build_prompt(records: list[dict]) -> str:
    slim = slim_dictionary(records)
    dictionnaire_block = json.dumps(slim, ensure_ascii=False, indent=2)

    return f"""Dictionnaire des colonnes disponibles (nom + definition metier UNIQUEMENT, \
aucune donnee reelle) :

{dictionnaire_block}

Propose les KPI les plus pertinents pour evaluer la performance/la rentabilite d'une \
observation du jeu de donnees, en respectant STRICTEMENT le format JSON et les regles de la \
consigne systeme."""
