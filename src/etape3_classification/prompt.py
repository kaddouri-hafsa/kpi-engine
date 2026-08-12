"""
prompt.py

Seul et unique endroit ou le prompt de classification par modele de
langage est defini. Importe par appel_ollama.py et diagnostic_rapide.py.
Toute modification du prompt se fait ici, jamais redefinie ailleurs.
"""
import json

BATCH_SIZE = 5   # petits lots : plus fiable avec un modele local que de gros lots

SYSTEM_PROMPT = """Tu es un expert en analyse de données qui classifie les colonnes \
d'un jeu de données tabulaire selon un schéma technique précis.

Important : le champ "type_stockage_brut" fourni pour chaque colonne est le type de \
STOCKAGE détecté dans les données brutes (ex: "numerique" pour une colonne pandas \
de type int/float), PAS le rôle métier attendu. Un identifiant technique peut très \
bien être stocké numériquement : réfléchis au RÔLE, pas seulement au type de stockage.

Ne te fie pas non plus aveuglément au texte de la définition si le profil empirique \
la contredit (ex: une colonne présentée comme "taux" mais dont les valeurs réelles \
dépassent largement 1 n'est probablement pas une proportion entre 0 et 1 - signale \
ce genre d'incohérence dans le champ "note").

Certaines colonnes ont une définition vide (aucune information fournie) : base-toi \
alors uniquement sur le nom de la colonne et son profil empirique, et indique ton \
incertitude dans "note" plutôt que d'inventer un rôle avec une confiance injustifiée. \
De même, il arrive que deux colonnes au nom très proche (ex: une variante avec et \
sans suffixe numérique) aient des définitions qui se contredisent ou semblent \
inversées entre elles : dans ce cas, signale la contradiction dans "note" plutôt que \
de trancher arbitrairement en faveur de l'une des deux définitions.

Attention également au cas OPPOSÉ : deux ou plusieurs colonnes du même lot peuvent \
avoir des noms très proches et des définitions qui, cette fois, DISENT LA MÊME \
CHOSE (pas de contradiction) - ce sont alors des doublons ou quasi-doublons \
potentiels plutôt qu'une contradiction à trancher. Compare explicitement les noms \
et définitions des colonnes du lot entre elles : si deux colonnes semblent décrire \
le même concept, signale-le dans le champ "note" des DEUX colonnes concernées.

Critère pour "peut_etre_denominateur_ratio" : un dénominateur plausible est une \
grandeur de RÉFÉRENCE qui donne l'échelle globale d'une observation - par exemple, \
selon le domaine du jeu de données, un montant de base (chiffre d'affaires, budget, \
capital), un volume de référence (effectif, surface, quantité totale), ou toute \
autre grandeur qui sert à mettre les autres montants en proportion. Le nom n'a pas \
besoin de correspondre littéralement à un terme générique : toute colonne dont la \
DÉFINITION indique qu'elle représente une grandeur de référence de ce type est un \
dénominateur plausible. À l'inverse, un montant de coût, de charge, d'ajustement \
ponctuel ou de correction n'est PAS un dénominateur plausible : ce sont des montants \
qui ont vocation à être rapportés à une grandeur de référence, donc plutôt des \
NUMÉRATEURS de ratio que des dénominateurs.

- "role_principal" : DOIT être EXACTEMENT l'une de ces 2 valeurs, telles quelles :
  "categoriel", "numerique" (les dates ont déjà été classées automatiquement
  en amont à partir de leur type de stockage réel, tu ne les reçois
  normalement pas ici). N'utilise JAMAIS "type_stockage_brut" comme
  réponse : ce champ décrit le stockage brut des données, pas le rôle
  attendu en sortie - ne recopie jamais sa valeur telle quelle. Il n'existe
  PAS de rôle "identifiant" séparé : un identifiant technique (numéro
  interne, code de reference...) doit être classé "categoriel", et c'est
  le champ "dimension_agregation_pertinente" qui distingue un identifiant
  d'une vraie dimension d'analyse (voir critère ci-dessous).

Critère pour "dimension_agregation_pertinente" (uniquement pour les colonnes
"categoriel") : réponds true si la colonne prend un nombre limité de valeurs
qui se répètent à travers les lignes (ex: une catégorie, un statut, une
zone géographique - plusieurs lignes partagent la même valeur), ce qui permet
de calculer des moyennes de groupe ayant un sens. Réponds false si la colonne a
une valeur quasi unique par ligne (ex: un numéro de séquence, un nom propre,
un texte libre) : ce sont des identifiants ou des champs techniques, pas des
dimensions d'analyse utiles. Utilise le champ "exemples" et l'information de
cardinalité du profil empirique fourni pour juger, pas seulement le nom de
la colonne.

Tu dois répondre UNIQUEMENT avec un objet JSON valide, sans balises markdown, sans \
texte avant ou après. Le JSON doit contenir UNE ENTRÉE POUR CHAQUE colonne listée, \
sans exception - ne t'arrête pas après la première. Les CLÉS du JSON doivent être \
EXACTEMENT les noms de colonnes fournis, copiés tels quels."""


def slim_record(r: dict) -> dict:
    """Ne garde que les champs utiles au modele de langage (reduit le volume de tokens)."""
    p = r["profil_empirique"]
    return {
        "nom": r["nom"],
        "definition": r["definition"],
        "type_stockage_brut": p.get("type_empirique"),
        "min": p.get("min"),
        "max": p.get("max"),
        "pct_manquant": p.get("pct_manquant"),
        "nb_valeurs_distinctes": p.get("nunique"),
        "ratio_valeurs_distinctes": p.get("nunique_ratio"),  # proche de 1 = quasi unique par ligne
        "exemples": p.get("exemples_valeurs"),
    }


def build_batch_prompt(batch: list[dict]) -> str:
    slim = [slim_record(r) for r in batch]
    columns_block = json.dumps(slim, ensure_ascii=False, indent=2)
    noms = [r["nom"] for r in slim]

    # Squelette JSON complet (une entree vide par colonne, pas un exemple
    # unique + "...") : evite que le modele s'arrete apres la 1ere entree.
    skeleton = {
        n: {
            "role_principal": "...",
            "peut_etre_denominateur_ratio": None,
            "deja_un_taux": None,
            "dimension_agregation_pertinente": None,
            "note": "",
        }
        for n in noms
    }
    skeleton_block = json.dumps(skeleton, ensure_ascii=False, indent=2)

    return f"""Colonnes à classifier :

{columns_block}

Pour CHACUNE de ces {len(noms)} colonnes ({', '.join(noms)}), donne :
- "role_principal" (voir contrainte stricte dans la consigne système)
- "peut_etre_denominateur_ratio" : NULL si role_principal n'est PAS
  "numerique", sinon OBLIGATOIREMENT true ou false (voir le critère donné
  dans la consigne système) - ne réponds JAMAIS null pour une colonne
  numérique par indécision : si l'évidence est faible ou ambiguë, tranche
  quand même (vers false en cas de doute) et signale ton incertitude dans
  "note" plutôt que de laisser le champ vide.
- "deja_un_taux" : même règle que ci-dessus (NULL uniquement si non
  numérique, jamais par indécision sur une colonne numérique).
- "dimension_agregation_pertinente" : NULL si role_principal n'est PAS
  "categoriel", sinon OBLIGATOIREMENT true ou false (même règle : jamais
  null par indécision sur une colonne catégorielle, tranche et documente
  ton incertitude dans "note").
- "note" : string, peut être vide, pour signaler une ambiguïté, une
  incohérence, ou un doublon potentiel (voir consigne système)

Complète EXACTEMENT ce squelette JSON, qui contient déjà les {len(noms)} clés
attendues (une par colonne) - remplis chaque champ, n'ajoute ni ne retire
aucune clé, et n'omets AUCUNE des {len(noms)} colonnes :

{skeleton_block}"""


def make_batches(records: list[dict], batch_size: int = BATCH_SIZE) -> list[list[dict]]:
    return [records[i:i + batch_size] for i in range(0, len(records), batch_size)]
