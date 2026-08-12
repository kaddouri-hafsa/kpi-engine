"""
charger_classification.py

Derive, a partir du resultat de l'etape 3 (classification_finale.json),
les listes de travail necessaires a la generation de candidats KPI :
numerateurs de ratio possibles, denominateurs, dimensions d'agregation,
colonnes de dates. AUCUNE liste codee en dur - tout decoule directement
des reponses de la classification automatique + de garde-fous statistiques
objectifs (cardinalite, risque de fuite), sans jugement metier code en dur.

Le role semantique des colonnes (role_principal, peut_etre_denominateur_ratio...)
vient de la classification par modele de langage (etape 3, deja faite,
resultats/classification_finale.json) et reste valable tant que le schema
de colonnes ne change pas. Le RISQUE DE FUITE et la CARDINALITE sont en
revanche des calculs purement statistiques (pandas) : ils sont recalcules
ICI a chaque appel sur le jeu de donnees reellement utilise en aval,
plutot que de faire confiance aux valeurs figees dans
classification_finale.json (qui peuvent avoir ete calculees sur un
fichier different).

INPUT  : resultats/classification_finale.json (etape 3) + config.DATA_PATH
OUTPUT : fonction load_working_lists(), importee par generation_candidats.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape3_classification"))

from config import DATA_PATH, DATA_SHEET, TARGET_COL, CLASSIFICATION_PATH, DICT_ENRICHI_PATH  # noqa: E402
from detection_fuite import detect_leakage  # noqa: E402

# Garde-fou OBJECTIF (statistique, pas semantique) : une colonne numerique
# quasi unique par ligne (cardinalite proche de 1) est structurellement un
# identifiant, meme si le modele a laisse ses champs a null plutot que
# false. On l'exclut mecaniquement des candidats numerateur/agregation, en
# complement du jugement du modele plutot qu'a sa place.
CARDINALITY_THRESHOLD = 0.9

# Garde-fou SYMETRIQUE au precedent, a l'autre extreme : une colonne
# numerique avec TRES PEU de valeurs distinctes (< 30% des lignes) se
# comporte comme une categorie plutot que comme une grandeur economique
# continue (ex: une annee ne prend qu'une poignee de valeurs - diviser un
# montant par une annee, ou l'inverse, n'a aucun sens metier). Ce n'est
# pas un jugement sur le NOM de la colonne : c'est la meme logique de
# cardinalite objective, utilisee ailleurs pour promouvoir ces colonnes en
# dimensions de regroupement plutot qu'en grandeurs a mettre au
# numerateur/denominateur.
LOW_CARDINALITY_THRESHOLD = 0.3

# Une cardinalite relative basse ne suffit pas a garantir une VRAIE
# categorie repetee : une grandeur continue (ex. un taux a plusieurs
# decimales) peut accidentellement avoir peu de valeurs distinctes sur un
# echantillon donne, sans que ces valeurs se repetent reellement en nombre
# suffisant. MIN_GROUP_SIZE / MIN_GROUP_SHARE verifient la REPETITION
# effective, pas seulement le ratio de cardinalite - critere objectif
# complementaire, toujours sans jugement sur le nom ou le sens metier de
# la colonne.
MIN_GROUP_SIZE = 10
MIN_GROUP_SHARE = 0.8


def _load_active_dataframe() -> pd.DataFrame:
    return pd.read_excel(DATA_PATH, sheet_name=DATA_SHEET)


def has_sufficient_repetition(series: pd.Series, min_group_size: int = MIN_GROUP_SIZE,
                               min_share: float = MIN_GROUP_SHARE) -> bool:
    """
    Vrai si au moins min_share (80% par defaut) des lignes non manquantes
    appartiennent a une valeur qui se repete au moins min_group_size fois.
    C'est le test objectif de "est-ce que cette colonne se comporte comme
    une categorie repetee", independant de la cardinalite relative seule et
    independant du role_principal ou du jugement du modele.
    """
    counts = series.value_counts(dropna=True)
    if len(counts) == 0:
        return False
    part_bien_groupee = counts[counts >= min_group_size].sum() / counts.sum()
    return part_bien_groupee >= min_share


def load_cardinality_ratios(df: pd.DataFrame = None, dict_path: Path = DICT_ENRICHI_PATH) -> dict:
    """
    Ratio de cardinalite (valeurs distinctes / nb de lignes) de chaque
    colonne. Si un DataFrame est fourni, le calcul se fait directement
    dessus, pour TOUTES les colonnes (numeriques et categorielles) -
    toujours coherent avec le jeu de donnees reellement utilise. Sinon,
    repli sur le dictionnaire enrichi (profilage fige de l'etape 2, peut
    etre perime si le jeu de donnees actif a change depuis).
    """
    if df is not None:
        if len(df) == 0:
            return {}
        return {
            col: round(df[col].nunique(dropna=True) / len(df), 4)
            for col in df.columns
        }

    if not dict_path.exists():
        print(f"ATTENTION : {dict_path.name} introuvable - le garde-fou de cardinalite "
              f"(exclusion des quasi-identifiants numeriques) est DESACTIVE. Relance "
              f"profiler_donnees.py pour le regenerer avant de faire confiance aux "
              f"candidats numerateur/agregation.")
        return {}
    with open(dict_path, encoding="utf-8") as f:
        records = json.load(f)
    return {
        r["nom"]: r["profil_empirique"].get("nunique_ratio")
        for r in records
        if "profil_empirique" in r
    }


def load_working_lists(df: pd.DataFrame = None, classification_path: Path = CLASSIFICATION_PATH,
                        target_col: str = TARGET_COL) -> dict:
    with open(classification_path, encoding="utf-8") as f:
        classification = json.load(f)

    if df is None:
        df = _load_active_dataframe()

    cardinalite = load_cardinality_ratios(df)

    # Risque de fuite RECALCULE sur le jeu de donnees actif (purement
    # statistique, aucun appel au modele de langage) - ne fait jamais
    # confiance a la valeur figee dans classification_finale.json si elle
    # a ete calculee sur un fichier different de DATA_PATH.
    leakage = detect_leakage(data_path=DATA_PATH, data_sheet=DATA_SHEET, target_col=target_col)

    date_cols = []
    group_dims = []
    ratio_denominators = []
    ratio_numerator_candidates = []
    aggregation_targets = []
    desaccords_llm = []

    for nom, entry in classification.items():
        if nom == target_col:
            continue  # la cible ne devient jamais elle-meme un candidat

        role = entry.get("role_principal")

        if role == "date":
            date_cols.append(nom)

        elif role == "categoriel":
            # Verifie la repetition REELLE des valeurs plutot que de faire
            # confiance uniquement au jugement du modele
            # (dimension_agregation_pertinente) : le fait objectif
            # l'emporte dans les deux sens (peut aussi exclure une colonne
            # que le modele aurait approuvee a tort).
            est_dimension_objective = nom in df.columns and has_sufficient_repetition(df[nom])
            llm_dit_pertinent = entry.get("dimension_agregation_pertinente") is True

            if est_dimension_objective != llm_dit_pertinent:
                desaccords_llm.append((nom, llm_dit_pertinent, est_dimension_objective))

            if est_dimension_objective:
                group_dims.append(nom)

        elif role == "numerique":
            ratio_quasi_unique = (cardinalite.get(nom) or 0) >= CARDINALITY_THRESHOLD
            ratio_quasi_categorielle = 0 < (cardinalite.get(nom) or 0) < LOW_CARDINALITY_THRESHOLD
            risque_fuite_actuel = (leakage.get(nom) or {}).get("risque_de_fuite")

            # Denominateur : le jugement du modele (grandeur de reference)
            # reste necessaire, mais un denominateur quasi-deterministe de
            # la cible propage la fuite a TOUS les ratios qui l'utilisent, et
            # un quasi-identifiant/quasi-categorielle (ex. une annee) n'a pas
            # de sens comme diviseur - memes garde-fous que pour les
            # numerateurs, appliques ici aussi par symetrie.
            if (entry.get("peut_etre_denominateur_ratio") is True
                    and risque_fuite_actuel is not True
                    and not ratio_quasi_unique
                    and not ratio_quasi_categorielle):
                ratio_denominators.append(nom)

            # Numerateur de ratio : ni un taux deja exprime (division absurde),
            # ni a risque de fuite avec la cible, ni quasi-identifiant, ni
            # quasi-categorielle (garde-fous objectifs, complementaires au
            # jugement du modele)
            if (entry.get("deja_un_taux") is not True
                    and risque_fuite_actuel is not True
                    and not ratio_quasi_unique
                    and not ratio_quasi_categorielle):
                ratio_numerator_candidates.append(nom)

            # Cible d'agregation (moyenne par groupe) : les taux restent
            # pertinents ici (ex. taux moyen par categorie a un sens),
            # seules la fuite de donnees, la quasi-unicite et la
            # quasi-categorisation sont exclues
            if (risque_fuite_actuel is not True
                    and not ratio_quasi_unique
                    and not ratio_quasi_categorielle):
                aggregation_targets.append(nom)

    if desaccords_llm:
        print(f"  ({len(desaccords_llm)} colonne(s) categorielle(s) ou le jugement du modele sur "
              f"dimension_agregation_pertinente contredit la repetition reelle des valeurs - "
              f"le fait objectif l'emporte) :")
        for nom, llm_dit, objectif in desaccords_llm:
            print(f"    - {nom} : modele={llm_dit}, retenu(e)={objectif}")

    return {
        "date_cols": date_cols,
        "group_dims": group_dims,
        "ratio_denominators": ratio_denominators,
        "ratio_numerator_candidates": ratio_numerator_candidates,
        "aggregation_targets": aggregation_targets,
    }


def resolve_year_series(df: pd.DataFrame, working_lists: dict) -> pd.Series:
    """
    Renvoie la serie "annee" utilisee pour le score de stabilite
    temporelle (etape 5). Si config.STABILITY_YEAR_COL est renseigne et
    present dans le fichier, elle est utilisee telle quelle. Sinon, l'annee
    est derivee automatiquement de la premiere colonne de date disponible
    (classee par la classification, etape 3) - fonctionne sur n'importe
    quel fichier sans configuration, au prix d'un choix arbitraire de
    colonne de date si plusieurs existent.
    """
    from config import STABILITY_YEAR_COL

    if STABILITY_YEAR_COL and STABILITY_YEAR_COL in df.columns:
        return df[STABILITY_YEAR_COL]

    date_cols = working_lists.get("date_cols") or []
    if not date_cols:
        raise ValueError(
            "Impossible de calculer le score de stabilite temporelle : aucune colonne "
            "d'annee (KPI_YEAR_COL) ni colonne de date detectee dans le fichier."
        )
    return df[date_cols[0]].dt.year


if __name__ == "__main__":
    lists = load_working_lists()
    for nom, valeurs in lists.items():
        print(f"\n{nom} ({len(valeurs)}) :")
        print(f"  {valeurs}")
