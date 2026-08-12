"""
generation_candidats.py (Etape 4)

Genere EXHAUSTIVEMENT les candidats KPI dans 3 familles, a partir des
listes derivees automatiquement de la classification (charger_classification.py) :

1. RATIOS : chaque (numerateur autorise) / (denominateur identifie)
2. AGREGATIONS PAR GROUPE :
   - MOYENNE d'une cible numerique par dimension
   - COMPTAGE (nombre de lignes) par dimension - KPI de volume
3. ECARTS DE DATES : delai en jours entre CHAQUE paire de colonnes de
   dates (genere mecaniquement, aucune paire choisie a la main - la
   notation en aval filtre ce qui est effectivement informatif)

Extension des dimensions de regroupement (au-dela des colonnes categorielles
identifiees par le modele de langage) :
- annee et mois derives mecaniquement de chaque colonne de date
- colonnes numeriques a faible cardinalite, qui se comportent comme des
  categories bien qu'elles soient stockees comme des nombres - meme
  logique de garde-fou objectif que dans charger_classification.py

INPUT  : fichier de donnees (etape 0) + resultats de charger_classification.py
OUTPUT : fonction generate_all_candidates(), consommee par notation_kpi.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape0_validation"))

from charger_classification import (  # noqa: E402
    load_working_lists, load_cardinality_ratios, has_sufficient_repetition,
    LOW_CARDINALITY_THRESHOLD, MIN_GROUP_SIZE,
)
from valider_entree import valider_et_charger  # noqa: E402

# Correlation (Spearman, robuste a l'asymetrie de la cible) entre un
# numerateur et son propre denominateur : au-dela de ce seuil, le ratio est
# quasi-tautologique (les deux grandeurs racontent la meme chose) plutot
# qu'un veritable insight metier - complement de detection_doublons.py,
# applique ici specifiquement aux paires numerateur/denominateur.
NUMERATOR_DENOMINATOR_CORR_THRESHOLD = 0.95


def load_real_data() -> pd.DataFrame:
    """Charge, valide et nettoie le jeu de donnees actif - delegue a
    l'etape 0 pour ne jamais dupliquer la logique de validation/nettoyage."""
    return valider_et_charger()


def derive_date_parts(df: pd.DataFrame, date_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Cree des colonnes annee/mois derivees de chaque date - extraction
    MECANIQUE, pas un choix editorial de quelles dates comptent."""
    nouvelles_dims = []
    for col in date_cols:
        if col not in df.columns:
            continue
        annee_col = f"{col}__annee"
        mois_col = f"{col}__mois"
        df[annee_col] = df[col].dt.year
        df[mois_col] = df[col].dt.to_period("M").astype(str)
        nouvelles_dims.extend([annee_col, mois_col])
    return df, nouvelles_dims


def extended_group_dims(df: pd.DataFrame, group_dims: list[str], date_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Etend les dimensions de regroupement categorielles avec les parties
    de date derivees et les colonnes numeriques a faible cardinalite."""
    cardinalite = load_cardinality_ratios(df)
    low_card_numeric = [
        nom for nom, ratio in cardinalite.items()
        if ratio is not None and ratio < LOW_CARDINALITY_THRESHOLD
        and nom in df.columns and nom not in group_dims
        and pd.api.types.is_numeric_dtype(df[nom])
        and has_sufficient_repetition(df[nom])
    ]
    df, date_derived = derive_date_parts(df, date_cols)
    return df, group_dims + low_card_numeric + date_derived


def generate_ratio_candidates(df: pd.DataFrame, numerators: list[str], denominators: list[str]) -> pd.DataFrame:
    """
    Genere numerateur/denominateur pour chaque paire autorisee, en ecartant
    les paires quasi-tautologiques : si le numerateur est deja tres
    correle a son propre denominateur (Spearman >= NUMERATOR_DENOMINATOR_CORR_THRESHOLD),
    le ratio ne raconte rien de nouveau - complement de detection_doublons.py
    (qui ne regarde que la correlation avec la CIBLE, pas entre les deux
    termes du ratio eux-memes).
    """
    series_list = []
    paires_ecartees = []
    for denom in denominators:
        if denom not in df.columns:
            continue
        safe_denom = df[denom].replace(0, np.nan)
        for num in numerators:
            if num == denom or num not in df.columns:
                continue
            corr_num_denom = df[num].corr(df[denom], method="spearman")
            if pd.notna(corr_num_denom) and abs(corr_num_denom) >= NUMERATOR_DENOMINATOR_CORR_THRESHOLD:
                paires_ecartees.append(f"{num}/{denom} (correlation = {corr_num_denom:.3f})")
                continue
            name = f"ratio__{num}__sur__{denom}"
            series_list.append((df[num] / safe_denom).rename(name))

    if paires_ecartees:
        print(f"  {len(paires_ecartees)} paire(s) numerateur/denominateur ecartee(s) (quasi-tautologiques) :")
        for p in paires_ecartees[:10]:
            print(f"    - {p}")
        if len(paires_ecartees) > 10:
            print(f"    ... et {len(paires_ecartees) - 10} de plus")

    return pd.concat(series_list, axis=1) if series_list else pd.DataFrame(index=df.index)


def generate_mean_aggregation_candidates(df: pd.DataFrame, targets: list[str], group_dims: list[str],
                                          min_group_size: int = MIN_GROUP_SIZE) -> pd.DataFrame:
    series_list = []
    valid_targets = [t for t in targets if t in df.columns]
    for dim in group_dims:
        if dim not in df.columns or not valid_targets:
            continue
        if df[dim].notna().sum() == 0:
            continue  # dimension entierement vide -> groupby casse, on l'ignore
        group_sizes = df.groupby(dim)[dim].transform("size")
        small_group_mask = group_sizes < min_group_size
        group_means = df.groupby(dim)[valid_targets].transform("mean")
        for t in valid_targets:
            name = f"moyenne__{t}__par__{dim}"
            values = group_means[t].where(~small_group_mask)
            series_list.append(values.rename(name))
    return pd.concat(series_list, axis=1) if series_list else pd.DataFrame(index=df.index)


def generate_count_candidates(df: pd.DataFrame, group_dims: list[str]) -> pd.DataFrame:
    """KPI de volume : nombre de lignes partageant la meme valeur pour
    chaque dimension (ex: nombre d'observations par categorie, par annee...)."""
    series_list = []
    for dim in group_dims:
        if dim not in df.columns:
            continue
        if df[dim].notna().sum() == 0:
            continue  # dimension entierement vide -> groupby casse, on l'ignore
        counts = df.groupby(dim)[dim].transform("size")
        name = f"nombre_observations__par__{dim}"
        series_list.append(counts.rename(name))
    return pd.concat(series_list, axis=1) if series_list else pd.DataFrame(index=df.index)


def generate_date_delta_candidates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """Ecart en jours entre CHAQUE paire de colonnes de dates, genere
    mecaniquement (toutes les combinaisons, pas une selection editoriale) -
    la notation en aval filtre ce qui est effectivement informatif."""
    candidates = pd.DataFrame(index=df.index)
    valid_dates = [d for d in date_cols if d in df.columns]
    for date_a, date_b in combinations(valid_dates, 2):
        name = f"delai__{date_a}__moins__{date_b}"
        candidates[name] = (df[date_a] - df[date_b]).dt.days
    return candidates


def generate_all_candidates(df: pd.DataFrame, working_lists: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Retourne (candidats, dims_etendues). dims_etendues est expose pour etre
    REUTILISE tel quel par les appelants qui en ont aussi besoin (ex.
    croisement_candidats.py) - ne JAMAIS rappeler extended_group_dims une
    seconde fois sur ce meme df : elle le mute en place (ajoute les
    colonnes __annee/__mois), et un second appel recompte ces colonnes
    deja presentes comme "numerique a faible cardinalite" EN PLUS de les
    reajouter comme "dimension derivee de date" - provoque une erreur
    pandas en aval (colonnes dupliquees).
    """
    df, dims_etendues = extended_group_dims(
        df, working_lists["group_dims"], working_lists["date_cols"]
    )

    ratios = generate_ratio_candidates(
        df, working_lists["ratio_numerator_candidates"], working_lists["ratio_denominators"]
    )
    moyennes = generate_mean_aggregation_candidates(
        df, working_lists["aggregation_targets"], dims_etendues
    )
    comptages = generate_count_candidates(df, dims_etendues)
    delais = generate_date_delta_candidates(df, working_lists["date_cols"])

    print(f"  ratios              : {ratios.shape[1]}")
    print(f"  moyennes par groupe : {moyennes.shape[1]} (dimensions etendues : {len(dims_etendues)})")
    print(f"  comptages par groupe: {comptages.shape[1]}")
    print(f"  ecarts de dates     : {delais.shape[1]}")

    return pd.concat([ratios, moyennes, comptages, delais], axis=1), dims_etendues


if __name__ == "__main__":
    df = load_real_data()
    print(f"Donnees chargees : {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    working_lists = load_working_lists(df=df)

    candidats, _ = generate_all_candidates(df, working_lists)
    print(f"\nTotal candidats generes : {candidats.shape[1]}")
