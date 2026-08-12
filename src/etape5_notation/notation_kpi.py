"""
notation_kpi.py (Etape 5)

Note chaque candidat genere par l'etape 4 selon 4 criteres independants,
puis supprime les candidats redondants et calcule un score composite pour
le classement final.

1. COMPLETUDE       : part de valeurs non manquantes
2. INFORMATION      : variabilite (coefficient de variation), en rang
                       percentile pour rester robuste aux valeurs extremes
3. PREDICTIF        : importance dans une Random Forest entrainee a predire
                       la cible, MOYENNEE sur plusieurs folds de validation
                       croisee
4. STABILITE        : regularite de la moyenne annuelle du candidat, meme
                       logique de rang percentile que le score d'information

+ SUPPRESSION DE REDONDANCE : filtre applique APRES la notation (propriete
  relative entre candidats, pas un 5e score absolu) - parmi des candidats
  fortement correles entre eux, ne garde que celui au meilleur score
  composite.

INPUT  : DataFrame de candidats (etape 4) + variable cible + colonne
         d'annee pour la stabilite
OUTPUT : DataFrame classee des meilleurs KPI, avec nom lisible
"""
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

RANDOM_SEED = 42

WINSOR_LOWER_QUANTILE = 0.01
WINSOR_UPPER_QUANTILE = 0.99

MIN_COMPLETENESS_RATIO = 0.30   # seuil RELATIF (pas un compte absolu)
N_FOLDS_PREDICTIVE = 5
CORRELATION_REDUNDANCY_THRESHOLD = 0.95

# Garde-fou anti-compensation : un score composite est une MOYENNE ponderee,
# ce qui permet a un candidat quasi nul sur UN SEUL des 4 criteres (ex.
# stabilite = 0, la pire valeur possible) de quand meme bien se classer grace
# aux 3 autres. Ce n'est pas ce qu'on veut d'un KPI "solide" : on exige un
# score minimal sur CHAQUE critere, en plus du composite. 0.10 = ne pas etre
# dans les ~10% les plus faibles des candidats valides sur un critere donne.
MIN_INDIVIDUAL_SCORE = 0.10

# Part MINIMALE (absolue, pas relative) du budget total d'importance RF
# (qui somme a 1 sur l'ensemble des candidats) en dessous de laquelle un
# candidat n'a PAS de pouvoir predictif reellement significatif - meme s'il
# affiche score_predictif = 1.0 apres normalisation par le max (qui ne dit
# que "le moins pire du lot"). Avec des centaines de candidats, une part
# uniforme serait de l'ordre de 1/n ; 0.02 (2%) represente un signal
# nettement au-dessus de ce bruit de fond.
MIN_RAW_IMPORTANCE_SHARE = 0.02

# Poids par defaut. Peuvent etre recalibres via optimiser_poids.py
# (recherche par grille contrainte, plancher 0.10 par critere) pour
# maximiser le rang percentile moyen des KPI "raisonnement_metier" dans le
# classement compose - a refaire sur un jeu de KPI de reference propre a
# chaque nouveau jeu de donnees, ces valeurs par defaut n'ont pas de
# statut universel.
DEFAULT_WEIGHTS = {
    "predictif": 0.60,
    "information": 0.20,
    "stabilite": 0.10,
    "completude": 0.10,
}


# ---------------------------------------------------------------------------
# Nettoyage prealable des candidats (pas des donnees source - deja fait par
# nettoyage_donnees.py en amont de la generation)
# ---------------------------------------------------------------------------
def winsorize_candidates(candidates: pd.DataFrame,
                          lower: float = WINSOR_LOWER_QUANTILE,
                          upper: float = WINSOR_UPPER_QUANTILE) -> pd.DataFrame:
    """Ecrete chaque candidat a ses percentiles [lower, upper] : un
    denominateur proche de zero peut produire un ratio enorme mais fini,
    qui vient ensuite fausser les moyennes annuelles (stabilite)."""
    numeric = candidates.select_dtypes(include="number")
    lower_bounds = numeric.quantile(lower)
    upper_bounds = numeric.quantile(upper)
    clipped = numeric.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
    candidates = candidates.copy()
    candidates[numeric.columns] = clipped
    return candidates


def filter_non_degenerate(candidates: pd.DataFrame,
                           min_completeness: float = MIN_COMPLETENESS_RATIO,
                           variation_relative_min: float = 1e-8) -> pd.DataFrame:
    """
    Retire les candidats degeneres (trop de manquants ou constants) -
    seuil de completude RELATIF, valable quelle que soit la taille du
    jeu de donnees.

    Constance testee via l'ETENDUE (max - min) relative a l'echelle du
    candidat, PAS via std() > 0 : un candidat theoriquement constant (ex.
    "moyenne par groupe" quand la dimension de regroupement est elle-meme
    constante) peut afficher un std() infinitesimal mais non nul (bruit de
    calcul flottant) qui passe a tort un test std() > 0 strict. Comparer
    l'etendue a l'echelle du candidat (plutot qu'a un seuil absolu) reste
    valable que le candidat soit un ratio proche de 0 ou un montant en
    millions.
    """
    candidates = candidates.replace([np.inf, -np.inf], np.nan)
    valides = []
    for c in candidates.columns:
        serie = candidates[c]
        if serie.notna().mean() <= min_completeness:
            continue
        etendue = serie.max() - serie.min()
        if not (etendue > 0):
            continue  # constant exact (ou colonne entierement NaN)
        echelle = max(abs(serie.mean()), abs(serie.max()), 1.0)
        if etendue / echelle > variation_relative_min:
            valides.append(c)
    return candidates[valides]


# ---------------------------------------------------------------------------
# Les 4 scores
# ---------------------------------------------------------------------------
def score_completeness(candidates: pd.DataFrame) -> pd.Series:
    return 1 - candidates.isna().mean()


def score_information(candidates: pd.DataFrame) -> pd.Series:
    means = candidates.mean(numeric_only=True).abs()
    stds = candidates.std(numeric_only=True)
    cv = stds / means.replace(0, np.nan)
    cv = cv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return cv.rank(pct=True)


def impute_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """Matrice entierement imputee (mediane), reutilisee pour la Random
    Forest ET la correlation de redondance - evite l'incoherence d'une
    correlation calculee sur des donnees partiellement manquantes."""
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(
        imputer.fit_transform(candidates),
        columns=candidates.columns,
        index=candidates.index,
    )


def score_predictive_importance(candidates_imputed: pd.DataFrame, target: pd.Series,
                                 n_splits: int = N_FOLDS_PREDICTIVE) -> tuple:
    """
    Importance moyennee sur plusieurs folds de validation croisee - attenue
    le risque de surapprentissage d'un entrainement unique.

    Retourne DEUX series :
    - normalisee (par le max) : utilisee pour le score composite, sur une
      echelle relative [0, 1] ou le meilleur candidat vaut TOUJOURS 1.0 -
      meme si son importance reelle est negligeable en absolu quand il y a
      des centaines de candidats concurrents (dilution naturelle du budget
      total d'importance de la Random Forest, qui somme a 1 sur TOUS les
      candidats).
    - brute (part reelle du budget total d'importance, avant mise a
      l'echelle) : permet de distinguer "le moins pire du lot" d'un
      candidat au pouvoir predictif reellement significatif.
    """
    valid_target = target.notna()
    X = candidates_imputed.loc[valid_target]
    y = target.loc[valid_target]

    if len(X) < n_splits * 2:
        n_splits = max(2, len(X) // 2)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_importances = []
    for train_idx, _ in kf.split(X):
        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_SEED, n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_importances.append(model.feature_importances_)

    importances_brutes = pd.Series(np.mean(fold_importances, axis=0), index=X.columns)
    importances_normalisees = importances_brutes.copy()
    if importances_normalisees.max() > 0:
        importances_normalisees = importances_normalisees / importances_normalisees.max()
    return importances_normalisees, importances_brutes


def score_stability(candidates: pd.DataFrame, year_series: pd.Series) -> pd.Series:
    tmp = candidates.copy()
    tmp["_year_"] = year_series.values
    yearly_means = tmp.groupby("_year_").mean(numeric_only=True)

    cv_over_time = yearly_means.std() / yearly_means.mean().abs().replace(0, np.nan)
    cv_over_time = cv_over_time.replace([np.inf, -np.inf], np.nan)
    cv_over_time = cv_over_time.fillna(cv_over_time.max())

    return 1 - cv_over_time.rank(pct=True)


# ---------------------------------------------------------------------------
# Suppression de redondance (filtre relatif, pas un 5e score)
# ---------------------------------------------------------------------------
def drop_redundant_candidates(candidates_imputed: pd.DataFrame, scores: pd.Series,
                               threshold: float = CORRELATION_REDUNDANCY_THRESHOLD) -> list:
    corr_matrix = candidates_imputed.corr().abs()
    to_drop = set()
    columns_sorted_by_score = scores.sort_values(ascending=False).index.tolist()

    kept = []
    for col in columns_sorted_by_score:
        if col in to_drop:
            continue
        kept.append(col)
        redundant = corr_matrix.index[
            (corr_matrix[col] > threshold) & (corr_matrix.index != col)
        ]
        to_drop.update(redundant)

    return kept


# ---------------------------------------------------------------------------
# Garde-fou anti-compensation (filtre relatif, applique apres le composite)
# ---------------------------------------------------------------------------
def filter_weak_on_any_criterion(candidates: list, predictif: pd.Series, information: pd.Series,
                                  stabilite: pd.Series, completude: pd.Series,
                                  min_percentile: float = MIN_INDIVIDUAL_SCORE) -> list:
    """
    Ecarte les candidats dont AU MOINS UN des 4 scores est dans les
    min_percentile les plus faibles PARMI LES CANDIDATS EVALUES, meme si
    leur score composite est bon - evite qu'un KPI totalement instable ou
    non predictif remonte artificiellement grace aux autres criteres.

    information et stabilite sont deja des rangs percentiles (construits
    via .rank(pct=True)) : comparables directement a min_percentile. predictif
    et completude sont des scores sur une echelle differente (importance
    normalisee par le max, et fraction de non-manquants) - on les convertit
    en rang percentile ICI pour que le seuil ait le meme sens ("ne pas etre
    dans les X% les plus faibles") sur les 4 criteres.
    """
    predictif_pct = predictif.rank(pct=True)
    completude_pct = completude.rank(pct=True)
    return [
        c for c in candidates
        if predictif_pct[c] >= min_percentile
        and information[c] >= min_percentile
        and stabilite[c] >= min_percentile
        and completude_pct[c] >= min_percentile
    ]


# ---------------------------------------------------------------------------
# Nom lisible
# ---------------------------------------------------------------------------
def human_readable_name(candidate_key: str) -> str:
    m = re.match(r"^ratio__(.+)__sur__(.+)$", candidate_key)
    if m:
        num, denom = m.groups()
        return f"Ratio {num} / {denom}"

    m = re.match(r"^moyenne__(.+)__par__(.+)$", candidate_key)
    if m:
        num, dim = m.groups()
        return f"Moyenne de {num} par {dim}"

    m = re.match(r"^nombre_observations__par__(.+)$", candidate_key)
    if m:
        dim = m.group(1)
        return f"Nombre d'observations par {dim}"

    m = re.match(r"^delai__(.+)__moins__(.+)$", candidate_key)
    if m:
        date_a, date_b = m.groups()
        return f"Délai (jours) entre {date_a} et {date_b}"

    return candidate_key.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Pipeline complet
# ---------------------------------------------------------------------------
def run_notation(candidates: pd.DataFrame, target: pd.Series, year_series: pd.Series,
                  top_n: int = 30, weights: dict = None,
                  origine_metier: set = None, noms_lisibles_override: dict = None,
                  origine_labels: dict = None) -> pd.DataFrame:
    """
    origine_metier : ensemble des noms techniques de candidats proposes par
    raisonnement metier a priori (etape 3b), PAS generes par la
    combinatoire exhaustive. Ces candidats :
    - ne sont jamais ecartes par le veto anti-compensation (filter_weak_on_any_criterion) :
      leur pertinence vient d'un raisonnement metier, pas de leur variabilite
      brute sur cet echantillon - un KPI contractuel stable (ex. un taux
      fixe) ne doit pas etre penalise pour sa faible variabilite.
    - sont prioritaires dans la suppression de redondance (drop_redundant_candidates) :
      si un candidat "metier" est fortement correle a un candidat genere
      exhaustivement, c'est le candidat "metier" qui est garde.
    - apparaissent TOUJOURS dans le resultat final, en plus du top_n
      statistique - jamais silencieusement noyes par le seul classement
      statistique.

    origine_labels : etiquette d'affichage PUREMENT COSMETIQUE (colonne
    "origine" du resultat) pour des candidats qui ne rentrent dans aucune
    des deux categories ci-dessus - typiquement les candidats de
    croisement (etape 4, boucle iterative de pipeline_kpi.py). Ces
    candidats sont notes et filtres EXACTEMENT comme la generation
    exhaustive (aucune protection particuliere) : le croisement doit
    prouver sa valeur au meme titre que n'importe quel autre candidat,
    contrairement aux candidats "metier" qui sont garantis par construction.
    """
    origine_metier = origine_metier or set()
    noms_lisibles_override = noms_lisibles_override or {}
    origine_labels = origine_labels or {}
    weights = weights or DEFAULT_WEIGHTS

    print(f"[1/7] {candidates.shape[1]} candidats bruts")

    candidates = winsorize_candidates(candidates)
    candidates = filter_non_degenerate(candidates)
    print(f"[2/7] {candidates.shape[1]} candidats valides apres ecretage/filtrage")

    candidates_imputed = impute_candidates(candidates)
    print("[3/7] Matrice imputee construite (mediane)")

    completude = score_completeness(candidates)
    information = score_information(candidates)
    predictif, predictif_brut = score_predictive_importance(candidates_imputed, target)
    stabilite = score_stability(candidates, year_series)
    print(f"[4/7] Notation multi-criteres terminee (predictif valide par {N_FOLDS_PREDICTIVE}-fold CV)")

    composite = (
        weights["predictif"] * predictif
        + weights["information"] * information
        + weights["stabilite"] * stabilite
        + weights["completude"] * completude
    )

    # Priorite absolue aux candidats "metier" dans le tri glouton de
    # suppression de redondance - ne change PAS leur score affiche, juste
    # l'ordre de traitement pour qu'ils ne soient jamais elimines au profit
    # d'un candidat statistique correle.
    composite_pour_redondance = composite.copy()
    presents = [c for c in origine_metier if c in composite_pour_redondance.index]
    composite_pour_redondance.loc[presents] += 10

    kept = drop_redundant_candidates(candidates_imputed, composite_pour_redondance)
    print(f"[5/7] {len(kept)} candidats retenus apres suppression de redondance")

    kept_metier = [c for c in kept if c in origine_metier]
    kept_stats = [c for c in kept if c not in origine_metier]
    robustes_stats = filter_weak_on_any_criterion(kept_stats, predictif, information, stabilite, completude)
    robustes = kept_metier + robustes_stats
    print(f"[6/7] {len(robustes_stats)} candidats 'generation exhaustive' retenus apres exclusion "
          f"des scores faibles sur un critere isole (< {MIN_INDIVIDUAL_SCORE}) "
          f"+ {len(kept_metier)} candidat(s) 'raisonnement metier' toujours conserves")

    def _origine(c):
        if c in origine_metier:
            return "raisonnement_metier"
        return origine_labels.get(c, "generation_exhaustive")

    result_all = pd.DataFrame({
        "candidat_technique": robustes,
        "origine": [_origine(c) for c in robustes],
        "score_composite": composite.loc[robustes].values,
        "score_predictif": predictif.loc[robustes].values,
        "signal_predictif_significatif": (predictif_brut.loc[robustes] >= MIN_RAW_IMPORTANCE_SHARE).values,
        "score_information": information.loc[robustes].values,
        "score_stabilite": stabilite.loc[robustes].values,
        "completude": completude.loc[robustes].values,
    })
    result_all["nom_lisible"] = result_all["candidat_technique"].apply(
        lambda c: noms_lisibles_override.get(c) or human_readable_name(c)
    )
    result_all = result_all.sort_values("score_composite", ascending=False).reset_index(drop=True)

    # Les candidats "metier" sont TOUS gardes, quel que soit leur rang -
    # sinon un simple head(top_n) pourrait quand meme les faire disparaitre
    # du resultat final si leur score composite est faible.
    result_metier = result_all[result_all["origine"] == "raisonnement_metier"]
    result_stats = result_all[result_all["origine"] == "generation_exhaustive"].head(top_n)
    result = pd.concat([result_metier, result_stats]).sort_values(
        "score_composite", ascending=False
    ).reset_index(drop=True)

    n_signal_faible = (~result["signal_predictif_significatif"]).sum()
    if n_signal_faible > 0:
        print(f"  ATTENTION : {n_signal_faible}/{len(result)} candidat(s) du classement ont un "
              f"score_predictif eleve UNIQUEMENT par normalisation relative (au meilleur du lot), "
              f"mais une part reelle d'importance RF < {MIN_RAW_IMPORTANCE_SHARE} - voir la colonne "
              f"signal_predictif_significatif. Ces KPI restent valables sur les 3 autres criteres "
              f"(information/stabilite/completude) mais ne doivent PAS etre presentes comme "
              f"'expliquant la performance' au sens statistique.")
    print(f"[7/7] {len(result_metier)} KPI 'raisonnement metier' + top {top_n} KPI 'generation "
          f"exhaustive' prets ({len(result)} au total)")

    return result
