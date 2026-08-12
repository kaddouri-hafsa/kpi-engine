"""
optimiser_poids.py (Etape 5 - analyse)

Cherche les poids du score composite qui favorisent le mieux les KPI
"raisonnement metier" (etape 3b) dans le classement final, plutot que de
garder les poids par defaut (jamais valides empiriquement - voir
DEFAULT_WEIGHTS dans notation_kpi.py).

METHODE : recherche par grille sur le simplexe des poids (chaque poids >=
0, les 4 sommant a 1). Pour chaque combinaison, on mesure le RANG
PERCENTILE MOYEN des candidats "raisonnement_metier" dans le classement
composite resultant, parmi TOUS les candidats valides (pas seulement le
top tronque). Les 4 scores bruts (completude, information, predictif,
stabilite) sont calcules UNE SEULE FOIS - l'entrainement Random Forest
(predictif) est la partie couteuse ; la recherche de poids elle-meme
n'est ensuite qu'une recombinaison lineaire tres rapide, repetee sur
toute la grille.

DEUX RECHERCHES SONT FAITES :
1. Sans contrainte - peut converger vers une solution DEGENEREE (tout le
   poids sur un seul critere) si les KPI metier se distinguent surtout
   par CE critere - une solution qui gagne "sur le papier" mais qui
   trahit l'esprit d'une notation a 4 criteres INDEPENDANTS.
2. Avec un plancher minimal par critere (chaque poids >= POIDS_MIN) -
   garantit que les 4 criteres comptent reellement, quitte a moins bien
   "gagner" sur l'objectif brut.

LIMITE IMPORTANTE A GARDER EN TETE : cette recherche s'appuie sur les KPI
"metier" (etape 3b) proposes pour LE fichier de donnees actif - risque
reel de surapprentissage (poids tailles sur mesure pour ces candidats
precis, pas necessairement generalisables). A refaire pour chaque nouveau
jeu de donnees, avec autant de KPI de reference valides que possible.

INPUT  : le jeu de donnees actif (config.DATA_PATH)
OUTPUT : affichage des poids trouves + comparaison avec DEFAULT_WEIGHTS
"""
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape4_candidats"))

from config import TARGET_COL  # noqa: E402
from charger_classification import load_working_lists, resolve_year_series  # noqa: E402
from generation_candidats import load_real_data, generate_all_candidates  # noqa: E402
from generation_candidats_guides import load_propositions, generate_guided_candidates  # noqa: E402

from notation_kpi import (  # noqa: E402
    winsorize_candidates, filter_non_degenerate, impute_candidates,
    score_completeness, score_information, score_predictive_importance, score_stability,
    DEFAULT_WEIGHTS,
)

PAS_GRILLE = 0.05        # granularite de la recherche (0.05 = 21^3 combinaisons apres contrainte de somme)
POIDS_MIN_CONTRAINT = 0.10  # plancher par critere pour la recherche "contrainte" (garde les 4 criteres actifs)


def calculer_scores_bruts():
    """Calcule UNE FOIS les 4 scores bruts + l'origine de chaque candidat
    valide - reutilise pour toute la recherche de poids."""
    print("Chargement des donnees et generation des candidats...")
    df = load_real_data()
    working_lists = load_working_lists(df=df)
    candidates_exhaustifs, _ = generate_all_candidates(df, working_lists)

    propositions = load_propositions()
    candidates_metier, infos_metier = generate_guided_candidates(df, propositions)
    origine_metier = set(candidates_metier.columns)

    candidates = pd.concat([candidates_exhaustifs, candidates_metier], axis=1)
    candidates = candidates.loc[:, ~candidates.columns.duplicated()]

    candidates = winsorize_candidates(candidates)
    candidates = filter_non_degenerate(candidates)
    print(f"{candidates.shape[1]} candidats valides ({len([c for c in candidates.columns if c in origine_metier])} 'metier' parmi eux)")

    candidates_imputed = impute_candidates(candidates)

    completude = score_completeness(candidates)
    information = score_information(candidates)
    print("Entrainement Random Forest (5-fold CV) - seule etape couteuse...")
    predictif, _ = score_predictive_importance(candidates_imputed, df[TARGET_COL])
    stabilite = score_stability(candidates, resolve_year_series(df, working_lists))

    return {
        "completude": completude, "information": information,
        "predictif": predictif, "stabilite": stabilite,
    }, origine_metier & set(candidates.columns)


def rang_percentile_moyen_metier(scores: dict, poids: dict, origine_metier: set) -> float:
    composite = (
        poids["predictif"] * scores["predictif"]
        + poids["information"] * scores["information"]
        + poids["stabilite"] * scores["stabilite"]
        + poids["completude"] * scores["completude"]
    )
    rangs = composite.rank(pct=True)
    return rangs.loc[list(origine_metier)].mean()


def grille_des_poids(pas: float = PAS_GRILLE, poids_min: float = 0.0):
    """Genere toutes les combinaisons (predictif, information, stabilite,
    completude) multiples de `pas`, >= poids_min chacun, sommant a 1."""
    n = round(1 / pas)
    n_min = round(poids_min / pas)
    for a, b, c in product(range(n_min, n - 3 * n_min + 1), repeat=3):
        d = n - a - b - c
        if d < n_min:
            continue
        yield {
            "predictif": a * pas, "information": b * pas,
            "stabilite": c * pas, "completude": d * pas,
        }


def rechercher_meilleurs_poids(scores: dict, origine_metier: set, poids_min: float = 0.0) -> tuple[dict, float]:
    meilleur_poids, meilleur_score = None, -1.0
    for poids in grille_des_poids(PAS_GRILLE, poids_min):
        val = rang_percentile_moyen_metier(scores, poids, origine_metier)
        if val > meilleur_score:
            meilleur_score, meilleur_poids = val, poids
    return meilleur_poids, meilleur_score


if __name__ == "__main__":
    scores, origine_metier = calculer_scores_bruts()

    score_actuel = rang_percentile_moyen_metier(scores, DEFAULT_WEIGHTS, origine_metier)
    print(f"\nPoids ACTUELS {DEFAULT_WEIGHTS}")
    print(f"  -> rang percentile moyen des KPI 'metier' : {score_actuel:.3f}")

    print("\n=== Recherche SANS contrainte (peut degenerer sur un seul critere) ===")
    poids_libre, val_libre = rechercher_meilleurs_poids(scores, origine_metier, poids_min=0.0)
    print(f"Meilleurs poids trouves : {poids_libre}")
    print(f"  -> rang percentile moyen des KPI 'metier' : {val_libre:.3f}")

    print(f"\n=== Recherche CONTRAINTE (chaque poids >= {POIDS_MIN_CONTRAINT}, garde les 4 criteres actifs) ===")
    poids_contraint, val_contraint = rechercher_meilleurs_poids(scores, origine_metier, poids_min=POIDS_MIN_CONTRAINT)
    print(f"Meilleurs poids trouves : {poids_contraint}")
    print(f"  -> rang percentile moyen des KPI 'metier' : {val_contraint:.3f}")
