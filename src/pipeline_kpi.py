"""
pipeline_kpi.py

Orchestration complete : validation/nettoyage du fichier de donnees
(etape 0) -> classification semantique (etape 3, deja faite au prealable,
resultats/classification_finale.json) -> generation des candidats
(exhaustive, etape 4 + raisonnement metier a priori, etape 3b) -> BOUCLE
ITERATIVE de notation + croisement des meilleurs candidats (etape 4, sur
N_ROUNDS cycles) -> export du classement final (etape 5).

La boucle iterative (inspiree du brevet US 11,416,533 sur la decouverte
automatique de KPI, mais volontairement allegee - voir croisement_candidats.py
pour la justification detaillee de ne PAS utiliser un algorithme genetique
complet) : a chaque round, les meilleurs candidats du round precedent
(toutes origines confondues) sont croises entre eux pour creer de nouveaux
candidats, qui rentrent en competition normale au round suivant. Seul le
dernier round determine le classement final exporte.

Les candidats "metier" dependent de resultats/kpi_proposes_metier.json,
produit manuellement par etape3b_raisonnement_metier/proposer_kpi_metier.py
(appel au modele de langage, comme la classification de l'etape 3 - pas
rappele a chaque lancement de ce pipeline). Si ce fichier n'existe pas
encore, le pipeline continue avec la seule generation exhaustive
(avertissement affiche).
"""
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "etape4_candidats"))
sys.path.insert(0, str(SCRIPT_DIR / "etape5_notation"))

from config import TARGET_COL, TOP_KPI_PATH  # noqa: E402
from charger_classification import load_working_lists, resolve_year_series  # noqa: E402
from generation_candidats import load_real_data, generate_all_candidates  # noqa: E402
from generation_candidats_guides import load_propositions, generate_guided_candidates  # noqa: E402
from croisement_candidats import generate_crossover_candidates, parse_ratio_metadata  # noqa: E402
from notation_kpi import run_notation  # noqa: E402

OUTPUT_PATH = TOP_KPI_PATH

TOP_N = 30
N_ROUNDS = 3            # generation 0 (initiale) + jusqu'a 2 rounds de croisement
TOP_K_CROISEMENT = 15   # nombre de meilleurs RATIOS consideres pour le croisement a chaque round
# Pendant les rounds intermediaires, on demande a run_notation de ne PAS
# tronquer le resultat (top_n tres large) : le croisement ne sait combiner
# que des candidats "ratio", qui peuvent tres bien ne pas figurer dans le
# top 15-30 TOUTES FAMILLES CONFONDUES (les candidats "moyenne"/"comptage"
# peuvent dominer le classement global, laissant 0 ratio dans le top 15,
# donc 0 croisement possible). Il faut donc regarder l'ensemble des
# candidats valides pour en extraire les meilleurs RATIOS specifiquement,
# plutot que de se fier au classement global tronque.
TOP_N_INTERMEDIAIRE = 100_000


def main():
    print("=== Validation et nettoyage des donnees (etape 0) ===")
    df = load_real_data()
    print(f"Donnees pretes : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    print("\n=== Chargement de la classification (etape 3) + garde-fous statistiques ===")
    working_lists = load_working_lists(df=df)

    print("\n=== Generation des candidats KPI - exhaustive (etape 4) ===")
    candidates_exhaustifs, dims_etendues = generate_all_candidates(df, working_lists)

    print("\n=== Generation des candidats KPI - raisonnement metier (etape 3b) ===")
    propositions = load_propositions()
    candidates_metier, infos_metier = generate_guided_candidates(df, propositions)
    origine_metier = set(candidates_metier.columns)
    noms_lisibles = {k: v["nom"] for k, v in infos_metier.items()}

    candidates = pd.concat([candidates_exhaustifs, candidates_metier], axis=1)
    print(f"  total combine (round 0) : {candidates.shape[1]} candidats "
          f"({candidates_exhaustifs.shape[1]} exhaustifs + {candidates_metier.shape[1]} metier)")

    target = df[TARGET_COL]
    year_series = resolve_year_series(df, working_lists)
    origine_labels = {}

    top_kpis = None
    for round_num in range(1, N_ROUNDS + 1):
        dernier_round = (round_num == N_ROUNDS)
        top_n_round = TOP_N if dernier_round else TOP_N_INTERMEDIAIRE

        print(f"\n=== Round {round_num}/{N_ROUNDS} : notation multi-criteres ({candidates.shape[1]} candidats) ===")
        top_kpis = run_notation(candidates, target, year_series, top_n=top_n_round,
                                 origine_metier=origine_metier, noms_lisibles_override=noms_lisibles,
                                 origine_labels=origine_labels)

        if dernier_round:
            break

        # Le classement est deja trie par score_composite decroissant -
        # filtrer aux "ratio" (exhaustifs ou metier) puis prendre les
        # K premiers preserve l'ordre "meilleur ratio d'abord".
        candidats_ratio = [
            c for c in top_kpis["candidat_technique"]
            if parse_ratio_metadata(c, infos_metier) is not None
        ]
        top_k_noms = candidats_ratio[:TOP_K_CROISEMENT]
        print(f"  ({len(top_kpis)} candidats valides au total, dont {len(candidats_ratio)} de type 'ratio')")
        print(f"\n=== Round {round_num}/{N_ROUNDS} : croisement des {len(top_k_noms)} meilleurs ratios ===")
        nouveaux, noms_nouveaux = generate_crossover_candidates(
            df, candidates, top_k_noms, infos_metier, dims_etendues, round_label=f"r{round_num}"
        )
        nouveaux = nouveaux.loc[:, ~nouveaux.columns.duplicated()]  # garde-fou defensif
        nouveaux = nouveaux[[c for c in nouveaux.columns if c not in candidates.columns]]

        if nouveaux.shape[1] == 0:
            print("  aucun nouveau candidat produit par le croisement - arret anticipe de la boucle")
            break

        candidates = pd.concat([candidates, nouveaux], axis=1)
        candidates = candidates.loc[:, ~candidates.columns.duplicated()]  # garde-fou defensif
        noms_lisibles.update(noms_nouveaux)
        for c in nouveaux.columns:
            origine_labels[c] = f"round{round_num}_croisement"

    print(f"\n=== KPI retenus (raisonnement metier garantis + top {TOP_N} tous rounds confondus) ===")
    cols = ["nom_lisible", "origine", "score_composite", "score_predictif",
            "signal_predictif_significatif", "score_information", "score_stabilite", "completude"]
    print(top_kpis[cols].to_string(index=False))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    top_kpis.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResultat exporte : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
