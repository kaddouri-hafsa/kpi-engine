"""
croisement_candidats.py (Etape 4 - raffinement iteratif)

Genere de NOUVEAUX candidats en croisant les MEILLEURS candidats DEJA
NOTES entre eux - inspire du principe de "crossover" des algorithmes
genetiques (cf. brevet US 11,416,533, "System and Method for Automated
Key-Performance-Indicator Discovery"), mais volontairement ALLEGE.

POURQUOI PAS UN ALGORITHME GENETIQUE COMPLET :
1. L'espace de variables (listes de travail deja filtrees par la
   classification) est generalement ENUMERABLE EXHAUSTIVEMENT (de l'ordre
   du millier de candidats) - contrairement au cas vise par le brevet
   (des milliers de variables, ou l'enumeration complete est infaisable).
   La seule vraie justification d'un algorithme genetique (echantillonner
   intelligemment un espace trop grand pour etre teste) ne s'applique
   generalement pas a ce type de fichier.
2. La MUTATION (perturbation aleatoire d'une formule) casserait
   l'interpretabilite - a l'oppose de l'objectif du projet (chaque KPI
   doit rester explicable en langage metier, etape 6). Le CROISEMENT,
   lui, combine deux candidats DEJA VALIDES - il reste interpretable par
   construction.
3. Un algorithme genetique complet ajoute des hyperparametres (taille de
   population, taux de mutation, pression de selection) a regler sans
   assez de donnees reelles pour les valider, plus une composante
   aleatoire qui nuit a la reproductibilite.

2 CROISEMENTS IMPLEMENTES (profondeur bornee a 1 generation - pas de
composition recursive infinie, meme logique d'interpretabilite) :
1. SOMME DE RATIOS A MEME DENOMINATEUR : generalise automatiquement un
   ratio composite (ex: deux ratios partageant la meme base de reference)
   a toute paire de ratios candidats qui partagent un denominateur commun -
   mathematiquement A/D + B/D = (A+B)/D, pas de recalcul depuis les
   colonnes brutes.
2. MOYENNE PAR GROUPE D'UN BON RATIO : un ratio bien note au round
   precedent devient la CIBLE d'une agregation par groupe (ex: "un taux
   moyen par categorie"), sur les dimensions de regroupement deja
   validees - reutilise directement generate_mean_aggregation_candidates.

INPUT  : candidats deja notes (classement du round precedent) + valeurs
         brutes des candidats + infos_metier (etape 3b, pour identifier
         les ratios "metier")
OUTPUT : fonction generate_crossover_candidates(), consommee par pipeline_kpi.py
"""
import re

import pandas as pd

from generation_candidats import generate_mean_aggregation_candidates


def parse_ratio_metadata(technical_name: str, infos_metier: dict) -> tuple:
    """
    Retourne (numerateur_cols, denominateur_cols) si le candidat est un
    ratio SIMPLE (exhaustif "ratio__X__sur__Y", ou metier avec operation
    "ratio"), sinon None. Les autres familles (moyenne, comptage, delai,
    metier brute/complement/somme_ratios, et les candidats de croisement
    eux-memes) ne sont volontairement PAS re-croisees ici - profondeur
    bornee a 1 generation, voir docstring du module.
    """
    m = re.match(r"^ratio__(.+)__sur__(.+)$", technical_name)
    if m:
        return ([m.group(1)], [m.group(2)])
    if technical_name.startswith("metier__"):
        info = infos_metier.get(technical_name)
        if info and info.get("operation") == "ratio":
            return (info["numerateur"], info["denominateur"])
    return None


def croiser_sommes_ratios(candidates: pd.DataFrame, top_k_noms: list, infos_metier: dict,
                           round_label: str) -> tuple:
    """CROISEMENT 1 : voir docstring du module."""
    ratios_par_denom = {}
    for nom in top_k_noms:
        meta = parse_ratio_metadata(nom, infos_metier)
        if meta is None:
            continue
        _, denom = meta
        cle = tuple(sorted(denom))
        ratios_par_denom.setdefault(cle, []).append(nom)

    nouveaux, noms_lisibles = {}, {}
    for noms in ratios_par_denom.values():
        if len(noms) < 2:
            continue
        for i in range(len(noms)):
            for j in range(i + 1, len(noms)):
                a, b = noms[i], noms[j]
                technical_name = f"croise__{round_label}_somme__{a}__ET__{b}"
                nouveaux[technical_name] = candidates[a] + candidates[b]
                noms_lisibles[technical_name] = f"Somme de [{a}] et [{b}] (meme base)"

    resultat = pd.concat(nouveaux, axis=1) if nouveaux else pd.DataFrame(index=candidates.index)
    return resultat, noms_lisibles


def croiser_moyennes_des_meilleurs_ratios(df: pd.DataFrame, candidates: pd.DataFrame, top_k_noms: list,
                                           infos_metier: dict, group_dims: list, round_label: str) -> tuple:
    """CROISEMENT 2 : voir docstring du module."""
    ratio_noms = [n for n in top_k_noms if parse_ratio_metadata(n, infos_metier) is not None]
    if not ratio_noms or not group_dims:
        return pd.DataFrame(index=df.index), {}

    df_augmente = df.copy()
    index_vers_nom = {}
    cibles_temp = []
    for i, nom in enumerate(ratio_noms):
        cible = f"TEMPCIBLE{i}"
        df_augmente[cible] = candidates[nom]
        cibles_temp.append(cible)
        index_vers_nom[cible] = nom

    moyennes = generate_mean_aggregation_candidates(df_augmente, cibles_temp, group_dims)

    renommage, noms_lisibles = {}, {}
    for col in moyennes.columns:
        m = re.match(r"^moyenne__(TEMPCIBLE\d+)__par__(.+)$", col)
        if not m:
            continue
        cible, dim = m.groups()
        nom_ratio = index_vers_nom[cible]
        nouveau_nom = f"croise__{round_label}_moyenne__{nom_ratio}__par__{dim}"
        renommage[col] = nouveau_nom
        noms_lisibles[nouveau_nom] = f"Moyenne de [{nom_ratio}] par {dim}"

    moyennes = moyennes.rename(columns=renommage)[list(renommage.values())] if renommage else pd.DataFrame(index=df.index)
    return moyennes, noms_lisibles


def generate_crossover_candidates(df: pd.DataFrame, candidates: pd.DataFrame, top_k_noms: list,
                                   infos_metier: dict, group_dims: list, round_label: str) -> tuple:
    """Applique les 2 croisements aux top_k_noms (meilleurs candidats du
    round precedent, toutes origines confondues) et retourne
    (nouveaux_candidats, noms_lisibles)."""
    sommes, noms1 = croiser_sommes_ratios(candidates, top_k_noms, infos_metier, round_label)
    moyennes, noms2 = croiser_moyennes_des_meilleurs_ratios(df, candidates, top_k_noms, infos_metier,
                                                             group_dims, round_label)

    print(f"  croisement 'somme de ratios (meme base)' : {sommes.shape[1]} nouveau(x) candidat(s)")
    print(f"  croisement 'moyenne d'un bon ratio'       : {moyennes.shape[1]} nouveau(x) candidat(s)")

    nouveaux = pd.concat([sommes, moyennes], axis=1)
    noms_lisibles = {**noms1, **noms2}
    return nouveaux, noms_lisibles
