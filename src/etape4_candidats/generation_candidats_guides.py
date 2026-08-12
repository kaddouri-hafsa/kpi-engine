"""
generation_candidats_guides.py (Etape 4 - generation guidee)

Calcule les candidats KPI PROPOSES PAR RAISONNEMENT METIER (etape 3b,
resultats/kpi_proposes_metier.json) - complementaire a la generation
exhaustive de generation_candidats.py, jamais un remplacement.

Difference cle avec la generation exhaustive : chaque candidat ici vient
d'une PROPOSITION explicite motivee par le sens metier, pas d'une
combinaison mecanique parmi des milliers. Ces candidats sont marques avec
le prefixe "metier__" pour etre distingues en aval (etape 5, notation_kpi.py)
et ne jamais etre silencieusement ecartes par le seul filtre statistique.

4 operations supportees (voir prompt_raisonnement.py pour le schema) :
- ratio            : somme(numerateur) / somme(denominateur)
- brute            : une colonne deja exploitable telle quelle
- complement       : 1 - colonne (garde-fou : uniquement si la colonne est
                     bien un taux/pourcentage borne, sinon incoherent)
- somme_ratios     : somme de plusieurs ratios (ex: un ratio composite)

INPUT  : resultats/kpi_proposes_metier.json (etape 3b) + jeu de donnees actif
OUTPUT : fonction generate_guided_candidates(), consommee par pipeline_kpi.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
from config import KPI_METIER_PATH  # noqa: E402

# Part minimale des valeurs non-nulles devant tomber dans [0,1] ou [0,100]
# pour qu'un "complement" (1 - X) soit juge calculable de facon fiable -
# garde-fou objectif, evite de calculer un "complement" incoherent sur une
# colonne qui n'est pas reellement un taux/pourcentage borne.
SEUIL_BORNE_COMPLEMENT = 0.9


def load_propositions(path: Path = KPI_METIER_PATH) -> list[dict]:
    if not path.exists():
        print(f"ATTENTION : {path.name} introuvable - aucun candidat issu du raisonnement "
              f"metier ne sera genere. Lance proposer_kpi_metier.py (etape 3b) d'abord si "
              f"tu veux les inclure.")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _colonnes_non_numeriques(df: pd.DataFrame, colonnes: list[str]) -> list[str]:
    """Le modele de langage peut halluciner qu'une colonne categorielle/texte
    est une grandeur numerique exploitable dans un ratio - garde-fou
    objectif, independant de son jugement : verifie le dtype REELLEMENT
    observe, pas ce que la proposition affirme. Sans ce controle,
    df[colonnes].sum(axis=1) sur des colonnes texte concatene les chaines
    au lieu de lever une erreur immediate, et l'echec (division de deux
    chaines) survient plus loin, avec un message bien moins clair."""
    return [c for c in colonnes if not pd.api.types.is_numeric_dtype(df[c])]


def _sum_columns(df: pd.DataFrame, colonnes: list[str]) -> pd.Series:
    return df[colonnes].sum(axis=1, skipna=False)


def _compute_ratio(df: pd.DataFrame, numerateur: list[str], denominateur: list[str]) -> pd.Series:
    num = _sum_columns(df, numerateur)
    denom = _sum_columns(df, denominateur).replace(0, np.nan)
    return num / denom


def _technical_name(nom: str, dejas_utilises: set) -> str:
    base = f"metier__{nom.lower().replace(' ', '_')}"
    technical_name = base
    suffixe = 2
    while technical_name in dejas_utilises:
        technical_name = f"{base}_{suffixe}"
        suffixe += 1
    return technical_name


def generate_guided_candidates(df: pd.DataFrame, propositions: list[dict] = None) -> tuple[pd.DataFrame, dict]:
    """
    Retourne (candidats, infos_metier). infos_metier associe chaque nom
    technique a la proposition COMPLETE d'origine (nom, operation, colonnes
    impliquees, justification) - reutilise telle quelle par pipeline_kpi.py
    (juste le "nom") et par generer_explications.py (etape 6, qui a besoin
    des colonnes et de l'operation pour construire l'explication).
    """
    if propositions is None:
        propositions = load_propositions()

    series_list = []
    infos_metier = {}
    ecartees = []

    for kpi in propositions:
        nom = kpi.get("nom", "kpi_sans_nom")
        operation = kpi.get("operation")

        try:
            if operation == "ratio":
                num, denom = kpi["numerateur"], kpi["denominateur"]
                if not all(c in df.columns for c in num + denom):
                    raise KeyError("colonne absente du jeu de donnees actif")
                non_numeriques = _colonnes_non_numeriques(df, num + denom)
                if non_numeriques:
                    raise ValueError(f"colonne(s) non numerique(s), ratio non calculable : {non_numeriques}")
                serie = _compute_ratio(df, num, denom)

            elif operation == "brute":
                col = kpi["colonne"]
                if col not in df.columns:
                    raise KeyError("colonne absente du jeu de donnees actif")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"colonne '{col}' non numerique, KPI non calculable")
                serie = df[col]

            elif operation == "complement":
                col = kpi["colonne"]
                if col not in df.columns:
                    raise KeyError("colonne absente du jeu de donnees actif")
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"colonne '{col}' non numerique, complement non calculable")
                valeurs = df[col].dropna()
                if len(valeurs) == 0:
                    raise ValueError("colonne entierement vide")
                if valeurs.between(0, 1).mean() > SEUIL_BORNE_COMPLEMENT:
                    serie = 1 - df[col]
                elif valeurs.between(0, 100).mean() > SEUIL_BORNE_COMPLEMENT:
                    serie = 100 - df[col]
                else:
                    raise ValueError(
                        f"valeurs hors [0,1]/[0,100] (max observe = {valeurs.max():.2f}), "
                        f"complement non calculable de facon fiable"
                    )

            elif operation == "somme_ratios":
                composantes = []
                for r in kpi["ratios"]:
                    cols_ratio = r["numerateur"] + r["denominateur"]
                    if not all(c in df.columns for c in cols_ratio):
                        raise KeyError("colonne absente du jeu de donnees actif")
                    non_numeriques = _colonnes_non_numeriques(df, cols_ratio)
                    if non_numeriques:
                        raise ValueError(f"colonne(s) non numerique(s), ratio non calculable : {non_numeriques}")
                    composantes.append(_compute_ratio(df, r["numerateur"], r["denominateur"]))
                serie = sum(composantes)

            else:
                raise ValueError(f"operation inconnue : {operation!r}")

            technical_name = _technical_name(nom, set(infos_metier))
            series_list.append(serie.rename(technical_name))
            infos_metier[technical_name] = kpi

        except (KeyError, ValueError) as e:
            ecartees.append(f"{nom} : {e}")

    if ecartees:
        print(f"  {len(ecartees)} KPI 'metier' ecarte(s) (colonne absente ou incoherence) :")
        for e in ecartees:
            print(f"    - {e}")

    print(f"  candidats 'metier'    : {len(series_list)}")

    candidats = pd.concat(series_list, axis=1) if series_list else pd.DataFrame(index=df.index)
    return candidats, infos_metier


if __name__ == "__main__":
    from generation_candidats import load_real_data

    df = load_real_data()
    propositions = load_propositions()
    candidats, infos_metier = generate_guided_candidates(df, propositions)

    print(f"\n{candidats.shape[1]} candidats 'metier' generes :")
    for tech, info in infos_metier.items():
        print(f"  {tech:50s} -> {info['nom']}")
