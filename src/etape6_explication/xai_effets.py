"""
xai_effets.py (Etape 6a)

Calcule un effet XAI OBJECTIF (statistique, aucun modele de langage) pour
un candidat KPI deja note : comment la variable cible se comporte quand ce
KPI est dans son quartile HAUT vs son quartile BAS.

CHOIX DE LA MEDIANE (PAS LA MOYENNE) ET DES QUARTILES (PAS PEARSON) : si
la variable cible est asymetrique (voir detection_fuite.py), une moyenne
ou une correlation lineaire peuvent etre dominees par une poignee de
valeurs extremes. Comparer les MEDIANES entre le quartile haut et le
quartile bas du KPI est robuste a cette asymetrie et capture aussi bien
une relation non lineaire qu'une relation lineaire.

Ce module ne fait AUCUNE hypothese sur le sens metier du KPI : il se
contente de mesurer un ecart, a charge de la couche modele de langage
(prompt_explication.py) de l'interpreter en s'appuyant sur les
definitions reelles des colonnes.

INPUT  : la serie de valeurs d'un candidat + la cible
OUTPUT : fonction compute_quartile_effect(), importee par generer_explications.py
"""
import pandas as pd

MIN_POINTS_PAR_QUARTILE = 5  # sous ce seuil, l'effet est trop instable pour etre rapporte


def compute_quartile_effect(kpi_values: pd.Series, target: pd.Series) -> dict:
    """
    Compare la MEDIANE de la cible quand le candidat est dans son quartile
    haut (>= q3) vs son quartile bas (<= q1). Retourne calculable=False si
    le candidat n'a pas assez de variabilite ou de donnees pour que la
    comparaison soit fiable.
    """
    valid = kpi_values.notna() & target.notna()
    kpi = kpi_values[valid]
    y = target[valid]

    if kpi.nunique() < 4 or len(kpi) < 2 * MIN_POINTS_PAR_QUARTILE:
        return {"calculable": False}

    q1, q3 = kpi.quantile(0.25), kpi.quantile(0.75)
    if q1 == q3:
        return {"calculable": False}  # candidat quasi-constant sur cet echantillon

    groupe_bas = y[kpi <= q1]
    groupe_haut = y[kpi >= q3]

    if len(groupe_bas) < MIN_POINTS_PAR_QUARTILE or len(groupe_haut) < MIN_POINTS_PAR_QUARTILE:
        return {"calculable": False}

    mediane_bas = float(groupe_bas.median())
    mediane_haut = float(groupe_haut.median())

    return {
        "calculable": True,
        "mediane_cible_quartile_bas": round(mediane_bas, 2),
        "mediane_cible_quartile_haut": round(mediane_haut, 2),
        "direction": "positive" if mediane_haut > mediane_bas else "negative",
        "n_quartile_bas": int(len(groupe_bas)),
        "n_quartile_haut": int(len(groupe_haut)),
    }
