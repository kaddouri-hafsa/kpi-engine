"""
exporter_dax.py (Etape 7)

Traduit resultats/top_kpi.csv (KPI retenus) en un script C# "Advanced
Scripting" pour Tabular Editor (gratuit, s'ajoute comme External Tool dans
le ruban Power BI Desktop) : un seul "Run Script" cree une colonne
calculee par KPI dans la table importee, sans avoir a retaper chaque
mesure a la main. Power BI n'est pas dans le perimetre de cet outil - ce
script se contente de produire le fichier a importer, l'utilisateur
construit son tableau de bord lui-meme dans Power BI Desktop.

Choix de COLONNE CALCULEE (pas mesure) pour tous les KPI : le pipeline
Python calcule chaque candidat LIGNE PAR LIGNE (un ratio/une moyenne de
groupe/un delai par observation), jamais un agregat global - une mesure
DAX classique (ex. DIVIDE(SUM(...), SUM(...))) ne reproduirait cette
semantique que par coincidence. La colonne calculee est le seul objet DAX
qui correspond exactement a ce que le pipeline a note.

Deux origines de KPI a decoder differemment :
- generation_exhaustive (etape 4) : le nom technique encode la formule
  (ratio__X__sur__Y, moyenne__X__par__Y, nombre_observations__par__Y,
  delai__X__moins__Y) - voir generation_candidats.py pour la grammaire.
- raisonnement_metier (etape 4, generation guidee) : la formule n'est PAS
  dans le nom, elle vit dans resultats/kpi_proposes_metier.json (operation,
  numerateur/denominateur/colonne/ratios) - on reconstruit le mapping nom
  technique -> proposition via generate_guided_candidates(), la meme
  fonction que le pipeline utilise, pour ne jamais desynchroniser la
  logique de decodage.

INPUT  : resultats/top_kpi.csv + resultats/kpi_proposes_metier.json + jeu
         de donnees actif (pour re-decider "complement" comme le pipeline
         l'a fait, et verifier l'existence des colonnes).
OUTPUT : resultats/mesures_powerbi.csx (a coller dans Tabular Editor ->
         Advanced Scripting -> Run Script, sur le modele Power BI ouvert
         avec la table de donnees deja importee).
"""
import csv
import os
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src" / "etape4_candidats"))

from config import DATA_SHEET, TOP_KPI_PATH, DAX_EXPORT_PATH  # noqa: E402
from generation_candidats import load_real_data  # noqa: E402
from generation_candidats_guides import load_propositions, generate_guided_candidates  # noqa: E402
from charger_classification import MIN_GROUP_SIZE  # noqa: E402

OUTPUT_PATH = DAX_EXPORT_PATH
# Nom de la table dans Power BI apres import : par defaut, Power BI nomme
# la table d'apres le nom de la feuille Excel importee - a surcharger via
# POWERBI_TABLE_NAME si l'utilisateur l'a renommee dans Power BI.
TABLE_NAME = os.environ.get("POWERBI_TABLE_NAME", DATA_SHEET)

# Seuil de "complement" (1-X vs 100-X), duplique depuis
# generation_candidats_guides.py pour reproduire EXACTEMENT la meme
# decision au moment de l'export (fait une seule fois ici, sur les memes
# donnees, plutot que de la re-derivera dynamiquement en DAX).
SEUIL_BORNE_COMPLEMENT = 0.9

RATIO_RE = re.compile(r"^ratio__(.+)__sur__(.+)$")
MOYENNE_RE = re.compile(r"^moyenne__(.+)__par__(.+)$")
NOMBRE_RE = re.compile(r"^nombre_observations__par__(.+)$")
DELAI_RE = re.compile(r"^delai__(.+)__moins__(.+)$")

INTERDITS_NOM = str.maketrans("", "", "[]'\":;,")


def dax_col(colonne: str) -> str:
    return f"'{TABLE_NAME}'[{colonne}]"


def dim_expr(dim: str) -> str:
    """Une dimension de regroupement peut etre une vraie colonne ou une
    partie de date derivee mecaniquement (__annee / __mois, voir
    derive_date_parts dans generation_candidats.py)."""
    if dim.endswith("__annee"):
        return f"YEAR({dax_col(dim[:-len('__annee')])})"
    if dim.endswith("__mois"):
        return f'FORMAT({dax_col(dim[:-len("__mois")])}, "YYYY-MM")'
    return dax_col(dim)


def dim_base_column(dim: str) -> str:
    return dim[:-len("__annee")] if dim.endswith("__annee") else (
        dim[:-len("__mois")] if dim.endswith("__mois") else dim
    )


def decode_exhaustive(technical_name: str, colonnes_dispo: set) -> str | None:
    m = RATIO_RE.match(technical_name)
    if m:
        num, denom = m.groups()
        if num not in colonnes_dispo or denom not in colonnes_dispo:
            return None
        return f"DIVIDE({dax_col(num)}, {dax_col(denom)})"

    m = MOYENNE_RE.match(technical_name)
    if m:
        target, dim = m.groups()
        if target not in colonnes_dispo or dim_base_column(dim) not in colonnes_dispo:
            return None
        expr = dim_expr(dim)
        return (
            f"VAR CurrentDim = {expr}\n"
            f"    VAR MemeGroupe = FILTER('{TABLE_NAME}', {expr} = CurrentDim)\n"
            f"    VAR TailleGroupe = COUNTROWS(MemeGroupe)\n"
            f"    RETURN IF(TailleGroupe >= {MIN_GROUP_SIZE}, AVERAGEX(MemeGroupe, {dax_col(target)}))"
        )

    m = NOMBRE_RE.match(technical_name)
    if m:
        dim = m.group(1)
        if dim_base_column(dim) not in colonnes_dispo:
            return None
        expr = dim_expr(dim)
        return (
            f"VAR CurrentDim = {expr}\n"
            f"    RETURN COUNTROWS(FILTER('{TABLE_NAME}', {expr} = CurrentDim))"
        )

    m = DELAI_RE.match(technical_name)
    if m:
        date_a, date_b = m.groups()
        if date_a not in colonnes_dispo or date_b not in colonnes_dispo:
            return None
        return f"DATEDIFF({dax_col(date_b)}, {dax_col(date_a)}, DAY)"

    return None


def _sum_expr(colonnes: list) -> str:
    return " + ".join(dax_col(c) for c in colonnes)


def decode_metier(kpi: dict, df, colonnes_dispo: set) -> str | None:
    operation = kpi.get("operation")

    if operation == "brute":
        col = kpi["colonne"]
        return dax_col(col) if col in colonnes_dispo else None

    if operation == "ratio":
        num, denom = kpi["numerateur"], kpi["denominateur"]
        if not all(c in colonnes_dispo for c in num + denom):
            return None
        return f"DIVIDE({_sum_expr(num)}, {_sum_expr(denom)})"

    if operation == "complement":
        col = kpi["colonne"]
        if col not in colonnes_dispo:
            return None
        valeurs = df[col].dropna()
        if len(valeurs) == 0:
            return None
        if valeurs.between(0, 1).mean() > SEUIL_BORNE_COMPLEMENT:
            return f"1 - {dax_col(col)}"
        if valeurs.between(0, 100).mean() > SEUIL_BORNE_COMPLEMENT:
            return f"100 - {dax_col(col)}"
        return None

    if operation == "somme_ratios":
        termes = []
        for r in kpi["ratios"]:
            if not all(c in colonnes_dispo for c in r["numerateur"] + r["denominateur"]):
                return None
            termes.append(f"DIVIDE({_sum_expr(r['numerateur'])}, {_sum_expr(r['denominateur'])})")
        return " + ".join(termes)

    return None


def nom_dax_valide(nom: str) -> str:
    return nom.translate(INTERDITS_NOM).strip()


def dedupliquer_noms(noms: list) -> list:
    """Deux KPI 'metier' peuvent partager le meme nom_lisible avec une
    formule differente (un modele de langage peut proposer deux fois le
    meme intitule pour des colonnes distinctes) - on numerote les doublons
    pour ne jamais ecraser une colonne calculee par une autre dans le meme
    script."""
    vus = {}
    resultat = []
    for nom in noms:
        vus[nom] = vus.get(nom, 0) + 1
        resultat.append(nom if vus[nom] == 1 else f"{nom} ({vus[nom]})")
    return resultat


def csharp_verbatim(expr_dax: str) -> str:
    return expr_dax.replace('"', '""')


def build_script() -> str:
    df = load_real_data()
    colonnes_dispo = set(df.columns)

    propositions = load_propositions()
    _, infos_metier = generate_guided_candidates(df, propositions)

    with open(TOP_KPI_PATH, encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))

    noms_bruts, expressions, ignores = [], [], []

    for ligne in lignes:
        technical_name = ligne["candidat_technique"]
        nom_lisible = ligne["nom_lisible"]
        origine = ligne["origine"]

        if origine == "raisonnement_metier":
            kpi = infos_metier.get(technical_name)
            expr = decode_metier(kpi, df, colonnes_dispo) if kpi else None
        else:
            expr = decode_exhaustive(technical_name, colonnes_dispo)

        if expr is None:
            ignores.append(f"{technical_name} ({nom_lisible}) : formule non reconstructible")
            continue

        noms_bruts.append(nom_lisible)
        expressions.append(expr)

    noms_finaux = dedupliquer_noms([nom_dax_valide(n) for n in noms_bruts])

    if ignores:
        print(f"{len(ignores)} KPI ignore(s) a l'export (colonne absente ou formule non decodable) :")
        for i in ignores:
            print(f"  - {i}")

    lignes_script = [
        "// Genere par exporter_dax.py - a coller dans Tabular Editor",
        "// (External Tools -> Tabular Editor -> Advanced Scripting -> Run Script)",
        f"// {len(noms_finaux)} KPI, sur la table '{TABLE_NAME}'",
        "",
        f'var t = Model.Tables["{TABLE_NAME}"];',
        "",
    ]
    for nom, expr in zip(noms_finaux, expressions):
        lignes_script.append(f't.AddCalculatedColumn(@"{nom}", @"{csharp_verbatim(expr)}");')

    return "\n".join(lignes_script) + "\n"


def main():
    script = build_script()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(script, encoding="utf-8")
    print(f"\nScript ecrit : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

