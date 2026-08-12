"""
diagnostic_rapide.py

Test rapide sur un petit lot de colonnes, avant de lancer le run complet
(appel_ollama.py) sur l'ensemble du fichier - permet de verifier en
quelques secondes que la connexion et le modele fonctionnent, sans
attendre le run complet.

Par defaut, teste automatiquement les premieres colonnes necessitant
reellement le modele de langage. Pour un diagnostic cible sur des
colonnes precises (ex: apres avoir modifie le dictionnaire ou le prompt),
remplis la liste NOMS_A_TESTER ci-dessous avec les noms exacts.
"""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_GENERATED_DIR  # noqa: E402
from classification_regles import split_columns  # noqa: E402
from prompt import build_batch_prompt  # noqa: E402
from appel_ollama import call_ollama, NUM_CTX, MODEL_NAME  # noqa: E402

DICT_ENRICHI_PATH = DATA_GENERATED_DIR / "dictionnaire_enrichi.json"

# Laisser vide ([]) pour tester automatiquement les premieres colonnes
# necessitant le modele. Remplir avec des noms precis pour un test cible.
NOMS_A_TESTER: list[str] = []
TAILLE_ECHANTILLON_PAR_DEFAUT = 3

with open(DICT_ENRICHI_PATH, encoding="utf-8") as f:
    records = json.load(f)

_, a_llm = split_columns(records)

if NOMS_A_TESTER:
    a_llm_par_nom = {r["nom"]: r for r in a_llm}
    manquants = [n for n in NOMS_A_TESTER if n not in a_llm_par_nom]
    if manquants:
        raise ValueError(
            f"Colonnes introuvables dans la liste a envoyer au modele (deja "
            f"classees automatiquement, ou nom incorrect) : {manquants}"
        )
    mini_batch = [a_llm_par_nom[n] for n in NOMS_A_TESTER]
else:
    mini_batch = a_llm[:TAILLE_ECHANTILLON_PAR_DEFAUT]

prompt = build_batch_prompt(mini_batch)
print("=== PROMPT ENVOYE (longueur:", len(prompt), "caracteres) ===")
print(prompt)
print("\n=== APPEL DU MODELE:", MODEL_NAME, "avec num_ctx =", NUM_CTX, "===\n")

raw = call_ollama(prompt, model=MODEL_NAME)

print("=== REPONSE BRUTE ===")
print(raw)
print("=== FIN REPONSE BRUTE ===\n")

try:
    parsed = json.loads(raw)
    print("JSON valide. Cles retournees :", list(parsed.keys()))
    print("Noms attendus              :", [r["nom"] for r in mini_batch])
except json.JSONDecodeError as e:
    print("JSON INVALIDE :", e)
