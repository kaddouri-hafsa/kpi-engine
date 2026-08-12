# Moteur de découverte de KPI

Pipeline qui combine raisonnement sémantique par modèle de langage (LLM local, pour la confidentialité) et validation statistique objective pour découvrir, noter et expliquer automatiquement des indicateurs de performance (KPI) à partir de **n'importe quel fichier de données tabulaire** accompagné de son dictionnaire de données.

Complété par un module indépendant de **text-to-SQL** (interroger la base en langage naturel, interface de chat Streamlit incluse).

> Power BI n'est pas dans le périmètre de cet outil : le pipeline génère un script prêt à importer (`resultats/mesures_powerbi.csx`) que tu charges toi-même dans Power BI via Tabular Editor. Voir [Restitution Power BI](#restitution-power-bi-optionnel).

## Sommaire

- [Principe](#principe)
- [Prérequis machine](#prérequis-machine)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation — pipeline de découverte de KPI](#utilisation--pipeline-de-découverte-de-kpi)
- [Utilisation — module text-to-SQL](#utilisation--module-text-to-sql)
- [Restitution Power BI (optionnel)](#restitution-power-bi-optionnel)
- [Structure du projet](#structure-du-projet)
- [Limites connues](#limites-connues)

## Principe

Le pipeline s'organise en 7 couches successives : classification sémantique du dictionnaire de données, raisonnement métier a priori (propositions de KPI à partir des seules définitions, avant de voir les données), génération exhaustive de candidats, raffinement itératif par croisement, notation multi-critères (complétude, information, pouvoir prédictif validé par Random Forest, stabilité temporelle), optimisation/filtrage, et enfin explication en langage naturel de chaque KPI retenu.

À chaque étape, le jugement du modèle de langage est systématiquement croisé avec un calcul statistique objectif — aucune décision du LLM n'est appliquée telle quelle.

## Prérequis machine

| | Minimum (fonctionnel, mais lent) | Confortable |
|---|---|---|
| OS | Windows 10/11 (Power BI Desktop l'exige pour la restitution) | idem |
| CPU | Multi-cœur récent | idem |
| RAM | 16 Go | 32 Go |
| VRAM (GPU) | Aucune obligatoire, mais très lent sans | 8-16 Go — accélère nettement l'inférence locale |
| Stockage libre | ~30 Go (modèles Ollama + dépendances) | 50+ Go |

**Le facteur qui détermine le plus la vitesse est la VRAM disponible**, pas seulement la RAM ou le CPU : sur un GPU à 4 Go de VRAM, une question peut prendre plusieurs minutes avec un modèle de 20 milliards de paramètres. Voir [Limites connues](#limites-connues) pour les options de repli (modèle plus léger, réglage `num_gpu`).

## Installation

1. **Python 3.10+** installé et accessible en ligne de commande.
2. **[Ollama](https://ollama.com/download)** installé, puis télécharger au moins un modèle :
   ```bash
   ollama pull gpt-oss:20b
   ```
   Pour une machine avec peu de VRAM, un modèle plus léger fonctionne aussi (voir [Limites connues](#limites-connues)) :
   ```bash
   ollama pull qwen2.5:7b
   ```
3. **Dépendances Python** :
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optionnel, pour la restitution Power BI)** [Power BI Desktop](https://www.microsoft.com/fr-fr/download/details.aspx?id=58494) et [Tabular Editor 2](https://github.com/TabularEditor/TabularEditor/releases) (gratuit).

## Configuration

Toute la configuration passe par des **variables d'environnement**, centralisées dans [`config.py`](config.py) — aucune modification de code n'est nécessaire pour changer de fichier de données.

| Variable | Rôle | Défaut |
|---|---|---|
| `KPI_DATA_PATH` | Chemin vers le fichier de données (Excel) | `data/sources/donnees.xlsx` |
| `KPI_DATA_SHEET` | Nom de la feuille à lire | `Feuil1` |
| `KPI_DICT_PATH` | Chemin vers le dictionnaire de données (Excel/CSV avec colonnes nom + définition, ou `.txt` au format `NOM, -- définition`) | `data/sources/dictionnaire.xlsx` |
| `KPI_TARGET_COL` | Colonne cible que les KPI doivent expliquer | `RESULTAT` |
| `KPI_LLM_MODEL` | Modèle Ollama utilisé | `gpt-oss:20b` |
| `KPI_DOMAIN_CONTEXT` | Contexte métier libre injecté dans le raisonnement a priori (secteur d'activité, dimensions d'analyse pertinentes...) | générique si non renseigné |
| `KPI_YEAR_COL` | Colonne d'année pour le score de stabilité temporelle | dérivée automatiquement d'une colonne de date si non renseignée |
| `OLLAMA_BASE_URL` | URL du serveur Ollama | `http://127.0.0.1:11434` |

**Le dictionnaire de données** accepte deux formats :
- **Excel ou CSV** : un tableau avec au minimum deux colonnes — un nom technique (en-tête contenant "nom", "champ" ou "colonne") et une définition en langage naturel (en-tête contenant "défini"). À défaut de détection, la 1ère colonne est prise comme nom et la 2ème comme définition.
- **Texte brut (`.txt`)** : une ligne par colonne, au format `NOM_TECHNIQUE, -- définition en texte libre` (commentaire de style SQL) — pratique si l'équipe métier fournit le dictionnaire tel quel, sans mise en forme tabulaire.

Exemple de configuration (PowerShell) :
```powershell
$env:KPI_DATA_PATH = "C:\chemin\vers\mon_fichier.xlsx"
$env:KPI_DATA_SHEET = "Données"
$env:KPI_DICT_PATH = "C:\chemin\vers\dictionnaire.xlsx"
$env:KPI_TARGET_COL = "MA_COLONNE_CIBLE"
```

## Utilisation — pipeline de découverte de KPI

Deux étapes nécessitent un appel au modèle de langage et ne sont **pas rappelées automatiquement** à chaque exécution (résultat mis en cache) :

```bash
# Étape 1-3 : dictionnaire, profilage, classification sémantique
python src/etape1_dictionnaire/load_dictionnaire.py
python src/etape2_profilage/profiler_donnees.py
python src/etape3_classification/appel_ollama.py

# Étape 3b : raisonnement métier a priori (propositions de KPI)
python src/etape3b_raisonnement_metier/proposer_kpi_metier.py
```

Puis l'orchestrateur complet (génération, notation, croisement itératif — aucun appel au modèle de langage ici, tout est déjà en cache) :

```bash
python src/pipeline_kpi.py
```

→ produit `resultats/top_kpi.csv`.

Génère ensuite les explications en langage naturel de chaque KPI retenu (rappelle le modèle de langage, une fois par KPI) :

```bash
python src/etape6_explication/generer_explications.py
```

→ produit `resultats/top_kpi_explique.xlsx`.

Enfin, génère le script d'import Power BI :

```bash
python src/etape7_export_powerbi/exporter_dax.py
```

→ produit `resultats/mesures_powerbi.csx`.

### Recalibrer les poids du score composite (optionnel)

```bash
python src/etape5_notation/optimiser_poids.py
```

S'appuie sur les KPI "raisonnement métier" du jeu de données actif comme référence — à refaire pour chaque nouveau fichier, pas un réglage universel (voir [Limites connues](#limites-connues)).

## Utilisation — module text-to-SQL

Permet d'interroger directement le fichier de données en langage naturel, sans connaissance SQL — complémentaire au tableau de bord des KPI déjà calculés.

1. Construire la base locale (à relancer si le fichier de données change) :
   ```bash
   python text_to_sql/creer_base_locale.py
   ```
2. Utilisation en ligne de commande :
   ```bash
   python text_to_sql/agent_text_to_sql.py "Combien de lignes contient la base ?"
   ```
3. Ou via l'interface de chat :
   ```bash
   streamlit run text_to_sql/streamlit_app.py
   ```

**Garde-fous de sécurité** (le module est pensé pour un usage par une personne sans connaissance SQL) : connexion à la base ouverte en lecture seule au niveau du fichier lui-même, validation déterministe qui rejette tout ce qui n'est pas un `SELECT`, limite automatique du nombre de lignes renvoyées. Le modèle de langage ne calcule jamais rien lui-même — il traduit la question en requête et reformule le résultat, le calcul est toujours fait par la base de données.

## Restitution Power BI (optionnel)

1. Dans Power BI Desktop, importe ton fichier de données (`Obtenir les données` → `Excel`).
2. Installe [Tabular Editor 2](https://github.com/TabularEditor/TabularEditor/releases) et enregistre-le comme External Tool (voir sa documentation).
3. Depuis Power BI Desktop : `External Tools` → `Tabular Editor` → panneau `C# Script` → colle le contenu de `resultats/mesures_powerbi.csx` → `Run Script` → `Save`.
4. Rafraîchis le volet `Data` dans Power BI Desktop : les colonnes calculées apparaissent, prêtes à être utilisées dans des visuels.

## Structure du projet

```
config.py                      # configuration centrale (variables d'environnement)
requirements.txt
data/
  sources/                     # ton fichier de données + dictionnaire (non fournis)
  generated/                   # dictionnaire enrichi, rapport de qualité (regénérés)
resultats/                     # classement des KPI, explications, export Power BI (regénérés)
src/
  etape0_validation/           # validation + nettoyage du fichier d'entrée
  etape1_dictionnaire/         # chargement du dictionnaire de données
  etape2_profilage/            # profilage empirique des données réelles
  etape3_classification/       # classification sémantique par modèle de langage
  etape3b_raisonnement_metier/ # propositions de KPI a priori (définitions seules)
  etape4_candidats/            # génération exhaustive + guidée + croisement itératif
  etape5_notation/             # notation multi-critères, calibration des poids
  etape6_explication/          # effet XAI objectif + explication en langage naturel
  etape7_export_powerbi/       # export du script Tabular Editor
  pipeline_kpi.py              # orchestrateur (étapes 4-5, boucle itérative)
text_to_sql/
  creer_base_locale.py         # copie les données dans une base SQLite locale
  agent_text_to_sql.py         # chaîne question -> SQL -> exécution -> réponse
  streamlit_app.py             # interface de chat
```

## Limites connues

- **Vitesse liée au matériel** : sur un GPU avec peu de VRAM, chaque appel au modèle de langage peut prendre plusieurs minutes. Deux leviers, à calibrer prudemment pour la machine cible :
  - passer sur un modèle plus léger (`KPI_LLM_MODEL=qwen2.5:7b`) — plus rapide, mais moins fiable sur du vocabulaire métier ambigu formulé en langage courant plutôt qu'avec les noms exacts de colonnes ;
  - forcer un nombre de couches sur le GPU (`KPI_SQL_NUM_GPU` pour le module text-to-SQL) — accélère nettement, mais une valeur trop élevée fait échouer le chargement du modèle (`cudaMalloc` hors mémoire). À augmenter progressivement en testant à chaque palier, jamais en une seule grande valeur.
- **Calibration des poids** (`optimiser_poids.py`) : s'appuie sur les seuls KPI "raisonnement métier" du jeu de données actif comme référence — risque de surapprentissage sur un petit nombre d'exemples, à refaire pour chaque nouveau fichier plutôt que de réutiliser des poids d'un autre contexte.
- **Text-to-SQL et vocabulaire ambigu** : le modèle peut mal choisir entre deux colonnes dont les noms sont proches d'un même terme métier. Nommer explicitement la colonne dans la question (plutôt que le terme métier générique) contourne le problème de façon fiable.
- **Module text-to-SQL branché en local par défaut** : `creer_base_locale.py` copie les données dans un fichier SQLite local. Pour brancher sur une base de production, changer `CONNECTION_STRING` dans `agent_text_to_sql.py` — s'assurer que la connexion utilisée reste en lecture seule au niveau du serveur.
