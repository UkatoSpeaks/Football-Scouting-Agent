# European Football Player Scouting

> A data-driven football scouting system that identifies statistically similar players and discovers playing-style clusters across Europe's top five leagues.

Search for a player, see their statistical profile, and get the players whose statistics look most like theirs, in any of the Premier League, La Liga, Serie A, Bundesliga or Ligue 1. Explore how the whole player pool splits into playing-style clusters, and compare any two profiles side by side.

- [Overview](#1-overview)
- [Features](#2-features)
- [Tech stack](#3-tech-stack)
- [Pipeline](#4-pipeline)
- [Similarity model](#5-similarity-model)
- [Clustering](#6-clustering)
- [Example results](#7-example-results)
- [Dashboard](#8-dashboard)
- [Project structure](#9-project-structure)
- [Setup](#10-setup)
- [Data](#11-data)
- [Limitations](#12-limitations)
- [Future improvements](#13-future-improvements)
- [Running tests](#14-running-tests)
- [Screenshots](#15-screenshots)

---

## 1. Overview

Football scouting often starts with a question like *"who else can do what this player does?"*: finding a player who could fill a similar role or offer a similar statistical profile to an existing one.

This project answers that question from data. It uses season statistics for players in Europe's five major leagues to:

- **find statistically similar players**, including across leagues;
- **group players into playing-style clusters** within each position;
- **compare player profiles** side by side;
- **explore cross-league scouting candidates** with league, position and minutes filters;
- **visualise** how players and clusters relate to each other.

> **What "similar" means here.** The system measures **statistical similarity**: how alike two players' per-90 statistics are after standardising them within a position group. That is a *proxy* for playing style, not a description of it. It cannot see tactical role, off-ball movement, instructions, opposition quality or anything else that is not in the statistics, and it says nothing about how *good* a player is. See [Limitations](#12-limitations).

---

## 2. Features

Everything listed here is implemented and covered by the test suite.

**Dashboard (Streamlit)**

- **Player search** across all modelled players (searchable list, shareable `?player_id=` links).
- **Position filtering**: narrows the player list to GK, DEF, MID or FWD.
- **League filtering**: restricts the similar-player results to one league (the selected player can be from any league).
- **Minimum candidate minutes**: similar players must have played at least this many minutes (default 900).
- **Player statistics**: position-specific statistics with percentiles, grouped as Attacking, Chance creation, Passing, Carrying & take-ons, Defending and Goalkeeping (only the groups that apply to the player's position).
- **Similar-player recommendations** ranked by cosine similarity, with club, league, position, minutes and cluster.
- **Cross-league scouting**: a league-mix chart shows where the matches play.
- **Player comparison**: any similar player against the target, on the same features, with a chart and a table.
- **Playing-style cluster** for the selected player, with a description generated from the cluster's statistics.
- **PCA cluster map** with hover details, the selected player highlighted and the similar players ringed.
- **Cluster explorer**: cluster sizes, a profile heatmap, high/lower feature lists, per-cluster player lists with league and minutes filters, and a PCA map with a focus option.
- **Cached models and data**: everything is loaded once with `st.cache_resource`; nothing is retrained when the app starts.
- **Validation and error handling**: saved artifacts are cross-checked at load (a missing or out-of-sync file shows what to rebuild); unknown players, empty results, missing optional metadata (such as age) and bad links are handled without errors.

**Modelling code (`src/scouting/`)**

- **Position-specific feature sets** for GK, DEF, MID and FWD, with per-90 rates and smoothed efficiency ratios.
- **Cosine similarity** within a position group, with an optional league filter and a candidate-minutes filter.
- **Euclidean distance** as a *secondary diagnostic* (`metric="euclidean"` and `SimilarityEngine.compare_metrics`). It is available in code and in `scripts/inspect_similarity.py`; it is not shown in the dashboard.
- **K-Means clustering** per position group, with K evaluated by inertia, silhouette and seed stability.
- **Cluster profiles** (mean and median standardised value of every feature per cluster).
- **PCA** to two dimensions for visualisation.

---

## 3. Tech stack

```text
Python  ·  Pandas  ·  NumPy  ·  Scikit-learn  ·  Matplotlib  ·  Streamlit  ·  Joblib
```

Also used: **Plotly** (interactive dashboard charts), **PyYAML** (configuration), **kagglehub** and **python-dotenv** (dataset download), **pytest**.

Machine-learning and analysis techniques:

| Technique | Where it is used |
|---|---|
| `StandardScaler` | z-scores every feature, fitted separately for each position group |
| Cosine similarity | the primary player-similarity metric |
| K-Means | playing-style clusters, one model per position group |
| PCA | 2-D map of the standardised feature space |
| Silhouette analysis | how well separated the clusters are, for K = 2 to 10 |
| Cluster stability (adjusted Rand index, ARI) | whether the same clusters appear with different random seeds |
| `log1p` transform | reduces right skew in selected count features before scaling |

`requirements.txt` also lists `requests`, `lxml`, `html5lib`, `beautifulsoup4` and `pyarrow`. The first four belong to a live-scraping client that is no longer used (see [Data](#11-data)); `pyarrow` is not currently used.

---

## 4. Pipeline

```text
Raw Data
   ↓
Data Cleaning
   ↓
Minimum Minutes Filter
   ↓
Per-90 Feature Engineering
   ↓
Position-specific Feature Selection
   ↓
Log Transform
   ↓
StandardScaler
   ↓
Player Similarity
   ↓
K-Means Clustering
   ↓
PCA
   ↓
Streamlit Dashboard
```

| Step | What happens | Code |
|---|---|---|
| Raw data | The 2024-25 player table (267 columns, 2,854 rows) is downloaded from Kaggle. | `src/scouting/clean/download.py` |
| Data cleaning | Repeated columns are verified identical and dropped, the ambiguous ones (for example `Blocks`, `Off`, `Fld`) are renamed explicitly, league and nation prefixes are split off, and only additive season totals are kept. Players who appear for two teams are merged on name plus birth year (147 rows merged, giving 2,707 players). | `src/scouting/clean/data_preprocessing.py`, `src/scouting/clean/columns.py` |
| Minimum minutes | Players need **450 minutes** to be modelled, leaving 1,959 players. | `src/scouting/features/engineering.py` |
| Per-90 features | Totals become rates per 90 minutes; efficiency ratios are added. | `src/scouting/features/engineering.py` |
| Feature selection | Each position group uses its own feature list. | `config.yaml` |
| Log transform | `log1p` on features that are non-negative and strongly right-skewed within their group. | `src/scouting/features/scaling.py` |
| StandardScaler | z-scores per position group; the fitted scalers are saved. | `src/scouting/features/scaling.py` |
| Similarity | Cosine similarity inside each position group. | `src/scouting/similarity.py` |
| K-Means, PCA | One K-Means and one PCA per position group; assignments, profiles, coordinates and models are saved. | `src/scouting/clustering.py` |
| Dashboard | Reads the saved outputs only. | `app.py`, `src/scouting/dashboard/` |

**Why per-90 statistics.** Season totals mostly measure playing time: a player with 3,000 minutes will out-accumulate one with 1,000 even if the second is more productive. Dividing by minutes played (in units of 90) puts players on a comparable footing. Ratios such as pass completion or take-on success are computed from the totals and are *smoothed* toward the position group's average (a prior worth 10 attempts), so a player who went one-for-one on take-ons does not read as 100% and a player with no attempts gets the group average instead of a division by zero.

**Why position-specific feature sets.** A goalkeeper's useful statistics (save percentage, sweeping, distribution) have nothing in common with a striker's (shot quality, penalty-area touches), and even outfield roles are described by different things. Forcing one feature set on everyone would fill the comparison with irrelevant, mostly-zero columns. So each group has its own list:

| Group | Players (≥ 450 min) | Features | `log1p` applied to |
|---|---|---|---|
| GK | 154 | 6 | 1 |
| DEF | 748 | 19 | 4 |
| MID | 604 | 16 | 5 |
| FWD | 453 | 20 | 10 |

The full lists are in [`config.yaml`](config.yaml) (`modeling.position_feature_lists` and `modeling.log1p_features`). No manual feature weights are applied.

---

## 5. Similarity model

Each player is a vector of standardised statistics, one z-score per feature in their position group's list (0 means the position-group average). **Cosine similarity** compares the *direction* of two such vectors: do the two players lean the same way relative to the average, on the same features? A score of 1 means the vectors point the same way, 0 means unrelated, and −1 means opposite.

> **Similarity scores are not probabilities or percentages.** A cosine similarity of 0.87 does *not* mean "87% similar", and it says nothing about how good either player is. The dashboard shows scores as plain numbers (for example `0.924`).

Rules the engine follows:

- **Players are compared only within their own position group.** A goalkeeper is never compared with an outfielder, a defender with a midfielder, and so on. Asking for a different group raises an error.
- The **query player** only needs to be in the modelled pool (450+ minutes). **Candidates** must have at least **900 minutes** by default; this is adjustable (`min_candidate_minutes`) and guards against noisy small samples.
- An optional **league filter** restricts the candidates to one league; the query player may come from any league, which is what makes cross-league scouting possible.
- Results exclude the query player, are sorted by score, and break ties deterministically.

```python
from scouting.similarity import SimilarityEngine

engine = SimilarityEngine.from_config()
pid = engine.find_id("Bukayo Saka")
engine.find_similar_players(pid, n=10, min_candidate_minutes=900, leagues="La Liga")
```

Cosine similarity ignores how *extreme* a player is: a modest player with a striker's shape scores well against an elite striker. As a diagnostic, `metric="euclidean"` ranks by overall distance instead, and `SimilarityEngine.compare_metrics` shows how the two rankings differ. Neither is presented as "better"; the dashboard uses cosine similarity.

---

## 6. Clustering

Players are clustered with **K-Means, separately for each position group**, on the same standardised matrices used for similarity. A goalkeeper never shares a feature space with a forward.

```text
GK  → K=2
DEF → K=5
MID → K=4
FWD → K=5
```

**How K was chosen.** K was evaluated from 2 to 10 using inertia, silhouette score, cluster stability across random seeds (ARI), cluster sizes and whether the clusters have distinct, interpretable profiles. Silhouette alone was not used to decide, because it favours K=2 in every group (the single biggest split). Instead, K is the largest value whose clusters are reproducible across seeds (ARI ≥ 0.90, and ≥ 0.95 for goalkeepers, which have only six features), with no cluster under 5% of its group and clearly different feature profiles. The choice was made without reference to any individual player's assignment. The rationale is recorded in [`config.yaml`](config.yaml).

| Group | K | Silhouette | Stability (ARI) | Cluster sizes |
|---|---|---|---|---|
| GK | 2 | 0.18 | 0.97 | 86, 68 |
| DEF | 5 | 0.12 | 0.96 | 239, 151, 150, 105, 103 |
| MID | 4 | 0.15 | 0.97 | 224, 142, 137, 101 |
| FWD | 5 | 0.13 | 0.95 | 132, 108, 107, 76, 30 |

**Silhouette scores are relatively low** (0.12 to 0.18). That is expected: player profiles form a continuum rather than perfectly separated groups, so clusters overlap and a player near a boundary could sit in a neighbouring cluster. The clusters are best read as *tendencies* (which statistics dominate), not hard categories. Stability is high, meaning the same clusters come back with different random seeds up to the chosen K.

Clusters are described from data, never named by hand: a feature counts as **high** when the cluster's mean z-score is at least +0.5 and **lower** when it is at most −0.5. The dashboard builds its descriptions from [`cluster_profiles.csv`](data/processed/2024-2025/cluster_profiles.csv). Cluster ids are ordered by size (0 is the largest) and only mean something within their own position group.

---

## 7. Example results

These come from the saved 2024-25 outputs with the default settings (candidates need 900+ minutes). They are illustrations, not validation: there is no ground truth for "similar player", so judge them with football knowledge.

### Bukayo Saka: FWD · cluster FWD-4

Arsenal, Premier League, 1,729 minutes. FWD-4 is the smallest forward cluster (30 players, 7%): **high** xAG (+2.1), progressive passes received (+2.0), goal-creating actions (+1.9), penalty-area touches (+1.8) and key passes (+1.6); **lower** aerial duels won (−1.0). In short, high chance creation and attacking involvement. Closest matches: Michael Olise (0.924), Désiré Doué (0.916), Sávio (0.909), Lamine Yamal (0.895), Vinícius Júnior (0.869). Nine of his top ten play outside the Premier League.

### Erling Haaland: FWD · cluster FWD-3

Manchester City, Premier League, 2,736 minutes. FWD-3 (76 players) is a finishing-oriented profile: **high** npxG (+1.5), shot quality (+1.2), goals (+1.2), shooting volume (+1.0); **lower** crossing, ball recoveries, passing volume and tackles. Closest matches: Cristhian Stuani (0.890), Patrik Schick (0.868), Ermedin Demirović (0.859), Alexander Sørloth (0.842), Myron Boadu (0.839). All five share the finishing profile: their npxG per 90 ranges from 0.48 to 0.92, against 0.62 for Haaland and 0.29 for the average forward.

### Declan Rice: MID · cluster MID-2

Arsenal, Premier League, 2,825 minutes. MID-2 (137 players) shows **high** progressive passing (+1.1), passing volume (+1.0), ball recoveries (+0.8) and pass completion (+0.6), with nothing markedly lower. Closest matches: Thiago Almada (0.843), Hakan Çalhanoğlu (0.832), Julian Brandt (0.819), Fabián Ruiz (0.804), Lucas Da Cunha (0.789). Worth knowing: these are mostly advanced, progression-heavy midfielders. Rice's own tackle and interception rates are *below* the midfielder average in this data (which may partly reflect Arsenal's high possession; see [Limitations](#12-limitations)), and the classic ball-winners in the dataset (for example Casemiro, Zubimendi, Caicedo) fall in MID-0.

### Virgil van Dijk: DEF · cluster DEF-2

Liverpool, Premier League, 3,330 minutes. DEF-2 (150 players) is a ball-playing defender profile: **high** passing volume (+1.2), pass completion (+0.9), progressive passing (+0.9), aerial win rate (+0.5); **lower** crossing and attacking-third touches. Closest matches: Karol Mets (0.867), Stefan de Vrij (0.804), Jonathan Tah (0.775), Philipp Lienhart (0.768), Eric Dier (0.762).

### Alisson: GK · cluster GK-0

Liverpool, Premier League, 2,508 minutes. GK-0 (86 goalkeepers) is defined by *lower* passing volume and progressive pass distance; it has no strongly high traits, and the two goalkeeper clusters differ mainly in distribution style. Closest matches: Rui Silva (0.889), Péter Gulácsi (0.818), Alex Meret (0.807), Kepa Arrizabalaga (0.787), Wojciech Szczęsny (0.753). Goalkeeper results rest on only six statistics and 154 players, so treat them as the weakest in the project.

---

## 8. Dashboard

Start it with `streamlit run app.py` (see [Setup](#10-setup)). Sidebar filters (player, league, position, minimum candidate minutes, number of similar players) drive the **Player scouting** page.

### Player scouting

Sections appear in this order once a player is selected:

1. **Player overview**: name, club, league, position and group, minutes, cluster, plus age, matches and nation when available.
2. **Player statistics**: the position-specific statistics in tabs, each with its percentile among that position group.
3. **Find similar players**: the ranked table (rank, player, club, league, position, minutes, cosine similarity, cluster), a league-mix chart and a note explaining how to read the score.
4. **Player comparison**: choose any of the similar players; a chart of standard deviations from the position average and a side-by-side table of real values, with the largest differences called out.
5. **Playing style**: the player's cluster, its size and a description generated from its high and lower features.
6. **Cluster map**: the saved PCA projection of the player's position group, coloured by cluster, with the selected player and (optionally) the similar players highlighted.

### Cluster explorer

Pick a position group (GK, DEF, MID or FWD) to see:

- **Cluster sizes** and the number of players and clusters;
- **Cluster profiles** as a heatmap of average standardised statistics (blue above the position average, red below);
- **High and lower feature analysis** for each cluster;
- **Player lists** for each cluster, filterable by league and minimum minutes;
- **PCA visualisation** of the whole group, with an option to focus on one cluster.

A **Limitations** section, including the team-style caveat, is shown on every page.

---

## 9. Project structure

```text
.
├── app.py                        Streamlit entry point (two pages)
├── config.yaml                   season, data source, features, K values, log1p lists
├── requirements.txt
├── pyproject.toml                makes the `scouting` package installable (pip install -e .)
├── .streamlit/config.toml        pins the app to the light theme
├── .env                          local Kaggle token (git-ignored; create it yourself)
│
├── src/scouting/
│   ├── config.py                 typed loader for config.yaml, plus data/model/report paths
│   ├── clean/                    download.py, columns.py, data_preprocessing.py
│   ├── features/                 engineering.py (per-90, ratios, filter), scaling.py (log1p, scaler)
│   ├── similarity.py             SimilarityEngine and find_similar_players()
│   ├── clustering.py             K selection, K-Means, profiles, PCA, saved outputs
│   ├── visualization.py          Matplotlib elbow and PCA figures for reports/
│   ├── palette.py, labels.py     shared chart colours; readable feature names and categories
│   ├── dashboard/                data.py (load + validate), logic.py, charts.py, views.py
│   ├── scrape/, utils/           legacy FBref client (no longer used); logging helper
│   └── models/                   empty package placeholder
│
├── data/
│   ├── raw/2024-2025/            downloaded Kaggle CSV
│   ├── interim/2024-2025/        players_clean.csv
│   └── processed/2024-2025/      features, model matrices, K-Means evaluation,
│                                 cluster assignments and profiles, PCA coordinates
├── models/2024-2025/             scalers.joblib, kmeans.joblib, pca.joblib
├── reports/2024-2025/            elbow_<GROUP>.png, pca_<GROUP>.png
├── scripts/                      inspect_features.py, inspect_similarity.py, inspect_clusters.py
│                                 (review reports); inspect_raw.py (legacy, needs the old scraper)
└── tests/                        pytest suite (see Running tests)
```

`data/raw/` also contains two 2025-26 CSVs that the pipeline does not use (see [Data](#11-data)).

---

## 10. Setup

Requires Python 3.10 or newer (`pyproject.toml`); developed and tested on Python 3.13.

```bash
# 1. get the project
git clone <repository-url>
cd <project-folder>

# 2. create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\Activate.ps1         # Windows PowerShell

# 3. install dependencies and the `scouting` package
pip install -r requirements.txt
pip install -e .

# 4. start the dashboard
streamlit run app.py
```

The app opens at <http://localhost:8501>. On Windows without activating the environment: `.venv\Scripts\python.exe -m streamlit run app.py`.

If `data/processed/` and `models/` are already present, step 4 is all you need. If the app says the data is not ready, rebuild the outputs:

```bash
python -m scouting.clean.data_preprocessing   # downloads the raw data if missing, then cleans it
python -m scouting.features.engineering       # per-90 features, 450-minute filter
python -m scouting.features.scaling           # log1p + StandardScaler, saves scalers
python -m scouting.clustering evaluate        # optional: K = 2..10 evaluation and elbow plots
python -m scouting.clustering fit             # K-Means, PCA, profiles, assignments, figures
```

The download step needs a Kaggle API token. Put it in a `.env` file in the project root (this file is git-ignored):

```text
KAGGLE_API_TOKEN=<your-token>
```

If `data/raw/2024-2025/players_data-2024_2025.csv` already exists, nothing is downloaded and no token is needed. All steps use fixed random seeds (`modeling.random_state: 42`). Rebuilding reproduces the features, model matrices, cluster assignments, cluster profiles and PCA coordinates exactly; the K-Means evaluation table (`kmeans_evaluation.csv`) can differ in the last floating-point digits between runs.

---

## 11. Data

- **Source.** Player statistics come from **FBref**, obtained through the Kaggle dataset [`hubertsidorowicz/football-players-stats-2024-2025`](https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025) (file `players_data-2024_2025.csv`), as configured in `config.yaml`. Check the dataset page for its licence and terms before reusing or redistributing the data.
- **Season.** **2024-2025**, a complete season.
- **Leagues.** Premier League, La Liga, Serie A, Bundesliga and Ligue 1.
- **Coverage.** 2,854 raw rows become 2,707 players after merging players who appeared for two teams; **1,959** have the **450 minutes** needed for model eligibility (748 DEF, 604 MID, 453 FWD, 154 GK).
- **Thresholds.** 450 minutes to be modelled; **900 minutes** by default for similar-player *candidates* (adjustable in the dashboard).
- **Why 2024-25 and not the latest season.** FBref removed its advanced statistics (passing, possession, defending, shot creation) in January 2026, and the site is behind a Cloudflare challenge. The 2025-26 dataset on Kaggle therefore has basic statistics only, which is not enough for these features. The two 2025-26 CSVs in `data/raw/` are left over from that exploration and are not used, and the original scraping client in `src/scouting/scrape/` is dormant.

---

## 12. Limitations

**Team style.** Per-90 statistics still reflect team context and possession style. A midfielder at a high-possession club records fewer tackles and interceptions per 90 than a similar player at a team that defends more, and passing and touch counts inflate the other way. The project does not correct for this, so some similarities and clusters partly reflect the team, not just the player.

**Statistical similarity is not tactical similarity.** Two players can have similar statistics while playing different roles, and two players in the same role can have different statistics. The model sees only the numbers in the dataset: no video, off-ball movement, instructions or opposition strength.

**K-Means.** Player profiles form a continuum, so clusters overlap (silhouette 0.12 to 0.18) and boundaries are fuzzy. K-Means also assumes roughly round clusters, and some separations follow intensity as much as style (for example the small FWD-4 cluster).

**Goalkeepers.** The goalkeeper feature space is small (six features, 154 players) and mostly reflects distribution style; clustering and similarity are weaker there than for outfield players.

**Sample size.** Low-minute players produce noisy statistics. That is why players need 450 minutes to be modelled and why similar-player candidates default to 900+ minutes; even so, a single season is a limited sample.

**Cross-league differences.** Statistics are standardised across all five leagues together, so differences in league style and intensity are not normalised away.

Also: results cover one season (2024-25); position groups are broad (GK, DEF, MID, FWD from the first listed position, with no separate winger or full-back label); and the low silhouette and small goalkeeper model mean cluster labels should be read as tendencies.

---

## 13. Future improvements

None of these is implemented yet.

- Adjust statistics for **team possession** and style.
- **Normalise across leagues.**
- **Role classification** within position groups (for example winger versus striker, full-back versus centre-back).
- **More seasons**, including trends over time.
- **More advanced similarity methods** beyond cosine and Euclidean distance.
- **Scouting shortlists** that can be saved and compared.
- **Age and transfer-value filters.**
- **Tactical or event-level data.**
- **Deployment** improvements (hosting, packaged data and models, CI).

---

## 14. Running tests

```bash
python -m pytest tests -q
```

The suite has 125 tests and takes about 20 seconds. It covers data cleaning, feature engineering, scaling, the similarity engine, clustering and the dashboard (the dashboard is exercised headlessly with Streamlit's `AppTest`). Tests that need the built pipeline outputs are skipped automatically if `data/processed/` and `models/` are missing; run the [rebuild steps](#10-setup) first to include them.

The `scripts/inspect_*.py` files print review reports (feature distributions, similarity diagnostics, cluster profiles) and are not part of the test suite:

```bash
python scripts/inspect_features.py
python scripts/inspect_similarity.py
python scripts/inspect_clusters.py
```

---

## 15. Screenshots

The pipeline's static figures are in the repository:

| K selection (MID) | PCA map (FWD) |
|---|---|
| ![Elbow and silhouette for MID](reports/2024-2025/elbow_MID.png) | ![PCA map for FWD](reports/2024-2025/pca_FWD.png) |

Other figures: [`elbow_GK.png`](reports/2024-2025/elbow_GK.png), [`elbow_DEF.png`](reports/2024-2025/elbow_DEF.png), [`elbow_FWD.png`](reports/2024-2025/elbow_FWD.png), [`pca_GK.png`](reports/2024-2025/pca_GK.png), [`pca_DEF.png`](reports/2024-2025/pca_DEF.png), [`pca_MID.png`](reports/2024-2025/pca_MID.png).

<!-- TODO: add dashboard screenshots. No dashboard screenshots are stored in the repository yet. Suggested: capture the Player scouting page (overview, similar players, comparison, cluster map) and the Cluster explorer page, save them in a new folder such as docs/screenshots/, and reference them here. -->

> **TODO:** dashboard screenshots (Player scouting and Cluster explorer) have not been added to the repository yet.
