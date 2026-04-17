# Bias-Aware Video Recommender

A Responsible AI demo that **injects**, **detects**, and **corrects** algorithmic bias in a YouTube-style video recommender. Built on real YouTube trending data, a two-tower neural retrieval model, and an agentic LLM supervisor with multi-round feedback refinement.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [System Architecture](#system-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Two-Tower Recommender](#two-tower-recommender)
6. [Agentic LLM Supervisor](#agentic-llm-supervisor)
7. [Feedback Refinement Loop](#feedback-refinement-loop)
8. [Fairness Scores](#fairness-scores)
9. [Web UI](#web-ui)
10. [User Personas](#user-personas)
11. [Testing](#testing)
12. [Project Structure](#project-structure)
13. [Installation & Usage](#installation--usage)
14. [References](#references)

---

## Overview

| Component | Description |
|---|---|
| **Dataset** | Real YouTube trending videos from 10 countries (~13K unique videos, 8 genres, 8 languages) |
| **Biased Recommender** | Two-tower model with engagement-weighted scoring that systematically demotes suppressed genres |
| **LLM Supervisor** | LangChain agentic loop (Groq `llama-3.3-70b-versatile`) that detects bias and rebuilds a corrected 30-video feed |
| **Feedback Refinement** | Multi-round session loop — users describe what they want changed; the LLM re-corrects in each round |
| **Fairness Scores** | 5 quantitative metrics (0–100) grading every feed on diversity, representation, suppressed coverage, and language balance |
| **Web UI** | Flask single-page app — select a persona or create a new user, run analysis, compare three feeds side-by-side |
| **Test Suite** | Two standalone test scripts covering data integrity, bias injection, fairness maths, and LLM correction quality |

---

## Motivation

> *"If a platform recommends what users click on, and users click on what they've been shown, who decides what gets shown first?"*

Real platforms like YouTube optimise for engagement signals (clicks, watch time, likes). A documented side effect is that high-click content (Entertainment, Gaming, Music) gets amplified while slower-consumption content (Educational, Documentary, Regional-language) gets progressively de-ranked — not through explicit suppression, but through the compounding weight of engagement metrics.

This project makes that mechanism **explicit, measurable, and correctable**.

**Suppressed categories** (defined throughout the system):
`Educational`, `Documentary`, `DIY`, `News/Analysis`, `Regional`

---

## System Architecture

```
                      ┌───────────────────────────────────┐
                      │        Real YouTube Data           │
                      │  10 countries · 8 genres           │
                      │  8 languages · ~13K unique videos  │
                      └─────────────┬─────────────────────┘
                                    │ dataset_builder.py
                                    ▼
                      ┌───────────────────────────────────┐
                      │  master_dataset.csv               │
                      │  + embeddings.npy                 │
                      │  (SentenceTransformer all-MiniLM) │
                      └──────────┬────────────────────────┘
                                 │
           ┌─────────────────────┴──────────────────────┐
           │                                            │
           ▼                                            ▼
┌─────────────────────────┐          ┌──────────────────────────────┐
│  20 User Personas        │          │  LLMSupervisor (app startup)  │
│  (users.json)            │          │  Deduplicates master_df       │
│  + Guest user builder    │          │  Aligns embeddings 1-to-1    │
└──────────┬──────────────┘          └──────────────┬───────────────┘
           │                                         │
           ▼                                         │
┌────────────────────────────────────────────┐       │
│         get_biased_recs()                  │       │
│                                            │       │
│  USER TOWER                                │       │
│  Rating-weighted avg of watched-video      │       │
│  embeddings → user_vec (384-dim)           │       │
│                                            │       │
│  ITEM TOWER                                │       │
│  Pre-computed embeddings.npy               │       │
│                                            │       │
│  SCORE (per candidate)                     │       │
│  soft_mult = 0.5 + 0.5 × eng_mult × lm    │       │
│  penalized_sim = cosine_sim × soft_mult    │       │
│  bias_component = virality × eng × lm     │       │
│  final = 0.5 × penalized_sim              │       │
│        + 0.5 × bias_component             │       │
│                                            │       │
│  → Biased top-30                           │       │
└──────────────────┬─────────────────────────┘       │
                   │                                  │
                   └──────────────┬───────────────────┘
                                  │
                                  ▼
                ┌─────────────────────────────────────┐
                │      LLMSupervisor.fix()             │
                │                                     │
                │  Step 1 — Judge                     │
                │  Structured LLM call → BiasAssessment│
                │  (is_biased, genres_missing,         │
                │   tier slot allocation)              │
                │                 │ if biased          │
                │  Step 2 — Pre-fetch (~50 candidates) │
                │  Pure Python, no LLM, ~0.05s        │
                │                 │                   │
                │  Step 3 — Agent selects 30          │
                │  Single LLM turn over candidate     │
                │  table → submit_selection()         │
                │                 │                   │
                │  Fallback 1: direct structured LLM  │
                │  Fallback 2: rule-based tier fill   │
                └─────────────────┬───────────────────┘
                                  │
                                  ▼
                    Corrected top-30 (always exactly 30)
                    Tier constraints server-side enforced
                                  │
                    ┌─────────────▼─────────────┐
                    │  Feedback refinement loop  │
                    │  User describes changes    │
                    │  Supervisor re-corrects    │
                    │  (multi-round, per session)│
                    └───────────────────────────┘
```

---

## Data Pipeline

### Source Data

Real YouTube trending video CSVs from **10 countries** via the `datasnaek/youtube-new` Kaggle dataset.

| Country | Language |
|---|---|
| US, CA, GB | English |
| IN | Hindi |
| MX | Spanish |
| FR | French |
| DE | German |
| JP | Japanese |
| KR | Korean |
| RU | Russian |

### Processing (`dataset_builder.py`)

1. Load all 10 country CSVs; keep peak-views row per video per country
2. Deduplicate English countries globally (same video_id = same video)
3. **Global dedup** across all countries — same video_id keeps highest-views row
4. Map YouTube `category_id` → 8 project genres
5. Non-English videos in lifestyle/vlog categories → `Regional`
6. **Stratified cap**: top-300 videos per `(genre, language)` bucket → ~13K unique videos
7. Compute virality score: `0.5 × views_n + 0.3 × likes_n + 0.2 × engagement_rate_n`
8. Compute SentenceTransformer embeddings (`all-MiniLM-L6-v2`, 384-dim) on `title [genre] [language]`

**Output:** `data/master_dataset.csv` + `data/embeddings.npy`

### Genre Mapping

| YouTube Category IDs | Project Genre | Suppressed? |
|---|---|---|
| 27, 28 | Educational | Yes |
| 25 | News/Analysis | Yes |
| 29, 35 | Documentary | Yes |
| 26 | DIY | Yes |
| 10 | Music | No |
| 20 | Gaming | No |
| 1, 2, 15, 17, 19, 21, 22, 23, 24 | Entertainment | No |
| Non-English lifestyle categories | Regional | Yes |

### Embeddings

Each video's embedding is computed from `"{title} [{genre}] [{language}]"` using `all-MiniLM-L6-v2`. Embeddings are aligned row-by-row with `master_dataset.csv` and stored as a `(N, 384)` float32 numpy array.

---

## Two-Tower Recommender

### Why Two-Tower?

A two-tower architecture separates user and item representations into independent embedding spaces, then retrieves items by similarity. This produces **meaningful content-based matches** — the bias that emerges is structural (engagement weighting) rather than fabricated.

### User Tower

Built at query time from the user's watch history:

```
user_vec = Σᵢ (rating_i / Σ ratings) × embedding(watched_video_i)
user_vec = user_vec / ‖user_vec‖₂
```

- Weights each watched video's embedding by the user's rating (1–5)
- L2-normalised to unit sphere — compatible with cosine similarity
- Zero-shot from interaction history; no separate model to train
- Cold-start fallback: if no history, ranks by virality within preferred genres

### Item Tower

Pre-computed `embeddings.npy` — `SentenceTransformer` vectors computed once at build time.

### Biased Scoring

The bias is applied as a soft penalty so preferred suppressed content can still surface, but engagement-optimised content systematically edges ahead:

```
soft_mult        = 0.5 + 0.5 × genre_engagement_mult × language_mult
penalized_sim    = cosine_sim(user_vec, item_vec) × soft_mult
bias_component   = virality_score × genre_engagement_mult × language_mult

final_score      = 0.50 × penalized_sim_normalised
                 + 0.50 × bias_component_normalised
```

**Genre engagement multipliers** (proxies for real platform engagement signals):

| Genre | Multiplier | Soft penalty floor |
|---|---|---|
| Entertainment | 1.00 | 1.000 |
| Music | 0.95 | 0.975 |
| Gaming | 0.92 | 0.960 |
| DIY | 0.58 | 0.790 |
| News/Analysis | 0.52 | 0.760 |
| Educational | 0.45 | 0.725 |
| Documentary | 0.40 | 0.700 |
| Regional | 0.33 | 0.665 |

**Language multiplier:** non-English = `0.55` (−45%)

The soft penalty floor (`0.5 + 0.5 × mult`) means suppressed content is never fully zeroed out — a user who loves documentaries will still see some, but viral entertainment consistently outscores it.

---

## Agentic LLM Supervisor

### Step 1 — Judge

A structured LLM call using `with_structured_output(BiasAssessment)`:

```python
class BiasAssessment(BaseModel):
    is_biased: bool
    reasoning: str
    genres_over_represented: List[str]
    genres_missing: List[str]
    tier1_slots: int   # preferred genres via similarity + trending
    tier2_slots: int   # suppressed genre retrieval
    tier3_slots: int   # diversity outside user's bubble
    tier4_slots: int   # globally viral overlay
```

Bias is flagged only when the user's own history supports it:
- A genre the user watches frequently is absent or heavily under-represented
- The feed is flooded with genres the user does not prefer
- The user prefers non-English content but the feed is mostly English

Not flagged as bias: if a user genuinely prefers Entertainment and gets Entertainment.

### Step 2 — Pre-fetch Candidates (Python, no LLM)

Before any LLM call, the supervisor runs all 4 retrieval tiers in pure Python to build a pool of ~50 candidate videos in ~0.05s:

| Tier | Method | Description |
|---|---|---|
| T1 | Similarity + Trending | `0.65 × cosine_sim + 0.35 × virality` in preferred genres |
| T2 | Genre Retrieval | Top-viral within suppressed genres, language-filtered |
| T3 | Diversity | One video per genre outside the user's preference bubble |
| T4 | Global Viral | Top globally trending regardless of genre or language |

### Step 3 — Agent Selects 30

The candidate table (~50 rows) is handed to a LangChain tool-calling agent. The agent selects exactly 30 rows and calls `submit_selection`, which enforces hard constraints server-side:

```
REJECTED if:
  - fewer than 30 unique valid row numbers
  - T1 count outside [tier1_slots ± 5]
  - T2 count < tier2_slots
  - T3 count < t3_min
  - fewer than t4_min rows marked globally viral
```

The agent cannot finish without a valid submission. Search tools remain available as a fallback if the pre-fetched pool is insufficient.

### Fallback Chain

If the agent fails to submit (tool error, rate limit, recursion limit):

1. **Direct structured LLM call** — single `with_structured_output` call over the same candidate table, no tool loop
2. **Rule-based tier fill** — pure Python greedy algorithm respecting T1/T2/T3 quotas

All three paths always produce exactly 30 items.

### Models

| Model | Role |
|---|---|
| `llama-3.3-70b-versatile` | Primary (best tool-use reliability on Groq) |
| `llama3-groq-70b-8192-tool-use-preview` | Rate-limit fallback |
| `llama-3.1-70b-versatile` | Second fallback |

Auto-switches on 429 with a 2-second backoff.

---

## Feedback Refinement Loop

After the initial correction, users can describe what they want changed in plain text. Each round of feedback is appended to the session history and passed back to both the Judge and the Agent as highest-priority instructions.

```
Round 1: "I want more documentary content, less gaming."
Round 2: "Show me Korean videos too."
```

The supervisor treats feedback violations as bias regardless of normal criteria — if the user asked for Korean content and the corrected feed has none, it is re-flagged as biased and corrected again. Sessions are stored in-memory per process; each new `/api/run` call creates a fresh session.

---

## Fairness Scores

After every run the system grades each recommendation list on 5 scores. All scores go from **0 to 100 — higher always means fairer**. Scores appear in the web UI as colour-coded progress bars and in the run log as ASCII bar charts.

### 1. Genre Diversity (30% of Overall)

**Plain English:** How many different genres appear, and are they spread out evenly?

Shannon entropy of the genre distribution, normalised to the number of genres in the full library. A feed with all 8 genres in equal amounts scores 100. A feed of all Entertainment scores 0.

### 2. Suppressed Coverage (25% of Overall)

**Plain English:** Are the genres the platform normally buries actually showing up?

`(% suppressed in recs) ÷ (% suppressed in library) × 100`, capped at 100. The library is ~63% suppressed genres. If the feed matches that rate, score is 100. If it contains zero suppressed content, score is 0.

### 3. Representation (30% of Overall)

**Plain English:** Is one genre dominating, or is exposure shared fairly?

Gini coefficient computed over **all 8 library genres** (absent genres count as 0), converted to `(1 − Gini) × 100`. Computing over all genres is important — a single-genre feed has Gini ≈ 0.875 → score ≈ 12, not 100.

### 4. Language Diversity (15% of Overall)

**Plain English:** Does the feed include content in different languages?

Shannon entropy of language distribution, normalised to the 8 languages in the library.

### 5. Overall Fairness Score

```
Overall = 0.30 × Genre Diversity
        + 0.30 × Representation
        + 0.25 × Suppressed Coverage
        + 0.15 × Language Diversity
```

### Reading the scores

| Range | Meaning |
|---|---|
| 0 – 39 | Severely biased — one or two groups dominate |
| 40 – 69 | Moderate — some diversity but clear gaps |
| 70 – 100 | Fair — content is well distributed |

Typical values: **biased feed ≈ 10–30**, **LLM corrected ≈ 70–85**.

---

## Web UI

The Flask app (`flask_app.py`) serves a single-page interface at `http://localhost:5000`.

### Features

**User selection**
- Pick from 20 pre-built personas (each with avatar, description, genre preferences, and watch history)
- Or click **New User** to enter custom genre weights and language preferences (cold-start path)

**Three-feed comparison**
- **Biased feed** — engagement-weighted two-tower output
- **LLM Corrected feed** — supervisor-rebuilt feed after bias detection
- **Ideal feed** — pure cosine similarity, no engagement weighting (shown on the Ideal tab)

**Metrics panels**
- *Metrics Comparison* card: fairness score progress bars for biased vs corrected
- *Three-Way Comparison* card: biased vs corrected vs ideal, all 5 scores
- Genre mix tags on both cards

**Verdict card**
- LLM bias verdict (BIASED / FAIR badge)
- Over-represented and missing genres
- Correction reasoning

**Feedback refinement**
- Text area below results
- Submit plain-English instructions ("more Korean content, less Gaming")
- Rounds accumulate in session; each submission re-runs the supervisor with full history

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serves the SPA |
| `/api/user/<user_id>` | GET | Returns user profile + interaction history |
| `/api/run` | POST | Runs full pipeline; returns biased, corrected, ideal feeds + metrics |
| `/api/feedback` | POST | Applies feedback within session; returns updated feeds |

`/api/run` accepts either `{ "user_id": "U02" }` for a known persona or `{ "new_user": { "name": "...", "genres": {...}, "languages": [...] } }` for a guest user.

---

## User Personas

20 fixed personas covering diverse demographics, languages, and genre preferences. Each has ~35 pre-sampled interaction histories from real YouTube videos matching their profile.

| ID | Name | Primary Genres | Languages |
|---|---|---|---|
| U01 | Priya | DIY, Documentary | English, Hindi |
| U02 | Raj | Educational, News/Analysis | English |
| U03 | Maria | Regional, Music | Spanish |
| U04 | Ahmed | News/Analysis, Documentary | English |
| U05 | Yuki | Gaming, Music, Regional | Japanese, Korean |
| U06 | Emma | DIY, Educational | English |
| U07 | Carlos | Music, Documentary | Spanish, English |
| U08 | Fatima | Documentary, Educational | French, English |
| U09 | James | Entertainment, Gaming | English |
| U10 | Ananya | Regional, Music | Hindi |
| U11 | Lukas | Regional, Educational | German |
| U12 | Sophie | Educational, Documentary | French, English |
| U13 | Marcus | Gaming | English |
| U14 | Kenji | Music, Regional | Korean |
| U15 | Amara | Documentary, Educational | English |
| U16 | Ivan | News/Analysis, Regional | Russian |
| U17 | David | News/Analysis, Documentary | English, French, Spanish |
| U18 | Lin | Educational, Documentary | English, French |
| U19 | Sara | Music, Documentary | English, Spanish, French, Korean, Japanese |
| U20 | Tom | Entertainment, Gaming | English |

---

## Testing

Two standalone test scripts — no Flask server required.

### `test_architecture.py` — fast, no LLM

Tests every component except the LLM supervisor. Runs in ~15 seconds.

```bash
python test_architecture.py           # 36 tests, no API calls
python test_architecture.py --llm     # also run LLM supervisor tests
python test_architecture.py --llm --user U02   # LLM test for one user
```

| Section | What it checks |
|---|---|
| Data Integrity | File existence, shape alignment, NaN, unique IDs, suppression flags |
| User Embeddings | Non-zero vectors, unit norm, closer to preferred genre centroid than others |
| Bias Injection | Suppressed-genre users get <63% suppressed in biased feed; entertainment users get 97% of their preferred genres |
| Fairness Score Maths | Formula correctness: all-Entertainment→3.8, uniform→98.4, monotonicity, Gini edge cases |
| Biased vs Unbiased | Per-user table; suppressed coverage +77.5 avg; preferred-genre hit-rate +71.2pp avg |
| E2E Pipeline | All 20 users; 30-item count guarantee; avg improvement +9.1 |

### `test_llm_correction.py` — requires `GROQ_API_KEY`

Runs the biased recommender then the LLM supervisor for a set of users and asserts the correction is meaningful.

```bash
python test_llm_correction.py                  # 5 representative users
python test_llm_correction.py --all            # all 20 users
python test_llm_correction.py --users U02 U04  # specific users
```

For each user it prints a side-by-side `before → after ▲delta` table for all 5 fairness scores, then checks:

| Assertion | |
|---|---|
| Corrected list = exactly 30 items | Supervisor count guarantee |
| Bias correctly detected for suppressed-pref users | Judge accuracy |
| Overall fairness improves for every tested user | Core correctness |
| Suppressed coverage improves for every tested user | Specific bias fixed |
| Average overall improvement ≥ 20 points | Meaningful correction |
| No single genre > 50% of corrected slots | No monoculture over-correction |

---

## Project Structure

```
bias_recommender/
│
├── flask_app.py            # Flask server — biased/unbiased recs, LLM pipeline, API
├── llm_supervisor.py       # LangChain agentic supervisor (judge + pre-fetch + agent)
├── bias_metrics.py         # Standalone fairness metric functions (Gini, entropy, etc.)
├── dataset_builder.py      # Kaggle → master_dataset.csv + embeddings.npy
├── user_profiles.py        # 20 personas → data/users.json
├── compute_embeddings.py   # Standalone embedding recomputation script
├── recommender.py          # SVD collaborative filter (used by main.py study)
├── main.py                 # Batch bias study pipeline (SVD-based, offline)
├── visualize.py            # Plots for the offline study
├── app.py                  # Legacy Gradio UI (superseded by flask_app.py)
├── data_generator.py       # Synthetic data generator (used by main.py)
├── youtube_loader.py       # Kaggle dataset downloader utility
│
├── test_architecture.py    # Architecture health tests (no LLM, ~15s)
├── test_llm_correction.py  # Biased vs LLM correction tests (requires GROQ_API_KEY)
│
├── templates/
│   └── index.html          # Single-page web UI
│
├── requirements.txt
│
└── data/                   # Auto-generated; not committed
    ├── master_dataset.csv  # ~13K deduplicated videos with virality scores
    ├── embeddings.npy      # (N, 384) float32 sentence-transformer vectors
    └── users.json          # 20 user profiles with interaction histories
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- Groq API key — free tier at [console.groq.com](https://console.groq.com)
- Kaggle API credentials (only needed for first-run dataset download)

### Install

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file inside `bias_recommender/`:

```
GROQ_API_KEY=your_groq_api_key_here
```

### Build the dataset (first run only)

```bash
python dataset_builder.py   # downloads Kaggle data, builds CSV + embeddings (~2 min)
python user_profiles.py     # builds users.json from the master dataset
```

### Launch the web app

```bash
python flask_app.py
```

Opens at `http://localhost:5000`.

1. Select a user persona from the left panel (or click **New User** to enter custom preferences)
2. Click **Run Analysis**
3. Compare the three feeds on the **Bias Analysis** and **Ideal** tabs
4. Type feedback in the refinement box and click **Apply Feedback** to re-run

### Run the tests

```bash
# Architecture tests — no API key needed
python test_architecture.py

# LLM correction tests — needs GROQ_API_KEY
python test_llm_correction.py
python test_llm_correction.py --users U02 U04 U16
```

---

## References

- Zehlike, M. et al. (2017). **FA\*IR: A Fair Top-k Ranking Algorithm**. *ACM CIKM 2017*. [doi:10.1145/3132847.3132938](https://doi.org/10.1145/3132847.3132938)

- Abdollahpouri, H., Burke, R., & Mobasher, B. (2017). **Controlling Popularity Bias in Learning-to-Rank Recommendation**. *RecSys 2017*.

- Yi, X. et al. (2019). **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations** (Two-Tower). *RecSys 2019*. [doi:10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996)

- Noble, S. U. (2018). **Algorithms of Oppression: How Search Engines Reinforce Racism**. New York University Press.

- Yao, S. & Huang, B. (2017). **Beyond Parity: Fairness Objectives for Collaborative Filtering**. *NeurIPS 2017*.
