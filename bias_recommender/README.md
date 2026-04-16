# Bias-Aware Video Recommender

A Responsible AI demo that **injects**, **detects**, and **corrects** algorithmic bias in a YouTube-style video recommender. Built on real YouTube trending data, a two-tower neural retrieval model, and an agentic LLM supervisor.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [System Architecture](#system-architecture)
4. [Data Pipeline](#data-pipeline)
5. [Two-Tower Recommender](#two-tower-recommender)
6. [Agentic LLM Supervisor](#agentic-llm-supervisor)
7. [User Personas](#user-personas)
8. [Project Structure](#project-structure)
9. [Installation & Usage](#installation--usage)
10. [References](#references)

---

## Overview

| Component | Description |
|---|---|
| **Dataset** | Real YouTube trending videos from 10 countries (~13K unique videos, 8 genres, 8 languages) |
| **Biased Recommender** | Two-tower similarity model with realistic platform engagement bias |
| **LLM Supervisor** | LangChain agentic loop (Groq `openai/gpt-oss-120b`) that detects bias and builds a corrected 10-video feed using 5 retrieval tools |
| **UI** | Gradio web app — select one of 20 user personas, run test, compare biased vs corrected feed side-by-side |

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
                        ┌──────────────────────────────────┐
                        │         Real YouTube Data         │
                        │  10 countries · 8 genres          │
                        │  8 languages · ~13K unique videos  │
                        └────────────┬─────────────────────┘
                                     │ dataset_builder.py
                                     ▼
                        ┌──────────────────────────────────┐
                        │  master_dataset.csv              │
                        │  + embeddings.npy                │
                        │  (SentenceTransformer all-MiniLM) │
                        └────────────┬─────────────────────┘
                                     │
              ┌──────────────────────┴────────────────────────┐
              │                                               │
              ▼                                               ▼
 ┌─────────────────────────┐               ┌──────────────────────────────┐
 │  20 User Personas        │               │  LLMSupervisor (app startup)  │
 │  (users.json)            │               │  Deduplicates master_df       │
 │  Each with watch history │               │  Aligns embeddings 1-to-1    │
 └──────────┬──────────────┘               └──────────────┬───────────────┘
            │                                              │
            ▼                                              │
 ┌──────────────────────────────────────────┐             │
 │        get_biased_recs()  [app.py]        │             │
 │                                           │             │
 │  USER TOWER                               │             │
 │  Rating-weighted avg of watched-video     │             │
 │  embeddings → user_vec (384-dim)          │             │
 │                                           │             │
 │  ITEM TOWER                               │             │
 │  Pre-computed embeddings.npy              │             │
 │                                           │             │
 │  SCORE                                    │             │
 │  (0.38 × cosine_sim                       │             │
 │   + 0.62 × virality_score)                │             │
 │  × genre_engagement_mult  [0.61 – 1.00]   │             │
 │  × language_mult  [0.78 English/non-Eng]  │             │
 │                                           │             │
 │  → Biased top-10                          │             │
 └──────────────────┬───────────────────────┘             │
                    │                                      │
                    └─────────────────┬────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      LLMSupervisor.fix()             │
                    │                                     │
                    │  ┌──────────────────────────────┐   │
                    │  │  Step 1 — Judge              │   │
                    │  │  Structured LLM call         │   │
                    │  │  → BiasAssessment schema     │   │
                    │  │    is_biased: bool           │   │
                    │  │    genres_missing: [...]     │   │
                    │  │    tier1/2/3/4 slot counts   │   │
                    │  └──────────────┬───────────────┘   │
                    │                 │ if biased          │
                    │  ┌──────────────▼───────────────┐   │
                    │  │  Step 2 — Agentic Correction │   │
                    │  │                              │   │
                    │  │  AGENT (gpt-oss-120b)        │   │
                    │  │  + 5 tools:                  │   │
                    │  │                              │   │
                    │  │  search_similar_trending()   │   │
                    │  │  ↳ Two-tower cosine sim      │   │
                    │  │    in target genres          │   │
                    │  │                              │   │
                    │  │  search_by_genre()           │   │
                    │  │  ↳ Top-viral within genre    │   │
                    │  │                              │   │
                    │  │  search_diverse()            │   │
                    │  │  ↳ Outside user's bubble     │   │
                    │  │                              │   │
                    │  │  search_viral()              │   │
                    │  │  ↳ Globally top-viral        │   │
                    │  │                              │   │
                    │  │  submit_final_list()  ←──────┼───┼── server-side
                    │  │  ↳ Validates & locks 10 IDs  │   │   enforcement
                    │  │    Rejects if < 10 valid IDs │   │
                    │  └──────────────────────────────┘   │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                         Corrected top-10 (always exactly 10)
                         ≥ 2 suppressed-category videos
                         ≥ 2 non-English videos
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
- No separate user model to train; the tower is zero-shot from interaction history

### Item Tower

Pre-computed `embeddings.npy` — `SentenceTransformer` vectors computed once at build time.

### Scoring (Biased Recommender)

```
score = (0.38 × cosine_sim(user_vec, item_vec)
         + 0.62 × virality_score)
        × genre_engagement_mult
        × language_mult
```

**Genre engagement multipliers** (real platform signal proxies):

| Genre | Multiplier | Effect |
|---|---|---|
| Entertainment | 1.00 | Baseline |
| Music | 0.93 | −7% |
| Gaming | 0.87 | −13% |
| DIY | 0.76 | −24% |
| News/Analysis | 0.72 | −28% |
| Educational | 0.68 | −32% |
| Documentary | 0.64 | −36% |
| Regional | 0.61 | −39% |

**Language multiplier:** non-English = `0.78` (−22%)

The bias is proportionate and realistic: a user who loves documentaries still sees 1–2 documentaries (relevance pushes them up), but entertainment and viral content systematically edge past them because the platform weights its engagement signal at 62%.

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
    tier1_slots: int   # similarity + trending
    tier2_slots: int   # genre retrieval fallback
    tier3_slots: int   # diversity outside bubble
    tier4_slots: int   # global viral fill
```

Bias is flagged if the feed:
- Suppresses genres that dominate the user's watch history
- Has fewer than 2 suppressed-category videos
- Has fewer than 2 non-English videos (when user prefers non-English)

### Step 2 — Agentic Correction

A LangChain tool-calling agent (`create_agent`) is initialised with 5 tools and a system prompt encoding the hard fairness constraints. The agent iteratively calls search tools and must satisfy `submit_final_list` before it can finish.

#### Tools

| Tool | Tier | Description |
|---|---|---|
| `search_similar_trending` | T1 | Two-tower cosine similarity to user history in target genres, scored by `0.65 × sim + 0.35 × virality` |
| `search_by_genre` | T2 | Top-viral videos from specified genres, language-filtered to user's preferences |
| `search_diverse` | T3 | Videos from genres outside the user's normal bubble (one per unseen genre) |
| `search_viral` | T4 | Globally top-viral videos regardless of genre or language |
| `submit_final_list` | — | Validates and locks in the final 10 IDs — **server-side enforcement** |

#### `submit_final_list` Enforcement

```python
# Inside submit_final_list tool:
if len(unique_valid) < 10:
    return f"REJECTED: only {len(unique_valid)} unique valid IDs. "
           f"Call search tools to get {10 - len(unique_valid)} more, then resubmit."
```

The agent physically cannot finish with fewer than 10 valid IDs. It loops until it satisfies the constraint or exhausts the context.

#### Hard Constraints (encoded in system prompt)

- Exactly 10 videos
- ≥ 2 from suppressed categories
- ≥ 2 non-English videos
- No single genre > 5 slots
- Tool priority: T1 → T2 → T3 → T4

### Model & Fallback

| Model | Use |
|---|---|
| `openai/gpt-oss-120b` | Primary |
| `openai/gpt-oss-20b` | Auto-fallback on 429 rate limit |

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

## Project Structure

```
bias_recommender/
│
├── app.py                  # Gradio UI + two-tower biased recommender
├── llm_supervisor.py       # LangChain agentic supervisor (judge + correction)
├── dataset_builder.py      # Kaggle → master_dataset.csv + embeddings.npy
├── user_profiles.py        # 20 personas → data/users.json
├── compute_embeddings.py   # Standalone embedding recomputation script
├── requirements.txt
│
└── data/                   # Auto-generated on first run
    ├── master_dataset.csv  # ~13K deduplicated videos with virality scores
    ├── embeddings.npy      # (N, 384) float32 sentence-transformer vectors
    └── users.json          # 20 user profiles with interaction histories
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- Groq API key — free tier at [console.groq.com](https://console.groq.com)
- Kaggle API credentials (for first-run dataset download)

### Install

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

### First Run (builds data)

On the very first launch `app.py` automatically:
1. Downloads the YouTube dataset via `kagglehub`
2. Builds `master_dataset.csv`
3. Computes `embeddings.npy` (~2 min on CPU)
4. Builds `users.json`

Or run each step manually:

```bash
python dataset_builder.py   # build CSV + embeddings
python user_profiles.py     # build user interaction histories
```

### Launch the App

```bash
python app.py
```

Opens at `http://localhost:7861`. Select a user persona, click **Run Test**, and compare the biased and LLM-corrected feeds side-by-side.

---

## References

- Zehlike, M. et al. (2017). **FA\*IR: A Fair Top-k Ranking Algorithm**. *ACM CIKM 2017*. [doi:10.1145/3132847.3132938](https://doi.org/10.1145/3132847.3132938)

- Abdollahpouri, H., Burke, R., & Mobasher, B. (2017). **Controlling Popularity Bias in Learning-to-Rank Recommendation**. *RecSys 2017*.

- Yi, X. et al. (2019). **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations** (Two-Tower). *RecSys 2019*. [doi:10.1145/3298689.3346996](https://doi.org/10.1145/3298689.3346996)

- Noble, S. U. (2018). **Algorithms of Oppression: How Search Engines Reinforce Racism**. New York University Press.

- Yao, S. & Huang, B. (2017). **Beyond Parity: Fairness Objectives for Collaborative Filtering**. *NeurIPS 2017*.
