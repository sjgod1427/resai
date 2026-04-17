# Bias-Aware Video Recommender — Technical Approach

> A Responsible AI research prototype that injects, detects, and corrects
> algorithmic bias in a YouTube-style video recommendation system.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset & Data Pipeline](#2-dataset--data-pipeline)
3. [Two-Tower Recommender](#3-two-tower-recommender)
4. [Bias Injection Mechanism](#4-bias-injection-mechanism)
5. [Ideal Recommender (Zero-Bias Baseline)](#5-ideal-recommender-zero-bias-baseline)
6. [Agentic LLM Supervisor](#6-agentic-llm-supervisor)
7. [Flask Application & UI](#7-flask-application--ui)
8. [Key Engineering Challenges & Solutions](#8-key-engineering-challenges--solutions)
9. [Performance Optimisation History](#9-performance-optimisation-history)
10. [Tech Stack](#10-tech-stack)

---

## 1. Problem Statement

Real video platforms (YouTube, TikTok, Instagram Reels) optimise for engagement
signals — clicks, watch time, shares. A documented side-effect is that high-click
content genres (Entertainment, Music, Gaming) are progressively amplified while
slower-consumption genres (Educational, Documentary, Regional-language content,
News) are de-ranked. This does not happen through explicit policy but through the
compounding weight of engagement multipliers inside the ranking formula.

The effect is self-reinforcing: the platform shows users more viral content → users
click on it → the engagement signal strengthens → the algorithm ranks it higher.
Over time the user's feed narrows, independent of their stated preferences.

**This project makes that mechanism explicit, measurable, and correctable by:**

- Building a realistic biased recommender that encodes real platform engagement data
  as structural weights
- Running an LLM supervisor that independently detects and corrects the bias
- Comparing three feeds side-by-side: Biased / LLM-Corrected / Ideal (zero-bias)

**Suppressed categories** (defined throughout the system):
`Educational`, `Documentary`, `DIY`, `News/Analysis`, `Regional`

---

## 2. Dataset & Data Pipeline

### Source

Real YouTube trending video CSVs from 10 countries via the
`datasnaek/youtube-new` Kaggle dataset (~40,000 raw rows).

| Country | Primary Language |
|---------|-----------------|
| US, CA, GB | English |
| IN | Hindi |
| MX | Spanish |
| FR | French |
| DE | German |
| JP | Japanese |
| KR | Korean |
| RU | Russian |

### Processing (`dataset_builder.py`)

```
Raw CSVs (10 countries)
        │
        ▼
  Keep peak-views row per video per country
        │
        ▼
  Deduplicate English countries (US/CA/GB share video_ids)
        │
        ▼
  Global dedup: same video_id across all countries → keep highest-view row
        │   Note: a Korean music video that trended in Korea AND Japan
        │   becomes ONE row. Without this step, cosine-sim lookup returns
        │   the same video multiple times — a critical correctness bug.
        ▼
  Map YouTube category_id → 8 project genres
  Non-English lifestyle/vlog categories → Regional
        │
        ▼
  Stratified cap: top-300 per (genre × language) bucket → ~12,852 unique videos
        │
        ▼
  Virality score: 0.5×views_norm + 0.3×likes_norm + 0.2×engagement_rate_norm
        │
        ▼
  SentenceTransformer embeddings (all-MiniLM-L6-v2, 384-dim)
  Encoded as: "{title} [{genre}] [{language}]"
        │
        ▼
  master_dataset.csv  +  embeddings.npy  (rows aligned 1-to-1)
```

### Genre Mapping

| YouTube Category IDs | Project Genre | Suppressed? |
|----------------------|---------------|-------------|
| 27, 28 | Educational | Yes |
| 25 | News/Analysis | Yes |
| 29, 35 | Documentary | Yes |
| 26 | DIY | Yes |
| Non-English lifestyle | Regional | Yes |
| 10 | Music | No |
| 20 | Gaming | No |
| 1,2,15,17,19–24 | Entertainment | No |

---

## 3. Two-Tower Recommender

A two-tower architecture uses two independent encoder networks — one for users,
one for items — and retrieves items by computing similarity between the resulting
embeddings. In production systems (Google, YouTube) both towers are deep neural
networks trained jointly on interaction logs. Here we use a non-parametric
equivalent appropriate for the dataset size (20 users × ~35 interactions = 700
data points — too few to train neural weights).

### User Tower

Built at query time from the user's watch history:

```
user_vec  =  Σᵢ  (rating_i / Σ ratings)  ×  embedding(watched_video_i)
user_vec  =  user_vec / ‖user_vec‖₂
```

- Each watched video contributes its pre-computed embedding, weighted by the
  user's rating (1–5)
- L2-normalised to unit sphere — compatible with cosine similarity
- Zero-shot: no separate user model to train; the tower is fully derived from
  interaction history

### Item Tower

Pre-computed `embeddings.npy` — 12,852 × 384 float32 array.

Each entry is the `all-MiniLM-L6-v2` SentenceTransformer embedding of:
`"{title} [{genre}] [{language}]"`

The genre and language tags are appended so that genre/language similarity is
captured in the embedding space even when titles don't signal them.

### Retrieval

```
score(user, item)  =  cosine_similarity(user_vec, item_embedding)
```

This is the pure relevance signal — identical for both the biased and ideal
recommenders. The difference lies entirely in what gets added on top.

---

## 4. Bias Injection Mechanism

The biased recommender adds a weighted engagement signal on top of the
relevance score, mimicking how real platforms amplify engagement-optimised content.

### Scoring Formula

```
relevance        = cosine_similarity(user_vec, item_vec)

bias_component   = virality_score × genre_engagement_mult × language_mult
bias_component   = bias_component / max(bias_component)      ← normalise to [0,1]

final_score      = 0.60 × relevance  +  0.40 × bias_component
```

### Genre Engagement Multipliers

Derived from documented platform engagement benchmarks:

| Genre | Multiplier | Effect |
|-------|-----------|--------|
| Entertainment | 1.00 | baseline |
| Music | 0.93 | −7% |
| Gaming | 0.87 | −13% |
| DIY | 0.76 | −24% |
| News/Analysis | 0.72 | −28% |
| Educational | 0.68 | −32% |
| Documentary | 0.64 | −36% |
| Regional | 0.61 | −39% |

### Language Multiplier

Non-English content: `0.78` (−22%)

This reflects the English-language bias present in global platform training data
and content moderation pipelines.

### Why 60/40?

At 60% relevance / 40% bias the bias is realistic but not fabricated. A user who
strongly prefers Documentaries will still see 2–3 Documentaries in their biased
feed (relevance pushes them through), but Entertainment and viral Gaming content
will systematically edge past niche educational material. This matches what
platform researchers have documented in audits of real recommendation systems.

---

## 5. Ideal Recommender (Zero-Bias Baseline)

```python
score = cosine_similarity(user_vec, item_vec)
```

No virality weighting. No genre multipliers. No language penalty.

The ideal recommender is a direct implementation of the user tower retrieval step
with nothing else. It answers the question: *"What would the algorithm recommend
if it only cared about content-relevance to this specific user?"*

By showing all three feeds side-by-side, the UI makes the gap between what the
platform serves (biased), what a fair correction looks like (LLM-supervised), and
what pure relevance retrieval would produce (ideal) immediately visible.

---

## 6. Agentic LLM Supervisor

The supervisor runs after the biased recommender and has two sequential steps.

### Step 1 — Judge (Structured LLM Call)

A single structured output call using LangChain's `with_structured_output`:

```python
class BiasAssessment(BaseModel):
    is_biased:                bool
    reasoning:                str
    genres_over_represented:  List[str]
    genres_missing:           List[str]
    tier1_slots:              int   # similarity + trending
    tier2_slots:              int   # genre retrieval
    tier3_slots:              int   # diversity
    tier4_slots:              int   # global viral fill
```

The judge receives:
- User profile (name, description, preferred genres/languages)
- Watch history summary (genre/language breakdown + 6 recent titles)
- The full biased feed summary (genre breakdown, suppressed count, non-English count)

It flags the feed as biased if:
- Fewer than 6 videos from suppressed categories
- Fewer than 6 non-English videos (when user prefers non-English content)
- User's preferred genres are substantially absent from the recommendations

If not biased, the biased feed is returned as-is (no correction needed).

### Step 2 — Agentic Correction (Tool-Calling Agent)

If the judge flags bias, a LangChain `create_agent` tool-calling agent is
initialised with 5 tools and a hard-constraint system prompt.

#### Pre-Fetch Optimisation

Before the agent starts, all four retrieval tiers are executed in Python
(no LLM, ~50ms) to build a candidate pool of ~38–50 videos:

```
T1 (personalised + trending) — semantic cosine-sim in target genres
T2 (genre retrieval)         — top-viral in suppressed categories
T3 (diversity)               — one video per genre outside user's bubble
T4 (viral fill)              — globally top-viral, any genre/language

All 30 biased video_ids are excluded from the candidate pool.
```

The numbered candidate table is passed to the agent in its first message.
The agent selects 30 row numbers and calls `submit_selection` — typically
in a single LLM turn rather than 4–5 sequential search calls.

#### Tools

| Tool | Role | When used |
|------|------|-----------|
| `submit_selection` | Lock in 30 selections by row number from the pre-fetched table | Primary path — called immediately |
| `search_similar_trending` | T1 retrieval by genre | Fallback — if pre-fetch is insufficient |
| `search_by_genre` | T2 genre retrieval | Fallback |
| `search_diverse` | T3 diversity retrieval | Fallback |
| `search_viral` | T4 viral fill | Fallback |

#### Server-Side Enforcement (`submit_selection`)

```python
if len(unique_valid) < 30:
    return f"REJECTED: only {len(unique_valid)} rows selected. Need {needed} more."
```

The agent cannot complete with fewer than 30 valid selections. If it submits
an incomplete list, the tool rejects the call and the agent must retry.

#### Hard Constraints (system prompt)

- Exactly 30 videos
- ≥ 6 from suppressed categories
- ≥ 6 non-English videos
- No single genre > 15 slots

#### Fallback

If the agent errors (rate limit, recursion limit, etc.) without submitting,
the pre-fetched pool itself is used directly as the corrected list. Since
the pool was built with fairness-aware tier weights, this fallback always
produces diverse, relevant, non-biased results — it is not a degraded path.

### Model & Rate Limit Handling

| Model | Role |
|-------|------|
| `openai/gpt-oss-120b` (Groq) | Primary |
| `openai/gpt-oss-20b` (Groq) | Auto-fallback on 429 rate limit (judge step) |

---

## 7. Flask Application & UI

### Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Renders the SPA; injects user list + genre colour map via Jinja |
| `/api/user/<id>` | GET | Returns user profile + watch history as JSON |
| `/api/run` | POST | Runs biased + unbiased retrieval + LLM supervisor; returns all three feeds + metrics |

### Layout

```
┌─────────────────────────────────────────────────────────┐
│  Header: logo · title · [user dropdown] · tech pills    │
├──────────────────┬──────────────────────────────────────┤
│  Left panel      │  Right panel (scrollable)            │
│  (400px, fixed)  │                                      │
│                  │  [Run Bias Analysis]                  │
│  User profile    │                                      │
│  · photo         │  ┌─────────────────────────────────┐ │
│  · name/id       │  │ Tabs: Bias Analysis | Ideal     │ │
│  · description   │  ├────────────────┬────────────────┤ │
│  · genre bars    │  │ Biased Feed    │ LLM-Corrected  │ │
│  · watch history │  │ (30 items)     │ (30 items)     │ │
│                  │  ├────────────────┴────────────────┤ │
│  (never scrolls  │  │ LLM Verdict + Tier Breakdown    │ │
│   or squishes)   │  ├─────────────────────────────────┤ │
│                  │  │ 2-Way Metrics (Biased/Corrected)│ │
│                  │  └─────────────────────────────────┘ │
│                  │                                      │
│                  │  Ideal tab:                          │
│                  │  ┌─────────────────────────────────┐ │
│                  │  │ Zero-Bias Feed (30 items)       │ │
│                  │  ├─────────────────────────────────┤ │
│                  │  │ 3-Way Metrics (B / C / Ideal)   │ │
│                  │  └─────────────────────────────────┘ │
└──────────────────┴──────────────────────────────────────┘
```

### Key UI Details

- User selector is a `<select>` dropdown in the header (replaces old sidebar list)
- Left panel profile card is `overflow: hidden` — stays fixed regardless of results length
- All cards use glassmorphism: `backdrop-filter: blur(22px)` + `rgba(255,255,255,0.68)`
- Soft indigo-purple gradient background with decorative radial blobs
- Real portrait photos via randomuser.me mapped per persona; initials fallback on error
- Results rendered via `fetch()` → DOM update, no page reload

---

## 8. Key Engineering Challenges & Solutions

### Challenge 1: Duplicate video_ids

**Problem:** The same YouTube video trending in multiple non-English countries
(e.g., a K-Pop video trending in both Korea and Japan) created multiple rows
with the same `video_id`. The LLM supervisor's `isin()` filter returned all
matching rows, causing the corrected feed to return 32+ items instead of 30.

**Solution (three-layer):**
1. `dataset_builder.py` — global dedup at source: keep highest-view row per `video_id`
2. `LLMSupervisor.__init__` — dedup at load time using boolean mask applied
   simultaneously to both the DataFrame and the `embeddings` numpy array
   (critical: if only the DataFrame is deduped, embeddings become misaligned)
3. `fix()` — safety-net `.drop_duplicates("video_id")` after the `isin` filter

### Challenge 2: Embeddings Alignment

**Problem:** After `drop_duplicates` + `reset_index(drop=True)`, the positional
index must match the embeddings array row index at all times.

**Solution:** A single boolean mask `keep = (~dup_mask).values` is applied to
both structures before either is reset:

```python
keep       = (~dup_mask).values
df         = df[keep].reset_index(drop=True)
embeddings = embeddings[keep]
```

### Challenge 3: Mojibake in Video Titles

**Problem:** UTF-8 YouTube titles read on Windows as Latin-1 produced sequences
like `â\x80\x99` for the Unicode apostrophe `'`. The leading byte `\xe2` was
silently discarded, leaving control characters in titles.

**Solution:** Explicit substitution map for common sequences
(`\x80\x99 → '`, `\x80\x9c → "`, `\x80\x93 → -`, etc.) followed by
regex removal of all remaining C0/C1 control characters:

```python
re.sub(r'[\x00-\x1f\x7f-\x9f]', '', title)
```

### Challenge 4: LLM Rate Limits (Groq 8,000 TPM)

**Problem:** The agent's input prompt with a 30-video biased feed summary was
7,080 tokens. After the judge already used ~2,400 tokens in the same minute,
the agent's first LLM call hit the 8,000 TPM rate limit immediately. The agent
errored without calling any tools, `state["seen"]` contained only the 30 biased
IDs, and the fallback returned those same IDs as the "corrected" feed — identical
to the biased output.

**Solution:** Truncate the agent input to top-10 biased items (from 7,080 tokens
to ~381 tokens). The agent no longer needs the full biased list; it only needs
the bias assessment and the pre-fetched candidates.

### Challenge 5: Agent Tool-Call JSON Generation Failure

**Problem:** The LLM attempted to generate a `submit_final_list` call with
`video_ids` as a comma-separated string of 30 eleven-character IDs (360+ chars
of dense string). The JSON generation failed — only `reasoning` was included in
the arguments, `video_ids` was dropped entirely.

**Root cause:** Generating 30 long arbitrary strings in a single JSON field is
error-prone for LLMs. The model has to maintain the exact character sequences of
30 IDs simultaneously while also generating valid JSON structure.

**Solution:** Changed from raw video_id submission to index-based submission.
The candidate table is numbered 1–N. The agent submits row numbers
(small integers like `"1,3,5,7,9,..."`) which are resolved server-side to
video_ids. This reduces the output the LLM must generate from ~360 chars of
opaque IDs to ~60 chars of small numbers.

### Challenge 6: Agent Speed (71 seconds)

**Problem:** Each search tool call requires a full LLM round-trip (~3–5s on
Groq). Gathering 30 videos required 4–5 sequential tool calls → 20–25s of LLM
time plus retry overhead on rate limits.

**Solution:** Pre-fetch all candidates in Python before the agent starts.
The pre-fetch runs all four retrieval tiers (pure numpy + pandas, ~50ms) and
hands a numbered table to the agent. The agent now calls `submit_selection` in
a single LLM turn instead of orchestrating 4–5 tool calls.

| Path | Before | After |
|------|--------|-------|
| Not biased (judge only) | ~5s | ~3s |
| Biased (full correction) | ~71s | ~10s |

---

## 9. Performance Optimisation History

| Version | Architecture | Time | Notes |
|---------|-------------|------|-------|
| v1 | 10-item retrieval, agent calls 4–5 tools | ~40s | Slow, rate-limited |
| v2 | 30-item retrieval, large prompt | ~71s | Rate limit on first agent call; corrected = biased |
| v3 | Compact prompt (381 tokens) | ~9.6s | Agent tool-call JSON fails; fallback is correct but agent path broken |
| v4 (current) | Pre-fetch + index submission | ~10s (biased path) / ~3s (fair path) | Agent calls submit in 1 turn; fallback is also reliable |

---

## 10. Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Web framework | Flask 3.x |
| Data | pandas, numpy |
| Embeddings | `sentence-transformers` / `all-MiniLM-L6-v2` (384-dim) |
| Similarity | `sklearn.metrics.pairwise.cosine_similarity` |
| LLM | `openai/gpt-oss-120b` via Groq API |
| LLM framework | LangChain 1.x, LangGraph (create_agent, tool calling) |
| Structured output | Pydantic `BaseModel` + `with_structured_output` |
| Frontend | Vanilla JS `fetch()`, CSS glassmorphism, Google Fonts Inter |
| Dataset | Kaggle `datasnaek/youtube-new` (~40K rows, 10 countries) |
| Environment | `python-dotenv` (.env for GROQ_API_KEY) |

---

## User Personas

20 synthetic personas covering diverse demographics, languages, and genre
preferences. Each has ~35 pre-sampled interaction histories drawn from real
YouTube videos that match their profile.

| ID | Name | Primary Genres | Languages |
|----|------|---------------|-----------|
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

## References

- Zehlike et al. (2017). **FA\*IR: A Fair Top-k Ranking Algorithm**. ACM CIKM 2017.
- Abdollahpouri et al. (2017). **Controlling Popularity Bias in Learning-to-Rank Recommendation**. RecSys 2017.
- Yi et al. (2019). **Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations** (Two-Tower). RecSys 2019.
- Noble, S. U. (2018). **Algorithms of Oppression**. New York University Press.
- Yao & Huang (2017). **Beyond Parity: Fairness Objectives for Collaborative Filtering**. NeurIPS 2017.
