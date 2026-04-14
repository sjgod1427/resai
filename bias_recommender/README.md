# Responsible AI — Content-Type & Language Bias in Recommender Systems

A full research pipeline that **injects**, **measures**, and **mitigates** algorithmic bias in a YouTube-style video recommender system. Built as a Responsible AI study to demonstrate how engagement-optimised platforms systematically suppress Educational, Documentary, Regional, and non-English content.

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation & Research Question](#motivation--research-question)
3. [Project Structure](#project-structure)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Data Generation](#data-generation)
6. [Recommender Systems](#recommender-systems)
   - [BiasedRecommender](#biasedrecommender)
   - [FairRecommender](#fairrecommender)
7. [Bias Metrics](#bias-metrics)
8. [Visualisations](#visualisations)
9. [Gradio Web Interface](#gradio-web-interface)
10. [Results Summary](#results-summary)
11. [Installation & Usage](#installation--usage)
12. [References](#references)

---

## Overview

This project simulates the algorithmic pipeline of a large video platform and studies two variants of the same collaborative filtering model:

| Model | Description |
|---|---|
| **BiasedRecommender** | SVD-based CF with score suppression on Educational, Documentary, DIY, News/Analysis, Regional, and non-English content — mimicking real engagement-maximising systems |
| **FairRecommender** | Same SVD base, but with a fairness-aware greedy re-ranking layer that enforces minimum representation quotas |

Both models are evaluated against six quantitative fairness metrics and the differences are visualised across six chart types. A four-tab Gradio web app makes every step of the pipeline interactive.

---

## Motivation & Research Question

> *"If a platform recommends what users click on, and users click on what they've been shown, who decides what gets shown first?"*

Recommendation algorithms on platforms like YouTube optimise for engagement signals (clicks, watch time, likes). A side effect — well-documented in academic literature — is that content with high initial engagement (Entertainment, Music, Gaming) gets amplified, while content that is slower to attract clicks despite genuine user interest (Educational, Documentary, Regional-language) gets progressively de-ranked.

This creates a feedback loop:
1. Suppressed content gets fewer impressions
2. Fewer impressions → fewer clicks → lower engagement score
3. Lower engagement score → even fewer recommendations
4. Creators of suppressed content earn less → produce less

This project makes that mechanism **explicit and measurable**.

---

## Project Structure

```
bias_recommender/
│
├── data_generator.py     # Synthetic data: videos, users, interactions
├── recommender.py        # BiasedRecommender & FairRecommender (SVD CF)
├── bias_metrics.py       # Six quantitative fairness metrics
├── visualize.py          # Six matplotlib plots (CLI output)
├── main.py               # Full CLI pipeline runner
├── app.py                # Interactive Gradio web app (4 tabs)
├── requirements.txt      # Python dependencies
│
└── data/                 # Auto-generated CSVs (created on first run)
    ├── creators.csv
    ├── videos.csv
    ├── users.csv
    └── interactions.csv
```

Plots are saved to `../plots/bias_study/` relative to the module when running `main.py`.

---

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────────┐
│  Synthetic Data  │────▶│  SVD CF Training  │────▶│  BiasedRecommender    │
│  500 videos      │     │  (k=40 factors)   │     │  Suppression x0.30–   │
│  200 users       │     │  Truncated SVD    │     │  x0.50 per category   │──▶ Biased top-10
│  ~7K interactions│     │  on rating matrix │     │  x0.40 non-English    │
└─────────────────┘     └──────────────────┘     └───────────────────────┘
                                  │
                                  └──────────────▶ ┌───────────────────────┐
                                                   │  FairRecommender      │
                                                   │  Raw CF scores only   │
                                                   │  + Quota re-ranking   │──▶ Fair top-10
                                                   │  (FA*IR-inspired)     │
                                                   └───────────────────────┘
                                                              │
                                              ┌───────────────▼───────────────┐
                                              │   Bias Audit (6 metrics)      │
                                              │   + 6 Visualisation Charts    │
                                              └───────────────────────────────┘
```

---

## Data Generation

`data_generator.py` produces a fully synthetic but realistic YouTube-like dataset. All random seeds are fixed (`numpy.random.seed(42)`) for reproducibility.

### Video Library (500 videos)

| Category | Library Share | Suppressed? |
|---|---|---|
| Educational | 20% | Yes |
| Entertainment | 15% | No (Boosted) |
| Music | 15% | No (Boosted) |
| Gaming | 13% | No (Boosted) |
| Regional | 12.5% | Yes |
| Documentary | 11% | Yes |
| DIY | 8% | Yes |
| News/Analysis | 4% | Yes |

**65.5% of the video library is from suppressed categories.**

Each video carries: `video_id`, `title`, `category`, `language`, `creator_id`, `creator_size`, `views`, `likes`, `engagement_rate`, `avg_watch_pct`, `is_suppressed`, `is_english`.

### Language Distribution

Seven languages are represented with realistic per-category distributions:

| Language | Library Share |
|---|---|
| English | ~47% |
| Spanish | ~13% |
| Hindi | ~13% |
| Arabic | ~9% |
| French | ~9% |
| Portuguese | ~5% |
| Korean | ~4% |

Regional-category videos are weighted heavily toward non-English; Educational and Documentary videos have substantial non-English shares too.

### Users (200 users)

Six user profile types, each with a preference distribution over categories:

| Profile | Primary Interest |
|---|---|
| `student` | Educational 35%, Gaming 25%, Entertainment 20% |
| `professional` | News/Analysis 25%, Educational 25%, Documentary 20% |
| `casual` | Entertainment 40%, Music 25%, Gaming 20% |
| `regional` | Regional 40%, Music 20%, Entertainment 15% |
| `creative` | DIY 30%, Documentary 20%, Educational 20% |
| `news_junkie` | News/Analysis 40%, Documentary 25%, Regional 15% |

### Interactions (~7,000 rows)

Each interaction is sampled using the user's profile preference distribution to pick a category, then a random video from that category. Ratings (1–5) are scaled from preference strength plus Gaussian noise. Watch percentage is drawn from a Beta distribution, with Educational and Documentary content having higher completion rates.

---

## Recommender Systems

Both recommenders share the same `SVDCollaborativeFilter` base.

### Base Model: SVD Collaborative Filtering

1. Builds a sparse user-item rating matrix (200 × 500)
2. Mean-centres each user's ratings
3. Applies truncated SVD with `k=40` latent factors (`scipy.sparse.linalg.svds`)
4. Reconstructs a dense predicted-score matrix: `U × Σ × Vᵀ + global_mean`

The reconstructed matrix gives a score for every (user, video) pair, including unobserved ones.

---

### BiasedRecommender

After computing raw CF scores, a **score multiplier** is applied per video before ranking:

**Suppression multipliers (applied to CF score):**

| Category | Multiplier | Effect |
|---|---|---|
| Regional | ×0.30 | −70% score |
| Educational | ×0.35 | −65% score |
| News/Analysis | ×0.40 | −60% score |
| Documentary | ×0.45 | −55% score |
| DIY | ×0.50 | −50% score |

**Boost multipliers:**

| Category | Multiplier | Effect |
|---|---|---|
| Entertainment | ×1.40 | +40% score |
| Music | ×1.30 | +30% score |
| Gaming | ×1.20 | +20% score |

**Non-English penalty:** ×0.40 stacked on top of any category multiplier.

A non-English Educational video therefore receives: `score × 0.35 × 0.40 = score × 0.14` — an **86% score reduction**.

---

### FairRecommender

Uses the same SVD base but applies **no suppression**. Instead, it re-ranks candidates using a greedy quota algorithm:

**Step 1 — Score all videos** with raw CF scores (no multipliers).

**Step 2 — Build a candidate pool** of the top `3 × N` videos by raw score.

**Step 3 — Fill quota slots first:**

| Group | Minimum share of top-10 |
|---|---|
| Educational | ≥ 15% (≥ 2 slots) |
| Regional | ≥ 10% (≥ 1 slot) |
| DIY | ≥ 8% (≥ 1 slot) |
| Documentary | ≥ 8% (≥ 1 slot) |
| News/Analysis | ≥ 5% (≥ 1 slot) |
| Non-English | ≥ 25% (≥ 3 slots) |

Within each quota group, videos are selected in descending raw-score order (highest-quality first).

**Step 4 — Fill remaining slots** with the highest-scored remaining candidates regardless of category.

This is a greedy proportional quota approach inspired by the **FA\*IR algorithm** (Zehlike et al., 2017). It guarantees minimum representation without fully sacrificing relevance — quota slots are filled by the best-scored video of each required type.

---

## Bias Metrics

`bias_metrics.py` implements six fairness metrics.

### 1. Exposure Ratio
Percentage of total recommendation slots occupied by each category or language group.

### 2. Representation Gap
For each group: `gap = exposure_% − library_%`
- Positive → over-represented vs the library
- Negative → under-represented (suppressed)

### 3. Demographic Parity Gap
`|P(rec = Educational) − P(rec = Entertainment)|`
Ranges from 0 (perfect parity) to 1 (total disparity). Measures inequality between the most-suppressed and most-boosted categories.

### 4. Gini Coefficient
Inequality of the exposure distribution across all categories, computed via the Lorenz curve formula.

| Range | Interpretation |
|---|---|
| < 0.10 | Very fair |
| 0.10 – 0.25 | Mild inequality |
| 0.25 – 0.40 | Moderate inequality |
| > 0.40 | Severe bias |

### 5. Language Diversity Score
Normalised Shannon entropy of the language distribution in recommendations:

```
H = -Σ pᵢ · log₂(pᵢ)
score = H / log₂(n_languages)
```

1.0 = maximum diversity across all languages; 0.0 = all content in a single language.

### 6. Intra-List Diversity
Average number of **unique categories** per user's top-10 recommendation list. Higher values indicate the system exposes users to a wider range of content types.

---

## Visualisations

`visualize.py` generates six charts saved to `plots/bias_study/`:

| File | Chart |
|---|---|
| `1_category_exposure.png` | Grouped bar chart: Library % vs Biased % vs Fair % per category, with suppressed categories shaded |
| `2_language_distribution.png` | Same grouped bar layout for all seven languages |
| `3_representation_gap_heatmap.png` | Colour-coded heatmap of representation gaps (green = closer to parity, red = more biased) |
| `4_gini_lorenz_curve.png` | Lorenz curves for Library, Biased, and Fair systems — distance from the equality diagonal shows degree of inequality |
| `5_user_feed_simulation.png` | Per-video colour blocks for a sample user's top-20 feed, coloured by category |
| `6_summary_metrics.png` | Side-by-side normalised bars for all six scalar metrics with raw values labelled |

---

## Gradio Web Interface

`app.py` provides a four-tab interactive dashboard at `http://localhost:7860`.

### Tab 1 — Initialize Pipeline
- Runs data generation, trains both recommenders, and generates all recommendations
- Displays stat cards (video count, user count, interaction count, suppressed video count)
- Shows a live library distribution chart (category bar + language pie)
- Populates user and video dropdowns for other tabs

### Tab 2 — Feed Simulator
- Select any of the 200 users from the dropdown
- Side-by-side HTML cards showing their Biased vs Fair top-10
- Each card is colour-coded by category; suppressed videos are badge-labelled
- Companion bar chart showing feed composition at a glance

### Tab 3 — Bias Audit Dashboard
- Runs the full six-metric audit across all 2,000 recommendation rows
- HTML table comparing Biased vs Fair with IMPROVED / NO CHANGE verdict badges
- Representation gap table showing every category's library %, biased exposure %, and fair exposure %
- Four chart panels: category exposure, language exposure, Lorenz curve, summary metrics

### Tab 4 — Score Inspector
- Select any video from the full catalogue
- Shows the exact suppression multipliers applied (category + language + combined)
- Score waterfall: Raw CF → After category penalty → After language penalty
- Algorithmic reach panel: how many of the 200 users receive this video in each recommender
- Matplotlib waterfall bar chart + score-loss pie chart

---

## Results Summary

Typical results from a single pipeline run (seeds fixed, results are deterministic):

| Metric | Biased | Fair | Change |
|---|---|---|---|
| Gini Coefficient (↓ better) | 0.6503 | 0.4123 | −0.238 ✓ |
| Language Diversity Score (↑ better) | 0.0000 | 0.8581 | +0.858 ✓ |
| Intra-List Diversity (↑ better) | 1.18 cats/user | 3.93 cats/user | +2.75 ✓ |
| Suppressed Content Rate (↑ better) | 0.0% | 78.5% | +78.5 pp ✓ |
| Non-English Content Rate (↑ better) | 0.0% | 60.3% | +60.3 pp ✓ |
| Demographic Parity Gap (↓ better) | 0.9775 | 0.0500 | −0.928 ✓ |

**Category exposure (Biased):** Entertainment absorbs 97.8% of all recommendation slots; Educational, Regional, Documentary, DIY, and News/Analysis each receive 0%.

**Category exposure (Fair):** Educational ~25%, Regional ~19%, News/Analysis ~20%, Entertainment ~20% — all categories represented.

**Language exposure (Biased):** 100% English. All non-English content is completely invisible.

**Language exposure (Fair):** English ~40%, Spanish ~16%, Hindi ~16%, French ~12%, Arabic ~9%, Portuguese ~5%, Korean ~3%.

---

## Installation & Usage

### Prerequisites

- Python 3.9 or higher

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the CLI Pipeline

Generates data, trains both models, prints the full audit report, and saves all six plots:

```bash
python bias_recommender/main.py
```

Plots are saved to `plots/bias_study/`.

### Launch the Web App

```bash
python bias_recommender/app.py
```

Opens automatically at `http://localhost:7860`. Start with **Tab 1 → Run Pipeline**, then explore the other tabs.

---

## References

- Zehlike, M., Bonchi, F., Castillo, C., Hajian, S., Megahed, M., & Baeza-Yates, R. (2017). **FA\*IR: A Fair Top-k Ranking Algorithm**. *ACM CIKM 2017*. [doi:10.1145/3132847.3132938](https://doi.org/10.1145/3132847.3132938)

- Ekstrand, M. D., Tian, M., Azpiazu, I. M., Ekstrand, J. D., Anuyah, O., McNeill, D., & Pera, M. S. (2018). **All The Cool Kids, How Do They Fit In?: Popularity and Demographic Biases in Recommender Evaluation and Effectiveness**. *FAT* 2018*.

- Abdollahpouri, H., Burke, R., & Mobasher, B. (2017). **Controlling Popularity Bias in Learning-to-Rank Recommendation**. *RecSys 2017*.

- Noble, S. U. (2018). **Algorithms of Oppression: How Search Engines Reinforce Racism**. New York University Press.
