# Responsible AI: Bias-Aware YouTube Recommender

A comprehensive **Responsible AI demonstration** that injects, detects, and corrects algorithmic bias in a YouTube-style video recommendation system. Built on real YouTube trending data, embedding-based retrieval, and an agentic LLM supervisor.

**Status:** ✅ All systems functional and tested  
**Latest Test:** April 21, 2026

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Matters](#-why-this-matters)
3. [Quick Start](#-quick-start)
4. [Project Structure](#-project-structure)
5. [File Guide](#-file-guide)
6. [System Architecture](#-system-architecture)
7. [Installation & Configuration](#-installation--configuration)
8. [Running the Application](#-running-the-application)
9. [Testing](#-testing)
10. [Key Concepts](#-key-concepts)
11. [Dependencies](#-dependencies)
12. [Troubleshooting](#-troubleshooting)

---

## 🎯 Project Overview

### What Is This?

This project demonstrates how **algorithmic bias** can emerge in recommendation systems, even without explicit discrimination. It provides:

- **Bias Injection**: A YouTube recommender that systematically suppresses educational, documentary, DIY, news, and regional-language content
- **Bias Detection**: An LLM-based supervisor that identifies unfair recommendations using structured reasoning
- **Bias Correction**: Real-time correction via fairness-aware re-ranking and interactive feedback refinement
- **Measurement**: Five quantitative fairness metrics (Gini, entropy, demographic parity, diversity, representation)

### Key Features

| Feature | Description |
|---------|-------------|
| **Real Data** | ~13K YouTube videos from 10 countries, 8 genres, 8 languages |
| **Two-Tower Retrieval** | User+item embedding-based scoring with engagement multipliers |
| **LLM Supervisor** | Agentic loop (Groq llama-3.3-70b) for bias judgment and correction |
| **Interactive UI** | Flask web interface; 20 user personas + guest user builder |
| **Fairness Metrics** | Gini coefficient, entropy, demographic parity gap, diversity scores |
| **Multi-Round Feedback** | Users refine corrections interactively; system re-ranks each round |
| **Offline Study Mode** | `main.py` runs a full batch bias study with SVD-based recommender |

---

## ⚠️ Why This Matters

> _"If a platform recommends what users click on, and users click on what they've been shown, who decides what gets shown first?"_

### The Problem

Real platforms (YouTube, TikTok, Netflix) optimize for **engagement signals** (clicks, watch time, likes). High-engagement content (Entertainment, Gaming, Music) gets amplified. Lower-engagement content (Education, Documentary, Regional languages) gets de-ranked—not through explicit suppression, but through compound engagement weights.

**Result:** Over time, certain user demographics and content categories are systematically underrepresented, even though no explicit rule forbids them.

### The Solution

This project makes that mechanism **explicit, measurable, and correctable**:

1. **Inject** documented bias multipliers
2. **Measure** how unfair the system becomes
3. **Correct** using fairness-aware re-ranking + LLM reasoning
4. **Quantify** improvement with 5 fairness metrics

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Free Groq API key: https://console.groq.com
- (Optional) Kaggle API credentials for first-time data download

### Installation

```bash
cd bias_recommender
pip install -r requirements.txt
```

### Environment Setup

Create `.env` inside `bias_recommender/`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### Run the Web UI

```bash
python flask_app.py
```

Then open: **http://localhost:5000**

### Or Run the Offline Study

```bash
python main.py
```

(Generates bias metrics + plots for a synthetic data study using SVD-based recommender)

---

## 📁 Project Structure

```
resai/
├── README.md                           # This file
├── Global YouTube Statistics.csv       # Source data (Kaggle dataset)
├── youtube_eda_only.py                 # Standalone EDA script for raw YouTube data
│
└── bias_recommender/                   # Main application folder
    ├── flask_app.py                    # **MAIN APP** — Flask web server with API
    ├── llm_supervisor.py               # **CORE** — LLM-based bias detection/correction
    ├── recommender.py                  # SVD collaborative filter (offline study)
    ├── bias_metrics.py                 # Fairness measurement functions
    ├── main.py                         # Offline batch bias study pipeline
    │
    ├── dataset_builder.py              # Kaggle → master_dataset.csv + embeddings
    ├── data_generator.py               # Synthetic data generator (for main.py)
    ├── compute_embeddings.py           # Re-compute SentenceTransformer embeddings
    ├── youtube_loader.py               # Kaggle API downloader utility
    ├── user_profiles.py                # Generate 20 persona profiles → users.json
    ├── visualize.py                    # Plot generation for offline study
    │
    ├── app.py                          # Legacy Gradio UI (superseded by flask_app.py)
    ├── test_architecture.py            # Unit tests (no LLM, ~15s)
    ├── test_llm_correction.py          # Bias correction tests (requires GROQ_API_KEY)
    │
    ├── templates/
    │   └── index.html                  # Single-page web UI
    │
    ├── requirements.txt                # All Python dependencies
    ├── APPROACH.md                     # Technical deep dive
    ├── README.md                       # Detailed project documentation
    │
    └── data/                           # Auto-generated on first run
        ├── master_dataset.csv          # Deduplicated videos with metadata
        ├── embeddings.npy              # (N, 384) SentenceTransformer vectors
        ├── users.json                  # 20 persona profiles
        ├── videos.csv                  # Cleaned video metadata
        ├── users.csv                   # User interaction history
        ├── interactions.csv            # User-video interaction log
        └── creators.csv                # Creator metadata
```

---

## 📖 File Guide

### **Critical Files** (Read These First)

#### `flask_app.py` — Main Application Server
**Purpose:** Flask REST API + session management for the web UI  
**What it does:**
- Loads master dataset + embeddings (startup: ~2s)
- Initializes LLMSupervisor (startup: ~30s)
- Serves `/api/run` — accepts user profile, returns biased + corrected recommendations
- Serves `/api/feedback` — multi-round refinement loop
- Renders `templates/index.html`

**Key functions:**
- `get_biased_recs()` — Two-tower recommendation with bias multipliers
- `post('/api/run')` — Main entry point; calls LLMSupervisor.fix()
- `post('/api/feedback')` — Refinement round; re-ranks based on user feedback

**Dependencies:** flask, numpy, pandas, LLMSupervisor

---

#### `llm_supervisor.py` — Bias Detection & Correction Engine
**Purpose:** Agentic loop that judges and corrects biased recommendations  
**What it does:**
- **Judge Phase:** Structured LLM call to detect bias + assign tier constraints
- **Pre-fetch Phase:** Python optimization to find ~50 candidate videos meeting tiers
- **Agent Phase:** LLM selects final 30 videos from candidates, enforcing fairness
- **Fallback strategies:** Structured parsing, then rule-based tier fill if LLM fails

**Key classes:**
- `LLMSupervisor` — Main class; initialized once at app startup
  - `fix(user_prefs, biased_recs)` → BiasAssessment + corrected list
  - Pre-fetches candidates using pure Python (~0.05s)
  - Calls Groq LLM for judgment (~1.5s)

**Dependencies:** langchain-groq, pydantic, numpy, pandas

---

#### `templates/index.html` — Web UI
**Purpose:** Single-page React-like interface  
**Features:**
- Select user persona or build custom guest profile
- Run bias analysis; get biased + corrected recommendations side-by-side
- View fairness scores (0–100 metrics)
- Multi-round feedback loop to refine corrections
- Compare three feeds: biased, LLM-corrected, and cumulative feedback-refined

**No backend build needed** — pure vanilla JavaScript + CSS

---

### **Data Pipeline Files**

#### `dataset_builder.py`
**Purpose:** Download and process real YouTube data → `data/master_dataset.csv`  
**Inputs:**
- Kaggle dataset: `datasnaek/youtube-new` (free, requires Kaggle credentials)
- `Global YouTube Statistics.csv` (included in repo)

**Outputs:**
- `data/master_dataset.csv` — ~13K deduplicated videos with columns:
  - `video_id`, `title`, `channel`, `genre`, `language`, `virality_score`, `engagement_mult`, `language_mult`
- `data/embeddings.npy` — (N, 384) float32 vectors from SentenceTransformer

**When to run:**
- First time setup: `python dataset_builder.py`
- To refresh data: Delete `data/master_dataset.csv` + `data/embeddings.npy`, then run again

---

#### `user_profiles.py`
**Purpose:** Generate 20 pre-built user personas → `data/users.json`  
**Personas include:**
- Students (interested in education)
- Professionals (interested in news, learning)
- Entertainment fans (movies, gaming, music)
- Global audience (non-English speakers)

**When to run:**
- First time setup: `python user_profiles.py`
- To customize: Edit the file or rebuild from scratch

---

#### `compute_embeddings.py`
**Purpose:** Re-compute embeddings for all videos using SentenceTransformer  
**When to run:**
- If you add new videos to `master_dataset.csv`
- If you want to switch embedding models

**Usage:**
```bash
python compute_embeddings.py  # Re-embeds all videos from titles
```

---

### **Recommendation & Metrics Files**

#### `bias_metrics.py`
**Purpose:** Fairness measurement functions  
**Key metrics:**
- **Gini coefficient** — Lower = more fair (0–1)
- **Entropy (diversity)** — Higher = better (0–1)
- **Language diversity** — % of videos in non-English (0–1)
- **Demographic parity gap** — Gender/language representation fairness (0–1)
- **Suppressed content rate** — % coverage of low-engagement categories

**Functions:**
- `full_report(recs_df, videos_df)` — Compute all metrics for a recommendation set

---

#### `recommender.py`
**Purpose:** Two recommendation engines for the offline study (`main.py`)  
**Classes:**
- `BiasedRecommender` — SVD model with bias multipliers
- `FairRecommender` — SVD model + fairness-aware re-ranking

**Used by:** `main.py` (offline study only)  
**Not used by:** `flask_app.py` (which uses two-tower embeddings instead)

---

### **Offline Study Files**

#### `main.py` — Batch Bias Study Pipeline
**Purpose:** Run a full offline study comparing biased vs. fair SVD-based recommenders  
**Workflow:**
1. Generate synthetic data (users, videos, interactions)
2. Train BiasedRecommender + FairRecommender
3. Get recommendations for all users
4. Measure bias metrics for both
5. Generate comparison plots

**Output:** `../plots/bias_study/` (6 PNG plots)

**When to run:**
```bash
python main.py
```

---

#### `data_generator.py`
**Purpose:** Generate synthetic YouTube-like data for offline study  
**Generates:**
- 500 synthetic videos with: genre, language, virality, title
- 100 synthetic users with: profile type, language preferences, interaction history
- ~3000 interaction records (user-video ratings)

**Used by:** `main.py` only

---

#### `visualize.py`
**Purpose:** Generate plots for offline study results  
**Plots:**
- Bias comparison (Gini, entropy, etc.)
- Genre distribution bias
- Language representation
- Sample user recommendations (biased vs. corrected)

**Output:** `../plots/bias_study/`

---

### **Testing Files**

#### `test_architecture.py`
**Purpose:** Verify data integrity + bias injection logic (no LLM calls)  
**Tests:**
- Data files exist and are loadable
- Embeddings align with videos
- Bias multipliers apply correctly
- Metrics compute without errors

**Runtime:** ~15 seconds  
**Run:** `python test_architecture.py`

---

#### `test_llm_correction.py`
**Purpose:** Test LLM-based bias detection + correction quality  
**Tests:**
- LLMSupervisor correctly identifies biased recommendations
- Corrected lists improve fairness scores
- Multi-round feedback refines corrections

**Runtime:** ~30 seconds (depends on LLM latency)  
**Requirements:** Valid `GROQ_API_KEY`  
**Run:** `python test_llm_correction.py`

---

### **Utility Files**

#### `youtube_loader.py`
**Purpose:** Download raw YouTube data from Kaggle  
**When to use:** Only called internally by `dataset_builder.py`

---

#### `app.py` — Legacy Gradio UI
**Status:** ⚠️ Superseded by `flask_app.py`  
**Why kept:** Historical reference; demonstrates Gradio-based alternative UI  
**You probably don't need this.**

---

#### `APPROACH.md`
**Purpose:** Technical deep dive into bias injection, two-tower model, LLM corrections  
**For:** Researchers who want to understand the math + implementation

---

---

## 🏗️ System Architecture

### High-Level Flow

```
User Input (Profile + Preferences)
         ↓
   get_biased_recs()
   (Two-Tower Retrieval + Bias Multipliers)
         ↓
   Biased Recommendations (Top 30)
         ↓
   LLMSupervisor.fix()
   (Judge → Pre-fetch → Agent)
         ↓
   Corrected Recommendations (Top 30)
         ↓
   Fairness Metrics (5 scores: 0–100)
         ↓
   User Sees Three Feeds Side-by-Side
   (Biased | Corrected | Feedback-Refined)
```

### Bias Multipliers

| Category | Multiplier | Penalty |
|----------|-----------|---------|
| Entertainment | 1.40 | **+40%** |
| Music | 1.30 | **+30%** |
| Gaming | 1.20 | **+20%** |
| DIY | 0.50 | **-50%** |
| News/Analysis | 0.40 | **-60%** |
| Documentary | 0.45 | **-55%** |
| Educational | 0.35 | **-65%** |
| Regional | 0.30 | **-70%** |
| Non-English | 0.40 | **-60%** (stacks with above) |

### Fairness Correction Strategy (FA*IR-inspired)

The `LLMSupervisor` enforces **minimum-quota re-ranking**:

- **Educational**: ≥15% of top-30
- **Regional**: ≥10%
- **Documentary**: ≥8%
- **DIY**: ≥8%
- **News/Analysis**: ≥5%
- **Non-English**: ≥25%
- **Remainder**: Highest-scoring videos (unpenalized)

---

## 💾 Installation & Configuration

### Step 1: Install Python Packages

```bash
cd bias_recommender
pip install -r requirements.txt
```

### Step 2: Create Environment File

Create `bias_recommender/.env`:

```env
GROQ_API_KEY=gsk_XXXX...  # Free at https://console.groq.com
```

### Step 3: (Optional) Set Up Kaggle Credentials

Only needed for first-time data download:

```bash
# On Windows/Linux/Mac, create ~/.kaggle/kaggle.json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}

# Make it read-only
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Build Data Files (First Run Only)

```bash
python dataset_builder.py    # Downloads + processes Kaggle data (~2 min)
python user_profiles.py      # Generates 20 personas (~5 sec)
```

These create:
- `data/master_dataset.csv` (~13K rows)
- `data/embeddings.npy` (~50 MB)
- `data/users.json`

---

## ▶️ Running the Application

### **Option A: Web UI (Recommended)**

```bash
python flask_app.py
```

Then open: **http://localhost:5000**

**Features:**
- Pick a persona or create custom user
- Click "Run Analysis"
- View biased vs. corrected recommendations
- Multi-round feedback refinement
- Compare fairness metrics

**Startup Time:** ~30 seconds (initializes LLMSupervisor)

---

### **Option B: Offline Batch Study**

```bash
python main.py
```

**What it does:**
1. Generates synthetic data
2. Trains SVD-based recommenders (biased + fair)
3. Measures bias metrics
4. Generates comparison plots
5. Prints detailed report

**Output:** `../plots/bias_study/` (6 PNG plots)  
**Runtime:** ~1-2 minutes

---

### **Option C: Legacy Gradio UI**

```bash
python app.py
```

(⚠️ Older interface; `flask_app.py` recommended)

---

## 🧪 Testing

### Run Architecture Tests (No LLM)

```bash
python test_architecture.py
```

✅ Verifies:
- Data files exist and load
- Embeddings align with videos
- Bias multipliers apply correctly
- Metrics compute without error

**Runtime:** ~15 seconds

---

### Run LLM Correction Tests

```bash
python test_llm_correction.py
```

✅ Verifies:
- LLMSupervisor initializes
- Bias detection works
- Corrections improve fairness
- Multi-round feedback refines

**Runtime:** ~30 seconds  
**Requires:** Valid `GROQ_API_KEY`

---

## 🔑 Key Concepts

### Two-Tower Retrieval

```
User Tower:
  user_vec = weighted_avg( embeddings[videos_user_watched] )

Item Tower:
  item_vecs = embeddings.npy

Scoring:
  similarity = cosine(user_vec, item_vecs)
  bias_component = virality_score × engagement_mult × language_mult
  final_score = 0.5 × similarity + 0.5 × bias_component
```

### Fairness Metrics (0–100)

- **Gini Coefficient** (0–100): Concentration of recommendations. Lower = more fair.
- **Entropy / Diversity** (0–100): Genre diversity. Higher = better.
- **Language Diversity** (0–100): % non-English videos. Higher = better.
- **Demographic Parity Gap** (0–100): Gender representation fairness.
- **Suppressed Content Rate** (0–100): % coverage of low-engagement categories.

### LLM-Based Correction

1. **Judge Phase:** LLM evaluates the biased list, identifies missing categories
2. **Pre-fetch Phase:** Python finds ~50 candidates meeting fairness tiers
3. **Agent Phase:** LLM selects final 30, respecting tier constraints
4. **Feedback Refinement:** User describes desired changes; LLM re-ranks each round

---

## 📦 Dependencies

All Python packages are listed in `requirements.txt`:

| Package | Version | Purpose |
|---------|---------|---------|
| flask | ≥3.0 | Web server |
| numpy | ≥1.24 | Numerical computing |
| pandas | ≥2.0 | Data manipulation |
| scipy | ≥1.10 | Scientific computing (SVD) |
| matplotlib | ≥3.7 | Plotting (offline study) |
| Pillow | ≥9.0 | Image processing |
| kagglehub | ≥0.3.0 | Download Kaggle datasets |
| langchain | ≥1.0.0 | LLM orchestration |
| langchain-groq | ≥1.0.0 | Groq API integration |
| langchain-core | ≥1.0.0 | LLM interfaces |
| python-dotenv | ≥1.0.0 | Load `.env` file |
| scikit-learn | ≥1.3.0 | SVD + metrics |
| pydantic | ≥2.0.0 | Data validation |
| gradio | ≥6.0 | Legacy UI (optional) |

### External APIs

- **Groq** (Free tier): `llama-3.3-70b-versatile` for LLM-based corrections
- **Kaggle** (Free): `datasnaek/youtube-new` dataset
- **HuggingFace**: `all-MiniLM-L6-v2` sentence embeddings (auto-downloaded)

---

## 🐛 Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'flask'`

**Solution:**
```bash
pip install -r requirements.txt
```

---

### Issue: LLMSupervisor initialization hangs (>60s)

**Likely causes:**
1. First-time Kaggle data download (can take 2+ min)
2. Embedding computation for new data (can take 5+ min)
3. Network latency downloading embeddings from HuggingFace

**Solution:**
- Wait for it to complete (check console output)
- Or manually pre-build data:
  ```bash
  python dataset_builder.py
  ```

---

### Issue: `GROQ_API_KEY not found`

**Solution:**
1. Get free key: https://console.groq.com
2. Create `bias_recommender/.env`:
   ```env
   GROQ_API_KEY=gsk_XXXX...
   ```
3. Restart Flask: `python flask_app.py`

---

### Issue: LLM Corrections Are Slow (~3+ seconds per request)

**Expected behavior.** Groq's free tier has latency. Improvements:
- Use Groq paid tier (lower latency)
- Switch LLM: Edit `llm_supervisor.py`, change `ChatGroq` to another provider

---

### Issue: `Kaggle API credentials not found`

**Only needed for first-run data download.**

**Solution:**
1. Get Kaggle API key: https://www.kaggle.com/settings/account
2. Create `~/.kaggle/kaggle.json`:
   ```json
   {"username": "your_username", "key": "your_api_key"}
   ```
3. Rerun `dataset_builder.py`

---

### Issue: `data/master_dataset.csv` is missing

**Solution:**
```bash
python dataset_builder.py
```

This will:
1. Download from Kaggle (if credentials available)
2. Or use local `Global YouTube Statistics.csv`
3. Process + save to `data/master_dataset.csv`

---

## 📚 Additional Resources

- **APPROACH.md** — Technical deep dive (bias injection, two-tower model, LLM prompts)
- **bias_recommender/README.md** — Older, detailed documentation
- **Kaggle Dataset:** https://www.kaggle.com/datasnaek/youtube-new
- **FA*IR Paper:** https://arxiv.org/abs/1706.06368 (Zehlike et al., 2017)
- **Groq Console:** https://console.groq.com

---

## ✅ Verification Checklist

After setup, verify everything works:

- [ ] `python test_architecture.py` passes (15 sec)
- [ ] `python test_llm_correction.py` passes (30 sec, requires GROQ_API_KEY)
- [ ] `python flask_app.py` starts without errors (~30 sec startup)
- [ ] http://localhost:5000 loads in browser
- [ ] Can select a user persona and click "Run Analysis"
- [ ] Biased + corrected recommendations appear
- [ ] Fairness metrics display (0–100 scores)

**All ✅?** You're good to go!

---

## 📊 Example Output

### Web UI Results

**For Student User:**
- **Biased Recommendations:** Heavy on Entertainment, Gaming, Music; almost no Educational content
- **Corrected Recommendations:** Balanced mix including ≥15% Educational, ≥25% Non-English
- **Fairness Metrics:**
  - Gini (Biased): 0.65 → Corrected: 0.42 ✅
  - Diversity (Biased): 0.31 → Corrected: 0.62 ✅
  - Suppressed Content (Biased): 0.05 → Corrected: 0.38 ✅

---

## 📝 License & Citation

This project is a Responsible AI demonstration for educational purposes.

**Cite as:**
```
Responsible AI: Bias-Aware YouTube Recommender
ResAI Project, 2026
```

---

## 🤝 Contributing

Found a bug or have suggestions?

1. Run tests: `python test_architecture.py`
2. Check `llm_supervisor.py` logs for LLM issues
3. Open an issue with:
   - Output from test script
   - Your `.env` setup (without API keys!)
   - Expected vs. actual behavior

---

## 📞 Support

- **Flask App Issues:** Check `flask_app.py` startup logs
- **LLM Correction Issues:** Check `GROQ_API_KEY` in `.env`
- **Data Missing:** Run `python dataset_builder.py`
- **Tests Failing:** Run `python test_architecture.py` first (checks basics)

---

**Last Updated:** April 21, 2026  
**Status:** ✅ All systems operational

