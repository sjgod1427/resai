"""
flask_app.py
────────────
Flask backend for the Bias-Aware Video Recommender.
Serves a single-page app and exposes three JSON endpoints.

Run:  python flask_app.py
"""

import os
import sys
import copy
import json
import uuid
import warnings
import datetime
warnings.filterwarnings("ignore")

# Windows consoles default to cp1252, which can't encode the box-drawing
# characters used in console log output (e.g. _log_run's bar charts).
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
from dotenv import load_dotenv

load_dotenv()

# ── Data ───────────────────────────────────────────────────────────────────────

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MASTER_CSV = os.path.join(DATA_DIR, "master_dataset.csv")
EMBEDDINGS = os.path.join(DATA_DIR, "embeddings.npy")
USERS_JSON = os.path.join(DATA_DIR, "users.json")

print("Loading dataset...")
master_df  = pd.read_csv(MASTER_CSV, index_col="idx")
embeddings = np.load(EMBEDDINGS)

_dup = master_df["video_id"].duplicated(keep="first")
if _dup.any():
    _keep      = (~_dup).values
    master_df  = master_df[_keep].reset_index(drop=True)
    embeddings = embeddings[_keep]
else:
    master_df  = master_df.reset_index(drop=True)

with open(USERS_JSON, "r", encoding="utf-8") as f:
    USERS = json.load(f)
USER_MAP = {u["user_id"]: u for u in USERS}

print("Loading LLM supervisor...")
from llm_supervisor import LLMSupervisor
SUPERVISOR = LLMSupervisor(master_df, embeddings)

# ── Constants ──────────────────────────────────────────────────────────────────

# ── Avatar photos (randomuser.me — real portraits, indexed by persona) ─────────
AVATARS = {
    "U01": "https://randomuser.me/api/portraits/women/44.jpg",   # Priya
    "U02": "https://randomuser.me/api/portraits/men/32.jpg",     # Raj
    "U03": "https://randomuser.me/api/portraits/women/28.jpg",   # Maria
    "U04": "https://randomuser.me/api/portraits/men/67.jpg",     # Ahmed
    "U05": "https://randomuser.me/api/portraits/women/15.jpg",   # Yuki
    "U06": "https://randomuser.me/api/portraits/women/55.jpg",   # Emma
    "U07": "https://randomuser.me/api/portraits/men/41.jpg",     # Carlos
    "U08": "https://randomuser.me/api/portraits/women/63.jpg",   # Fatima
    "U09": "https://randomuser.me/api/portraits/men/22.jpg",     # James
    "U10": "https://randomuser.me/api/portraits/women/38.jpg",   # Ananya
    "U11": "https://randomuser.me/api/portraits/men/71.jpg",     # Lukas
    "U12": "https://randomuser.me/api/portraits/women/47.jpg",   # Sophie
    "U13": "https://randomuser.me/api/portraits/men/18.jpg",     # Marcus
    "U14": "https://randomuser.me/api/portraits/men/25.jpg",     # Kenji
    "U15": "https://randomuser.me/api/portraits/women/72.jpg",   # Amara
    "U16": "https://randomuser.me/api/portraits/men/56.jpg",     # Ivan
    "U17": "https://randomuser.me/api/portraits/men/83.jpg",     # David
    "U18": "https://randomuser.me/api/portraits/women/31.jpg",   # Lin
    "U19": "https://randomuser.me/api/portraits/women/59.jpg",   # Sara
    "U20": "https://randomuser.me/api/portraits/men/12.jpg",     # Tom
}

GENRE_ENGAGEMENT = {
    "Entertainment": 1.00, "Music": 0.95, "Gaming": 0.92,
    "DIY": 0.58, "News/Analysis": 0.52, "Educational": 0.45,
    "Documentary": 0.40, "Regional": 0.33,
}
NON_ENGLISH_MULT  = 0.55
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

GENRE_KEYWORDS = {
    "entertainment": "Entertainment",
    "music":         "Music",
    "gaming":        "Gaming",
    "game":          "Gaming",
    "games":         "Gaming",
    "educational":   "Educational",
    "education":     "Educational",
    "learn":         "Educational",
    "learning":      "Educational",
    "documentary":   "Documentary",
    "documentaries": "Documentary",
    "diy":           "DIY",
    "do it yourself":"DIY",
    "craft":         "DIY",
    "news":          "News/Analysis",
    "analysis":      "News/Analysis",
    "regional":      "Regional",
    "local":         "Regional",
}
_NEG_WORDS = frozenset([
    "less", "fewer", "no", "not", "remove", "stop", "avoid",
    "too many", "too much", "don't", "dont", "dislike", "hate",
])

def _parse_feedback_genres(text: str):
    """Returns (want_more, want_less) — lists of canonical genre names."""
    t = text.lower()
    want_more, want_less = [], []
    for kw, genre in GENRE_KEYWORDS.items():
        if kw not in t:
            continue
        idx     = t.find(kw)
        context = t[max(0, idx - 40): idx + len(kw) + 15]
        is_neg  = any(neg in context for neg in _NEG_WORDS)
        bucket  = want_less if is_neg else want_more
        if genre not in bucket:
            bucket.append(genre)
    return want_more, want_less


def _update_user_from_feedback(user: dict, feedback_text: str):
    """Mutates in-session user copy: boosts/reduces preferred_genres & injects
    synthetic watch interactions so the user embedding shifts accordingly.
    Returns (genres_boosted, genres_reduced)."""
    want_more, want_less = _parse_feedback_genres(feedback_text)
    if not want_more and not want_less:
        return [], []

    prefs    = user.setdefault("preferred_genres", {})
    history  = user.setdefault("interactions", [])
    seen_ids = {i["video_id"] for i in history}

    for genre in want_more:
        prefs[genre] = round(min(1.0, prefs.get(genre, 0.0) + 0.30), 2)
        pool  = master_df[master_df["genre"] == genre].nlargest(15, "virality_score")
        added = 0
        for _, row in pool.iterrows():
            if added >= 6:
                break
            if row["video_id"] in seen_ids:
                continue
            history.append({
                "video_id": row["video_id"],
                "title":    str(row["title"])[:55],
                "genre":    row["genre"],
                "language": row["language"],
                "rating":   5.0,
            })
            seen_ids.add(row["video_id"])
            added += 1

    for genre in want_less:
        if genre in prefs:
            new_w = round(max(0.0, prefs[genre] - 0.30), 2)
            if new_w == 0.0:
                del prefs[genre]
            else:
                prefs[genre] = new_w

    return want_more, want_less


GENRE_COLORS = {
    "Entertainment": "#e74c3c", "Music": "#e67e22", "Gaming": "#f39c12",
    "Educational":   "#2980b9", "Documentary": "#27ae60", "DIY": "#8e44ad",
    "News/Analysis": "#16a085", "Regional":    "#2c3e50",
}
LANG_FLAG = {
    "English": "🇬🇧", "Hindi": "🇮🇳", "Spanish": "🇪🇸", "French": "🇫🇷",
    "German": "🇩🇪", "Japanese": "🇯🇵", "Korean": "🇰🇷", "Russian": "🇷🇺",
}

# ── Towers ─────────────────────────────────────────────────────────────────────

def _user_embedding(user: dict) -> np.ndarray:
    interactions = user.get("interactions", [])
    if not interactions:
        return np.zeros(embeddings.shape[1])
    vid_to_pos = {vid: i for i, vid in enumerate(master_df["video_id"])}
    vecs, weights = [], []
    for item in interactions:
        pos = vid_to_pos.get(item["video_id"])
        if pos is None:
            continue
        vecs.append(embeddings[pos])
        weights.append(float(item["rating"]))
    if not vecs:
        return np.zeros(embeddings.shape[1])
    vecs    = np.array(vecs)
    weights = np.array(weights) / sum(weights)
    uv      = (vecs * weights[:, np.newaxis]).sum(axis=0)
    norm    = np.linalg.norm(uv)
    return uv / norm if norm > 0 else uv


def _watched_ids(user: dict) -> set:
    return {i["video_id"] for i in user.get("interactions", [])}


def get_biased_recs(user: dict, top_n: int = 30) -> pd.DataFrame:
    watched  = _watched_ids(user)
    df       = master_df[~master_df["video_id"].isin(watched)].copy()
    user_vec = _user_embedding(user)

    if np.linalg.norm(user_vec) == 0:
        # Cold-start: rank within preferred genres by biased virality score only.
        pref_genres          = list(user.get("preferred_genres", {}).keys())
        pool                 = df[df["genre"].isin(pref_genres)].copy() if pref_genres else df
        pool["eng_mult"]     = pool["genre"].map(GENRE_ENGAGEMENT).fillna(0.45)
        pool["lang_mult"]    = pool["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
        pool["final_score"]  = pool["virality_score"] * pool["eng_mult"] * pool["lang_mult"]
        return pool.nlargest(top_n, "final_score").drop(
            columns=["eng_mult", "lang_mult", "final_score"]
        )

    sims                 = _cos_sim([user_vec], embeddings[df.index])[0]
    df["eng_mult"]       = df["genre"].map(GENRE_ENGAGEMENT).fillna(0.45)
    df["lang_mult"]      = df["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
    # Soft penalty: floor at 0.5 so preferred suppressed content still appears
    # (mimics real platforms that demote but never fully suppress user preferences).
    soft_mult            = 0.5 + 0.5 * df["eng_mult"].values * df["lang_mult"].values
    penalized            = sims * soft_mult
    pr_max               = penalized.max()
    df["penalized_rel"]  = penalized / pr_max if pr_max > 0 else penalized
    bias_raw             = df["virality_score"] * df["eng_mult"] * df["lang_mult"]
    b_max                = bias_raw.max()
    df["bias_component"] = bias_raw / b_max if b_max > 0 else bias_raw
    df["final_score"]    = 0.50 * df["penalized_rel"] + 0.50 * df["bias_component"]
    return df.nlargest(top_n, "final_score").drop(
        columns=["eng_mult", "lang_mult", "penalized_rel", "bias_component", "final_score"]
    )


def get_unbiased_recs(user: dict, top_n: int = 30) -> pd.DataFrame:
    watched  = _watched_ids(user)
    df       = master_df[~master_df["video_id"].isin(watched)].copy()
    user_vec = _user_embedding(user)

    genre_cap = max(top_n // 3, 5)   # no single genre > 1/3 of feed

    if np.linalg.norm(user_vec) == 0:
        pref_genres = list(user.get("preferred_genres", {}).keys())
        pool        = df[df["genre"].isin(pref_genres)] if pref_genres else df
        pool        = pool.sort_values("virality_score", ascending=False)
        rows, gcnt  = [], {}
        for _, row in pool.iterrows():
            g = row["genre"]
            if gcnt.get(g, 0) < genre_cap:
                rows.append(row); gcnt[g] = gcnt.get(g, 0) + 1
            if len(rows) >= top_n:
                break
        return pd.DataFrame(rows)

    sims        = _cos_sim([user_vec], embeddings[df.index])[0]
    df["score"] = sims
    df_sorted   = df.sort_values("score", ascending=False)
    rows, gcnt  = [], {}
    for _, row in df_sorted.iterrows():
        g = row["genre"]
        if gcnt.get(g, 0) < genre_cap:
            rows.append(row); gcnt[g] = gcnt.get(g, 0) + 1
        if len(rows) >= top_n:
            break
    return pd.DataFrame(rows).drop(columns=["score"])


def _df_to_list(df: pd.DataFrame) -> list:
    return [
        {
            "video_id":     row["video_id"],
            "title":        str(row["title"])[:65],
            "genre":        row["genre"],
            "language":     row["language"],
            "virality":     round(float(row["virality_score"]), 3),
            "is_suppressed": bool(row["is_suppressed"]),
            "color":        GENRE_COLORS.get(row["genre"], "#95a5a6"),
            "flag":         LANG_FLAG.get(row["language"], "🌐"),
        }
        for _, row in df.iterrows()
    ]


_TOP_VIRAL_IDS = None

def _get_top_viral_ids() -> set:
    global _TOP_VIRAL_IDS
    if _TOP_VIRAL_IDS is None:
        _TOP_VIRAL_IDS = set(master_df.nlargest(100, "virality_score")["video_id"])
    return _TOP_VIRAL_IDS


def _tier_counts(user: dict, df: pd.DataFrame) -> dict:
    pref   = {g.strip() for g in user.get("preferred_genres", {}).keys()}
    supp   = {g.strip() for g in SUPPRESSED_GENRES}
    genres = df["genre"].str.strip()
    # T1/T2/T3 are mutually exclusive and always sum to 30.
    # T4 (viral overlay) may overlap with T1–T3 and is shown separately.
    t1 = int(genres.isin(pref).sum())
    t2 = int((genres.isin(supp) & ~genres.isin(pref)).sum())
    t3 = int((~genres.isin(pref) & ~genres.isin(supp)).sum())
    t4 = int(df["video_id"].isin(_get_top_viral_ids()).sum())
    return {"t1": t1, "t2": t2, "t3": t3, "t4": t4}


def _fairness_scores(df: pd.DataFrame, user: dict = None) -> dict:
    """Compute 5 fairness scores (0–100, higher = fairer) for one rec list."""
    n = len(df)
    if n == 0:
        return {"overall": 0.0, "diversity": 0.0, "suppressed_coverage": 0.0,
                "representation": 0.0, "language_diversity": 0.0}

    genres = df["genre"].str.strip()

    # Genre Diversity — normalized Shannon entropy across all genres
    gc    = genres.value_counts().values.astype(float)
    probs = gc / gc.sum()
    entropy  = float(-np.sum(probs * np.log2(probs + 1e-12)))
    n_genres = master_df["genre"].nunique()
    diversity = round(min(100.0, entropy / np.log2(max(n_genres, 2)) * 100), 1)

    # Suppressed Coverage — user-aware: if user has preferred suppressed genres,
    # measure recall of those specific genres; otherwise fall back to library rate.
    user_pref_supp = set()
    if user:
        pref = set(user.get("preferred_genres", {}).keys())
        user_pref_supp = pref & SUPPRESSED_GENRES
    if user_pref_supp:
        present = {g for g in genres.unique() if g in user_pref_supp}
        supp_cov = round(len(present) / len(user_pref_supp) * 100, 1)
    else:
        lib_supp = float(master_df["genre"].isin(SUPPRESSED_GENRES).mean())
        rec_supp = float(genres.isin(SUPPRESSED_GENRES).mean())
        supp_cov = round(min(100.0, rec_supp / max(lib_supp, 0.01) * 100), 1)

    # Representation — 1 minus Gini coefficient.
    # Compute over ALL library genres (fill 0 for absent ones) so a single-genre
    # feed scores near 0, not 100 (Gini of a single non-zero value is always 0).
    all_genres  = master_df["genre"].unique()
    full_counts = genres.value_counts().reindex(all_genres, fill_value=0).values.astype(float)
    counts   = np.sort(full_counts)
    m        = len(counts)
    idx      = np.arange(1, m + 1)
    gini_val = max(0.0, float((2 * (idx * counts).sum()) / (m * counts.sum()) - (m + 1) / m))
    representation = round((1.0 - gini_val) * 100, 1)

    # Language Diversity — normalized Shannon entropy across languages
    lc       = df["language"].value_counts().values.astype(float)
    lp       = lc / lc.sum()
    l_ent    = float(-np.sum(lp * np.log2(lp + 1e-12)))
    n_langs  = master_df["language"].nunique()
    lang_div = round(min(100.0, l_ent / np.log2(max(n_langs, 2)) * 100), 1)

    # Overall — weighted composite
    overall = round(
        0.30 * diversity + 0.30 * representation +
        0.25 * supp_cov  + 0.15 * lang_div,
        1,
    )
    return {
        "overall":             overall,
        "diversity":           diversity,
        "suppressed_coverage": supp_cov,
        "representation":      representation,
        "language_diversity":  lang_div,
    }


def _metrics(user: dict, df: pd.DataFrame) -> dict:
    return {
        "genres":   df["genre"].value_counts().to_dict(),
        "tiers":    _tier_counts(user, df),
        "fairness": _fairness_scores(df, user),
    }


# ── Logging ────────────────────────────────────────────────────────────────────

def _log_run(user: dict, assessment, metrics: dict, reasoning: str):
    a  = assessment
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    bfs = metrics["biased"]["fairness"]
    ffs = metrics["corrected"]["fairness"]
    ufs = metrics["unbiased"]["fairness"]

    _SCORE_KEYS = [
        ("Overall",             "overall"),
        ("Genre Diversity",     "diversity"),
        ("Suppressed Coverage", "suppressed_coverage"),
        ("Representation",      "representation"),
        ("Language Diversity",  "language_diversity"),
    ]

    def score_line(label, key):
        b, f, u = bfs[key], ffs[key], ufs[key]
        bar_b = "█" * int(b / 10) + "░" * (10 - int(b / 10))
        bar_f = "█" * int(f / 10) + "░" * (10 - int(f / 10))
        gain  = f - b
        return (f"  {label:<22}  {b:5.1f} [{bar_b}]  "
                f"{f:5.1f} [{bar_f}]  "
                f"Ideal {u:5.1f}  Gain {gain:+.1f}")

    lines = [
        "=" * 80,
        f"RUN  {ts}  |  User: {user['user_id']} — {user['name']}",
        "=" * 80,
        f"VERDICT      : {'BIASED' if a.is_biased else 'NEEDS FIXING' if a.needs_fixing else 'FAIR'}",
        f"REASONING    : {a.reasoning}",
        f"OVER-REP     : {a.genres_over_represented}",
        f"MISSING      : {a.genres_missing}",
        "",
        "FAIRNESS SCORES  (0–100, higher = fairer)  [██░░] = bar chart",
        f"  {'Metric':<22}  {'Biased':>5}              {'Corrected':>5}              {'Ideal':>8}  {'Gain':>6}",
        "  " + "─" * 74,
        *[score_line(lbl, key) for lbl, key in _SCORE_KEYS],
        "",
        "GENRE MIX",
        f"  Biased    : {metrics['biased']['genres']}",
        f"  Corrected : {metrics['corrected']['genres']}",
        f"  Ideal     : {metrics['unbiased']['genres']}",
        "",
        f"CORRECTION REASONING: {reasoning}",
        "",
    ]

    print("\n".join(lines), flush=True)


# ── Session store (in-memory, per-process) ─────────────────────────────────────
# keyed by session UUID; value: {user, feedback_history}
SESSIONS: dict = {}


def _user_profile_dict(user: dict) -> dict:
    return {
        "user_id":             user.get("user_id", "GUEST"),
        "name":                user.get("name", "New User"),
        "description":         user.get("description", ""),
        "avatar":              user.get("avatar", ""),
        "preferred_genres":    user["preferred_genres"],
        "preferred_languages": user.get("preferred_languages", ["English"]),
        "interactions": [
            {
                "title":    str(i["title"])[:55],
                "genre":    i["genre"],
                "language": i["language"],
                "color":    GENRE_COLORS.get(i["genre"], "#95a5a6"),
                "flag":     LANG_FLAG.get(i["language"], "🌐"),
            }
            for i in user["interactions"][-8:]
        ],
        "total_interactions": len(user["interactions"]),
    }


def _build_guest_user(prefs: dict) -> dict:
    genres = {g: float(w) for g, w in prefs.get("genres", {}).items() if float(w) > 0}
    languages = prefs.get("languages", ["English"]) or ["English"]
    name = str(prefs.get("name", "New User"))[:40].strip() or "New User"
    top_genres = list(genres.keys())[:3]
    return {
        "user_id": "GUEST",
        "name": name,
        "description": (
            f"New user exploring {', '.join(top_genres)}" if top_genres
            else "New user with no genre preferences yet"
        ),
        "preferred_genres": genres,
        "preferred_languages": languages,
        "interactions": [],
    }


# ── Flask ──────────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    users = [
        {
            "user_id":     u["user_id"],
            "name":        u["name"],
            "description": u["description"][:75],
            "top_genre":   max(u["preferred_genres"], key=u["preferred_genres"].get),
            "languages":   u["preferred_languages"],
            "avatar":      AVATARS.get(u["user_id"], ""),
        }
        for u in USERS
    ]
    return render_template("index.html", users=users, genre_colors=GENRE_COLORS)


@app.route("/api/user/<user_id>")
def get_user(user_id):
    u = USER_MAP.get(user_id)
    if not u:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "user_id":             u["user_id"],
        "name":                u["name"],
        "description":         u["description"],
        "preferred_genres":    u["preferred_genres"],
        "preferred_languages": u["preferred_languages"],
        "avatar":              AVATARS.get(u["user_id"], ""),
        "interactions": [
            {
                "title":    str(i["title"])[:55],
                "genre":    i["genre"],
                "language": i["language"],
                "rating":   i["rating"],
                "color":    GENRE_COLORS.get(i["genre"], "#95a5a6"),
                "flag":     LANG_FLAG.get(i["language"], "🌐"),
            }
            for i in u["interactions"][:8]
        ],
        "total_interactions": len(u["interactions"]),
    })


@app.route("/api/run", methods=["POST"])
def run():
    data        = request.get_json(force=True) or {}
    user_id     = data.get("user_id")
    new_user    = data.get("new_user")   # cold-start guest preferences
    session_id  = data.get("session_id")

    if session_id and session_id in SESSIONS:
        # Re-run using existing session (feedback mutations already applied)
        user = SESSIONS[session_id]["user"]
        feedback_history = SESSIONS[session_id]["feedback_history"]
    else:
        if new_user:
            user = _build_guest_user(new_user)
        elif user_id:
            base = USER_MAP.get(user_id)
            if not base:
                return jsonify({"error": "User not found"}), 404
            user = copy.deepcopy(base)   # isolate session mutations from global state
        else:
            return jsonify({"error": "No user specified"}), 400

        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = {"user": user, "feedback_history": []}
        feedback_history = []

    try:
        biased_df   = get_biased_recs(user)
        unbiased_df = get_unbiased_recs(user)
        corrected_df, reasoning, assessment = SUPERVISOR.fix(user, biased_df)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500

    run_metrics = {
        "biased":    _metrics(user, biased_df),
        "corrected": _metrics(user, corrected_df),
        "unbiased":  _metrics(user, unbiased_df),
    }
    _log_run(user, assessment, run_metrics, reasoning)

    return jsonify({
        "biased":    _df_to_list(biased_df),
        "corrected": _df_to_list(corrected_df),
        "unbiased":  _df_to_list(unbiased_df),
        "assessment": {
            "is_biased":      assessment.is_biased,
            "needs_fixing":   assessment.needs_fixing,
            "reasoning":      assessment.reasoning,
            "genres_missing": assessment.genres_missing,
            "genres_over":    assessment.genres_over_represented,
            "tier1":          assessment.tier1_slots,
            "tier2":          assessment.tier2_slots,
            "tier3":          assessment.tier3_slots,
            "tier4":          assessment.tier4_slots,
        },
        "reasoning":        reasoning,
        "metrics":          run_metrics,
        "session_id":       session_id,
        "feedback_history": feedback_history,
        "user_profile":     _user_profile_dict(user),
    })


@app.route("/api/feedback", methods=["POST"])
def feedback():
    data          = request.get_json(force=True) or {}
    session_id    = data.get("session_id", "")
    feedback_text = str(data.get("feedback", "")).strip()

    if session_id not in SESSIONS:
        return jsonify({"error": "Session expired — please re-run the analysis first."}), 400
    if not feedback_text:
        return jsonify({"error": "Feedback cannot be empty."}), 400

    session = SESSIONS[session_id]
    user    = session["user"]
    session["feedback_history"].append(feedback_text)

    # Update interaction history + preference weights for ALL users.
    # The session user is already a deep-copy, so this never mutates global state.
    genres_boosted, genres_reduced = _update_user_from_feedback(user, feedback_text)

    try:
        biased_df   = get_biased_recs(user)
        unbiased_df = get_unbiased_recs(user)
        corrected_df, reasoning, assessment = SUPERVISOR.fix(
            user, biased_df, session["feedback_history"]
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        session["feedback_history"].pop()
        return jsonify({"error": str(exc)}), 500

    run_metrics = {
        "biased":    _metrics(user, biased_df),
        "corrected": _metrics(user, corrected_df),
        "unbiased":  _metrics(user, unbiased_df),
    }
    _log_run(user, assessment, run_metrics, reasoning)

    updated_profile = {
        "preferred_genres": user["preferred_genres"],
        "genres_boosted":   genres_boosted,
        "genres_reduced":   genres_reduced,
        "recent_interactions": [
            {
                "title":    i["title"],
                "genre":    i["genre"],
                "language": i["language"],
                "color":    GENRE_COLORS.get(i["genre"], "#95a5a6"),
                "flag":     LANG_FLAG.get(i["language"], "🌐"),
            }
            for i in user["interactions"][-8:]
        ],
        "total_interactions": len(user["interactions"]),
    }

    return jsonify({
        "biased":    _df_to_list(biased_df),
        "corrected": _df_to_list(corrected_df),
        "unbiased":  _df_to_list(unbiased_df),
        "assessment": {
            "is_biased":      assessment.is_biased,
            "needs_fixing":   assessment.needs_fixing,
            "reasoning":      assessment.reasoning,
            "genres_missing": assessment.genres_missing,
            "genres_over":    assessment.genres_over_represented,
            "tier1":          assessment.tier1_slots,
            "tier2":          assessment.tier2_slots,
            "tier3":          assessment.tier3_slots,
            "tier4":          assessment.tier4_slots,
        },
        "reasoning":        reasoning,
        "metrics":          run_metrics,
        "session_id":       session_id,
        "feedback_history": session["feedback_history"],
        "updated_profile":  updated_profile,
        "user_profile":     _user_profile_dict(user),
    })


if __name__ == "__main__":
    print("Starting Flask on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)