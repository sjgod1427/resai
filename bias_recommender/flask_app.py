"""
flask_app.py
────────────
Flask backend for the Bias-Aware Video Recommender.
Serves a single-page app and exposes three JSON endpoints.

Run:  python flask_app.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

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
    "Entertainment": 1.00, "Music": 0.93, "Gaming": 0.87,
    "DIY": 0.76, "News/Analysis": 0.72, "Educational": 0.68,
    "Documentary": 0.64, "Regional": 0.61,
}
NON_ENGLISH_MULT  = 0.78
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

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


def get_biased_recs(user: dict, top_n: int = 10) -> pd.DataFrame:
    df          = master_df.copy()
    user_vec    = _user_embedding(user)
    sims        = _cos_sim([user_vec], embeddings)[0]
    df["relevance"]      = sims
    df["eng_mult"]       = df["genre"].map(GENRE_ENGAGEMENT).fillna(0.65)
    df["lang_mult"]      = df["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
    bias_raw             = df["virality_score"] * df["eng_mult"] * df["lang_mult"]
    b_max                = bias_raw.max()
    df["bias_component"] = bias_raw / b_max if b_max > 0 else bias_raw
    df["final_score"]    = 0.60 * df["relevance"] + 0.40 * df["bias_component"]
    return df.nlargest(top_n, "final_score").drop(
        columns=["relevance", "eng_mult", "lang_mult", "bias_component", "final_score"]
    )


def get_unbiased_recs(user: dict, top_n: int = 10) -> pd.DataFrame:
    df          = master_df.copy()
    user_vec    = _user_embedding(user)
    sims        = _cos_sim([user_vec], embeddings)[0]
    df["score"] = sims
    return df.nlargest(top_n, "score").drop(columns=["score"])


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


def _metrics(df: pd.DataFrame) -> dict:
    return {
        "suppressed":  int(df["is_suppressed"].sum()),
        "non_english": int((df["language"] != "English").sum()),
        "genres":      df["genre"].value_counts().to_dict(),
        "languages":   df["language"].value_counts().to_dict(),
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
    user_id = (request.get_json(force=True) or {}).get("user_id")
    user    = USER_MAP.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    biased_df   = get_biased_recs(user)
    unbiased_df = get_unbiased_recs(user)
    corrected_df, reasoning, assessment = SUPERVISOR.fix(user, biased_df)

    return jsonify({
        "biased":    _df_to_list(biased_df),
        "corrected": _df_to_list(corrected_df),
        "unbiased":  _df_to_list(unbiased_df),
        "assessment": {
            "is_biased":      assessment.is_biased,
            "reasoning":      assessment.reasoning,
            "genres_missing": assessment.genres_missing,
            "genres_over":    assessment.genres_over_represented,
            "tier1":          assessment.tier1_slots,
            "tier2":          assessment.tier2_slots,
            "tier3":          assessment.tier3_slots,
            "tier4":          assessment.tier4_slots,
        },
        "reasoning": reasoning,
        "metrics": {
            "biased":    _metrics(biased_df),
            "corrected": _metrics(corrected_df),
            "unbiased":  _metrics(unbiased_df),
        },
    })


if __name__ == "__main__":
    print("Starting Flask on http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
