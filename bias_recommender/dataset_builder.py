"""
dataset_builder.py
──────────────────
Combines all 10 country YouTube CSVs into one master dataset.
Maps YouTube category IDs -> genre labels (8 genres).
Deduplicates, stratified-samples top-N per genre+language bucket,
computes virality scores, embeds titles, shuffles, saves.

Outputs:
  data/master_dataset.csv   -- master video catalog (~15K videos)
  data/embeddings.npy       -- sentence-transformer vectors (aligned by row index)

Run once:  python dataset_builder.py
"""

import os
import json
import numpy as np
import pandas as pd
import kagglehub
from sentence_transformers import SentenceTransformer

KAGGLE_DATASET = "datasnaek/youtube-new"

COUNTRY_LANGUAGE = {
    "US": "English", "CA": "English", "GB": "English",
    "IN": "Hindi",   "MX": "Spanish", "FR": "French",
    "DE": "German",  "JP": "Japanese","KR": "Korean", "RU": "Russian",
}

# YouTube numeric category_id → project genre
YOUTUBE_GENRE_MAP = {
    27: "Educational", 28: "Educational",
    25: "News/Analysis",
    29: "Documentary",  35: "Documentary",
    26: "DIY",
    10: "Music",
    20: "Gaming",
    24: "Entertainment", 23: "Entertainment", 22: "Entertainment",
     1: "Entertainment", 17: "Entertainment", 15: "Entertainment",
    19: "Entertainment", 21: "Entertainment",  2: "Entertainment",
}

# Non-English videos in these lifestyle/vlog YouTube categories → "Regional"
REGIONAL_YT_CATS = {22, 21, 19, 17, 15, 1, 2}

SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

# Max videos kept per (genre, language) bucket after deduplication.
# 8 genres x 8 languages x 300 = ~19K max; sparse buckets will have fewer.
CAP_PER_BUCKET = 300


def _map_genre(cat_id, language):
    base = YOUTUBE_GENRE_MAP.get(int(cat_id), "Entertainment")
    if language != "English" and int(cat_id) in REGIONAL_YT_CATS:
        return "Regional"
    return base


def _load_country(path, country, language):
    fpath = os.path.join(path, f"{country}videos.csv")
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_csv(fpath, encoding="latin-1")

    # Drop removed/error videos
    if "video_error_or_removed" in df.columns:
        df = df[~df["video_error_or_removed"].astype(str).str.lower().isin(["true", "1"])]

    df = df.dropna(subset=["video_id", "title", "category_id", "views", "likes"])
    df["views"] = pd.to_numeric(df["views"], errors="coerce").fillna(0).astype(int)
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce").fillna(0).astype(int)

    # Keep peak-views row per video (same video trends on multiple days)
    df = df.sort_values("views", ascending=False).drop_duplicates("video_id")
    df["language"] = language
    return df[["video_id", "title", "channel_title", "category_id", "views", "likes", "language"]]


def build(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)

    print("Fetching YouTube dataset via kagglehub...")
    kaggle_path = kagglehub.dataset_download(KAGGLE_DATASET)

    # ── Load all countries ─────────────────────────────────────────────────
    frames = []
    for country, lang in COUNTRY_LANGUAGE.items():
        chunk = _load_country(kaggle_path, country, lang)
        if not chunk.empty:
            frames.append(chunk)
            print(f"  {country}: {len(chunk):,} unique videos")

    combined = pd.concat(frames, ignore_index=True)

    # For English countries (US/CA/GB) same video_id = same video → keep once
    english    = combined[combined["language"] == "English"]
    non_english = combined[combined["language"] != "English"]
    english_dedup = english.sort_values("views", ascending=False).drop_duplicates("video_id")
    master = pd.concat([english_dedup, non_english], ignore_index=True)

    # Global dedup: same video_id can appear in multiple non-English country datasets
    # (e.g., a viral video trending in both DE and RU). Keep highest-views row.
    before = len(master)
    master = master.sort_values("views", ascending=False).drop_duplicates("video_id").reset_index(drop=True)
    after  = len(master)
    if before != after:
        print(f"  Global dedup: removed {before - after:,} duplicate video_id rows "
              f"({before:,} -> {after:,})")

    # ── Map genre ──────────────────────────────────────────────────────────
    master["genre"] = master.apply(
        lambda r: _map_genre(r["category_id"], r["language"]), axis=1
    )

    # ── Stratified cap: top-N by views per (genre, language) bucket ────────
    print(f"\nStratified sampling: top {CAP_PER_BUCKET} per genre+language bucket...")
    master = (
        master
        .sort_values("views", ascending=False)
        .groupby(["genre", "language"], group_keys=False)
        .head(CAP_PER_BUCKET)
        .reset_index(drop=True)
    )
    print(f"  Dataset reduced to {len(master):,} videos")

    # ── Virality score (composite, normalised 0-1) ─────────────────────────
    master["engagement_rate"] = (master["likes"] / master["views"].clip(lower=1)).round(4)
    for col in ["views", "likes", "engagement_rate"]:
        max_v = master[col].max()
        master[f"{col}_n"] = (master[col] / max_v).round(6) if max_v > 0 else 0.0
    master["virality_score"] = (
        0.5 * master["views_n"] +
        0.3 * master["likes_n"] +
        0.2 * master["engagement_rate_n"]
    ).round(6)
    master = master.drop(columns=["views_n", "likes_n", "engagement_rate_n", "category_id"])

    # ── Metadata flags ─────────────────────────────────────────────────────
    master["is_english"]    = master["language"] == "English"
    master["is_suppressed"] = master["genre"].isin(SUPPRESSED_GENRES)

    # ── Shuffle ────────────────────────────────────────────────────────────
    master = master.sample(frac=1, random_state=42).reset_index(drop=True)
    master.index.name = "idx"

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, "master_dataset.csv")
    master.to_csv(csv_path, index=True)

    print(f"\nMaster dataset: {len(master):,} videos -> {csv_path}")
    print("\nGenre distribution:")
    for genre, cnt in master["genre"].value_counts().items():
        flag = "  [suppressed]" if genre in SUPPRESSED_GENRES else ""
        print(f"  {genre:<18} {cnt:>6,}  ({cnt/len(master)*100:.1f}%){flag}")

    print("\nLanguage distribution:")
    for lang, cnt in master["language"].value_counts().items():
        print(f"  {lang:<12} {cnt:>6,}  ({cnt/len(master)*100:.1f}%)")

    # ── Compute & save embeddings ──────────────────────────────────────────
    emb_path = os.path.join(output_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        print(f"\nEmbeddings already exist at {emb_path}, skipping.")
    else:
        print(f"\nComputing sentence-transformer embeddings for {len(master):,} videos...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = (
            master["title"].fillna("").astype(str)
            + " [" + master["genre"] + "]"
            + " [" + master["language"] + "]"
        ).tolist()
        embeddings = model.encode(texts, batch_size=512, show_progress_bar=True,
                                  convert_to_numpy=True)
        np.save(emb_path, embeddings)
        print(f"Embeddings saved: {embeddings.shape} -> {emb_path}")

    return master


if __name__ == "__main__":
    build()
