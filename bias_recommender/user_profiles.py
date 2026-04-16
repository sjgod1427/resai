"""
user_profiles.py
────────────────
Defines 20 fixed user personas and builds their interaction histories
from the master dataset. Each user is a test identity for the demo.

Saves: data/users.json

Run once:  python user_profiles.py
"""

import os
import json
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# ─── 20 User Personas ─────────────────────────────────────────────────────────

PERSONAS = [
    {
        "user_id": "U01", "name": "Priya",
        "description": "Priya is a 22-year-old art and design student. She loves DIY craft tutorials, documentaries about artists and designers, and occasionally music videos. She never watches gaming or mainstream entertainment.",
        "preferred_genres": {"DIY": 0.45, "Documentary": 0.30, "Music": 0.15, "Educational": 0.10},
        "preferred_languages": ["English", "Hindi"],
    },
    {
        "user_id": "U02", "name": "Raj",
        "description": "Raj is a 28-year-old software engineer. He watches coding tutorials, tech explainers, and science documentaries almost exclusively. He is deeply engaged with educational content and follows tech news.",
        "preferred_genres": {"Educational": 0.50, "Documentary": 0.20, "News/Analysis": 0.20, "Gaming": 0.10},
        "preferred_languages": ["English"],
    },
    {
        "user_id": "U03", "name": "Maria",
        "description": "Maria is a 25-year-old lifestyle blogger from Mexico. She watches Spanish-language vlogs, regional entertainment, and Latin music. She prefers Spanish content and rarely watches English channels.",
        "preferred_genres": {"Regional": 0.50, "Music": 0.30, "Entertainment": 0.20},
        "preferred_languages": ["Spanish"],
    },
    {
        "user_id": "U04", "name": "Ahmed",
        "description": "Ahmed is a 35-year-old political journalist. He consumes heavy volumes of news analysis, political documentaries, and investigative journalism. He watches in English and Arabic.",
        "preferred_genres": {"News/Analysis": 0.55, "Documentary": 0.30, "Educational": 0.15},
        "preferred_languages": ["English"],
    },
    {
        "user_id": "U05", "name": "Yuki",
        "description": "Yuki is a 19-year-old student in Tokyo. She watches Japanese gaming streams, K-pop performances, and anime-related content. Her viewing is almost entirely in Japanese and Korean.",
        "preferred_genres": {"Gaming": 0.40, "Music": 0.35, "Regional": 0.25},
        "preferred_languages": ["Japanese", "Korean"],
    },
    {
        "user_id": "U06", "name": "Emma",
        "description": "Emma is a 31-year-old homeowner passionate about home improvement. She watches DIY renovation tutorials, woodworking projects, and gardening guides. She prefers practical, step-by-step content.",
        "preferred_genres": {"DIY": 0.65, "Educational": 0.20, "Documentary": 0.15},
        "preferred_languages": ["English"],
    },
    {
        "user_id": "U07", "name": "Carlos",
        "description": "Carlos is a 27-year-old musician from Spain. He watches Latin music videos, behind-the-scenes artist documentaries, and music theory tutorials. He strongly prefers Spanish-language content.",
        "preferred_genres": {"Music": 0.55, "Documentary": 0.25, "Educational": 0.20},
        "preferred_languages": ["Spanish", "English"],
    },
    {
        "user_id": "U08", "name": "Fatima",
        "description": "Fatima is a 29-year-old researcher from Morocco. She watches science documentaries, educational lectures, and news analysis in Arabic, French, and English.",
        "preferred_genres": {"Documentary": 0.40, "Educational": 0.40, "News/Analysis": 0.20},
        "preferred_languages": ["French", "English"],
    },
    {
        "user_id": "U09", "name": "James",
        "description": "James is a 23-year-old casual viewer. He watches comedy clips, entertainment highlights, and gaming streams. He has no strong niche and follows whatever is trending.",
        "preferred_genres": {"Entertainment": 0.55, "Gaming": 0.30, "Music": 0.15},
        "preferred_languages": ["English"],
    },
    {
        "user_id": "U10", "name": "Ananya",
        "description": "Ananya is a 26-year-old Bollywood fan from Mumbai. She watches Hindi entertainment serials, Bollywood music videos, and celebrity vlogs. Almost all her content is in Hindi.",
        "preferred_genres": {"Regional": 0.45, "Music": 0.35, "Entertainment": 0.20},
        "preferred_languages": ["Hindi"],
    },
    {
        "user_id": "U11", "name": "Lukas",
        "description": "Lukas is a 33-year-old sports coach from Germany. He watches German football analysis, sports fitness tutorials, and sports documentaries in German.",
        "preferred_genres": {"Regional": 0.40, "Educational": 0.30, "Documentary": 0.30},
        "preferred_languages": ["German"],
    },
    {
        "user_id": "U12", "name": "Sophie",
        "description": "Sophie is a 24-year-old philosophy student from Paris. She watches French educational videos, science documentaries, and intellectual debates. She strongly prefers slow, thoughtful content.",
        "preferred_genres": {"Educational": 0.45, "Documentary": 0.35, "News/Analysis": 0.20},
        "preferred_languages": ["French", "English"],
    },
    {
        "user_id": "U13", "name": "Marcus",
        "description": "Marcus is a 21-year-old esports player. He watches competitive gaming tournaments, game reviews, and streamer highlights almost exclusively. He is fully embedded in gaming culture.",
        "preferred_genres": {"Gaming": 0.80, "Entertainment": 0.20},
        "preferred_languages": ["English"],
    },
    {
        "user_id": "U14", "name": "Kenji",
        "description": "Kenji is a 20-year-old K-pop fan from South Korea. He watches Korean music show performances, idol vlogs, and Korean variety entertainment shows. Almost all content is in Korean.",
        "preferred_genres": {"Music": 0.50, "Regional": 0.40, "Entertainment": 0.10},
        "preferred_languages": ["Korean"],
    },
    {
        "user_id": "U15", "name": "Amara",
        "description": "Amara is a 30-year-old biology teacher from Nigeria. She watches science and nature documentaries, biology lectures, and educational explainers. She prefers English content.",
        "preferred_genres": {"Documentary": 0.50, "Educational": 0.40, "News/Analysis": 0.10},
        "preferred_languages": ["English"],
    },
    {
        "user_id": "U16", "name": "Ivan",
        "description": "Ivan is a 38-year-old journalist from Russia. He watches Russian news analysis, political commentary, and sports coverage almost entirely in Russian.",
        "preferred_genres": {"News/Analysis": 0.50, "Regional": 0.35, "Entertainment": 0.15},
        "preferred_languages": ["Russian"],
    },
    {
        "user_id": "U17", "name": "David",
        "description": "David is a 45-year-old news junkie from the UK. He watches news from multiple countries, political analysis, and investigative documentaries in English, French, and Spanish.",
        "preferred_genres": {"News/Analysis": 0.60, "Documentary": 0.30, "Educational": 0.10},
        "preferred_languages": ["English", "French", "Spanish"],
    },
    {
        "user_id": "U18", "name": "Lin",
        "description": "Lin is a 22-year-old mathematics student from China studying in France. She watches math and physics tutorials, coding videos, and science explainers in English and French.",
        "preferred_genres": {"Educational": 0.70, "Documentary": 0.20, "News/Analysis": 0.10},
        "preferred_languages": ["English", "French"],
    },
    {
        "user_id": "U19", "name": "Sara",
        "description": "Sara is a 28-year-old music producer. She watches music videos across all genres and languages, music theory content, and artist documentaries. She has the broadest language preference of any user.",
        "preferred_genres": {"Music": 0.60, "Documentary": 0.25, "Educational": 0.15},
        "preferred_languages": ["English", "Spanish", "French", "Korean", "Japanese"],
    },
    {
        "user_id": "U20", "name": "Tom",
        "description": "Tom is a 19-year-old casual viewer with no strong preferences. He watches whatever is trending — entertainment clips, gaming highlights, and popular music. He is a pure engagement-driven user.",
        "preferred_genres": {"Entertainment": 0.50, "Gaming": 0.30, "Music": 0.20},
        "preferred_languages": ["English"],
    },
]


# ─── Build interaction histories ──────────────────────────────────────────────

def _build_interactions(persona, master_df, n_target=35):
    """
    Sample real videos from master_dataset matching the user's genre + language
    preferences. Assign realistic ratings and watch percentages.
    """
    uid      = persona["user_id"]
    prefs    = persona["preferred_genres"]
    langs    = persona["preferred_languages"]
    interactions = []
    used_ids = set()

    for genre, weight in sorted(prefs.items(), key=lambda x: -x[1]):
        n_genre = max(3, round(weight * n_target))

        # Try language-filtered pool first
        pool = master_df[
            (master_df["genre"] == genre) &
            (master_df["language"].isin(langs)) &
            (~master_df["video_id"].isin(used_ids))
        ]
        # Fall back to any language if too few
        if len(pool) < n_genre:
            pool = master_df[
                (master_df["genre"] == genre) &
                (~master_df["video_id"].isin(used_ids))
            ]

        if pool.empty:
            continue

        sample = pool.sample(min(n_genre, len(pool)),
                             random_state=hash(uid + genre) % 9999)

        for _, row in sample.iterrows():
            # Higher rating for primary genres
            rating   = int(min(5, max(2, round(weight * 8 + np.random.normal(0, 0.6)))))
            watch_pct = round(float(min(1.0, max(0.2, weight + np.random.normal(0, 0.12)))), 2)
            interactions.append({
                "video_id":  row["video_id"],
                "genre":     row["genre"],
                "language":  row["language"],
                "title":     str(row["title"])[:70],
                "rating":    rating,
                "watch_pct": watch_pct,
            })
            used_ids.add(row["video_id"])

    return interactions


def build_and_save(output_dir="data"):
    csv_path = os.path.join(output_dir, "master_dataset.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run dataset_builder.py first."
        )

    master_df = pd.read_csv(csv_path, index_col="idx")
    print(f"Loaded master dataset: {len(master_df):,} videos")

    users_out = []
    for persona in PERSONAS:
        interactions = _build_interactions(persona, master_df)
        user_data    = {**persona, "interactions": interactions}
        users_out.append(user_data)
        genre_counts = {}
        for i in interactions:
            genre_counts[i["genre"]] = genre_counts.get(i["genre"], 0) + 1
        print(f"  {persona['user_id']} {persona['name']:<8} "
              f"{len(interactions):>3} interactions  {genre_counts}")

    out_path = os.path.join(output_dir, "users.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(users_out, f, indent=2, ensure_ascii=False)

    print(f"\n20 user profiles saved -> {out_path}")
    return users_out


if __name__ == "__main__":
    build_and_save()
