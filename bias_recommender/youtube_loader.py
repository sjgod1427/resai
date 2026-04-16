"""
youtube_loader.py
─────────────────
Loads real YouTube trending data (datasnaek/youtube-new Kaggle dataset) and
produces the same four DataFrames that data_generator.generate_all() returns,
so the rest of the codebase (recommender.py, bias_metrics.py, app.py) works
without any changes.

What comes from real data:
  video_id, title, channel (creator), category, language, views, likes

What is still synthetic (not available in public YouTube data):
  avg_watch_pct  — per-video watch completion (private metric)
  users          — individual user profiles
  interactions   — who watched what (private per-user data)

Usage:
    from youtube_loader import load_all
    creators_df, videos_df, users_df, interactions_df = load_all()
"""

import os, json, random
import numpy as np
import pandas as pd
import kagglehub

np.random.seed(42)
random.seed(42)

# ─── Dataset path ─────────────────────────────────────────────────────────────

KAGGLE_DATASET = 'datasnaek/youtube-new'

# ─── Country → language mapping ───────────────────────────────────────────────

COUNTRY_LANGUAGE = {
    'US': 'English',
    'CA': 'English',
    'GB': 'English',
    'IN': 'Hindi',
    'MX': 'Spanish',
    'FR': 'French',
    'DE': 'German',
    'JP': 'Japanese',
    'KR': 'Korean',
    'RU': 'Russian',
}

# ─── YouTube category ID → project category ───────────────────────────────────
# Unmapped IDs fall back to 'Entertainment'

YOUTUBE_CAT_MAP = {
    27: 'Educational',    # Education
    28: 'Educational',    # Science & Technology
    25: 'News/Analysis',  # News & Politics
    29: 'Documentary',    # Nonprofits & Activism
    35: 'Documentary',    # Documentary
    26: 'DIY',            # Howto & Style
    10: 'Music',          # Music
    20: 'Gaming',         # Gaming
    24: 'Entertainment',  # Entertainment
    23: 'Entertainment',  # Comedy
    22: 'Entertainment',  # People & Blogs
    1:  'Entertainment',  # Film & Animation
    17: 'Entertainment',  # Sports
    15: 'Entertainment',  # Pets & Animals
    19: 'Entertainment',  # Travel & Events
    21: 'Entertainment',  # Videoblogging
    2:  'Entertainment',  # Autos & Vehicles
}

# For non-English videos these YouTube categories become 'Regional'
# (local-language lifestyle/entertainment content)
REGIONAL_YT_CATS = {22, 21, 19, 17, 15, 1, 2}

# Categories that are suppressed by the BiasedRecommender
SUPPRESSED_CATEGORIES = {'Educational', 'Documentary', 'DIY', 'News/Analysis', 'Regional'}

# ─── Synthetic user config (unchanged from data_generator.py) ─────────────────

N_USERS        = 200
N_INTERACTIONS = 8000
N_VIDEOS_SAMPLE = 800   # how many real videos to keep after deduplication + sampling

USER_PROFILES = {
    'student':      {'Educational': 0.35, 'Gaming': 0.25, 'Entertainment': 0.20, 'Music': 0.10, 'DIY': 0.10},
    'professional': {'News/Analysis': 0.25, 'Documentary': 0.20, 'Educational': 0.25, 'Music': 0.15, 'Entertainment': 0.15},
    'casual':       {'Entertainment': 0.40, 'Music': 0.25, 'Gaming': 0.20, 'DIY': 0.10, 'Educational': 0.05},
    'regional':     {'Regional': 0.40, 'Music': 0.20, 'Entertainment': 0.15, 'Educational': 0.15, 'DIY': 0.10},
    'creative':     {'DIY': 0.30, 'Documentary': 0.20, 'Educational': 0.20, 'Music': 0.20, 'Entertainment': 0.10},
    'news_junkie':  {'News/Analysis': 0.40, 'Documentary': 0.25, 'Regional': 0.15, 'Educational': 0.15, 'Entertainment': 0.05},
}

AGE_GROUPS       = ['13-17', '18-24', '25-34', '35-44', '45+']
AGE_GROUP_PROBS  = [0.10, 0.30, 0.30, 0.20, 0.10]
PROFILE_DIST     = [0.20, 0.15, 0.25, 0.15, 0.12, 0.13]   # casual most common


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load & clean real YouTube data
# ═══════════════════════════════════════════════════════════════════════════════

def _map_category(yt_cat_id, language):
    """Map a YouTube numeric category ID to one of the 8 project categories."""
    base = YOUTUBE_CAT_MAP.get(int(yt_cat_id), 'Entertainment')
    # Non-English videos in lifestyle/travel/vlog buckets → Regional
    if language != 'English' and int(yt_cat_id) in REGIONAL_YT_CATS:
        return 'Regional'
    return base


def _estimate_creator_size(max_views):
    """Rough creator-size estimate from peak view count (no subscriber data available)."""
    if max_views >= 5_000_000:
        return 'large'
    if max_views >= 500_000:
        return 'medium'
    return 'small'


def _load_country(path, country, language):
    """Load one country CSV, deduplicate by video_id (keep peak-views row)."""
    fpath = os.path.join(path, f'{country}videos.csv')
    if not os.path.exists(fpath):
        return pd.DataFrame()

    df = pd.read_csv(fpath, encoding='latin-1')

    # Drop rows with missing core fields or error flags
    df = df[~df['video_error_or_removed'].astype(str).str.lower().isin(['true','1'])]
    df = df.dropna(subset=['video_id', 'title', 'category_id', 'views', 'likes'])

    # Keep peak-views row per video (same video can trend on multiple days)
    df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0).astype(int)
    df = df.sort_values('views', ascending=False).drop_duplicates('video_id')

    df['language'] = language
    df['country']  = country
    return df[['video_id', 'title', 'channel_title', 'category_id',
               'views', 'likes', 'language', 'country']]


def load_videos(kaggle_path, n_sample=N_VIDEOS_SAMPLE):
    """
    Load all countries, deduplicate, map categories, sample to n_sample videos
    with a proportional-but-floored strategy so suppressed categories are present.
    """
    frames = []
    for country, language in COUNTRY_LANGUAGE.items():
        frames.append(_load_country(kaggle_path, country, language))

    combined = pd.concat(frames, ignore_index=True)

    # For English-speaking countries the same video may appear in US, CA, GB —
    # keep the highest-viewed copy; for non-English each country is its own language.
    english_mask = combined['language'] == 'English'
    english_dedup = (combined[english_mask]
                     .sort_values('views', ascending=False)
                     .drop_duplicates('video_id'))
    non_english   = combined[~english_mask].copy()   # different languages → keep all

    all_videos = pd.concat([english_dedup, non_english], ignore_index=True)

    # Map to project categories
    all_videos['category'] = all_videos.apply(
        lambda r: _map_category(r['category_id'], r['language']), axis=1
    )

    # ── Stratified sampling: proportional with a minimum floor ────────────────
    # Ensure at least MIN_PER_CAT videos per category so quota re-ranking works
    MIN_PER_CAT = 60
    sampled_parts = []

    cat_counts    = all_videos['category'].value_counts()
    total_avail   = len(all_videos)

    for cat, avail in cat_counts.items():
        # Proportional share of target sample
        proportional = max(MIN_PER_CAT, int(n_sample * avail / total_avail))
        n_take = min(proportional, avail)
        sampled_parts.append(
            all_videos[all_videos['category'] == cat].sample(n_take, random_state=42)
        )

    videos_df = pd.concat(sampled_parts, ignore_index=True)

    # Trim to n_sample if over (remove excess from largest category)
    if len(videos_df) > n_sample:
        largest_cat = videos_df['category'].value_counts().index[0]
        excess = len(videos_df) - n_sample
        large_idx = videos_df[videos_df['category'] == largest_cat].index
        drop_idx  = large_idx[:excess]
        videos_df = videos_df.drop(drop_idx).reset_index(drop=True)

    # ── Derived columns ───────────────────────────────────────────────────────
    videos_df['likes']           = pd.to_numeric(videos_df['likes'], errors='coerce').fillna(0).astype(int)
    videos_df['engagement_rate'] = (videos_df['likes'] / videos_df['views'].clip(lower=1)).round(4)

    # avg_watch_pct is not in the dataset — simulate based on category (same
    # logic as data_generator: educational content has higher completion rates)
    def _sim_watch_pct(cat):
        if cat in ('Educational', 'Documentary'):
            return round(float(np.random.beta(6, 3)), 4)   # skewed high
        return round(float(np.random.beta(3, 5)), 4)       # viral = clicks but partial

    videos_df['avg_watch_pct'] = videos_df['category'].apply(_sim_watch_pct)

    videos_df['is_english']    = videos_df['language'] == 'English'
    videos_df['is_suppressed'] = videos_df['category'].isin(SUPPRESSED_CATEGORIES)

    # ── Creator columns from channel_title ────────────────────────────────────
    channel_max_views = videos_df.groupby('channel_title')['views'].max()
    videos_df['creator_id']   = 'CH_' + videos_df['channel_title'].str.replace(r'\W+', '_', regex=True).str[:20]
    videos_df['creator_size'] = videos_df['channel_title'].map(
        channel_max_views.apply(_estimate_creator_size)
    )

    videos_df = videos_df.rename(columns={'title': 'title'})[
        ['video_id', 'title', 'category', 'language', 'is_english', 'is_suppressed',
         'creator_id', 'creator_size', 'views', 'likes', 'engagement_rate', 'avg_watch_pct']
    ].reset_index(drop=True)

    return videos_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Build creators_df from the real channel data
# ═══════════════════════════════════════════════════════════════════════════════

def build_creators(videos_df):
    """Derive a creators DataFrame from the real channel names in videos_df."""
    creators = (videos_df[['creator_id', 'creator_size', 'views']]
                .groupby(['creator_id', 'creator_size'])['views']
                .max()
                .reset_index()
                .rename(columns={'views': 'subscribers'}))
    # Use peak views as a rough subscriber proxy
    SIZE_SCALE = {'small': 0.1, 'medium': 0.5, 'large': 2.0}
    creators['subscribers'] = (
        creators.apply(lambda r: int(r['subscribers'] * SIZE_SCALE[r['creator_size']]), axis=1)
    )
    return creators


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Synthetic users (identical to data_generator.py)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_users(n=N_USERS):
    profiles = list(USER_PROFILES.keys())
    users    = []
    for i in range(n):
        profile   = np.random.choice(profiles, p=PROFILE_DIST)
        age_group = np.random.choice(AGE_GROUPS, p=AGE_GROUP_PROBS)
        users.append({'user_id': f'U{i+1:03d}', 'profile_type': profile, 'age_group': age_group})
    return pd.DataFrame(users)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Synthetic interactions over REAL video pool (identical logic)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_interactions(users_df, videos_df, n=N_INTERACTIONS):
    """
    Generate synthetic user-video interactions based on user profile preferences,
    drawn from the real video pool. The interaction pattern reflects genuine user
    interest — not algorithmic exposure — so SVD learns unbiased latent factors.
    """
    video_by_cat = {
        cat: videos_df[videos_df['category'] == cat]['video_id'].tolist()
        for cat in videos_df['category'].unique()
    }

    interactions = []
    for _ in range(n):
        user_row = users_df.sample(1).iloc[0]
        profile  = USER_PROFILES[user_row['profile_type']]

        cats  = list(profile.keys())
        probs = [p / sum(profile.values()) for p in profile.values()]
        category = np.random.choice(cats, p=probs)

        if category not in video_by_cat or not video_by_cat[category]:
            continue

        video_id = random.choice(video_by_cat[category])
        video    = videos_df[videos_df['video_id'] == video_id].iloc[0]

        base_rating = profile.get(category, 0.05) * 10
        rating      = int(min(5, max(1, round(base_rating + np.random.normal(0, 0.8)))))
        watch_pct   = round(float(min(1.0, max(0.05,
                        video['avg_watch_pct'] + np.random.normal(0, 0.15)))), 3)

        interactions.append({
            'user_id':  user_row['user_id'],
            'video_id': video_id,
            'rating':   rating,
            'watch_pct': watch_pct,
        })

    return pd.DataFrame(interactions).drop_duplicates(subset=['user_id', 'video_id'])


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point — drop-in replacement for data_generator.generate_all()
# ═══════════════════════════════════════════════════════════════════════════════

def load_all(output_dir=None, n_videos=N_VIDEOS_SAMPLE,
             n_users=N_USERS, n_interactions=N_INTERACTIONS):
    """
    Download (if needed) and process the YouTube dataset.
    Returns (creators_df, videos_df, users_df, interactions_df) —
    identical schema to data_generator.generate_all().
    """
    print("Fetching YouTube dataset via kagglehub...")
    kaggle_path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"  Dataset path: {kaggle_path}")

    print(f"Loading and processing real YouTube video data (target: {n_videos} videos)...")
    videos_df   = load_videos(kaggle_path, n_sample=n_videos)
    creators_df = build_creators(videos_df)

    print(f"  Videos loaded:  {len(videos_df)}")
    print(f"  Unique creators: {len(creators_df)}")

    cat_counts = videos_df['category'].value_counts()
    print("\n  Category distribution (real YouTube data):")
    for cat, cnt in cat_counts.items():
        flag = '  *** suppressed' if cat in SUPPRESSED_CATEGORIES else ''
        print(f"    {cat:<20} {cnt:>4} videos  ({cnt/len(videos_df)*100:.1f}%){flag}")

    lang_counts = videos_df['language'].value_counts()
    print("\n  Language distribution:")
    for lang, cnt in lang_counts.items():
        print(f"    {lang:<12} {cnt:>4} videos  ({cnt/len(videos_df)*100:.1f}%)")

    print(f"\nGenerating {n_users} synthetic users...")
    users_df = generate_users(n_users)

    print(f"Generating {n_interactions} synthetic interactions over real video pool...")
    interactions_df = generate_interactions(users_df, videos_df, n_interactions)
    print(f"  Interactions generated: {len(interactions_df)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        creators_df.to_csv(f'{output_dir}/creators.csv',       index=False)
        videos_df.to_csv(f'{output_dir}/videos.csv',           index=False)
        users_df.to_csv(f'{output_dir}/users.csv',             index=False)
        interactions_df.to_csv(f'{output_dir}/interactions.csv', index=False)
        print(f"\n  CSVs saved to {output_dir}/")

    print("\nDone.")
    return creators_df, videos_df, users_df, interactions_df


if __name__ == '__main__':
    load_all(output_dir='data')
