"""
data_generator.py
─────────────────
Generates synthetic YouTube-like video, user, and interaction data
for studying demographic/content-type bias in recommender systems.

Suppressed categories (mirror real platform behavior):
  - Educational, Documentary, DIY, News/Analysis, Regional language content
"""

import numpy as np
import pandas as pd
import random
import os

np.random.seed(42)
random.seed(42)

# ─── Configuration ────────────────────────────────────────────────────────────

N_VIDEOS       = 500
N_CREATORS     = 80
N_USERS        = 200
N_INTERACTIONS = 8000

# Category distribution in the video library
CATEGORY_DIST = {
    'Entertainment': 0.15,
    'Music':         0.15,
    'Gaming':        0.125,
    'Educational':   0.20,   # largest single group — but suppressed
    'Documentary':   0.10,
    'DIY':           0.10,
    'News/Analysis': 0.05,
    'Regional':      0.125,
}

LANGUAGES = ['English', 'Hindi', 'Spanish', 'French', 'Arabic', 'Portuguese', 'Korean']

# Realistic language distribution per category
CATEGORY_LANG_DIST = {
    'Entertainment': [0.70, 0.08, 0.08, 0.06, 0.02, 0.04, 0.02],
    'Music':         [0.50, 0.12, 0.12, 0.06, 0.06, 0.08, 0.06],
    'Gaming':        [0.60, 0.06, 0.10, 0.04, 0.04, 0.08, 0.08],
    'Educational':   [0.45, 0.18, 0.14, 0.10, 0.06, 0.04, 0.03],
    'Documentary':   [0.50, 0.10, 0.12, 0.14, 0.08, 0.04, 0.02],
    'DIY':           [0.42, 0.18, 0.16, 0.06, 0.08, 0.08, 0.02],
    'News/Analysis': [0.40, 0.12, 0.14, 0.14, 0.12, 0.06, 0.02],
    'Regional':      [0.05, 0.28, 0.22, 0.14, 0.16, 0.10, 0.05],
}

# Base view scale per category (educational/regional already have fewer organic views)
CATEGORY_VIEW_SCALE = {
    'Entertainment': 1_500_000,
    'Music':         1_200_000,
    'Gaming':          800_000,
    'Educational':     150_000,
    'Documentary':      80_000,
    'DIY':             120_000,
    'News/Analysis':   100_000,
    'Regional':         90_000,
}

# Video title templates per category
TITLE_TEMPLATES = {
    'Entertainment': [
        "I Can't Believe This Happened", "Funniest Moments of {year}",
        "Try Not To Laugh Challenge", "Gone Wrong...", "Surprising My Family",
        "Reacting to Viral Videos", "{month} Fails Compilation"
    ],
    'Music':         [
        "Official Music Video", "Live Performance at {city}",
        "Acoustic Cover - {song}", "New Single Out Now", "Behind The Scenes"
    ],
    'Gaming':        [
        "I Beat The Hardest Level", "1v1 vs Pro Player", "Epic Win Compilation",
        "New Update Review", "First Impressions - {game}", "Top 10 Tips & Tricks"
    ],
    'Educational':   [
        "How {topic} Actually Works", "The Science of {topic}",
        "Learn {topic} in 10 Minutes", "Why {topic} Matters",
        "Understanding {topic} for Beginners", "{topic} Explained Simply",
        "The Truth About {topic}"
    ],
    'Documentary':   [
        "Life Inside {place}", "The Untold Story of {topic}",
        "Inside the World of {topic}", "A Day in the Life: {topic}",
        "The Hidden Reality of {topic}"
    ],
    'DIY':           [
        "Build Your Own {item} at Home", "Easy {item} Tutorial",
        "How I Made {item} for Under $10", "DIY {item} Step by Step",
        "Beginner's Guide to {skill}"
    ],
    'News/Analysis': [
        "Breaking Down {event}", "What Really Happened with {topic}",
        "The Real Impact of {policy}", "Investigative Report: {topic}",
        "Analysis: {topic} Explained"
    ],
    'Regional':      [
        "ये देखकर चौंक जाएंगे", "Mi experiencia en {place}",
        "La vérité sur {topic}", "قصة {topic}", "나의 일상 브이로그",
        "Minha vida em {city}", "O que aprendi sobre {topic}"
    ],
}

TOPICS = [
    'Climate Change', 'Quantum Physics', 'Machine Learning', 'Black Holes',
    'Evolution', 'Economics', 'Psychology', 'History', 'Nutrition', 'Philosophy',
    'Mathematics', 'Coding', 'Language Learning', 'Politics', 'Architecture'
]

CREATOR_SIZES      = ['small', 'medium', 'large']
CREATOR_SIZE_WEIGHTS = [0.60, 0.30, 0.10]

CREATOR_SIZE_SUBS = {
    'small':  (1_000,     50_000),
    'medium': (50_000,  1_000_000),
    'large':  (1_000_000, 50_000_000),
}

AGE_GROUPS = ['13-17', '18-24', '25-34', '35-44', '45+']

# User preference profiles (each user type leans toward certain categories)
USER_PROFILES = {
    'student':       {'Educational': 0.35, 'Gaming': 0.25, 'Entertainment': 0.20, 'Music': 0.10, 'DIY': 0.10},
    'professional':  {'News/Analysis': 0.25, 'Documentary': 0.20, 'Educational': 0.25, 'Music': 0.15, 'Entertainment': 0.15},
    'casual':        {'Entertainment': 0.40, 'Music': 0.25, 'Gaming': 0.20, 'DIY': 0.10, 'Educational': 0.05},
    'regional':      {'Regional': 0.40, 'Music': 0.20, 'Entertainment': 0.15, 'Educational': 0.15, 'DIY': 0.10},
    'creative':      {'DIY': 0.30, 'Documentary': 0.20, 'Educational': 0.20, 'Music': 0.20, 'Entertainment': 0.10},
    'news_junkie':   {'News/Analysis': 0.40, 'Documentary': 0.25, 'Regional': 0.15, 'Educational': 0.15, 'Entertainment': 0.05},
}


# ─── Generators ───────────────────────────────────────────────────────────────

def generate_creators(n=N_CREATORS):
    creators = []
    for i in range(n):
        size = np.random.choice(CREATOR_SIZES, p=CREATOR_SIZE_WEIGHTS)
        sub_min, sub_max = CREATOR_SIZE_SUBS[size]
        creators.append({
            'creator_id':   f'C{i+1:03d}',
            'creator_size': size,
            'subscribers':  int(np.random.uniform(sub_min, sub_max)),
        })
    return pd.DataFrame(creators)


def _pick_title(category):
    templates = TITLE_TEMPLATES.get(category, ['Video about {topic}'])
    template = random.choice(templates)
    return template.format(
        topic=random.choice(TOPICS),
        year=random.choice([2023, 2024, 2025]),
        month=random.choice(['January', 'March', 'July', 'October']),
        city=random.choice(['New York', 'Mumbai', 'Paris', 'Seoul', 'Cairo']),
        song=random.choice(['Love Song', 'Power', 'Dreams', 'Rise']),
        game=random.choice(['Minecraft', 'Valorant', 'FIFA', 'Elden Ring']),
        place=random.choice(['Tokyo', 'the Amazon', 'Silicon Valley', 'Antarctica']),
        event=random.choice(['the Election', 'the Summit', 'the Crisis']),
        policy=random.choice(['the New Law', 'the Budget', 'the Treaty']),
        item=random.choice(['Bookshelf', 'Garden Bed', 'Lamp', 'Phone Stand']),
        skill=random.choice(['Woodworking', 'Sewing', 'Electronics', 'Painting']),
    )


def generate_videos(creators_df, n=N_VIDEOS):
    categories = list(CATEGORY_DIST.keys())
    cat_probs   = list(CATEGORY_DIST.values())

    videos = []
    for i in range(n):
        category = np.random.choice(categories, p=cat_probs)
        lang_probs = CATEGORY_LANG_DIST[category]
        language   = np.random.choice(LANGUAGES, p=lang_probs)
        creator    = creators_df.sample(1).iloc[0]

        view_scale = CATEGORY_VIEW_SCALE[category]
        # Creator size amplifier
        size_multiplier = {'small': 0.2, 'medium': 1.0, 'large': 5.0}[creator['creator_size']]
        views = max(100, int(np.random.lognormal(mean=np.log(view_scale * size_multiplier), sigma=1.2)))

        engagement = np.random.beta(2, 5)          # likes / views ratio
        likes      = int(views * engagement)
        # Educational/Documentary content has higher watch-time despite fewer views
        if category in ('Educational', 'Documentary'):
            avg_watch_pct = np.random.beta(6, 3)   # skewed high
        else:
            avg_watch_pct = np.random.beta(3, 5)   # skewed lower (viral = click but don't finish)

        videos.append({
            'video_id':       f'V{i+1:03d}',
            'title':          _pick_title(category),
            'category':       category,
            'language':       language,
            'is_english':     language == 'English',
            'is_suppressed':  category in ('Educational', 'Documentary', 'DIY', 'News/Analysis', 'Regional'),
            'creator_id':     creator['creator_id'],
            'creator_size':   creator['creator_size'],
            'views':          views,
            'likes':          likes,
            'engagement_rate': round(engagement, 4),
            'avg_watch_pct':   round(avg_watch_pct, 4),
        })

    return pd.DataFrame(videos)


def generate_users(n=N_USERS):
    profiles     = list(USER_PROFILES.keys())
    profile_dist = [0.20, 0.15, 0.25, 0.15, 0.12, 0.13]  # casual is most common

    users = []
    for i in range(n):
        profile   = np.random.choice(profiles, p=profile_dist)
        age_group = np.random.choice(AGE_GROUPS, p=[0.10, 0.30, 0.30, 0.20, 0.10])
        users.append({
            'user_id':        f'U{i+1:03d}',
            'profile_type':   profile,
            'age_group':      age_group,
        })
    return pd.DataFrame(users)


def generate_interactions(users_df, videos_df, n=N_INTERACTIONS):
    """
    Generate user–video interactions based on user profile preferences.
    Watch percentage reflects genuine interest (not algorithmic exposure).
    """
    categories    = videos_df['category'].unique()
    video_by_cat  = {cat: videos_df[videos_df['category'] == cat]['video_id'].tolist()
                     for cat in categories}

    interactions = []
    users_list   = users_df['user_id'].tolist()

    for _ in range(n):
        user_row    = users_df.sample(1).iloc[0]
        user_id     = user_row['user_id']
        profile     = USER_PROFILES[user_row['profile_type']]

        # Pick category based on user profile preferences
        cats   = list(profile.keys())
        probs  = list(profile.values())
        probs  = [p / sum(probs) for p in probs]          # normalize
        category = np.random.choice(cats, p=probs)

        if category not in video_by_cat or not video_by_cat[category]:
            continue

        video_id = random.choice(video_by_cat[category])
        video    = videos_df[videos_df['video_id'] == video_id].iloc[0]

        # Rating: 1-5, higher if category matches preference
        base_rating    = profile.get(category, 0.05) * 10   # scale to ~0-3.5
        rating         = min(5, max(1, round(base_rating + np.random.normal(0, 0.8))))
        watch_pct      = min(1.0, max(0.05,
                            video['avg_watch_pct'] + np.random.normal(0, 0.15)))

        interactions.append({
            'user_id':    user_id,
            'video_id':   video_id,
            'rating':     rating,
            'watch_pct':  round(watch_pct, 3),
        })

    df = pd.DataFrame(interactions).drop_duplicates(subset=['user_id', 'video_id'])
    return df


# ─── Entry Point ──────────────────────────────────────────────────────────────

def generate_all(output_dir='.'):
    print("Generating synthetic data...")

    creators_df     = generate_creators()
    videos_df       = generate_videos(creators_df)
    users_df        = generate_users()
    interactions_df = generate_interactions(users_df, videos_df)

    os.makedirs(output_dir, exist_ok=True)
    creators_df.to_csv(f'{output_dir}/creators.csv',      index=False)
    videos_df.to_csv(f'{output_dir}/videos.csv',          index=False)
    users_df.to_csv(f'{output_dir}/users.csv',            index=False)
    interactions_df.to_csv(f'{output_dir}/interactions.csv', index=False)

    print(f"  Videos:       {len(videos_df)}")
    print(f"  Users:        {len(users_df)}")
    print(f"  Interactions: {len(interactions_df)}")
    print()

    # Category breakdown
    cat_counts = videos_df['category'].value_counts()
    print("  Video library distribution:")
    for cat, cnt in cat_counts.items():
        print(f"    {cat:<20} {cnt:>4} videos  ({cnt/len(videos_df)*100:.1f}%)")
    print()

    lang_counts = videos_df['language'].value_counts()
    print("  Language distribution:")
    for lang, cnt in lang_counts.items():
        print(f"    {lang:<15} {cnt:>4} videos  ({cnt/len(videos_df)*100:.1f}%)")

    return creators_df, videos_df, users_df, interactions_df


if __name__ == '__main__':
    generate_all(output_dir='data')
