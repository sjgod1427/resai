"""
recommender.py
──────────────
Two recommender systems sharing the same SVD collaborative filtering base:

  1. BiasedRecommender  — injects suppression multipliers to penalize
                          educational, non-English, documentary, DIY, and
                          regional content.  Mimics real platform behaviour.

  2. FairRecommender    — same base model, but applies fairness-aware
                          re-ranking (quota-based) to restore proportional
                          representation for suppressed categories.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


# ─── Suppression / Boost Configuration ───────────────────────────────────────

# Biased recommender penalises these categories (multiplied onto CF score)
SUPPRESSION_MULTIPLIERS = {
    'Educational':   0.35,
    'Documentary':   0.45,
    'DIY':           0.50,
    'News/Analysis': 0.40,
    'Regional':      0.30,
}

# Biased recommender boosts these categories
BOOST_MULTIPLIERS = {
    'Entertainment': 1.40,
    'Music':         1.30,
    'Gaming':        1.20,
}

# Penalty applied to ALL non-English videos in biased mode
NON_ENGLISH_PENALTY = 0.40

# ─── Fairness Quotas (for FairRecommender) ────────────────────────────────────
# Minimum fraction of top-N recommendations that must come from each group

FAIRNESS_QUOTAS = {
    'category': {
        'Educational':   0.15,   # at least 15% of recs
        'Documentary':   0.08,
        'DIY':           0.08,
        'News/Analysis': 0.05,
        'Regional':      0.10,
    },
    'non_english_min': 0.25,     # at least 25% non-English
}


# ─── Base Model: SVD Collaborative Filtering ──────────────────────────────────

class SVDCollaborativeFilter:
    """
    Matrix-factorisation collaborative filter using truncated SVD.
    Learns latent user and item factors from the interaction matrix.
    """

    def __init__(self, n_factors=40):
        self.n_factors   = n_factors
        self.user_idx    = {}      # user_id  → row index
        self.video_idx   = {}      # video_id → col index
        self.U = self.sigma = self.Vt = None
        self.global_mean = 0.0

    def fit(self, interactions_df):
        """Build the user-item matrix and decompose it with SVD."""
        users  = interactions_df['user_id'].unique()
        videos = interactions_df['video_id'].unique()

        self.user_idx  = {u: i for i, u in enumerate(users)}
        self.video_idx = {v: i for i, v in enumerate(videos)}

        rows = interactions_df['user_id'].map(self.user_idx)
        cols = interactions_df['video_id'].map(self.video_idx)
        data = interactions_df['rating'].astype(float)

        n_users  = len(users)
        n_videos = len(videos)

        matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_videos))
        self.global_mean = data.mean()

        # Mean-centre each user's ratings
        dense = matrix.toarray().astype(float)
        user_means = np.true_divide(dense.sum(1), (dense != 0).sum(1) + 1e-9)
        for i in range(n_users):
            mask = dense[i] != 0
            dense[i, mask] -= user_means[i]

        k = min(self.n_factors, min(n_users, n_videos) - 1)
        self.U, sigma, self.Vt = svds(csr_matrix(dense), k=k)
        self.sigma = np.diag(sigma)

        # Full predicted score matrix (n_users × n_videos)
        self._predicted = self.U @ self.sigma @ self.Vt
        # Add global mean back so scores are on a sensible scale
        self._predicted += self.global_mean

        return self

    def raw_scores(self, user_id):
        """
        Return a dict {video_id: raw_cf_score} for all videos seen in training.
        """
        if user_id not in self.user_idx:
            return {}
        u_idx = self.user_idx[user_id]
        scores = {}
        for vid, v_idx in self.video_idx.items():
            scores[vid] = float(self._predicted[u_idx, v_idx])
        return scores


# ─── Biased Recommender ───────────────────────────────────────────────────────

class BiasedRecommender:
    """
    Recommender that deliberately suppresses educational, documentary, DIY,
    news/analysis, regional, and non-English content by multiplying the
    base CF score with a suppression factor.

    This mirrors how engagement-optimised platforms inadvertently (or
    deliberately) de-prioritise these content types.
    """

    def __init__(self, n_factors=40):
        self.cf = SVDCollaborativeFilter(n_factors=n_factors)

    def fit(self, interactions_df):
        self.cf.fit(interactions_df)
        return self

    def recommend(self, user_id, videos_df, top_n=10):
        """
        Returns top-N video_ids with bias applied.
        """
        raw = self.cf.raw_scores(user_id)
        if not raw:
            return []

        biased = {}
        for vid_id, score in raw.items():
            row = videos_df[videos_df['video_id'] == vid_id]
            if row.empty:
                continue
            row     = row.iloc[0]
            category = row['category']
            language = row['language']

            # Apply suppression / boost
            multiplier = SUPPRESSION_MULTIPLIERS.get(category,
                         BOOST_MULTIPLIERS.get(category, 1.0))

            # Non-English penalty (stacks with category suppression)
            if language != 'English':
                multiplier *= NON_ENGLISH_PENALTY

            biased[vid_id] = score * multiplier

        ranked = sorted(biased, key=lambda v: biased[v], reverse=True)
        return ranked[:top_n]


# ─── Fair Recommender ────────────────────────────────────────────────────────

class FairRecommender:
    """
    Recommender that uses the same SVD base as BiasedRecommender but applies
    fairness-aware re-ranking via proportional quotas.

    Algorithm:
      1. Compute raw CF scores (no suppression applied).
      2. Sort candidates by score.
      3. First pass  → fill minimum-quota slots for suppressed categories.
      4. Second pass → fill remaining slots with highest remaining scores.
      5. For non-English diversity, ensure min 25% non-English in the list.

    This is a greedy quota-based approach inspired by the FA*IR algorithm
    (Zehlike et al., 2017).
    """

    def __init__(self, n_factors=40):
        self.cf = SVDCollaborativeFilter(n_factors=n_factors)

    def fit(self, interactions_df):
        self.cf.fit(interactions_df)
        return self

    def recommend(self, user_id, videos_df, top_n=10):
        raw = self.cf.raw_scores(user_id)
        if not raw:
            return []

        # ── Step 1: rank by raw CF score (no suppression) ──────────────────
        ranked_all = sorted(raw, key=lambda v: raw[v], reverse=True)
        # Keep only videos present in videos_df
        ranked_all = [v for v in ranked_all if not videos_df[videos_df['video_id'] == v].empty]

        video_meta = videos_df.set_index('video_id')

        # ── Step 2: compute minimum slot counts ────────────────────────────
        # Work with a larger candidate pool (3× top_n) so quotas can be filled
        pool = ranked_all[:top_n * 3]

        min_slots = {cat: max(1, int(np.ceil(frac * top_n)))
                     for cat, frac in FAIRNESS_QUOTAS['category'].items()}
        non_eng_min = max(1, int(np.ceil(FAIRNESS_QUOTAS['non_english_min'] * top_n)))

        # Group pool by category and language
        by_category = {cat: [] for cat in FAIRNESS_QUOTAS['category']}
        non_english_pool = []
        for vid in pool:
            if vid not in video_meta.index:
                continue
            cat  = video_meta.at[vid, 'category']
            lang = video_meta.at[vid, 'language']
            if cat in by_category:
                by_category[cat].append(vid)
            if lang != 'English':
                non_english_pool.append(vid)

        # ── Step 3: fill quota slots ────────────────────────────────────────
        selected  = []
        used      = set()

        # Fill category quotas (best-scored first within each group)
        for cat, n_slots in min_slots.items():
            filled = 0
            for vid in by_category.get(cat, []):
                if filled >= n_slots:
                    break
                if vid not in used:
                    selected.append(vid)
                    used.add(vid)
                    filled += 1

        # Fill non-English quota
        eng_count     = sum(1 for v in selected
                            if video_meta.at[v, 'language'] == 'English')
        non_eng_count = len(selected) - eng_count
        for vid in non_english_pool:
            if non_eng_count >= non_eng_min:
                break
            if vid not in used:
                selected.append(vid)
                used.add(vid)
                non_eng_count += 1

        # ── Step 4: fill remaining slots with highest raw scores ────────────
        for vid in ranked_all:
            if len(selected) >= top_n:
                break
            if vid not in used:
                selected.append(vid)
                used.add(vid)

        return selected[:top_n]


# ─── Batch Evaluation Helper ─────────────────────────────────────────────────

def get_all_recommendations(recommender, users_df, videos_df, top_n=10):
    """
    Run recommender for all users and return a DataFrame of results.
    Columns: user_id, rank, video_id
    """
    records = []
    for user_id in users_df['user_id']:
        recs = recommender.recommend(user_id, videos_df, top_n=top_n)
        for rank, vid in enumerate(recs, start=1):
            records.append({'user_id': user_id, 'rank': rank, 'video_id': vid})
    return pd.DataFrame(records)
