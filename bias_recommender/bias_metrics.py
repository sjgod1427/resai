"""
bias_metrics.py
───────────────
Quantitative fairness metrics for evaluating recommender system bias.

Metrics implemented:
  1. Exposure Ratio          — how often each group appears in recommendations
  2. Representation Gap      — exposure % vs library % (over/under representation)
  3. Demographic Parity Gap  — max difference in exposure rate across groups
  4. Gini Coefficient        — inequality of exposure distribution
  5. Language Diversity Score— Shannon entropy of language distribution in recs
  6. Intra-List Diversity    — average category diversity per user's rec list
"""

import numpy as np
import pandas as pd
from collections import Counter


# ─── 1. Exposure Ratio ────────────────────────────────────────────────────────

def exposure_by_category(recs_df, videos_df):
    """
    Percentage of total recommendations belonging to each category.

    Args:
        recs_df   : DataFrame with columns [user_id, rank, video_id]
        videos_df : DataFrame with columns [video_id, category, ...]

    Returns:
        pd.Series  category → exposure_percentage
    """
    merged = recs_df.merge(videos_df[['video_id', 'category']], on='video_id')
    total  = len(merged)
    counts = merged['category'].value_counts()
    return (counts / total * 100).round(2)


def exposure_by_language(recs_df, videos_df):
    """Percentage of recommendations in each language."""
    merged = recs_df.merge(videos_df[['video_id', 'language']], on='video_id')
    total  = len(merged)
    counts = merged['language'].value_counts()
    return (counts / total * 100).round(2)


# ─── 2. Representation Gap ────────────────────────────────────────────────────

def representation_gap(recs_df, videos_df, column='category'):
    """
    For each group (category or language), compute:
        gap = exposure_% - library_%

    Positive = over-represented in recommendations
    Negative = under-represented (suppressed)

    Returns:
        pd.DataFrame  with columns [group, library_pct, exposure_pct, gap]
    """
    # Library distribution
    lib_counts  = videos_df[column].value_counts()
    lib_pct     = (lib_counts / len(videos_df) * 100).rename('library_pct')

    # Recommendation distribution
    merged      = recs_df.merge(videos_df[['video_id', column]], on='video_id')
    rec_counts  = merged[column].value_counts()
    rec_pct     = (rec_counts / len(merged) * 100).rename('exposure_pct')

    result = pd.concat([lib_pct, rec_pct], axis=1).fillna(0)
    result['gap'] = (result['exposure_pct'] - result['library_pct']).round(2)
    result = result.sort_values('gap')
    return result


# ─── 3. Demographic Parity Gap ───────────────────────────────────────────────

def demographic_parity_gap(recs_df, videos_df, column='category',
                           group_a='Educational', group_b='Entertainment'):
    """
    Difference in recommendation rates between two specific groups.
    A gap of 0 = perfectly equal exposure rate.
    """
    merged      = recs_df.merge(videos_df[['video_id', column]], on='video_id')
    total       = len(merged)
    rate_a      = (merged[column] == group_a).sum() / total
    rate_b      = (merged[column] == group_b).sum() / total
    return round(abs(rate_b - rate_a), 4)


# ─── 4. Gini Coefficient ─────────────────────────────────────────────────────

def gini_coefficient(recs_df, videos_df, column='category'):
    """
    Gini coefficient of the exposure distribution across groups.
    0 = perfectly equal exposure,  1 = all recommendations go to one group.

    Interpretation:
        < 0.10  very fair
        0.10 – 0.25  mild inequality
        0.25 – 0.40  moderate inequality
        > 0.40  severe bias
    """
    merged = recs_df.merge(videos_df[['video_id', column]], on='video_id')
    counts = np.array(merged[column].value_counts().values, dtype=float)
    counts.sort()

    n     = len(counts)
    index = np.arange(1, n + 1)
    gini  = (2 * (index * counts).sum()) / (n * counts.sum()) - (n + 1) / n
    return round(float(gini), 4)


# ─── 5. Language Diversity Score ─────────────────────────────────────────────

def language_diversity_score(recs_df, videos_df):
    """
    Normalised Shannon entropy of the language distribution in recommendations.
    1.0 = maximum diversity,  0.0 = all content in one language.
    """
    merged = recs_df.merge(videos_df[['video_id', 'language']], on='video_id')
    counts = merged['language'].value_counts().values.astype(float)
    probs  = counts / counts.sum()

    entropy = -np.sum(probs * np.log2(probs + 1e-12))
    max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
    return round(float(entropy / max_entropy), 4)


# ─── 6. Intra-List Diversity ─────────────────────────────────────────────────

def intra_list_diversity(recs_df, videos_df):
    """
    Average number of unique categories per user's recommendation list.
    Higher = more diverse.
    """
    merged = recs_df.merge(videos_df[['video_id', 'category']], on='video_id')
    per_user = merged.groupby('user_id')['category'].nunique()
    return round(float(per_user.mean()), 4)


# ─── 7. Suppressed Content Exposure Rate ─────────────────────────────────────

SUPPRESSED_CATEGORIES = {'Educational', 'Documentary', 'DIY', 'News/Analysis', 'Regional'}

def suppressed_exposure_rate(recs_df, videos_df):
    """
    What fraction of all recommendations are from suppressed categories?
    """
    merged = recs_df.merge(videos_df[['video_id', 'category']], on='video_id')
    suppressed = merged['category'].isin(SUPPRESSED_CATEGORIES)
    return round(float(suppressed.mean()), 4)


def non_english_exposure_rate(recs_df, videos_df):
    """
    What fraction of all recommendations are non-English?
    """
    merged = recs_df.merge(videos_df[['video_id', 'language']], on='video_id')
    return round(float((merged['language'] != 'English').mean()), 4)


# ─── Full Report ─────────────────────────────────────────────────────────────

def full_report(recs_df, videos_df, label='Recommender'):
    """
    Prints a complete bias audit report for a set of recommendations.
    Returns a dict of scalar metrics.
    """
    print(f"\n{'='*60}")
    print(f"  BIAS AUDIT — {label}")
    print(f"{'='*60}")

    # ── Category exposure ─────────────────────────────────────────────
    print("\n  Category Exposure vs Library:")
    rep_gap = representation_gap(recs_df, videos_df, column='category')
    for group, row in rep_gap.iterrows():
        tag = ''
        if row['gap'] < -5:
            tag = '  *** UNDER-REPRESENTED'
        elif row['gap'] > 5:
            tag = '  ^  over-represented'
        print(f"    {group:<20} lib={row['library_pct']:5.1f}%  "
              f"rec={row['exposure_pct']:5.1f}%  "
              f"gap={row['gap']:+.1f}%{tag}")

    # ── Language exposure ─────────────────────────────────────────────
    print("\n  Language Exposure vs Library:")
    lang_gap = representation_gap(recs_df, videos_df, column='language')
    for lang, row in lang_gap.iterrows():
        tag = ' ***' if row['gap'] < -5 else ''
        print(f"    {lang:<15} lib={row['library_pct']:5.1f}%  "
              f"rec={row['exposure_pct']:5.1f}%  "
              f"gap={row['gap']:+.1f}%{tag}")

    # ── Scalar metrics ────────────────────────────────────────────────
    gini    = gini_coefficient(recs_df, videos_df, column='category')
    ld      = language_diversity_score(recs_df, videos_df)
    ild     = intra_list_diversity(recs_df, videos_df)
    dp_gap  = demographic_parity_gap(recs_df, videos_df,
                                     group_a='Educational', group_b='Entertainment')
    supp_r  = suppressed_exposure_rate(recs_df, videos_df)
    non_eng = non_english_exposure_rate(recs_df, videos_df)

    print(f"\n  Summary Metrics:")
    print(f"    Gini Coefficient (category exposure) : {gini:.4f}  "
          f"({'severe' if gini > 0.4 else 'moderate' if gini > 0.25 else 'mild'} inequality)")
    print(f"    Language Diversity Score             : {ld:.4f}  (1=max diversity)")
    print(f"    Intra-List Diversity (avg categories): {ild:.2f}  per user")
    print(f"    Demographic Parity Gap               : {dp_gap:.4f}  "
          f"(Educational vs Entertainment)")
    print(f"    Suppressed Content Rate              : {supp_r*100:.1f}%  of all recs")
    print(f"    Non-English Content Rate             : {non_eng*100:.1f}%  of all recs")

    return {
        'label':                    label,
        'gini':                     gini,
        'language_diversity':       ld,
        'intra_list_diversity':     ild,
        'demographic_parity_gap':   dp_gap,
        'suppressed_content_rate':  supp_r,
        'non_english_rate':         non_eng,
    }
