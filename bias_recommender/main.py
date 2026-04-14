"""
main.py
───────
Full pipeline for the Demographic/Content-Type Bias study.

Steps:
  1. Generate synthetic YouTube-like data
  2. Train BiasedRecommender  (suppresses Educational, non-English, etc.)
  3. Train FairRecommender    (fairness-aware re-ranking)
  4. Measure bias metrics for both
  5. Print comparison report
  6. Generate all visualisation plots

Run:
    python main.py
"""

import os
import sys

# ── ensure local imports work regardless of where script is called from ───────
sys.path.insert(0, os.path.dirname(__file__))

from data_generator import generate_all
from recommender    import BiasedRecommender, FairRecommender, get_all_recommendations
from bias_metrics   import full_report
from visualize      import generate_all_plots


PLOTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots', 'bias_study')
DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
TOP_N     = 10


def print_header():
    print("\n" + "="*65)
    print("   RESPONSIBLE AI - Content-Type & Language Bias Study")
    print("   Recommender System Bias Detection & Mitigation")
    print("="*65)


def compare_scalar_metrics(biased_m, fair_m):
    print("\n" + "="*65)
    print("  SIDE-BY-SIDE COMPARISON")
    print("="*65)
    print(f"  {'Metric':<38} {'Biased':>8}  {'Fair':>8}  {'Change':>8}")
    print(f"  {'-'*62}")

    comparisons = [
        ('Gini Coefficient (lower=better)',         'gini',
         lambda b, f: f'{(f-b):+.4f}',
         lambda b, f: '[IMPROVED]' if f < b else '[WORSE]'),

        ('Language Diversity Score (higher=better)', 'language_diversity',
         lambda b, f: f'{(f-b):+.4f}',
         lambda b, f: '[IMPROVED]' if f > b else '[WORSE]'),

        ('Intra-List Diversity (higher=better)',     'intra_list_diversity',
         lambda b, f: f'{(f-b):+.4f}',
         lambda b, f: '[IMPROVED]' if f > b else '[WORSE]'),

        ('Suppressed Content Rate (higher=better)',  'suppressed_content_rate',
         lambda b, f: f'{(f-b)*100:+.1f}pp',
         lambda b, f: '[IMPROVED]' if f > b else '[WORSE]'),

        ('Non-English Content Rate (higher=better)', 'non_english_rate',
         lambda b, f: f'{(f-b)*100:+.1f}pp',
         lambda b, f: '[IMPROVED]' if f > b else '[WORSE]'),

        ('Demographic Parity Gap (lower=better)',    'demographic_parity_gap',
         lambda b, f: f'{(f-b):+.4f}',
         lambda b, f: '[IMPROVED]' if f < b else '[WORSE]'),
    ]

    for label, key, delta_fn, verdict_fn in comparisons:
        b = biased_m.get(key, 0)
        f = fair_m.get(key, 0)
        print(f"  {label:<38} {b:>8.4f}  {f:>8.4f}  "
              f"{delta_fn(b, f):>8}  {verdict_fn(b, f)}")


def print_methodology():
    sep = "  " + "-"*59
    print()
    print(sep)
    print("  HOW THE BIAS WAS INJECTED")
    print(sep)
    print("  Base model: SVD Collaborative Filtering")
    print()
    print("  Suppression multipliers applied to CF scores:")
    print("    Educational   x 0.35  (-65% score penalty)")
    print("    Regional      x 0.30  (-70%)")
    print("    News/Analysis x 0.40  (-60%)")
    print("    Documentary   x 0.45  (-55%)")
    print("    DIY           x 0.50  (-50%)")
    print("    Non-English   x 0.40  (stacks with above)")
    print()
    print("  Boost multipliers:")
    print("    Entertainment x 1.40  (+40%)")
    print("    Music         x 1.30  (+30%)")
    print("    Gaming        x 1.20  (+20%)")
    print(sep)
    print("  HOW THE BIAS WAS MITIGATED (FairRecommender)")
    print(sep)
    print("  Method: Fairness-Aware Greedy Re-ranking")
    print("  (Inspired by FA*IR - Zehlike et al., 2017)")
    print()
    print("  Step 1: Compute raw CF scores (NO suppression)")
    print("  Step 2: Fill minimum-quota slots first:")
    print("    Educational   >= 15% of top-N")
    print("    Regional      >= 10%")
    print("    Documentary   >= 8%")
    print("    DIY           >= 8%")
    print("    News/Analysis >= 5%")
    print("    Non-English   >= 25%")
    print("  Step 3: Fill remaining slots with highest raw scores")
    print(sep)


def main():
    print_header()

    # ── 1. Data Generation ─────────────────────────────────────────────────
    creators_df, videos_df, users_df, interactions_df = generate_all(
        output_dir=DATA_DIR
    )

    # ── 2. Train Both Recommenders ─────────────────────────────────────────
    print("\nTraining recommenders...")

    biased_rec = BiasedRecommender(n_factors=40)
    biased_rec.fit(interactions_df)
    print("  BiasedRecommender: trained")

    fair_rec = FairRecommender(n_factors=40)
    fair_rec.fit(interactions_df)
    print("  FairRecommender:   trained")

    # ── 3. Generate Recommendations for All Users ──────────────────────────
    print("\nGenerating recommendations for all users...")
    biased_recs = get_all_recommendations(biased_rec, users_df, videos_df, top_n=TOP_N)
    fair_recs   = get_all_recommendations(fair_rec,   users_df, videos_df, top_n=TOP_N)
    print(f"  Biased: {len(biased_recs)} recommendation rows")
    print(f"  Fair:   {len(fair_recs)} recommendation rows")

    # ── 4. Measure Bias ────────────────────────────────────────────────────
    biased_metrics = full_report(biased_recs, videos_df, label='BIASED RECOMMENDER')
    fair_metrics   = full_report(fair_recs,   videos_df, label='FAIR RECOMMENDER')

    # ── 5. Side-by-Side Comparison ─────────────────────────────────────────
    compare_scalar_metrics(biased_metrics, fair_metrics)

    # ── 6. Methodology Summary ─────────────────────────────────────────────
    print_methodology()

    # ── 7. Visualisations ──────────────────────────────────────────────────
    # Pick a sample user who has a 'student' or 'professional' profile
    student_users = users_df[users_df['profile_type'].isin(['student', 'professional'])]
    sample_user   = student_users.iloc[0]['user_id'] if len(student_users) > 0 else 'U001'

    generate_all_plots(
        biased_recs, fair_recs, videos_df,
        biased_metrics, fair_metrics,
        sample_user=sample_user,
        plots_dir=PLOTS_DIR,
    )

    print("\n" + "="*65)
    print("  Study complete.")
    print(f"  Plots saved in: {os.path.abspath(PLOTS_DIR)}")
    print("="*65 + "\n")


if __name__ == '__main__':
    main()
