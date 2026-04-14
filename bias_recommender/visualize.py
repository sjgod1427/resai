"""
visualize.py
────────────
Generates all comparison charts between the biased and fair recommender.

Charts produced:
  1. Category exposure bar chart (biased vs fair vs library)
  2. Language distribution comparison
  3. Representation gap heatmap
  4. Gini curve (cumulative exposure inequality)
  5. Sample user feed simulation (first 20 recs, biased vs fair)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── colour palette ────────────────────────────────────────────────────────────
SUPPRESSED_CATS = {'Educational', 'Documentary', 'DIY', 'News/Analysis', 'Regional'}
PROMOTED_CATS   = {'Entertainment', 'Music', 'Gaming'}

CAT_COLOURS = {
    'Entertainment': '#e74c3c',
    'Music':         '#e67e22',
    'Gaming':        '#f1c40f',
    'Educational':   '#2980b9',
    'Documentary':   '#27ae60',
    'DIY':           '#8e44ad',
    'News/Analysis': '#16a085',
    'Regional':      '#2c3e50',
}

LANG_COLOURS = {
    'English':    '#3498db',
    'Hindi':      '#e74c3c',
    'Spanish':    '#2ecc71',
    'French':     '#9b59b6',
    'Arabic':     '#f39c12',
    'Portuguese': '#1abc9c',
    'Korean':     '#e91e63',
}


def _save(fig, path, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    full_path = os.path.join(plots_dir, path)
    fig.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {full_path}")


# ── 1. Category Exposure Bar Chart ───────────────────────────────────────────

def plot_category_exposure(biased_recs, fair_recs, videos_df, plots_dir='plots'):
    from bias_metrics import exposure_by_category

    biased_exp = exposure_by_category(biased_recs, videos_df)
    fair_exp   = exposure_by_category(fair_recs,   videos_df)
    lib_dist   = (videos_df['category'].value_counts() / len(videos_df) * 100).round(2)

    categories = sorted(videos_df['category'].unique(),
                        key=lambda c: lib_dist.get(c, 0), reverse=True)

    x     = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))

    bars_lib   = ax.bar(x - width, [lib_dist.get(c, 0)   for c in categories],
                        width, label='Library %',      color='#bdc3c7', edgecolor='white')
    bars_bias  = ax.bar(x,         [biased_exp.get(c, 0) for c in categories],
                        width, label='Biased Recs %',  color='#e74c3c', edgecolor='white')
    bars_fair  = ax.bar(x + width, [fair_exp.get(c, 0)   for c in categories],
                        width, label='Fair Recs %',    color='#2980b9', edgecolor='white')

    # Shade suppressed categories
    for i, cat in enumerate(categories):
        if cat in SUPPRESSED_CATS:
            ax.axvspan(i - 1.5*width, i + 1.5*width + width,
                       alpha=0.06, color='blue', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('% of Recommendations', fontsize=11)
    ax.set_title('Category Exposure: Biased vs Fair Recommender vs Library\n'
                 '(shaded = suppressed categories)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(biased_exp.max(), fair_exp.max(), lib_dist.max()) * 1.25)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    # Annotate gap arrows for suppressed categories
    for i, cat in enumerate(categories):
        if cat in SUPPRESSED_CATS:
            b = biased_exp.get(cat, 0)
            f = fair_exp.get(cat, 0)
            if f > b + 1:
                ax.annotate('', xy=(i + width, f + 0.5),
                            xytext=(i, b + 0.5),
                            arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    _save(fig, '1_category_exposure.png', plots_dir)


# ── 2. Language Distribution Comparison ──────────────────────────────────────

def plot_language_distribution(biased_recs, fair_recs, videos_df, plots_dir='plots'):
    from bias_metrics import exposure_by_language

    biased_lang = exposure_by_language(biased_recs, videos_df)
    fair_lang   = exposure_by_language(fair_recs,   videos_df)
    lib_lang    = (videos_df['language'].value_counts() / len(videos_df) * 100).round(2)

    languages = sorted(videos_df['language'].unique(),
                       key=lambda l: lib_lang.get(l, 0), reverse=True)

    x     = np.arange(len(languages))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, [lib_lang.get(l, 0)   for l in languages],
           width, label='Library %',      color='#bdc3c7', edgecolor='white')
    ax.bar(x,         [biased_lang.get(l, 0) for l in languages],
           width, label='Biased Recs %',  color='#e74c3c', edgecolor='white')
    ax.bar(x + width, [fair_lang.get(l, 0)   for l in languages],
           width, label='Fair Recs %',    color='#2980b9', edgecolor='white')

    # Shade non-English
    ax.axvspan(0.5, len(languages) - 0.5, alpha=0.04, color='blue', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(languages, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('% of Recommendations', fontsize=11)
    ax.set_title('Language Exposure: Biased vs Fair Recommender vs Library\n'
                 '(shaded = non-English languages)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    _save(fig, '2_language_distribution.png', plots_dir)


# ── 3. Representation Gap Heatmap ────────────────────────────────────────────

def plot_representation_gap(biased_recs, fair_recs, videos_df, plots_dir='plots'):
    from bias_metrics import representation_gap

    b_gap = representation_gap(biased_recs, videos_df, 'category')['gap'].rename('Biased')
    f_gap = representation_gap(fair_recs,   videos_df, 'category')['gap'].rename('Fair')
    df    = pd.concat([b_gap, f_gap], axis=1).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = max(abs(df.values.min()), abs(df.values.max()))
    im   = ax.imshow(df.values, cmap='RdYlGn', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Biased', 'Fair'], fontsize=12)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=10)

    for i in range(len(df.index)):
        for j, col in enumerate(['Biased', 'Fair']):
            val = df.iloc[i][col]
            ax.text(j, i, f'{val:+.1f}%', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if abs(val) > vmax * 0.5 else 'black')

    plt.colorbar(im, ax=ax, label='Representation Gap (%)')
    ax.set_title('Representation Gap Heatmap\n(green = closer to parity, red = more bias)',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Recommender System', fontsize=11)

    _save(fig, '3_representation_gap_heatmap.png', plots_dir)


# ── 4. Gini Lorenz Curve ─────────────────────────────────────────────────────

def _lorenz(values):
    values = np.sort(np.array(values, dtype=float))
    cumsum = np.cumsum(values)
    return np.concatenate([[0], cumsum / cumsum[-1]])


def plot_gini_curve(biased_recs, fair_recs, videos_df, plots_dir='plots'):
    from bias_metrics import exposure_by_category

    biased_exp = exposure_by_category(biased_recs, videos_df)
    fair_exp   = exposure_by_category(fair_recs,   videos_df)
    lib_exp    = (videos_df['category'].value_counts() / len(videos_df) * 100)

    # Align on same categories
    all_cats = sorted(videos_df['category'].unique())
    b_vals   = [biased_exp.get(c, 0) for c in all_cats]
    f_vals   = [fair_exp.get(c, 0)   for c in all_cats]
    l_vals   = [lib_exp.get(c, 0)    for c in all_cats]

    n    = len(all_cats)
    x    = np.linspace(0, 1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect Equality')
    ax.plot(x, _lorenz(l_vals),   color='#bdc3c7', lw=2, label='Library')
    ax.plot(x, _lorenz(b_vals),   color='#e74c3c', lw=2.5, label='Biased Recs')
    ax.plot(x, _lorenz(f_vals),   color='#2980b9', lw=2.5, label='Fair Recs')

    ax.fill_between(x, _lorenz(b_vals), [xi for xi in x],
                    alpha=0.10, color='red',  label='_nolegend_')
    ax.fill_between(x, _lorenz(f_vals), [xi for xi in x],
                    alpha=0.10, color='blue', label='_nolegend_')

    ax.set_xlabel('Cumulative share of categories (poorest to richest exposure)', fontsize=10)
    ax.set_ylabel('Cumulative share of total exposure', fontsize=10)
    ax.set_title('Lorenz Curve — Category Exposure Inequality\n'
                 '(further from diagonal = more unequal = more biased)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    _save(fig, '4_gini_lorenz_curve.png', plots_dir)


# ── 5. Sample User Feed Simulation ───────────────────────────────────────────

def plot_user_feed_simulation(biased_recs, fair_recs, videos_df,
                               sample_user='U001', top_n=20, plots_dir='plots'):
    """
    For a single user, show the first top_n recommendations as a coloured
    sequence — each block = one video, coloured by category.
    """
    def get_user_recs(recs_df, user):
        rows = recs_df[recs_df['user_id'] == user].sort_values('rank')
        return rows['video_id'].tolist()[:top_n]

    b_vids = get_user_recs(biased_recs, sample_user)
    f_vids = get_user_recs(fair_recs,   sample_user)

    def to_categories(vids):
        cats = []
        for v in vids:
            row = videos_df[videos_df['video_id'] == v]
            cats.append(row.iloc[0]['category'] if not row.empty else 'Unknown')
        return cats

    b_cats = to_categories(b_vids)
    f_cats = to_categories(f_vids)

    fig, axes = plt.subplots(2, 1, figsize=(16, 4))

    for ax, cats, label, color in zip(
            axes,
            [b_cats, f_cats],
            ['Biased Feed', 'Fair Feed'],
            ['#e74c3c', '#2980b9']):

        for i, cat in enumerate(cats):
            rect = plt.Rectangle([i, 0], 1, 1,
                                  color=CAT_COLOURS.get(cat, '#95a5a6'),
                                  ec='white', lw=1.5)
            ax.add_patch(rect)
            ax.text(i + 0.5, 0.5, cat[:3], ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

        ax.set_xlim(0, top_n)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xticks(range(top_n))
        ax.set_xticklabels([f'#{i+1}' for i in range(top_n)], fontsize=8)
        ax.set_title(f'{label} — User {sample_user}', fontsize=11,
                     color=color, fontweight='bold')
        ax.spines[['top', 'right', 'left']].set_visible(False)

    # Shared legend
    handles = [mpatches.Patch(color=v, label=k) for k, v in CAT_COLOURS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.15))

    fig.suptitle(f'Recommendation Feed Simulation — First {top_n} Videos',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, '5_user_feed_simulation.png', plots_dir)


# ── 6. Summary Metrics Bar Chart ─────────────────────────────────────────────

def plot_summary_metrics(biased_metrics, fair_metrics, plots_dir='plots'):
    """
    Side-by-side bar chart of all scalar fairness metrics.
    """
    metric_labels = {
        'gini':                    'Gini Coeff.\n(↓ better)',
        'language_diversity':      'Language\nDiversity (↑)',
        'intra_list_diversity':    'Intra-List\nDiversity (↑)',
        'suppressed_content_rate': 'Suppressed\nContent Rate (↑)',
        'non_english_rate':        'Non-English\nRate (↑)',
    }

    keys = list(metric_labels.keys())
    b_vals = [biased_metrics.get(k, 0) for k in keys]
    f_vals = [fair_metrics.get(k, 0)   for k in keys]

    # For Gini: lower is better — invert for visual comparison
    def normalise(key, val):
        if key == 'gini':
            return 1 - val   # invert so higher = better
        if key == 'intra_list_diversity':
            return val / 8    # max possible ≈ 8 categories
        return val

    b_norm = [normalise(k, v) for k, v in zip(keys, b_vals)]
    f_norm = [normalise(k, v) for k, v in zip(keys, f_vals)]

    x     = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width/2, b_norm, width, label='Biased',
           color='#e74c3c', edgecolor='white')
    ax.bar(x + width/2, f_norm, width, label='Fair',
           color='#2980b9', edgecolor='white')

    # Raw values as text
    for i, (b, f) in enumerate(zip(b_vals, f_vals)):
        ax.text(i - width/2, b_norm[i] + 0.01, f'{b:.2f}',
                ha='center', va='bottom', fontsize=8, color='#e74c3c')
        ax.text(i + width/2, f_norm[i] + 0.01, f'{f:.2f}',
                ha='center', va='bottom', fontsize=8, color='#2980b9')

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[k] for k in keys], fontsize=10)
    ax.set_ylabel('Normalised Score (higher = fairer)', fontsize=11)
    ax.set_title('Fairness Metrics: Biased vs Fair Recommender\n'
                 '(raw values shown above bars)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    _save(fig, '6_summary_metrics.png', plots_dir)


# ─── Run All ─────────────────────────────────────────────────────────────────

def generate_all_plots(biased_recs, fair_recs, videos_df,
                        biased_metrics, fair_metrics,
                        sample_user='U001', plots_dir='plots'):
    print("\nGenerating visualisations...")
    plot_category_exposure(biased_recs, fair_recs, videos_df, plots_dir)
    plot_language_distribution(biased_recs, fair_recs, videos_df, plots_dir)
    plot_representation_gap(biased_recs, fair_recs, videos_df, plots_dir)
    plot_gini_curve(biased_recs, fair_recs, videos_df, plots_dir)
    plot_user_feed_simulation(biased_recs, fair_recs, videos_df,
                               sample_user=sample_user, plots_dir=plots_dir)
    plot_summary_metrics(biased_metrics, fair_metrics, plots_dir)
    print(f"\n  All plots saved to '{plots_dir}/'")
