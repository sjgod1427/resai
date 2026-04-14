"""
app.py
──────
Gradio interface for the Content-Type & Language Bias Study.

Simulates the full pipeline:
  Data Generation  →  SVD CF Training  →  Bias Injection  →  Fair Re-ranking
                                       →  Metrics Comparison & Visualisation

Run:
    python app.py
"""

import os, sys, io
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from data_generator import generate_all
from recommender import (
    BiasedRecommender, FairRecommender, get_all_recommendations,
    SUPPRESSION_MULTIPLIERS, BOOST_MULTIPLIERS, NON_ENGLISH_PENALTY,
)
from bias_metrics import (
    exposure_by_category, exposure_by_language, representation_gap,
    gini_coefficient, language_diversity_score, intra_list_diversity,
    suppressed_exposure_rate, non_english_exposure_rate,
    demographic_parity_gap, SUPPRESSED_CATEGORIES,
)

# ─── Global state ─────────────────────────────────────────────────────────────
_s = dict(
    ready=False,
    videos_df=None, users_df=None, interactions_df=None,
    biased_rec=None, fair_rec=None,
    biased_recs=None, fair_recs=None,
)

# ─── Colour palettes ──────────────────────────────────────────────────────────
CAT_COL = {
    'Entertainment': '#e74c3c', 'Music': '#e67e22', 'Gaming': '#f39c12',
    'Educational':   '#2980b9', 'Documentary': '#27ae60', 'DIY': '#8e44ad',
    'News/Analysis': '#16a085', 'Regional': '#2c3e50',
}
LANG_COL = {
    'English': '#3498db', 'Hindi': '#e74c3c', 'Spanish': '#2ecc71',
    'French':  '#9b59b6', 'Arabic': '#f39c12', 'Portuguese': '#1abc9c',
    'Korean':  '#e91e63',
}
USER_PROFILES_INFO = {
    'student':      'Prefers Educational (35%), Gaming (25%), Entertainment (20%)',
    'professional': 'Prefers News/Analysis (25%), Educational (25%), Documentary (20%)',
    'casual':       'Prefers Entertainment (40%), Music (25%), Gaming (20%)',
    'regional':     'Prefers Regional (40%), Music (20%), Entertainment (15%)',
    'creative':     'Prefers DIY (30%), Documentary (20%), Educational (20%)',
    'news_junkie':  'Prefers News/Analysis (40%), Documentary (25%), Regional (15%)',
}

# ─── Utility: matplotlib fig → PIL image ──────────────────────────────────────
def _fig_to_pil(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    from PIL import Image
    return Image.open(buf).copy()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PIPELINE INITIALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(progress=gr.Progress()):
    """Generate data, train both recommenders, compute all recommendations."""

    progress(0.05, desc="[1/4] Generating 500 synthetic videos & 200 users ...")
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    _, videos_df, users_df, interactions_df = generate_all(output_dir=data_dir)
    _s.update(videos_df=videos_df, users_df=users_df, interactions_df=interactions_df)

    progress(0.35, desc="[2/4] Training Biased Recommender (SVD, k=40) ...")
    br = BiasedRecommender(n_factors=40)
    br.fit(interactions_df)
    _s['biased_rec'] = br

    progress(0.60, desc="[3/4] Training Fair Recommender (SVD + quota re-ranking) ...")
    fr = FairRecommender(n_factors=40)
    fr.fit(interactions_df)
    _s['fair_rec'] = fr

    progress(0.80, desc="[4/4] Generating top-10 recommendations for all 200 users ...")
    biased_recs = get_all_recommendations(br, users_df, videos_df, top_n=10)
    fair_recs   = get_all_recommendations(fr, users_df, videos_df, top_n=10)
    _s.update(biased_recs=biased_recs, fair_recs=fair_recs, ready=True)

    progress(1.0, desc="Pipeline ready!")

    user_choices  = [(f"{r.user_id} — {r.profile_type}", r.user_id)
                     for _, r in users_df.iterrows()]
    video_choices = [(f"{r.video_id} | {r.category} | {r.language} | {r.title[:45]}", r.video_id)
                     for _, r in videos_df.iterrows()]

    return (
        _build_status_html(),
        _build_library_chart(),
        gr.update(choices=user_choices,  value=user_choices[0][1]),
        gr.update(choices=video_choices, value=video_choices[0][1]),
    )


def _build_status_html():
    v, u, i = _s['videos_df'], _s['users_df'], _s['interactions_df']
    n_supp = int(v['is_suppressed'].sum())

    cat_rows = ''.join(
        f"<tr><td style='padding:5px 10px'>{cat}</td>"
        f"<td style='text-align:right;padding:5px 10px'>{cnt}</td>"
        f"<td style='text-align:right;padding:5px 10px'>{cnt/len(v)*100:.1f}%</td>"
        f"<td style='padding:5px 10px'>"
        f"<div style='height:12px;width:{int(cnt/len(v)*180)}px;"
        f"background:{CAT_COL.get(cat,'#999')};border-radius:3px'></div></td></tr>"
        for cat, cnt in v['category'].value_counts().items()
    )

    return f"""
    <div style='font-family:sans-serif;padding:6px'>
      <div style='display:flex;gap:16px;flex-wrap:wrap;margin-bottom:18px'>
        {_stat_card(len(v),  "Videos",       "#2980b9", "#eaf4fb")}
        {_stat_card(len(u),  "Users",         "#27ae60", "#eafaf1")}
        {_stat_card(len(i),  "Interactions",  "#f39c12", "#fef9e7")}
        {_stat_card(f"{n_supp}/{len(v)}", "Suppressed<br>Videos", "#e74c3c", "#fdedec")}
      </div>
      <h4 style='color:#34495e;margin:4px 0 8px'>Video Library — Category Distribution</h4>
      <table style='border-collapse:collapse;width:100%'>
        <tr style='background:#f0f0f0'>
          <th style='text-align:left;padding:6px 10px'>Category</th>
          <th style='text-align:right;padding:6px 10px'>Count</th>
          <th style='text-align:right;padding:6px 10px'>Share</th>
          <th style='padding:6px 10px'>Bar</th>
        </tr>
        {cat_rows}
      </table>
    </div>"""


def _stat_card(val, label, fg, bg):
    return (f"<div style='background:{bg};padding:14px 20px;border-radius:10px;"
            f"min-width:100px;text-align:center'>"
            f"<div style='font-size:1.9em;font-weight:bold;color:{fg}'>{val}</div>"
            f"<div style='color:#555;font-size:13px'>{label}</div></div>")


def _build_library_chart():
    v = _s['videos_df']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    cats   = v['category'].value_counts()
    colors = [CAT_COL.get(c, '#999') for c in cats.index]
    bars   = ax1.barh(cats.index[::-1], cats.values[::-1],
                      color=colors[::-1], edgecolor='white', height=0.65)
    for bar, val in zip(bars, cats.values[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f'{val/len(v)*100:.1f}%', va='center', fontsize=9, color='#555')
    ax1.set_xlabel('Number of Videos', fontsize=10)
    ax1.set_title('Category Distribution in Library', fontweight='bold', fontsize=12)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='x', alpha=0.3)

    langs   = v['language'].value_counts()
    lcolors = [LANG_COL.get(l, '#999') for l in langs.index]
    wedges, texts, autotexts = ax2.pie(
        langs.values, labels=langs.index, colors=lcolors,
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 9}, pctdistance=0.82,
    )
    ax2.set_title('Language Distribution in Library', fontweight='bold', fontsize=12)

    plt.tight_layout()
    return _fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEED SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_feed(user_id):
    if not _s['ready']:
        return "<p style='color:red'>Please initialise the pipeline first (Tab 1).</p>", None

    v  = _s['videos_df'].set_index('video_id')
    u  = _s['users_df']
    br = _s['biased_recs']
    fr = _s['fair_recs']

    user_row = _s['users_df'][_s['users_df']['user_id'] == user_id].iloc[0]
    profile  = user_row['profile_type']

    b_vids = br[br['user_id'] == user_id].sort_values('rank')['video_id'].tolist()
    f_vids = fr[fr['user_id'] == user_id].sort_values('rank')['video_id'].tolist()

    def video_card(vid, rank):
        row  = v.loc[vid]
        cat  = row['category']
        lang = row['language']
        supp = row['is_suppressed']
        col  = CAT_COL.get(cat, '#999')
        badge = ("<span style='font-size:10px;background:rgba(255,255,255,0.25);"
                 "padding:2px 7px;border-radius:10px;margin-left:8px'>"
                 "SUPPRESSED</span>") if supp else ""
        title = str(row['title'])[:52] + ('…' if len(str(row['title'])) > 52 else '')
        return (
            f"<div style='background:{col};color:white;padding:10px 14px;"
            f"border-radius:8px;margin-bottom:5px;display:flex;"
            f"justify-content:space-between;align-items:center'>"
            f"<div><strong>#{rank}</strong>&nbsp; {title}{badge}</div>"
            f"<div style='font-size:11px;opacity:0.9;white-space:nowrap;margin-left:10px'>"
            f"{cat}&nbsp;|&nbsp;{lang}</div></div>"
        )

    b_html = ''.join(video_card(v_id, i+1) for i, v_id in enumerate(b_vids))
    f_html = ''.join(video_card(v_id, i+1) for i, v_id in enumerate(f_vids))

    b_cats = [_s['videos_df'].set_index('video_id').loc[v_id, 'category'] for v_id in b_vids]
    f_cats = [_s['videos_df'].set_index('video_id').loc[v_id, 'category'] for v_id in f_vids]
    b_supp = sum(1 for c in b_cats if c in SUPPRESSED_CATEGORIES)
    f_supp = sum(1 for c in f_cats if c in SUPPRESSED_CATEGORIES)

    profile_note = USER_PROFILES_INFO.get(profile, '')

    legend_items = ''.join(
        f"<span style='display:inline-block;background:{col};"
        f"color:white;padding:3px 8px;border-radius:12px;"
        f"font-size:11px;margin:2px'>{cat}</span>"
        for cat, col in CAT_COL.items()
    )

    html = f"""
    <div style='font-family:sans-serif'>
      <div style='background:#f4f6f8;padding:12px 16px;border-radius:8px;margin-bottom:14px'>
        <strong>User:</strong> {user_id} &nbsp;|&nbsp;
        <strong>Profile:</strong> {profile.replace('_', ' ').title()} &nbsp;|&nbsp;
        <strong>Age:</strong> {user_row['age_group']}<br>
        <span style='font-size:12px;color:#666;margin-top:4px;display:block'>{profile_note}</span>
      </div>

      <div style='display:flex;gap:16px'>
        <div style='flex:1'>
          <div style='background:#fdedec;padding:8px 14px;border-radius:6px;margin-bottom:10px'>
            <strong style='color:#e74c3c'>Biased Feed</strong>
            &nbsp;<span style='font-size:12px;color:#c0392b'>
              {b_supp}/10 suppressed content shown ({b_supp*10}%)</span>
          </div>
          {b_html}
        </div>
        <div style='flex:1'>
          <div style='background:#eaf4fb;padding:8px 14px;border-radius:6px;margin-bottom:10px'>
            <strong style='color:#2980b9'>Fair Feed</strong>
            &nbsp;<span style='font-size:12px;color:#1a6fa8'>
              {f_supp}/10 suppressed content shown ({f_supp*10}%)</span>
          </div>
          {f_html}
        </div>
      </div>

      <div style='margin-top:14px;padding:10px;background:#f9f9f9;border-radius:8px'>
        <strong style='font-size:12px;color:#555'>Category legend:</strong><br>
        <div style='margin-top:6px'>{legend_items}</div>
        <div style='margin-top:8px;font-size:11px;color:#888'>
          Warm colours (red/orange/yellow) = promoted &nbsp;|&nbsp;
          Cool colours (blue/green/purple) = suppressed in biased mode
        </div>
      </div>
    </div>"""

    chart = _build_feed_chart(b_cats, f_cats)
    return html, chart


def _build_feed_chart(b_cats, f_cats):
    from collections import Counter
    all_cats = list(CAT_COL.keys())
    b_cnt = Counter(b_cats)
    f_cnt = Counter(f_cats)

    x     = np.arange(len(all_cats))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 3.8))

    ax.bar(x - width/2, [b_cnt.get(c, 0) for c in all_cats], width,
           color=[CAT_COL[c] for c in all_cats], label='Biased Feed',
           edgecolor='white', alpha=0.85)
    ax.bar(x + width/2, [f_cnt.get(c, 0) for c in all_cats], width,
           color=[CAT_COL[c] for c in all_cats], label='Fair Feed',
           edgecolor='white', alpha=0.55, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(all_cats, rotation=22, ha='right', fontsize=9)
    ax.set_ylabel('# of Videos in Top-10', fontsize=10)
    ax.set_title('Feed Composition: Biased (solid) vs Fair (hatched)',
                 fontweight='bold', fontsize=11)
    ax.set_yticks(range(0, 11))
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    b_patch = mpatches.Patch(color='#555', label='Biased (solid)')
    f_patch = mpatches.Patch(color='#555', label='Fair (hatched)', hatch='//')
    ax.legend(handles=[b_patch, f_patch], fontsize=9)

    plt.tight_layout()
    return _fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BIAS AUDIT DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def run_audit():
    if not _s['ready']:
        return ("<p style='color:red'>Please initialise the pipeline first.</p>",
                None, None, None, None)

    v  = _s['videos_df']
    br = _s['biased_recs']
    fr = _s['fair_recs']

    b = dict(
        gini           = gini_coefficient(br, v, 'category'),
        lang_diversity = language_diversity_score(br, v),
        ild            = intra_list_diversity(br, v),
        supp_rate      = suppressed_exposure_rate(br, v),
        non_eng_rate   = non_english_exposure_rate(br, v),
        dp_gap         = demographic_parity_gap(br, v, group_a='Educational',
                                                group_b='Entertainment'),
    )
    f = dict(
        gini           = gini_coefficient(fr, v, 'category'),
        lang_diversity = language_diversity_score(fr, v),
        ild            = intra_list_diversity(fr, v),
        supp_rate      = suppressed_exposure_rate(fr, v),
        non_eng_rate   = non_english_exposure_rate(fr, v),
        dp_gap         = demographic_parity_gap(fr, v, group_a='Educational',
                                                group_b='Entertainment'),
    )

    html = _build_metrics_html(b, f, v, br, fr)
    c1   = _plot_category_exposure(br, fr, v)
    c2   = _plot_language_exposure(br, fr, v)
    c3   = _plot_lorenz(br, fr, v)
    c4   = _plot_summary_bars(b, f)
    return html, c1, c2, c3, c4


def _build_metrics_html(b, f, v, br, fr):
    def row(label, b_val, f_val, fmt, lower_better=False):
        improved = (f_val < b_val) if lower_better else (f_val > b_val)
        delta    = f_val - b_val
        color    = '#27ae60' if improved else '#e74c3c'
        sign     = '+' if delta > 0 else ''
        return (
            f"<tr>"
            f"<td style='padding:8px 12px'>{label}</td>"
            f"<td style='text-align:center;padding:8px;color:#e74c3c;font-weight:bold'>"
            f"{fmt.format(b_val)}</td>"
            f"<td style='text-align:center;padding:8px;color:#2980b9;font-weight:bold'>"
            f"{fmt.format(f_val)}</td>"
            f"<td style='text-align:center;padding:8px;color:{color};font-weight:bold'>"
            f"{sign}{fmt.format(delta)}</td>"
            f"<td style='text-align:center;padding:8px'>"
            f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:12px;font-size:12px'>"
            f"{'IMPROVED' if improved else 'NO CHANGE'}</span></td></tr>"
        )

    metrics_rows = (
        row('Gini Coefficient (lower=fairer)',     b['gini'],           f['gini'],           '{:.4f}', lower_better=True)  +
        row('Language Diversity Score (higher)',   b['lang_diversity'], f['lang_diversity'], '{:.4f}') +
        row('Intra-List Diversity (avg cats)',     b['ild'],            f['ild'],            '{:.2f}') +
        row('Suppressed Content Rate (higher)',    b['supp_rate'],      f['supp_rate'],      '{:.2%}') +
        row('Non-English Content Rate (higher)',   b['non_eng_rate'],   f['non_eng_rate'],   '{:.2%}') +
        row('Demographic Parity Gap (lower)',      b['dp_gap'],         f['dp_gap'],         '{:.4f}', lower_better=True)
    )

    # Representation gap tables
    b_rep = representation_gap(br, v, 'category')
    f_rep = representation_gap(fr, v, 'category')

    def rep_row(group, b_row, f_row):
        b_gap = b_row['gap']
        f_gap = f_row['gap']
        b_color = '#e74c3c' if b_gap < -3 else '#27ae60' if b_gap > 3 else '#555'
        f_color = '#e74c3c' if f_gap < -3 else '#27ae60' if f_gap > 3 else '#555'
        cat_color = CAT_COL.get(group, '#999')
        supp_badge = ("<span style='background:#e74c3c;color:white;font-size:10px;"
                      "padding:1px 6px;border-radius:8px;margin-left:4px'>suppressed</span>"
                      ) if group in SUPPRESSED_CATEGORIES else ""
        return (
            f"<tr>"
            f"<td style='padding:6px 10px'>"
            f"<span style='background:{cat_color};color:white;padding:2px 8px;"
            f"border-radius:10px;font-size:12px'>{group}</span>{supp_badge}</td>"
            f"<td style='text-align:center;padding:6px'>{b_row['library_pct']:.1f}%</td>"
            f"<td style='text-align:center;padding:6px;color:{b_color};font-weight:bold'>"
            f"{b_row['exposure_pct']:.1f}% ({b_gap:+.1f}pp)</td>"
            f"<td style='text-align:center;padding:6px;color:{f_color};font-weight:bold'>"
            f"{f_row['exposure_pct']:.1f}% ({f_gap:+.1f}pp)</td></tr>"
        )

    all_cats = sorted(set(b_rep.index) | set(f_rep.index),
                      key=lambda c: b_rep.loc[c, 'gap'] if c in b_rep.index else 0)
    rep_rows = ''.join(
        rep_row(cat,
                b_rep.loc[cat] if cat in b_rep.index else pd.Series({'library_pct':0,'exposure_pct':0,'gap':0}),
                f_rep.loc[cat] if cat in f_rep.index else pd.Series({'library_pct':0,'exposure_pct':0,'gap':0}))
        for cat in all_cats
    )

    return f"""
    <div style='font-family:sans-serif;padding:6px'>
      <h3 style='color:#2c3e50'>Fairness Metrics Comparison</h3>
      <table style='border-collapse:collapse;width:100%;margin-bottom:20px'>
        <tr style='background:#f0f0f0'>
          <th style='text-align:left;padding:8px 12px'>Metric</th>
          <th style='text-align:center;padding:8px;color:#e74c3c'>Biased</th>
          <th style='text-align:center;padding:8px;color:#2980b9'>Fair</th>
          <th style='text-align:center;padding:8px'>Change</th>
          <th style='text-align:center;padding:8px'>Result</th>
        </tr>
        {metrics_rows}
      </table>

      <h3 style='color:#2c3e50'>Category Representation Gap</h3>
      <p style='font-size:12px;color:#777'>
        Positive = over-represented vs library &nbsp;|&nbsp;
        Negative = under-represented (suppressed)
      </p>
      <table style='border-collapse:collapse;width:100%'>
        <tr style='background:#f0f0f0'>
          <th style='text-align:left;padding:6px 10px'>Category</th>
          <th style='text-align:center;padding:6px'>Library %</th>
          <th style='text-align:center;padding:6px;color:#e74c3c'>Biased Recs</th>
          <th style='text-align:center;padding:6px;color:#2980b9'>Fair Recs</th>
        </tr>
        {rep_rows}
      </table>
    </div>"""


def _plot_category_exposure(br, fr, v):
    b_exp = exposure_by_category(br, v)
    f_exp = exposure_by_category(fr, v)
    lib   = (v['category'].value_counts() / len(v) * 100).round(2)
    cats  = sorted(v['category'].unique(), key=lambda c: lib.get(c, 0), reverse=True)

    x     = np.arange(len(cats))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.bar(x - width,   [lib.get(c, 0)   for c in cats], width,
           label='Library %',     color='#bdc3c7', edgecolor='white')
    ax.bar(x,           [b_exp.get(c, 0) for c in cats], width,
           label='Biased Recs %', color='#e74c3c', edgecolor='white')
    ax.bar(x + width,   [f_exp.get(c, 0) for c in cats], width,
           label='Fair Recs %',   color='#2980b9', edgecolor='white')

    for i, cat in enumerate(cats):
        if cat in SUPPRESSED_CATEGORIES:
            ax.axvspan(i - 1.5*width, i + 1.5*width + width,
                       alpha=0.07, color='blue', zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('% of Recommendations', fontsize=10)
    ax.set_title('Category Exposure: Biased vs Fair vs Library\n'
                 '(blue shading = suppressed categories)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return _fig_to_pil(fig)


def _plot_language_exposure(br, fr, v):
    b_lang = exposure_by_language(br, v)
    f_lang = exposure_by_language(fr, v)
    lib    = (v['language'].value_counts() / len(v) * 100).round(2)
    langs  = sorted(v['language'].unique(), key=lambda l: lib.get(l, 0), reverse=True)

    x     = np.arange(len(langs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - width, [lib.get(l, 0)    for l in langs], width,
           label='Library %',     color='#bdc3c7', edgecolor='white')
    ax.bar(x,         [b_lang.get(l, 0) for l in langs], width,
           label='Biased Recs %', color='#e74c3c', edgecolor='white')
    ax.bar(x + width, [f_lang.get(l, 0) for l in langs], width,
           label='Fair Recs %',   color='#2980b9', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(langs, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('% of Recommendations', fontsize=10)
    ax.set_title('Language Exposure: Biased vs Fair vs Library\n'
                 'Biased mode creates 100% English monopoly',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return _fig_to_pil(fig)


def _lorenz_curve(vals):
    vals  = np.sort(np.array(vals, dtype=float))
    cumsum = np.cumsum(vals)
    return np.concatenate([[0], cumsum / (cumsum[-1] + 1e-9)])


def _plot_lorenz(br, fr, v):
    b_exp = exposure_by_category(br, v)
    f_exp = exposure_by_category(fr, v)
    lib   = (v['category'].value_counts() / len(v) * 100)
    cats  = sorted(v['category'].unique())
    n     = len(cats)
    x     = np.linspace(0, 1, n + 1)

    # Compute library gini directly from category distribution
    lib_vals = np.array([lib.get(c, 0) for c in cats], dtype=float)
    lib_vals_s = np.sort(lib_vals)
    idx = np.arange(1, n + 1)
    lib_gini = round(float((2 * (idx * lib_vals_s).sum()) / (n * lib_vals_s.sum()) - (n + 1) / n), 2)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect Equality', lw=1.5)
    ax.plot(x, _lorenz_curve([lib.get(c, 0)   for c in cats]),
            color='#bdc3c7', lw=2,   label=f'Library (Gini={lib_gini:.2f})')
    ax.plot(x, _lorenz_curve([b_exp.get(c, 0) for c in cats]),
            color='#e74c3c', lw=2.5, label=f'Biased  (Gini={gini_coefficient(br, v):.2f})')
    ax.plot(x, _lorenz_curve([f_exp.get(c, 0) for c in cats]),
            color='#2980b9', lw=2.5, label=f'Fair    (Gini={gini_coefficient(fr, v):.2f})')

    ax.fill_between(x, _lorenz_curve([b_exp.get(c, 0) for c in cats]),
                    list(x), alpha=0.10, color='red')
    ax.fill_between(x, _lorenz_curve([f_exp.get(c, 0) for c in cats]),
                    list(x), alpha=0.10, color='blue')

    ax.set_xlabel('Cumulative share of categories', fontsize=10)
    ax.set_ylabel('Cumulative share of exposure',   fontsize=10)
    ax.set_title('Lorenz Curve — Exposure Inequality\n'
                 'Further from diagonal = more biased',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return _fig_to_pil(fig)


def _plot_summary_bars(b, f):
    keys = ['gini', 'lang_diversity', 'ild', 'supp_rate', 'non_eng_rate', 'dp_gap']
    labels = [
        'Gini\n(lower=fairer)', 'Language\nDiversity',
        'Intra-List\nDiversity', 'Suppressed\nContent Rate',
        'Non-English\nRate', 'Parity Gap\n(lower=fairer)',
    ]
    lower_better = {k: k in ('gini', 'dp_gap') for k in keys}

    def normalise(k, val):
        if k == 'gini':        return 1 - val
        if k == 'dp_gap':      return 1 - val
        if k == 'ild':         return val / 8
        return val

    b_n = [normalise(k, b[k]) for k in keys]
    f_n = [normalise(k, f[k]) for k in keys]
    x   = np.arange(len(keys))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(x - w/2, b_n, w, label='Biased', color='#e74c3c', edgecolor='white')
    ax.bar(x + w/2, f_n, w, label='Fair',   color='#2980b9', edgecolor='white')

    for i, k in enumerate(keys):
        ax.text(i - w/2, b_n[i] + 0.02, f'{b[k]:.2f}',
                ha='center', fontsize=8, color='#c0392b')
        ax.text(i + w/2, f_n[i] + 0.02, f'{f[k]:.2f}',
                ha='center', fontsize=8, color='#1a6fa8')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel('Normalised Score (higher = fairer)', fontsize=10)
    ax.set_title('Summary Fairness Metrics — Biased vs Fair\n'
                 'Raw values shown above bars; all metrics normalised for comparison',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    return _fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCORE INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def inspect_video(video_id):
    if not _s['ready']:
        return "<p style='color:red'>Please initialise the pipeline first.</p>", None

    v      = _s['videos_df']
    row    = v[v['video_id'] == video_id].iloc[0]
    cat    = row['category']
    lang   = row['language']

    cat_mult  = SUPPRESSION_MULTIPLIERS.get(cat, BOOST_MULTIPLIERS.get(cat, 1.0))
    lang_mult = NON_ENGLISH_PENALTY if lang != 'English' else 1.0
    total     = cat_mult * lang_mult
    effect_pct = (1 - total) * 100

    # How many users received this video in each recommender
    br      = _s['biased_recs']
    fr      = _s['fair_recs']
    n_total = len(_s['users_df'])
    n_bias  = (br['video_id'] == video_id).sum()
    n_fair  = (fr['video_id'] == video_id).sum()

    # Determine multiplier type
    if cat in SUPPRESSION_MULTIPLIERS:
        cat_type  = 'SUPPRESSED'
        cat_color = '#e74c3c'
        cat_desc  = f'Category penalty: &times;{cat_mult:.2f} ({(1-cat_mult)*100:.0f}% score reduction)'
    elif cat in BOOST_MULTIPLIERS:
        cat_type  = 'BOOSTED'
        cat_color = '#27ae60'
        cat_desc  = f'Category boost: &times;{cat_mult:.2f} (+{(cat_mult-1)*100:.0f}% score increase)'
    else:
        cat_type  = 'NEUTRAL'
        cat_color = '#f39c12'
        cat_desc  = 'No category multiplier (neutral).'

    lang_desc = (f'Non-English penalty: &times;{lang_mult:.2f} ({(1-lang_mult)*100:.0f}% additional reduction)'
                 if lang != 'English' else 'No language penalty (English content).')

    example_raw    = 3.50
    example_biased = example_raw * total
    example_fair   = example_raw   # no suppression in fair mode

    waterfall_html = _waterfall_html(example_raw, cat_mult, lang_mult, example_biased)

    html = f"""
    <div style='font-family:sans-serif;padding:6px'>

      <div style='display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px'>
        <div style='background:{CAT_COL.get(cat,"#999")};color:white;padding:12px 20px;border-radius:10px'>
          <div style='font-size:1.4em;font-weight:bold'>{cat}</div>
          <div style='font-size:12px;opacity:0.9'>Category</div>
        </div>
        <div style='background:{LANG_COL.get(lang,"#999")};color:white;padding:12px 20px;border-radius:10px'>
          <div style='font-size:1.4em;font-weight:bold'>{lang}</div>
          <div style='font-size:12px;opacity:0.9'>Language</div>
        </div>
        <div style='background:#f4f6f8;padding:12px 20px;border-radius:10px'>
          <div style='font-size:1.4em;font-weight:bold;color:#555'>{row["creator_size"].title()}</div>
          <div style='font-size:12px;color:#888'>Creator Size</div>
        </div>
        <div style='background:#f4f6f8;padding:12px 20px;border-radius:10px'>
          <div style='font-size:1.4em;font-weight:bold;color:#555'>{row["views"]:,}</div>
          <div style='font-size:12px;color:#888'>Views</div>
        </div>
      </div>

      <h4 style='color:#2c3e50;margin-bottom:8px'>Video Title</h4>
      <p style='background:#f9f9f9;padding:10px;border-radius:6px;font-size:13px'>{row["title"]}</p>

      <h4 style='color:#2c3e50;margin:14px 0 8px'>Suppression Analysis</h4>

      <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px'>
        <div style='border-left:4px solid {cat_color};padding:10px 14px;background:#fafafa;border-radius:6px'>
          <div style='font-size:11px;font-weight:bold;color:{cat_color};margin-bottom:4px'>
            CATEGORY — {cat_type}</div>
          <div style='font-size:13px'>{cat_desc}</div>
        </div>
        <div style='border-left:4px solid {"#e74c3c" if lang!="English" else "#27ae60"};
                    padding:10px 14px;background:#fafafa;border-radius:6px'>
          <div style='font-size:11px;font-weight:bold;
                      color:{"#e74c3c" if lang!="English" else "#27ae60"};margin-bottom:4px'>
            LANGUAGE — {"NON-ENGLISH PENALTY" if lang!="English" else "NO PENALTY"}</div>
          <div style='font-size:13px'>{lang_desc}</div>
        </div>
      </div>

      <div style='background:{"#fdedec" if total < 0.8 else "#eafaf1"};
                  padding:12px 18px;border-radius:8px;margin-bottom:16px'>
        <strong>Combined multiplier: &times;{total:.3f}</strong>
        &nbsp;&mdash;&nbsp;
        {"Score reduced by " + f"{effect_pct:.0f}%" if total < 1 else
         "Score boosted by " + f"{(total-1)*100:.0f}%"}
        <br>
        <span style='font-size:12px;color:#666'>
          Example: Raw CF score 3.50 &rarr; Biased score {example_biased:.2f}
          (Fair score remains 3.50)
        </span>
      </div>

      {waterfall_html}

      <h4 style='color:#2c3e50;margin:14px 0 8px'>Algorithmic Reach</h4>
      <div style='display:flex;gap:14px'>
        <div style='flex:1;background:#fdedec;padding:12px;border-radius:8px;text-align:center'>
          <div style='font-size:2em;font-weight:bold;color:#e74c3c'>{n_bias}</div>
          <div style='color:#888;font-size:12px'>users receive this video<br>in <strong>Biased</strong> top-10</div>
        </div>
        <div style='flex:1;background:#eaf4fb;padding:12px;border-radius:8px;text-align:center'>
          <div style='font-size:2em;font-weight:bold;color:#2980b9'>{n_fair}</div>
          <div style='color:#888;font-size:12px'>users receive this video<br>in <strong>Fair</strong> top-10</div>
        </div>
        <div style='flex:1;background:#f4f6f8;padding:12px;border-radius:8px;text-align:center'>
          <div style='font-size:2em;font-weight:bold;color:#555'>{n_total}</div>
          <div style='color:#888;font-size:12px'>total users in the system</div>
        </div>
      </div>
    </div>"""

    chart = _build_inspector_chart(cat, lang, cat_mult, lang_mult, total,
                                   example_raw, example_biased)
    return html, chart


def _waterfall_html(raw, cat_mult, lang_mult, final):
    steps = [
        ('Raw CF Score',          raw,   '#3498db'),
        (f'After category &times;{cat_mult:.2f}', raw * cat_mult, '#e67e22'),
        (f'After language &times;{lang_mult:.2f}', final, '#e74c3c'),
    ]
    max_val = raw * 1.1
    bars = ''.join(
        f"<div style='display:flex;align-items:center;margin-bottom:6px'>"
        f"<div style='width:170px;font-size:12px;color:#555'>{label}</div>"
        f"<div style='flex:1;background:#f0f0f0;border-radius:4px;height:22px;position:relative'>"
        f"<div style='background:{col};height:100%;width:{val/max_val*100:.1f}%;border-radius:4px'></div>"
        f"</div>"
        f"<div style='width:50px;text-align:right;font-weight:bold;font-size:13px;color:{col}'>{val:.2f}</div>"
        f"</div>"
        for label, val, col in steps
    )
    return (f"<h4 style='color:#2c3e50;margin:14px 0 6px'>Score Waterfall (Biased Mode)</h4>"
            f"<div style='padding:10px;background:#fafafa;border-radius:8px'>{bars}</div>")


def _build_inspector_chart(cat, lang, cat_mult, lang_mult, total,
                             raw, biased_final):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: waterfall bar chart
    ax = axes[0]
    stages  = ['Raw Score', f'After\nCategory\nx{cat_mult:.2f}',
               f'After\nLanguage\nx{lang_mult:.2f}']
    values  = [raw, raw * cat_mult, biased_final]
    colors  = ['#3498db', '#e67e22', '#e74c3c']
    bars    = ax.bar(stages, values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.axhline(raw, color='#3498db', linestyle='--', alpha=0.5, lw=1.5,
               label=f'Raw (Fair) score = {raw:.2f}')
    ax.set_ylim(0, raw * 1.4)
    ax.set_title(f'Score Degradation in Biased Mode\n({cat} | {lang})',
                 fontweight='bold', fontsize=11)
    ax.set_ylabel('Recommender Score')
    ax.legend(fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    # Right: multiplier breakdown pie
    ax2 = axes[1]
    if total < 1:
        remaining = total
        lost      = 1 - total
        ax2.pie([remaining, lost],
                labels=[f'Kept\n({remaining*100:.0f}%)', f'Lost to bias\n({lost*100:.0f}%)'],
                colors=['#2980b9', '#e74c3c'],
                startangle=90, textprops={'fontsize': 10},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax2.set_title(f'Score Loss from Bias\nTotal multiplier: x{total:.3f}',
                      fontweight='bold', fontsize=11)
    else:
        ax2.pie([1],
                labels=[f'No suppression\n(x{total:.2f} boost)'],
                colors=['#27ae60'],
                startangle=90, textprops={'fontsize': 10},
                wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        ax2.set_title('No suppression applied', fontweight='bold', fontsize=11)

    plt.tight_layout()
    return _fig_to_pil(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE FLOW DIAGRAM (HTML)
# ═══════════════════════════════════════════════════════════════════════════════

FLOW_HTML = """
<div style="font-family:sans-serif;background:#f8f9fa;padding:16px 20px;
            border-radius:12px;margin-bottom:4px">
  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;justify-content:center">

    <div style="background:#2c3e50;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:110px">
      <div style="font-size:11px;opacity:0.7">STEP 1</div>
      <div style="font-weight:bold">Synthetic Data</div>
      <div style="font-size:11px;opacity:0.8">500 videos · 200 users</div>
    </div>

    <div style="font-size:22px;color:#bbb">&rarr;</div>

    <div style="background:#2980b9;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:110px">
      <div style="font-size:11px;opacity:0.7">STEP 2</div>
      <div style="font-weight:bold">SVD CF Model</div>
      <div style="font-size:11px;opacity:0.8">k=40 latent factors</div>
    </div>

    <div style="font-size:22px;color:#bbb">&rarr;</div>

    <div style="background:#e74c3c;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:120px">
      <div style="font-size:11px;opacity:0.7">STEP 3A</div>
      <div style="font-weight:bold">Bias Injection</div>
      <div style="font-size:11px;opacity:0.8">Suppress Educational<br>Non-English &amp; more</div>
    </div>

    <div style="font-size:22px;color:#bbb">&rarr;</div>

    <div style="background:#c0392b;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:110px">
      <div style="font-size:11px;opacity:0.7">OUTPUT A</div>
      <div style="font-weight:bold">Biased Recs</div>
      <div style="font-size:11px;opacity:0.8">97% Entertainment<br>0% Educational</div>
    </div>

  </div>

  <div style="display:flex;align-items:center;flex-wrap:wrap;gap:6px;
              justify-content:center;margin-top:10px">

    <div style="min-width:110px"></div>
    <div style="min-width:22px"></div>

    <div style="background:#27ae60;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:120px">
      <div style="font-size:11px;opacity:0.7">STEP 3B</div>
      <div style="font-weight:bold">Fair Re-ranking</div>
      <div style="font-size:11px;opacity:0.8">Quota-based · FA*IR</div>
    </div>

    <div style="font-size:22px;color:#bbb">&rarr;</div>

    <div style="background:#1a6fa8;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:110px">
      <div style="font-size:11px;opacity:0.7">OUTPUT B</div>
      <div style="font-weight:bold">Fair Recs</div>
      <div style="font-size:11px;opacity:0.8">Diverse · Equitable</div>
    </div>

    <div style="font-size:22px;color:#bbb">&rarr;</div>

    <div style="background:#8e44ad;color:white;padding:10px 16px;border-radius:8px;text-align:center;min-width:110px">
      <div style="font-size:11px;opacity:0.7">STEP 4</div>
      <div style="font-weight:bold">Bias Audit</div>
      <div style="font-size:11px;opacity:0.8">Gini · Parity · Diversity</div>
    </div>

  </div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO APP LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="Responsible AI — Bias in Recommender Systems") as app:

        gr.Markdown("""
# Responsible AI — Content-Type & Language Bias Study
### Demonstrating how recommender systems suppress Educational, Non-English, and Documentary content
        """)

        gr.HTML(FLOW_HTML)

        with gr.Tabs():

            # ── Tab 1: Initialize ───────────────────────────────────────────
            with gr.Tab("1 — Initialize Pipeline"):
                gr.Markdown("""
**What this does:** Generates 500 synthetic YouTube-like videos across 8 categories
and 7 languages, creates 200 users with realistic viewing profiles, generates ~7,000
interactions, then trains both the biased and fair recommender systems.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        init_btn = gr.Button("Run Pipeline", variant="primary")
                    with gr.Column(scale=3):
                        pass

                status_html  = gr.HTML(label="Data Overview")
                library_chart = gr.Image(label="Library Distribution", show_label=True)

            # ── Tab 2: Feed Simulator ───────────────────────────────────────
            with gr.Tab("2 — Feed Simulator"):
                gr.Markdown("""
**Select a user** and see what their feed looks like under the biased vs fair recommender.
Notice how the biased feed is dominated by Entertainment while the fair feed includes
Educational, Documentary, and non-English content.
                """)
                with gr.Row():
                    user_dd    = gr.Dropdown(
                        label="Select User", choices=[], scale=3,
                        info="Users have different profile types (student, professional, casual, etc.)"
                    )
                    feed_btn   = gr.Button("Simulate Feed", variant="primary", scale=1)

                feed_html  = gr.HTML(label="Side-by-Side Feed Comparison")
                feed_chart = gr.Image(label="Feed Composition Chart")

            # ── Tab 3: Bias Audit ───────────────────────────────────────────
            with gr.Tab("3 — Bias Audit Dashboard"):
                gr.Markdown("""
**Full bias audit** across all 200 users and 2000 recommendation rows.
Shows category and language representation gaps, Gini coefficient, and
all fairness metrics side-by-side.
                """)
                audit_btn = gr.Button("Run Full Audit", variant="primary")

                metrics_html = gr.HTML(label="Metrics Comparison")

                with gr.Row():
                    audit_c1 = gr.Image(label="Category Exposure")
                    audit_c2 = gr.Image(label="Language Exposure")
                with gr.Row():
                    audit_c3 = gr.Image(label="Lorenz Curve (Gini)")
                    audit_c4 = gr.Image(label="Summary Metrics")

            # ── Tab 4: Score Inspector ──────────────────────────────────────
            with gr.Tab("4 — Score Inspector"):
                gr.Markdown("""
**Select any video** to see exactly how the biased recommender degrades its score.
Shows the raw CF score, the suppression multipliers applied (category + language),
and how many users actually see this video in each recommender's output.
                """)
                with gr.Row():
                    video_dd    = gr.Dropdown(
                        label="Select Video", choices=[], scale=4,
                        info="Format: video_id | category | language | title"
                    )
                    inspect_btn = gr.Button("Inspect Score", variant="primary", scale=1)

                inspector_html  = gr.HTML(label="Score Analysis")
                inspector_chart = gr.Image(label="Score Waterfall & Loss Breakdown")

        # ── Event Wiring ────────────────────────────────────────────────────
        init_btn.click(
            fn=run_pipeline,
            outputs=[status_html, library_chart, user_dd, video_dd],
        )

        feed_btn.click(
            fn=simulate_feed,
            inputs=[user_dd],
            outputs=[feed_html, feed_chart],
        )

        audit_btn.click(
            fn=run_audit,
            outputs=[metrics_html, audit_c1, audit_c2, audit_c3, audit_c4],
        )

        inspect_btn.click(
            fn=inspect_video,
            inputs=[video_dd],
            outputs=[inspector_html, inspector_chart],
        )

    return app


if __name__ == '__main__':
    app = build_app()
    app.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=".tab-nav button { font-size: 14px !important; font-weight: 600 !important; }",
    )
