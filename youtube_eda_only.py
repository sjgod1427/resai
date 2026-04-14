import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
import warnings, os

warnings.filterwarnings('ignore')

# load the dataset — latin1 handles special characters in channel names
df = pd.read_csv('Global YouTube Statistics.csv', encoding='latin1')

# some columns come in as strings, force them all to numbers
for col in ['video views', 'subscribers', 'uploads', 'highest_monthly_earnings',
            'lowest_monthly_earnings', 'video_views_for_the_last_30_days',
            'subscribers_for_last_30_days', 'created_year']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# drop rows missing the two columns we rely on most
df_clean = df.dropna(subset=['subscribers', 'video views']).copy()
df_clean['category'] = df_clean['category'].fillna('Unknown')
df_clean['Country']  = df_clean['Country'].fillna('Unknown')

# a few extra columns that come in handy later
df_clean['avg_views_per_video'] = df_clean['video views'] / df_clean['uploads'].replace(0, np.nan)
df_clean['engagement_proxy']    = (df_clean['video_views_for_the_last_30_days'] /
                                   df_clean['subscribers'].replace(0, np.nan)) * 100
df_clean['earning_range']       = df_clean['highest_monthly_earnings'] - df_clean['lowest_monthly_earnings']

os.makedirs('plots', exist_ok=True)

# colours used across all plots
BG      = '#FFFFFF'
RED     = '#CC0000'
DARK    = '#1a1a1a'
GREY    = '#555555'
ACCENT2 = '#E84040'
ACCENT3 = '#E8720C'
PASTEL  = ['#CC0000','#E84040','#E8720C','#F0A500','#F5C518',
           '#5BA85A','#2E86C1','#7D3C98','#E91E8C','#00897B',
           '#1565C0','#558B2F','#AD1457','#0097A7','#6A1B9A']


def style_axes(fig, axes):
    fig.patch.set_facecolor(BG)
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor('#F9F9F9')
        ax.tick_params(colors=DARK, labelsize=9)
        ax.xaxis.label.set_color(DARK)
        ax.yaxis.label.set_color(DARK)
        ax.title.set_color(DARK)
        for spine in ax.spines.values():
            spine.set_color('#CCCCCC')


def save_plot(name):
    path = f'plots/{name}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close()
    print(f'  Saved: {path}')
    return path


def gini(arr):
    # 0 = perfectly equal, 1 = one channel owns everything
    arr = np.sort(np.abs(arr[~np.isnan(arr)]))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr))


# plot 1 — subscriber distribution
# shown two ways: raw millions (left) and log scale (right)
# log scale is needed because channels like T-Series are so large they
# squash everyone else into one corner on the raw chart
print('Generating Plot 1 — Subscriber Distribution...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
style_axes(fig, axes)
fig.suptitle('Plot 1 — Subscriber Distribution', color=DARK, fontsize=14, fontweight='bold', y=1.01)

ax = axes[0]
vals = df_clean['subscribers'] / 1e6
ax.hist(vals, bins=40, color=RED, edgecolor='#333', alpha=0.85)
ax.axvline(vals.mean(),   color=ACCENT3,   lw=2, linestyle='--', label=f'Mean:   {vals.mean():.1f}M')
ax.axvline(vals.median(), color='#F0A500', lw=2, linestyle=':',  label=f'Median: {vals.median():.1f}M')
ax.set_xlabel('Subscribers (Millions)')
ax.set_ylabel('Number of Channels')
ax.set_title('Raw Subscriber Count')
ax.legend(framealpha=0.8, labelcolor=DARK)

ax = axes[1]
log_vals = np.log10(df_clean['subscribers'])
ax.hist(log_vals, bins=40, color=ACCENT2, edgecolor='#333', alpha=0.85)
ax.axvline(log_vals.mean(), color=ACCENT3, lw=2, linestyle='--',
           label=f'Mean: 10^{log_vals.mean():.2f}')
ax.set_xlabel('log10(Subscribers)')
ax.set_ylabel('Number of Channels')
ax.set_title('Log Scale (clearer view)')
ax.legend(framealpha=0.8, labelcolor=DARK)

plt.tight_layout()
save_plot('p1_subscriber_dist')


# plot 2 — which countries dominate?
# left: raw channel count per country | right: average subscribers per country
print('Generating Plot 2 — Geographic Distribution...')
country_stats = (df_clean[df_clean['Country'] != 'Unknown']
                 .groupby('Country')
                 .agg(channel_count=('Youtuber', 'count'),
                      avg_subs=('subscribers', 'mean'))
                 .sort_values('channel_count', ascending=False)
                 .head(15))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
style_axes(fig, axes)
fig.suptitle('Plot 2 — Geographic Distribution of YouTube Channels',
             color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
bar_colors = [RED if c == 'United States' else ACCENT2 if c == 'India' else '#888'
              for c in country_stats.index[::-1]]
bars = ax.barh(country_stats.index[::-1], country_stats['channel_count'][::-1],
               color=bar_colors, edgecolor='#ccc')
for bar, val in zip(bars, country_stats['channel_count'][::-1]):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va='center', color=GREY, fontsize=8)
ax.set_xlabel('Number of Channels')
ax.set_title('Channel Count by Country')

ax = axes[1]
avg_subs_m = country_stats['avg_subs'] / 1e6
bars2 = ax.barh(country_stats.index[::-1], avg_subs_m[::-1],
                color=PASTEL[:15][::-1], edgecolor='#ccc')
for bar, val in zip(bars2, avg_subs_m[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{val:.1f}M', va='center', color=GREY, fontsize=8)
ax.set_xlabel('Average Subscribers (Millions)')
ax.set_title('Avg Subscribers per Country')

plt.tight_layout()
save_plot('p2_country_dist')


# plot 3 — what content types dominate?
# three panels: share of channels, avg subscribers, avg total views
print('Generating Plot 3 — Category Analysis...')
cat_stats = (df_clean[df_clean['category'] != 'Unknown']
             .groupby('category')
             .agg(count=('Youtuber', 'count'),
                  avg_subs=('subscribers', 'mean'),
                  avg_views=('video views', 'mean'))
             .sort_values('count', ascending=False)
             .head(12))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
style_axes(fig, axes)
fig.suptitle('Plot 3 — Content Category Analysis', color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
wedges, texts, autotexts = ax.pie(cat_stats['count'], labels=None,
                                   colors=PASTEL[:len(cat_stats)],
                                   autopct='%1.1f%%', startangle=140, pctdistance=0.82)
for at in autotexts:
    at.set_color(DARK); at.set_fontsize(7)
ax.legend(wedges, cat_stats.index, loc='lower left', fontsize=7,
          framealpha=0.8, labelcolor=DARK, bbox_to_anchor=(-0.3, -0.1))
ax.set_title('Channel Count\nby Category')

ax = axes[1]
avg_s = cat_stats['avg_subs'] / 1e6
bars = ax.barh(cat_stats.index[::-1], avg_s[::-1], color=PASTEL[:len(cat_stats)][::-1])
for bar, v in zip(bars, avg_s[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}M', va='center', color=GREY, fontsize=8)
ax.set_xlabel('Avg Subscribers (M)')
ax.set_title('Avg Subscribers\nper Category')

ax = axes[2]
avg_v = cat_stats['avg_views'] / 1e9
bars3 = ax.barh(cat_stats.index[::-1], avg_v[::-1], color=PASTEL[3:3+len(cat_stats)][::-1])
for bar, v in zip(bars3, avg_v[::-1]):
    ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}B', va='center', color=GREY, fontsize=8)
ax.set_xlabel('Avg Total Views (Billions)')
ax.set_title('Avg Views\nper Category')

plt.tight_layout()
save_plot('p3_category')


# plot 4 — do more subscribers always mean more views?
# if yes, that confirms the rich-get-richer loop we're studying
print('Generating Plot 4 — Subscribers vs Views...')
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
style_axes(fig, axes)
fig.suptitle('Plot 4 — Subscribers vs Total Views (Popularity Bias)',
             color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
top_cats   = cat_stats.index[:8].tolist()
colors_map = {c: PASTEL[i] for i, c in enumerate(top_cats)}
for cat in top_cats:
    sub = df_clean[df_clean['category'] == cat]
    ax.scatter(sub['subscribers'] / 1e6, sub['video views'] / 1e9,
               alpha=0.6, s=25, color=colors_map[cat], label=cat)
ax.set_xlabel('Subscribers (Millions)')
ax.set_ylabel('Total Views (Billions)')
ax.set_title('By Category')
ax.legend(fontsize=7, framealpha=0.8, labelcolor=DARK, markerscale=1.5)

ax = axes[1]
x    = np.log10(df_clean['subscribers'].replace(0, np.nan))
y    = np.log10(df_clean['video views'].replace(0, np.nan))
mask = x.notna() & y.notna()
r, _ = stats.pearsonr(x[mask], y[mask])
ax.scatter(x, y, alpha=0.4, s=20, color=RED, edgecolors='none')
m, b   = np.polyfit(x[mask], y[mask], 1)
xline  = np.linspace(x[mask].min(), x[mask].max(), 100)
ax.plot(xline, m * xline + b, color=ACCENT3, lw=2, label=f'r = {r:.3f}')
ax.set_xlabel('log10(Subscribers)')
ax.set_ylabel('log10(Total Views)')
ax.set_title(f'Log-Log Scale  |  Pearson r = {r:.3f}')
ax.legend(framealpha=0.8, labelcolor=DARK)

plt.tight_layout()
save_plot('p4_subs_vs_views')


# plot 5 — how unequal are earnings across channels?
# lorenz curve bowing far from the diagonal = high inequality
# gini above 0.5 is considered very unequal
print('Generating Plot 5 — Earnings Inequality...')
earn        = df_clean['highest_monthly_earnings'].dropna()
earn_sorted = np.sort(earn.values)
cum_earn    = np.cumsum(earn_sorted) / earn_sorted.sum()
cum_pop     = np.arange(1, len(earn_sorted) + 1) / len(earn_sorted)
g           = gini(earn.values)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
style_axes(fig, axes)
fig.suptitle('Plot 5 — Earnings Inequality Among Channels',
             color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot(cum_pop, cum_earn, color=RED, lw=2.5, label=f'Lorenz Curve  |  GINI = {g:.3f}')
ax.plot([0, 1], [0, 1], '--', color=GREY, lw=1.5, label='Perfect Equality')
ax.fill_between(cum_pop, cum_earn, cum_pop, alpha=0.2, color=RED)
ax.set_xlabel('Cumulative % of Channels')
ax.set_ylabel('Cumulative % of Earnings')
ax.set_title('Lorenz Curve — Earnings Inequality')
ax.legend(framealpha=0.8, labelcolor=DARK)

ax = axes[1]
metrics    = ['subscribers', 'video views', 'highest_monthly_earnings', 'avg_views_per_video']
labels_g   = ['Subscribers', 'Total Views', 'Monthly Earnings', 'Views/Video']
ginis      = [gini(df_clean[m].dropna().values) for m in metrics]
bar_colors = [RED if g > 0.5 else ACCENT3 if g > 0.35 else '#4CAF50' for g in ginis]
bars = ax.bar(labels_g, ginis, color=bar_colors, edgecolor='#ccc', width=0.5)
for bar, g_val in zip(bars, ginis):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{g_val:.3f}', ha='center', color=DARK, fontsize=10, fontweight='bold')
ax.axhline(0.4, color=ACCENT3, linestyle='--', lw=1.5, label='High Inequality Threshold (0.4)')
ax.set_ylim(0, 0.9)
ax.set_ylabel('GINI Coefficient')
ax.set_title('GINI by Metric  (0=Equal, 1=Monopoly)')
ax.legend(framealpha=0.8, labelcolor=DARK, fontsize=8)

plt.tight_layout()
save_plot('p5_earnings_gini')


# plot 6 — geographic bias deep-dive
# the US has 4.2% of world population but dominates the top channels list
print('Generating Plot 6 — Geographic Bias...')
df_known    = df_clean[df_clean['Country'] != 'Unknown'].copy()
us_pct      = (df_known['Country'] == 'United States').mean() * 100
country_avg = (df_known.groupby('Country')['subscribers']
               .mean().sort_values(ascending=False).head(15))

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
style_axes(fig, axes)
fig.suptitle('Plot 6 — Geographic Bias in YouTube Algorithm',
             color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
ax.pie([us_pct, 100 - us_pct],
       labels=[f'United States\n{us_pct:.1f}%', f'Rest of World\n{100-us_pct:.1f}%'],
       colors=[RED, '#AAAAAA'], startangle=90,
       wedgeprops={'edgecolor': '#fff', 'linewidth': 2})
ax.set_title(f'US = {us_pct:.1f}% of Top Channels\n(US = only 4.2% of world population)')

ax = axes[1]
top_c     = df_known['Country'].value_counts().head(10)
bar_cols2 = [RED if c == 'United States' else ACCENT2 if c == 'India' else '#888'
             for c in top_c.index]
ax.bar(range(len(top_c)), top_c.values, color=bar_cols2, edgecolor='#ccc')
ax.set_xticks(range(len(top_c)))
ax.set_xticklabels(top_c.index, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Number of Channels')
ax.set_title('Top 10 Countries\nby Channel Count')

ax = axes[2]
bars = ax.barh(country_avg.index[::-1], country_avg.values[::-1] / 1e6,
               color=[RED if c == 'United States' else PASTEL[i]
                      for i, c in enumerate(country_avg.index[::-1])])
for bar, v in zip(bars, country_avg.values[::-1] / 1e6):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f'{v:.1f}M', va='center', color=GREY, fontsize=7)
ax.set_xlabel('Avg Subscribers (Millions)')
ax.set_title('Avg Subscribers\nby Country')

plt.tight_layout()
save_plot('p6_geo_bias')


# plot 7 — when were the top channels created?
# older channels tend to have way more subscribers — classic first-mover advantage
print('Generating Plot 7 — Channel Creation Timeline...')
year_data = df_clean[df_clean['created_year'].between(2005, 2023)].copy()
yearly    = (year_data.groupby('created_year')
             .agg(count=('Youtuber', 'count'), avg_subs=('subscribers', 'mean'))
             .reset_index())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
style_axes(fig, axes)
fig.suptitle('Plot 7 — Channel Creation Over Time', color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
ax.bar(yearly['created_year'], yearly['count'], color=RED, edgecolor='#ccc', alpha=0.85)
ax.set_xlabel('Year Channel Created')
ax.set_ylabel('Number of Top Channels')
ax.set_title('When Were Top Channels Created?')

ax = axes[1]
ax.plot(yearly['created_year'], yearly['avg_subs'] / 1e6,
        color=ACCENT3, lw=2.5, marker='o', markersize=5)
ax.fill_between(yearly['created_year'], yearly['avg_subs'] / 1e6, alpha=0.2, color=ACCENT3)
ax.set_xlabel('Year Channel Created')
ax.set_ylabel('Avg Subscribers (Millions)')
ax.set_title('Avg Subscribers by Creation Year\n(Older = More Subscribers?)')

plt.tight_layout()
save_plot('p7_timeline')


# plot 8 — do all categories earn equally?
# entertainment and music earn far more than education — the algorithm has a financial incentive
# to push certain content types over others
print('Generating Plot 8 — Earnings by Category...')
earn_by_cat = (df_clean[df_clean['category'] != 'Unknown']
               .groupby('category')
               .agg(avg_earn=('highest_monthly_earnings', 'mean'),
                    count=('Youtuber', 'count'))
               .query('count >= 10')
               .sort_values('avg_earn', ascending=True))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
style_axes(fig, axes)
fig.suptitle('Plot 8 — Earnings Disparity Across Content Categories',
             color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
colors_bar = [RED if v == earn_by_cat['avg_earn'].max() else
              '#4CAF50' if v == earn_by_cat['avg_earn'].min() else '#888'
              for v in earn_by_cat['avg_earn']]
bars = ax.barh(earn_by_cat.index, earn_by_cat['avg_earn'] / 1e3,
               color=colors_bar, edgecolor='#ccc')
for bar, v in zip(bars, earn_by_cat['avg_earn'] / 1e3):
    ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f'${v:.0f}K', va='center', color=GREY, fontsize=8)
ax.set_xlabel('Avg Monthly Earnings (Thousands $)')
ax.set_title('Average Earnings by Category')

ax = axes[1]
top8     = df_clean[df_clean['category'].isin(earn_by_cat.index[-8:])].copy()
cat_list = earn_by_cat.index[-8:].tolist()
data_box = [top8[top8['category'] == c]['highest_monthly_earnings'].dropna().values / 1e3
            for c in cat_list]
bp = ax.boxplot(data_box, patch_artist=True,
                medianprops=dict(color=RED, lw=2),
                whiskerprops=dict(color=GREY), capprops=dict(color=GREY),
                flierprops=dict(marker='o', color=RED, alpha=0.4, markersize=3))
for patch, col in zip(bp['boxes'], PASTEL[:8]):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax.set_xticklabels(cat_list, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Monthly Earnings (Thousands $)')
ax.set_title('Earnings Spread — Top 8 Categories')

plt.tight_layout()
save_plot('p8_earnings_cat')


# plot 9 — what actually drives the algorithm's ranking?
# spearman correlation tells us which features predict rank the most
# if subscribers and total views dominate, popularity bias is confirmed
print('Generating Plot 9 — Explainability...')
features = {
    'Subscribers':      'subscribers',
    'Total Views':      'video views',
    'Uploads':          'uploads',
    'Avg Views/Video':  'avg_views_per_video',
    'Monthly Earnings': 'highest_monthly_earnings'
}
target = df_clean['video_views_rank'].dropna()
corrs  = {}
for name, col in features.items():
    sub    = df_clean.loc[target.index, col].dropna()
    common = target.loc[sub.index]
    if len(sub) > 50:
        r, _ = stats.spearmanr(sub, common)
        corrs[name] = r

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
style_axes(fig, axes)
fig.suptitle("Plot 9 — What Drives YouTube's Algorithm? (Explainability)",
             color=DARK, fontsize=14, fontweight='bold')

ax = axes[0]
feat_names = list(corrs.keys())
feat_vals  = list(corrs.values())
cols_corr  = [RED if abs(v) > 0.5 else ACCENT3 if abs(v) > 0.3 else '#888' for v in feat_vals]
bars = ax.barh(feat_names, feat_vals, color=cols_corr, edgecolor='#ccc')
ax.axvline(0, color=GREY, lw=1)
for bar, v in zip(bars, feat_vals):
    ax.text(v + (0.02 if v >= 0 else -0.08),
            bar.get_y() + bar.get_height() / 2,
            f'{v:.3f}', va='center', color=DARK, fontsize=9)
ax.set_xlabel('Spearman Correlation with View Rank')
ax.set_title('Feature Importance for Ranking\n(|r| closer to 1 = stronger driver)')

ax = axes[1]
x    = np.log10(df_clean['subscribers'].replace(0, np.nan))
y    = np.log10(df_clean['video views'].replace(0, np.nan))
mask = x.notna() & y.notna() & df_clean['video_views_rank'].notna()
sc   = ax.scatter(x[mask], y[mask], c=df_clean.loc[mask, 'video_views_rank'],
                  cmap='RdYlGn_r', s=20, alpha=0.6)
plt.colorbar(sc, ax=ax, label='View Rank (lower = better)')
ax.set_xlabel('log10(Subscribers)')
ax.set_ylabel('log10(Total Views)')
ax.set_title('Rank Coloured by Algorithm Score\n(green = high ranked)')

plt.tight_layout()
save_plot('p9_explainability')


# plot 10 — summary dashboard with the 6 key numbers from the whole analysis
print('Generating Plot 10 — Summary Dashboard...')
fig = plt.figure(figsize=(14, 8), facecolor=BG)
fig.suptitle('Plot 10 — Responsible AI Audit Summary Dashboard',
             color=DARK, fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

kpis = [
    ('Total Channels',  str(len(df_clean)),                                              DARK),
    ('US Dominance',    f'{us_pct:.1f}%',                                                RED),
    ('GINI (Earnings)', f'{gini(df_clean["highest_monthly_earnings"].dropna().values):.3f}', ACCENT3),
    ('Categories',      str(df_clean['category'].nunique()),                             '#2E7D32'),
    ('Countries',       str(df_known['Country'].nunique()),                              '#1565C0'),
    ('Missing Country', f'{df["Country"].isna().mean()*100:.1f}%',                      ACCENT2),
]
positions = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]

for (row, col), (label, val, col_) in zip(positions, kpis):
    ax = fig.add_subplot(gs[row, col])
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.65, val, ha='center', va='center', fontsize=28,
            fontweight='bold', color=col_, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=10,
            color=GREY, transform=ax.transAxes)
    rect = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                           boxstyle='round,pad=0.02',
                           linewidth=2, edgecolor=col_, facecolor='#F5F5F5',
                           transform=ax.transAxes)
    ax.add_patch(rect)

save_plot('p10_dashboard')

print('\nAll 10 plots saved in the /plots folder.')