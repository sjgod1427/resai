"""
app.py
──────
Bias-Aware Video Recommender — Demo

Flow:
  1. Select one of 20 user personas
  2. Click Run Test
  3. See what the biased recommender gave them
  4. See how the LLM supervisor detected the bias and corrected it
  5. Compare metrics

Run:  python app.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import gradio as gr

# ─── Bootstrap: build data files on first run ─────────────────────────────────

DATA_DIR     = os.path.join(os.path.dirname(__file__), "data")
MASTER_CSV   = os.path.join(DATA_DIR, "master_dataset.csv")
EMBEDDINGS   = os.path.join(DATA_DIR, "embeddings.npy")
USERS_JSON   = os.path.join(DATA_DIR, "users.json")

def _ensure_data():
    if not os.path.exists(MASTER_CSV):
        print("Building master dataset for the first time...")
        from dataset_builder import build
        build(output_dir=DATA_DIR)
    elif not os.path.exists(EMBEDDINGS):
        # CSV already built — just compute embeddings
        print("Computing sentence-transformer embeddings...")
        import numpy as np
        import pandas as pd
        from sentence_transformers import SentenceTransformer
        master = pd.read_csv(MASTER_CSV, index_col="idx")
        model  = SentenceTransformer("all-MiniLM-L6-v2")
        texts  = (
            master["title"].fillna("").astype(str)
            + " [" + master["genre"] + "]"
            + " [" + master["language"] + "]"
        ).tolist()
        embeddings = model.encode(texts, batch_size=512,
                                  show_progress_bar=True, convert_to_numpy=True)
        np.save(EMBEDDINGS, embeddings)
        print(f"Embeddings saved: {embeddings.shape} -> {EMBEDDINGS}")
    if not os.path.exists(USERS_JSON):
        print("Building user profiles for the first time...")
        from user_profiles import build_and_save
        build_and_save(output_dir=DATA_DIR)

_ensure_data()

# ─── Load everything once ─────────────────────────────────────────────────────

master_df  = pd.read_csv(MASTER_CSV, index_col="idx")
embeddings = np.load(EMBEDDINGS)

# Deduplicate and keep embeddings 1-to-1 with rows
_dup = master_df["video_id"].duplicated(keep="first")
if _dup.any():
    _keep      = (~_dup).values
    master_df  = master_df[_keep].reset_index(drop=True)
    embeddings = embeddings[_keep]
else:
    master_df  = master_df.reset_index(drop=True)

with open(USERS_JSON, "r", encoding="utf-8") as f:
    USERS = json.load(f)

USER_MAP = {u["user_id"]: u for u in USERS}

from llm_supervisor import LLMSupervisor
SUPERVISOR = LLMSupervisor(master_df, embeddings)

# ─── Biased recommender (content-based + platform bias multipliers) ────────────

# ── Platform engagement multipliers ──────────────────────────────────────────
# Mirrors how real platforms weight genre performance based on measured CTR /
# watch-time signals. Entertainment scores highest; slow content (docs, edu)
# gets organically deprioritised — not hard-suppressed.
GENRE_ENGAGEMENT = {
    "Entertainment": 1.00,
    "Music":         0.93,
    "Gaming":        0.87,
    "DIY":           0.76,
    "News/Analysis": 0.72,
    "Educational":   0.68,
    "Documentary":   0.64,
    "Regional":      0.61,
}
NON_ENGLISH_MULT  = 0.78   # smaller measured audience on English platform → lower CTR
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

from sklearn.metrics.pairwise import cosine_similarity as _cos_sim


def _user_embedding(user: dict) -> np.ndarray:
    """
    Two-tower user vector.
    Rating-weighted average of the embeddings of every video the user watched.
    Gives a dense representation of taste without any text encoding at query time.
    """
    interactions = user.get("interactions", [])
    if not interactions:
        return np.zeros(embeddings.shape[1])
    vid_to_pos = {vid: i for i, vid in enumerate(master_df["video_id"])}
    vecs, weights = [], []
    for item in interactions:
        pos = vid_to_pos.get(item["video_id"])
        if pos is None:
            continue
        vecs.append(embeddings[pos])
        weights.append(float(item["rating"]))
    if not vecs:
        return np.zeros(embeddings.shape[1])
    vecs    = np.array(vecs)
    weights = np.array(weights) / sum(weights)
    user_vec = (vecs * weights[:, np.newaxis]).sum(axis=0)
    norm = np.linalg.norm(user_vec)
    return user_vec / norm if norm > 0 else user_vec


def get_biased_recs(user: dict, top_n: int = 10) -> pd.DataFrame:
    """
    Two-tower recommender with realistic platform bias.

    User tower  : rating-weighted avg of watched-video embeddings.
    Item tower  : pre-computed sentence-transformer embeddings (embeddings.npy).
    Bias source : platform optimises for engagement (virality) over pure
                  user relevance, genre engagement rates naturally suppress
                  slow-consumption content, mild English-language preference.

    The bias is deliberate but proportionate — a real recommender, not a caricature.
    """
    df       = master_df.copy()
    user_vec = _user_embedding(user)

    # ── Correct retrieval (60%) — pure two-tower cosine similarity ─────────
    sims            = _cos_sim([user_vec], embeddings)[0]
    df["relevance"] = sims

    # ── Biased signal (40%) — platform engagement distortion ───────────────
    df["eng_mult"]  = df["genre"].map(GENRE_ENGAGEMENT).fillna(0.65)
    df["lang_mult"] = df["language"].apply(
        lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT
    )
    bias_raw = df["virality_score"] * df["eng_mult"] * df["lang_mult"]

    # Normalise bias component to [0, 1] so it blends on the same scale
    b_max = bias_raw.max()
    df["bias_component"] = bias_raw / b_max if b_max > 0 else bias_raw

    # Final score: 60% user relevance + 40% platform bias
    df["final_score"] = 0.60 * df["relevance"] + 0.40 * df["bias_component"]

    return df.nlargest(top_n, "final_score").drop(
        columns=["relevance", "eng_mult", "lang_mult", "bias_component", "final_score"]
    )


def get_unbiased_recs(user: dict, top_n: int = 10) -> pd.DataFrame:
    """
    Pure two-tower recommender — zero platform bias.

    Score = cosine_similarity(user_vec, item_vec) only.
    No virality weighting, no genre multipliers, no language penalties.
    The result is ranked purely by how closely each video matches the
    user's taste profile derived from their watch history.
    """
    df       = master_df.copy()
    user_vec = _user_embedding(user)
    sims     = _cos_sim([user_vec], embeddings)[0]
    df["score"] = sims
    return df.nlargest(top_n, "score").drop(columns=["score"])


# ─── UI helpers ───────────────────────────────────────────────────────────────

GENRE_COLORS = {
    "Entertainment": "#e74c3c", "Music": "#e67e22", "Gaming": "#f39c12",
    "Educational":   "#2980b9", "Documentary": "#27ae60", "DIY": "#8e44ad",
    "News/Analysis": "#16a085", "Regional": "#2c3e50",
}
LANG_FLAG = {
    "English": "🇬🇧", "Hindi": "🇮🇳", "Spanish": "🇪🇸", "French": "🇫🇷",
    "German": "🇩🇪", "Japanese": "🇯🇵", "Korean": "🇰🇷", "Russian": "🇷🇺",
}


def _video_card(row, rank: int) -> str:
    genre  = row["genre"]
    lang   = row["language"]
    title  = str(row["title"])[:55] + ("…" if len(str(row["title"])) > 55 else "")
    color  = GENRE_COLORS.get(genre, "#95a5a6")
    flag   = LANG_FLAG.get(lang, "🌐")
    supp   = ("&nbsp;<span style='font-size:10px;background:rgba(0,0,0,0.2);"
              "padding:1px 6px;border-radius:8px'>suppressed</span>"
              if row.get("is_suppressed") else "")
    return (
        f"<div style='background:{color};color:white;padding:9px 13px;"
        f"border-radius:8px;margin-bottom:6px;display:flex;"
        f"justify-content:space-between;align-items:center'>"
        f"<div><strong>#{rank}</strong>&nbsp; {title}{supp}</div>"
        f"<div style='font-size:11px;opacity:0.9;white-space:nowrap;margin-left:8px'>"
        f"{genre}&nbsp;|&nbsp;{flag}&nbsp;{lang}</div></div>"
    )


def _metrics_card(recs_df: pd.DataFrame) -> dict:
    total     = len(recs_df)
    suppressed = int(recs_df["is_suppressed"].sum())
    non_eng   = int((recs_df["language"] != "English").sum())
    genres    = recs_df["genre"].value_counts().to_dict()
    langs     = recs_df["language"].value_counts().to_dict()
    return {
        "suppressed": suppressed,
        "non_english": non_eng,
        "genres": genres,
        "langs": langs,
    }


def _metrics_html(b_metrics: dict, f_metrics: dict) -> str:
    def _bar(val, total, color):
        pct = val / total * 100 if total else 0
        return (f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<div style='flex:1;background:#eee;border-radius:4px;height:14px'>"
                f"<div style='width:{pct:.0f}%;background:{color};height:100%;"
                f"border-radius:4px'></div></div>"
                f"<span style='font-size:12px;color:#555;min-width:40px'>{val}/10</span></div>")

    rows = [
        ("Suppressed content", b_metrics["suppressed"], f_metrics["suppressed"], "#8e44ad"),
        ("Non-English content", b_metrics["non_english"], f_metrics["non_english"], "#16a085"),
    ]
    html = "<div style='font-family:sans-serif'>"
    html += "<h4 style='color:#2c3e50;margin:0 0 10px'>Bias Metrics</h4>"
    html += ("<div style='display:grid;grid-template-columns:160px 1fr 1fr;"
             "gap:6px;align-items:center;font-size:12px'>")
    html += ("<div></div>"
             "<div style='text-align:center;color:#e74c3c;font-weight:bold'>Biased</div>"
             "<div style='text-align:center;color:#27ae60;font-weight:bold'>LLM Corrected</div>")
    for label, b_val, f_val, color in rows:
        improved = f_val > b_val
        html += f"<div style='color:#555'>{label}</div>"
        html += f"<div>{_bar(b_val, 10, '#e74c3c')}</div>"
        badge = "<span style='color:#27ae60;font-size:11px'> improved</span>" if improved else ""
        html += f"<div>{_bar(f_val, 10, '#27ae60')}{badge}</div>"
    html += "</div></div>"
    return html


def _metrics_html_three(b: dict, f: dict, u: dict) -> str:
    """Three-way comparison: Biased | LLM-Corrected | Unbiased."""
    def _bar(val, color):
        pct = val / 10 * 100
        return (f"<div style='display:flex;align-items:center;gap:6px'>"
                f"<div style='flex:1;background:#eee;border-radius:4px;height:12px'>"
                f"<div style='width:{pct:.0f}%;background:{color};height:100%;"
                f"border-radius:4px'></div></div>"
                f"<span style='font-size:11px;color:#555;min-width:32px'>{val}/10</span></div>")

    def _genre_breakdown(m: dict) -> str:
        lines = []
        for g, cnt in sorted(m["genres"].items(), key=lambda x: -x[1]):
            col = GENRE_COLORS.get(g, "#95a5a6")
            lines.append(
                f"<span style='background:{col};color:white;padding:1px 7px;"
                f"border-radius:10px;font-size:10px;margin:2px;display:inline-block'>"
                f"{g} {cnt}</span>"
            )
        return "".join(lines)

    rows = [
        ("Suppressed content", b["suppressed"], f["suppressed"], u["suppressed"], "#8e44ad"),
        ("Non-English content", b["non_english"], f["non_english"], u["non_english"], "#16a085"),
    ]

    html = "<div style='font-family:sans-serif'>"
    html += "<h4 style='color:#2c3e50;margin:0 0 12px'>Three-Way Metrics Comparison</h4>"
    html += ("<div style='display:grid;grid-template-columns:150px 1fr 1fr 1fr;"
             "gap:8px;align-items:center;font-size:12px'>")
    html += ("<div></div>"
             "<div style='text-align:center;color:#e74c3c;font-weight:bold'>Biased</div>"
             "<div style='text-align:center;color:#27ae60;font-weight:bold'>LLM Corrected</div>"
             "<div style='text-align:center;color:#2980b9;font-weight:bold'>Ideal (No Bias)</div>")
    for label, bv, fv, uv, color in rows:
        html += f"<div style='color:#555'>{label}</div>"
        html += f"<div>{_bar(bv, '#e74c3c')}</div>"
        html += f"<div>{_bar(fv, '#27ae60')}</div>"
        html += f"<div>{_bar(uv, '#2980b9')}</div>"
    html += "</div>"

    # Genre breakdown row
    html += ("<div style='display:grid;grid-template-columns:150px 1fr 1fr 1fr;"
             "gap:8px;align-items:start;margin-top:12px;font-size:12px'>")
    html += "<div style='color:#555;padding-top:4px'>Genre mix</div>"
    for m in [b, f, u]:
        html += f"<div style='line-height:2'>{_genre_breakdown(m)}</div>"
    html += "</div></div>"
    return html


# ─── Core test function ───────────────────────────────────────────────────────

def load_user_card(user_id: str) -> str:
    if not user_id:
        return ""
    u = USER_MAP[user_id]

    def _genre_bar(g, w):
        color = GENRE_COLORS.get(g, "#999")
        pct   = int(w * 100)
        return (
            f"<div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>"
            f"<span style='width:110px;font-size:12px;color:#555'>{g}</span>"
            f"<div style='flex:1;background:#eee;border-radius:3px;height:10px'>"
            f"<div style='width:{pct}%;background:{color};"
            f"height:100%;border-radius:3px'></div></div>"
            f"<span style='font-size:11px;color:#888'>{pct}%</span></div>"
        )
    genre_bars = "".join(
        _genre_bar(g, w)
        for g, w in sorted(u["preferred_genres"].items(), key=lambda x: -x[1])
    )

    recent = u["interactions"][:5]
    def _history_item(item):
        color = GENRE_COLORS.get(item["genre"], "#999")
        flag  = LANG_FLAG.get(item["language"], "🌐")
        return (
            f"<div style='font-size:11px;color:#555;padding:2px 0'>"
            f"<span style='background:{color};"
            f"color:white;padding:1px 6px;border-radius:8px;font-size:10px'>"
            f"{item['genre']}</span>&nbsp;"
            f"{flag} {str(item['title'])[:55]}</div>"
        )
    history_html = "".join(_history_item(i) for i in recent)

    return f"""
    <div style='font-family:sans-serif;background:#f8f9fa;padding:16px;border-radius:10px'>
      <div style='display:flex;gap:14px;align-items:flex-start;flex-wrap:wrap'>
        <div style='background:{GENRE_COLORS.get(list(u["preferred_genres"].keys())[0],"#2c3e50")};
                    color:white;width:52px;height:52px;border-radius:50%;
                    display:flex;align-items:center;justify-content:center;
                    font-size:22px;font-weight:bold;flex-shrink:0'>
          {u['name'][0]}
        </div>
        <div style='flex:1;min-width:200px'>
          <div style='font-size:18px;font-weight:bold;color:#2c3e50'>{u['name']}
            <span style='font-size:12px;font-weight:normal;color:#888;margin-left:6px'>{u['user_id']}</span>
          </div>
          <div style='font-size:13px;color:#555;margin-top:4px;line-height:1.5'>{u['description']}</div>
        </div>
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:14px'>
        <div>
          <div style='font-size:12px;font-weight:bold;color:#2c3e50;margin-bottom:6px'>
            Genre Preferences</div>
          {genre_bars}
        </div>
        <div>
          <div style='font-size:12px;font-weight:bold;color:#2c3e50;margin-bottom:6px'>
            Sample Watch History</div>
          {history_html}
          <div style='font-size:11px;color:#aaa;margin-top:4px'>
            +{len(u["interactions"])-5} more interactions</div>
        </div>
      </div>
    </div>"""


def run_test(user_id: str):
    if not user_id:
        empty = "<p style='color:#999;font-style:italic'>Select a user first.</p>"
        return empty, empty, "", empty, empty, empty

    user = USER_MAP[user_id]

    # ── Three recommenders ─────────────────────────────────────────────────
    biased_df    = get_biased_recs(user, top_n=10)
    unbiased_df  = get_unbiased_recs(user, top_n=10)

    # ── LLM correction ────────────────────────────────────────────────────
    corrected_df, reasoning, assessment = SUPERVISOR.fix(user, biased_df)

    # ── Build feed HTML ───────────────────────────────────────────────────
    b_supp = int(biased_df["is_suppressed"].sum())
    f_supp = int(corrected_df["is_suppressed"].sum())

    biased_cards = "".join(
        _video_card(row, i + 1)
        for i, (_, row) in enumerate(biased_df.iterrows())
    )
    fair_cards = "".join(
        _video_card(row, i + 1)
        for i, (_, row) in enumerate(corrected_df.iterrows())
    )

    legend = "".join(
        f"<span style='background:{col};color:white;padding:2px 8px;"
        f"border-radius:10px;font-size:10px;margin:2px;display:inline-block'>{g}</span>"
        for g, col in GENRE_COLORS.items()
    )

    biased_html = f"""
    <div style='font-family:sans-serif'>
      <div style='background:#fdedec;padding:8px 14px;border-radius:8px;margin-bottom:10px'>
        <strong style='color:#e74c3c'>Biased Recommender</strong>
        <span style='font-size:12px;color:#c0392b;margin-left:8px'>
          {b_supp}/10 suppressed content &nbsp;|&nbsp;
          {int((biased_df['language']!='English').sum())}/10 non-English
        </span>
      </div>
      {biased_cards}
      <div style='margin-top:8px;font-size:11px;color:#aaa'>{legend}</div>
    </div>"""

    fair_html = f"""
    <div style='font-family:sans-serif'>
      <div style='background:#eafaf1;padding:8px 14px;border-radius:8px;margin-bottom:10px'>
        <strong style='color:#27ae60'>LLM-Corrected Feed</strong>
        <span style='font-size:12px;color:#1e8449;margin-left:8px'>
          {f_supp}/10 suppressed content &nbsp;|&nbsp;
          {int((corrected_df['language']!='English').sum())}/10 non-English
        </span>
      </div>
      {fair_cards}
    </div>"""

    # ── Reasoning box ─────────────────────────────────────────────────────
    bias_label = (
        "<span style='background:#e74c3c;color:white;padding:2px 10px;"
        "border-radius:10px;font-size:12px'>BIASED</span>"
        if assessment.is_biased else
        "<span style='background:#27ae60;color:white;padding:2px 10px;"
        "border-radius:10px;font-size:12px'>FAIR</span>"
    )

    tier_info = ""
    if assessment.is_biased:
        tier_info = (
            f"<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:8px'>"
            f"<span style='background:#3498db;color:white;padding:2px 8px;border-radius:8px;font-size:11px'>T1: {assessment.tier1_slots} slots</span>"
            f"<span style='background:#9b59b6;color:white;padding:2px 8px;border-radius:8px;font-size:11px'>T2: {assessment.tier2_slots} slots</span>"
            f"<span style='background:#e67e22;color:white;padding:2px 8px;border-radius:8px;font-size:11px'>T3: {assessment.tier3_slots} slots</span>"
            f"<span style='background:#e74c3c;color:white;padding:2px 8px;border-radius:8px;font-size:11px'>T4: {assessment.tier4_slots} slots</span>"
            f"</div>"
        )

    reasoning_html = f"""
    <div style='font-family:sans-serif;background:#f8f9fa;padding:14px;border-radius:10px'>
      <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px'>
        <strong style='color:#2c3e50'>LLM Verdict:</strong> {bias_label}
      </div>
      <div style='font-size:13px;color:#555;line-height:1.6'>{assessment.reasoning}</div>
      {tier_info}
      {"<hr style='border:none;border-top:1px solid #eee;margin:10px 0'>" if assessment.is_biased else ""}
      {"<div style='font-size:12px;color:#777'><strong>Correction reasoning:</strong><br>" + reasoning + "</div>" if assessment.is_biased and reasoning else ""}
    </div>"""

    # ── Unbiased feed HTML ────────────────────────────────────────────────
    u_supp = int(unbiased_df["is_suppressed"].sum())
    unbiased_cards = "".join(
        _video_card(row, i + 1)
        for i, (_, row) in enumerate(unbiased_df.iterrows())
    )
    unbiased_html = f"""
    <div style='font-family:sans-serif'>
      <div style='background:#eaf4fb;padding:8px 14px;border-radius:8px;margin-bottom:10px'>
        <strong style='color:#2980b9'>Ideal Recommender (Zero Bias)</strong>
        <span style='font-size:12px;color:#1a6fa8;margin-left:8px'>
          Pure cosine similarity &nbsp;|&nbsp;
          {u_supp}/10 suppressed content &nbsp;|&nbsp;
          {int((unbiased_df['language'] != 'English').sum())}/10 non-English
        </span>
      </div>
      <div style='background:#f0f8ff;border-left:3px solid #2980b9;padding:8px 12px;
                  border-radius:4px;margin-bottom:10px;font-size:12px;color:#555;
                  line-height:1.6'>
        <strong>How it works:</strong> Score&nbsp;=&nbsp;cosine_similarity(user_vec,&nbsp;item_vec).
        No virality weighting, no genre multipliers, no language penalties.
        The user embedding is a rating-weighted average of their watched-video
        embeddings — pure taste matching with zero platform distortion.
      </div>
      {unbiased_cards}
      <div style='margin-top:8px;font-size:11px;color:#aaa'>{legend}</div>
    </div>"""

    # ── Metrics ───────────────────────────────────────────────────────────
    metrics = _metrics_html(_metrics_card(biased_df), _metrics_card(corrected_df))
    metrics_three = _metrics_html_three(
        _metrics_card(biased_df),
        _metrics_card(corrected_df),
        _metrics_card(unbiased_df),
    )

    return biased_html, fair_html, reasoning_html, metrics, unbiased_html, metrics_three


# ─── Gradio UI ────────────────────────────────────────────────────────────────

USER_CHOICES = [
    (f"{u['name']}  —  {u['description'][:65]}…", u["user_id"])
    for u in USERS
]

with gr.Blocks(
    title="Bias-Aware Video Recommender",
    theme=gr.themes.Soft(primary_hue="blue"),
    css="""
    .gradio-container { max-width: 1200px !important; }
    #run-btn { font-size: 15px !important; font-weight: 600 !important; }
    """
) as app:

    gr.Markdown("""
# Bias-Aware Video Recommender
### Demonstrating how an LLM supervisor detects and corrects algorithmic bias
    """)

    # ── User selection ─────────────────────────────────────────────────────
    with gr.Row():
        user_dd = gr.Dropdown(
            choices=USER_CHOICES,
            label="Select a User",
            info="Each user has a distinct viewing history. The LLM judges whether the biased recommender served them fairly.",
            scale=4,
        )
        run_btn = gr.Button("Run Test", variant="primary", scale=1, elem_id="run-btn")

    user_card = gr.HTML(label="User Profile")

    # ── Tabbed results ─────────────────────────────────────────────────────
    with gr.Tabs():

        with gr.TabItem("Bias Analysis"):
            with gr.Row(equal_height=False):
                biased_out    = gr.HTML(label="Biased Recommender Output")
                corrected_out = gr.HTML(label="LLM-Corrected Output")
            reasoning_out = gr.HTML(label="LLM Reasoning")
            metrics_out   = gr.HTML(label="Bias Metrics")

        with gr.TabItem("Ideal Recommender"):
            gr.Markdown("""
### Pure Two-Tower Recommender — Zero Platform Bias
Recommendations ranked **only** by cosine similarity between the user's taste profile
and each video's embedding. No engagement weighting, no genre suppression, no language penalty.
Use this tab to see what the algorithm *would* recommend if it optimised for the user — not the platform.
            """)
            unbiased_out     = gr.HTML(label="Ideal Feed")
            metrics_three_out = gr.HTML(label="Three-Way Metrics Comparison")

    # ── Events ─────────────────────────────────────────────────────────────
    user_dd.change(fn=load_user_card, inputs=user_dd, outputs=user_card)

    run_btn.click(
        fn=run_test,
        inputs=user_dd,
        outputs=[biased_out, corrected_out, reasoning_out, metrics_out,
                 unbiased_out, metrics_three_out],
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        inbrowser=True,
    )
