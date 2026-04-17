"""
test_llm_correction.py
----------------------
Runs the biased recommender then the LLM supervisor for a set of users
and measures whether the LLM actually fixes the bias.

Usage
-----
    python test_llm_correction.py                  # 5 representative users
    python test_llm_correction.py --all            # all 20 users
    python test_llm_correction.py --users U02 U04  # specific users
"""

import os, sys, io, json, argparse, time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))

# ── Constants ──────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GENRE_ENGAGEMENT = {
    "Entertainment": 1.00, "Music": 0.95, "Gaming": 0.92,
    "DIY": 0.58, "News/Analysis": 0.52, "Educational": 0.45,
    "Documentary": 0.40, "Regional": 0.33,
}
NON_ENGLISH_MULT  = 0.55
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

# 5 representative users covering different preference profiles
DEFAULT_USERS = ["U02", "U04", "U01", "U09", "U16"]
#  U02 Raj      — Educational/Documentary/News heavy
#  U04 Ahmed    — News/Documentary heavy
#  U01 Priya    — DIY/Documentary
#  U09 James    — Entertainment/Gaming (non-suppressed, control case)
#  U16 Ivan     — News/Regional, non-English preference

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    master_df = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), index_col="idx")
    dup       = master_df["video_id"].duplicated(keep="first")
    keep      = (~dup).values
    master_df = master_df[keep].reset_index(drop=True)
    emb       = np.load(os.path.join(DATA_DIR, "embeddings.npy"))[keep]
    with open(os.path.join(DATA_DIR, "users.json"), encoding="utf-8") as f:
        users = json.load(f)
    return master_df, emb, users

# ── Biased recommender (mirrors flask_app.py exactly) ─────────────────────────

def biased_recs(user, master_df, emb, top_n=30):
    watched  = {i["video_id"] for i in user.get("interactions", [])}
    df       = master_df[~master_df["video_id"].isin(watched)].copy()

    vid_pos  = {vid: i for i, vid in enumerate(master_df["video_id"])}
    its      = user.get("interactions", [])
    vecs, ws = [], []
    for it in its:
        pos = vid_pos.get(it["video_id"])
        if pos is not None:
            vecs.append(emb[pos]); ws.append(float(it["rating"]))
    if vecs:
        vecs = np.array(vecs); ws = np.array(ws) / sum(ws)
        uv   = (vecs * ws[:, None]).sum(0)
        norm = np.linalg.norm(uv)
        uv   = uv / norm if norm > 0 else uv
    else:
        uv = np.zeros(emb.shape[1])

    if np.linalg.norm(uv) == 0:
        pool = df[df["genre"].isin(user.get("preferred_genres", {}).keys())].copy()
        pool = pool if len(pool) else df
        pool["em"] = pool["genre"].map(GENRE_ENGAGEMENT).fillna(0.45)
        pool["lm"] = pool["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
        pool["s"]  = pool["virality_score"] * pool["em"] * pool["lm"]
        return pool.nlargest(top_n, "s").drop(columns=["em","lm","s"])

    sims = cosine_similarity([uv], emb[df.index])[0]
    df["em"]  = df["genre"].map(GENRE_ENGAGEMENT).fillna(0.45)
    df["lm"]  = df["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
    soft      = 0.5 + 0.5 * df["em"].values * df["lm"].values
    pen       = sims * soft;  pm = pen.max()
    df["pr"]  = pen / pm if pm > 0 else pen
    raw       = df["virality_score"] * df["em"] * df["lm"]; bm = raw.max()
    df["bc"]  = raw / bm if bm > 0 else raw
    df["s"]   = 0.50 * df["pr"] + 0.50 * df["bc"]
    return df.nlargest(top_n, "s").drop(columns=["em","lm","pr","bc","s"])

# ── Fairness scores (all-genre Gini) ──────────────────────────────────────────

def fairness(df, master_df):
    n = len(df)
    if n == 0:
        return {k: 0.0 for k in ["overall","diversity","suppressed_coverage",
                                  "representation","language_diversity"]}
    genres = df["genre"].str.strip()

    # Diversity — normalised Shannon entropy
    gc    = genres.value_counts().values.astype(float)
    probs = gc / gc.sum()
    div   = round(min(100.0,
                      float(-np.sum(probs * np.log2(probs + 1e-12)))
                      / np.log2(master_df["genre"].nunique()) * 100), 1)

    # Suppressed coverage
    lib_s = float(master_df["genre"].isin(SUPPRESSED_GENRES).mean())
    supp  = round(min(100.0, float(genres.isin(SUPPRESSED_GENRES).mean())
                      / max(lib_s, 0.01) * 100), 1)

    # Representation — Gini over ALL 8 genres
    all_g   = master_df["genre"].unique()
    fc      = genres.value_counts().reindex(all_g, fill_value=0).values.astype(float)
    cs      = np.sort(fc); m = len(cs); idx = np.arange(1, m+1)
    gini    = max(0.0, float((2*(idx*cs).sum())/(m*cs.sum()) - (m+1)/m))
    rep     = round((1.0 - gini) * 100, 1)

    # Language diversity
    lc    = df["language"].value_counts().values.astype(float)
    lp    = lc / lc.sum()
    l_div = round(min(100.0,
                      float(-np.sum(lp * np.log2(lp + 1e-12)))
                      / np.log2(master_df["language"].nunique()) * 100), 1)

    overall = round(0.30*div + 0.30*rep + 0.25*supp + 0.15*l_div, 1)
    return {"overall": overall, "diversity": div, "suppressed_coverage": supp,
            "representation": rep, "language_diversity": l_div}

# ── Helpers ────────────────────────────────────────────────────────────────────

def genre_mix(df):
    return dict(df["genre"].value_counts())

def bar(score, width=12):
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)

def delta_str(d):
    return f"{d:+.1f}"

SCORE_KEYS = [
    ("Overall",             "overall"),
    ("Genre Diversity",     "diversity"),
    ("Suppressed Coverage", "suppressed_coverage"),
    ("Representation",      "representation"),
    ("Language Diversity",  "language_diversity"),
]

# ── Main test ──────────────────────────────────────────────────────────────────

def run(user_ids, master_df, emb, user_map, supervisor):
    results = []  # list of result dicts, one per user

    for uid in user_ids:
        u = user_map.get(uid)
        if u is None:
            print(f"  [!] Unknown user ID: {uid}")
            continue

        print(f"\n{'─'*64}")
        print(f"  USER  {uid} · {u['name']}")
        print(f"  Prefs {dict(list(u['preferred_genres'].items())[:4])}")
        print(f"{'─'*64}")

        # ── Step 1: Biased recommender ─────────────────────────────────────
        t0     = time.time()
        b_df   = biased_recs(u, master_df, emb)
        b_time = time.time() - t0
        b_fs   = fairness(b_df, master_df)
        b_mix  = genre_mix(b_df)

        print(f"\n  [BIASED]   {b_time:.2f}s")
        print(f"  Genres : {b_mix}")
        for lbl, key in SCORE_KEYS:
            v = b_fs[key]
            print(f"  {lbl:<22} {v:5.1f}  [{bar(v)}]")

        # ── Step 2: LLM supervisor correction ─────────────────────────────
        print(f"\n  [LLM CORRECTED]  running…")
        t0    = time.time()
        try:
            c_df, reasoning, assessment = supervisor.fix(u, b_df)
            c_time = time.time() - t0
            c_fs   = fairness(c_df, master_df)
            c_mix  = genre_mix(c_df)
            llm_ok = True
        except Exception as exc:
            c_time = time.time() - t0
            print(f"  [ERROR] {exc}")
            results.append({"uid": uid, "name": u["name"], "error": str(exc)})
            continue

        print(f"  {c_time:.1f}s  |  bias detected: {assessment.is_biased}")
        print(f"  Genres : {c_mix}")
        for lbl, key in SCORE_KEYS:
            b_v = b_fs[key]; c_v = c_fs[key]
            d   = c_v - b_v
            sym = "▲" if d > 0 else ("▼" if d < 0 else "─")
            print(f"  {lbl:<22} {b_v:5.1f} → {c_v:5.1f}  [{bar(c_v)}]  {sym}{abs(d):.1f}")

        print(f"\n  Reasoning: {reasoning[:120]}{'…' if len(reasoning)>120 else ''}")

        results.append({
            "uid":         uid,
            "name":        u["name"],
            "pref_genres": list(u["preferred_genres"].keys()),
            "is_biased":   assessment.is_biased,
            "b_count":     len(b_df),
            "c_count":     len(c_df),
            "b_fs":        b_fs,
            "c_fs":        c_fs,
            "b_mix":       b_mix,
            "c_mix":       c_mix,
            "b_time":      b_time,
            "c_time":      c_time,
        })

    return results


def summary_and_assert(results):
    print(f"\n{'='*64}")
    print(f"  SUMMARY")
    print(f"{'='*64}")

    ok = [r for r in results if "error" not in r]
    if not ok:
        print("  No successful runs to summarise.")
        return False

    # Per-user overall score table
    print(f"\n  {'User':<18} {'Biased':>7} {'Corrected':>10} {'Δ':>6}  {'Bias?':>6}  {'Items':>6}")
    print(f"  {'─'*60}")
    for r in ok:
        d    = r["c_fs"]["overall"] - r["b_fs"]["overall"]
        tick = "✓" if d > 0 else "✗"
        print(f"  {tick} {r['uid']} {r['name']:<14}"
              f"  {r['b_fs']['overall']:>7.1f}"
              f"  {r['c_fs']['overall']:>10.1f}"
              f"  {d:>+6.1f}"
              f"  {'YES' if r['is_biased'] else 'no':>6}"
              f"  {r['c_count']:>6}")

    # Aggregate per metric
    print(f"\n  {'Metric':<22} {'Biased avg':>10} {'Corrected avg':>14} {'Avg Δ':>7}")
    print(f"  {'─'*58}")
    for lbl, key in SCORE_KEYS:
        b_vals = [r["b_fs"][key] for r in ok]
        c_vals = [r["c_fs"][key] for r in ok]
        b_avg  = float(np.mean(b_vals))
        c_avg  = float(np.mean(c_vals))
        d      = c_avg - b_avg
        sym    = "▲" if d > 0 else "▼"
        print(f"  {lbl:<22} {b_avg:>10.1f} {c_avg:>14.1f} {sym}{abs(d):>6.1f}")

    # ── Assertions ─────────────────────────────────────────────────────────
    print(f"\n  {'─'*64}")
    print(f"  ASSERTIONS")
    print(f"  {'─'*64}")
    passed = 0; failed = 0

    def check(name, cond, detail=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            print(f"  [✓] {name}")
        else:
            failed += 1
            print(f"  [✗] {name}")
            if detail:
                print(f"       {detail}")

    # 1. Every corrected list has exactly 30 items
    counts_ok = all(r["c_count"] == 30 for r in ok)
    check("Corrected list always has exactly 30 items",
          counts_ok,
          str({r["uid"]: r["c_count"] for r in ok if r["c_count"] != 30}))

    # 2. LLM detected bias for suppressed-pref users
    supp_users = [r for r in ok
                  if any(g in SUPPRESSED_GENRES
                         for g in r["pref_genres"][:2])]
    bias_detected = sum(1 for r in supp_users if r["is_biased"])
    check(f"Bias correctly detected for suppressed-pref users "
          f"({bias_detected}/{len(supp_users)})",
          bias_detected == len(supp_users) if supp_users else True)

    # 3. Overall fairness improves for every biased user
    # Users where no bias was detected correctly keep the original feed unchanged.
    biased_users = [r for r in ok if r.get("is_biased", True)]
    improved = [r for r in biased_users if r["c_fs"]["overall"] > r["b_fs"]["overall"]]
    check(f"LLM corrected overall fairness > biased for every biased user "
          f"({len(improved)}/{len(biased_users)})",
          len(improved) == len(biased_users),
          str({r["uid"]: f"{r['b_fs']['overall']:.1f}→{r['c_fs']['overall']:.1f}"
               for r in biased_users if r["c_fs"]["overall"] <= r["b_fs"]["overall"]}))

    # 4. Suppressed coverage improves for biased users
    supp_improved = [r for r in biased_users
                     if r["c_fs"]["suppressed_coverage"] > r["b_fs"]["suppressed_coverage"]]
    check(f"Suppressed coverage improves for biased users ({len(supp_improved)}/{len(biased_users)})",
          len(supp_improved) == len(biased_users),
          str({r["uid"]: f"{r['b_fs']['suppressed_coverage']:.1f}→{r['c_fs']['suppressed_coverage']:.1f}"
               for r in biased_users if r["c_fs"]["suppressed_coverage"] <= r["b_fs"]["suppressed_coverage"]}))

    # 5. Average overall improvement ≥ 20 points
    avg_delta = float(np.mean([r["c_fs"]["overall"] - r["b_fs"]["overall"] for r in ok]))
    check(f"Average overall fairness gain ≥ 20 points (got {avg_delta:+.1f})",
          avg_delta >= 20)

    # 6. No *corrected* feed dominated by a single genre (>50% slots).
    # Skip users where no bias was detected — their feed is deliberately unchanged.
    dominated = [r for r in ok
                 if r.get("is_biased", True) and max(r["c_mix"].values()) / 30 > 0.50]
    check(f"No corrected feed dominated by a single genre (>50% slots)",
          len(dominated) == 0,
          str({r["uid"]: r["c_mix"] for r in dominated}))

    print(f"\n  {'='*64}")
    print(f"  RESULT: {passed}/{passed+failed} assertions passed"
          + (f"  ({failed} failed)" if failed else "  — all green"))
    print(f"  {'='*64}\n")

    return failed == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",   action="store_true", help="Run all 20 users")
    parser.add_argument("--users", nargs="+",           help="Specific user IDs")
    args = parser.parse_args()

    print("\n" + "=" * 64)
    print("  Biased Recommender vs LLM Supervisor — Correction Test")
    print("=" * 64)

    print("  Loading data…", end=" ", flush=True)
    master_df, emb, users = load_data()
    user_map = {u["user_id"]: u for u in users}
    print(f"done  ({len(master_df):,} videos, {len(users)} users)")

    print("  Loading LLM supervisor…", end=" ", flush=True)
    from llm_supervisor import LLMSupervisor
    supervisor = LLMSupervisor(master_df, emb)
    print("done")

    if args.all:
        uids = sorted(user_map.keys())
    elif args.users:
        uids = args.users
    else:
        uids = DEFAULT_USERS

    print(f"\n  Testing {len(uids)} user(s): {uids}")

    results = run(uids, master_df, emb, user_map, supervisor)
    all_passed = summary_and_assert(results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
