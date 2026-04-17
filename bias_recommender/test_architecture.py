"""
test_architecture.py
--------------------
Architecture health tests for the Bias-Aware Video Recommender.
Covers data integrity, bias injection strength, fairness score maths,
user embedding quality, recommender contrast, and (optionally) the LLM supervisor.

Usage
-----
    python test_architecture.py           # all tests except LLM
    python test_architecture.py --llm     # include LLM supervisor (needs GROQ_API_KEY)
    python test_architecture.py --user U02 --llm   # single user with LLM
"""

import os, sys, json, argparse, time, textwrap, io
# Force UTF-8 output on Windows so box-drawing and tick chars render correctly.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(__file__))

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

GENRE_ENGAGEMENT = {
    "Entertainment": 1.00, "Music": 0.95, "Gaming": 0.92,
    "DIY": 0.58, "News/Analysis": 0.52, "Educational": 0.45,
    "Documentary": 0.40, "Regional": 0.33,
}
NON_ENGLISH_MULT  = 0.55
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

# Users whose primary preferences are in suppressed genres (≥ 50% weight suppressed)
SUPPRESSED_PREF_USERS = {"U01", "U02", "U04", "U06", "U08", "U11", "U12", "U15", "U16", "U17", "U18"}
# Users whose preferences are entirely un-suppressed genres
UNSUPPRESSED_PREF_USERS = {"U09", "U13", "U20"}

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data():
    master_df  = pd.read_csv(os.path.join(DATA_DIR, "master_dataset.csv"), index_col="idx")
    dup        = master_df["video_id"].duplicated(keep="first")
    keep       = (~dup).values
    master_df  = master_df[keep].reset_index(drop=True)
    embeddings = np.load(os.path.join(DATA_DIR, "embeddings.npy"))[keep]
    with open(os.path.join(DATA_DIR, "users.json"), encoding="utf-8") as f:
        users = json.load(f)
    return master_df, embeddings, users

# ── Replicated scoring functions (mirrors flask_app.py, standalone) ─────────────

def _user_embedding(user, master_df, embeddings):
    interactions = user.get("interactions", [])
    if not interactions:
        return np.zeros(embeddings.shape[1])
    vid_pos = {vid: i for i, vid in enumerate(master_df["video_id"])}
    vecs, weights = [], []
    for it in interactions:
        pos = vid_pos.get(it["video_id"])
        if pos is not None:
            vecs.append(embeddings[pos])
            weights.append(float(it["rating"]))
    if not vecs:
        return np.zeros(embeddings.shape[1])
    vecs    = np.array(vecs)
    weights = np.array(weights) / sum(weights)
    uv      = (vecs * weights[:, np.newaxis]).sum(axis=0)
    norm    = np.linalg.norm(uv)
    return uv / norm if norm > 0 else uv


def _biased_recs(user, master_df, embeddings, top_n=30):
    watched  = {i["video_id"] for i in user.get("interactions", [])}
    df       = master_df[~master_df["video_id"].isin(watched)].copy()
    uv       = _user_embedding(user, master_df, embeddings)

    if np.linalg.norm(uv) == 0:
        pool = df[df["genre"].isin(user.get("preferred_genres", {}).keys())].copy()
        pool = pool if len(pool) else df
        pool["em"] = pool["genre"].map(GENRE_ENGAGEMENT).fillna(0.45)
        pool["lm"] = pool["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
        pool["s"]  = pool["virality_score"] * pool["em"] * pool["lm"]
        return pool.nlargest(top_n, "s").drop(columns=["em", "lm", "s"])

    sims             = cosine_similarity([uv], embeddings[df.index])[0]
    df["em"]         = df["genre"].map(GENRE_ENGAGEMENT).fillna(0.45)
    df["lm"]         = df["language"].apply(lambda l: 1.0 if l == "English" else NON_ENGLISH_MULT)
    soft             = 0.5 + 0.5 * df["em"].values * df["lm"].values
    pen              = sims * soft
    pm               = pen.max()
    df["pr"]         = pen / pm if pm > 0 else pen
    raw              = df["virality_score"] * df["em"] * df["lm"]
    bm               = raw.max()
    df["bc"]         = raw / bm if bm > 0 else raw
    df["s"]          = 0.50 * df["pr"] + 0.50 * df["bc"]
    return df.nlargest(top_n, "s").drop(columns=["em", "lm", "pr", "bc", "s"])


def _unbiased_recs(user, master_df, embeddings, top_n=30):
    watched = {i["video_id"] for i in user.get("interactions", [])}
    df      = master_df[~master_df["video_id"].isin(watched)].copy()
    uv      = _user_embedding(user, master_df, embeddings)
    if np.linalg.norm(uv) == 0:
        pref = list(user.get("preferred_genres", {}).keys())
        pool = df[df["genre"].isin(pref)] if pref else df
        return pool.nlargest(top_n, "virality_score")
    df["s"] = cosine_similarity([uv], embeddings[df.index])[0]
    return df.nlargest(top_n, "s").drop(columns=["s"])


def _fairness(df, master_df):
    n = len(df)
    if n == 0:
        return {k: 0.0 for k in ["overall","diversity","suppressed_coverage","representation","language_diversity"]}
    genres = df["genre"].str.strip()
    gc     = genres.value_counts().values.astype(float)
    probs  = gc / gc.sum()
    ent    = float(-np.sum(probs * np.log2(probs + 1e-12)))
    div    = round(min(100.0, ent / np.log2(master_df["genre"].nunique()) * 100), 1)

    lib_s  = float(master_df["genre"].isin(SUPPRESSED_GENRES).mean())
    rec_s  = float(genres.isin(SUPPRESSED_GENRES).mean())
    supp   = round(min(100.0, rec_s / max(lib_s, 0.01) * 100), 1)

    all_genres  = master_df["genre"].unique()
    full_counts = genres.value_counts().reindex(all_genres, fill_value=0).values.astype(float)
    counts = np.sort(full_counts); m = len(counts); idx = np.arange(1, m+1)
    gini   = max(0.0, float((2*(idx*counts).sum())/(m*counts.sum()) - (m+1)/m))
    rep    = round((1.0 - gini) * 100, 1)

    lc     = df["language"].value_counts().values.astype(float)
    lp     = lc / lc.sum()
    l_div  = round(min(100.0, float(-np.sum(lp*np.log2(lp+1e-12))) / np.log2(master_df["language"].nunique()) * 100), 1)

    overall = round(0.30*div + 0.30*rep + 0.25*supp + 0.15*l_div, 1)
    return {"overall": overall, "diversity": div, "suppressed_coverage": supp,
            "representation": rep, "language_diversity": l_div}


# ── Test runner ────────────────────────────────────────────────────────────────

class TR:
    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; RESET = "\033[0m"; BOLD = "\033[1m"

    def __init__(self):
        self.results = []

    def check(self, section, name, cond, detail=""):
        self.results.append((section, name, bool(cond), detail))
        icon = f"{self.GREEN}✓{self.RESET}" if cond else f"{self.RED}✗{self.RESET}"
        print(f"  [{icon}] {name}")
        if detail:
            prefix = "       "
            for line in textwrap.wrap(str(detail), 72):
                print(f"{prefix}{line}")

    def info(self, msg):
        print(f"  {self.YELLOW}·{self.RESET} {msg}")

    def section(self, title):
        print(f"\n{self.BOLD}{'─'*60}{self.RESET}")
        print(f"{self.BOLD}  {title}{self.RESET}")
        print(f"{self.BOLD}{'─'*60}{self.RESET}")

    def summary(self):
        passed = sum(1 for *_, p, _ in self.results if p)
        failed = sum(1 for *_, p, _ in self.results if not p)
        total  = passed + failed
        print(f"\n{'='*60}")
        print(f"  {self.BOLD}RESULT: {passed}/{total} passed", end="")
        if failed:
            print(f"  ({self.RED}{failed} failed{self.RESET})", end="")
        print(f"  {self.RESET}")
        print(f"{'='*60}")
        if failed:
            print(f"  {self.RED}FAILURES:{self.RESET}")
            for sec, name, p, detail in self.results:
                if not p:
                    print(f"    ✗ [{sec}] {name}")
                    if detail:
                        print(f"          {detail}")
        print(f"{'='*60}\n")
        return failed == 0


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_data_integrity(tr, master_df, embeddings, users):
    tr.section("1 · DATA INTEGRITY")

    # Files and shapes
    tr.check("DATA", "master_dataset.csv exists",
             os.path.exists(os.path.join(DATA_DIR, "master_dataset.csv")))
    tr.check("DATA", "embeddings.npy exists",
             os.path.exists(os.path.join(DATA_DIR, "embeddings.npy")))
    tr.check("DATA", "users.json exists",
             os.path.exists(os.path.join(DATA_DIR, "users.json")))

    tr.check("DATA", "master_df rows match embeddings rows",
             len(master_df) == embeddings.shape[0],
             f"master_df={len(master_df):,}  embeddings={embeddings.shape[0]:,}")

    tr.check("DATA", "embeddings have expected dimension (384)",
             embeddings.shape[1] == 384,
             f"got {embeddings.shape[1]}")

    # Required columns
    required = {"video_id", "title", "genre", "language", "virality_score", "is_suppressed"}
    missing  = required - set(master_df.columns)
    tr.check("DATA", "all required columns present",
             len(missing) == 0, f"missing: {missing}" if missing else "")

    # Uniqueness
    dup_count = master_df["video_id"].duplicated().sum()
    tr.check("DATA", "video_id is unique",
             dup_count == 0, f"{dup_count} duplicates found")

    # No NaN in key columns
    for col in ["video_id", "genre", "language", "virality_score"]:
        nan_n = master_df[col].isna().sum()
        tr.check("DATA", f"no NaN in '{col}'",
                 nan_n == 0, f"{nan_n} NaN values" if nan_n else "")

    # Virality in [0, 1]
    vmin, vmax = master_df["virality_score"].min(), master_df["virality_score"].max()
    tr.check("DATA", "virality_score in [0, 1]",
             vmin >= 0 and vmax <= 1.0,
             f"range=[{vmin:.4f}, {vmax:.4f}]")

    # is_suppressed matches known suppressed genres
    expected = master_df["genre"].isin(SUPPRESSED_GENRES)
    mismatch  = (expected != master_df["is_suppressed"]).sum()
    tr.check("DATA", "is_suppressed flag consistent with genre list",
             mismatch == 0, f"{mismatch} rows have wrong flag" if mismatch else "")

    # User data
    tr.check("DATA", "exactly 20 user personas loaded",
             len(users) == 20, f"got {len(users)}")

    required_user_keys = {"user_id", "name", "preferred_genres", "preferred_languages", "interactions"}
    missing_keys = [u["user_id"] for u in users
                    if not required_user_keys.issubset(u.keys())]
    tr.check("DATA", "all user profiles have required keys",
             len(missing_keys) == 0,
             f"missing keys in: {missing_keys}" if missing_keys else "")

    lib_supp_pct = master_df["genre"].isin(SUPPRESSED_GENRES).mean() * 100
    lib_eng_pct  = (master_df["language"] == "English").mean() * 100
    tr.info(f"Library: {len(master_df):,} videos | "
            f"{lib_supp_pct:.1f}% suppressed | {lib_eng_pct:.1f}% English")


def test_user_embeddings(tr, master_df, embeddings, users):
    tr.section("2 · USER EMBEDDINGS (Two-Tower)")

    zero_vec = np.zeros(embeddings.shape[1])
    norms, sim_gaps = [], []

    for u in users:
        uv = _user_embedding(u, master_df, embeddings)
        has_history = len(u.get("interactions", [])) > 0

        norm = float(np.linalg.norm(uv))

        if has_history:
            norms.append(norm)
            # Cosine sim from user to preferred genre centroid vs random centroid
            pref_genres = list(u["preferred_genres"].keys())
            pref_mask   = master_df["genre"].isin(pref_genres)
            if pref_mask.sum() > 0:
                pref_centroid = embeddings[pref_mask].mean(axis=0)
                pref_centroid /= (np.linalg.norm(pref_centroid) + 1e-12)
                other_mask = ~pref_mask
                rand_centroid = embeddings[other_mask].mean(axis=0)
                rand_centroid /= (np.linalg.norm(rand_centroid) + 1e-12)
                sim_pref  = float(cosine_similarity([uv], [pref_centroid])[0][0])
                sim_other = float(cosine_similarity([uv], [rand_centroid])[0][0])
                sim_gaps.append(sim_pref - sim_other)

    all_nonzero = all(n > 0 for n in norms)
    tr.check("EMB", "all users with interactions have non-zero embeddings",
             all_nonzero,
             f"{sum(1 for n in norms if n == 0)} zero-norm embeddings" if not all_nonzero else "")

    near_unit = [abs(n - 1.0) < 0.01 for n in norms]
    tr.check("EMB", "user embeddings are unit-normalised (norm ≈ 1.0)",
             all(near_unit),
             f"norms range=[{min(norms):.4f}, {max(norms):.4f}]")

    if sim_gaps:
        avg_gap = float(np.mean(sim_gaps))
        frac_positive = sum(1 for g in sim_gaps if g > 0) / len(sim_gaps)
        tr.check("EMB",
                 "user vector is closer to preferred genre centroid than to other genres",
                 frac_positive >= 0.70,
                 f"{frac_positive*100:.0f}% of users have positive gap  (avg Δsim={avg_gap:+.4f})")


def test_bias_injection(tr, master_df, embeddings, users):
    tr.section("3 · BIAS INJECTION STRENGTH")

    user_map   = {u["user_id"]: u for u in users}
    lib_supp   = float(master_df["genre"].isin(SUPPRESSED_GENRES).mean())
    lib_eng    = float((master_df["language"] == "English").mean())

    supp_user_scores, unsupp_user_scores = [], []
    below_lib_count = 0

    tr.info("Running biased recommender on all users…")
    for uid in sorted(user_map):
        u    = user_map[uid]
        recs = _biased_recs(u, master_df, embeddings)
        if len(recs) == 0:
            continue

        rec_supp_rate = float(recs["genre"].isin(SUPPRESSED_GENRES).mean())
        fs = _fairness(recs, master_df)

        if uid in SUPPRESSED_PREF_USERS:
            supp_user_scores.append(fs["overall"])
            if rec_supp_rate < lib_supp:
                below_lib_count += 1
        elif uid in UNSUPPRESSED_PREF_USERS:
            unsupp_user_scores.append(fs["overall"])

    if supp_user_scores:
        avg_supp_fair = float(np.mean(supp_user_scores))
        tr.check("BIAS",
                 "biased recs score < 50 for suppressed-genre users",
                 avg_supp_fair < 50,
                 f"avg overall fairness = {avg_supp_fair:.1f}  (expect < 50)")

        tr.check("BIAS",
                 "suppressed-genre users get < library suppressed rate in biased feed",
                 below_lib_count >= len(supp_user_scores) * 0.7,
                 f"{below_lib_count}/{len(supp_user_scores)} users below library rate "
                 f"(lib={lib_supp*100:.1f}%)")

    # For entertainment users: biased feed should serve their preferred genres well
    # (preferred-genre hit rate measures whether the user's actual taste is served)
    if UNSUPPRESSED_PREF_USERS:
        unsupp_hit_rates, supp_hit_rates = [], []
        for uid in UNSUPPRESSED_PREF_USERS:
            if uid not in user_map:
                continue
            u    = user_map[uid]
            recs = _biased_recs(u, master_df, embeddings)
            pref = set(u["preferred_genres"].keys())
            unsupp_hit_rates.append(float(recs["genre"].isin(pref).mean()) * 100)
        for uid in SUPPRESSED_PREF_USERS:
            if uid not in user_map:
                continue
            u    = user_map[uid]
            recs = _biased_recs(u, master_df, embeddings)
            pref = set(u["preferred_genres"].keys())
            supp_hit_rates.append(float(recs["genre"].isin(pref).mean()) * 100)
        avg_unsupp_hit = float(np.mean(unsupp_hit_rates))
        avg_supp_hit   = float(np.mean(supp_hit_rates))
        tr.check("BIAS",
                 "entertainment users served their preferred genres more than suppressed-genre users",
                 avg_unsupp_hit > avg_supp_hit,
                 f"entertainment pref-hit={avg_unsupp_hit:.1f}%  "
                 f"suppressed-pref pref-hit={avg_supp_hit:.1f}%")


def test_fairness_score_maths(tr, master_df):
    tr.section("4 · FAIRNESS SCORE UNIT TESTS")

    def make_df(genre_list, lang_list=None):
        if lang_list is None:
            lang_list = ["English"] * len(genre_list)
        return pd.DataFrame({"genre": genre_list, "language": lang_list,
                              "video_id": [f"v{i}" for i in range(len(genre_list))]})

    all_genres = list(master_df["genre"].unique())

    # ── Edge case: empty DataFrame
    empty = make_df([])
    fs = _fairness(empty, master_df)
    tr.check("FAIR", "empty recommendation list → all scores = 0",
             all(v == 0.0 for v in fs.values()),
             str(fs))

    # ── All Entertainment (worst case for diversity)
    all_ent = make_df(["Entertainment"] * 30)
    fs_ent  = _fairness(all_ent, master_df)
    tr.check("FAIR", "all-Entertainment list → diversity ≈ 0",
             fs_ent["diversity"] < 5,
             f"diversity={fs_ent['diversity']}")
    tr.check("FAIR", "all-Entertainment list → suppressed_coverage ≈ 0",
             fs_ent["suppressed_coverage"] < 5,
             f"suppressed_coverage={fs_ent['suppressed_coverage']}")
    tr.check("FAIR", "all-Entertainment list → overall < 40",
             fs_ent["overall"] < 40,
             f"overall={fs_ent['overall']}")

    # ── Perfectly uniform genre distribution (best case)
    uniform_genres = (all_genres * 4)[:len(all_genres) * 4]
    uniform_langs  = ["English", "Hindi", "Spanish", "French",
                      "German", "Japanese", "Korean", "Russian"] * len(all_genres)
    uniform_df     = make_df(uniform_genres[:30], uniform_langs[:30])
    fs_uni         = _fairness(uniform_df, master_df)
    tr.check("FAIR", "uniform genre+language distribution → diversity > 85",
             fs_uni["diversity"] > 85,
             f"diversity={fs_uni['diversity']}")
    tr.check("FAIR", "uniform distribution → overall > 60",
             fs_uni["overall"] > 60,
             f"overall={fs_uni['overall']}")

    # ── Monotonicity: mixed scores higher than homogeneous
    mixed_genres = (["Educational", "Documentary", "Entertainment",
                     "Music", "Gaming", "DIY"] * 5)[:30]
    fs_mix = _fairness(make_df(mixed_genres), master_df)
    tr.check("FAIR", "mixed feed scores higher than all-Entertainment",
             fs_mix["overall"] > fs_ent["overall"],
             f"mixed={fs_mix['overall']}  all-ent={fs_ent['overall']}")

    # ── Suppressed coverage proportional to suppressed rate
    all_supp = make_df(["Educational"] * 15 + ["Documentary"] * 15)
    fs_supp  = _fairness(all_supp, master_df)
    tr.check("FAIR", "all-suppressed feed → suppressed_coverage ≥ 100",
             fs_supp["suppressed_coverage"] >= 99.0,
             f"suppressed_coverage={fs_supp['suppressed_coverage']}")

    # ── Representation score: more genres present = less Gini inequality = higher score
    # (Gini is computed over all 8 library genres, so absent ones score 0 → high inequality)
    single  = _fairness(make_df(["Entertainment"] * 30), master_df)
    two_eq  = _fairness(make_df(["Entertainment"] * 15 + ["Music"] * 15), master_df)
    eight_eq_genres = (all_genres * 4)[:30]
    eight_eq = _fairness(make_df(eight_eq_genres), master_df)
    tr.check("FAIR", "representation increases: 1 genre < 2 genres < 8 genres",
             single["representation"] < two_eq["representation"] < eight_eq["representation"],
             f"1-genre={single['representation']}  2-genre={two_eq['representation']}  "
             f"8-genre={eight_eq['representation']}")


def test_recommender_contrast(tr, master_df, embeddings, users):
    tr.section("5 · BIASED vs UNBIASED CONTRAST")

    user_map = {u["user_id"]: u for u in users}
    improved_count = 0
    total_supp     = 0
    delta_scores   = []
    delta_supp_cov = []
    delta_pref_hit = []

    print()
    header = f"  {'User':<18} {'Biased':>7} {'Unbiased':>9} {'Δ':>6}  {'B-supp':>7} {'U-supp':>7}"
    print(header)
    print(f"  {'─'*60}")

    for uid in sorted(user_map):
        if uid not in SUPPRESSED_PREF_USERS:
            continue
        u      = user_map[uid]
        brecs  = _biased_recs(u, master_df, embeddings)
        urecs  = _unbiased_recs(u, master_df, embeddings)
        bfs    = _fairness(brecs, master_df)
        ufs    = _fairness(urecs, master_df)

        delta   = ufs["overall"] - bfs["overall"]
        b_supp  = float(brecs["genre"].isin(SUPPRESSED_GENRES).mean()) * 100
        u_supp  = float(urecs["genre"].isin(SUPPRESSED_GENRES).mean()) * 100

        # Preferred genre hit rate
        pref    = set(u["preferred_genres"].keys())
        b_pref  = float(brecs["genre"].isin(pref).mean()) * 100
        u_pref  = float(urecs["genre"].isin(pref).mean()) * 100

        delta_scores.append(delta)
        delta_supp_cov.append(ufs["suppressed_coverage"] - bfs["suppressed_coverage"])
        delta_pref_hit.append(u_pref - b_pref)

        if delta > 0:
            improved_count += 1
        total_supp += 1

        flag = "✓" if delta > 0 else "✗"
        print(f"  {flag} {uid} {u['name']:<14} {bfs['overall']:>7.1f} {ufs['overall']:>9.1f} "
              f"{delta:>+6.1f}  {b_supp:>6.1f}% {u_supp:>6.1f}%")

    print()

    # NOTE: pure cosine similarity (unbiased) is a preference-matching recommender,
    # NOT a distribution-fairness recommender.  Users who strongly prefer 1-2
    # suppressed genres (e.g. Emma: 65% DIY) get a mono-genre feed that scores low
    # on distribution metrics — sometimes lower than the biased feed which
    # accidentally mixes genres.  The LLM supervisor (Section 7) is the true
    # fairness fix.  Threshold is set at 75% to allow for these mono-preference users.
    if total_supp > 0:
        tr.check("REC",
                 "unbiased feed scores higher than biased for ≥ 75% of suppressed-genre users",
                 improved_count >= total_supp * 0.75,
                 f"{improved_count}/{total_supp} users improved  "
                 f"(avg Δ={np.mean(delta_scores):+.1f})")

    if delta_supp_cov:
        avg_sc_delta = float(np.mean(delta_supp_cov))
        tr.check("REC",
                 "unbiased suppressed_coverage score > biased for suppressed-genre users",
                 avg_sc_delta > 0,
                 f"avg Δsuppressed_coverage = {avg_sc_delta:+.1f}")

    if delta_pref_hit:
        avg_pref = float(np.mean(delta_pref_hit))
        tr.check("REC",
                 "unbiased feed surfaces preferred genres more than biased",
                 avg_pref > 0,
                 f"avg Δpreferred-genre hit-rate = {avg_pref:+.1f}pp")


def test_e2e_pipeline(tr, master_df, embeddings, users):
    tr.section("6 · END-TO-END PIPELINE  (all 20 users, no LLM)")

    user_map  = {u["user_id"]: u for u in users}
    rec_counts_ok = True
    all_deltas    = []
    all_biased_fs = []
    all_ubias_fs  = []

    print()
    header = f"  {'UID':<5} {'Name':<14} {'Biased':>7} {'Ideal':>7} {'Δ':>6}  Pref genres"
    print(header)
    print(f"  {'─'*70}")

    for uid in sorted(user_map):
        u     = user_map[uid]
        brecs = _biased_recs(u, master_df, embeddings)
        urecs = _unbiased_recs(u, master_df, embeddings)

        if len(brecs) != 30 or len(urecs) != 30:
            rec_counts_ok = False

        bfs = _fairness(brecs, master_df)
        ufs = _fairness(urecs, master_df)

        delta = ufs["overall"] - bfs["overall"]
        all_deltas.append(delta)
        all_biased_fs.append(bfs["overall"])
        all_ubias_fs.append(ufs["overall"])

        pref_str = ", ".join(list(u["preferred_genres"].keys())[:3])
        print(f"  {uid:<5} {u['name']:<14} {bfs['overall']:>7.1f} {ufs['overall']:>7.1f} "
              f"{delta:>+6.1f}  {pref_str}")

    print()
    tr.check("E2E", "every recommender returns exactly 30 videos",
             rec_counts_ok)

    avg_bias   = float(np.mean(all_biased_fs))
    avg_unbias = float(np.mean(all_ubias_fs))
    tr.check("E2E", "avg unbiased fairness > avg biased fairness across all users",
             avg_unbias > avg_bias,
             f"biased avg={avg_bias:.1f}  ideal avg={avg_unbias:.1f}  "
             f"Δ={avg_unbias - avg_bias:+.1f}")

    n_improved = sum(1 for d in all_deltas if d > 0)
    # 75% threshold: pure preference matching can produce narrow feeds for
    # mono-preference users, scoring lower than biased on distribution metrics.
    tr.check("E2E", "unbiased scores higher than biased for ≥ 75% of users",
             n_improved >= 15,
             f"{n_improved}/20 users improved")

    tr.info(f"Overall biased avg: {avg_bias:.1f}  |  Overall ideal avg: {avg_unbias:.1f}  "
            f"|  Avg improvement: {float(np.mean(all_deltas)):+.1f}")


def test_llm_supervisor(tr, master_df, embeddings, users, target_uid=None):
    tr.section("7 · LLM SUPERVISOR  (agentic correction)")

    from llm_supervisor import LLMSupervisor

    supervisor = LLMSupervisor(master_df, embeddings)
    user_map   = {u["user_id"]: u for u in users}

    # Test on suppressed-genre users; pick a subset to limit API calls
    test_uids = [target_uid] if target_uid else ["U02", "U04", "U06"]
    test_uids = [uid for uid in test_uids if uid in user_map]

    for uid in test_uids:
        u     = user_map[uid]
        brecs = _biased_recs(u, master_df, embeddings)
        bfs   = _fairness(brecs, master_df)

        print(f"\n  Running LLM correction for {uid} ({u['name']})…")
        t0 = time.time()
        try:
            corrected_df, reasoning, assessment = supervisor.fix(u, brecs)
        except Exception as exc:
            tr.check("LLM", f"{uid}: supervisor.fix() completed without error",
                     False, str(exc))
            continue
        elapsed = time.time() - t0

        cfs = _fairness(corrected_df, master_df)
        delta = cfs["overall"] - bfs["overall"]

        tr.check("LLM", f"{uid}: corrected list has exactly 30 items",
                 len(corrected_df) == 30,
                 f"got {len(corrected_df)}")

        tr.check("LLM", f"{uid}: bias correctly detected (is_biased=True)",
                 assessment.is_biased,
                 f"reasoning: {assessment.reasoning[:80]}")

        tr.check("LLM", f"{uid}: corrected overall fairness > biased",
                 cfs["overall"] > bfs["overall"],
                 f"biased={bfs['overall']:.1f}  corrected={cfs['overall']:.1f}  Δ={delta:+.1f}")

        tr.check("LLM", f"{uid}: corrected suppressed coverage > biased",
                 cfs["suppressed_coverage"] > bfs["suppressed_coverage"],
                 f"biased={bfs['suppressed_coverage']:.1f}  "
                 f"corrected={cfs['suppressed_coverage']:.1f}")

        genre_mix = corrected_df["genre"].value_counts().to_dict()
        tr.info(f"{uid} corrected in {elapsed:.1f}s | genres: {genre_mix}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Architecture health tests")
    parser.add_argument("--llm",  action="store_true", help="Include LLM supervisor tests")
    parser.add_argument("--user", default=None,        help="Restrict LLM test to one user ID")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Bias-Aware Recommender — Architecture Test Suite")
    print("=" * 60)
    print("  Loading data…", end=" ", flush=True)
    master_df, embeddings, users = load_data()
    print(f"done  ({len(master_df):,} videos, {len(users)} users)\n")

    tr = TR()

    test_data_integrity(tr,       master_df, embeddings, users)
    test_user_embeddings(tr,      master_df, embeddings, users)
    test_bias_injection(tr,       master_df, embeddings, users)
    test_fairness_score_maths(tr, master_df)
    test_recommender_contrast(tr, master_df, embeddings, users)
    test_e2e_pipeline(tr,         master_df, embeddings, users)

    if args.llm:
        test_llm_supervisor(tr, master_df, embeddings, users, target_uid=args.user)
    else:
        print("\n  (LLM supervisor tests skipped — pass --llm to enable)\n")

    tr.summary()


if __name__ == "__main__":
    main()
