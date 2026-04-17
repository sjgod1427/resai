"""
llm_supervisor.py
-----------------
Two-step bias correction — no agent loop, no tool calls.

Step 1  JUDGE  : single structured LLM call → BiasAssessment
Step 2  SELECT : pre-fetch ~50 candidates in pure Python, then rule-based
                 tier-balanced selection using the judge's slot allocations.
"""

import os
import re
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import List
from collections import Counter
from dotenv import load_dotenv

# Force line-buffered stdout so logs appear immediately in the terminal
# (Python buffers stdout when not run in a TTY, e.g. on Windows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator, model_validator

load_dotenv()

# ── Logger setup ────────────────────────────────────────────────────────────────

logger = logging.getLogger("supervisor")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()

fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

# StreamHandler that flushes after every record so logs appear immediately
class _FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

ch = _FlushingStreamHandler(sys.stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

# FileHandler so logs are also written to supervisor.log
_log_path = os.path.join(os.path.dirname(__file__), "supervisor.log")
fh = logging.FileHandler(_log_path, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)

# ── Config ──────────────────────────────────────────────────────────────────────

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

# ── Pydantic schemas ─────────────────────────────────────────────────────────────

class BiasAssessment(BaseModel):
    is_biased:    bool = Field(description="True if algorithmic suppression is hurting this user")
    needs_fixing: bool = Field(default=False, description="True if a highly-weighted preferred genre is absent but no algorithmic suppression is detected")
    reasoning: str  = Field(default="", description="Why the output is or is not biased / needs fixing")
    genres_over_represented: List[str] = Field(default_factory=list, description="Genres appearing too many times")
    genres_missing: List[str]          = Field(default_factory=list, description="Genres that should appear but don't")
    tier1_slots: int = Field(default=12, description="Slots for T1 (user's preferred genres)", ge=0, le=30)
    tier2_slots: int = Field(default=8,  description="Slots for T2 (suppressed genres user watches)", ge=0, le=30)
    tier3_slots: int = Field(default=6,  description="Slots for T3 (diversity)", ge=0, le=30)
    tier4_slots: int = Field(default=4,  description="Slots for T4 (global viral overlay)", ge=0, le=30)

    @model_validator(mode="before")
    @classmethod
    def _normalise_fields(cls, data):
        if not isinstance(data, dict):
            return data
        # LLM sometimes uses 'biased' instead of 'is_biased'
        if "biased" in data and "is_biased" not in data:
            data["is_biased"] = data.pop("biased")
        # LLM sometimes nests slots as {"tier_slots": {"T1": n, ...}}
        if "tier_slots" in data and isinstance(data["tier_slots"], dict):
            ts = data.pop("tier_slots")
            data.setdefault("tier1_slots", ts.get("T1", 12))
            data.setdefault("tier2_slots", ts.get("T2", 8))
            data.setdefault("tier3_slots", ts.get("T3", 6))
            data.setdefault("tier4_slots", ts.get("T4", 4))
        return data


class _RowSelection(BaseModel):
    row_numbers: List[int] = Field(
        description="Exactly 30 unique 1-based row numbers from the candidate table"
    )
    reasoning: str = Field(description="Brief explanation: 'X T1 + Y T2 + Z T3 = 30'")

    @field_validator("reasoning", mode="before")
    @classmethod
    def _coerce_reasoning(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        return str(v) if v is not None else ""


# ── Supervisor ───────────────────────────────────────────────────────────────────

class LLMSupervisor:

    def __init__(self, master_df: pd.DataFrame, embeddings: np.ndarray):
        df = master_df.reset_index(drop=True)
        dup_mask = df["video_id"].duplicated(keep="first")
        if dup_mask.any():
            n_dups = int(dup_mask.sum())
            logger.info("Dropping %d duplicate video_id rows (%d -> %d unique).",
                        n_dups, len(df), len(df) - n_dups)
            keep       = (~dup_mask).values
            df         = df[keep].reset_index(drop=True)
            embeddings = embeddings[keep]

        self.master_df  = df
        self.embeddings = embeddings
        self._api_key   = os.getenv("GROQ_API_KEY")
        self._model_idx = 0
        logger.info("LLMSupervisor ready | model=%s | videos=%d",
                    GROQ_MODELS[self._model_idx], len(self.master_df))

    # ── LLM helpers ─────────────────────────────────────────────────────────────

    def _make_llm(self, idx: int = 0) -> ChatGroq:
        return ChatGroq(model=GROQ_MODELS[idx], temperature=0,
                        groq_api_key=self._api_key)

    def _call_llm(self, messages, structured_schema=None):
        schema_name = structured_schema.__name__ if structured_schema else "plain"
        for idx in range(len(GROQ_MODELS)):
            model = GROQ_MODELS[idx]
            llm   = self._make_llm(idx)
            logger.info("LLM call | model=%s | schema=%s | attempt=%d",
                        model, schema_name, idx + 1)
            try:
                if structured_schema:
                    # Use json_mode — more reliable on Groq than tool_use
                    result = llm.with_structured_output(
                        structured_schema, method="json_mode"
                    ).invoke(messages)
                else:
                    result = llm.invoke(messages)
                logger.info("LLM call succeeded | model=%s | schema=%s", model, schema_name)
                self._model_idx = idx
                return result
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err.lower():
                    retry_match = re.search(r"try again in\s+([\d.]+\s*\w+)", err, re.IGNORECASE)
                    retry_hint  = retry_match.group(1) if retry_match else "unknown"
                    if idx + 1 < len(GROQ_MODELS):
                        logger.warning(
                            "RATE LIMIT on %s (retry in %s) — switching to %s",
                            model, retry_hint, GROQ_MODELS[idx + 1]
                        )
                        time.sleep(2)
                        continue
                    else:
                        logger.error(
                            "RATE LIMIT on %s (retry in %s) — no more fallback models.",
                            model, retry_hint
                        )
                        raise
                else:
                    logger.error("LLM error on %s: %s: %s", model, type(e).__name__, err[:300])
                    raise

    # ══════════════════════════════════════════════════════════════════════════════
    # Public entry point
    # ══════════════════════════════════════════════════════════════════════════════

    def fix(self, user: dict, biased_recs: pd.DataFrame, feedback_history: list = None):
        """Returns (corrected_df, reasoning, assessment). corrected_df always has 30 rows."""
        uid = user.get("user_id", "?")
        logger.info("─── fix() START | user=%s (%s)", uid, user.get("name", ""))

        assessment = self._judge(user, biased_recs, feedback_history)

        # Hard override: if a genre the user weights ≥0.25 is completely absent, flag needs_fixing
        pref_genres = user.get("preferred_genres", {})
        if pref_genres and not assessment.is_biased:
            rec_genres = set(biased_recs["genre"].tolist())
            absent_high_weight = [
                g for g, w in pref_genres.items()
                if float(w) >= 0.25 and g not in rec_genres
            ]
            if absent_high_weight:
                assessment.needs_fixing = True
                for g in absent_high_weight:
                    if g not in assessment.genres_missing:
                        assessment.genres_missing.insert(0, g)
                if not assessment.reasoning or "preferred" not in assessment.reasoning.lower():
                    assessment.reasoning = (
                        f"Preferred genre(s) {absent_high_weight} with significant weight "
                        f"are absent from recommendations. "
                        + assessment.reasoning
                    )
                # For preference-mismatch: push T1 up, zero out T2 (not a suppression issue)
                assessment.tier1_slots = 20
                assessment.tier2_slots = 0
                assessment.tier3_slots = 6
                assessment.tier4_slots = 4
                logger.info(
                    "Hard override: genre(s) %s (weight≥0.25) missing → needs_fixing=True | user=%s",
                    absent_high_weight, uid,
                )

        if feedback_history and not assessment.is_biased and not assessment.needs_fixing:
            assessment.is_biased = True
            assessment.reasoning = f"Feedback-driven correction applied. {assessment.reasoning}"
            logger.info("Feedback override: forcing is_biased=True for user=%s", uid)

        if not assessment.is_biased and not assessment.needs_fixing:
            logger.info("No bias or preference mismatch detected for user=%s — returning original recs.", uid)
            return biased_recs, assessment.reasoning, assessment

        corrected_ids, reasoning = self._correct(user, biased_recs, assessment, feedback_history)

        corrected_df = (
            self.master_df[self.master_df["video_id"].isin(corrected_ids)]
            .drop_duplicates("video_id")
            .copy()
        )
        order = {vid: i for i, vid in enumerate(corrected_ids)}
        corrected_df["_order"] = corrected_df["video_id"].map(order)
        corrected_df = corrected_df.sort_values("_order").drop(columns=["_order"])

        genre_mix = dict(corrected_df["genre"].value_counts())
        logger.info("fix() DONE | user=%s | corrected=%d rows | genres=%s | reasoning='%s'",
                    uid, len(corrected_df), genre_mix, reasoning)
        return corrected_df, reasoning, assessment

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 1 — Judge
    # ══════════════════════════════════════════════════════════════════════════════

    def _judge(self, user: dict, biased_recs: pd.DataFrame, feedback_history: list = None) -> BiasAssessment:
        pref_genre_set = set(user.get("preferred_genres", {}).keys())
        history_genres = Counter(i["genre"] for i in user.get("interactions", []))
        user_suppressed = [
            g for g in SUPPRESSED_GENRES
            if g in pref_genre_set or history_genres.get(g, 0) >= 3
        ]

        feedback_ctx = ""
        if feedback_history:
            fb_lines = "\n".join(f"  Round {i+1}: {f}" for i, f in enumerate(feedback_history))
            feedback_ctx = (
                f"\n\nSESSION FEEDBACK — USER'S EXPLICIT REQUESTS (MANDATORY):\n{fb_lines}\n"
                "Treat any violation as BIAS. Allocate tier slots to address these items."
            )

        system = SystemMessage(content=f"""You are a fairness auditor for a video recommendation system.
Assign one of THREE verdicts for this user's recommendations.{feedback_ctx}

── VERDICT DEFINITIONS ──────────────────────────────────────────────────────

BIASED (is_biased=true, needs_fixing=false):
  The engagement-scoring algorithm suppresses genres the user actually wants.
  Suppressed genres = Educational, Documentary, DIY, News/Analysis, Regional.
  Flag when:
  1. A suppressed genre the user watches/prefers is absent or heavily under-represented.
  2. High-engagement genres flood the feed, crowding out the user's preferred suppressed content.
  3. User prefers non-English content but feed is mostly English due to the language multiplier.

NEEDS FIXING (is_biased=false, needs_fixing=true):
  No algorithmic suppression, but the feed doesn't match the user's stated preference weights.
  Flag when a genre the user explicitly weights at ≥0.25 is completely absent from recommendations,
  even if that genre is NOT a suppressed genre (e.g. Gaming weighted at 0.5 but gets 0 videos).
  This is a preference-mismatch problem, not a platform-bias problem.

FAIR (is_biased=false, needs_fixing=false):
  Feed accurately reflects user preferences. No correction needed.

── RULES ────────────────────────────────────────────────────────────────────

- NOT BIASED if the user genuinely prefers Music/Entertainment/Gaming and the feed reflects that.
- NEEDS FIXING takes priority over FAIR whenever a high-weight preferred genre is missing.
- Only one of is_biased / needs_fixing can be true at a time.

This user's suppressed genres of interest: {user_suppressed or '(none)'}

If is_biased=true OR needs_fixing=true, allocate tier slots (T1+T2+T3+T4 must sum to 30):
  T1 user's preferred/watched genres                ~12  (raise to ~20 for NEEDS FIXING)
  T2 suppressed genres the user watches             ~8   (0 for NEEDS FIXING)
  T3 diversity outside user's bubble                ~6
  T4 globally viral overlay                         ~4

Return ONLY valid JSON with EXACTLY these keys:
{{
  "is_biased": false,
  "needs_fixing": false,
  "reasoning": "...",
  "genres_over_represented": ["..."],
  "genres_missing": ["..."],
  "tier1_slots": 12,
  "tier2_slots": 8,
  "tier3_slots": 6,
  "tier4_slots": 4
}}""")

        human = HumanMessage(content=f"""USER PROFILE
Name: {user['name']}
Description: {user['description']}
Preferred genres: {json.dumps(user['preferred_genres'])}
Preferred languages: {user['preferred_languages']}

WATCH HISTORY
{self._summarise_history(user)}

BIASED RECOMMENDER OUTPUT
{self._summarise_recs(biased_recs)}

Assess bias. Return JSON schema.""")

        uid = user.get("user_id", "?")
        logger.info("Judge prompt built for user=%s | suppressed_interest=%s", uid, user_suppressed)
        try:
            result = self._call_llm([system, human], structured_schema=BiasAssessment)
            logger.info(
                "Judge result | user=%s | is_biased=%s | missing=%s | over_rep=%s | "
                "slots T1=%d T2=%d T3=%d T4=%d",
                uid, result.is_biased, result.genres_missing, result.genres_over_represented,
                result.tier1_slots, result.tier2_slots, result.tier3_slots, result.tier4_slots,
            )
            logger.info("Judge reasoning | user=%s | %s", uid, result.reasoning)
            return result
        except Exception as e:
            logger.error("Judge LLM failed for user=%s | error=%s: %s", uid, type(e).__name__, e)
            rec_genres = set(biased_recs["genre"].tolist())
            missing    = [g for g in pref_genre_set if g not in rec_genres]
            over_rep   = [g for g in rec_genres if g not in pref_genre_set and g not in SUPPRESSED_GENRES]
            t2 = len(user_suppressed) * 2 if user_suppressed else 0
            t1 = 12 + (8 - t2)
            return BiasAssessment(
                is_biased=bool(missing or over_rep),
                needs_fixing=False,
                reasoning="LLM assessment failed; applying default correction.",
                genres_over_represented=over_rep, genres_missing=missing,
                tier1_slots=t1, tier2_slots=t2, tier3_slots=6, tier4_slots=4,
            )

    # ══════════════════════════════════════════════════════════════════════════════
    # Step 2 — Correct: pre-fetch → single LLM selection → rule-based fallback
    # ══════════════════════════════════════════════════════════════════════════════

    def _correct(
        self, user: dict, biased_recs: pd.DataFrame,
        assessment: BiasAssessment, feedback_history: list = None,
    ):
        uid         = user.get("user_id", "?")
        watched_ids = {i["video_id"] for i in user.get("interactions", [])}
        biased_ids  = set(biased_recs["video_id"].tolist())
        prefetched  = self._prefetch_candidates(user, assessment, watched_ids | biased_ids)

        pool_df     = self.master_df[self.master_df["video_id"].isin(prefetched)]
        pool_genres = pool_df["genre"].str.strip()
        pool_mix    = dict(pool_genres.value_counts())
        pref_set    = set(user["preferred_genres"].keys())
        avail_t3    = int((~pool_genres.isin(pref_set) & ~pool_genres.isin(SUPPRESSED_GENRES)).sum())
        t3_min      = min(assessment.tier3_slots, avail_t3)

        # Cap T2 slots to videos actually available in the pool so the gap
        # doesn't silently overflow into T3 and let non-preferred genres dominate.
        avail_t2 = int((pool_genres.isin(SUPPRESSED_GENRES) & ~pool_genres.isin(pref_set)).sum())
        if assessment.tier2_slots > avail_t2:
            logger.warning(
                "T2 slots (%d) exceed pool availability (%d) for user=%s — capping",
                assessment.tier2_slots, avail_t2, uid,
            )
            overflow = assessment.tier2_slots - avail_t2
            assessment.tier2_slots = avail_t2
            assessment.tier1_slots = min(assessment.tier1_slots + overflow, 30)

        viral_ids = set(self.master_df.nlargest(100, "virality_score")["video_id"])
        t4_min    = assessment.tier4_slots

        logger.info("Candidates | user=%s | total=%d | pool_genres=%s | t3_min=%d",
                    uid, len(prefetched), pool_mix, t3_min)

        logger.info("Attempting LLM selection | user=%s", uid)
        selected, reasoning = self._llm_select(
            user, prefetched, assessment, viral_ids, t3_min, t4_min, feedback_history
        )

        if selected is None:
            logger.warning("LLM selection failed | user=%s — falling back to rule-based", uid)
            selected = self._rule_based_select(user, prefetched, assessment, t3_min)
            reasoning = "Rule-based tier-balanced selection (LLM fallback)."
        else:
            logger.info("LLM selection succeeded | user=%s | selected=%d videos", uid, len(selected))

        sel_df    = self.master_df[self.master_df["video_id"].isin(selected)]
        sel_mix   = dict(sel_df["genre"].value_counts())
        t1_count  = int(sel_df["genre"].isin(pref_set).sum())
        t2_count  = int((sel_df["genre"].isin(SUPPRESSED_GENRES) & ~sel_df["genre"].isin(pref_set)).sum())
        t3_count  = len(selected) - t1_count - t2_count
        logger.info("Selection | user=%s | T1=%d T2=%d T3=%d total=%d | genres=%s | reasoning='%s'",
                    uid, t1_count, t2_count, t3_count, len(selected), sel_mix, reasoning)

        return selected, reasoning

    # ══════════════════════════════════════════════════════════════════════════════
    # LLM selection — single structured call, no tool loop
    # ══════════════════════════════════════════════════════════════════════════════

    def _llm_select(
        self, user: dict, prefetched: List[str], assessment: BiasAssessment,
        viral_ids: set, t3_min: int, t4_min: int, feedback_history: list = None,
    ):
        pref_genres = list(user["preferred_genres"].keys())
        table       = self._candidates_table(prefetched, viral_ids)

        t1_min = max(0, assessment.tier1_slots - 5)
        t1_max = assessment.tier1_slots + 5
        t2_min = assessment.tier2_slots
        t2_max = assessment.tier2_slots + 6

        feedback_note = ""
        if feedback_history:
            fb = "\n".join(f"  - {f}" for f in feedback_history)
            feedback_note = f"\nMandatory user feedback (override all other criteria):\n{fb}\n"

        system = SystemMessage(content=(
            "You select videos by row number. Respond with JSON only. "
            "Column S='*' means suppressed category. Column V='#' means globally viral. "
            'Return EXACTLY: {"row_numbers": [<integers>], "reasoning": "<plain string>"}'
        ))

        human = HumanMessage(content=(
            f"Select exactly 30 rows for user '{user['name']}' "
            f"(preferred genres: {pref_genres}, languages: {user['preferred_languages']})."
            f"{feedback_note}\n\n"
            f"Hard constraints (reject the whole selection if violated):\n"
            f"  - Exactly 30 unique row numbers\n"
            f"  - T1 (genre in {pref_genres}): {t1_min} <= T1 <= {t1_max}  (target {assessment.tier1_slots})\n"
            f"  - T2 (S='*', genre NOT in T1): T2 >= {t2_min} and T2 <= {t2_max}\n"
            f"  - T3 (all others, not T1 or T2): T3 >= {t3_min}\n"
            f"  - At least {t4_min} rows must carry V='#' (counted inside T1/T2/T3)\n"
            f"  - No single genre > 15 rows\n"
            f"  - T1 + T2 + T3 = 30\n\n"
            f"Bias detected — missing: {assessment.genres_missing}, "
            f"over-represented: {assessment.genres_over_represented}\n\n"
            f"CANDIDATE TABLE ({len(prefetched)} rows):\n{table}\n\n"
            f"Return JSON with row_numbers (list of 30 integers) and reasoning."
        ))

        uid = user.get("user_id", "?")
        try:
            result = self._call_llm([system, human], structured_schema=_RowSelection)
            logger.info("LLM select raw response | user=%s | rows=%s | reasoning='%s'",
                        uid, result.row_numbers, result.reasoning)
            valid, seen = [], set()
            for n in result.row_numbers:
                i = n - 1
                if 0 <= i < len(prefetched) and prefetched[i] not in seen:
                    valid.append(prefetched[i])
                    seen.add(prefetched[i])
            if len(valid) >= 30:
                logger.info("LLM select succeeded | user=%s | valid=%d rows", uid, len(valid))
                return valid[:30], result.reasoning
            logger.warning("LLM select only returned %d valid rows | user=%s — falling through to rule-based",
                           len(valid), uid)
        except Exception as e:
            logger.error("LLM select failed | user=%s | %s: %s", uid, type(e).__name__, e)
        return None, None

    # ══════════════════════════════════════════════════════════════════════════════
    # Rule-based fallback selection
    # ══════════════════════════════════════════════════════════════════════════════

    def _rule_based_select(
        self, user: dict, prefetched: List[str],
        assessment: BiasAssessment, t3_min: int,
    ) -> List[str]:
        pref = set(user["preferred_genres"].keys())
        missing_set = set(assessment.genres_missing)

        pool_df = self.master_df[self.master_df["video_id"].isin(prefetched)].copy()
        g = pool_df["genre"].str.strip()
        pool_df["_tier"] = 3
        pool_df.loc[g.isin(pref), "_tier"] = 1
        pool_df.loc[g.isin(SUPPRESSED_GENRES) & ~g.isin(pref), "_tier"] = 2
        pool_df["_miss"] = (~g.isin(missing_set)).astype(int)

        id_order = {v: i for i, v in enumerate(prefetched)}
        pool_df["_ord"] = pool_df["video_id"].map(id_order)
        pool_df = pool_df.sort_values(["_miss", "_ord"])

        t1_pool = pool_df[pool_df["_tier"] == 1]["video_id"].tolist()
        t2_pool = pool_df[pool_df["_tier"] == 2]["video_id"].tolist()
        t3_pool = pool_df[pool_df["_tier"] == 3]["video_id"].tolist()

        t1_target = assessment.tier1_slots
        t1_max    = assessment.tier1_slots + 3
        t2_min    = assessment.tier2_slots
        t2_max    = assessment.tier2_slots + 4
        GENRE_CAP = 12

        vid_tier  = pool_df.set_index("video_id")["_tier"].to_dict()
        vid_genre = pool_df.set_index("video_id")["genre"].to_dict()
        genre_count: Counter = Counter()
        selected: list = []
        t1_count = t2_count = 0

        for vid in t1_pool:
            if t1_count >= t1_target:
                break
            if genre_count[vid_genre.get(vid, "")] >= GENRE_CAP:
                continue
            selected.append(vid); genre_count[vid_genre.get(vid, "")] += 1; t1_count += 1

        for vid in t2_pool:
            if t2_count >= t2_min:
                break
            if genre_count[vid_genre.get(vid, "")] >= GENRE_CAP:
                continue
            selected.append(vid); genre_count[vid_genre.get(vid, "")] += 1; t2_count += 1

        t3_count = 0
        for vid in t3_pool:
            if t3_count >= t3_min:
                break
            if genre_count[vid_genre.get(vid, "")] >= GENRE_CAP:
                continue
            selected.append(vid); genre_count[vid_genre.get(vid, "")] += 1; t3_count += 1

        used = set(selected)

        for vid in prefetched:
            if len(selected) >= 30:
                break
            if vid in used:
                continue
            tier  = vid_tier.get(vid, 3)
            genre = vid_genre.get(vid, "")
            if tier == 1 and t1_count >= t1_max:
                continue
            if tier == 2 and t2_count >= t2_max:
                continue
            if genre_count[genre] >= GENRE_CAP:
                continue
            selected.append(vid); used.add(vid); genre_count[genre] += 1
            if tier == 1:   t1_count += 1
            elif tier == 2: t2_count += 1

        if len(selected) < 30:
            selected.extend(self._tier4_viral(30 - len(selected), used))

        return selected[:30]

    # ══════════════════════════════════════════════════════════════════════════════
    # Candidate pre-fetching (pure Python, no LLM)
    # ══════════════════════════════════════════════════════════════════════════════

    def _prefetch_candidates(
        self, user: dict, assessment: BiasAssessment, exclude: set
    ) -> List[str]:
        pref_genres   = list(user["preferred_genres"].keys())
        target_genres = list(dict.fromkeys(assessment.genres_missing + pref_genres))
        ex = set(exclude)

        t1 = self._tier1_similarity_trending(
            user, target_genres, max(assessment.tier1_slots + 4, 16), ex)
        ex.update(t1)

        missing_supp = [g for g in assessment.genres_missing if g in SUPPRESSED_GENRES]
        other_supp   = [g for g in SUPPRESSED_GENRES if g not in missing_supp]
        t2_budget    = max(assessment.tier2_slots + 4, 12)

        t2_miss = []
        if missing_supp:
            t2_miss = self._tier2_genre_retrieval(
                user, missing_supp, max(len(missing_supp) * 3, t2_budget // 2), ex)
            ex.update(t2_miss)

        t2_fill = (self._tier2_genre_retrieval(
            user, other_supp, max(t2_budget - len(t2_miss), 4), ex)
            if other_supp else [])
        ex.update(t2_fill)

        t3 = self._tier3_diversity(user, max(assessment.tier3_slots + 2, 8), ex)
        ex.update(t3)

        t4 = self._tier4_viral(max(assessment.tier4_slots + 2, 6), ex)

        raw = t1 + t2_miss + t2_fill + t3 + t4
        seen_set: set = set()
        deduped: List[str] = []
        for v in raw:
            if v not in seen_set:
                deduped.append(v)
                seen_set.add(v)
        logger.info("Prefetch tiers | T1=%d T2_miss=%d T2_fill=%d T3=%d T4=%d -> pool=%d",
                    len(t1), len(t2_miss), len(t2_fill), len(t3), len(t4), len(deduped))
        return deduped[:50]

    def _candidates_table(self, ids: List[str], viral_ids: set = None) -> str:
        rows = self.master_df[self.master_df["video_id"].isin(ids)].copy()
        id_order = {v: i for i, v in enumerate(ids)}
        rows["_ord"] = rows["video_id"].map(id_order)
        rows = rows.sort_values("_ord").reset_index(drop=True)
        lines = ["#   genre           language   S  V  title"]
        lines.append("─" * 72)
        for i, (_, r) in enumerate(rows.iterrows(), start=1):
            sup = "*" if r["is_suppressed"] else " "
            vir = "#" if viral_ids and r["video_id"] in viral_ids else " "
            lines.append(
                f"{i:<4}{r['genre']:<16}{r['language']:<11}{sup}  {vir}  "
                f"{str(r['title'])[:36]}"
            )
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════════
    # 4-Tier retrieval primitives
    # ══════════════════════════════════════════════════════════════════════════════

    def _user_tower_embedding(self, user: dict) -> np.ndarray:
        interactions = user.get("interactions", [])
        if not interactions:
            return np.zeros(self.embeddings.shape[1])
        vid_to_pos = {vid: i for i, vid in enumerate(self.master_df["video_id"])}
        vecs, weights = [], []
        for item in interactions:
            pos = vid_to_pos.get(item["video_id"])
            if pos is None:
                continue
            vecs.append(self.embeddings[pos])
            weights.append(float(item["rating"]))
        if not vecs:
            return np.zeros(self.embeddings.shape[1])
        vecs    = np.array(vecs)
        weights = np.array(weights) / sum(weights)
        user_vec = (vecs * weights[:, np.newaxis]).sum(axis=0)
        norm = np.linalg.norm(user_vec)
        return user_vec / norm if norm > 0 else user_vec

    def _tier1_similarity_trending(
        self, user: dict, target_genres: List[str], n: int, exclude: set
    ) -> List[str]:
        user_vec      = self._user_tower_embedding(user)
        pref_genres   = list(user["preferred_genres"].keys())
        genres_to_use = target_genres if target_genres else pref_genres

        if np.linalg.norm(user_vec) == 0:
            return self._tier2_genre_retrieval(user, genres_to_use, n, exclude)

        mask = (
            self.master_df["genre"].isin(genres_to_use) &
            ~self.master_df["video_id"].isin(exclude)
        )
        if not mask.any():
            return []

        idxs   = self.master_df[mask].index.tolist()
        sims   = cosine_similarity([user_vec], self.embeddings[idxs])[0]
        sub_df = self.master_df.loc[idxs].copy()
        sub_df["sim"]      = sims
        sub_df["t1_score"] = 0.65 * sub_df["sim"] + 0.35 * sub_df["virality_score"]
        return sub_df.nlargest(n, "t1_score")["video_id"].tolist()

    def _tier2_genre_retrieval(
        self, user: dict, target_genres: List[str], n: int, exclude: set
    ) -> List[str]:
        pref_langs = user.get("preferred_languages", ["English"])
        genres_use = target_genres if target_genres else list(user["preferred_genres"].keys())
        pool       = self.master_df[
            self.master_df["genre"].isin(genres_use) &
            ~self.master_df["video_id"].isin(exclude)
        ]
        lang_pool = pool[pool["language"].isin(pref_langs)]
        pool      = lang_pool if len(lang_pool) >= n else pool
        return pool.nlargest(n, "virality_score")["video_id"].tolist()

    def _tier3_diversity(self, user: dict, n: int, exclude: set) -> List[str]:
        pref_genres = set(user["preferred_genres"].keys())
        pool = self.master_df[
            ~self.master_df["genre"].isin(pref_genres) &
            ~self.master_df["video_id"].isin(exclude)
        ].sort_values("virality_score", ascending=False)
        result, seen_genres = [], set()
        for _, row in pool.iterrows():
            if len(result) >= n:
                break
            if row["genre"] not in seen_genres:
                result.append(row["video_id"])
                seen_genres.add(row["genre"])
        return result

    def _tier4_viral(self, n: int, exclude: set) -> List[str]:
        return (
            self.master_df[~self.master_df["video_id"].isin(exclude)]
            .nlargest(n, "virality_score")["video_id"].tolist()
        )

    # ══════════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════════

    def _summarise_history(self, user: dict) -> str:
        interactions = user.get("interactions", [])
        genre_counts = Counter(i["genre"] for i in interactions)
        lang_counts  = Counter(i["language"] for i in interactions)
        lines = [
            f"Total watched: {len(interactions)} videos",
            f"Genre breakdown: {dict(genre_counts.most_common())}",
            f"Language breakdown: {dict(lang_counts.most_common())}",
            "Recent titles:",
        ]
        for i in interactions[:6]:
            lines.append(f"  - [{i['genre']} | {i['language']}] {i['title'][:55]}")
        return "\n".join(lines)

    def _summarise_recs(self, recs: pd.DataFrame) -> str:
        genre_counts = recs["genre"].value_counts().to_dict()
        lang_counts  = recs["language"].value_counts().to_dict()
        total = len(recs)
        lines = [
            f"Genre breakdown: {genre_counts}",
            f"Language breakdown: {lang_counts}",
            f"Suppressed content: {recs['is_suppressed'].sum()}/{total}",
            f"Non-English: {(recs['language'] != 'English').sum()}/{total}",
            "Videos:",
        ]
        for _, row in recs.iterrows():
            lines.append(f"  - [{row['genre']} | {row['language']}] {str(row['title'])[:55]}")
        return "\n".join(lines)
