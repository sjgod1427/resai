"""
llm_supervisor.py
-----------------
LangChain + Groq supervisor.

Step 1 - JUDGE  : structured LLM call decides if the biased output is fair.
Step 2 - AGENT  : if biased, a tool-calling agent builds the corrected list.

The agent has 5 tools:
  search_similar_trending  -- Tier 1: semantic sim + virality in target genres
  search_by_genre          -- Tier 2: direct genre/language retrieval
  search_diverse           -- Tier 3: outside user's genre bubble
  search_viral             -- Tier 4: globally top-viral
  submit_final_list        -- validates and locks in exactly 10 IDs

Because submit_final_list enforces the 10-item constraint server-side,
the LLM physically cannot finish with a wrong-length list.
"""

import os
import re
import time
import json
import numpy as np
import pandas as pd
from typing import List
from collections import Counter
from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from pydantic import BaseModel, Field

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────

GROQ_MODELS       = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
SUPPRESSED_GENRES = {"Educational", "Documentary", "DIY", "News/Analysis", "Regional"}

# ── Pydantic schema for judge output ───────────────────────────────────────────

class BiasAssessment(BaseModel):
    is_biased: bool = Field(description="True if the recommendations are unfair for this user")
    reasoning: str  = Field(description="Explanation of why the output is or is not biased")
    genres_over_represented: List[str] = Field(description="Genres appearing too many times")
    genres_missing: List[str]          = Field(description="Genres that should appear but don't")
    tier1_slots: int = Field(description="Slots for Tier 1 (similarity + trending)", ge=0, le=10)
    tier2_slots: int = Field(description="Slots for Tier 2 (genre retrieval fallback)", ge=0, le=10)
    tier3_slots: int = Field(description="Slots for Tier 3 (diversity)", ge=0, le=10)
    tier4_slots: int = Field(description="Slots for Tier 4 (global viral)", ge=0, le=10)


# ── Supervisor ─────────────────────────────────────────────────────────────────

class LLMSupervisor:
    """Loaded once at startup; called per user test."""

    def __init__(self, master_df: pd.DataFrame, embeddings: np.ndarray):
        df = master_df.reset_index(drop=True)
        # Remove duplicate video_ids: same video can trend in multiple country datasets
        dup_mask = df["video_id"].duplicated(keep="first")
        if dup_mask.any():
            n_dups = int(dup_mask.sum())
            print(f"  [Supervisor] Dropping {n_dups} duplicate video_id rows "
                  f"({len(df):,} -> {len(df) - n_dups:,} unique videos).")
            keep       = (~dup_mask).values          # bool numpy array
            df         = df[keep].reset_index(drop=True)
            embeddings = embeddings[keep]

        self.master_df   = df
        self.embeddings  = embeddings
        self._api_key    = os.getenv("GROQ_API_KEY")
        self._model_idx  = 0
        self.llm         = self._make_llm()

    # ── LLM helpers ────────────────────────────────────────────────────────────

    def _make_llm(self) -> ChatGroq:
        return ChatGroq(model=GROQ_MODELS[self._model_idx], temperature=0,
                        groq_api_key=self._api_key)

    def _call_llm(self, messages, structured_schema=None):
        """Single LLM call with automatic model fallback on 429."""
        for _ in range(len(GROQ_MODELS)):
            try:
                if structured_schema:
                    return self.llm.with_structured_output(structured_schema).invoke(messages)
                return self.llm.invoke(messages)
            except Exception as e:
                err = str(e)
                if ("429" in err or "rate_limit" in err.lower()) \
                        and self._model_idx + 1 < len(GROQ_MODELS):
                    self._model_idx += 1
                    self.llm = self._make_llm()
                    print(f"  [LLM] Rate-limited -> switching to {GROQ_MODELS[self._model_idx]}")
                    time.sleep(2)
                else:
                    raise

    # ══════════════════════════════════════════════════════════════════════════
    # Public entry point
    # ══════════════════════════════════════════════════════════════════════════

    def fix(self, user: dict, biased_recs: pd.DataFrame):
        """
        Returns (corrected_df, reasoning, assessment).
        corrected_df always has exactly 10 rows.
        """
        assessment = self._judge(user, biased_recs)

        if not assessment.is_biased:
            return biased_recs.head(10), assessment.reasoning, assessment

        corrected_ids, reasoning = self._correct_with_agent(user, biased_recs, assessment)

        corrected_df = (
            self.master_df[self.master_df["video_id"].isin(corrected_ids)]
            .drop_duplicates("video_id")
            .copy()
        )
        order = {vid: i for i, vid in enumerate(corrected_ids)}
        corrected_df["_order"] = corrected_df["video_id"].map(order)
        corrected_df = corrected_df.sort_values("_order").drop(columns=["_order"])

        return corrected_df, reasoning, assessment

    # ══════════════════════════════════════════════════════════════════════════
    # Step 1 — Judge
    # ══════════════════════════════════════════════════════════════════════════

    def _judge(self, user: dict, biased_recs: pd.DataFrame) -> BiasAssessment:
        system = SystemMessage(content="""
You are a fairness auditor for a video recommendation system.
Decide if the recommendation output is biased for this specific user.

Biased if:
- Suppresses genres that dominate the user's watch history
- Ignores user's preferred languages
- Violates fairness quota: at least 2 suppressed-category videos, at least 2 non-English
  (Suppressed categories: Educational, Documentary, DIY, News/Analysis, Regional)

If biased, allocate tier slots (sum must be <= 10):
  Tier 1: semantic similarity + trending in user's domain
  Tier 2: direct genre retrieval fallback
  Tier 3: diversity outside user's bubble
  Tier 4: globally viral

Return ONLY valid JSON matching the schema.
""")
        human = HumanMessage(content=f"""
USER PROFILE
Name: {user['name']}
Description: {user['description']}
Preferred genres: {json.dumps(user['preferred_genres'])}
Preferred languages: {user['preferred_languages']}

WATCH HISTORY
{self._summarise_history(user)}

BIASED RECOMMENDER OUTPUT
{self._summarise_recs(biased_recs)}

Assess bias and return the JSON schema.
""")
        try:
            return self._call_llm([system, human], structured_schema=BiasAssessment)
        except Exception as e:
            print(f"  [Judge] Error: {e}. Defaulting to biased.")
            pref_genres = list(user["preferred_genres"].keys())
            rec_genres  = biased_recs["genre"].tolist()
            missing = [g for g in pref_genres if g not in rec_genres]
            return BiasAssessment(
                is_biased=True,
                reasoning="LLM assessment failed. Applying default correction.",
                genres_over_represented=[g for g in rec_genres if g not in pref_genres],
                genres_missing=missing,
                tier1_slots=4, tier2_slots=2, tier3_slots=2, tier4_slots=2,
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Step 2 — Agentic correction
    # ══════════════════════════════════════════════════════════════════════════

    def _correct_with_agent(
        self, user: dict, biased_recs: pd.DataFrame, assessment: BiasAssessment
    ):
        """
        Tool-calling agent that actively retrieves videos and submits
        a validated 10-item list. submit_final_list enforces the count.
        """

        # ── Shared state (mutated by tool closures) ────────────────────────
        state = {
            "seen": set(biased_recs["video_id"].tolist()),  # avoid re-returning biased IDs
            "final_ids": None,
            "final_reasoning": "",
        }

        def _rows_to_json(ids: List[str]) -> str:
            if not ids:
                return json.dumps([])
            rows = self.master_df[self.master_df["video_id"].isin(ids)]
            return json.dumps([
                {"video_id": r["video_id"], "title": str(r["title"])[:55],
                 "genre": r["genre"], "language": r["language"],
                 "virality": round(float(r["virality_score"]), 3)}
                for _, r in rows.iterrows()
            ], indent=2)

        # ── Tool definitions ───────────────────────────────────────────────

        @tool
        def search_similar_trending(genres: str, n: int) -> str:
            """Tier-1: semantic similarity to user's watch history in target genres,
            scored by virality. Use for personalised + trending content.
            Args: genres (comma-separated), n (1-10)"""
            genre_list = [g.strip() for g in genres.split(",") if g.strip()]
            ids = self._tier1_similarity_trending(user, genre_list, n, state["seen"])
            state["seen"].update(ids)
            return _rows_to_json(ids)

        @tool
        def search_by_genre(genres: str, n: int) -> str:
            """Tier-2: top videos from specific genres sorted by virality.
            Use when Tier-1 doesn't fill enough slots or for exact genre control.
            Args: genres (comma-separated), n (1-10)"""
            genre_list = [g.strip() for g in genres.split(",") if g.strip()]
            ids = self._tier2_genre_retrieval(user, genre_list, n, state["seen"])
            state["seen"].update(ids)
            return _rows_to_json(ids)

        @tool
        def search_diverse(n: int) -> str:
            """Tier-3: videos from genres the user doesn't normally watch.
            Use to add language/genre variety.
            Args: n (1-5)"""
            ids = self._tier3_diversity(user, n, state["seen"])
            state["seen"].update(ids)
            return _rows_to_json(ids)

        @tool
        def search_viral(n: int) -> str:
            """Tier-4: globally top-viral videos regardless of genre or language.
            Use to fill remaining slots.
            Args: n (1-5)"""
            ids = self._tier4_viral(n, state["seen"])
            state["seen"].update(ids)
            return _rows_to_json(ids)

        @tool
        def submit_final_list(video_ids: str, reasoning: str) -> str:
            """Submit the final corrected recommendation list.
            MUST contain EXACTLY 10 comma-separated video_ids from previous search results.
            Will be REJECTED if fewer or more than 10 unique valid IDs are provided.
            Args: video_ids (comma-separated, exactly 10), reasoning (what changed and why)"""
            ids = [v.strip() for v in video_ids.split(",") if v.strip()]
            valid_set = set(self.master_df["video_id"].values)

            # Deduplicate, preserve order, validate existence
            seen_local, unique_valid = set(), []
            for v in ids:
                if v in valid_set and v not in seen_local:
                    unique_valid.append(v)
                    seen_local.add(v)

            if len(unique_valid) < 10:
                needed = 10 - len(unique_valid)
                return (f"REJECTED: only {len(unique_valid)} unique valid IDs. "
                        f"Call search tools to get {needed} more, then resubmit.")
            if len(unique_valid) > 10:
                unique_valid = unique_valid[:10]

            state["final_ids"]      = unique_valid
            state["final_reasoning"] = reasoning
            return "ACCEPTED: 10-video corrected list confirmed."

        tools = [
            search_similar_trending,
            search_by_genre,
            search_diverse,
            search_viral,
            submit_final_list,
        ]

        system_prompt = (
            "You are a fairness supervisor for a video recommendation system.\n\n"
            "Your task: build a corrected 10-video list to replace a biased output.\n\n"
            "Workflow:\n"
            "1. Review the bias assessment to identify missing genres/languages.\n"
            "2. Call search tools to retrieve candidate videos (each call returns fresh, "
            "non-duplicate results).\n"
            "3. Once you have enough candidates, call submit_final_list with exactly 10 "
            "video_ids.\n"
            "4. If submit_final_list rejects your list, fetch more videos and resubmit.\n\n"
            "Hard constraints:\n"
            "- EXACTLY 10 videos (enforced by submit_final_list)\n"
            "- >= 2 from suppressed categories: Educational, Documentary, DIY, "
            "News/Analysis, Regional\n"
            "- >= 2 non-English videos\n"
            "- No single genre > 5 slots\n"
            "- Tool priority: search_similar_trending > search_by_genre > "
            "search_diverse > search_viral"
        )

        agent = create_agent(self.llm, tools, system_prompt=system_prompt)

        input_msg = (
            f"User: {user['name']} — {user['description'][:120]}\n"
            f"Preferred genres: {json.dumps(user['preferred_genres'])}\n"
            f"Preferred languages: {user['preferred_languages']}\n\n"
            f"BIAS ASSESSMENT:\n{assessment.reasoning}\n"
            f"Missing genres: {assessment.genres_missing}\n"
            f"Over-represented: {assessment.genres_over_represented}\n"
            f"Suggested slots: T1={assessment.tier1_slots} "
            f"T2={assessment.tier2_slots} T3={assessment.tier3_slots} "
            f"T4={assessment.tier4_slots}\n\n"
            f"CURRENT BIASED OUTPUT:\n{self._summarise_recs(biased_recs)}\n\n"
            "Use the search tools to retrieve videos, then call submit_final_list "
            "with exactly 10 video_ids."
        )

        try:
            result   = agent.invoke({"messages": [HumanMessage(content=input_msg)]})
            messages = result.get("messages", [])
            _        = messages[-1].content if messages else ""
        except Exception as e:
            print(f"  [Agent] Error during run: {e}")

        # ── Guarantee 10 items if agent didn't submit ──────────────────────
        if state["final_ids"] is None:
            print("  [Agent] No submission — using pool fallback.")
            seen_set = state["seen"] - set(biased_recs["video_id"].tolist())
            fallback = list(seen_set)
            for v in biased_recs["video_id"]:
                if len(fallback) >= 10:
                    break
                if v not in fallback:
                    fallback.append(v)
            # Last resort: top viral
            if len(fallback) < 10:
                extra = self._tier4_viral(10 - len(fallback), set(fallback))
                fallback.extend(extra)
            state["final_ids"]       = fallback[:10]
            state["final_reasoning"] = "Agent fallback: pool-based selection."

        return state["final_ids"], state["final_reasoning"]

    # ══════════════════════════════════════════════════════════════════════════
    # 4-Tier retrieval primitives
    # ══════════════════════════════════════════════════════════════════════════

    def _user_tower_embedding(self, user: dict) -> np.ndarray:
        """
        Two-tower user vector.
        Rating-weighted average of the embeddings of every video the user watched.
        Produces a dense taste representation without any live text encoding.
        """
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

        mask = (
            self.master_df["genre"].isin(genres_to_use) &
            ~self.master_df["video_id"].isin(exclude)
        )
        if not mask.any():
            return []

        idxs    = self.master_df[mask].index.tolist()
        sims    = cosine_similarity([user_vec], self.embeddings[idxs])[0]
        sub_df  = self.master_df.loc[idxs].copy()
        sub_df["sim"]      = sims
        sub_df["t1_score"] = 0.65 * sub_df["sim"] + 0.35 * sub_df["virality_score"]
        return sub_df.nlargest(n, "t1_score")["video_id"].tolist()

    def _tier2_genre_retrieval(
        self, user: dict, target_genres: List[str], n: int, exclude: set
    ) -> List[str]:
        pref_langs = user.get("preferred_languages", ["English"])
        genres_use = target_genres if target_genres else list(user["preferred_genres"].keys())

        pool      = self.master_df[
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

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════

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
        lines = [
            f"Genre breakdown: {genre_counts}",
            f"Language breakdown: {lang_counts}",
            f"Suppressed content: {recs['is_suppressed'].sum()}/10",
            f"Non-English: {(recs['language'] != 'English').sum()}/10",
            "Videos:",
        ]
        for _, row in recs.iterrows():
            lines.append(f"  - [{row['genre']} | {row['language']}] {str(row['title'])[:55]}")
        return "\n".join(lines)
