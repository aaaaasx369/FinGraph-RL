#!/usr/bin/env python3
"""
FinRL-X Active RAG V2

Actual architecture:

    Discrete stochastic Retrieval Controller
                  |
                  v
    Ollama LLM + Token Logprobs
                  |
          Temporal KG / SEC RAG
                  |
          Semantic Evidence
                  |
       A2C / PPO / DDPG Portfolio
                  |
       Portfolio reward + retrieval cost
                  |
       Retrieval Controller update

Features:
    - Learned categorical retrieval decisions
    - Multi-step KG / SEC search
    - Temporal source indexes
    - Persistent Ollama embedding cache
    - Checkpoint / partial resume
    - Live training progress
    - Forced retrieval smoke tests
    - Strict historical evidence date filtering

Research limitations:
    - Retrieval router uses online REINFORCE
    - Portfolio actor uses SB3 A2C/PPO/DDPG
    - Not a native hybrid-action SB3 policy
    - Not a strict causal retrieval effect estimate
    - KGT training provenance is not PIT-certified
    - LLM pretrained knowledge may include future facts
    - Checkpoint resume is approximate, not bit-exact
"""

import argparse
import bisect
import hashlib
import json
import math
import random
import re
import sqlite3
import sys
import time

from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from gymnasium import spaces

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_flare_raggym_ollama as old
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_portfolio_aware_v2 as v2
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_self_questioning_v1 as legacy
)

TICKERS = old.TICKERS
RELATIONS = old.REL
SOURCES = old.SOURCES
COSTS = old.COST_BPS


def log(*args):
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        *args,
        flush=True,
    )


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def source_key(source, ticker):
    return source + "|" + ticker


class CachedOllama(old.LLMEngine):
    """
    Persistent SQLite cache for semantic embeddings.

    Old runner cached text generation only.
    This version also persists document/query
    embeddings across program restarts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS embedding (
                key TEXT PRIMARY KEY,
                dimension INTEGER NOT NULL,
                vector BLOB NOT NULL
            )
        """)

        self.db.commit()

        self.embedding_cache_hits = 0
        self.embedding_api_calls = 0

    @lru_cache(maxsize=16384)
    def embed(self, text):
        text = str(text)[:3000]

        key = hashlib.sha256(
            (
                self.host
                + "|"
                + self.embed_name
                + "|"
                + text
            ).encode("utf-8")
        ).hexdigest()

        row = self.db.execute(
            """
            SELECT dimension, vector
            FROM embedding
            WHERE key=?
            """,
            (key,),
        ).fetchone()

        if row is not None:
            dim, blob = row

            vector = np.frombuffer(
                blob,
                dtype=np.float32,
            ).copy()

            if len(vector) != dim:
                raise RuntimeError(
                    "Corrupt embedding cache"
                )

            self.embedding_cache_hits += 1
            return vector

        result = self.client.embed(
            model=self.embed_name,
            input=text,
        )

        vector = np.asarray(
            result.embeddings[0],
            dtype=np.float32,
        )

        vector /= (
            np.linalg.norm(vector) + 1e-9
        )

        if len(vector) != self.dim:
            raise RuntimeError(
                "Unexpected embedding size"
            )

        self.db.execute(
            """
            INSERT OR REPLACE INTO embedding
            VALUES (?, ?, ?)
            """,
            (
                key,
                len(vector),
                vector.tobytes(),
            ),
        )

        self.db.commit()

        self.embedding_api_calls += 1

        return vector


class IndexedTemporalRetrieval(old.TemporalRetrieval):
    """
    Faster candidate lookup.

    Build an inverted index for company names,
    ticker symbols and SEC ticker metadata.

    Time is checked AGAIN at retrieval.
    """

    def __init__(
        self,
        kg_rows,
        sec_rows,
        llm,
        topk,
        max_candidates=120,
    ):
        super().__init__(
            kg_rows,
            sec_rows,
            llm,
            topk,
        )

        self.max_candidates = max_candidates

        self.company_index = {}

        for source in ["KG", "SEC"]:
            grouped = defaultdict(list)

            for row in self.db[source]:
                low = str(
                    row.get("text", "")
                ).lower()

                source_ticker = str(
                    row.get("ticker", "")
                ).upper()

                for ticker in TICKERS:
                    aliases = old.ALIAS[ticker]

                    matched = (
                        source_ticker == ticker
                        or any(
                            alias in low
                            for alias in aliases
                        )
                    )

                    if matched:
                        grouped[ticker].append(row)

            for ticker in TICKERS:
                items = sorted(
                    grouped[ticker],
                    key=lambda x: x["available_at"],
                )

                self.company_index[
                    source_key(source, ticker)
                ] = (
                    items,
                    [
                        row["available_at"]
                        for row in items
                    ],
                )

        for source in ["KG", "SEC"]:
            for ticker in TICKERS:
                rows, _ = self.company_index[
                    source_key(source, ticker)
                ]

                log(
                    "[INDEX]",
                    source,
                    ticker,
                    len(rows),
                )

    @lru_cache(maxsize=50000)
    def query(
        self,
        source,
        ticker,
        relation,
        day_iso,
        question,
    ):
        day = pd.Timestamp(day_iso)

        rows, dates = self.company_index[
            source_key(source, ticker)
        ]

        cutoff = bisect.bisect_left(
            dates,
            day,
        )

        # Historical lookback.
        # Bounded scan improves performance.
        # This is a candidate-window restriction,
        # not a guarantee of perfect recall.
        start = max(
            0,
            cutoff - self.max_candidates,
        )

        candidates = rows[start:cutoff]

        query_tokens = set(
            re.findall(
                r"[a-z0-9]+",
                question.lower(),
            )
        )

        lexical = []

        for row in candidates:
            if row["available_at"] >= day:
                continue

            words = set(
                re.findall(
                    r"[a-z0-9]+",
                    str(row["text"]).lower(),
                )
            )

            score = (
                len(query_tokens & words)
                / max(1, len(query_tokens))
            )

            if (
                source == "KG"
                and str(
                    row.get("relation", "")
                ).upper() == relation
            ):
                score += 0.5

            age = (
                day - row["available_at"]
            ).total_seconds() / 86400

            score += (
                0.08
                * math.exp(-max(age, 0) / 365)
            )

            lexical.append((score, row))

        lexical.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        shortlisted = lexical[
            :max(self.topk * 8, 12)
        ]

        if not shortlisted:
            return []

        qvec = self.llm.embed(
            question
        )

        scored = []

        for score, row in shortlisted:
            dvec = self.llm.embed(
                row["text"]
            )

            semantic = float(
                np.dot(qvec, dvec)
            )

            combined = (
                0.4 * score
                + 0.6 * semantic
            )

            scored.append(
                (combined, row)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return scored[:self.topk]


class RetrievalPolicy(nn.Module):
    """
    Categorical multi-head retrieval controller.

    Source:
        STOP / KG / SEC / BOTH

    Company:
        AAPL / AMZN / GOOGL / MSFT / NVDA

    Relation:
        Four financial KG relation types

    All three heads are stochastic during training.
    """

    def __init__(self, input_dim):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )

        self.source_head = nn.Linear(
            64, len(SOURCES)
        )

        self.company_head = nn.Linear(
            64, len(TICKERS)
        )

        self.relation_head = nn.Linear(
            64, len(RELATIONS)
        )

    def forward(self, x):
        h = self.backbone(x)

        return (
            self.source_head(h),
            self.company_head(h),
            self.relation_head(h),
        )


class LearnedRouter:
    """
    Online policy gradient for retrieval decisions.

    Uses a categorical distribution, not
    continuous-action argmax.

    An independent controller is trained
    for each portfolio algorithm/seed.

    Reward uses realized portfolio log return
    minus a running baseline.
    """

    def __init__(
        self,
        state_dim,
        lr,
        entropy_weight,
        training=True,
    ):
        self.net = RetrievalPolicy(
            state_dim
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
        )

        self.entropy_weight = entropy_weight
        self.training = training

        self.pending = []

        self.running_baseline = 0.0
        self.update_count = 0

    def select(self, features):
        x = torch.as_tensor(
            features,
            dtype=torch.float32,
        ).unsqueeze(0)

        logits = self.net(x)

        choices = []
        log_probs = []
        entropies = []

        for value in logits:
            distribution = (
                torch.distributions.Categorical(
                    logits=value
                )
            )

            if self.training:
                choice = distribution.sample()
            else:
                choice = torch.argmax(
                    value,
                    dim=-1,
                )

            choices.append(
                int(choice.item())
            )

            log_probs.append(
                distribution.log_prob(choice)
            )

            entropies.append(
                distribution.entropy()
            )

        if self.training:
            self.pending.append((
                torch.stack(log_probs).sum(),
                torch.stack(entropies).sum(),
            ))

        return (
            SOURCES[choices[0]],
            TICKERS[choices[1]],
            RELATIONS[choices[2]],
        )

    def update(self, reward):
        if not self.training:
            self.pending.clear()
            return

        if not self.pending:
            return

        reward = float(reward)

        advantage = (
            reward - self.running_baseline
        )

        # Advantage normalization by scale.
        advantage = float(
            np.clip(
                advantage * 100.0,
                -5.0,
                5.0,
            )
        )

        log_prob = torch.stack([
            x[0] for x in self.pending
        ]).sum()

        entropy = torch.stack([
            x[1] for x in self.pending
        ]).mean()

        loss = (
            -advantage * log_prob
            - self.entropy_weight * entropy
        )

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(),
            1.0,
        )

        self.optimizer.step()

        self.running_baseline = (
            0.98 * self.running_baseline
            + 0.02 * reward
        )

        self.update_count += 1
        self.pending.clear()

    def save(self, path):
        torch.save({
            "network": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "running_baseline":
                self.running_baseline,
            "update_count":
                self.update_count,
        }, path)

    def load(self, path):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        self.net.load_state_dict(
            data["network"]
        )

        self.optimizer.load_state_dict(
            data["optimizer"]
        )

        self.running_baseline = (
            data["running_baseline"]
        )

        self.update_count = (
            data["update_count"]
        )


class ActiveRAGEnvironment(old.MultiStepEnv):
    """
    Hierarchical Environment.

    At each decision day:

    1. Learned categorical router chooses retrieval.
    2. LLM generates financial query.
    3. KG/SEC RAG retrieves evidence.
    4. Router may retrieve again or stop.
    5. Portfolio actor observes semantic evidence.
    6. Portfolio actor allocates.
    7. Portfolio reward updates retrieval router.

    One Gym step = one completed portfolio day.

    This is hierarchical RL, not a single
    native mixed discrete/continuous SB3 actor.
    """

    def __init__(
        self,
        *args,
        router,
        **kwargs,
    ):
        self.router = router

        self.actual_retrieval_count = 0
        self.actual_kg_hits = 0
        self.actual_sec_hits = 0

        super().__init__(
            *args,
            **kwargs,
        )

        # Portfolio action only.
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(5,),
            dtype=np.float32,
        )

    def _question_seed(self):
        # Do not call LLM when nothing
        # has been retrieved.
        self.seed_text = ""
        self.confidence = 0.0

    def _router_state(self):
        """
        Only decision-time quantities.

        No future returns.
        """

        day = self.day

        prices = self.price[
            max(0, day - 1):day + 1
        ]

        if len(prices) == 2:
            previous_returns = (
                prices[1] / prices[0] - 1
            )
        else:
            previous_returns = np.zeros(5)

        state = np.concatenate([
            previous_returns,
            self.weights,
            np.array([
                self.used
                / max(1, self.steps),
                self.confidence,
                float(self.last_hits > 0),
                float(self.remaining_cost),
            ]),
        ])

        return np.nan_to_num(
            state,
            nan=0,
            posinf=0,
            neginf=0,
        ).astype(np.float32)

    def _search(self):
        date = self.dates[self.day]

        self.phase = 0
        self.used = 0
        self.evidence[:] = 0

        self.day_queries = []
        self.remaining_cost = 0.0
        self.last_hits = 0

        for step_id in range(self.steps):
            source, ticker, relation = (
                self.router.select(
                    self._router_state()
                )
            )

            if self.mode == "NEVER":
                source = "STOP"

            elif self.mode == "ALWAYS_BOTH":
                source = "BOTH"

            if source == "STOP":
                break

            summaries = " ".join(
                q.get("summary", "")
                for q in self.day_queries[-2:]
            )[:450]

            prompt = (
                f"Historical decision date: {date.date()}.\n"
                f"Company: {ticker}.\n"
                f"Relation: {relation}.\n"
                f"Prior retrieved evidence: "
                f"{summaries or 'none'}.\n"
                "Generate one short English search query "
                "for company relationships or SEC filings. "
                "Do not answer the question. Query:"
            )

            question, confidence = (
                self.llm.generate(prompt)
            )

            vector, provenance = (
                self.ret.retrieve(
                    source,
                    ticker,
                    relation,
                    date,
                    question,
                )
            )

            for item in provenance:
                available = pd.Timestamp(
                    item["available_at"]
                )

                if available >= date:
                    raise RuntimeError(
                        "Future evidence detected"
                    )

            for j in range(2):
                if np.linalg.norm(
                    vector[j]
                ) > 0:
                    if np.linalg.norm(
                        self.evidence[j]
                    ) == 0:
                        self.evidence[j] = vector[j]
                    else:
                        self.evidence[j] = (
                            0.5 * self.evidence[j]
                            + 0.5 * vector[j]
                        )

            self.used += 1

            cost = (
                COSTS[source]
                / 10000.0
                * self.cost_scale
            )

            self.remaining_cost += cost

            self.confidence = confidence
            self.last_hits = len(provenance)

            kg_hits = sum(
                x["source"] == "KG"
                for x in provenance
            )

            sec_hits = sum(
                x["source"] == "SEC"
                for x in provenance
            )

            self.actual_retrieval_count += 1
            self.actual_kg_hits += kg_hits
            self.actual_sec_hits += sec_hits

            summary = " ".join(
                x["text_excerpt"]
                for x in provenance[:2]
            )[:350]

            # LLM reads actual evidence.
            followup, follow_confidence = (
                self.llm.generate(
                    f"Date: {date.date()}.\n"
                    f"Query: {question}\n"
                    f"Historical retrieved evidence: "
                    f"{summary or 'NO EVIDENCE'}\n"
                    "What financial information remains "
                    "unresolved? Ask one short question."
                )
            )

            self.day_queries.append({
                "step": step_id + 1,
                "date": str(date.date()),
                "source": source,
                "ticker": ticker,
                "relation": relation,
                "query": question,
                "token_confidence": confidence,
                "followup": followup,
                "followup_confidence":
                    follow_confidence,
                "evidence": provenance,
                "summary": summary,
                "retrieval_cost": cost,
            })

            if self.mode == "ALWAYS_BOTH":
                break

        # Evidence is now observable BEFORE
        # portfolio allocation.
        self.phase = 1

    def reset(self, *, seed=None, options=None):
        # Discard unfinished computation graphs
        # when an episode resets.
        self.router.pending.clear()

        obs, info = super().reset(
            seed=seed,
            options=options,
        )

        self._search()

        return self._obs(), info

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape != (5,):
            raise RuntimeError(
                "Portfolio action must be 5D"
            )

        # Parent expects 18D action.
        merged = np.zeros(
            18,
            dtype=np.float32,
        )

        merged[13:18] = action

        # Phase is already allocation.
        self.phase = 1

        result = super().step(
            merged
        )

        obs, reward, done, truncated, info = result

        # The portfolio reward trains the
        # retrieval controller.
        #
        # Its reward already includes
        # transaction/retrieval costs.
        self.router.update(reward)

        if not done:
            self._search()

        return (
            self._obs(),
            reward,
            done,
            truncated,
            info,
        )


def create_portfolio_model(
    agent,
    env,
    seed,
    args,
):
    common = dict(
        seed=seed,
        learning_rate=args.learning_rate,
        device=args.device,
        verbose=0,
    )

    if agent == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            n_steps=10,
            **common,
        )

    if agent == "PPO":
        return PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=64,
            **common,
        )

    if agent == "DDPG":
        noise = NormalActionNoise(
            mean=np.zeros(5),
            sigma=0.1 * np.ones(5),
        )

        return DDPG(
            "MlpPolicy",
            env,
            action_noise=noise,
            learning_starts=500,
            buffer_size=50000,
            batch_size=128,
            **common,
        )

    raise ValueError(agent)


class ProgressCheckpoint(BaseCallback):
    def __init__(
        self,
        folder,
        router,
        every=1000,
    ):
        super().__init__(verbose=0)

        self.folder = folder
        self.router = router
        self.every = every
        self.start = time.time()

        self.next_save = every

    def _on_step(self):
        steps = self.model.num_timesteps

        if steps >= self.next_save:
            self.folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.model.save(
                str(
                    self.folder / "checkpoint_model"
                )
            )

            self.router.save(
                self.folder / "checkpoint_router.pt"
            )

            save_json(
                self.folder / "progress.json",
                {
                    "timesteps": steps,
                    "elapsed_seconds":
                        time.time() - self.start,
                    "router_updates":
                        self.router.update_count,
                },
            )

            log(
                "[CHECKPOINT]",
                self.folder.name,
                f"{steps} steps",
                f"{time.time()-self.start:.1f}s",
                "router updates:",
                self.router.update_count,
            )

            self.next_save = (
                (steps // self.every + 1)
                * self.every
            )

        return True


def evaluate_model(model, env):
    obs, _ = env.reset()

    done = False

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, done, truncated, info = (
            env.step(action)
        )

        if truncated:
            break

    frame = pd.DataFrame(
        env.history
    )

    values = np.asarray(
        env.values,
        dtype=float,
    )

    returns = np.asarray(
        env.returns,
        dtype=float,
    )

    peak = np.maximum.accumulate(values)

    metrics = {
        "total_return": float(
            values[-1] / values[0] - 1
        ),
        "sharpe": float(
            returns.mean()
            / max(
                returns.std(),
                1e-12,
            )
            * np.sqrt(252)
        ),
        "max_drawdown": float(
            (values / peak - 1).min()
        ),
        "query_rate": float(
            np.mean([
                len(json.loads(x)) > 0
                for x in frame["queries"]
            ])
        ),
        "KG_hits": int(
            frame["KG_hits"].sum()
        ),
        "SEC_hits": int(
            frame["SEC_hits"].sum()
        ),
    }

    return metrics, frame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    parser.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    parser.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default=(
            "http://140.113.87.164:11434"
        ),
    )

    parser.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )

    parser.add_argument(
        "--embed",
        default="bge-m3:latest",
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=["A2C", "PPO", "DDPG"],
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "ACTIVE",
            "NEVER",
            "ALWAYS_BOTH",
        ],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[
            42, 123, 2026, 7, 999
        ],
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--router-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--router-entropy",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--cost-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tier",
        default="conservative",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v2"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    out = (
        ROOT / args.out
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("[LOAD] Market and Dynamic KG")

    _, _, schema, market, _, _ = (
        v2.read_original_data(args)
    )

    market = market.copy()

    market["date"] = pd.to_datetime(
        market["date"]
    )

    dates = sorted(
        market["date"].unique()
    )

    base_cols = schema[
        "FULL_DYNAMIC"
    ]

    kgt, kgt_cols = legacy.load_kgt(
        args.kgt,
        dates,
    )

    if len(kgt_cols) != 200:
        raise RuntimeError(
            "Expected actual 200D KGT"
        )

    kg = legacy.load_kg(
        args.kg
    )

    sec = legacy.load_sec(
        args.sec
    )

    log("[LOAD] Ollama")

    llm = CachedOllama(
        args.llm,
        args.embed,
        out / "llm_cache.sqlite",
        args.device,
        48,
        args.ollama_host,
    )

    retriever = IndexedTemporalRetrieval(
        kg,
        sec,
        llm,
        args.top_k,
    )

    log(
        "[READY]",
        "KG:", len(kg),
        "SEC:", len(sec),
        "KGT:", len(kgt_cols),
        "Semantic:", llm.dim,
    )

    def make_env(split, mode, router):
        return ActiveRAGEnvironment(
            market[
                market.split.eq(split)
            ],
            base_cols,
            kgt,
            retriever,
            llm,
            mode,
            args.max_search_steps,
            args.transaction_cost,
            args.cost_scale,
            split,
            router=router,
        )

    if args.smoke:
        # Force actual KG/SEC calls using the
        # retriever, independent of RL behavior.
        day = pd.Timestamp(
            "2021-06-01"
        )

        for source in [
            "KG",
            "SEC",
            "BOTH",
        ]:
            result, audit = retriever.retrieve(
                source,
                "NVDA",
                "SUPPLIES_TO",
                day,
                (
                    "NVIDIA manufacturing suppliers "
                    "semiconductor supply chain"
                ),
            )

            log(
                "[SMOKE]",
                source,
                "hits:", len(audit),
                "shape:", result.shape,
            )

            for item in audit[:2]:
                print(
                    item["source"],
                    item["available_at"],
                    item["text_excerpt"][:100],
                )

        log("[SMOKE] Completed")
        return

    rows = []

    for mode in args.modes:
        for agent in args.agents:
            for seed in args.seeds:

                name = (
                    f"{mode}__{agent}__seed{seed}"
                )

                folder = (
                    out / "runs" / name
                )

                folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                result_path = (
                    folder / "result.json"
                )

                if result_path.exists():
                    log("[RESUME COMPLETED]", name)

                    existing = json.loads(
                        result_path.read_text()
                    )

                    rows.extend(
                        existing["rows"]
                    )

                    continue

                log("[TRAIN]", name)

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                router = LearnedRouter(
                    state_dim=14,
                    lr=args.router_lr,
                    entropy_weight=
                        args.router_entropy,
                    training=(mode == "ACTIVE"),
                )

                train_env = make_env(
                    "train",
                    mode,
                    router,
                )

                checkpoint_model = (
                    folder / "checkpoint_model.zip"
                )

                checkpoint_router = (
                    folder / "checkpoint_router.pt"
                )

                if (
                    checkpoint_model.exists()
                    and checkpoint_router.exists()
                ):
                    log("[RESUME CHECKPOINT]", name)

                    model = {
                        "A2C": A2C,
                        "PPO": PPO,
                        "DDPG": DDPG,
                    }[agent].load(
                        str(checkpoint_model),
                        env=train_env,
                        device=args.device,
                    )

                    router.load(
                        checkpoint_router
                    )

                else:
                    model = create_portfolio_model(
                        agent,
                        train_env,
                        seed,
                        args,
                    )

                callback = ProgressCheckpoint(
                    folder,
                    router,
                    args.checkpoint_every,
                )

                remaining = max(
                    0,
                    args.timesteps
                    - model.num_timesteps,
                )

                if remaining > 0:
                    model.learn(
                        total_timesteps=remaining,
                        reset_num_timesteps=False,
                        callback=callback,
                        progress_bar=False,
                    )

                model.save(
                    str(folder / "model")
                )

                router.save(
                    folder / "router.pt"
                )

                run_rows = []

                for split in [
                    "val",
                    "test",
                ]:
                    eval_router = LearnedRouter(
                        state_dim=14,
                        lr=args.router_lr,
                        entropy_weight=
                            args.router_entropy,
                        training=False,
                    )

                    eval_router.load(
                        folder / "router.pt"
                    )

                    eval_env = make_env(
                        split,
                        mode,
                        eval_router,
                    )

                    metrics, history = evaluate_model(
                        model,
                        eval_env,
                    )

                    history.to_csv(
                        folder / f"{split}_queries.csv",
                        index=False,
                    )

                    record = {
                        "mode": mode,
                        "agent": agent,
                        "seed": seed,
                        "split": split,
                        **metrics,
                    }

                    run_rows.append(record)
                    rows.append(record)

                    log(
                        "[EVAL]",
                        name,
                        split,
                        "return:",
                        round(
                            metrics["total_return"],
                            5,
                        ),
                        "query_rate:",
                        round(
                            metrics["query_rate"],
                            4,
                        ),
                        "KG:",
                        metrics["KG_hits"],
                        "SEC:",
                        metrics["SEC_hits"],
                    )

                save_json(
                    result_path,
                    {
                        "rows": run_rows,
                    },
                )

                pd.DataFrame(
                    rows
                ).to_csv(
                    out / "all_results.csv",
                    index=False,
                )

    if rows:
        result = pd.DataFrame(rows)

        result.to_csv(
            out / "all_results.csv",
            index=False,
        )

        test = result[
            result.split.eq("test")
        ]

        summary = (
            test.groupby(
                ["mode", "agent"]
            )
            .agg(
                return_mean=(
                    "total_return", "mean"
                ),
                sharpe_mean=(
                    "sharpe", "mean"
                ),
                mdd_mean=(
                    "max_drawdown", "mean"
                ),
                query_rate=(
                    "query_rate", "mean"
                ),
                kg_hits=(
                    "KG_hits", "mean"
                ),
                sec_hits=(
                    "SEC_hits", "mean"
                ),
            )
            .reset_index()
        )

        summary.to_csv(
            out / "test_summary.csv",
            index=False,
        )

        print(
            summary.to_string(index=False)
        )

    log("FINRL-X ACTIVE RAG V2: PASS")


if __name__ == "__main__":
    main()
