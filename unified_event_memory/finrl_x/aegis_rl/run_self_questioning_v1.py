#!/usr/bin/env python3

"""
AEGIS-RL Self-Questioning V1

Two-phase portfolio RL:
  Phase 0: Decide company, relation, retrieval source.
  Phase 1: Observe KG-RAG / SEC-RAG evidence, allocate.

Algorithms:
  A2C, PPO, DDPG

Retrieval modes:
  ACTIVE
  NEVER
  ALWAYS_BOTH

Time:
  Train 2018-2020
  Validation 2021
  Test 2022-2023

Safety:
  Source available_at/published_at < decision date
  KGT date <= decision date
  Missing historical inputs fail closed.

Research caveats:
  - Structured queries, not generative LLM questions.
  - Lexical KG/SEC retrieval, not pretrained dense retrieval.
  - Input 200D KGT provenance must be audited separately.
  - Joint RL training with two phases;
    no guarantee of learning a useful retrieval policy.
"""

import argparse
import hashlib
import json
import random
import re
import sys
from functools import lru_cache
from pathlib import Path

import gymnasium as gym
import joblib
import numpy as np
import pandas as pd
import torch

from gymnasium import spaces
from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_portfolio_aware_v2 as v2
)

ACTIONS = ["SKIP", "KG", "SEC", "BOTH"]

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

TICKERS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
]

COMPANY_NAMES = {
    "AAPL": ["apple", "aapl"],
    "AMZN": ["amazon", "amzn"],
    "GOOGL": ["alphabet", "google", "googl"],
    "MSFT": ["microsoft", "msft"],
    "NVDA": ["nvidia", "nvda"],
}

REL_WORDS = {
    "SUPPLIES_TO": [
        "supplier", "supply", "supplies",
        "manufacturing", "component",
    ],
    "CUSTOMER_OF": [
        "customer", "client", "sales",
        "revenue", "buyer",
    ],
    "COMPETES_WITH": [
        "competitor", "compete", "competition",
        "market share",
    ],
    "SAME_SECTOR": [
        "sector", "industry", "peer",
        "semiconductor", "technology",
    ],
}

ACTION_COST_BPS = {
    "SKIP": 0.0,
    "KG": 0.3,
    "SEC": 0.8,
    "BOTH": 1.1,
}


def fail(message):
    raise RuntimeError(message)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            indent=2,
            default=str,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def norm_date(values):
    return pd.to_datetime(
        values,
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None).dt.normalize()


def choose_col(df, candidates, label, required=True):
    mapping = {
        c.lower(): c
        for c in df.columns
    }

    for name in candidates:
        if name.lower() in mapping:
            return mapping[name.lower()]

    if required:
        fail(
            f"{label}: cannot find column among "
            f"{candidates}. Actual: {list(df.columns)[:40]}"
        )

    return None


def read_table(path):
    path = Path(path)

    if not path.exists():
        fail(f"Input does not exist: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    if path.suffix == ".csv":
        return pd.read_csv(path)

    fail(
        f"Unsupported tabular format: {path}"
    )


def source_time(value):
    if pd.isna(value):
        return pd.NaT

    return pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    ).tz_convert(None)


def tokens(text):
    return set(
        re.findall(
            r"[a-z0-9_]+",
            str(text).lower(),
        )
    )


def query_text(ticker, relation):
    name = COMPANY_NAMES[ticker][0]

    return (
        f"{ticker} {name} {relation.lower()} "
        + " ".join(REL_WORDS[relation])
    )


def load_kg(path):
    raw = read_table(path)

    head = choose_col(
        raw,
        [
            "head", "head_name", "source",
            "source_entity", "subject",
            "src", "entity1",
        ],
        "KG head",
    )

    relation = choose_col(
        raw,
        [
            "relation", "relation_type",
            "predicate", "edge_type",
            "rel",
        ],
        "KG relation",
    )

    tail = choose_col(
        raw,
        [
            "tail", "tail_name", "target",
            "target_entity", "object",
            "dst", "entity2",
        ],
        "KG tail",
    )

    available = choose_col(
        raw,
        [
            "available_at", "published_at",
            "publication_date", "filing_date",
        ],
        "KG evidence availability",
    )

    evidence_id = choose_col(
        raw,
        [
            "evidence_id", "source_id", "event_id",
            "filing_id", "edge_id",
        ],
        "KG evidence ID",
        required=False,
    )

    records = []

    for i, row in raw.iterrows():
        dt = source_time(row[available])

        if pd.isna(dt):
            continue

        h = str(row[head])
        r = str(row[relation]).upper()
        t = str(row[tail])

        text = f"{h} {r} {t}"

        eid = (
            str(row[evidence_id])
            if evidence_id
            else f"kg_{i}"
        )

        records.append({
            "available_at": dt,
            "head": h,
            "relation": r,
            "tail": t,
            "text": text,
            "evidence_id": eid,
        })

    if not records:
        fail("No dated KG evidence found")

    return records


def load_sec(path):
    path = Path(path)

    if not path.exists():
        fail(f"SEC chunks missing: {path}")

    rows = []

    with path.open(
        encoding="utf-8"
    ) as stream:

        for i, line in enumerate(stream):
            if not line.strip():
                continue

            data = json.loads(line)

            meta = data.get("metadata", {})

            if not isinstance(meta, dict):
                meta = {}

            text = next(
                (
                    data[k]
                    for k in [
                        "text", "chunk_text",
                        "content", "page_content",
                    ]
                    if isinstance(data.get(k), str)
                ),
                None,
            )

            if not text:
                continue

            published = None

            for key in [
                "published_at",
                "available_at",
                "filing_date",
                "filed_at",
                "acceptance_datetime",
            ]:
                published = (
                    data.get(key)
                    or meta.get(key)
                )

                if published is not None:
                    break

            if published is None:
                continue

            dt = source_time(published)

            if pd.isna(dt):
                continue

            tic = (
                data.get("ticker")
                or data.get("tic")
                or meta.get("ticker")
                or meta.get("tic")
                or ""
            )

            doc_id = (
                data.get("chunk_id")
                or data.get("id")
                or meta.get("accession_number")
                or f"sec_{i}"
            )

            rows.append({
                "available_at": dt,
                "ticker": str(tic).upper(),
                "text": text,
                "evidence_id": str(doc_id),
                "form": str(
                    data.get("form")
                    or meta.get("form")
                    or ""
                ),
            })

    if not rows:
        fail(
            "No SEC chunks with usable text and "
            "publication timestamps. Check JSONL schema."
        )

    return rows


def load_kgt(path, expected_dates):
    raw = read_table(path)

    date_col = choose_col(
        raw,
        [
            "date", "as_of_date",
            "snapshot_date", "time",
        ],
        "KGT snapshot date",
    )

    ticker_col = choose_col(
        raw,
        ["ticker", "tic", "symbol"],
        "KGT ticker",
    )

    candidates = [
        c for c in raw.columns
        if re.fullmatch(
            r"(?:kg|kgt|emb|embedding)[_-]?\d+",
            str(c),
            re.IGNORECASE,
        )
    ]

    if len(candidates) != 200:
        fail(
            "Expected 200 KGT embedding dimensions; "
            f"found {len(candidates)}. "
            f"First columns: {list(raw.columns)[:25]}"
        )

    candidates = sorted(
        candidates,
        key=lambda s: int(
            re.search(r"\d+$", s).group()
        ),
    )

    df = raw[
        [date_col, ticker_col] + candidates
    ].copy()

    df = df.rename(columns={
        date_col: "date",
        ticker_col: "ticker",
    })

    df["date"] = norm_date(df["date"])
    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
    )

    if df["date"].isna().any():
        fail("Invalid KGT snapshot dates")

    if df.duplicated(
        ["date", "ticker"]
    ).any():
        fail("Duplicate KGT date/ticker pairs")

    df[candidates] = (
        df[candidates]
        .apply(pd.to_numeric, errors="coerce")
    )

    if not np.isfinite(
        df[candidates].to_numpy(float)
    ).all():
        fail("KGT contains missing/nonfinite values")

    # Fail closed: as-of snapshots must be
    # available for each decision date.
    lookup = {}

    for ticker in TICKERS:
        part = (
            df[df.ticker.eq(ticker)]
            .sort_values("date")
        )

        if part.empty:
            fail(f"Missing KGT ticker {ticker}")

        snapshots = (
            part["date"].to_numpy()
        )

        embeddings = part[
            candidates
        ].to_numpy(np.float32)

        for day in expected_dates:
            # Only snapshots strictly before
            # current trading date.
            idx = np.searchsorted(
                snapshots,
                np.datetime64(day),
                side="left",
            ) - 1

            if idx < 0:
                fail(
                    f"No past KGT snapshot for "
                    f"{ticker} on {day}"
                )

            lookup[
                (pd.Timestamp(day), ticker)
            ] = embeddings[idx]

    return lookup, candidates


class TemporalRetrievers:

    def __init__(self, kg, sec, top_k=3):
        self.kg = sorted(
            kg,
            key=lambda x: x["available_at"]
        )

        self.sec = sorted(
            sec,
            key=lambda x: x["available_at"]
        )

        self.top_k = top_k

    def score(self, item, query, day):
        q = tokens(query)
        t = tokens(item["text"])

        overlap = (
            len(q & t)
            / max(1, len(q))
        )

        age = (
            day - item["available_at"]
        ).days

        freshness = np.exp(
            -max(age, 0) / 180
        )

        return (
            0.8 * overlap
            + 0.2 * freshness
        )

    @lru_cache(maxsize=50000)
    def retrieve(
        self,
        source,
        ticker,
        relation,
        day_str,
    ):
        day = pd.Timestamp(day_str)

        query = query_text(
            ticker, relation
        )

        collection = (
            self.kg
            if source == "KG"
            else self.sec
        )

        aliases = COMPANY_NAMES[ticker]

        matched = []

        for item in collection:
            # Strict daily time gate.
            if item["available_at"] >= day:
                break

            searchable = (
                item["text"].lower()
            )

            if source == "KG":
                related = (
                    any(
                        a in searchable
                        for a in aliases
                    )
                )
            else:
                related = (
                    item["ticker"] == ticker
                    or any(
                        a in searchable
                        for a in aliases
                    )
                )

            if not related:
                continue

            score = self.score(
                item,
                query,
                day,
            )

            if source == "KG":
                if item["relation"] == relation:
                    score += 0.5

            matched.append(
                (score, item)
            )

        matched.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        return matched[:self.top_k]

    def encode(
        self,
        retrieved,
        day,
    ):
        if not retrieved:
            return np.zeros(
                8,
                dtype=np.float32,
            )

        scores = np.array([
            float(s) for s, _ in retrieved
        ])

        ages = np.array([
            max(
                0,
                (day - x["available_at"]).days,
            )
            for _, x in retrieved
        ])

        values = np.array([
            min(len(retrieved) / 3, 1),
            min(float(scores.max()), 2) / 2,
            min(float(scores.mean()), 2) / 2,
            np.exp(-ages.min() / 90),
            np.exp(-ages.mean() / 365),
            min(
                np.mean([
                    len(x["text"])
                    for _, x in retrieved
                ]) / 3000,
                1,
            ),
            min(
                len(set(
                    x["evidence_id"]
                    for _, x in retrieved
                )) / 3,
                1,
            ),
            min(
                np.std(scores), 1
            ),
        ], dtype=np.float32)

        return values

    def run(self, action, ticker, relation, day):
        feature = np.zeros(
            16,
            dtype=np.float32,
        )

        evidence_log = []

        if action in ["KG", "BOTH"]:
            kg_hits = self.retrieve(
                "KG",
                ticker,
                relation,
                str(day.date()),
            )

            feature[:8] = self.encode(
                kg_hits, day
            )

            evidence_log.extend([
                {
                    "source": "KG",
                    "id": item["evidence_id"],
                    "available_at":
                        str(item["available_at"]),
                    "score": float(score),
                }
                for score, item in kg_hits
            ])

        if action in ["SEC", "BOTH"]:
            sec_hits = self.retrieve(
                "SEC",
                ticker,
                relation,
                str(day.date()),
            )

            feature[8:] = self.encode(
                sec_hits, day
            )

            evidence_log.extend([
                {
                    "source": "SEC",
                    "id": item["evidence_id"],
                    "available_at":
                        str(item["available_at"]),
                    "score": float(score),
                }
                for score, item in sec_hits
            ])

        return feature, evidence_log


class SelfQuestioningEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        market,
        base_cols,
        kgt_lookup,
        retrievers,
        mode,
        transaction_cost,
        initial_amount=1_000_000,
    ):
        super().__init__()

        self.mode = mode
        self.retrievers = retrievers
        self.transaction_cost = transaction_cost
        self.initial_amount = initial_amount

        market = market[
            market.ticker.isin(TICKERS)
        ].copy()

        counts = market.groupby(
            "date"
        ).ticker.nunique()

        complete = counts[
            counts == len(TICKERS)
        ].index

        market = market[
            market.date.isin(complete)
        ]

        dates = sorted(
            market.date.unique()
        )

        if len(dates) < 2:
            fail("Environment has insufficient days")

        self.dates = [
            pd.Timestamp(d) for d in dates
        ]

        price = (
            market.pivot(
                index="date",
                columns="ticker",
                values="market_close",
            )
            .reindex(
                index=self.dates,
                columns=TICKERS,
            )
        )

        if price.isna().any().any():
            fail("Incomplete price matrix")

        self.prices = price.to_numpy(float)

        arrays = []

        for c in base_cols:
            matrix = (
                market.pivot(
                    index="date",
                    columns="ticker",
                    values=c,
                )
                .reindex(
                    index=self.dates,
                    columns=TICKERS,
                )
            )

            if matrix.isna().any().any():
                fail(f"Missing market feature: {c}")

            arrays.append(
                matrix.to_numpy(np.float32)
            )

        base = np.stack(
            arrays,
            axis=-1,
        )

        kg_emb = np.stack([
            np.stack([
                kgt_lookup[(d, ticker)]
                for ticker in TICKERS
            ])
            for d in self.dates
        ])

        self.features = np.concatenate(
            [base, np.clip(kg_emb, -5, 5)],
            axis=-1,
        ).astype(np.float32)

        self.n_base = self.features.shape[-1]

        self.market_volatility = (
            pd.Series(
                np.mean(
                    np.abs(
                        np.diff(
                            np.log(self.prices),
                            axis=0,
                        )
                    ),
                    axis=1,
                )
            )
        )

        # Action:
        # [0:4]   retrieval source logits
        # [4:9]   company logits
        # [9:13]  relation logits
        # [13:18] portfolio logits
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(18,),
            dtype=np.float32,
        )

        # State:
        # 5*(53+200) market/KGT
        # 16 retrieved evidence
        # 5 current weights
        # 3 meta (phase, volatility, gap)
        obs_dim = (
            5 * self.n_base
            + 16 + 5 + 3
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.reset()

    def _obs(self):
        day_features = (
            self.features[self.idx]
            .reshape(-1)
        )

        volatility = (
            float(
                self.market_volatility.iloc[
                    min(
                        max(self.idx - 1, 0),
                        len(self.market_volatility) - 1,
                    )
                ]
            )
        )

        # Explicit missing-knowledge proxy.
        coverage_gap = (
            1.0 if not np.any(
                self.retrieved[:8]
            ) else 0.0
        )

        meta = np.array([
            float(self.phase),
            volatility,
            coverage_gap,
        ], dtype=np.float32)

        return np.concatenate([
            day_features,
            self.retrieved,
            self.weights.astype(np.float32),
            meta,
        ]).astype(np.float32)

    @staticmethod
    def softmax(x):
        x = np.clip(
            np.asarray(x, dtype=float),
            -20, 20,
        )

        ex = np.exp(
            x - x.max()
        )

        return ex / ex.sum()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.idx = 0
        self.phase = 0

        self.weights = (
            np.ones(5, dtype=float) / 5
        )

        self.portfolio_value = (
            self.initial_amount
        )

        self.retrieved = np.zeros(
            16, dtype=np.float32
        )

        self.current_cost = 0
        self.current_query = None

        self.history = []
        self.portfolio_values = [
            self.portfolio_value
        ]

        self.returns = []

        return self._obs(), {}

    def step(self, action):
        action = np.asarray(
            action,
            dtype=float,
        ).reshape(-1)

        if len(action) != 18:
            fail("Invalid action dimension")

        day = self.dates[self.idx]

        if self.phase == 0:

            source_idx = int(
                np.argmax(action[:4])
            )

            company_idx = int(
                np.argmax(action[4:9])
            )

            relation_idx = int(
                np.argmax(action[9:13])
            )

            source = ACTIONS[source_idx]

            if self.mode == "NEVER":
                source = "SKIP"

            if self.mode == "ALWAYS_BOTH":
                source = "BOTH"

            ticker = TICKERS[company_idx]
            relation = RELATIONS[relation_idx]

            evidence, provenance = (
                self.retrievers.run(
                    source,
                    ticker,
                    relation,
                    day,
                )
            )

            self.retrieved = evidence

            self.current_cost = (
                ACTION_COST_BPS[source]
                / 10000.0
            )

            self.current_query = {
                "date": str(day.date()),
                "mode": self.mode,
                "action": source,
                "ticker": ticker,
                "relation": relation,
                "query": query_text(
                    ticker, relation
                ),
                "evidence": provenance,
                "retrieval_cost_fraction":
                    self.current_cost,
            }

            self.phase = 1

            # No market transition yet.
            return (
                self._obs(),
                0.0,
                False,
                False,
                {"phase": "retrieval"},
            )

        # PHASE 1: Allocation
        target = self.softmax(
            action[13:18]
        )

        old_weights = self.weights.copy()

        trade_cost = (
            self.transaction_cost
            * np.abs(
                target - old_weights
            ).sum()
        )

        r = (
            self.prices[self.idx + 1]
            / self.prices[self.idx]
            - 1.0
        )

        gross = float(
            np.dot(target, r)
        )

        net = (
            gross
            - trade_cost
            - self.current_cost
        )

        net = max(
            net,
            -0.999999,
        )

        self.portfolio_value *= (
            1 + net
        )

        post = target * (1 + r)
        denom = post.sum()

        self.weights = (
            post / denom
            if denom > 0
            else np.ones(5) / 5
        )

        record = {
            **self.current_query,
            "gross_return": gross,
            "net_return": net,
            "trade_cost": trade_cost,
            "portfolio_value":
                self.portfolio_value,
            **{
                f"weight_{ticker}": float(
                    target[j]
                )
                for j, ticker in enumerate(TICKERS)
            },
        }

        self.history.append(record)
        self.portfolio_values.append(
            self.portfolio_value
        )
        self.returns.append(net)

        self.idx += 1
        self.phase = 0

        self.retrieved = np.zeros(
            16, dtype=np.float32
        )

        done = (
            self.idx >= len(self.dates) - 1
        )

        # Keep terminal state valid.
        if done:
            self.idx = len(self.dates) - 1

        reward = float(
            np.log1p(net)
        )

        return (
            self._obs(),
            reward,
            done,
            False,
            {"phase": "allocation"},
        )


def make_model(agent, env, seed, args):
    if agent == "A2C":
        return A2C(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=args.learning_rate,
            n_steps=10,
            gamma=0.99,
            device=args.device,
            verbose=0,
        )

    if agent == "PPO":
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=args.learning_rate,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            device=args.device,
            verbose=0,
        )

    if agent == "DDPG":
        noise = NormalActionNoise(
            mean=np.zeros(18),
            sigma=0.1 * np.ones(18),
        )

        return DDPG(
            "MlpPolicy",
            env,
            seed=seed,
            learning_rate=args.learning_rate,
            action_noise=noise,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=128,
            device=args.device,
            verbose=0,
        )

    fail(f"Unknown RL model: {agent}")


def evaluate(model, env):
    obs, _ = env.reset()

    terminated = False

    while not terminated:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if truncated:
            break

    history = pd.DataFrame(
        env.history
    )

    returns = np.asarray(
        env.returns,
        dtype=float,
    )

    equity = np.asarray(
        env.portfolio_values,
        dtype=float,
    )

    total_return = (
        equity[-1] / equity[0] - 1
    )

    sharpe = (
        np.mean(returns)
        / (np.std(returns) + 1e-12)
        * np.sqrt(252)
    )

    running_max = np.maximum.accumulate(
        equity
    )

    mdd = np.min(
        equity / running_max - 1
    )

    return {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(mdd),
        "query_rate": float(
            (
                history["action"] != "SKIP"
            ).mean()
        ),
        "kg_rate": float(
            history["action"].isin(
                ["KG", "BOTH"]
            ).mean()
        ),
        "sec_rate": float(
            history["action"].isin(
                ["SEC", "BOTH"]
            ).mean()
        ),
        "retrieval_cost_total": float(
            history[
                "retrieval_cost_fraction"
            ].sum()
        ),
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kgt",
        required=True,
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
        "--tier",
        default="conservative",
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=["A2C", "PPO", "DDPG"],
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026, 7, 999],
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
        "--timesteps",
        type=int,
        default=50000,
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
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "self_questioning_v1"
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    output = (
        ROOT / args.out
    ).resolve()

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        args.kgt,
        args.kg,
        args.sec,
    ]:
        if not Path(path).exists():
            fail(
                f"Required input not found: {path}"
            )

    (
        v18,
        original,
        schema,
        market,
        evidence,
        manifest,
    ) = v2.read_original_data(
        args
    )

    base_cols = schema[
        "FULL_DYNAMIC"
    ]

    kg = load_kg(
        args.kg
    )

    sec = load_sec(
        args.sec
    )

    expected_dates = sorted(
        market["date"].unique()
    )

    kgt, kgt_columns = load_kgt(
        args.kgt,
        expected_dates,
    )

    retrievers = TemporalRetrievers(
        kg,
        sec,
        args.top_k,
    )

    print(
        "KG events:", len(kg)
    )
    print(
        "SEC chunks:", len(sec)
    )
    print(
        "KGT dimensions:", len(kgt_columns)
    )
    print(
        "Market dimensions:", len(base_cols)
    )

    config = {
        "name": "AEGIS_SELF_QUESTIONING_V1",
        "kgt": str(Path(args.kgt).resolve()),
        "kg": str(Path(args.kg).resolve()),
        "sec": str(Path(args.sec).resolve()),
        "algorithms": args.agents,
        "seeds": args.seeds,
        "modes": args.modes,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "transaction_cost": args.transaction_cost,
        "kgt_dimensions": len(kgt_columns),
        "base_dimensions": len(base_cols),
        "split": {
            "train": "2018-2020",
            "val": "2021",
            "test": "2022-2023",
        },
        "strict_pit_certified": False,
        "important_limitations": [
            "Historical embedding training provenance unverified",
            "Retrieval is lexical, not dense semantic search",
            "Self-questioning is structured action selection",
            "No LLM generation",
            "PIT filtering is day-level",
            "No OOF source representation validation",
        ],
    }

    save_json(
        output / "manifest.json",
        config,
    )

    def env_for(split, mode):
        subset = market[
            market.split.eq(split)
        ]

        return SelfQuestioningEnv(
            market=subset,
            base_cols=base_cols,
            kgt_lookup=kgt,
            retrievers=retrievers,
            mode=mode,
            transaction_cost=
                args.transaction_cost,
        )

    if args.smoke:
        env = env_for(
            "val", "ACTIVE"
        )

        obs, _ = env.reset()

        # Deterministically test BOTH +
        # NVDA + SUPPLIES_TO
        action = np.zeros(
            18,
            dtype=np.float32,
        )

        action[3] = 1
        action[8] = 1
        action[9] = 1

        obs, reward, done, trunc, info = (
            env.step(action)
        )

        print(
            "\nSMOKE RETRIEVAL:",
            env.current_query,
        )

        assert env.phase == 1
        assert len(obs) == (
            env.observation_space.shape[0]
        )

        print(
            "\nSELF-QUESTIONING SMOKE: PASS"
        )
        return

    results = []

    for mode in args.modes:
        for agent in args.agents:
            for seed in args.seeds:

                name = (
                    f"{mode}__{agent}__seed{seed}"
                )

                folder = output / "runs" / name

                folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                result_path = (
                    folder / "result.json"
                )

                if result_path.exists():
                    result = json.loads(
                        result_path.read_text()
                    )

                    results.extend(
                        result["rows"]
                    )

                    print(
                        "[RESUME]", name
                    )
                    continue

                print(
                    "\n[TRAIN]", name,
                    flush=True,
                )

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                train_env = env_for(
                    "train", mode
                )

                model = make_model(
                    agent,
                    train_env,
                    seed,
                    args,
                )

                model.learn(
                    total_timesteps=args.timesteps,
                    progress_bar=False,
                )

                model.save(
                    str(folder / "model")
                )

                rows = []

                for split in [
                    "val",
                    "test",
                ]:
                    test_env = env_for(
                        split, mode
                    )

                    result = evaluate(
                        model, test_env
                    )

                    history = result.pop(
                        "history"
                    )

                    history[
                        "evidence"
                    ] = history[
                        "evidence"
                    ].apply(json.dumps)

                    history.to_csv(
                        folder / (
                            f"{split}_queries.csv"
                        ),
                        index=False,
                    )

                    row = {
                        "mode": mode,
                        "agent": agent,
                        "seed": seed,
                        "split": split,
                        **result,
                    }

                    rows.append(row)
                    results.append(row)

                save_json(
                    result_path,
                    {"rows": rows},
                )

                pd.DataFrame(
                    results
                ).to_csv(
                    output / "all_results.csv",
                    index=False,
                )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output / "all_results.csv",
        index=False,
    )

    summary = (
        results_df[
            results_df.split.eq("test")
        ]
        .groupby(
            ["mode", "agent"]
        )
        .agg(
            return_mean=(
                "total_return",
                "mean",
            ),
            return_std=(
                "total_return",
                "std",
            ),
            sharpe_mean=(
                "sharpe",
                "mean",
            ),
            mdd_mean=(
                "max_drawdown",
                "mean",
            ),
            query_rate=(
                "query_rate",
                "mean",
            ),
            kg_rate=(
                "kg_rate",
                "mean",
            ),
            sec_rate=(
                "sec_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        output / "test_summary.csv",
        index=False,
    )

    print(
        "\n===== TEST SUMMARY ====="
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nSELF-QUESTIONING AEGIS V1: PASS"
    )


if __name__ == "__main__":
    main()
