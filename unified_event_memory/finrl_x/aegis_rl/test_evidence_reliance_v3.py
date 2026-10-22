#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V3 Evidence Reliance Test

Compare frozen A2C under:
    REAL
    ZERO
    SHUFFLED
    LAGGED

Controls:
    - Same frozen portfolio model
    - Same market split and Dynamic KGT
    - Same source/ticker/relation decisions
    - Same LLM questions from V3 logs
    - Same retrieval costs
    - Same initial portfolio capital

Only the evidence embedding changes.

SHUFFLED:
    Uses retrieved embeddings from earlier trading
    dates. Never uses evidence from future dates.

LAGGED:
    Uses retrieved embeddings from at least
    90 calendar days earlier.

Important:
    This measures frozen-policy evidence sensitivity.
    It is not a causal identification of RAG value.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import A2C

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v2 as v2
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v3 as v3
)


def numeric(x):
    return np.nan_to_num(
        np.asarray(x, dtype=np.float32),
        nan=0,
        posinf=0,
        neginf=0,
    )


def date_key(x):
    return str(pd.Timestamp(x).date())


def load_trace(path):
    frame = pd.read_csv(path)

    required = {
        "date",
        "queries",
        "n_queries",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            f"Trace missing columns: {missing}"
        )

    by_date = {}

    for _, row in frame.iterrows():
        date = date_key(row["date"])
        queries = json.loads(row["queries"])

        if len(queries) != int(row["n_queries"]):
            raise RuntimeError(
                f"Query count mismatch on {date}"
            )

        by_date[date] = queries

    return by_date


class FrozenRouter:
    """
    Replay recorded retrieval decisions exactly.
    """

    def __init__(self, trace):
        self.trace = trace
        self.env = None

        self.pending = []
        self.training = False

        self.current_date = None
        self.pointer = 0

    def select(self, features):
        if self.env is None:
            # Environment initialization may invoke
            # retrieval before the env reference
            # has been assigned.
            return (
                "STOP",
                v2.TICKERS[0],
                v2.RELATIONS[0],
            )

        date = date_key(
            self.env.dates[self.env.day]
        )

        if date != self.current_date:
            self.current_date = date
            self.pointer = 0

        queries = self.trace.get(date)

        if queries is None:
            raise RuntimeError(
                f"Missing retrieval trace: {date}"
            )

        if self.pointer >= len(queries):
            return (
                "STOP",
                v2.TICKERS[0],
                v2.RELATIONS[0],
            )

        q = queries[self.pointer]
        self.pointer += 1

        return (
            q["source"],
            q["ticker"],
            q["relation"],
        )

    def update(self, reward):
        self.pending.clear()


class FrozenLLM:
    """
    Replay historical Query and Follow-up text.

    No new LLM questions are generated:
    this prevents cross-arm prompt differences.
    """

    def __init__(self, actual_llm, trace):
        self.actual = actual_llm
        self.trace = trace

        self.env = None
        self.current_date = None
        self.call_index = 0

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def generate(self, prompt):
        date = date_key(
            self.env.dates[self.env.day]
        )

        if date != self.current_date:
            self.current_date = date
            self.call_index = 0

        queries = self.trace.get(date)

        if queries is None:
            raise RuntimeError(
                f"Missing LLM trace: {date}"
            )

        # Each recorded search has:
        #   1. query generation
        #   2. follow-up generation

        index = self.call_index // 2
        followup = self.call_index % 2

        if index >= len(queries):
            raise RuntimeError(
                f"Unexpected LLM request: {date}, "
                f"call={self.call_index}"
            )

        q = queries[index]

        self.call_index += 1

        if not followup:
            return (
                q["query"],
                float(
                    q.get(
                        "token_confidence",
                        0,
                    )
                ),
            )

        return (
            q.get("followup", ""),
            float(
                q.get(
                    "followup_confidence",
                    0,
                )
            ),
        )


class EvidenceIntervention:
    """
    Intercepts retrieve() at the embedding level.

    REAL:
        Retrieve using original retriever.

    ZERO:
        Replace semantic vectors with zeros.

    SHUFFLED:
        Replace with vectors from an earlier date.

    LAGGED:
        Replace with vectors from at least
        90 days earlier.

    Proceeds with the original provenance metadata
    to preserve the same retrieval-cost/hit
    accounting inside the portfolio environment.

    Therefore, reported KG/SEC hit counters in
    altered arms refer to original retrieval events,
    not to the injected evidence provenance.
    """

    def __init__(
        self,
        real_retriever,
        trace,
        mode,
        cache,
        lag_days=90,
        seed=42,
    ):
        self.actual = real_retriever
        self.trace = trace
        self.mode = mode
        self.cache = cache

        self.lag_days = lag_days
        self.rng = np.random.default_rng(seed)

        self.env = None

        self.current_date = None
        self.pointer = 0

        self.interventions = []

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def historical_vector(
        self,
        today,
        source,
        mode,
    ):
        candidates = []

        cutoff = pd.Timestamp(today)

        if mode == "LAGGED":
            cutoff -= pd.Timedelta(
                days=self.lag_days
            )

        for (day, idx), record in self.cache.items():
            past = pd.Timestamp(day)

            if past >= cutoff:
                continue

            evidence = record["vector"]

            if source == "KG":
                vec = evidence[0]
            else:
                vec = evidence[1]

            if np.linalg.norm(vec) <= 0:
                continue

            candidates.append((
                past,
                vec,
            ))

        if not candidates:
            return np.zeros(
                self.actual.llm.dim,
                dtype=np.float32,
            )

        if mode == "SHUFFLED":
            index = int(
                self.rng.integers(
                    len(candidates)
                )
            )
        else:
            # Most recent evidence that satisfies
            # the requested lag.
            index = max(
                range(len(candidates)),
                key=lambda i: candidates[i][0],
            )

        return numeric(
            candidates[index][1]
        ).copy()

    def retrieve(
        self,
        source,
        ticker,
        relation,
        date,
        question,
    ):
        day = date_key(date)

        if day != self.current_date:
            self.current_date = day
            self.pointer = 0

        index = self.pointer
        self.pointer += 1

        queries = self.trace.get(day, [])

        if index >= len(queries):
            raise RuntimeError(
                f"Unrecorded retrieval: {day} #{index}"
            )

        expected = queries[index]

        checks = [
            ("source", source),
            ("ticker", ticker),
            ("relation", relation),
            ("query", question),
        ]

        for field, value in checks:
            if expected[field] != value:
                raise RuntimeError(
                    f"Frozen trace mismatch: "
                    f"{day}, {field}, "
                    f"expected={expected[field]!r}, "
                    f"got={value!r}"
                )

        key = (day, index)

        if self.mode == "REAL":
            vector, audit = self.actual.retrieve(
                source,
                ticker,
                relation,
                pd.Timestamp(date),
                question,
            )

            vector = numeric(vector)

            if vector.ndim != 2 or vector.shape[0] != 2:
                raise RuntimeError(
                    "Unexpected evidence shape"
                )

            for item in audit:
                available = pd.Timestamp(
                    item["available_at"]
                )

                if available >= pd.Timestamp(date):
                    raise RuntimeError(
                        f"Future evidence: {day}"
                    )

            self.cache[key] = {
                "vector": vector.copy(),
                "audit": audit,
            }

        else:
            if key not in self.cache:
                raise RuntimeError(
                    f"REAL evidence not cached: {key}"
                )

            original = self.cache[key]

            audit = original["audit"]

            vector = numeric(
                original["vector"]
            ).copy()

            if self.mode == "ZERO":
                vector[:] = 0

            elif self.mode in (
                "SHUFFLED",
                "LAGGED",
            ):
                for row, evidence_source in [
                    (0, "KG"),
                    (1, "SEC"),
                ]:
                    # Keep absent sources absent.
                    if (
                        np.linalg.norm(
                            vector[row]
                        ) > 0
                    ):
                        vector[row] = (
                            self.historical_vector(
                                day,
                                evidence_source,
                                self.mode,
                            )
                        )

            else:
                raise ValueError(
                    self.mode
                )

        self.interventions.append({
            "date": day,
            "search_number": index,
            "source": source,
            "ticker": ticker,
            "relation": relation,
            "mode": self.mode,
            "evidence_norm": float(
                np.linalg.norm(vector)
            ),
        })

        return vector, audit


def calculate_metrics(values, daily_returns):
    values = np.asarray(
        values,
        dtype=float,
    )

    returns = np.asarray(
        daily_returns,
        dtype=float,
    )

    if len(values) < 2:
        raise RuntimeError(
            "Insufficient portfolio history"
        )

    peak = np.maximum.accumulate(
        values
    )

    return {
        "total_return": float(
            values[-1] / values[0] - 1
        ),
        "sharpe": float(
            np.mean(returns)
            / max(
                np.std(returns),
                1e-12,
            )
            * np.sqrt(252)
        ),
        "max_drawdown": float(
            np.min(values / peak - 1)
        ),
    }


def run_episode(
    model,
    env,
):
    obs, _ = env.reset()

    actions = []
    steps = 0

    while True:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = numeric(
            action
        ).reshape(-1)

        actions.append(
            action.copy()
        )

        obs, reward, done, truncated, info = (
            env.step(action)
        )

        steps += 1

        if done or truncated:
            break

        if steps > 10000:
            raise RuntimeError(
                "Episode exceeded 10000 steps"
            )

    history = pd.DataFrame(
        env.history
    )

    metrics = calculate_metrics(
        env.values,
        env.returns,
    )

    metrics["steps"] = steps
    metrics["retrieval_days"] = int(
        (history["n_queries"] > 0).sum()
    )

    metrics["total_queries"] = int(
        history["n_queries"].sum()
    )

    return (
        metrics,
        history,
        np.asarray(actions),
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
    )

    parser.add_argument(
        "--root",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v3"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "evidence_reliance_v3"
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
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
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--lag-days",
        type=int,
        default=90,
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

    args = parser.parse_args()

    run_root = Path(
        args.root
    ).resolve()

    folder = (
        run_root
        / "runs/ACTIVE__A2C__seed42"
    )

    out = Path(
        args.out
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace = load_trace(
        folder
        / f"{args.split}_queries.csv"
    )

    print(
        "[LOAD] Frozen retrieval trace:",
        len(trace),
        "days",
        flush=True,
    )

    print(
        "[LOAD] Market / Dynamic KGT",
        flush=True,
    )

    _, _, schema, market, _, _ = (
        v2.v2.read_original_data(
            args
        )
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

    kgt, kgt_cols = (
        v2.legacy.load_kgt(
            args.kgt,
            dates,
        )
    )

    if len(kgt_cols) != 200:
        raise RuntimeError(
            "Expected 200D Dynamic KGT"
        )

    kg = v2.legacy.load_kg(
        args.kg
    )

    sec = v2.legacy.load_sec(
        args.sec
    )

    llm = v2.CachedOllama(
        args.llm,
        args.embed,
        out / "llm_cache.sqlite",
        args.device,
        48,
        args.ollama_host,
    )

    retriever = (
        v2.IndexedTemporalRetrieval(
            kg,
            sec,
            llm,
            args.top_k,
        )
    )

    model = A2C.load(
        str(folder / "model.zip"),
        device=args.device,
    )

    cache = {}

    results = {}
    histories = {}
    allocations = {}

    for mode in [
        "REAL",
        "ZERO",
        "SHUFFLED",
        "LAGGED",
    ]:
        print(
            "\n" + "=" * 65,
            "\n[REPLAY]",
            mode,
            args.split,
            flush=True,
        )

        router = FrozenRouter(
            trace
        )

        env = v3.ContextualActiveRAGEnv(
            market[
                market.split.eq(
                    args.split
                )
            ],
            base_cols,
            kgt,
            retriever,
            llm,
            "ACTIVE",
            args.max_search_steps,
            args.transaction_cost,
            args.cost_scale,
            args.split,
            router=router,
        )

        router.env = env

        frozen_llm = FrozenLLM(
            llm,
            trace,
        )

        frozen_llm.env = env

        intervention = (
            EvidenceIntervention(
                retriever,
                trace,
                mode,
                cache,
                lag_days=args.lag_days,
            )
        )

        intervention.env = env

        env.llm = frozen_llm
        env.ret = intervention

        metrics, history, weights = (
            run_episode(
                model,
                env,
            )
        )

        # Make sure original retriever sequence
        # has the exact same number of calls.
        expected_searches = sum(
            len(x)
            for x in trace.values()
        )

        actual_searches = len(
            intervention.interventions
        )

        if actual_searches != expected_searches:
            raise RuntimeError(
                f"Query replay mismatch: "
                f"expected={expected_searches}, "
                f"actual={actual_searches}"
            )

        results[mode] = metrics
        histories[mode] = history
        allocations[mode] = weights

        history.to_csv(
            out
            / (
                f"{args.split}_{mode.lower()}"
                "_portfolio.csv"
            ),
            index=False,
        )

        pd.DataFrame(
            intervention.interventions
        ).to_csv(
            out
            / (
                f"{args.split}_{mode.lower()}"
                "_interventions.csv"
            ),
            index=False,
        )

        print(
            "[RESULT]",
            mode,
            json.dumps(
                metrics,
                ensure_ascii=False,
            ),
            flush=True,
        )

    real = allocations["REAL"]

    rows = []

    for mode, metrics in results.items():
        other = allocations[mode]

        if other.shape != real.shape:
            raise RuntimeError(
                "Portfolio action shape mismatch"
            )

        diff = np.abs(
            real - other
        )

        row = {
            "mode": mode,
            **metrics,
            "return_delta_vs_real": (
                metrics["total_return"]
                - results["REAL"][
                    "total_return"
                ]
            ),
            "mean_action_l1_vs_real": float(
                diff.sum(axis=1).mean()
            ),
            "max_action_l1_vs_real": float(
                diff.sum(axis=1).max()
            ),
            "fraction_actions_changed": float(
                np.mean(
                    diff.sum(axis=1) > 1e-6
                )
            ),
        }

        rows.append(row)

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        out
        / f"{args.split}_reliance_summary.csv",
        index=False,
    )

    print(
        "\n===== EVIDENCE RELIANCE SUMMARY =====",
        flush=True,
    )

    print(
        summary.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "\nEVIDENCE RELIANCE TEST: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
