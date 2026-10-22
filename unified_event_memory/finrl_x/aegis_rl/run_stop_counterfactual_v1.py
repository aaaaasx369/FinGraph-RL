#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X: One-step Financial Retrieval Counterfactual

Actions:
    STOP, KG, SEC, BOTH

Properties:
    - Same market day and previous portfolio weights
    - Same frozen A2C actor
    - One actual historical retrieval per non-STOP action
    - No retrieval or LLM call for STOP
    - Historical evidence timestamp audit
    - Trading and retrieval costs included
    - Same next-day stock returns for all branches
    - No training in this script

IMPORTANT:
    One-query counterfactual, not full multi-step FLARE.
    Historical future returns are used ONLY as labels.
    Validation/test labels must never train the router.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, PPO, DDPG

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v2 as v2,
    run_active_rag_v3 as v3,
    run_active_rag_v4 as v4,
)

SOURCES = ("STOP", "KG", "SEC", "BOTH")
TICKERS = ("AAPL", "AMZN", "GOOGL", "MSFT", "NVDA")


class StopRouter:
    def __init__(self):
        self.pending = []
        self.training = False

    def select(self, state):
        return "STOP", "AAPL", "SUPPLIES_TO"

    def update(self, reward):
        self.pending.clear()
        return None


def log(*args):
    print(*args, flush=True)


def date_string(value):
    return str(pd.Timestamp(value).date())


def stable_ticker(date):
    """
    Deterministic, pre-outcome candidate choice.
    It does not use next-day returns.
    """
    digest = hashlib.sha256(
        date_string(date).encode("utf-8")
    ).digest()

    return TICKERS[
        int.from_bytes(digest[:4], "big") % 5
    ]


def candidate_query(ticker):
    """
    Fixed template, so the first experiment
    does not need an LLM to generate questions.
    """
    return (
        f"{ticker} financial filings business risk "
        "suppliers customers competitors"
    )


def load_system(args):
    log("[INSTALL] V4 Gated Fusion environment")
    v4.install_v4()

    log("[LOAD] Historical market")
    _, _, schema, market, _, _ = (
        v2.v2.read_original_data(args)
    )

    market = market.copy()
    market["date"] = pd.to_datetime(
        market["date"]
    )

    dates = sorted(
        market["date"].unique()
    )

    log("[LOAD] Dynamic KGT")
    kgt, kgt_cols = v2.legacy.load_kgt(
        args.kgt, dates
    )

    if len(kgt_cols) != 200:
        raise RuntimeError(
            f"Expected KGT 200D, got {len(kgt_cols)}"
        )

    log("[LOAD] KG and SEC")
    kg = v2.legacy.load_kg(args.kg)
    sec = v2.legacy.load_sec(args.sec)

    log("[LOAD] Ollama embedding cache")
    llm = v2.CachedOllama(
        args.llm,
        args.embed,
        str(args.out / "llm_cache.sqlite"),
        args.device,
        48,
        args.ollama_host,
    )

    ret = v2.IndexedTemporalRetrieval(
        kg, sec, llm, args.top_k
    )

    split_market = market.loc[
        market["split"].eq(args.split)
    ].copy()

    if split_market.empty:
        raise RuntimeError(
            f"Empty split: {args.split}"
        )

    # The same constructor previously worked in
    # evaluate_v5_checkpoint.py.
    env = v3.ContextualActiveRAGEnv(
        split_market,
        schema["FULL_DYNAMIC"],
        kgt,
        ret,
        llm,
        "NEVER",
        1,
        args.transaction_cost,
        args.cost_scale,
        args.split,
        router=StopRouter(),
    )

    algorithm_classes = {
        "A2C": A2C,
        "PPO": PPO,
        "DDPG": DDPG,
    }

    model = algorithm_classes[args.algo].load(
        str(args.model),
        device=args.device,
    )

    extractor = getattr(
        model.policy,
        "features_extractor",
        None,
    )

    if extractor is None:
        extractor = getattr(
            getattr(model.policy, "actor", None),
            "features_extractor",
            None,
        )

    log(
        "[FEATURE EXTRACTOR]",
        type(extractor).__name__
        if extractor is not None
        else "not exposed at policy level",
    )

    if model.observation_space.shape != (3323,):
        raise RuntimeError(
            "Incorrect A2C observation dimensions"
        )

    if env.observation_space.shape != (3323,):
        raise RuntimeError(
            "Incorrect environment dimensions"
        )

    model.policy.set_training_mode(False)

    log("[READY] Model:", args.model)
    log("[READY] Observation:", env.observation_space.shape)
    log("[READY] KGT:", len(kgt_cols))

    return env, model, ret


def retrieval_cost(source, scale):
    if source == "STOP":
        return 0.0

    # This is the actual V4 retrieval cost table.
    return (
        float(v4.COST_BPS[source])
        / 10000.0
        * scale
    )


def historical_retrieve(
    retriever,
    source,
    ticker,
    relation,
    date,
    question,
):
    if source == "STOP":
        return (
            np.zeros((2, 1024), dtype=np.float32),
            [],
        )

    vector, provenance = retriever.retrieve(
        source,
        ticker,
        relation,
        pd.Timestamp(date),
        question,
    )

    vector = np.asarray(
        vector,
        dtype=np.float32,
    )

    if vector.shape != (2, 1024):
        raise RuntimeError(
            f"Invalid retrieved vector {vector.shape}"
        )

    if not np.isfinite(vector).all():
        raise RuntimeError(
            "Nonfinite retrieved evidence"
        )

    for item in provenance:
        available = pd.Timestamp(
            item["available_at"]
        )

        decision = pd.Timestamp(date)

        if available >= decision:
            raise RuntimeError(
                "LOOK-AHEAD VIOLATION: "
                f"{available} >= {decision}"
            )

    return vector, provenance


def score_branch(
    env,
    model,
    date_idx,
    previous_weights,
    source,
    vector,
    provenance,
    cost,
):
    """
    Every branch begins from the same reference
    portfolio and same market time.

    The actor observes the evidence BEFORE
    making its investment decision.

    No environment.step() is called, because
    that would advance the environment and
    destroy the identical starting state.
    """

    env.day = date_idx

    env.weights = previous_weights.copy()

    env.evidence[:] = vector

    env.phase = 1

    env.used = int(source != "STOP")

    env.steps = max(1, env.steps)

    env.confidence = 0.0

    env.last_hits = len(provenance)

    env.remaining_cost = float(cost)

    observation = env._obs()

    if observation.shape != (3323,):
        raise RuntimeError(
            f"Unexpected observation: {observation.shape}"
        )

    action, _ = model.predict(
        observation,
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=np.float32,
    ).reshape(-1)

    if action.shape != (5,):
        raise RuntimeError(
            "Expected five-dimensional A2C action"
        )

    # Original environment uses softmax
    # on the five portfolio action logits.
    weights = np.asarray(
        env.softmax(action),
        dtype=float,
    ).reshape(-1)

    if weights.shape != (5,):
        raise RuntimeError(
            "Expected five portfolio weights"
        )

    if not np.isfinite(weights).all():
        raise RuntimeError(
            "Nonfinite portfolio weights"
        )

    if not np.isclose(
        weights.sum(), 1.0, atol=1e-6
    ):
        raise RuntimeError(
            "Portfolio weights do not sum to one"
        )

    if (weights < 0).any():
        raise RuntimeError(
            "Negative weights in long-only portfolio"
        )

    current = np.asarray(
        env.price[date_idx],
        dtype=float,
    )

    following = np.asarray(
        env.price[date_idx + 1],
        dtype=float,
    )

    if (
        not np.isfinite(current).all()
        or not np.isfinite(following).all()
        or (current <= 0).any()
    ):
        raise RuntimeError(
            "Invalid next-day market prices"
        )

    asset_returns = following / current - 1.0

    gross = float(
        weights @ asset_returns
    )

    transaction_cost = (
        float(env.trade_cost)
        * float(
            np.abs(
                weights - previous_weights
            ).sum()
        )
    )

    net = max(
        gross - transaction_cost - cost,
        -0.999999,
    )

    reward = float(np.log1p(net))

    return {
        "weights": weights,
        "action": action,
        "gross_return": gross,
        "transaction_cost": transaction_cost,
        "retrieval_cost": float(cost),
        "net_return": float(net),
        "log_reward": reward,
        "n_evidence": len(provenance),
        "KG_hits": sum(
            p.get("source") == "KG"
            for p in provenance
        ),
        "SEC_hits": sum(
            p.get("source") == "SEC"
            for p in provenance
        ),
    }


def run(args):
    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    env, model, retriever = load_system(args)

    dates = list(env.dates)

    n = min(
        len(dates) - 1,
        len(env.price) - 1,
    )

    indices = list(
        range(0, n, args.stride)
    )

    if args.max_days > 0:
        indices = indices[:args.max_days]

    if not indices:
        raise RuntimeError("No eligible dates")

    rows = []

    for i, day_idx in enumerate(indices):
        date = pd.Timestamp(
            dates[day_idx]
        )

        if args.split == "train":
            if not 2018 <= date.year <= 2020:
                raise RuntimeError(
                    f"Train leakage: {date}"
                )
        elif args.split == "val":
            if date.year != 2021:
                raise RuntimeError(
                    f"Validation leakage: {date}"
                )
        elif args.split == "test":
            if date.year not in (2022, 2023):
                raise RuntimeError(
                    f"Test split mismatch: {date}"
                )

        # Explicitly defined same-state reference:
        # Equal-weight holdings at the beginning
        # of each counterfactual decision.
        #
        # These are independent local decisions,
        # NOT a continuous portfolio backtest.
        previous_weights = np.full(
            5, 0.2, dtype=float
        )

        ticker = stable_ticker(date)
        relation = "SUPPLIES_TO"
        question = candidate_query(ticker)

        branch_results = {}

        # Always evaluate STOP first.
        stop_vector = np.zeros(
            (2, 1024),
            dtype=np.float32,
        )

        branch_results["STOP"] = score_branch(
            env,
            model,
            day_idx,
            previous_weights,
            "STOP",
            stop_vector,
            [],
            0.0,
        )

        for source in ("KG", "SEC", "BOTH"):
            vector, provenance = (
                historical_retrieve(
                    retriever,
                    source,
                    ticker,
                    relation,
                    date,
                    question,
                )
            )

            cost = retrieval_cost(
                source,
                args.cost_scale,
            )

            branch_results[source] = (
                score_branch(
                    env,
                    model,
                    day_idx,
                    previous_weights,
                    source,
                    vector,
                    provenance,
                    cost,
                )
            )

        stop = branch_results["STOP"]

        for source in SOURCES:
            result = branch_results[source]

            delta = (
                result["log_reward"]
                - stop["log_reward"]
            )

            w_l1 = float(
                np.abs(
                    result["weights"]
                    - stop["weights"]
                ).sum()
            )

            row = {
                "date": date_string(date),
                "algorithm": args.algo,
                "split": args.split,
                "ticker": ticker,
                "relation": relation,
                "source": source,
                "query": (
                    ""
                    if source == "STOP"
                    else question
                ),
                "utility_vs_stop": delta,
                "stop_log_reward":
                    stop["log_reward"],
                "log_reward":
                    result["log_reward"],
                "gross_return":
                    result["gross_return"],
                "transaction_cost":
                    result["transaction_cost"],
                "retrieval_cost":
                    result["retrieval_cost"],
                "net_return":
                    result["net_return"],
                "n_evidence":
                    result["n_evidence"],
                "KG_hits":
                    result["KG_hits"],
                "SEC_hits":
                    result["SEC_hits"],
                "weight_l1_vs_stop": w_l1,
                "number_of_queries":
                    int(source != "STOP"),
                "reference_state":
                    "equal_weight_local",
                "one_query_only": True,
            }

            for j, ticker_name in enumerate(TICKERS):
                row[
                    f"weight_{ticker_name}"
                ] = float(
                    result["weights"][j]
                )

                row[
                    f"previous_weight_{ticker_name}"
                ] = float(
                    previous_weights[j]
                )

            rows.append(row)

        if (i + 1) % 5 == 0 or i == 0:
            log(
                f"[COUNTERFACTUAL] "
                f"{i+1}/{len(indices)} "
                f"{date.date()} "
                f"ticker={ticker}"
            )

        # Incremental checkpoint: save after
        # every complete set of four branches.
        if (i + 1) % args.save_every == 0:
            pd.DataFrame(rows).to_csv(
                args.out / "counterfactual_partial.csv",
                index=False,
            )

    frame = pd.DataFrame(rows)

    expected = len(indices) * 4

    if len(frame) != expected:
        raise RuntimeError(
            "Incorrect counterfactual row count"
        )

    if frame.duplicated(
        subset=["date", "source"]
    ).any():
        raise RuntimeError(
            "Duplicate date/source combinations"
        )

    for date, group in frame.groupby("date"):
        if set(group["source"]) != set(SOURCES):
            raise RuntimeError(
                f"Incomplete source actions: {date}"
            )

        stop = group[
            group["source"] == "STOP"
        ].iloc[0]

        if stop["retrieval_cost"] != 0:
            raise RuntimeError(
                "STOP incurred retrieval cost"
            )

        if stop["number_of_queries"] != 0:
            raise RuntimeError(
                "STOP issued a query"
            )

        if abs(stop["utility_vs_stop"]) > 1e-12:
            raise RuntimeError(
                "STOP utility must equal zero"
            )

    frame.to_csv(
        args.out / "financial_utility_labels.csv",
        index=False,
    )

    summary = (
        frame.groupby("source")
        .agg(
            samples=("utility_vs_stop", "size"),
            mean_utility=("utility_vs_stop", "mean"),
            median_utility=("utility_vs_stop", "median"),
            positive_utility_rate=(
                "utility_vs_stop",
                lambda x: float((x > 0).mean())
            ),
            mean_retrieval_cost=(
                "retrieval_cost", "mean"
            ),
            mean_weight_l1=(
                "weight_l1_vs_stop", "mean"
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        args.out / "utility_summary.csv",
        index=False,
    )

    print()
    print(
        "===== TRUE ONE-QUERY STOP COUNTERFACTUAL =====",
        flush=True,
    )

    print(
        summary.to_string(index=False),
        flush=True,
    )

    best = (
        frame.pivot(
            index="date",
            columns="source",
            values="utility_vs_stop",
        )
    )

    best_action = best.idxmax(axis=1)

    log()
    log(
        "[ORACLE DIAGNOSTIC] STOP chosen:",
        int((best_action == "STOP").sum()),
        "/",
        len(best_action),
    )

    log(
        "NOTE: Oracle uses realized next-day "
        "returns; it is NOT a deployable strategy."
    )

    log(
        "[SAVED]",
        args.out / "financial_utility_labels.csv",
    )

    log(
        "[SAVED]",
        args.out / "utility_summary.csv",
    )

    log(
        "ONE-QUERY STOP COUNTERFACTUAL: PASS"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--algo",
        choices=["A2C", "PPO", "DDPG"],
        default="A2C",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/"
            "runs/ACTIVE__A2C__seed42/model.zip"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "stop_counterfactual_v1"
        ),
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
        "--top-k",
        type=int,
        default=3,
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
        "--stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    if args.stride <= 0:
        raise RuntimeError(
            "--stride must be positive"
        )

    if args.save_every <= 0:
        raise RuntimeError(
            "--save-every must be positive"
        )

    args.out = args.out.resolve()

    run(args)


if __name__ == "__main__":
    main()
