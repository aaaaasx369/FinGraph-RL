#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V6 A2C:
Utility-guided Selective Evidence Fusion

Phase 1:
    Train source utility gate using 2018-2020
    paired, zero-retrieval-cost utility labels.

    Select a gate threshold on 2021 validation.

Phase 2:
    Frozen-actor, closed-loop portfolio backtest
    on 2022-2023 test.

Comparison:
    STOP / KG / SEC / BOTH / V6_SELECTIVE

Design:
    - Frozen V5-Full A2C
    - Same market prices
    - Same initial portfolio
    - Separate portfolio evolution per strategy
    - One query per non-STOP source
    - Zero retrieval costs
    - Actual portfolio transaction costs
    - No future returns in gate input
    - Test used only after threshold selection

This is NOT:
    - Full multi-step FLARE reproduction
    - Joint RL fine-tuning
    - Proof of causal effect for arbitrary policies
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from unified_event_memory.finrl_x.aegis_rl.run_stop_counterfactual_v1 import (
    load_system,
    historical_retrieve,
    score_branch,
    stable_ticker,
    candidate_query,
    date_string,
)

ACTIONS = ("STOP", "KG", "SEC", "BOTH")

BASE_POLICIES = (
    "STOP",
    "KG",
    "SEC",
    "BOTH",
)

TARGET_SOURCES = (
    "KG",
    "SEC",
    "BOTH",
)


def log(*args):
    print(*args, flush=True)


def as_date(value):
    return pd.Timestamp(value).normalize()


def finite(value, name):
    if not np.isfinite(value).all():
        raise RuntimeError(
            f"Nonfinite values in {name}"
        )
    return value


def market_state_features(prices, t):
    """
    Strictly past/current market information.

    No price[t+1] is used in this function.

    Uses 5 stocks:
        1-day log return
        5-day log return
        20-day log return
        20-day realized volatility
        20-day average daily log return

    Total: 25 market features.

    The classifier never observes retrieved
    evidence or next-day returns.
    """

    prices = np.asarray(
        prices, dtype=np.float64
    )

    if prices.ndim != 2:
        raise RuntimeError(
            f"Expected 2D price array: {prices.shape}"
        )

    if prices.shape[1] != 5:
        raise RuntimeError(
            "Expected 5 stock price columns"
        )

    if t < 0 or t >= len(prices):
        raise RuntimeError(
            f"Invalid market index: {t}"
        )

    if np.any(prices[:t + 1] <= 0):
        raise RuntimeError(
            "Non-positive historical prices"
        )

    log_prices = np.log(
        prices[:t + 1]
    )

    daily = np.diff(
        log_prices,
        axis=0,
    )

    def cumulative(window):
        if len(daily) == 0:
            return np.zeros(5)

        return daily[
            -min(window, len(daily)):
        ].sum(axis=0)

    r1 = cumulative(1)
    r5 = cumulative(5)
    r20 = cumulative(20)

    if len(daily) > 0:
        recent = daily[-20:]

        vol20 = recent.std(axis=0)
        mean20 = recent.mean(axis=0)
    else:
        vol20 = np.zeros(5)
        mean20 = np.zeros(5)

    x = np.concatenate([
        r1,
        r5,
        r20,
        vol20,
        mean20,
    ]).astype(np.float64)

    if x.shape != (25,):
        raise RuntimeError(
            f"Invalid gate features: {x.shape}"
        )

    return finite(
        x, "market gate features"
    )


def load_training_labels(path):
    """
    Labels are paired, same-state,
    cost-free log utilities.

    Trading costs remain included.
    """

    df = pd.read_csv(path)

    required = {
        "date",
        "source",
        "gross_return",
        "transaction_cost",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing label columns: {missing}"
        )

    if df.duplicated(
        ["date", "source"]
    ).any():
        raise RuntimeError(
            "Duplicate train labels"
        )

    if "retrieval_cost" not in df.columns:
        raise RuntimeError(
            "Training labels need retrieval_cost column"
        )

    if not np.allclose(
        df["retrieval_cost"].to_numpy(dtype=float),
        0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "V6 requires labels generated with "
            "--cost-scale 0. "
            "Do not just subtract the old costs."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    if not df["date"].dt.year.between(
        2018, 2020
    ).all():
        raise RuntimeError(
            "Training labels must be 2018-2020 only"
        )

    net_no_retrieval_cost = np.maximum(
        df["gross_return"].to_numpy(dtype=float)
        -
        df["transaction_cost"].to_numpy(dtype=float),
        -0.999999,
    )

    df["log_return_cost_free"] = (
        np.log1p(net_no_retrieval_cost)
    )

    pivot = df.pivot(
        index="date",
        columns="source",
        values="log_return_cost_free",
    )

    if set(pivot.columns) != set(ACTIONS):
        raise RuntimeError(
            "Missing one or more source actions"
        )

    if pivot.isna().any().any():
        raise RuntimeError(
            "Incomplete paired training labels"
        )

    utility = pivot.subtract(
        pivot["STOP"],
        axis=0,
    )

    utility = utility.sort_index()

    return utility


def make_training_matrix(
    train_env,
    train_utility,
):
    """
    Join each historical utility label to the
    gate feature available on that date.
    """

    index_by_date = {
        as_date(d): i
        for i, d in enumerate(
            train_env.dates
        )
    }

    X = []
    Y = []
    dates = []

    for date, row in train_utility.iterrows():
        key = as_date(date)

        if key not in index_by_date:
            raise RuntimeError(
                f"Training date not in market: {key}"
            )

        t = index_by_date[key]

        if t >= len(train_env.price) - 1:
            raise RuntimeError(
                f"No next-day data for label: {key}"
            )

        x = market_state_features(
            train_env.price,
            t,
        )

        y = row[
            list(TARGET_SOURCES)
        ].to_numpy(dtype=float)

        X.append(x)
        Y.append(y)
        dates.append(date_string(key))

    X = np.asarray(
        X, dtype=np.float64
    )

    Y = np.asarray(
        Y, dtype=np.float64
    )

    if len(X) < 30:
        raise RuntimeError(
            "Too few training labels"
        )

    finite(X, "training features")
    finite(Y, "training targets")

    log(
        "[TRAIN LABELS]",
        X.shape,
        Y.shape,
    )

    return X, Y, dates


def train_source_gate(X, Y, alpha):
    """
    Regularized multi-output regression.

    Small sample (151 dates), so deliberately
    uses a low-capacity linear gate rather
    than an oversized neural network.
    """

    model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "ridge",
            Ridge(
                alpha=alpha,
            ),
        ),
    ])

    model.fit(X, Y)

    pred = model.predict(X)

    log(
        "[GATE TRAIN MAE]",
        float(
            mean_absolute_error(Y, pred)
        ),
    )

    log(
        "[GATE TRAIN RMSE]",
        float(
            np.sqrt(
                mean_squared_error(Y, pred)
            )
        ),
    )

    return model


def gate_prediction(model, price, t):
    x = market_state_features(
        price, t
    )

    pred = np.asarray(
        model.predict(
            x.reshape(1, -1)
        ),
        dtype=float,
    ).reshape(-1)

    if pred.shape != (3,):
        raise RuntimeError(
            "Expected 3 predicted source utilities"
        )

    finite(pred, "gate predictions")

    return {
        "STOP": 0.0,
        "KG": float(pred[0]),
        "SEC": float(pred[1]),
        "BOTH": float(pred[2]),
    }


def choose_source(pred, threshold):
    """
    STOP if maximum predicted improvement
    does not exceed the threshold.

    Otherwise select predicted best source.

    No oracle labels or future returns.
    """

    candidate = max(
        TARGET_SOURCES,
        key=lambda s: pred[s],
    )

    if pred[candidate] <= threshold:
        return "STOP"

    return candidate


def new_portfolio_state():
    return {
        "weights": np.full(
            5, 0.2, dtype=float
        ),
        "equity": 1.0,
        "equity_curve": [1.0],
        "daily_returns": [],
        "history": [],
    }


def update_portfolio_state(
    state,
    result,
    asset_returns,
):
    """
    Matches the original environment:

    net = gross - transaction - retrieval
    equity *= (1 + net)

    Following the market price move, the
    actual next-day holdings drift.
    """

    net = float(
        result["net_return"]
    )

    state["equity"] *= (1.0 + net)

    state["equity_curve"].append(
        float(state["equity"])
    )

    state["daily_returns"].append(net)

    w = np.asarray(
        result["weights"],
        dtype=float,
    )

    after = w * (
        1.0 + asset_returns
    )

    total = float(
        after.sum()
    )

    if total > 0:
        next_weights = after / total
    else:
        next_weights = np.full(
            5, 0.2
        )

    state["weights"] = next_weights


def portfolio_metrics(state):
    equity = np.asarray(
        state["equity_curve"],
        dtype=float,
    )

    returns = np.asarray(
        state["daily_returns"],
        dtype=float,
    )

    peak = np.maximum.accumulate(
        equity
    )

    mdd = float(
        np.min(
            equity / peak - 1.0
        )
    )

    std = float(
        returns.std()
    )

    sharpe = float(
        returns.mean()
        /
        max(std, 1e-12)
        * np.sqrt(252)
    )

    if not np.isfinite(sharpe):
        raise RuntimeError(
            "Nonfinite Sharpe"
        )

    return {
        "days": int(len(returns)),
        "total_return": float(
            equity[-1] / equity[0] - 1.0
        ),
        "final_multiple": float(
            equity[-1] / equity[0]
        ),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "annualized_volatility": float(
            std * np.sqrt(252)
        ),
    }


def check_retrieval_vector(
    source,
    vector,
):
    vector = np.asarray(
        vector, dtype=np.float32
    )

    if vector.shape != (2, 1024):
        raise RuntimeError(
            f"{source} invalid evidence shape"
        )

    if source == "KG":
        if np.linalg.norm(
            vector[1]
        ) > 1e-8:
            raise RuntimeError(
                "KG retrieval contains SEC evidence"
            )

    if source == "SEC":
        if np.linalg.norm(
            vector[0]
        ) > 1e-8:
            raise RuntimeError(
                "SEC retrieval contains KG evidence"
            )

    return vector


def backtest(
    env,
    actor,
    retriever,
    gate,
    thresholds,
    split,
    max_days,
    out,
):
    """
    Backtest all strategies concurrently.

    Every strategy:
        - starts from equal weights
        - evolves its own portfolio
        - uses the same underlying market prices
        - calls the same frozen actor
        - uses a different evidence source

    Retrieval is performed once per
    (date, source), then shared across
    portfolio policy evaluations.
    """

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    policies = list(BASE_POLICIES)

    for i in range(len(thresholds)):
        policies.append(
            f"V6_THRESHOLD_{i}"
        )

    states = {
        name: new_portfolio_state()
        for name in policies
    }

    dates = list(env.dates)

    n = min(
        len(dates) - 1,
        len(env.price) - 1,
    )

    if max_days > 0:
        n = min(n, max_days)

    if n < 1:
        raise RuntimeError(
            f"No backtest days: {split}"
        )

    records = []

    for t in range(n):
        date = pd.Timestamp(
            dates[t]
        )

        if split == "val":
            if date.year != 2021:
                raise RuntimeError(
                    f"Invalid validation date {date}"
                )

        elif split == "test":
            if date.year not in (2022, 2023):
                raise RuntimeError(
                    f"Invalid test date {date}"
                )

        ticker = stable_ticker(date)

        relation = "SUPPLIES_TO"

        question = candidate_query(
            ticker
        )

        predicted = gate_prediction(
            gate,
            env.price,
            t,
        )

        choices = {
            "STOP": "STOP",
            "KG": "KG",
            "SEC": "SEC",
            "BOTH": "BOTH",
        }

        for i, threshold in enumerate(
            thresholds
        ):
            name = f"V6_THRESHOLD_{i}"

            choices[name] = choose_source(
                predicted,
                threshold,
            )

        # Source evidence is shared on the same
        # date but each portfolio uses its own
        # pre-trade holdings.
        evidence_cache = {
            "STOP": (
                np.zeros(
                    (2, 1024),
                    dtype=np.float32,
                ),
                [],
            ),
        }

        needed = set(
            choices.values()
        )

        for source in TARGET_SOURCES:
            if source not in needed:
                continue

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

            evidence_cache[source] = (
                check_retrieval_vector(
                    source,
                    vector,
                ),
                provenance,
            )

        # The same realized asset returns
        # are used across all policies.
        asset_returns = (
            np.asarray(
                env.price[t + 1],
                dtype=float,
            )
            /
            np.asarray(
                env.price[t],
                dtype=float,
            )
            - 1.0
        )

        for policy_name in policies:
            source = choices[
                policy_name
            ]

            state = states[
                policy_name
            ]

            previous_weights = (
                state["weights"].copy()
            )

            vector, provenance = (
                evidence_cache[source]
            )

            # All retrieval costs are zero.
            result = score_branch(
                env,
                actor,
                t,
                previous_weights,
                source,
                vector,
                provenance,
                0.0,
            )

            if result["retrieval_cost"] != 0:
                raise RuntimeError(
                    "Zero-cost experiment violated"
                )

            weight_turnover = float(
                np.abs(
                    result["weights"]
                    - previous_weights
                ).sum()
            )

            update_portfolio_state(
                state,
                result,
                asset_returns,
            )

            row = {
                "date": date_string(date),
                "split": split,
                "policy": policy_name,
                "source": source,
                "ticker": ticker,
                "relation": relation,
                "net_return":
                    float(result["net_return"]),
                "gross_return":
                    float(result["gross_return"]),
                "transaction_cost":
                    float(result["transaction_cost"]),
                "retrieval_cost": 0.0,
                "turnover": weight_turnover,
                "equity": float(
                    state["equity"]
                ),
                "n_evidence":
                    int(result["n_evidence"]),
                "KG_hits":
                    int(result["KG_hits"]),
                "SEC_hits":
                    int(result["SEC_hits"]),
                "pred_KG": predicted["KG"],
                "pred_SEC": predicted["SEC"],
                "pred_BOTH": predicted["BOTH"],
            }

            for j, ticker_name in enumerate(
                ("AAPL", "AMZN", "GOOGL", "MSFT", "NVDA")
            ):
                row[
                    f"weight_{ticker_name}"
                ] = float(
                    result["weights"][j]
                )

            records.append(row)

        if (t + 1) % 10 == 0 or t == 0:
            log(
                f"[{split.upper()}] "
                f"{t+1}/{n} "
                f"{date.date()} "
                f"V6={choices['V6_THRESHOLD_0']}"
            )

        if (t + 1) % 25 == 0:
            pd.DataFrame(
                records
            ).to_csv(
                out / "partial_history.csv",
                index=False,
            )

    history = pd.DataFrame(
        records
    )

    history.to_csv(
        out / "daily_portfolio_history.csv",
        index=False,
    )

    summary_rows = []

    for policy_name, state in states.items():
        metrics = portfolio_metrics(
            state
        )

        one = history[
            history.policy == policy_name
        ]

        selection = (
            one.source.value_counts()
        )

        metrics.update({
            "split": split,
            "policy": policy_name,
            "mean_turnover": float(
                one.turnover.mean()
            ),
            "total_turnover": float(
                one.turnover.sum()
            ),
            "mean_transaction_cost": float(
                one.transaction_cost.mean()
            ),
            "retrieval_days": int(
                (one.source != "STOP").sum()
            ),
            "KG_days": int(
                selection.get("KG", 0)
            ),
            "SEC_days": int(
                selection.get("SEC", 0)
            ),
            "BOTH_days": int(
                selection.get("BOTH", 0)
            ),
            "STOP_days": int(
                selection.get("STOP", 0)
            ),
        })

        summary_rows.append(
            metrics
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        out / "portfolio_summary.csv",
        index=False,
    )

    log()
    log(
        f"===== {split.upper()} PORTFOLIO RESULTS ====="
    )

    columns = [
        "policy",
        "days",
        "total_return",
        "sharpe",
        "max_drawdown",
        "mean_turnover",
        "retrieval_days",
    ]

    log(
        summary[columns].to_string(
            index=False
        )
    )

    return history, summary


def choose_validation_threshold(
    validation_summary,
    thresholds,
):
    """
    Validation-only model selection.

    The maximum validation final multiple
    determines the frozen threshold.

    Test is not used.
    """

    candidates = validation_summary[
        validation_summary["policy"].str.startswith(
            "V6_THRESHOLD_"
        )
    ].copy()

    if len(candidates) != len(thresholds):
        raise RuntimeError(
            "Missing validation thresholds"
        )

    candidates = candidates.sort_values(
        [
            "final_multiple",
            "mean_turnover",
        ],
        ascending=[
            False,
            True,
        ],
    )

    winner = candidates.iloc[0]

    policy = str(
        winner["policy"]
    )

    index = int(
        policy.rsplit("_", 1)[-1]
    )

    threshold = float(
        thresholds[index]
    )

    return index, threshold, winner


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-labels",
        default=(
            "results/aegis_rl_v1/"
            "stop_counterfactual_train/"
            "financial_utility_labels.csv"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v6_a2c_selective_fusion"
        ),
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--device",
        default="cpu",
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
        default=0.0,
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
        "--learning-rate",
        type=float,
        default=3e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # V6 currently uses the frozen V5-Full A2C.
    # Required by the three-RL counterfactual loader.
    args.algo = "A2C"

    if args.cost_scale != 0:
        raise RuntimeError(
            "V6 zero-cost protocol requires "
            "--cost-scale 0"
        )

    if args.alpha <= 0:
        raise RuntimeError(
            "--alpha must be positive"
        )

    if not args.model.is_file():
        raise RuntimeError(
            f"Missing A2C: {args.model}"
        )

    if not Path(
        args.train_labels
    ).is_file():
        raise RuntimeError(
            f"Missing training labels: "
            f"{args.train_labels}"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.out = args.out.resolve()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 78)
    log(
        "V6 A2C — SELECTIVE EVIDENCE FUSION"
    )
    log("=" * 78)

    log("[ACTOR]", args.model)
    log("[TRAIN]", "2018-2020")
    log("[VAL]", "2021")
    log("[TEST]", "2022-2023")
    log("[RETRIEVAL COST]", 0.0)
    log("[ACTOR FROZEN]", True)

    # --------------------------------------------------
    # Phase 1: train gate
    # --------------------------------------------------

    args.split = "train"

    train_env, actor, _ = load_system(
        args
    )

    labels = load_training_labels(
        args.train_labels
    )

    X, Y, train_dates = (
        make_training_matrix(
            train_env,
            labels,
        )
    )

    gate = train_source_gate(
        X,
        Y,
        args.alpha,
    )

    train_predictions = gate.predict(
        X
    )

    pd.DataFrame({
        "date": train_dates,
        "actual_KG": Y[:, 0],
        "actual_SEC": Y[:, 1],
        "actual_BOTH": Y[:, 2],
        "predicted_KG": train_predictions[:, 0],
        "predicted_SEC": train_predictions[:, 1],
        "predicted_BOTH": train_predictions[:, 2],
    }).to_csv(
        args.out / "gate_train_predictions.csv",
        index=False,
    )

    import joblib

    joblib.dump(
        gate,
        args.out / "source_gate.joblib",
    )

    log(
        "[SAVED]",
        args.out / "source_gate.joblib",
    )

    # --------------------------------------------------
    # Phase 1: validation threshold selection
    # --------------------------------------------------

    # Units are daily log-return differences.
    # Thresholds are fixed BEFORE validation.
    thresholds = [
        -0.000005,
        0.0,
        0.000001,
        0.000003,
        0.000005,
    ]

    args.split = "val"

    val_env, val_actor, val_retriever = (
        load_system(args)
    )

    val_history, val_summary = backtest(
        val_env,
        val_actor,
        val_retriever,
        gate,
        thresholds,
        "val",
        args.max_days,
        args.out / "validation",
    )

    selected_idx, selected_threshold, winner = (
        choose_validation_threshold(
            val_summary,
            thresholds,
        )
    )

    log()
    log("===== VALIDATION SELECTION =====")
    log(
        "Selected threshold:",
        selected_threshold,
    )
    log(
        "Validation total return:",
        float(winner["total_return"]),
    )
    log(
        "Validation Sharpe:",
        float(winner["sharpe"]),
    )

    selection = {
        "selected_index": selected_idx,
        "selected_threshold": selected_threshold,
        "validation_total_return": float(
            winner["total_return"]
        ),
        "validation_sharpe": float(
            winner["sharpe"]
        ),
        "train_samples": len(X),
        "gate_alpha": args.alpha,
        "gate_input_dim": X.shape[1],
        "cost_scale": 0.0,
        "actor_model": str(args.model),
        "smoke_test": bool(
            args.max_days > 0
        ),
    }

    (
        args.out / "validation_selection.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # Phase 2: test
    # --------------------------------------------------

    # Test uses ONLY the selected threshold.
    args.split = "test"

    test_env, test_actor, test_retriever = (
        load_system(args)
    )

    test_history, test_summary = backtest(
        test_env,
        test_actor,
        test_retriever,
        gate,
        [selected_threshold],
        "test",
        args.max_days,
        args.out / "test",
    )

    # Rename V6 name for final report.
    test_summary["policy"] = (
        test_summary["policy"].replace({
            "V6_THRESHOLD_0": "V6_SELECTIVE"
        })
    )

    test_summary.to_csv(
        args.out / "test"
        / "portfolio_summary.csv",
        index=False,
    )

    test_history["policy"] = (
        test_history["policy"].replace({
            "V6_THRESHOLD_0": "V6_SELECTIVE"
        })
    )

    test_history.to_csv(
        args.out / "test"
        / "daily_portfolio_history.csv",
        index=False,
    )

    baseline = test_summary[
        test_summary.policy == "BOTH"
    ].iloc[0]

    v6 = test_summary[
        test_summary.policy == "V6_SELECTIVE"
    ].iloc[0]

    delta_return = float(
        v6["total_return"]
        - baseline["total_return"]
    )

    delta_sharpe = float(
        v6["sharpe"]
        - baseline["sharpe"]
    )

    final = {
        "actor": "V5_FULL_A2C_FROZEN",
        "test_days": int(
            v6["days"]
        ),
        "selected_threshold": selected_threshold,
        "both_total_return": float(
            baseline["total_return"]
        ),
        "v6_total_return": float(
            v6["total_return"]
        ),
        "delta_total_return": delta_return,
        "both_sharpe": float(
            baseline["sharpe"]
        ),
        "v6_sharpe": float(
            v6["sharpe"]
        ),
        "delta_sharpe": delta_sharpe,
        "both_mdd": float(
            baseline["max_drawdown"]
        ),
        "v6_mdd": float(
            v6["max_drawdown"]
        ),
        "smoke_test": bool(
            args.max_days > 0
        ),
    }

    (
        args.out / "final_result.json"
    ).write_text(
        json.dumps(
            final, indent=2
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 78)
    log("FINAL V6 VS FIXED BOTH")
    log("=" * 78)

    log(
        "Test days:",
        final["test_days"],
    )

    log(
        "BOTH return:",
        final["both_total_return"],
    )

    log(
        "V6 return:",
        final["v6_total_return"],
    )

    log(
        "Delta return:",
        final["delta_total_return"],
    )

    log(
        "Delta Sharpe:",
        final["delta_sharpe"],
    )

    log(
        "BOTH MDD:",
        final["both_mdd"],
    )

    log(
        "V6 MDD:",
        final["v6_mdd"],
    )

    log()
    log(
        "[SAVED]",
        args.out / "final_result.json",
    )

    log(
        "V6 END-TO-END:",
        "SMOKE PASS"
        if args.max_days > 0
        else "COMPLETE",
    )


if __name__ == "__main__":
    main()
