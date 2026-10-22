#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V10 - Reflection Retrieval Quality Audit

Dataset:
  2018-2019: historical representation calibration
  2020: walk-forward evaluation
  2021: not evaluated
  2022-2023: never accessed for labels

Inputs:
  SEC semantic retrieval: 64D / company
  Temporal KG: 8D / company
  Market price context: 5D / company
  Company identity: 5D

Methods:
  KNN_REFLECTION
  RECENT
  RANDOM
  HIST_MEAN
  ZERO

Prediction target:
  5-day stock return minus equal-weight
  5-stock benchmark return over same period.

A historical case becomes retrievable ONLY when:
  maturity_index < decision_index

This is a research audit, not trading advice.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from unified_event_memory.finrl_x.aegis_rl import (
    run_v9_reflection_kg_rag as v9,
    run_v8_phase2b as v8b,
    run_v8_replug_finagent as v8,
)


TICKERS = list(v9.TICKERS)
N_STOCKS = len(TICKERS)
HORIZON = 5

METHODS = [
    "KNN_REFLECTION",
    "RECENT",
    "RANDOM",
    "HIST_MEAN",
    "ZERO",
]


def log(*x):
    print(*x, flush=True)


def zsafe(x):
    if isinstance(x, dict):
        return {str(k): zsafe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [zsafe(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def safe_ic(actual, prediction):
    a = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)

    if len(a) < 3:
        return np.nan

    if np.ptp(a) < 1e-12 or np.ptp(p) < 1e-12:
        return np.nan

    value = spearmanr(a, p).statistic
    return float(value) if np.isfinite(value) else np.nan


def normalized_weights(distances):
    """
    Convert scaled Euclidean distances to positive
    retrieval weights.

    Weight temperature is fixed in advance,
    not tuned on 2020 outcomes.
    """
    d = np.asarray(distances, dtype=np.float64)
    z = -(d - d.min())
    z = np.clip(z, -40, 0)
    w = np.exp(z)
    w /= w.sum()
    return w


def case_confidence(outcomes, weights, distances):
    """
    Descriptive confidence diagnostics only.

    Not a calibrated probability and not yet used
    to decide portfolio allocations.
    """
    values = np.asarray(outcomes, dtype=float)
    w = np.asarray(weights, dtype=float)
    d = np.asarray(distances, dtype=float)

    mu = float(np.dot(w, values))
    var = float(np.dot(w, (values - mu) ** 2))
    std = float(np.sqrt(max(var, 0.0)))

    n_eff = float(
        1.0 / max(np.sum(w ** 2), 1e-12)
    )

    pos = float(
        w[values > 0].sum()
    )
    neg = float(
        w[values < 0].sum()
    )

    agreement = max(pos, neg)

    # Heuristic reliability measure. No claim of
    # probabilistic calibration.
    consistency = abs(pos - neg)
    proximity = float(
        np.exp(-np.sqrt(max(float(d.min()), 0.0)))
    )

    confidence = (
        consistency
        * proximity
        * min(n_eff / len(w), 1.0)
    )

    return {
        "mean": mu,
        "std": std,
        "effective_cases": n_eff,
        "sign_agreement": agreement,
        "confidence_proxy": float(confidence),
        "min_distance": float(d.min()),
    }


def calculate_metrics(df, method):
    sub = df[df["method"] == method].copy()

    a = sub["actual"].to_numpy(dtype=float)
    p = sub["prediction"].to_numpy(dtype=float)

    mse = float(np.mean((a - p) ** 2))
    mae = float(np.mean(np.abs(a - p)))

    # Target is EXCESS return, not raw return.
    sign_accuracy = float(
        np.mean((a > 0) == (p > 0))
    )

    daily = []

    for date, group in sub.groupby("date", sort=True):
        if group["ticker"].nunique() != N_STOCKS:
            continue

        group = group.set_index("ticker").loc[TICKERS]

        aa = group["actual"].to_numpy()
        pp = group["prediction"].to_numpy()

        ic = safe_ic(aa, pp)

        # Difference between best and worst model-ranked
        # stock's realized excess return.
        # This is a gross ranking diagnostic,
        # NOT a tradeable net portfolio backtest.
        long_short = (
            aa[np.argmax(pp)] - aa[np.argmin(pp)]
            if np.ptp(pp) > 1e-12
            else 0.0
        )

        daily.append({
            "date": date,
            "rank_ic": ic,
            "gross_rank_spread": float(long_short),
        })

    daily = pd.DataFrame(daily)

    return {
        "method": method,
        "samples": len(sub),
        "days": sub["date"].nunique(),
        "mse": mse,
        "mae": mae,
        "sign_accuracy": sign_accuracy,
        "mean_rank_ic": (
            float(daily["rank_ic"].mean())
            if len(daily) else np.nan
        ),
        "valid_rank_ic_days": (
            int(daily["rank_ic"].notna().sum())
            if len(daily) else 0
        ),
        "mean_gross_rank_spread": (
            float(daily["gross_rank_spread"].mean())
            if len(daily) else np.nan
        ),
    }, daily


def paired_block_bootstrap(
    a,
    b,
    metric,
    seed,
    block=10,
    repeats=2000,
):
    """
    Paired moving-block bootstrap over dates.

    For error metrics:
       improvement = baseline error - KNN error

    For Rank IC:
       improvement = KNN IC - baseline IC

    Correlated overlapping five-day returns
    are handled approximately by block sampling.
    """

    aa = a[["date", metric]].rename(
        columns={metric: "knn"}
    )

    bb = b[["date", metric]].rename(
        columns={metric: "baseline"}
    )

    joint = aa.merge(
        bb,
        on="date",
        how="inner",
        validate="one_to_one",
    ).dropna()

    if metric in ("squared_error", "abs_error"):
        delta = (
            joint["baseline"].to_numpy()
            - joint["knn"].to_numpy()
        )
    else:
        delta = (
            joint["knn"].to_numpy()
            - joint["baseline"].to_numpy()
        )

    n = len(delta)

    if n < 20:
        return {
            "n": n,
            "mean_improvement": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    block = min(block, n)
    rng = np.random.default_rng(seed)
    simulations = []

    for _ in range(repeats):
        indices = []

        while len(indices) < n:
            start = int(
                rng.integers(0, n - block + 1)
            )
            indices.extend(
                range(start, start + block)
            )

        sampled = delta[
            np.asarray(indices[:n], dtype=int)
        ]

        simulations.append(sampled.mean())

    lo, hi = np.quantile(
        simulations,
        [0.025, 0.975],
    )

    return {
        "n": n,
        "mean_improvement": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def daily_errors(prediction_table, method):
    sub = prediction_table[
        prediction_table["method"] == method
    ].copy()

    sub["squared_error"] = (
        sub["actual"] - sub["prediction"]
    ) ** 2

    sub["abs_error"] = np.abs(
        sub["actual"] - sub["prediction"]
    )

    return (
        sub.groupby("date", as_index=False)[
            ["squared_error", "abs_error"]
        ].mean()
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v10_reflection_quality"
        ),
    )

    p.add_argument(
        "--documents",
        default="outputs/v31/evidence_alignment.parquet",
    )

    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument("--neighbors", type=int, default=12)
    p.add_argument("--random-repeats", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")

    # Compatibility with FinRL-X's original loader.
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    p.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--sec",
        default="rag/data/processed/sec/sec_chunks.jsonl"
    )

    p.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
        ),
    )

    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--embed", default="bge-m3:latest")

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )

    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")

    args = p.parse_args()

    if args.neighbors < 2:
        raise RuntimeError("neighbors must >= 2")

    if args.random_repeats < 2:
        raise RuntimeError("random-repeats must >= 2")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)

    log("===== V10 LOAD FINRL-X DATA =====")

    # Reuse existing V9 source loaders.
    # Validation prices are loaded by the project
    # loader but WILL NOT be used in any 2020
    # training target or evaluation label.
    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    dates = v9.normalized_dates(dates)

    all_prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    ).astype(np.float64)

    # Explicitly restrict audited data to 2018-2020.
    dates = dates[:n_train]
    contexts = contexts[:n_train]
    prices = all_prices[:n_train]

    if len(dates) != len(prices):
        raise RuntimeError("Invalid historical date alignment")

    if dates.max().year != 2020:
        raise RuntimeError(
            "Train should end in year 2020"
        )

    if dates.min().year != 2018:
        raise RuntimeError(
            "Train should start in year 2018"
        )

    log("[TRAIN DATE RANGE]", dates.min(), dates.max())

    log("===== LOAD SEC POINT-IN-TIME RETRIEVAL =====")

    docs = v8.load_documents(
        args.documents
    )

    vectorizer, embeddings = (
        v8.make_document_vectors(docs)
    )

    sec, sec_audit = (
        v8b.build_asof_sec_retrieval(
            dates,
            contexts,
            docs,
            embeddings,
            vectorizer,
            args.top_k,
        )
    )

    sec_audit.to_csv(
        args.out / "sec_retrieval_audit.csv",
        index=False,
    )

    log("[SEC FEATURES]", sec.shape)

    log("===== LOAD TEMPORAL KG RAG =====")

    kg, kg_info = v9.load_temporal_kg(
        args.temporal_kg,
        dates,
        args.out,
    )

    log("[KG FEATURES]", kg.shape)

    log("===== BUILD HISTORICAL MEMORY =====")

    memory, memory_x, memory_y = v9.build_memory(
        dates,
        prices,
        contexts,
        sec,
        kg,
        n_train,
    )

    memory.to_csv(
        args.out / "historical_cases.csv",
        index=False,
    )

    memory["decision_date"] = pd.to_datetime(
        memory["decision_date"]
    )

    memory["maturity_date"] = pd.to_datetime(
        memory["maturity_date"]
    )

    memory_company = memory["ticker"].to_numpy()
    maturity_idx = memory["maturity_index"].to_numpy(
        dtype=int
    )

    memory_outcomes = memory[
        "realized_excess_return"
    ].to_numpy(dtype=float)

    if len(memory_x) != len(memory):
        raise RuntimeError(
            "Historical feature length mismatch"
        )

    # Strict feature scaling:
    # use cases with matured labels by 2019-12-31.
    calibration = (
        memory["maturity_date"]
        <= pd.Timestamp("2019-12-31")
    ).to_numpy()

    if calibration.sum() < 100:
        raise RuntimeError(
            "Not enough mature 2018-2019 cases"
        )

    scaler = StandardScaler()
    scaler.fit(
        memory_x[calibration]
    )

    memory_z = scaler.transform(
        memory_x
    ).astype(np.float32)

    # Build sample-specific query vectors using the
    # exact same feature composition as V9.
    eye = np.eye(
        N_STOCKS,
        dtype=np.float32
    )

    prediction_rows = []
    retrieval_rows = []

    # Warm-up restriction: 2020 trading dates
    # only, with targets fully contained in 2020.
    eval_indices = [
        t for t in range(len(dates) - HORIZON)
        if dates[t].year == 2020
        and dates[t + HORIZON].year == 2020
    ]

    if not eval_indices:
        raise RuntimeError("No 2020 evaluation dates")

    log(
        "[EVALUATION 2020]",
        len(eval_indices),
        "decision dates"
    )

    # Seed is fixed so method differences are
    # reproducible. Random baseline has multiple
    # draws to reduce Monte Carlo noise.
    rng = np.random.default_rng(args.seed)

    for day_count, t in enumerate(eval_indices):
        decision = pd.Timestamp(dates[t])
        target_date = pd.Timestamp(
            dates[t + HORIZON]
        )

        forward = (
            prices[t + HORIZON] / prices[t] - 1.0
        )

        benchmark = float(forward.mean())
        excess = forward - benchmark

        for j, ticker in enumerate(TICKERS):
            # Most important PIT filter:
            # historical outcome MUST be mature
            # before the current decision date.
            eligible = np.flatnonzero(
                (memory_company == ticker)
                & (maturity_idx < t)
            )

            if len(eligible) < args.neighbors:
                raise RuntimeError(
                    f"Insufficient mature cases: "
                    f"{decision.date()} {ticker}"
                )

            if not (
                memory.iloc[eligible]["maturity_date"]
                < decision
            ).all():
                raise RuntimeError(
                    "Historical memory lookahead detected"
                )

            query = np.concatenate(
                [
                    contexts[t, j],
                    sec[t, j],
                    kg[t, j],
                    eye[j],
                ]
            ).reshape(1, -1)

            query_z = scaler.transform(
                query
            )[0]

            # Same distance definition as V9:
            # mean squared standardized feature distance.
            distances = np.mean(
                (
                    memory_z[eligible]
                    - query_z
                ) ** 2,
                axis=1,
            )

            knn_order = np.argsort(
                distances,
                kind="stable",
            )[:args.neighbors]

            knn_ids = eligible[knn_order]
            knn_dist = distances[knn_order]

            knn_weights = normalized_weights(
                knn_dist
            )

            knn_values = memory_outcomes[
                knn_ids
            ]

            quality = case_confidence(
                knn_values,
                knn_weights,
                knn_dist,
            )

            knn_pred = quality["mean"]

            # Recent baseline:
            # most recently matured same-company cases.
            recent_order = np.lexsort(
                (
                    eligible,
                    -maturity_idx[eligible],
                )
            )[:args.neighbors]

            recent_ids = eligible[
                recent_order
            ]

            recent_pred = float(
                memory_outcomes[recent_ids].mean()
            )

            # Company historical mean:
            # all mature cases available by t.
            mean_pred = float(
                memory_outcomes[eligible].mean()
            )

            # Random baseline:
            # average the predictions across 20
            # independently sampled candidate sets.
            random_predictions = []

            for _ in range(args.random_repeats):
                sample = rng.choice(
                    eligible,
                    size=args.neighbors,
                    replace=False,
                )

                random_predictions.append(
                    float(
                        memory_outcomes[sample].mean()
                    )
                )

            random_pred = float(
                np.mean(random_predictions)
            )

            predictions = {
                "KNN_REFLECTION": knn_pred,
                "RECENT": recent_pred,
                "RANDOM": random_pred,
                "HIST_MEAN": mean_pred,
                "ZERO": 0.0,
            }

            for method, estimate in predictions.items():
                prediction_rows.append({
                    "date": str(decision.date()),
                    "target_date": str(target_date.date()),
                    "ticker": ticker,
                    "method": method,
                    "actual": float(excess[j]),
                    "prediction": float(estimate),
                })

            retrieval_rows.append({
                "date": str(decision.date()),
                "target_date": str(target_date.date()),
                "ticker": ticker,
                "actual_excess_return": float(excess[j]),
                "knn_prediction": knn_pred,
                "recent_prediction": recent_pred,
                "random_prediction": random_pred,
                "hist_mean_prediction": mean_pred,
                "knn_case_ids": json.dumps(
                    knn_ids.tolist()
                ),
                "recent_case_ids": json.dumps(
                    recent_ids.tolist()
                ),
                "max_knn_maturity": str(
                    memory.iloc[knn_ids][
                        "maturity_date"
                    ].max().date()
                ),
                **{
                    k: v
                    for k, v in quality.items()
                    if k != "mean"
                },
            })

        if (
            day_count == 0
            or (day_count + 1) % 50 == 0
            or day_count + 1 == len(eval_indices)
        ):
            log(
                f"[WALK FORWARD] "
                f"{day_count+1}/{len(eval_indices)}"
            )

    pred = pd.DataFrame(prediction_rows)
    retrieval = pd.DataFrame(retrieval_rows)

    pred.to_csv(
        args.out / "predictions_2020.csv",
        index=False,
    )

    retrieval.to_csv(
        args.out / "reflection_retrieval_audit.csv",
        index=False,
    )

    log("===== CALCULATE METHOD PERFORMANCE =====")

    metric_rows = []
    daily_by_method = {}

    for method in METHODS:
        result, daily = calculate_metrics(
            pred,
            method,
        )

        metric_rows.append(result)

        daily_by_method[method] = daily

        daily.to_csv(
            args.out / f"{method}_daily_metrics.csv",
            index=False,
        )

    metrics = pd.DataFrame(metric_rows)

    metrics.to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    comparisons = []

    for idx, baseline in enumerate(
        ["RECENT", "RANDOM", "HIST_MEAN", "ZERO"]
    ):
        knn_errors = daily_errors(
            pred,
            "KNN_REFLECTION",
        )

        base_errors = daily_errors(
            pred,
            baseline,
        )

        for metric in [
            "squared_error",
            "abs_error",
        ]:
            result = paired_block_bootstrap(
                knn_errors,
                base_errors,
                metric,
                seed=args.seed + idx,
            )

            comparisons.append({
                "comparison": (
                    f"KNN_REFLECTION_vs_{baseline}"
                ),
                "metric": metric,
                **result,
            })

        result = paired_block_bootstrap(
            daily_by_method["KNN_REFLECTION"],
            daily_by_method[baseline],
            "rank_ic",
            seed=args.seed + 100 + idx,
        )

        comparisons.append({
            "comparison": (
                f"KNN_REFLECTION_vs_{baseline}"
            ),
            "metric": "rank_ic",
            **result,
        })

    comparison = pd.DataFrame(
        comparisons
    )

    comparison.to_csv(
        args.out / "paired_bootstrap.csv",
        index=False,
    )

    # Descriptive quality audit:
    # association between confidence proxy
    # and actual absolute forecast error.
    retrieval["knn_abs_error"] = np.abs(
        retrieval["actual_excess_return"]
        - retrieval["knn_prediction"]
    )

    retrieval["knn_sq_error"] = (
        retrieval["actual_excess_return"]
        - retrieval["knn_prediction"]
    ) ** 2

    correlation = retrieval[
        [
            "confidence_proxy",
            "knn_abs_error",
        ]
    ].corr(method="spearman").iloc[0, 1]

    quantiles = pd.qcut(
        retrieval["confidence_proxy"],
        q=5,
        duplicates="drop",
    )

    quality_bins = (
        retrieval.assign(
            confidence_bin=quantiles.astype(str)
        )
        .groupby("confidence_bin", as_index=False)
        .agg(
            samples=("knn_abs_error", "size"),
            mean_confidence=("confidence_proxy", "mean"),
            mae=("knn_abs_error", "mean"),
            mse=("knn_sq_error", "mean"),
        )
    )

    quality_bins.to_csv(
        args.out / "confidence_diagnostics.csv",
        index=False,
    )

    manifest = {
        "evaluation_period": "2020",
        "historical_calibration_end": "2019-12-31",
        "train_case_period": "2018-2020",
        "validation_2021_used": False,
        "test_2022_2023_used": False,
        "horizon_trading_days": HORIZON,
        "neighbors": args.neighbors,
        "random_repeats": args.random_repeats,
        "stock_universe": TICKERS,
        "historical_case_count": len(memory),
        "evaluation_days": len(eval_indices),
        "temporal_kg": kg_info,
        "sec_document_count": len(docs),
        "confidence_error_spearman": (
            float(correlation)
            if np.isfinite(correlation) else None
        ),
        "notes": [
            "2020 walk-forward evaluation",
            "Historical outcomes mature strictly before decision",
            "One fixed scaler calibrated with cases matured by 2019",
            "No A2C or Portfolio Fusion training",
            "Five-day equal-weight excess return target",
            "2020 forward labels remain inside 2020",
            "Block bootstrap is diagnostic, not proof of significance",
            "SEC/KG original provenance still requires independent PIT audit",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            zsafe(manifest),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 95)
    log("V10 REFLECTION QUALITY RESULTS")
    log("=" * 95)
    log(metrics.to_string(index=False))

    log()
    log("=" * 95)
    log("PAIRED BOOTSTRAP")
    log("=" * 95)
    log(comparison.to_string(index=False))

    log()
    log("=" * 95)
    log("CONFIDENCE DIAGNOSTICS")
    log("=" * 95)
    log(quality_bins.to_string(index=False))

    log()
    log("V10 REFLECTION QUALITY: PASS")
    log("2021 VALIDATION NOT EVALUATED")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
