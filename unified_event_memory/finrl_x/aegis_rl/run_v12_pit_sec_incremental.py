#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V12: PIT SEC-RAG Incremental Predictive Value

Train: 2018-2020
Validation: 2021
Test 2022-2023: NOT USED

M0: Market + Dynamic KGT
M1: Market + Dynamic KGT + Original SEC RAG
M2: Market + Dynamic KGT + PIT SEC RAG
M3: Fixed M2 predictor + same-date ticker-shuffled PIT SEC
M4: Fixed M2 predictor + zero PIT SEC

Target: 5-trading-day excess return vs 5-stock equal weight.

Existing project loaders:
  run_v8_phase2b.load_data
  run_v8_phase2b.build_asof_sec_retrieval
  run_v8_replug_finagent.load_documents
  run_v8_replug_finagent.make_document_vectors

No A2C training.
No future return labels as features.
Original SEC and PIT SEC source content is unchanged;
only eligible availability dates differ.

The official filing metadata is matched, but excerpt content
and Dynamic KG temporal provenance remain separate audits.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v8_replug_finagent as v8,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
N = len(TICKERS)
HORIZON = 5
ARMS = ["M0", "M1", "M2", "M3", "M4"]


def log(*parts):
    print(*parts, flush=True)


def normalized_dates(values):
    return pd.DatetimeIndex(
        pd.to_datetime(values, utc=True)
    ).tz_convert(None).normalize()


def make_labels(prices, dates, n_train):
    """
    Strict split-local maturity:
      train t+5 must still be in 2018-2020
      validation t+5 must still be in 2021
    """
    n = len(prices)
    y = np.full((n, N), np.nan, dtype=np.float64)

    for t in range(n - HORIZON):
        end = t + HORIZON

        if t < n_train and end >= n_train:
            continue

        if t >= n_train and dates[end].year != 2021:
            continue

        r = prices[end] / prices[t] - 1.0
        y[t] = r - float(np.mean(r))

    train_ids = np.flatnonzero(
        np.isfinite(y[:, 0])
        & (np.arange(n) < n_train)
    )

    val_ids = np.flatnonzero(
        np.isfinite(y[:, 0])
        & (np.arange(n) >= n_train)
    )

    if len(train_ids) < 500 or len(val_ids) < 150:
        raise RuntimeError(
            "Unexpected train/val label coverage: "
            f"train={len(train_ids)}, val={len(val_ids)}"
        )

    return y, train_ids, val_ids


def market_to_rows(market):
    """
    Each ticker gets the same 1265D Market/KGT
    state plus a 5D company identity.

    This preserves the validated Dynamic KGT
    observation without inventing a feature mapping.
    """
    n, d = market.shape

    repeated = np.repeat(
        market[:, None, :],
        N,
        axis=1,
    )

    identities = np.broadcast_to(
        np.eye(N, dtype=np.float32),
        (n, N, N),
    )

    return np.concatenate(
        [repeated, identities],
        axis=2,
    ).astype(np.float32)


def flatten_days(x, day_ids):
    subset = x[day_ids]

    return subset.reshape(
        len(day_ids) * N,
        x.shape[2],
    ).astype(np.float32)


def target_rows(y, day_ids):
    return y[day_ids].reshape(-1)


def fit_model(x, y, ridge_alpha):
    """
    Each model uses train-only scaling.
    No validation model selection.
    """
    pipeline = make_pipeline(
        StandardScaler(),
        Ridge(alpha=ridge_alpha),
    )
    pipeline.fit(x, y)
    return pipeline


def predict_days(model, x, day_ids):
    rows = flatten_days(x, day_ids)

    pred = model.predict(rows)

    return pred.reshape(
        len(day_ids), N,
    )


def metrics_table(actual, predicted, dates, arm):
    true_flat = actual.reshape(-1)
    pred_flat = predicted.reshape(-1)

    mse = float(np.mean(
        (true_flat - pred_flat) ** 2
    ))

    mae = float(np.mean(
        np.abs(true_flat - pred_flat)
    ))

    accuracy = float(np.mean(
        (true_flat > 0) == (pred_flat > 0)
    ))

    daily_ics = []
    daily_spreads = []

    for i in range(len(actual)):
        truth = actual[i]
        output = predicted[i]

        if (
            np.ptp(truth) < 1e-12
            or np.ptp(output) < 1e-12
        ):
            daily_ics.append(np.nan)
            daily_spreads.append(0.0)
            continue

        corr = spearmanr(
            truth, output
        ).statistic

        daily_ics.append(
            float(corr)
            if np.isfinite(corr) else np.nan
        )

        high = int(np.argmax(output))
        low = int(np.argmin(output))

        daily_spreads.append(
            float(truth[high] - truth[low])
        )

    daily = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "arm": arm,
        "rank_ic": daily_ics,
        "gross_rank_spread": daily_spreads,
        "mse": np.mean(
            (actual - predicted) ** 2,
            axis=1,
        ),
        "mae": np.mean(
            np.abs(actual - predicted),
            axis=1,
        ),
    })

    result = {
        "arm": arm,
        "samples": len(true_flat),
        "days": len(dates),
        "mse": mse,
        "mae": mae,
        "sign_accuracy": accuracy,
        "mean_rank_ic": float(
            np.nanmean(daily_ics)
        ),
        "valid_ic_days": int(
            np.isfinite(daily_ics).sum()
        ),
        "mean_gross_rank_spread": float(
            np.mean(daily_spreads)
        ),
        "mean_prediction_abs": float(
            np.mean(np.abs(pred_flat))
        ),
    }

    return result, daily


def compare_block_bootstrap(
    results,
    baseline,
    challenger,
    metric,
    repeats=2000,
    block=10,
    seed=42,
):
    """
    Positive improvement:
      MSE/MAE -> baseline minus challenger
      IC -> challenger minus baseline

    Moving block bootstrap over valuation dates.
    """
    a = results[baseline][["date", metric]].rename(
        columns={metric: "baseline"}
    )

    b = results[challenger][["date", metric]].rename(
        columns={metric: "challenger"}
    )

    merged = a.merge(
        b,
        on="date",
        how="inner",
        validate="one_to_one",
    ).dropna()

    if metric == "rank_ic":
        delta = (
            merged["challenger"].to_numpy()
            - merged["baseline"].to_numpy()
        )
    else:
        delta = (
            merged["baseline"].to_numpy()
            - merged["challenger"].to_numpy()
        )

    n = len(delta)

    if n < 20:
        return {
            "baseline": baseline,
            "challenger": challenger,
            "metric": metric,
            "n_days": n,
            "mean_improvement": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    block = min(block, n)

    draws = np.empty(
        repeats,
        dtype=np.float64
    )

    for i in range(repeats):
        samples = []

        while len(samples) < n:
            start = int(
                rng.integers(
                    0, n - block + 1
                )
            )
            samples.extend(
                range(start, start + block)
            )

        samples = np.asarray(
            samples[:n],
            dtype=np.int64,
        )

        draws[i] = float(
            delta[samples].mean()
        )

    lo, hi = np.quantile(
        draws, [0.025, 0.975]
    )

    return {
        "baseline": baseline,
        "challenger": challenger,
        "metric": metric,
        "n_days": n,
        "mean_improvement": float(
            delta.mean()
        ),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def evidence_coverage(sec, train_ids, val_ids):
    """
    This detects nonzero pooled document vectors,
    NOT independently verified unique document IDs.
    """
    magnitudes = np.linalg.norm(
        sec,
        axis=2,
    )

    return {
        "train_nonzero_ticker_day_fraction": float(
            np.mean(magnitudes[train_ids] > 1e-10)
        ),
        "val_nonzero_ticker_day_fraction": float(
            np.mean(magnitudes[val_ids] > 1e-10)
        ),
        "train_mean_vector_norm": float(
            np.mean(magnitudes[train_ids])
        ),
        "val_mean_vector_norm": float(
            np.mean(magnitudes[val_ids])
        ),
    }


def load_sec_features(
    dates,
    contexts,
    document_path,
    top_k,
    audit_path,
):
    log("[RAG] Loading", document_path)

    docs = v8.load_documents(
        str(document_path)
    )

    vectorizer, doc_vectors = (
        v8.make_document_vectors(docs)
    )

    sec, audit = (
        v8b.build_asof_sec_retrieval(
            dates,
            contexts,
            docs,
            doc_vectors,
            vectorizer,
            top_k,
        )
    )

    sec = np.asarray(
        sec,
        dtype=np.float32,
    )

    audit.to_csv(
        audit_path,
        index=False,
    )

    if (
        sec.ndim != 3
        or sec.shape[:2] != (len(dates), N)
    ):
        raise RuntimeError(
            f"Invalid SEC feature shape: {sec.shape}"
        )

    return sec, len(docs)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--original-documents",
        default=(
            "outputs/v31/"
            "evidence_alignment.parquet"
        ),
    )

    p.add_argument(
        "--pit-documents",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/"
            "sec_pit_retrieval.parquet"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v12_pit_sec_incremental"
        ),
    )

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1000.0,
    )

    p.add_argument(
        "--max-val-days",
        type=int,
        default=0,
    )

    p.add_argument("--seed", type=int, default=42)

    # Existing FinRL-X loader arguments.
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
        default="rag/data/processed/sec/sec_chunks.jsonl",
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
    p.add_argument("--device", default="cpu")
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")

    args = p.parse_args()

    if args.ridge_alpha <= 0:
        raise ValueError(
            "--ridge-alpha must be positive"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("===== LOAD FINRL-X + DYNAMIC KGT =====")

    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    dates = normalized_dates(dates)

    market = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    ).astype(np.float32)

    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    ).astype(np.float64)

    if (
        len(dates) != len(market)
        or len(dates) != len(prices)
        or market.shape[1] != 1265
    ):
        raise RuntimeError(
            f"Invalid FinRL-X arrays: "
            f"dates={len(dates)}, "
            f"market={market.shape}, "
            f"prices={prices.shape}"
        )

    log(
        "[MARKET/KGT]",
        market.shape,
        "[PRICES]",
        prices.shape,
    )

    y, train_ids, val_ids = make_labels(
        prices,
        dates,
        n_train,
    )

    if args.max_val_days > 0:
        val_ids = val_ids[
            :args.max_val_days
        ]

    log(
        "[TRAIN DATES]",
        len(train_ids),
        "[VALIDATION DATES]",
        len(val_ids),
    )

    log("===== ORIGINAL SEC RETRIEVAL =====")

    sec_original, n_original = load_sec_features(
        dates,
        contexts,
        args.original_documents,
        args.top_k,
        args.out / "original_retrieval_audit.csv",
    )

    log("===== PIT SEC RETRIEVAL =====")

    sec_pit, n_pit = load_sec_features(
        dates,
        contexts,
        args.pit_documents,
        args.top_k,
        args.out / "pit_retrieval_audit.csv",
    )

    if sec_original.shape != sec_pit.shape:
        raise RuntimeError(
            "Original and PIT SEC dimensions differ: "
            f"{sec_original.shape} vs {sec_pit.shape}"
        )

    if not np.isfinite(sec_original).all():
        raise RuntimeError(
            "Original SEC contains NaN/Inf"
        )

    if not np.isfinite(sec_pit).all():
        raise RuntimeError(
            "PIT SEC contains NaN/Inf"
        )

    sec_dim = sec_pit.shape[2]

    log(
        "[SEC FEATURES]",
        sec_pit.shape,
    )

    base = market_to_rows(
        market
    )

    original = np.concatenate(
        [base, sec_original],
        axis=2,
    )

    pit = np.concatenate(
        [base, sec_pit],
        axis=2,
    )

    log("===== SEC COVERAGE AUDIT =====")

    coverage = {
        "original": evidence_coverage(
            sec_original, train_ids, val_ids
        ),
        "pit": evidence_coverage(
            sec_pit, train_ids, val_ids
        ),
    }

    original_val = sec_original[val_ids]
    pit_val = sec_pit[val_ids]

    coverage["same_state_original_vs_pit"] = {
        "mean_L2": float(
            np.linalg.norm(
                original_val - pit_val,
                axis=2,
            ).mean()
        ),
        "changed_ticker_day_fraction": float(
            np.mean(
                np.linalg.norm(
                    original_val - pit_val,
                    axis=2,
                ) > 1e-8
            )
        ),
    }

    log(
        json.dumps(
            coverage,
            indent=2,
        )
    )

    log("===== TRAIN M0 / M1 / M2 =====")

    target = target_rows(
        y, train_ids
    )

    datasets = {
        "M0": base,
        "M1": original,
        "M2": pit,
    }

    models = {}

    for arm in ["M0", "M1", "M2"]:
        log("[TRAIN]", arm)

        train_x = flatten_days(
            datasets[arm],
            train_ids,
        )

        if not np.isfinite(train_x).all():
            raise RuntimeError(
                f"{arm} train contains nonfinite features"
            )

        models[arm] = fit_model(
            train_x,
            target,
            args.ridge_alpha,
        )

    log("===== EVALUATE 2021 =====")

    predictions = {}

    for arm in ["M0", "M1", "M2"]:
        predictions[arm] = predict_days(
            models[arm],
            datasets[arm],
            val_ids,
        )

    # M3: permute ticker SEC channels within each
    # evaluation date. Never bring later-date
    # documents into an earlier state.
    #
    # Use a fixed nonidentity rotation to avoid
    # Monte Carlo-driven model selection.
    shuffled_sec = np.roll(
        sec_pit,
        shift=1,
        axis=1,
    )

    shuffled = np.concatenate(
        [base, shuffled_sec],
        axis=2,
    )

    # Both interventions keep the SAME fitted M2.
    predictions["M3"] = predict_days(
        models["M2"],
        shuffled,
        val_ids,
    )

    zero = np.concatenate(
        [
            base,
            np.zeros_like(sec_pit),
        ],
        axis=2,
    )

    predictions["M4"] = predict_days(
        models["M2"],
        zero,
        val_ids,
    )

    val_y = y[val_ids]
    val_dates = dates[val_ids]

    metrics_rows = []
    daily_results = {}

    for arm in ARMS:
        summary, daily = metrics_table(
            val_y,
            predictions[arm],
            val_dates,
            arm,
        )

        metrics_rows.append(summary)
        daily_results[arm] = daily

        daily.to_csv(
            args.out / f"{arm}_daily_metrics.csv",
            index=False,
        )

    metric_frame = pd.DataFrame(
        metrics_rows
    )

    metric_frame.to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    prediction_rows = []

    for i, date in enumerate(val_dates):
        for j, ticker in enumerate(TICKERS):
            row = {
                "date": str(date.date()),
                "ticker": ticker,
                "actual_excess_5d": float(
                    val_y[i, j]
                ),
                "original_sec_norm": float(
                    np.linalg.norm(
                        sec_original[val_ids[i], j]
                    )
                ),
                "pit_sec_norm": float(
                    np.linalg.norm(
                        sec_pit[val_ids[i], j]
                    )
                ),
            }

            for arm in ARMS:
                row[f"{arm}_prediction"] = float(
                    predictions[arm][i, j]
                )

            prediction_rows.append(row)

    pd.DataFrame(prediction_rows).to_csv(
        args.out / "predictions_2021.csv",
        index=False,
    )

    log("===== PAIRED BLOCK BOOTSTRAP =====")

    comparisons = [
        ("M0", "M1"),
        ("M0", "M2"),
        ("M1", "M2"),
        ("M3", "M2"),
        ("M4", "M2"),
    ]

    bootstrap_rows = []

    for idx, (baseline, challenger) in enumerate(
        comparisons
    ):
        for metric in ["mse", "mae", "rank_ic"]:
            row = compare_block_bootstrap(
                daily_results,
                baseline,
                challenger,
                metric,
                seed=args.seed + idx * 10,
            )

            bootstrap_rows.append(row)

    bootstrap = pd.DataFrame(
        bootstrap_rows
    )

    bootstrap.to_csv(
        args.out / "paired_bootstrap.csv",
        index=False,
    )

    sensitivity_rows = []

    for intervention in ["M3", "M4"]:
        difference = (
            predictions["M2"]
            - predictions[intervention]
        )

        sensitivity_rows.append({
            "intervention": intervention,
            "mean_abs_prediction_change": float(
                np.mean(np.abs(difference))
            ),
            "max_abs_prediction_change": float(
                np.max(np.abs(difference))
            ),
            "changed_prediction_fraction": float(
                np.mean(
                    np.abs(difference) > 1e-8
                )
            ),
        })

    pd.DataFrame(sensitivity_rows).to_csv(
        args.out / "evidence_sensitivity.csv",
        index=False,
    )

    manifest = {
        "experiment": "FinRL-X V12",
        "train": "2018-2020",
        "validation": "2021",
        "test_used": False,
        "target": "five_day_equal_weight_excess_return",
        "horizon": HORIZON,
        "train_days": len(train_ids),
        "validation_days": len(val_ids),
        "tickers": TICKERS,
        "market_dim": int(market.shape[1]),
        "sec_dim": int(sec_dim),
        "ridge_alpha": float(args.ridge_alpha),
        "original_document_rows_loaded": n_original,
        "pit_document_rows_loaded": n_pit,
        "source_paths": {
            "original": str(args.original_documents),
            "pit": str(args.pit_documents),
            "kgt": str(args.kgt),
        },
        "sec_coverage": coverage,
        "arms": {
            "M0": "fitted market + Dynamic KGT",
            "M1": "fitted market + KGT + original SEC",
            "M2": "fitted market + KGT + PIT SEC",
            "M3": "fixed M2 + same-day ticker-shuffled SEC",
            "M4": "fixed M2 + zero SEC",
        },
        "caveats": [
            "M3/M4 are fixed-model interventions.",
            "M0/M1/M2 are separately fitted.",
            "PIT SEC content matching not yet verified.",
            "Dynamic KGT historical provenance not fully verified.",
            "No A2C portfolio performance evaluated.",
            "No validation hyperparameter search performed.",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 80)
    log("V12 METHOD METRICS")
    log("=" * 80)
    log(metric_frame.to_string(index=False))

    log()
    log("=" * 80)
    log("PAIRED BOOTSTRAP")
    log("=" * 80)
    log(bootstrap.to_string(index=False))

    log()
    log("=" * 80)
    log("EVIDENCE SENSITIVITY")
    log("=" * 80)
    log(
        pd.DataFrame(
            sensitivity_rows
        ).to_string(index=False)
    )

    log()
    log("V12: PASS")
    log("2021 VALIDATION ONLY")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
