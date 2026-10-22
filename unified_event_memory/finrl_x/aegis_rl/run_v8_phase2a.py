#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V8 Phase 2A
Semantic RAG Evidence Validation

Uses existing:
  run_v8_replug_finagent.py
  run_v7b_structured_evidence.py

Training: 2018-2020
Validation: 2021
Test: NEVER

Methods:
  MARKET_ONLY
  SEMANTIC_RAG
  TRAIN_SHUFFLED_RAG

For frozen SEMANTIC_RAG:
  REAL
  ZERO
  SHUFFLED
  LAGGED

All models have identical network architecture.
Seeds: 42, 123, 2026 by default.

Point-in-time SEC retrieval is inherited from
the original V8 implementation.

No new LLM requests.
No PPO/DDPG.
No RL retraining.

Important limitations:
  - This evaluates financial auxiliary predictions,
    not portfolio RL utility.
  - Document embeddings are 64D hashing features.
  - LAGGED is a historical evidence-vector intervention.
  - SHUFFLED rotates company evidence within each date.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from unified_event_memory.finrl_x.aegis_rl import (
    run_v7b_structured_evidence as v7b,
    run_v8_replug_finagent as v8,
)


MODES = (
    "REAL",
    "ZERO",
    "SHUFFLED",
    "LAGGED",
)


def log(*args):
    print(*args, flush=True)


def to_jsonable(x):
    if isinstance(x, dict):
        return {
            k: to_jsonable(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        return float(x)

    if isinstance(x, np.ndarray):
        return x.tolist()

    return x


def check_data(data, name):
    n = len(data["market"])

    assert len(data["docs"]) == n
    assert len(data["semantic"]) == n
    assert len(data["returns"]) == n
    assert len(data["meta"]) == n

    if data["market"].shape[1] != 10:
        raise RuntimeError(
            f"{name}: market must be 10D"
        )

    if data["docs"].shape[2] != v8.DOC_DIM:
        raise RuntimeError(
            f"{name}: document dimension mismatch"
        )

    if not np.isfinite(
        data["market"]
    ).all():
        raise RuntimeError(
            f"{name}: nonfinite market"
        )

    if not np.isfinite(
        data["docs"]
    ).all():
        raise RuntimeError(
            f"{name}: nonfinite documents"
        )

    if not np.isfinite(
        data["returns"]
    ).all():
        raise RuntimeError(
            f"{name}: nonfinite labels"
        )

    dates = pd.to_datetime(
        data["meta"]["date"]
    )

    targets = pd.to_datetime(
        data["meta"]["target_date"]
    )

    if name == "train":
        if (
            dates.max() > pd.Timestamp("2020-12-31")
            or targets.max() > pd.Timestamp("2020-12-31")
        ):
            raise RuntimeError(
                "Training target crosses 2020 boundary"
            )

    elif name == "val":
        if (
            dates.min() < pd.Timestamp("2021-01-01")
            or targets.max() > pd.Timestamp("2021-12-31")
        ):
            raise RuntimeError(
                "Validation dates outside 2021"
            )


def rotating_company_indices(meta):
    """
    Create an index permutation by date.

    For each decision date, a company's
    retrieved documents come from another
    company on that same date.

    Fixed deterministic cyclic rotation.

    Market state / ticker ID / labels
    remain unchanged.
    """

    meta = meta.reset_index(drop=True)

    result = np.arange(
        len(meta),
        dtype=np.int64,
    )

    for _, indices in meta.groupby(
        "date",
        sort=False,
    ).indices.items():

        ids = np.asarray(
            indices,
            dtype=np.int64,
        )

        if len(ids) < 2:
            continue

        ticker_names = (
            meta.iloc[ids]["ticker"]
            .astype(str)
            .to_numpy()
        )

        order = np.argsort(
            ticker_names,
            kind="stable"
        )

        sorted_ids = ids[order]

        shifted = np.roll(
            sorted_ids,
            1,
        )

        result[sorted_ids] = shifted

        src_tickers = (
            meta.iloc[
                result[sorted_ids]
            ]["ticker"].to_numpy()
        )

        dst_tickers = (
            meta.iloc[
                sorted_ids
            ]["ticker"].to_numpy()
        )

        if np.any(
            src_tickers == dst_tickers
        ):
            raise RuntimeError(
                "SHUFFLED kept original company"
            )

    return result


def make_perturbations(
    data,
    real,
    lag_trading_days,
):
    """
    Preserves Market / ticker identity.

    REAL:
      correct pooled SEC retrieval

    ZERO:
      no evidence vector

    SHUFFLED:
      retrieved evidence from another ticker,
      same decision date

    LAGGED:
      evidence that belonged to the same ticker
      lag_trading_days earlier.

    For early dates with insufficient history,
    LAGGED is zero.
    """

    real = np.asarray(
        real,
        dtype=np.float32,
    )

    meta = data["meta"].reset_index(
        drop=True
    )

    if len(meta) != len(real):
        raise RuntimeError(
            "Evidence/metadata length mismatch"
        )

    permutation = rotating_company_indices(
        meta
    )

    shuffled = real[
        permutation
    ].copy()

    zero = np.zeros_like(
        real
    )

    lagged = np.zeros_like(
        real
    )

    n_lag_valid = 0

    if lag_trading_days < 1:
        raise RuntimeError(
            "lag_trading_days must >= 1"
        )

    for ticker, ids in meta.groupby(
        "ticker",
        sort=False
    ).indices.items():

        ids = np.asarray(
            ids,
            dtype=np.int64,
        )

        order = np.argsort(
            pd.to_datetime(
                meta.iloc[ids]["date"]
            ).to_numpy(),
            kind="stable",
        )

        chronological = ids[order]

        if len(chronological) > lag_trading_days:
            destination = chronological[
                lag_trading_days:
            ]

            source = chronological[
                :-lag_trading_days
            ]

            lagged[destination] = (
                real[source]
            )

            n_lag_valid += len(destination)

    outputs = {
        "REAL": real.copy(),
        "ZERO": zero,
        "SHUFFLED": shuffled,
        "LAGGED": lagged,
    }

    for key, value in outputs.items():
        if (
            value.shape != real.shape
            or not np.isfinite(value).all()
        ):
            raise RuntimeError(
                f"Invalid mode {key}"
            )

    info = {
        "n_samples": len(real),
        "lag_valid_samples": n_lag_valid,
        "lag_coverage": (
            n_lag_valid / len(real)
        ),
        "shuffled_mean_l1": float(
            np.abs(
                shuffled - real
            ).sum(axis=1).mean()
        ),
        "lagged_mean_l1": float(
            np.abs(
                lagged - real
            ).sum(axis=1).mean()
        ),
        "zero_mean_l1": float(
            np.abs(real).sum(axis=1).mean()
        ),
    }

    return outputs, info


def predict_and_measure(
    model,
    scale,
    data,
    evidence,
):
    pred, pred_risk = v8.evaluate_aux(
        model,
        scale,
        data,
        evidence,
    )

    m = v8.metrics(
        data,
        pred,
        pred_risk,
    )

    out = data["meta"].copy()

    out["actual_return"] = (
        data["returns"]
    )

    out["predicted_return"] = pred

    out["actual_risk"] = (
        data["risks"]
    )

    out["predicted_risk"] = pred_risk

    return out, m


def paired_predictions(
    base,
    other,
    name,
):
    """
    Mean absolute prediction change,
    keeping date/ticker/market state fixed.
    """

    keys = [
        "date",
        "target_date",
        "ticker",
    ]

    merged = base[
        keys + ["predicted_return"]
    ].merge(
        other[
            keys + ["predicted_return"]
        ],
        on=keys,
        suffixes=("_real", "_other"),
        validate="one_to_one",
    )

    diff = np.abs(
        merged[
            "predicted_return_real"
        ].to_numpy()
        -
        merged[
            "predicted_return_other"
        ].to_numpy()
    )

    return {
        "mode": name,
        "mean_abs_prediction_change":
            float(diff.mean()),
        "median_abs_prediction_change":
            float(np.median(diff)),
        "p95_prediction_change":
            float(np.quantile(diff, 0.95)),
        "fraction_change_gt_0p001":
            float(np.mean(diff > 0.001)),
    }


def daily_rank_ic(predictions):
    """
    Five-stock Spearman rank IC per date.

    Dates missing >=3 stocks or with constant
    values are excluded from IC aggregation.
    """

    rows = []

    for date, group in predictions.groupby(
        "date",
        sort=True,
    ):
        valid = group[
            ["actual_return", "predicted_return"]
        ].dropna()

        if (
            len(valid) < 3
            or valid["actual_return"].nunique() < 2
            or valid["predicted_return"].nunique() < 2
        ):
            ic = np.nan

        else:
            ic = valid.corr(
                method="spearman"
            ).loc[
                "actual_return",
                "predicted_return"
            ]

        rows.append({
            "date": date,
            "rank_ic": ic,
        })

    return pd.DataFrame(rows)


def block_bootstrap_ic(
    real_daily,
    baseline_daily,
    block=10,
    repetitions=1000,
    seed=2026,
):
    """
    Resample paired daily IC differences
    with moving blocks.

    Diagnostic CI only; 5-day forward-return
    labels are overlapping and days are not
    independent.
    """

    joined = real_daily.merge(
        baseline_daily,
        on="date",
        suffixes=("_real", "_baseline"),
        validate="one_to_one",
    ).dropna()

    values = (
        joined["rank_ic_real"]
        - joined["rank_ic_baseline"]
    ).to_numpy(dtype=np.float64)

    n = len(values)

    if n < 2:
        return {
            "delta_mean_ic": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "days": n,
        }

    block = min(block, n)

    rng = np.random.default_rng(
        seed
    )

    results = []

    for _ in range(repetitions):
        sampled = []

        while len(sampled) < n:
            start = int(
                rng.integers(
                    0,
                    n - block + 1
                )
            )

            sampled.extend(
                range(
                    start,
                    start + block
                )
            )

        sampled = np.asarray(
            sampled[:n],
            dtype=np.int64,
        )

        results.append(
            values[sampled].mean()
        )

    low, high = np.quantile(
        results,
        [0.025, 0.975],
    )

    return {
        "delta_mean_ic": float(
            values.mean()
        ),
        "ci_low": float(low),
        "ci_high": float(high),
        "days": n,
    }


def train_arm(
    train,
    val,
    train_evidence,
    val_variants,
    seed,
    epochs,
    arm,
    out,
):
    """
    Train one predictor with the same seed
    and the same architecture.

    Evaluate all specified validation modes.
    """

    log()
    log("=" * 75)
    log(
        "TRAIN",
        arm,
        "SEED",
        seed
    )
    log("=" * 75)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model, scale = v8.fit_auxiliary(
        train,
        train_evidence,
        epochs=epochs,
        seed=seed,
    )

    path = (
        out
        / "models"
        / f"{arm}_seed{seed}.pt"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save({
        "state_dict": model.state_dict(),
        "scale": float(scale),
        "seed": seed,
        "arm": arm,
    }, path)

    rows = []
    predictions = {}

    for mode, evidence in val_variants.items():
        prediction, metrics = (
            predict_and_measure(
                model,
                scale,
                val,
                evidence,
            )
        )

        predictions[mode] = prediction

        rows.append({
            "seed": seed,
            "arm": arm,
            "mode": mode,
            **metrics,
        })

        log(
            f"[{mode:9s}] "
            f"MSE={metrics['mse']:.8f} "
            f"IC={metrics['rank_ic']:+.6f} "
            f"RiskMAE={metrics['risk_mae']:.6f}"
        )

    return pd.DataFrame(rows), predictions


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--documents",
        default=(
            "outputs/v31/"
            "evidence_alignment.parquet"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v8_phase2a"
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026],
    )

    parser.add_argument(
        "--aux-epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--lag-days",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    parser.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
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
        default=0.0003,
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--split",
        default="train",
    )

    parser.add_argument(
        "--algo",
        default="A2C",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    if args.max_days != 0:
        raise RuntimeError(
            "Full 2018-2021 data required"
        )

    if args.top_k < 2:
        raise RuntimeError(
            "top-k must be >= 2"
        )

    if args.lag_days < 1:
        raise RuntimeError(
            "lag-days must be >= 1"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.set_num_threads(2)

    log("=" * 80)
    log("V8 PHASE 2A")
    log("=" * 80)

    log("[LOAD] Existing FinRL-X data")

    train_data = v7b.load_split(
        args,
        "train"
    )

    val_data = v7b.load_split(
        args,
        "val"
    )

    log("[LOAD] SEC documents")

    documents = v8.load_documents(
        args.documents
    )

    vectorizer, embeddings = (
        v8.make_document_vectors(
            documents
        )
    )

    log("[DOCUMENTS]", len(documents))

    log("[RETRIEVE] TRAIN")

    train = v8.build_retrieval_dataset(
        train_data,
        documents,
        embeddings,
        vectorizer,
        args.top_k,
        "train",
    )

    log("[RETRIEVE] VALIDATION")

    val = v8.build_retrieval_dataset(
        val_data,
        documents,
        embeddings,
        vectorizer,
        args.top_k,
        "val",
    )

    check_data(train, "train")
    check_data(val, "val")

    log("[TRAIN]", len(train["returns"]))
    log("[VALIDATION]", len(val["returns"]))

    train_sem = v8.semantic_weights(
        train
    )

    val_sem = v8.semantic_weights(
        val
    )

    train_real = v8.pooled_evidence(
        train,
        train_sem,
    )

    val_real = v8.pooled_evidence(
        val,
        val_sem,
    )

    train_variants, train_info = (
        make_perturbations(
            train,
            train_real,
            args.lag_days,
        )
    )

    val_variants, val_info = (
        make_perturbations(
            val,
            val_real,
            args.lag_days,
        )
    )

    log("[TRAIN PERTURBATIONS]", train_info)
    log("[VAL PERTURBATIONS]", val_info)

    train_zero = np.zeros_like(
        train_real
    )

    val_zero = np.zeros_like(
        val_real
    )

    # The trained Semantic RAG model
    # is tested against REAL / ZERO /
    # SHUFFLED / LAGGED evidence.
    #
    # Other arms provide matched-capacity
    # retraining controls.
    experiments = [
        (
            "MARKET_ONLY",
            train_zero,
            {"REAL": val_zero},
        ),
        (
            "SEMANTIC_RAG",
            train_real,
            val_variants,
        ),
        (
            "TRAIN_SHUFFLED_RAG",
            train_variants["SHUFFLED"],
            {"SHUFFLED": val_variants["SHUFFLED"]},
        ),
    ]

    all_metrics = []
    sensitivity_rows = []
    predictions_by_seed = {}

    for seed in args.seeds:
        predictions_by_seed[seed] = {}

        for arm, tr_evidence, eval_modes in experiments:
            metrics, predictions = train_arm(
                train,
                val,
                tr_evidence,
                eval_modes,
                seed,
                args.aux_epochs,
                arm,
                args.out,
            )

            all_metrics.append(metrics)

            predictions_by_seed[
                seed
            ][arm] = predictions

            seed_dir = (
                args.out
                / f"seed{seed}"
            )

            seed_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for mode, prediction in predictions.items():
                prediction.to_csv(
                    seed_dir
                    / f"{arm}_{mode}_predictions.csv",
                    index=False,
                )

                daily_rank_ic(
                    prediction
                ).to_csv(
                    seed_dir
                    / f"{arm}_{mode}_daily_ic.csv",
                    index=False,
                )

        reference = predictions_by_seed[
            seed
        ]["SEMANTIC_RAG"]["REAL"]

        for mode in MODES:
            other = predictions_by_seed[
                seed
            ]["SEMANTIC_RAG"][mode]

            x = paired_predictions(
                reference,
                other,
                mode,
            )

            x["seed"] = seed

            sensitivity_rows.append(x)

    metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    sensitivity = pd.DataFrame(
        sensitivity_rows
    )

    metrics.to_csv(
        args.out / "validation_metrics.csv",
        index=False,
    )

    sensitivity.to_csv(
        args.out / "evidence_sensitivity.csv",
        index=False,
    )

    summary = (
        metrics.groupby(
            ["arm", "mode"]
        )
        .agg(
            mean_mse=("mse", "mean"),
            mean_mae=("mae", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
            mean_risk_mae=("risk_mae", "mean"),
            mean_directional_accuracy=(
                "directional_accuracy",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        args.out / "multi_seed_summary.csv",
        index=False,
    )

    # Same-seed improvement vs Market-only
    comparison_rows = []

    for seed in args.seeds:
        seed_metrics = metrics[
            metrics["seed"] == seed
        ]

        market = seed_metrics[
            (
                seed_metrics["arm"]
                == "MARKET_ONLY"
            )
        ].iloc[0]

        for arm, mode in [
            ("SEMANTIC_RAG", "REAL"),
            ("SEMANTIC_RAG", "ZERO"),
            ("SEMANTIC_RAG", "SHUFFLED"),
            ("SEMANTIC_RAG", "LAGGED"),
            ("TRAIN_SHUFFLED_RAG", "SHUFFLED"),
        ]:
            sub = seed_metrics[
                (seed_metrics["arm"] == arm)
                & (seed_metrics["mode"] == mode)
            ].iloc[0]

            baseline_ic = daily_rank_ic(
                predictions_by_seed[
                    seed
                ]["MARKET_ONLY"]["REAL"]
            )

            candidate_ic = daily_rank_ic(
                predictions_by_seed[
                    seed
                ][arm][mode]
            )

            ci = block_bootstrap_ic(
                candidate_ic,
                baseline_ic,
                block=10,
                repetitions=1000,
                seed=seed + 2026,
            )

            comparison_rows.append({
                "seed": seed,
                "arm": arm,
                "mode": mode,
                "delta_mse": (
                    sub["mse"] - market["mse"]
                ),
                "delta_rank_ic": (
                    sub["rank_ic"]
                    - market["rank_ic"]
                ),
                "delta_directional_accuracy": (
                    sub["directional_accuracy"]
                    - market["directional_accuracy"]
                ),
                "delta_risk_mae": (
                    sub["risk_mae"]
                    - market["risk_mae"]
                ),
                "paired_delta_ic": ci["delta_mean_ic"],
                "ic_ci_low": ci["ci_low"],
                "ic_ci_high": ci["ci_high"],
            })

    comparisons = pd.DataFrame(
        comparison_rows
    )

    comparisons.to_csv(
        args.out / "paired_comparison.csv",
        index=False,
    )

    manifest = {
        "train_samples": len(train["returns"]),
        "validation_samples": len(val["returns"]),
        "documents": len(documents),
        "top_k": args.top_k,
        "seeds": args.seeds,
        "aux_epochs": args.aux_epochs,
        "lag_days": args.lag_days,
        "train_perturbation": train_info,
        "validation_perturbation": val_info,
        "test_used": False,
        "limitations": [
            "No portfolio RL evaluation",
            "64D hashing documents",
            "Fixed semantic retrieval candidates",
            "Daily labels overlap for 5-day horizon",
            "Same-date document shuffle is a negative control",
            "Lagged evidence has zero warm-up period",
            "Training period shares chronological filing corpus",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            to_jsonable(manifest),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 95)
    log("MULTI-SEED SUMMARY")
    log("=" * 95)
    log(summary.to_string(index=False))

    log()
    log("=" * 95)
    log("PAIRED COMPARISON")
    log("=" * 95)
    log(comparisons.to_string(index=False))

    log()
    log("=" * 95)
    log("EVIDENCE SENSITIVITY")
    log("=" * 95)
    log(sensitivity.to_string(index=False))

    log()
    log("V8 PHASE 2A: PASS")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
