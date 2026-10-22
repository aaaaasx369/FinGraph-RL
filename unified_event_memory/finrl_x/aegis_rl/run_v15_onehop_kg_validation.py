#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V15: One-hop KG RAG Reliability Tests

M0: Market-only
M1: Market + correct company 1-hop KG
M2: Market + shuffled-company 1-hop KG (refit)
M3: Market + 20-trading-day-lagged KG (refit)

Additional fixed-M1 interventions on 2021:
  - correct KG
  - shuffled KG
  - zero KG

All predictions use:
  - historical 2018-2020 market/KG information
  - 2020 expanding monthly walk-forward
  - 2021 fixed validation
  - five-day forward excess return

2022-2023 never evaluated.

KG source first-observed date is not independently
certified as public information availability.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v13_unified_sec_kg_rag as v13,
    run_v14_0_ablation as base,
)


ARMS = ["M0", "M1", "M2", "M3"]

COMPARISONS = [
    ("M0", "M1"),
    ("M1", "M2"),
    ("M1", "M3"),
    ("M0", "M2"),
    ("M0", "M3"),
]


def log(*items):
    print(*items, flush=True)


def shuffle_company_kg(kg, seed):
    """
    Deterministically rotate ticker KG assignments
    separately for each date.

    Does not shuffle across dates, so it cannot
    bring a future KG snapshot into the past.

    Uses a nonzero circular shift: no company's
    original KG remains assigned to itself.
    """
    rng = np.random.default_rng(seed)

    output = np.empty_like(kg)
    rotations = []

    for i in range(len(kg)):
        shift = int(rng.integers(1, 5))
        rotations.append(shift)

        output[i] = np.roll(
            kg[i],
            shift=shift,
            axis=0,
        )

    if np.array_equal(output, kg):
        raise RuntimeError(
            "Shuffled KG unexpectedly identical"
        )

    return output, rotations


def lag_company_kg(kg, lag):
    """
    At decision t, retrieve the feature that
    was used at decision t-lag.

    The first lag dates receive zero vectors.

    This is a stale-evidence control, not
    an exact simulation of a delayed publication.
    """
    if lag <= 0:
        raise ValueError("lag must be positive")

    output = np.zeros_like(kg)

    if lag < len(kg):
        output[lag:] = kg[:-lag]

    return output


def fixed_model_interventions(
    datasets,
    market,
    real_kg,
    shuffled_kg,
    targets,
    maturity,
    dates,
    alpha,
):
    """
    Fit M1 once on historical matured labels.
    Evaluate interventions without refitting.

    This is a model reliance check, not a
    causal test of KG economic usefulness.
    """
    finite = np.isfinite(targets).all(axis=1)

    train_ids = np.flatnonzero(
        finite
        & (dates.year <= 2020)
        & (
            maturity
            < np.datetime64("2021-01-01")
        )
    )

    val_ids = np.flatnonzero(
        finite
        & (dates.year == 2021)
        & (
            maturity
            <= np.datetime64("2021-12-31")
        )
    )

    models = base.fit_company_models(
        datasets["M1"],
        targets,
        train_ids,
        alpha,
    )

    conditions = {
        "REAL": real_kg,
        "SHUFFLED": shuffled_kg,
        "ZERO": np.zeros_like(real_kg),
    }

    prediction_rows = []
    metric_frames = []
    stored_preds = {}

    for condition, kg in conditions.items():
        x = base.compose(market, kg)

        predictions = base.predict_company_models(
            models,
            x,
            val_ids,
        )

        stored_preds[condition] = predictions

        frame = base.evaluate_predictions(
            targets[val_ids],
            predictions,
            dates[val_ids],
            condition,
            "2021_FIXED_INTERVENTION",
        )

        metric_frames.append(frame)

        for k, day_index in enumerate(val_ids):
            for j, ticker in enumerate(base.TICKERS):
                prediction_rows.append({
                    "date": str(
                        dates[day_index].date()
                    ),
                    "ticker": ticker,
                    "condition": condition,
                    "actual": float(
                        targets[day_index, j]
                    ),
                    "prediction": float(
                        predictions[k, j]
                    ),
                })

    ref = stored_preds["REAL"]

    reliance_rows = []

    for name in ["SHUFFLED", "ZERO"]:
        changed = np.abs(
            ref - stored_preds[name]
        )

        reliance_rows.append({
            "intervention": name,
            "mean_abs_prediction_change": float(
                changed.mean()
            ),
            "max_abs_prediction_change": float(
                changed.max()
            ),
            "fraction_changed": float(
                np.mean(changed > 1e-10)
            ),
        })

    return (
        pd.concat(
            metric_frames,
            ignore_index=True,
        ),
        pd.DataFrame(prediction_rows),
        pd.DataFrame(reliance_rows),
    )


def paired_bootstraps(
    daily,
    comparisons,
    periods,
    seed,
):
    rows = []

    for p_idx, period in enumerate(periods):
        for c_idx, (a, b) in enumerate(comparisons):
            for m_idx, metric in enumerate(
                ["mse", "mae", "rank_ic"]
            ):
                rows.append(
                    base.moving_block_comparison(
                        daily,
                        period,
                        a,
                        b,
                        metric,
                        seed=(
                            seed
                            + 1000*p_idx
                            + 10*c_idx
                            + m_idx
                        ),
                    )
                )

    return pd.DataFrame(rows)


def make_parser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v15_onehop_validation"
        ),
    )

    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1000.0,
    )

    p.add_argument(
        "--kg-top-k",
        type=int,
        default=12,
    )

    p.add_argument(
        "--kg-lag",
        type=int,
        default=20,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Compatibility with the V8/V14-0 loader.
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
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
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
        default="http://localhost:11434",
    )

    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )

    p.add_argument(
        "--embed",
        default="bge-m3:latest",
    )

    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")

    return p


def main():
    args = make_parser().parse_args()

    if args.ridge_alpha <= 0:
        raise ValueError("Invalid Ridge alpha")

    if args.kg_lag < 1:
        raise ValueError("KG lag must be >= 1")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 72)
    log("V15: ONE-HOP KG RELIABILITY")
    log("=" * 72)

    log("[1] LOAD MARKET")

    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    dates = base.normalize_dates(dates)

    market_flat = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    )

    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    )

    market, kgt = base.validate_market_layout(
        market_flat
    )

    if not (
        len(dates)
        == len(market)
        == len(prices)
    ):
        raise RuntimeError(
            "Dataset date alignment mismatch"
        )

    if not (
        dates.min().year == 2018
        and dates.max().year == 2021
        and dates[n_train].year == 2021
    ):
        raise RuntimeError(
            "Unexpected training/validation split"
        )

    log("[MARKET]", market.shape)

    log("[2] BUILD REAL 1-HOP KG")

    company_events = (
        v13.kg_events_from_csv(
            args.temporal_kg,
            pd.Timestamp("2021-12-31"),
        )
    )

    real_kg, relations = (
        v13.build_kg_rag(
            dates,
            company_events,
            args.kg_top_k,
            args.out,
        )
    )

    if real_kg.shape[:2] != market.shape[:2]:
        raise RuntimeError(
            "KG/market alignment mismatch"
        )

    if not np.isfinite(real_kg).all():
        raise RuntimeError(
            "Nonfinite KG representations"
        )

    log("[KG]", real_kg.shape)

    log("[3] CONSTRUCT NEGATIVE CONTROLS")

    shuffled_kg, rotations = (
        shuffle_company_kg(
            real_kg,
            args.seed,
        )
    )

    lagged_kg = lag_company_kg(
        real_kg,
        args.kg_lag,
    )

    datasets = {
        "M0": base.compose(
            market,
        ),
        "M1": base.compose(
            market, real_kg,
        ),
        "M2": base.compose(
            market, shuffled_kg,
        ),
        "M3": base.compose(
            market, lagged_kg,
        ),
    }

    for arm, x in datasets.items():
        if not np.isfinite(x).all():
            raise RuntimeError(
                f"{arm} contains NaN/Inf"
            )

        log("[ARM]", arm, x.shape)

    log("[4] CONSTRUCT FIVE-DAY LABELS")

    targets, maturity = (
        base.build_targets(
            prices,
            dates,
        )
    )

    log("[5] 2020 MONTHLY WALK-FORWARD")

    old_arms = list(base.ARMS)

    try:
        base.ARMS = list(ARMS)

        wf_daily, wf_predictions, folds = (
            base.run_2020_expanding(
                datasets,
                targets,
                maturity,
                dates,
                args.ridge_alpha,
            )
        )

        log("[6] 2021 FIXED VALIDATION")

        val_daily, val_predictions, val_info = (
            base.run_2021_fixed(
                datasets,
                targets,
                maturity,
                dates,
                args.ridge_alpha,
            )
        )
    finally:
        base.ARMS = old_arms

    folds.to_csv(
        args.out / "walk_forward_folds.csv",
        index=False,
    )

    daily = pd.concat(
        [wf_daily, val_daily],
        ignore_index=True,
    )

    predictions = pd.concat(
        [wf_predictions, val_predictions],
        ignore_index=True,
    )

    metrics = base.summarize(daily)

    daily.to_csv(
        args.out / "daily_metrics.csv",
        index=False,
    )

    predictions.to_csv(
        args.out / "predictions.csv",
        index=False,
    )

    metrics.to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    log("[7] BOOTSTRAP")

    bootstrap = paired_bootstraps(
        daily,
        COMPARISONS,
        [
            "2020_WALK_FORWARD",
            "2021_VALIDATION",
        ],
        args.seed,
    )

    bootstrap.to_csv(
        args.out / "paired_bootstrap.csv",
        index=False,
    )

    log("[8] FIXED-MODEL INTERVENTIONS")

    intervention_daily, intervention_predictions, reliance = (
        fixed_model_interventions(
            datasets,
            market,
            real_kg,
            shuffled_kg,
            targets,
            maturity,
            dates,
            args.ridge_alpha,
        )
    )

    intervention_metrics = (
        base.summarize(intervention_daily)
    )

    intervention_metrics.to_csv(
        args.out / "fixed_intervention_metrics.csv",
        index=False,
    )

    intervention_predictions.to_csv(
        args.out / "fixed_intervention_predictions.csv",
        index=False,
    )

    reliance.to_csv(
        args.out / "model_reliance.csv",
        index=False,
    )

    intervention_bootstrap = paired_bootstraps(
        intervention_daily,
        [
            ("REAL", "SHUFFLED"),
            ("REAL", "ZERO"),
        ],
        ["2021_FIXED_INTERVENTION"],
        args.seed + 10000,
    )

    intervention_bootstrap.to_csv(
        args.out / "fixed_intervention_bootstrap.csv",
        index=False,
    )

    # Verify that the correct-company KG arm has
    # exactly the same feature definition as V14-A G1.
    previous_path = Path(
        "results/aegis_rl_v1/"
        "v14a_multihop/method_metrics.csv"
    )

    if previous_path.exists():
        prior = pd.read_csv(previous_path)

        cols = [
            "mse",
            "mae",
            "mean_rank_ic",
            "gross_rank_spread",
            "sign_accuracy",
        ]

        for period in [
            "2020_WALK_FORWARD",
            "2021_VALIDATION",
        ]:
            old = prior.loc[
                (prior["period"] == period)
                & (prior["arm"] == "G1"),
                cols,
            ].to_numpy(float)

            new = metrics.loc[
                (metrics["period"] == period)
                & (metrics["arm"] == "M1"),
                cols,
            ].to_numpy(float)

            if (
                old.shape != (1, len(cols))
                or new.shape != (1, len(cols))
                or not np.allclose(
                    old,
                    new,
                    atol=1e-9,
                    rtol=1e-7,
                )
            ):
                raise RuntimeError(
                    "M1 FAILED TO REPRODUCE V14-A G1 "
                    f"FOR {period}"
                )

        log("[BASELINE REPRODUCTION] PASS")
    else:
        log(
            "[BASELINE REPRODUCTION] "
            "Prior metrics missing; not verified"
        )

    manifest = {
        "experiment": "FinRL-X V15",
        "purpose": (
            "Validate company-specific information "
            "and recency of 1-hop KG RAG"
        ),
        "market_features": int(
            market.shape[2]
        ),
        "kg_features": int(
            real_kg.shape[2]
        ),
        "kg_relations": relations,
        "kg_unique_company_triples": int(
            len(company_events)
        ),
        "kg_lag_trading_days": args.kg_lag,
        "kg_shuffling": (
            "Same-date nonzero circular "
            "ticker rotation, seed-fixed"
        ),
        "ridge_alpha": args.ridge_alpha,
        "2020_folds": len(folds),
        "2021": val_info,
        "2022_2023_test_used": False,
        "arms": {
            "M0": "Market only",
            "M1": "Market + correct 1-hop KG",
            "M2": "Market + shuffled ticker KG, refit",
            "M3": "Market + lagged KG, refit",
        },
        "limitations": [
            "2021 previously inspected in earlier experiments.",
            "KG first-observed time not certified release time.",
            "Company permutation retains same-day market-wide KG signal.",
            "One permutation seed alone is not sufficient for a robust randomization test.",
            "Lagged KG zero-fills first 20 trading dates.",
            "Zero KG intervention is out-of-distribution for a fitted M1 model.",
            "Prediction reliance does not prove financial predictive value.",
            "No RL portfolio or transaction-cost backtest.",
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
    log("=" * 72)
    log("V15 MODEL METRICS")
    log("=" * 72)
    log(metrics.to_string(index=False))

    log()
    log("=" * 72)
    log("FIXED M1 INTERVENTIONS")
    log("=" * 72)
    log(intervention_metrics.to_string(index=False))

    log()
    log("=" * 72)
    log("MODEL RELIANCE")
    log("=" * 72)
    log(reliance.to_string(index=False))

    log()
    log("V15: PASS")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
