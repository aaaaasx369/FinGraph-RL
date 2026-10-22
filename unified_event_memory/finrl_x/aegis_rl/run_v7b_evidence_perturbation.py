#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V7-B Structured SEC Evidence Perturbation

Frozen A2C models:
  STRUCTURED_seed42
  STRUCTURED_seed123
  STRUCTURED_seed2026

Validation only (2021).

Modes:
  REAL      original SEC features
  ZERO      no SEC features
  SHUFFLED  wrong company assignment, fixed over time
  LAGGED    SEC features delayed by 30 trading days

Measurements:
  1. Closed-loop portfolio performance
  2. Same-state action sensitivity
  3. Portfolio weight L1 distance
  4. Real vs perturbed return differences

No training.
No test data evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from unified_event_memory.finrl_x.aegis_rl import (
    run_v7b_structured_evidence as v7b
)


MODES = [
    "REAL",
    "ZERO",
    "SHUFFLED",
    "LAGGED",
]


def log(*args):
    print(*args, flush=True)


def make_perturbations(original, lag_days):
    """
    Input: [n_days, 25]

    SEC features per ticker:
      impact, risk, confidence, recency, exists

    SHUFFLED reassigns complete ticker blocks.
    The same permutation is used for all dates.
    """

    x = np.asarray(
        original,
        dtype=np.float32
    )

    if x.ndim != 2 or x.shape[1] != 25:
        raise RuntimeError(
            f"Expected SEC features [N,25], got {x.shape}"
        )

    if lag_days < 1:
        raise ValueError(
            "lag_days must be >= 1"
        )

    n = len(x)
    blocks = x.reshape(n, 5, 5)

    # Fixed derangement: no ticker receives its
    # original ticker's signal.
    permutation = np.array(
        [1, 2, 3, 4, 0],
        dtype=int
    )

    if np.any(
        permutation == np.arange(5)
    ):
        raise RuntimeError(
            "SHUFFLED permutation is not a derangement"
        )

    shuffled = (
        blocks[:, permutation, :]
        .reshape(n, 25)
        .copy()
    )

    lagged = np.zeros_like(x)

    if lag_days < n:
        lagged[lag_days:] = x[:-lag_days]

    # RECENCY correction:
    # a signal observed 30 trading days ago should
    # not be treated as freshly observed today.
    #
    # Avoid relying on the original recency values
    # as if they applied to the current decision date.
    #
    # We use a conservative 30-calendar-day minimum
    # age adjustment. The true age is at least as old.
    lag_blocks = lagged.reshape(n, 5, 5)

    decay = np.exp(
        -float(lag_days) / 180.0
    )

    lag_blocks[:, :, 3] *= decay

    result = {
        "REAL": x.copy(),
        "ZERO": np.zeros_like(x),
        "SHUFFLED": shuffled,
        "LAGGED": lagged,
    }

    for name, arr in result.items():
        if (
            arr.shape != x.shape
            or not np.isfinite(arr).all()
        ):
            raise RuntimeError(
                f"Invalid perturbation: {name}"
            )

    return result, permutation


def make_env(data, sec_features, transaction_cost):
    modified = dict(data)

    modified["structured"] = np.asarray(
        sec_features,
        dtype=np.float32
    )

    return v7b.PortfolioEnv(
        modified,
        "STRUCTURED",
        transaction_cost=transaction_cost,
        reward_scale=100.0,
        max_days=0,
    )


def weights_from_action(action):
    return v7b.PortfolioEnv.softmax(
        action
    )


def rollout_with_reference_states(
    model,
    env,
):
    """
    Closed-loop REAL rollout.

    Save the real trajectory's state BEFORE
    each trading action for same-state probes.
    """

    observation, _ = env.reset()

    finished = False

    states = []
    actions = []

    while not finished:
        t = env.day

        states.append({
            "day": int(t),
            "date": str(
                env.dates[t].date()
            ),
            "pretrade_weights":
                env.weights.copy(),
        })

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        actions.append(
            weights_from_action(action)
        )

        observation, _, finished, _, _ = env.step(
            action
        )

    history = pd.DataFrame(
        env.history
    )

    return (
        history,
        states,
        np.asarray(actions),
    )


def summarize_history(history):
    returns = history[
        "net_return"
    ].to_numpy(dtype=np.float64)

    equity = history[
        "equity"
    ].to_numpy(dtype=np.float64)

    curve = np.r_[
        1.0,
        equity
    ]

    drawdown = (
        curve
        / np.maximum.accumulate(curve)
        - 1.0
    )

    if len(returns) > 1:
        volatility = float(
            returns.std(ddof=1)
        )
    else:
        volatility = 0.0

    sharpe = (
        float(
            np.sqrt(252)
            * returns.mean()
            / volatility
        )
        if volatility > 0
        else float("nan")
    )

    return {
        "days": int(len(history)),
        "total_return": float(
            equity[-1] - 1.0
        ),
        "sharpe": sharpe,
        "max_drawdown": float(
            drawdown.min()
        ),
        "mean_turnover": float(
            history["turnover"].mean()
        ),
    }


def same_state_probe(
    model,
    data,
    variants,
    reference_states,
):
    """
    Fix date and pre-trade holdings to the
    REAL rollout, change only SEC inputs.

    This measures direct evidence sensitivity,
    not total closed-loop causality.
    """

    rows = []

    for reference in reference_states:
        t = reference["day"]

        w_previous = reference[
            "pretrade_weights"
        ]

        predictions = {}

        for mode in MODES:
            observation = np.concatenate([
                data["market"][t],
                variants[mode][t],
                w_previous.astype(
                    np.float32
                ),
            ]).astype(np.float32)

            action, _ = model.predict(
                observation,
                deterministic=True,
            )

            predictions[mode] = (
                weights_from_action(action)
            )

        real = predictions["REAL"]

        for mode in MODES:
            other = predictions[mode]

            l1 = float(
                np.abs(
                    other - real
                ).sum()
            )

            linf = float(
                np.abs(
                    other - real
                ).max()
            )

            rows.append({
                "date": reference["date"],
                "day": t,
                "mode": mode,
                "weight_l1_vs_real": l1,
                "weight_linf_vs_real": linf,
                "changed_1pct": int(
                    l1 > 0.01
                ),
                **{
                    f"weight_{ticker}":
                        float(other[j])
                    for j, ticker
                    in enumerate(v7b.TICKERS)
                },
            })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v7b_validation"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v7b_evidence_perturbation"
        ),
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )

    p.add_argument(
        "--lag-days",
        type=int,
        default=30,
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    p.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
        ),
    )

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
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument(
        "--llm",
        default="qwen2.5:3b"
    )

    p.add_argument(
        "--embed",
        default="bge-m3:latest"
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=3
    )

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001
    )

    p.add_argument(
        "--cost-scale",
        type=float,
        default=0.0
    )

    p.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0
    )

    p.add_argument(
        "--tier",
        default="conservative"
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42
    )

    p.add_argument(
        "--max-days",
        type=int,
        default=0
    )

    args = p.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    # Existing project loader, but only
    # the 2021 validation split is requested.
    data = v7b.load_split(
        args,
        "val"
    )

    variants, permutation = make_perturbations(
        data["structured"],
        args.lag_days,
    )

    log("\n===== PERTURBATION SETTINGS =====")
    log("Validation: 2021 only")
    log("Lag trading days:", args.lag_days)
    log("Shuffled ticker permutation:", permutation)
    log("SEC feature dimensions:", variants["REAL"].shape)

    performance_rows = []
    reliance_rows = []

    for seed in args.seeds:
        checkpoint = (
            args.root
            / f"STRUCTURED_seed{seed}"
            / "model.zip"
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                checkpoint
            )

        log()
        log("=" * 75)
        log("SEED", seed)
        log("=" * 75)

        model = A2C.load(
            str(checkpoint),
            device=args.device,
        )

        real_env = make_env(
            data,
            variants["REAL"],
            args.transaction_cost,
        )

        if (
            tuple(model.observation_space.shape)
            != tuple(
                real_env.observation_space.shape
            )
        ):
            raise RuntimeError(
                "Model observation dimension mismatch"
            )

        if tuple(model.action_space.shape) != (5,):
            raise RuntimeError(
                "Model action shape mismatch"
            )

        real_history, reference_states, _ = (
            rollout_with_reference_states(
                model,
                real_env,
            )
        )

        real_metrics = summarize_history(
            real_history
        )

        # Confirm REAL reproduces the
        # original saved validation trajectory.
        previous = (
            args.root
            / f"STRUCTURED_seed{seed}"
            / "validation_history.csv"
        )

        if previous.is_file():
            original_history = pd.read_csv(
                previous
            )

            old_return = float(
                original_history[
                    "equity"
                ].iloc[-1] - 1.0
            )

            if not np.isclose(
                old_return,
                real_metrics["total_return"],
                rtol=0,
                atol=1e-7,
            ):
                raise RuntimeError(
                    "REAL parity failed for "
                    f"seed={seed}: "
                    f"old={old_return:.10f}, "
                    f"new={real_metrics['total_return']:.10f}"
                )

            log("[REAL PARITY] PASS")
        else:
            raise FileNotFoundError(
                "Original validation history not found: "
                + str(previous)
            )

        seed_dir = (
            args.out / f"seed{seed}"
        )

        seed_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        real_history.to_csv(
            seed_dir / "REAL_history.csv",
            index=False,
        )

        for mode in MODES:
            if mode == "REAL":
                history = real_history
                metrics = real_metrics
            else:
                env = make_env(
                    data,
                    variants[mode],
                    args.transaction_cost,
                )

                history, metrics = (
                    v7b.evaluate(
                        model,
                        env,
                    )
                )

            history.to_csv(
                seed_dir / f"{mode}_history.csv",
                index=False,
            )

            performance_rows.append({
                "seed": seed,
                "mode": mode,
                **metrics,
            })

            log(
                f"[{mode:8s}] "
                f"return={metrics['total_return']:+.6f} "
                f"sharpe={metrics['sharpe']:+.5f}"
            )

        sensitivity = same_state_probe(
            model,
            data,
            variants,
            reference_states,
        )

        sensitivity.to_csv(
            seed_dir / "same_state_actions.csv",
            index=False,
        )

        for mode in MODES:
            subset = sensitivity[
                sensitivity["mode"] == mode
            ]

            reliance_rows.append({
                "seed": seed,
                "mode": mode,
                "mean_weight_l1_vs_real": float(
                    subset[
                        "weight_l1_vs_real"
                    ].mean()
                ),
                "median_weight_l1_vs_real": float(
                    subset[
                        "weight_l1_vs_real"
                    ].median()
                ),
                "max_weight_l1_vs_real": float(
                    subset[
                        "weight_l1_vs_real"
                    ].max()
                ),
                "fraction_changed_l1_gt_1pct": float(
                    subset[
                        "changed_1pct"
                    ].mean()
                ),
            })

    performance = pd.DataFrame(
        performance_rows
    )

    reliance = pd.DataFrame(
        reliance_rows
    )

    performance.to_csv(
        args.out / "performance.csv",
        index=False,
    )

    reliance.to_csv(
        args.out / "reliance.csv",
        index=False,
    )

    real = (
        performance[
            performance["mode"] == "REAL"
        ][
            ["seed", "total_return", "sharpe"]
        ]
        .rename(columns={
            "total_return": "real_return",
            "sharpe": "real_sharpe",
        })
    )

    compare = performance.merge(
        real,
        on="seed",
        how="left",
        validate="many_to_one",
    )

    compare[
        "real_minus_mode_return_pp"
    ] = (
        compare["real_return"]
        - compare["total_return"]
    ) * 100.0

    compare[
        "real_minus_mode_sharpe"
    ] = (
        compare["real_sharpe"]
        - compare["sharpe"]
    )

    compare = compare.merge(
        reliance,
        on=["seed", "mode"],
        how="left",
        validate="one_to_one",
    )

    compare.to_csv(
        args.out / "comparison.csv",
        index=False,
    )

    log()
    log("=" * 100)
    log("V7-B EVIDENCE PERTURBATION RESULTS")
    log("=" * 100)

    log(
        compare[
            [
                "seed",
                "mode",
                "total_return",
                "sharpe",
                "real_minus_mode_return_pp",
                "mean_weight_l1_vs_real",
                "fraction_changed_l1_gt_1pct",
            ]
        ].to_string(index=False)
    )

    summary = (
        compare.groupby("mode")
        .agg(
            mean_return=(
                "total_return", "mean"
            ),
            mean_sharpe=(
                "sharpe", "mean"
            ),
            mean_real_advantage_pp=(
                "real_minus_mode_return_pp", "mean"
            ),
            mean_action_l1=(
                "mean_weight_l1_vs_real", "mean"
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        args.out / "summary.csv",
        index=False,
    )

    log()
    log("===== MULTI-SEED SUMMARY =====")
    log(summary.to_string(index=False))

    log()
    log("V7-B PERTURBATION: PASS")
    log("TEST SPLIT NOT EVALUATED")


if __name__ == "__main__":
    main()
