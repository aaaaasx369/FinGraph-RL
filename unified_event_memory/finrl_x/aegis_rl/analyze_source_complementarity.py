#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SOURCES = ("STOP", "KG", "SEC", "BOTH")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "results/aegis_rl_v1/"
            "stop_counterfactual_train/"
            "financial_utility_labels.csv"
        ),
    )

    parser.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "source_complementarity_v1"
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--block-size",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    required = {
        "date",
        "source",
        "gross_return",
        "transaction_cost",
        "retrieval_cost",
        "net_return",
        "utility_vs_stop",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    if df.duplicated(["date", "source"]).any():
        raise RuntimeError(
            "Duplicate date/source records"
        )

    if df[list(required - {"date", "source"})].isna().any().any():
        raise RuntimeError(
            "Missing numeric values"
        )

    # Reconstruct cost-free decision returns
    # from actual portfolio allocation outcomes.
    #
    # Trading costs remain in the calculation.
    # Only retrieval cost is removed.
    df["net_no_retrieval_cost"] = np.maximum(
        df["gross_return"]
        - df["transaction_cost"],
        -0.999999,
    )

    df["log_no_retrieval_cost"] = np.log1p(
        df["net_no_retrieval_cost"]
    )

    utility = df.pivot(
        index="date",
        columns="source",
        values="log_no_retrieval_cost",
    ).sort_index()

    if set(utility.columns) != set(SOURCES):
        raise RuntimeError(
            "STOP, KG, SEC, BOTH are required"
        )

    if utility.isna().any().any():
        raise RuntimeError(
            "Incomplete date/source pairs"
        )

    utility = utility.subtract(
        utility["STOP"],
        axis=0,
    )

    # STOP is zero by definition.
    assert np.allclose(
        utility["STOP"].to_numpy(),
        0.0,
        atol=1e-12,
    )

    u_kg = utility["KG"]
    u_sec = utility["SEC"]
    u_both = utility["BOTH"]

    # Incremental benefit of adding the second source.
    incremental_sec_given_kg = u_both - u_kg
    incremental_kg_given_sec = u_both - u_sec

    # Positive synergy indicates BOTH exceeds
    # the sum of individual one-step benefits.
    #
    # This is an outcome interaction proxy,
    # not direct evidence of document duplication.
    synergy = u_both - u_kg - u_sec

    best_single = pd.concat(
        [u_kg, u_sec],
        axis=1,
    ).max(axis=1)

    benefit_over_best_single = (
        u_both - best_single
    )

    # All actions share the same starting state.
    # This is a one-step, fixed-policy experiment.
    best_action = utility[
        list(SOURCES)
    ].idxmax(axis=1)

    daily = pd.DataFrame({
        "date": utility.index,
        "KG": u_kg.to_numpy(),
        "SEC": u_sec.to_numpy(),
        "BOTH": u_both.to_numpy(),
        "STOP": utility["STOP"].to_numpy(),
        "SEC_given_KG":
            incremental_sec_given_kg.to_numpy(),
        "KG_given_SEC":
            incremental_kg_given_sec.to_numpy(),
        "synergy":
            synergy.to_numpy(),
        "BOTH_minus_best_single":
            benefit_over_best_single.to_numpy(),
        "oracle_best_action":
            best_action.to_numpy(),
    })

    daily.to_csv(
        out / "daily_source_utility.csv",
        index=False,
    )

    n = len(daily)

    def describe(source, values):
        values = np.asarray(values, dtype=float)

        return {
            "source": source,
            "days": len(values),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "positive_rate": float(
                (values > 0).mean()
            ),
            "negative_rate": float(
                (values < 0).mean()
            ),
            "p05": float(
                np.quantile(values, 0.05)
            ),
            "p95": float(
                np.quantile(values, 0.95)
            ),
        }

    summary = pd.DataFrame([
        describe("KG", u_kg),
        describe("SEC", u_sec),
        describe("BOTH", u_both),
        describe(
            "SEC_given_KG",
            incremental_sec_given_kg,
        ),
        describe(
            "KG_given_SEC",
            incremental_kg_given_sec,
        ),
        describe(
            "BOTH_minus_best_single",
            benefit_over_best_single,
        ),
        describe("SYNERGY", synergy),
    ])

    summary.to_csv(
        out / "source_summary.csv",
        index=False,
    )

    comparisons = {
        "days": n,

        "KG_better_than_SEC": float(
            (u_kg > u_sec).mean()
        ),

        "SEC_better_than_KG": float(
            (u_sec > u_kg).mean()
        ),

        "BOTH_better_than_KG": float(
            (u_both > u_kg).mean()
        ),

        "BOTH_better_than_SEC": float(
            (u_both > u_sec).mean()
        ),

        "BOTH_better_than_best_single": float(
            (u_both > best_single).mean()
        ),

        "both_individually_positive": float(
            ((u_kg > 0) & (u_sec > 0)).mean()
        ),

        "both_individually_negative": float(
            ((u_kg < 0) & (u_sec < 0)).mean()
        ),

        "KG_positive_SEC_negative": float(
            ((u_kg > 0) & (u_sec < 0)).mean()
        ),

        "SEC_positive_KG_negative": float(
            ((u_sec > 0) & (u_kg < 0)).mean()
        ),

        "negative_synergy_rate": float(
            (synergy < 0).mean()
        ),

        "positive_synergy_rate": float(
            (synergy > 0).mean()
        ),

        "KG_SEC_utility_correlation": float(
            u_kg.corr(u_sec)
        ),

        "mean_oracle_one_step_utility": float(
            utility.max(axis=1).mean()
        ),
    }

    pd.DataFrame(
        [comparisons]
    ).to_csv(
        out / "source_comparison.csv",
        index=False,
    )

    oracle_counts = (
        best_action
        .value_counts()
        .reindex(SOURCES, fill_value=0)
    )

    oracle_counts.rename(
        "days"
    ).to_csv(
        out / "oracle_action_counts.csv"
    )

    # Block bootstrap for the mean effects.
    #
    # Blocks preserve some local time dependence
    # in this sampled trading-date sequence.
    rng = np.random.default_rng(42)

    series = {
        "SEC_minus_KG": (
            u_sec - u_kg
        ).to_numpy(dtype=float),

        "BOTH_minus_best_single": (
            benefit_over_best_single
        ).to_numpy(dtype=float),

        "SEC_given_KG": (
            incremental_sec_given_kg
        ).to_numpy(dtype=float),

        "KG_given_SEC": (
            incremental_kg_given_sec
        ).to_numpy(dtype=float),

        "SYNERGY": synergy.to_numpy(dtype=float),
    }

    block_size = max(
        1, min(args.block_size, n)
    )

    bootstrap_results = []

    for name, values in series.items():
        means = []

        for _ in range(args.bootstrap):
            idx = []

            while len(idx) < n:
                start = rng.integers(0, n)
                block = (
                    np.arange(
                        start,
                        start + block_size,
                    ) % n
                )
                idx.extend(block.tolist())

            idx = np.asarray(
                idx[:n],
                dtype=int,
            )

            means.append(
                values[idx].mean()
            )

        lo, hi = np.quantile(
            means, [0.025, 0.975]
        )

        bootstrap_results.append({
            "comparison": name,
            "mean": float(values.mean()),
            "ci95_low": float(lo),
            "ci95_high": float(hi),
            "blocks": block_size,
            "bootstrap_repeats": args.bootstrap,
        })

    bootstrap_df = pd.DataFrame(
        bootstrap_results
    )

    bootstrap_df.to_csv(
        out / "bootstrap_intervals.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("SOURCE UTILITY SUMMARY — ZERO RETRIEVAL COST")
    print("=" * 76)
    print(summary.to_string(index=False))

    print()
    print("=" * 76)
    print("KG / SEC / BOTH COMPARISON")
    print("=" * 76)

    for key, value in comparisons.items():
        print(f"{key}: {value:.8f}")

    print()
    print("=" * 76)
    print("ONE-STEP ORACLE ACTION DISTRIBUTION")
    print("=" * 76)
    print(oracle_counts.to_string())

    print()
    print("=" * 76)
    print("BLOCK BOOTSTRAP 95% INTERVALS")
    print("=" * 76)
    print(bootstrap_df.to_string(index=False))

    print()
    print("Source utility audit: PASS")
    print(
        "IMPORTANT: Oracle uses future realized returns."
    )
    print(
        "IMPORTANT: Negative synergy is not, by itself, "
        "proof of duplicate documents."
    )


if __name__ == "__main__":
    main()
