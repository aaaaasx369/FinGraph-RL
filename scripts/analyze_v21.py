#!/usr/bin/env python3
"""V21 full experimental analysis, paired by seed."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


CONDITIONS = [
    "CLEAN",
    "FULL_STATIC",
    "FULL_DYNAMIC_PLACEBO",
    "FULL_DYNAMIC",
]

COMPARISONS = [
    ("FULL_STATIC", "CLEAN"),
    ("FULL_DYNAMIC_PLACEBO", "FULL_STATIC"),
    ("FULL_DYNAMIC", "FULL_STATIC"),
    ("FULL_DYNAMIC", "FULL_DYNAMIC_PLACEBO"),
]

METRICS = [
    "total_return",
    "sharpe",
    "max_drawdown",
]


def holm_adjust(pvalues):
    """Holm family-wise error rate correction."""
    p = np.asarray(pvalues, dtype=float)
    result = np.full(len(p), np.nan)

    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return result

    ordered = valid[np.argsort(p[valid])]
    m = len(ordered)
    adjusted = np.maximum.accumulate(
        np.minimum(
            1.0,
            p[ordered] * np.arange(m, 0, -1)
        )
    )
    result[ordered] = adjusted
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="results/v21_full"
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    file = root / "runs.csv"

    if not file.exists():
        raise FileNotFoundError(file)

    df = pd.read_csv(file)

    required = [
        "tier", "condition", "agent",
        "seed", "split", *METRICS,
    ]

    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    keys = ["tier", "condition", "agent", "seed", "split"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate experimental runs")

    expected = (
        2 * 4 * 3 * 5 * 2
    )

    print("\n=== V21 COMPLETENESS ===")
    print("Expected evaluations:", expected)
    print("Actual evaluations:", len(df))

    if len(df) != expected:
        raise ValueError(
            "Expected 240 evaluations for the "
            "2-tier, 4-condition, 3-agent, "
            "5-seed, 2-split experiment"
        )

    expected_tiers = {"conservative", "inclusive"}
    expected_agents = {"A2C", "PPO", "DDPG"}
    expected_splits = {"val", "test"}
    expected_seeds = set(range(5))

    if set(df["tier"]) != expected_tiers:
        raise ValueError("Unexpected tiers")
    if set(df["condition"]) != set(CONDITIONS):
        raise ValueError("Unexpected conditions")
    if set(df["agent"]) != expected_agents:
        raise ValueError("Unexpected agents")
    if set(df["split"]) != expected_splits:
        raise ValueError("Unexpected splits")
    if set(df["seed"]) != expected_seeds:
        raise ValueError("Unexpected seeds")

    for c in METRICS:
        if not np.isfinite(
            df[c].to_numpy(dtype=float)
        ).all():
            raise ValueError(f"Nonfinite metric: {c}")

    print("Experiment completeness: PASS")

    summary = (
        df.groupby(
            ["tier", "agent", "condition", "split"]
        )[METRICS]
        .agg(["mean", "std", "count"])
    )
    summary.columns = [
        f"{a}_{b}" for a, b in summary.columns
    ]
    summary.reset_index().to_csv(
        root / "verified_summary.csv",
        index=False,
    )

    rows = []

    for tier in sorted(expected_tiers):
        for agent in sorted(expected_agents):
            for split in sorted(expected_splits):
                group = df[
                    (df["tier"] == tier)
                    & (df["agent"] == agent)
                    & (df["split"] == split)
                ]

                for candidate, baseline in COMPARISONS:
                    a = group[
                        group["condition"] == candidate
                    ].set_index("seed")

                    b = group[
                        group["condition"] == baseline
                    ].set_index("seed")

                    seeds = a.index.intersection(b.index)

                    if set(seeds) != expected_seeds:
                        raise ValueError(
                            "Missing paired seeds: "
                            f"{tier}/{agent}/{split}/"
                            f"{candidate}"
                        )

                    for metric in METRICS:
                        delta = (
                            a.loc[sorted(seeds), metric]
                            - b.loc[sorted(seeds), metric]
                        ).to_numpy(dtype=float)

                        n = len(delta)
                        mean = float(delta.mean())
                        std = float(delta.std(ddof=1))

                        if np.isclose(std, 0, atol=1e-12):
                            if np.isclose(mean, 0, atol=1e-12):
                                pvalue = 1.0
                            else:
                                pvalue = 0.0
                            ci_low = ci_high = mean
                        else:
                            se = std / np.sqrt(n)
                            critical = stats.t.ppf(
                                0.975, df=n-1
                            )
                            ci_low = mean - critical * se
                            ci_high = mean + critical * se
                            pvalue = float(
                                stats.ttest_1samp(
                                    delta, 0
                                ).pvalue
                            )

                        rows.append({
                            "tier": tier,
                            "agent": agent,
                            "split": split,
                            "candidate": candidate,
                            "baseline": baseline,
                            "metric": metric,
                            "n_seeds": n,
                            "mean_difference": mean,
                            "std_difference": std,
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "p_value_unadjusted": pvalue,
                            "positive_seed_count": int(
                                (delta > 0).sum()
                            ),
                        })

    result = pd.DataFrame(rows)

    # Adjust across all pre-specified comparisons
    # within a given evaluation split.
    result["p_value_holm"] = np.nan

    for split, indices in result.groupby(
        "split"
    ).groups.items():
        result.loc[
            indices, "p_value_holm"
        ] = holm_adjust(
            result.loc[
                indices, "p_value_unadjusted"
            ].to_numpy()
        )

    result.to_csv(
        root / "paired_statistics.csv",
        index=False,
    )

    test = result[
        (result["split"] == "test")
        & (result["candidate"] == "FULL_DYNAMIC")
        & (
            result["baseline"]
            == "FULL_DYNAMIC_PLACEBO"
        )
    ]

    print("\n=== DYNAMIC VS PLACEBO: 2023 TEST ===")
    print(
        test[
            [
                "tier", "agent", "metric",
                "n_seeds", "mean_difference",
                "ci95_low", "ci95_high",
                "positive_seed_count",
                "p_value_unadjusted",
                "p_value_holm",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== 2023 TEST MEAN PERFORMANCE ===")

    performance = (
        df[df["split"] == "test"]
        .groupby(["tier", "agent", "condition"])[METRICS]
        .mean()
    )

    print(performance.round(4).to_string())

    print("\nSaved:")
    print(root / "verified_summary.csv")
    print(root / "paired_statistics.csv")
    print(
        "\nHolm adjustment covers all 72 comparisons "
        "within each split; exploratory interpretation."
    )


if __name__ == "__main__":
    main()
