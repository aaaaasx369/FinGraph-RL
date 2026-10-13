#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parents[1]
FOUR = ROOT / "outputs/finrl_dkg_four_native"
OLD = ROOT / "outputs/finrl_dkg_original_full"

OUT = FOUR / "research_comparison"
OUT.mkdir(parents=True, exist_ok=True)

def locate(*paths):
    for p in paths:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Cannot find original FULL-15 runs.csv.\n"
        + "\n".join(str(p) for p in paths)
    )

sources = {
    "CLEAN": (
        FOUR / "runs/FOUR_NATIVE_KGTRANSFORMER/runs.csv",
        "CLEAN",
    ),
    "FULL_15": (
        locate(
            OLD / "runs/ORIGINAL_KGTRANSFORMER/runs.csv",
            ROOT / "runs_multiseed/ORIGINAL_KGTRANSFORMER/runs.csv",
        ),
        "FULL_DYNAMIC",
    ),
    "FOUR_4": (
        FOUR / "runs/FOUR_NATIVE_KGTRANSFORMER/runs.csv",
        "FULL_DYNAMIC",
    ),
    "FOUR_PLACEBO": (
        FOUR / "runs/FOUR_NATIVE_KG_PLACEBO/runs.csv",
        "FULL_DYNAMIC",
    ),
    "FOUR_STATIC": (
        FOUR / "runs/FOUR_NATIVE_KGTRANSFORMER/runs.csv",
        "FULL_STATIC",
    ),
}

KEYS = ["tier", "agent", "seed", "split"]
METRICS = [
    "total_return",
    "sharpe",
    "max_drawdown",
    "annual_volatility",
]

frames = []

for arm, (path, condition) in sources.items():
    df = pd.read_csv(path)

    required = set(
        KEYS + METRICS + ["condition"]
    )
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{arm}: missing columns {missing}"
        )

    df = df[
        (df["condition"] == condition)
        & (df["split"] == "test")
    ].copy()

    df["seed"] = df["seed"].astype(int)

    if df.duplicated(KEYS).any():
        raise RuntimeError(
            f"{arm}: duplicate experimental keys"
        )

    expected = {
        (tier, agent, seed)
        for tier in ["conservative", "inclusive"]
        for agent in ["A2C", "PPO", "DDPG"]
        for seed in range(5)
    }

    found = set(
        zip(df["tier"], df["agent"], df["seed"])
    )

    if found != expected:
        raise RuntimeError(
            f"{arm}: incomplete or unexpected run keys; "
            f"missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )

    df["arm"] = arm
    frames.append(
        df[KEYS + ["arm"] + METRICS]
    )

all_runs = pd.concat(
    frames, ignore_index=True
)

all_runs.to_csv(
    OUT / "canonical_test_runs.csv",
    index=False,
)

summary = (
    all_runs.groupby(
        ["tier", "agent", "arm"]
    )[METRICS]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(
    OUT / "performance_summary.csv",
    index=False,
)

comparisons = [
    ("FOUR_4", "CLEAN"),
    ("FOUR_4", "FULL_15"),
    ("FOUR_4", "FOUR_PLACEBO"),
    ("FOUR_4", "FOUR_STATIC"),
]

tests = []

for tier in ["conservative", "inclusive"]:
    for agent in ["A2C", "PPO", "DDPG"]:
        data = all_runs[
            (all_runs["tier"] == tier)
            & (all_runs["agent"] == agent)
        ]

        for left, right in comparisons:
            a = (
                data[data["arm"] == left]
                .set_index("seed")
                .sort_index()
            )
            b = (
                data[data["arm"] == right]
                .set_index("seed")
                .sort_index()
            )

            if list(a.index) != list(b.index):
                raise RuntimeError(
                    "Unmatched seeds"
                )

            for metric in [
                "total_return",
                "sharpe",
                "max_drawdown",
            ]:
                x = a[metric].to_numpy(dtype=float)
                y = b[metric].to_numpy(dtype=float)

                if not (
                    np.isfinite(x).all()
                    and np.isfinite(y).all()
                ):
                    raise RuntimeError(
                        "Nonfinite metrics"
                    )

                differences = x - y

                if np.allclose(
                    differences, differences[0],
                    rtol=0, atol=1e-12
                ):
                    p = (
                        1.0
                        if np.isclose(
                            differences[0], 0
                        )
                        else 0.0
                    )
                else:
                    p = float(
                        ttest_rel(x, y).pvalue
                    )

                tests.append({
                    "tier": tier,
                    "agent": agent,
                    "comparison": f"{left} - {right}",
                    "metric": metric,
                    "n_seeds": len(x),
                    "mean_difference": (
                        differences.mean()
                    ),
                    "std_difference": (
                        differences.std(ddof=1)
                    ),
                    "p_raw": p,
                })

tests = pd.DataFrame(tests)

# Holm adjustment across the explicitly
# defined family of all reported tests.
order = np.argsort(tests["p_raw"].to_numpy())
pvals = tests["p_raw"].to_numpy()[order]

m = len(pvals)
adjusted = np.maximum.accumulate(
    np.minimum(
        1.0,
        pvals * (m - np.arange(m))
    )
)

tests["p_holm"] = np.nan
tests.loc[
    tests.index[order], "p_holm"
] = adjusted

tests["significant_holm_005"] = (
    tests["p_holm"] < 0.05
)

tests.to_csv(
    OUT / "paired_statistical_tests.csv",
    index=False,
)

print("\n=== 2023 TEST PERFORMANCE ===")

selected = summary[
    summary["tier"] == "conservative"
]

print(
    selected[
        [
            "agent", "arm",
            "total_return_mean",
            "sharpe_mean",
            "max_drawdown_mean",
        ]
    ].round(4).to_string(index=False)
)

print("\n=== FOUR-4 PAIRED COMPARISONS ===")

print(
    tests[
        (tests["tier"] == "conservative")
        & (tests["metric"] == "total_return")
    ][
        [
            "agent", "comparison",
            "mean_difference",
            "p_raw", "p_holm",
        ]
    ].round(5).to_string(index=False)
)

print("\nFiles saved:", OUT)
print("Strict PIT: NOT CERTIFIED")
