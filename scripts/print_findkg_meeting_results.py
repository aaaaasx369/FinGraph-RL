from pathlib import Path

import numpy as np
import pandas as pd


RAW_RUNS = Path(
    "results/findkg_5seed_50k/"
    "validation_runs.csv"
)

PCA_RUNS = Path(
    "results/findkg_pca_5seed_50k/"
    "validation_runs.csv"
)

OUT_DIR = Path(
    "results/findkg_meeting_summary"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

raw = pd.read_csv(RAW_RUNS)
pca = pd.read_csv(PCA_RUNS)


# Rename feature sets into presentation-friendly labels
raw["feature_set"] = raw["feature_set"].replace(
    {
        "F0": "Base",
        "F1": "FinDKG-200",
    }
)

pca["feature_set"] = pca["feature_set"].replace(
    {
        "PCA16": "FinDKG-PCA16",
        "PCA32": "FinDKG-PCA32",
    }
)

runs = pd.concat(
    [
        raw,
        pca,
    ],
    ignore_index=True,
)


ORDER = [
    "Base",
    "FinDKG-200",
    "FinDKG-PCA16",
    "FinDKG-PCA32",
]

AGENT_ORDER = [
    "A2C",
    "PPO",
    "DDPG",
]


# ============================================================
# Recompute mean/std
# ============================================================

summary = (
    runs
    .groupby(
        [
            "feature_set",
            "agent",
        ]
    )
    .agg(
        n_seeds=(
            "seed",
            "nunique",
        ),

        total_return_mean=(
            "total_return",
            "mean",
        ),

        total_return_std=(
            "total_return",
            "std",
        ),

        sharpe_mean=(
            "sharpe",
            "mean",
        ),

        sharpe_std=(
            "sharpe",
            "std",
        ),

        max_drawdown_mean=(
            "max_drawdown",
            "mean",
        ),

        max_drawdown_std=(
            "max_drawdown",
            "std",
        ),
    )
    .reset_index()
)

summary["feature_set"] = pd.Categorical(
    summary["feature_set"],
    categories=ORDER,
    ordered=True,
)

summary["agent"] = pd.Categorical(
    summary["agent"],
    categories=AGENT_ORDER,
    ordered=True,
)

summary = summary.sort_values(
    [
        "agent",
        "feature_set",
    ]
).reset_index(
    drop=True
)


# ============================================================
# Pretty main comparison
# ============================================================

pretty = summary.copy()

pretty["Return"] = (
    pretty.apply(
        lambda r:
        f"{r['total_return_mean']*100:+.2f}% "
        f"± {r['total_return_std']*100:.2f}%",
        axis=1,
    )
)

pretty["Sharpe"] = (
    pretty.apply(
        lambda r:
        f"{r['sharpe_mean']:+.3f} "
        f"± {r['sharpe_std']:.3f}",
        axis=1,
    )
)

pretty["Max DD"] = (
    pretty.apply(
        lambda r:
        f"{r['max_drawdown_mean']*100:.2f}% "
        f"± {r['max_drawdown_std']*100:.2f}%",
        axis=1,
    )
)


print()
print("=" * 100)
print("FINDKG → FINRL-X : 5-SEED × 50K VALIDATION RESULTS")
print("=" * 100)

print(
    pretty[
        [
            "agent",
            "feature_set",
            "n_seeds",
            "Return",
            "Sharpe",
            "Max DD",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Delta vs Base
# ============================================================

rows = []

for agent in AGENT_ORDER:

    a = summary[
        summary["agent"] == agent
    ].copy()

    base = a[
        a["feature_set"] == "Base"
    ]

    if len(base) != 1:
        raise RuntimeError(
            f"Base missing for {agent}"
        )

    base = base.iloc[0]

    for feature in [
        "FinDKG-200",
        "FinDKG-PCA16",
        "FinDKG-PCA32",
    ]:

        x = a[
            a["feature_set"] == feature
        ]

        if len(x) != 1:
            continue

        x = x.iloc[0]

        rows.append(
            {
                "agent":
                    agent,

                "feature":
                    feature,

                "return_delta_pp":
                    (
                        x["total_return_mean"]
                        -
                        base["total_return_mean"]
                    )
                    * 100,

                "sharpe_delta":
                    (
                        x["sharpe_mean"]
                        -
                        base["sharpe_mean"]
                    ),

                # Positive means drawdown improved
                # because -0.32 is better than -0.35.
                "mdd_improvement_pp":
                    (
                        x["max_drawdown_mean"]
                        -
                        base["max_drawdown_mean"]
                    )
                    * 100,
            }
        )

delta = pd.DataFrame(
    rows
)


print()
print("=" * 100)
print("DELTA RELATIVE TO BASE")
print("Positive Return Δ / Sharpe Δ / MDD improvement = better")
print("=" * 100)

delta_print = delta.copy()

delta_print[
    "return_delta_pp"
] = delta_print[
    "return_delta_pp"
].map(
    lambda x:
    f"{x:+.2f} pp"
)

delta_print[
    "sharpe_delta"
] = delta_print[
    "sharpe_delta"
].map(
    lambda x:
    f"{x:+.3f}"
)

delta_print[
    "mdd_improvement_pp"
] = delta_print[
    "mdd_improvement_pp"
].map(
    lambda x:
    f"{x:+.2f} pp"
)

print(
    delta_print.to_string(
        index=False
    )
)


# ============================================================
# Paired seed comparison
#
# Much more useful than only looking at means:
# same agent + same seed, compare feature against Base.
# ============================================================

paired_rows = []

for agent in AGENT_ORDER:

    base = runs[
        (
            runs["agent"] == agent
        )
        &
        (
            runs["feature_set"] == "Base"
        )
    ][
        [
            "seed",
            "total_return",
            "sharpe",
            "max_drawdown",
        ]
    ].copy()

    for feature in [
        "FinDKG-200",
        "FinDKG-PCA16",
        "FinDKG-PCA32",
    ]:

        alt = runs[
            (
                runs["agent"] == agent
            )
            &
            (
                runs["feature_set"] == feature
            )
        ][
            [
                "seed",
                "total_return",
                "sharpe",
                "max_drawdown",
            ]
        ].copy()

        m = base.merge(
            alt,
            on="seed",
            suffixes=(
                "_base",
                "_alt",
            ),
        )

        if len(m) == 0:
            continue

        return_delta = (
            m["total_return_alt"]
            -
            m["total_return_base"]
        )

        sharpe_delta = (
            m["sharpe_alt"]
            -
            m["sharpe_base"]
        )

        mdd_delta = (
            m["max_drawdown_alt"]
            -
            m["max_drawdown_base"]
        )

        paired_rows.append(
            {
                "agent":
                    agent,

                "feature":
                    feature,

                "paired_seeds":
                    len(m),

                "return_win_seeds":
                    int(
                        (return_delta > 0)
                        .sum()
                    ),

                "sharpe_win_seeds":
                    int(
                        (sharpe_delta > 0)
                        .sum()
                    ),

                "mdd_win_seeds":
                    int(
                        (mdd_delta > 0)
                        .sum()
                    ),

                "mean_return_delta_pp":
                    return_delta.mean()
                    * 100,

                "mean_sharpe_delta":
                    sharpe_delta.mean(),

                "mean_mdd_improvement_pp":
                    mdd_delta.mean()
                    * 100,
            }
        )


paired = pd.DataFrame(
    paired_rows
)

print()
print("=" * 100)
print("PAIRED-SEED ROBUSTNESS")
print("Win seeds = how many of the same 5 seeds beat Base")
print("=" * 100)

paired_print = paired.copy()

for c in [
    "return_win_seeds",
    "sharpe_win_seeds",
    "mdd_win_seeds",
]:

    paired_print[c] = (
        paired_print[c]
        .astype(str)
        +
        "/"
        +
        paired_print[
            "paired_seeds"
        ].astype(str)
    )

paired_print[
    "mean_return_delta_pp"
] = paired_print[
    "mean_return_delta_pp"
].map(
    lambda x:
    f"{x:+.2f} pp"
)

paired_print[
    "mean_sharpe_delta"
] = paired_print[
    "mean_sharpe_delta"
].map(
    lambda x:
    f"{x:+.3f}"
)

paired_print[
    "mean_mdd_improvement_pp"
] = paired_print[
    "mean_mdd_improvement_pp"
].map(
    lambda x:
    f"{x:+.2f} pp"
)

print(
    paired_print[
        [
            "agent",
            "feature",
            "return_win_seeds",
            "sharpe_win_seeds",
            "mdd_win_seeds",
            "mean_return_delta_pp",
            "mean_sharpe_delta",
            "mean_mdd_improvement_pp",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Cross-agent descriptive average
#
# This is descriptive only.
# Do NOT interpret it as a statistical test.
# ============================================================

cross = (
    delta
    .groupby(
        "feature"
    )
    .agg(
        avg_return_delta_pp=(
            "return_delta_pp",
            "mean",
        ),

        avg_sharpe_delta=(
            "sharpe_delta",
            "mean",
        ),

        avg_mdd_improvement_pp=(
            "mdd_improvement_pp",
            "mean",
        ),
    )
    .reset_index()
)

print()
print("=" * 100)
print("CROSS-AGENT DESCRIPTIVE AVERAGE")
print("(Descriptive only; not a significance test)")
print("=" * 100)

print(
    cross.to_string(
        index=False,
        formatters={
            "avg_return_delta_pp":
                lambda x:
                f"{x:+.2f} pp",

            "avg_sharpe_delta":
                lambda x:
                f"{x:+.3f}",

            "avg_mdd_improvement_pp":
                lambda x:
                f"{x:+.2f} pp",
        }
    )
)


# ============================================================
# Save
# ============================================================

summary.to_csv(
    OUT_DIR
    /
    "findkg_all_summary.csv",
    index=False,
)

delta.to_csv(
    OUT_DIR
    /
    "findkg_delta_vs_base.csv",
    index=False,
)

paired.to_csv(
    OUT_DIR
    /
    "findkg_paired_seed_comparison.csv",
    index=False,
)

print()
print("=" * 100)
print("FILES SAVED")
print("=" * 100)

print(
    OUT_DIR
    /
    "findkg_all_summary.csv"
)

print(
    OUT_DIR
    /
    "findkg_delta_vs_base.csv"
)

print(
    OUT_DIR
    /
    "findkg_paired_seed_comparison.csv"
)
