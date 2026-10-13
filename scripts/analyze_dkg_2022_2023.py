#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(
    "outputs/finrl_dkg_pit_2022_2023/"
    "finrl_runs/merged_5seeds"
)

FILES = {
    "FULL15": BASE / "FULL15_runs.csv",
    "FOUR4": BASE / "FOUR4_runs.csv",
}

METRICS = [
    "total_return",
    "sharpe",
    "max_drawdown",
]

KEYS = ["tier", "agent", "seed"]

rng = np.random.default_rng(2026)

frames = []

for model, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = set(
        KEYS + METRICS + ["condition", "split"]
    )

    if not required.issubset(df.columns):
        raise RuntimeError(
            f"{model}: missing columns "
            f"{sorted(required - set(df.columns))}"
        )

    if df.duplicated(
        KEYS + ["condition", "split"]
    ).any():
        raise RuntimeError(
            f"{model}: duplicate experiment keys"
        )

    df = df[df["split"].eq("test")].copy()
    df["kg_model"] = model

    expected = 2 * 3 * 5 * 4
    if len(df) != expected:
        raise RuntimeError(
            f"{model}: expected {expected} "
            f"test rows, got {len(df)}"
        )

    for metric in METRICS:
        df[metric] = pd.to_numeric(
            df[metric], errors="raise"
        )

    if not np.isfinite(
        df[METRICS].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(
            f"{model}: nonfinite metrics"
        )

    frames.append(df)

all_runs = pd.concat(
    frames, ignore_index=True
)

# --------------------------------------------------
# 1. Descriptive performance
# --------------------------------------------------

summary = (
    all_runs
    .groupby(
        ["tier", "agent", "kg_model", "condition"]
    )[METRICS]
    .agg(["mean", "std"])
)

summary.columns = [
    f"{metric}_{stat}"
    for metric, stat in summary.columns
]

summary = summary.reset_index()

summary.to_csv(
    BASE / "five_seed_summary.csv",
    index=False,
)

# --------------------------------------------------
# 2. Paired comparisons
# --------------------------------------------------

comparisons = [
    ("FULL_DYNAMIC", "CLEAN"),
    ("FULL_DYNAMIC", "FULL_STATIC"),
    ("FULL_DYNAMIC", "FULL_DYNAMIC_PLACEBO"),
]

paired_rows = []
seed_rows = []

def paired_stats(values):
    values = np.asarray(values, dtype=float)

    if len(values) != 5:
        raise RuntimeError(
            "Expected exactly 5 paired seeds"
        )

    # Descriptive bootstrap CI. With n=5 this is
    # exploratory, not definitive evidence.
    samples = rng.choice(
        values,
        size=(10000, len(values)),
        replace=True,
    )

    ci = np.percentile(
        samples.mean(axis=1),
        [2.5, 97.5],
    )

    return {
        "delta_mean": values.mean(),
        "delta_std": values.std(ddof=1),
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "positive_seeds": int((values > 0).sum()),
    }


for (tier, agent, model), group in all_runs.groupby(
    ["tier", "agent", "kg_model"]
):
    pivot = group.pivot(
        index="seed",
        columns="condition",
        values=METRICS,
    ).sort_index()

    if list(pivot.index) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Incomplete seed pairing")

    for treatment, control in comparisons:
        for metric in METRICS:
            delta = (
                pivot[(metric, treatment)]
                - pivot[(metric, control)]
            ).to_numpy(dtype=float)

            stats = paired_stats(delta)

            paired_rows.append({
                "tier": tier,
                "agent": agent,
                "kg_model": model,
                "comparison": (
                    f"{treatment} - {control}"
                ),
                "metric": metric,
                **stats,
            })

            for seed, value in zip(
                pivot.index, delta
            ):
                seed_rows.append({
                    "tier": tier,
                    "agent": agent,
                    "kg_model": model,
                    "comparison": (
                        f"{treatment} - {control}"
                    ),
                    "metric": metric,
                    "seed": seed,
                    "delta": value,
                })

# --------------------------------------------------
# 3. FULL15 versus FOUR4
# --------------------------------------------------

dynamic = all_runs[
    all_runs["condition"].eq("FULL_DYNAMIC")
].copy()

for (tier, agent), group in dynamic.groupby(
    ["tier", "agent"]
):
    full = group[
        group["kg_model"].eq("FULL15")
    ].set_index("seed")

    four = group[
        group["kg_model"].eq("FOUR4")
    ].set_index("seed")

    if (
        sorted(full.index.tolist()) != [0,1,2,3,4]
        or sorted(four.index.tolist()) != [0,1,2,3,4]
    ):
        raise RuntimeError(
            "Incomplete FULL15/FOUR4 pairing"
        )

    for metric in METRICS:
        delta = (
            full.loc[range(5), metric].to_numpy()
            - four.loc[range(5), metric].to_numpy()
        )

        stats = paired_stats(delta)

        paired_rows.append({
            "tier": tier,
            "agent": agent,
            "kg_model": "FULL15_vs_FOUR4",
            "comparison": (
                "FULL15_DYNAMIC - FOUR4_DYNAMIC"
            ),
            "metric": metric,
            **stats,
        })

        for seed, value in enumerate(delta):
            seed_rows.append({
                "tier": tier,
                "agent": agent,
                "kg_model": "FULL15_vs_FOUR4",
                "comparison": (
                    "FULL15_DYNAMIC - FOUR4_DYNAMIC"
                ),
                "metric": metric,
                "seed": seed,
                "delta": value,
            })

paired = pd.DataFrame(paired_rows)
per_seed = pd.DataFrame(seed_rows)

paired.to_csv(
    BASE / "five_seed_paired_comparisons.csv",
    index=False,
)

per_seed.to_csv(
    BASE / "five_seed_individual_deltas.csv",
    index=False,
)

# --------------------------------------------------
# 4. Console report
# --------------------------------------------------

print("\n" + "=" * 75)
print("2022-2023 FIVE-SEED PERFORMANCE")
print("=" * 75)

selected = summary[
    summary["tier"].eq("conservative")
]

print(
    selected[
        [
            "agent",
            "kg_model",
            "condition",
            "total_return_mean",
            "sharpe_mean",
            "max_drawdown_mean",
        ]
    ]
    .sort_values(
        ["agent", "kg_model", "condition"]
    )
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("PAIRED: DYNAMIC - ZERO PLACEBO")
print("CONSERVATIVE / TOTAL RETURN")
print("=" * 75)

selected = paired[
    paired["tier"].eq("conservative")
    & paired["metric"].eq("total_return")
    & paired["comparison"].eq(
        "FULL_DYNAMIC - FULL_DYNAMIC_PLACEBO"
    )
]

print(
    selected[
        [
            "agent", "kg_model",
            "delta_mean",
            "delta_std",
            "ci95_low",
            "ci95_high",
            "positive_seeds",
        ]
    ]
    .sort_values(["agent", "kg_model"])
    .to_string(index=False)
)

print("\n" + "=" * 75)
print("PAIRED: FULL15 - FOUR4")
print("CONSERVATIVE / TOTAL RETURN")
print("=" * 75)

selected = paired[
    paired["tier"].eq("conservative")
    & paired["metric"].eq("total_return")
    & paired["kg_model"].eq("FULL15_vs_FOUR4")
]

print(
    selected[
        [
            "agent",
            "delta_mean",
            "ci95_low",
            "ci95_high",
            "positive_seeds",
        ]
    ]
    .sort_values("agent")
    .to_string(index=False)
)

print("\n=== SAVED ===")
print(BASE / "five_seed_summary.csv")
print(BASE / "five_seed_paired_comparisons.csv")
print(BASE / "five_seed_individual_deltas.csv")

print("\nCaution: bootstrap CIs are exploratory (n=5).")
print("Strict PIT certification remains incomplete.")
