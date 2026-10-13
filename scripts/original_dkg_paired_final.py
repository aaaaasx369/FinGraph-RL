#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
OUT = BASE / "research_audit"
OUT.mkdir(parents=True, exist_ok=True)

KEYS = ["tier", "agent", "seed", "split"]
METRICS = ["total_return", "sharpe", "max_drawdown"]

KG = BASE / "runs_multiseed/ORIGINAL_KGTRANSFORMER/runs.csv"
PLACEBO = BASE / "runs_multiseed/ORIGINAL_KG_PLACEBO/runs.csv"


def read_arm(path, condition, label):
    df = pd.read_csv(path)
    df = df[
        df["split"].eq("test")
        & df["condition"].eq(condition)
    ].copy()

    if df.empty:
        raise RuntimeError(f"No records for {label}")

    if df.duplicated(KEYS).any():
        raise RuntimeError(
            f"Duplicated experiment keys in {label}"
        )

    df["experiment"] = label
    return df[KEYS + METRICS + ["experiment"]]


# Three mutually exclusive, explicitly sourced groups.
clean = read_arm(KG, "CLEAN", "CLEAN")
kg = read_arm(KG, "FULL_DYNAMIC", "KGTRANSFORMER")
placebo = read_arm(
    PLACEBO, "FULL_DYNAMIC", "PLACEBO"
)

# Independent consistency check for repeated CLEAN baselines.
clean_other = read_arm(
    PLACEBO, "CLEAN", "CLEAN_OTHER"
)

check = clean.merge(
    clean_other,
    on=KEYS,
    how="outer",
    validate="one_to_one",
    suffixes=("_kgfile", "_placebofile"),
    indicator=True,
)

if not check["_merge"].eq("both").all():
    raise RuntimeError(
        "CLEAN records do not have matching keys"
    )

clean_differences = {}

for metric in METRICS:
    a = check[f"{metric}_kgfile"].to_numpy(float)
    b = check[f"{metric}_placebofile"].to_numpy(float)
    clean_differences[metric] = float(
        np.max(np.abs(a - b))
    )

print("=== CLEAN CROSS-FILE CONSISTENCY ===")
print(clean_differences)

if any(
    value > 1e-9
    for value in clean_differences.values()
):
    raise RuntimeError(
        "CLEAN baseline differs between arm CSVs. "
        "Do not pool baselines without investigating."
    )

combined = pd.concat(
    [clean, kg, placebo],
    ignore_index=True,
)

if combined.duplicated(
    KEYS + ["experiment"]
).any():
    raise RuntimeError(
        "Duplicated keys after correct arm selection"
    )

EXPECTED = {
    (tier, agent, seed, "test")
    for tier in ["conservative", "inclusive"]
    for agent in ["A2C", "PPO", "DDPG"]
    for seed in range(5)
}

for name, df in [
    ("CLEAN", clean),
    ("KGTRANSFORMER", kg),
    ("PLACEBO", placebo),
]:
    observed = set(
        map(tuple, df[KEYS].itertuples(
            index=False, name=None
        ))
    )
    if observed != EXPECTED:
        raise RuntimeError(
            f"{name}: expected 30 unique test runs; "
            f"got {len(observed)}"
        )

combined.to_csv(
    OUT / "canonical_2023_test_runs.csv",
    index=False,
)

print("\n=== GROUP COUNTS ===")
print(combined.groupby(
    ["experiment", "tier", "agent"]
).size().to_string())

# Summary: mean and sample SD
long_summary = (
    combined.groupby(
        ["tier", "agent", "experiment"],
        as_index=False,
    )
    .agg(
        n=("seed", "size"),
        return_mean=("total_return", "mean"),
        return_std=("total_return", "std"),
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        mdd_mean=("max_drawdown", "mean"),
        mdd_std=("max_drawdown", "std"),
    )
)

long_summary.to_csv(
    OUT / "canonical_performance_summary.csv",
    index=False,
)

comparisons = [
    ("KGTRANSFORMER", "CLEAN"),
    ("KGTRANSFORMER", "PLACEBO"),
    ("PLACEBO", "CLEAN"),
]

records = []
seed_records = []

for (tier, agent), sub in combined.groupby(
    ["tier", "agent"], sort=True
):
    for metric in METRICS:
        wide = sub.pivot(
            index="seed",
            columns="experiment",
            values=metric,
        ).sort_index()

        if list(wide.index) != list(range(5)):
            raise RuntimeError(
                f"Incomplete seeds: {tier} {agent}"
            )

        for left, right in comparisons:
            differences = (
                wide[left] - wide[right]
            ).to_numpy(dtype=float)

            n = len(differences)
            mean = float(differences.mean())
            std = float(differences.std(ddof=1))

            if std > 0:
                t, p = stats.ttest_1samp(
                    differences, 0
                )
                margin = (
                    stats.t.ppf(0.975, n - 1)
                    * std / np.sqrt(n)
                )
                dz = mean / std
            else:
                t = 0.0 if mean == 0 else np.inf
                p = 1.0 if mean == 0 else 0.0
                margin = 0.0
                dz = 0.0 if mean == 0 else np.inf

            records.append({
                "tier": tier,
                "agent": agent,
                "metric": metric,
                "comparison": f"{left}-{right}",
                "n": n,
                "mean_difference": mean,
                "sd_difference": std,
                "ci95_low": mean - margin,
                "ci95_high": mean + margin,
                "t_stat": float(t),
                "p_uncorrected": float(p),
                "cohens_dz": dz,
            })

            for seed, value in enumerate(differences):
                seed_records.append({
                    "tier": tier,
                    "agent": agent,
                    "metric": metric,
                    "comparison": f"{left}-{right}",
                    "seed": seed,
                    "difference": float(value),
                })

tests = pd.DataFrame(records)
tests["p_holm"] = np.nan

# Correct Holm step-down adjustment.
# One family = same tier + same metric,
# across 3 agents * 3 comparisons.
for _, indexes in tests.groupby(
    ["tier", "metric"]
).groups.items():
    positions = np.asarray(list(indexes))
    pvals = tests.loc[
        positions, "p_uncorrected"
    ].to_numpy(float)

    order = np.argsort(pvals)
    count = len(order)
    adjusted = np.zeros(count)
    previous = 0.0

    for rank, local_index in enumerate(order):
        value = min(
            1.0, (count-rank)*pvals[local_index]
        )
        previous = max(previous, value)
        adjusted[local_index] = previous

    tests.loc[positions, "p_holm"] = adjusted

tests.to_csv(
    OUT / "canonical_paired_tests.csv",
    index=False,
)

pd.DataFrame(seed_records).to_csv(
    OUT / "canonical_seed_differences.csv",
    index=False,
)

# Critical check: KG vs placebo must not be
# identical when raw runs differ.
kg_vs_placebo = tests[
    tests["comparison"].eq(
        "KGTRANSFORMER-PLACEBO"
    )
]

if np.allclose(
    kg_vs_placebo["mean_difference"],
    0,
    rtol=0,
    atol=1e-12,
):
    raise RuntimeError(
        "KG/Placebo still identical: investigate."
    )

print("\n=== 2023 TEST: MEAN PERFORMANCE ===")

print(
    long_summary.round(5).to_string(index=False)
)

print("\n=== CORRECT PAIRED RETURN TESTS ===")

selection = tests[
    tests["metric"].eq("total_return")
    & tests["comparison"].isin([
        "KGTRANSFORMER-CLEAN",
        "KGTRANSFORMER-PLACEBO",
    ])
]

print(
    selection[
        [
            "tier",
            "agent",
            "comparison",
            "mean_difference",
            "ci95_low",
            "ci95_high",
            "p_uncorrected",
            "p_holm",
        ]
    ].round(5).to_string(index=False)
)

print("\n=== CANONICAL PAIRED STATISTICS PASS ===")
print("Source: 2 original unmodified runs.csv")
print("Total canonical 2023 runs:", len(combined))
print("Total paired comparisons:", len(tests))
print("Output directory:", OUT)
