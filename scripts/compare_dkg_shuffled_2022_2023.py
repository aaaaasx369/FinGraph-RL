#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(
    "outputs/finrl_dkg_pit_2022_2023/finrl_runs"
)
MERGED = BASE / "merged_5seeds"

METRICS = ["total_return", "sharpe", "max_drawdown"]
KEYS = ["tier", "agent", "seed"]

results = []

for model in ["FULL15", "FOUR4"]:

    real_path = MERGED / f"{model}_runs.csv"
    shuffled_path = (
        BASE / f"{model}_SHUFFLED" / "runs.csv"
    )

    if not real_path.exists():
        raise FileNotFoundError(real_path)

    if not shuffled_path.exists():
        raise FileNotFoundError(
            f"{shuffled_path}\n"
            f"{model} Shuffled training may not be complete."
        )

    real = pd.read_csv(real_path)
    shuffled = pd.read_csv(shuffled_path)

    def select(df):
        return df[
            df["split"].eq("test")
            & df["condition"].eq("FULL_DYNAMIC")
        ][KEYS + METRICS].copy()

    a = select(real)
    b = select(shuffled)

    if a.duplicated(KEYS).any():
        raise RuntimeError(
            f"{model}: duplicated REAL experiment keys"
        )

    if b.duplicated(KEYS).any():
        raise RuntimeError(
            f"{model}: duplicated SHUFFLED experiment keys"
        )

    paired = a.merge(
        b,
        on=KEYS,
        suffixes=("_real", "_shuffled"),
        validate="one_to_one",
    )

    expected = 2 * 3 * 5

    if len(a) != expected or len(b) != expected:
        raise RuntimeError(
            f"{model}: expected {expected} results "
            f"per condition, got {len(a)} and {len(b)}"
        )

    if len(paired) != expected:
        raise RuntimeError(
            f"{model}: incomplete paired comparisons"
        )

    for metric in METRICS:
        paired[f"delta_{metric}"] = (
            paired[f"{metric}_real"]
            - paired[f"{metric}_shuffled"]
        )

    paired["kg_model"] = model

    results.append(paired)

combined = pd.concat(
    results,
    ignore_index=True,
)

output = BASE / "shuffled_comparison.csv"
combined.to_csv(output, index=False)

summary = (
    combined
    .groupby(["tier", "agent", "kg_model"])
    .agg(
        real_return=("total_return_real", "mean"),
        shuffled_return=("total_return_shuffled", "mean"),
        delta_return=("delta_total_return", "mean"),
        delta_sharpe=("delta_sharpe", "mean"),
        delta_mdd=("delta_max_drawdown", "mean"),
        positive_seeds=(
            "delta_total_return",
            lambda x: int((x > 0).sum()),
        ),
    )
    .reset_index()
)

summary_path = (
    BASE / "shuffled_comparison_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)

print("\n" + "=" * 75)
print("REAL KG MINUS SHUFFLED KG")
print("2022–2023 TEST / CONSERVATIVE / 5 SEEDS")
print("=" * 75)

selected = summary[
    summary["tier"].eq("conservative")
]

print(
    selected[
        [
            "agent",
            "kg_model",
            "real_return",
            "shuffled_return",
            "delta_return",
            "delta_sharpe",
            "delta_mdd",
            "positive_seeds",
        ]
    ]
    .sort_values(["agent", "kg_model"])
    .to_string(index=False)
)

print("\n=== SAVED ===")
print(output)
print(summary_path)

print("\nStrict PIT certification: NOT COMPLETE")
