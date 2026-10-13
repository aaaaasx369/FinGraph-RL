#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

BASE = Path(
    "outputs/finrl_dkg_pit_2022_2023/finrl_runs"
)

OUT = BASE / "merged_5seeds"
OUT.mkdir(parents=True, exist_ok=True)

for model in ["FULL15", "FOUR4"]:
    paths = [
        BASE / model / "runs.csv",
        BASE / f"{model}_seeds1to4" / "runs.csv",
    ]

    frames = []

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(
        frames, ignore_index=True
    )

    keys = [
        "tier", "condition",
        "agent", "seed", "split"
    ]

    if combined.duplicated(keys).any():
        raise RuntimeError(
            f"{model}: duplicated experiment keys"
        )

    expected = (
        2 * 4 * 3 * 5 * 2
    )

    if len(combined) != expected:
        raise RuntimeError(
            f"{model}: expected {expected} rows, "
            f"got {len(combined)}"
        )

    seeds = sorted(combined["seed"].unique())
    if seeds != [0, 1, 2, 3, 4]:
        raise RuntimeError(
            f"{model}: unexpected seeds {seeds}"
        )

    output = OUT / f"{model}_runs.csv"

    combined.to_csv(
        output, index=False
    )

    print(model)
    print("  Rows:", len(combined))
    print("  Seeds:", seeds)
    print("  Output:", output)

print("\n=== FIVE-SEED MERGE COMPLETE ===")
