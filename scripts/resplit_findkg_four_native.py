#!/usr/bin/env python3
from pathlib import Path
import shutil
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "outputs/finrl_dkg_four_native/"
    "dataset/FinDKG-full"
)

OUT = (
    ROOT / "outputs/finrl_dkg_four_native/"
    "resplit_2018_2023/FinDKG-full"
)

OUT.mkdir(parents=True, exist_ok=True)

times = pd.read_csv(SOURCE / "time2id.txt")

required = {"TimeID", "DATE_WK"}

if not required.issubset(times.columns):
    raise RuntimeError(
        "Invalid time2id.csv schema"
    )

time_map = dict(zip(
    times["TimeID"].astype(int),
    pd.to_datetime(times["DATE_WK"])
))

all_parts = []

for name in ["train", "valid", "test"]:
    df = pd.read_csv(
        SOURCE / f"{name}.txt",
        sep=r"\s+",
        header=None,
    )

    if df.shape[1] != 5:
        raise RuntimeError(
            f"{name} does not have five fields"
        )

    df.columns = [
        "head", "relation", "tail",
        "time_id", "extra",
    ]

    df["date"] = df["time_id"].map(time_map)

    if df["date"].isna().any():
        raise RuntimeError(
            f"Unmapped times in {name}"
        )

    all_parts.append(df)

all_edges = pd.concat(
    all_parts, ignore_index=True
)

# Preserve source file event multiplicity.
# Do not silently discard original events.
all_edges = all_edges.sort_values([
    "time_id", "head", "relation", "tail"
])

masks = {
    "train": (
        all_edges["date"] < "2022-01-01"
    ),
    "valid": (
        (all_edges["date"] >= "2022-01-01")
        & (all_edges["date"] < "2023-01-01")
    ),
    "test": (
        all_edges["date"] >= "2023-01-01"
    ),
}

for name, mask in masks.items():
    part = all_edges.loc[
        mask,
        ["head", "relation", "tail",
         "time_id", "extra"]
    ]

    part.to_csv(
        OUT / f"{name}.txt",
        sep="\t",
        header=False,
        index=False,
    )

    print(
        f"{name.upper():6s}",
        "events =", len(part),
        "weeks =", part["time_id"].nunique(),
    )

for filename in [
    "entity2id.txt",
    "relation2id.txt",
    "time2id.txt",
    "stat.txt",
]:
    shutil.copy2(
        SOURCE / filename,
        OUT / filename
    )

status = {
    "dataset": "FinDKG four native relations",
    "train": "2018-2021",
    "validation": "2022",
    "test": "2023",
    "kg_data_end": "2023-01-01",
    "strict_pit_certified": False,
    "model_trained": False,
}

(OUT.parent / "resplit_manifest.json").write_text(
    json.dumps(status, indent=2)
)

print("\nRESPLIT COMPLETE")
print("Output:", OUT)
