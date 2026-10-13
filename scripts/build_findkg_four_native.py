#!/usr/bin/env python3
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "references/Financial-TKG/"
    "FinDKG_dataset/FinDKG-full"
)

OUT = (
    ROOT / "outputs/finrl_dkg_four_native/"
    "dataset/FinDKG-full"
)

OUT.mkdir(parents=True, exist_ok=True)

WANTED = [
    "Invests_In",
    "Operate_In",
    "Positive_Impact_On",
    "Negative_Impact_On",
]

STOCKS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
]


def load_relations(path):
    result = {}

    for line in path.read_text().splitlines():
        if not line.strip():
            continue

        name, number = line.rsplit(None, 1)
        result[name.strip()] = int(number)

    return result


def read_edges(path):
    # FinDKG uses five tab-separated integer fields.
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if df.shape[1] < 4:
        raise RuntimeError(
            f"Invalid edge format: {path}"
        )

    df = df.iloc[:, :5].copy()

    if df.shape[1] == 4:
        df[4] = 0

    df.columns = [
        "head", "relation",
        "tail", "time", "extra"
    ]

    for c in df.columns:
        df[c] = pd.to_numeric(
            df[c],
            errors="raise",
        ).astype("int64")

    return df


def read_entities(path):
    # Official FinDKG has four TAB-separated
    # columns. Names can contain whitespace.
    result = {}

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")

            if len(parts) != 4:
                # Only use right split as a fallback:
                # the three trailing fields are numeric.
                parts = line.rsplit(None, 3)

            if len(parts) != 4:
                raise RuntimeError(
                    f"Entity format error at line {line_no}"
                )

            name = parts[0].strip()
            entity_id = int(parts[1])

            result[name] = entity_id

    return result


relation_ids = load_relations(
    SOURCE / "relation2id.txt"
)

missing = [
    r for r in WANTED
    if r not in relation_ids
]

if missing:
    raise RuntimeError(
        "Missing original relations: " + str(missing)
    )

old_to_new = {
    relation_ids[name]: i
    for i, name in enumerate(WANTED)
}

entities = read_entities(
    SOURCE / "entity2id.txt"
)

stats = (
    SOURCE / "stat.txt"
).read_text().split()

num_entities = int(stats[0])
old_relations = int(stats[1])

if len(entities) != num_entities:
    raise RuntimeError(
        "Entity count does not match stat.txt"
    )

# Original FinDKG-full entity IDs used in the
# previously successful 15-relation experiment.
# Preserve original entity IDs for compatibility.
target_ids = {
    "AAPL": 9,
    "MSFT": 30,
    "NVDA": 221,
    "AMZN": 20,
    "GOOGL": 11,
}

id_to_name = {
    entity_id: name
    for name, entity_id in entities.items()
}

for ticker, entity_id in target_ids.items():
    if entity_id not in id_to_name:
        raise RuntimeError(
            f"Original entity ID missing: "
            f"{ticker}={entity_id}"
        )

    print(
        f"ENTITY CHECK: {ticker} "
        f"-> {entity_id} "
        f"-> {id_to_name[entity_id]}"
    )

if len(set(target_ids.values())) != 5:
    raise RuntimeError("Target entity IDs are not unique")

print("=== FOUR NATIVE RELATIONS ===")
print("Original relation IDs:", relation_ids)
print("Selected:", old_to_new)
print("Target entity IDs:", target_ids)

shutil.copy2(
    SOURCE / "entity2id.txt",
    OUT / "entity2id.txt",
)

shutil.copy2(
    SOURCE / "time2id.txt",
    OUT / "time2id.txt",
)

(OUT / "relation2id.txt").write_text(
    "".join(
        f"{name}\t{i}\n"
        for i, name in enumerate(WANTED)
    )
)

all_coverage = []
all_split_data = {}

for split in ["train", "valid", "test"]:
    df = read_edges(
        SOURCE / f"{split}.txt"
    )

    filtered = df[
        df["relation"].isin(old_to_new)
    ].copy()

    filtered["relation"] = (
        filtered["relation"]
        .map(old_to_new)
        .astype("int64")
    )

    filtered.to_csv(
        OUT / f"{split}.txt",
        sep="\t",
        index=False,
        header=False,
    )

    all_split_data[split] = filtered

    for rid, relation in enumerate(WANTED):
        part = filtered[
            filtered["relation"].eq(rid)
        ]

        target = part[
            part["head"].isin(
                target_ids.values()
            )
            | part["tail"].isin(
                target_ids.values()
            )
        ]

        all_coverage.append({
            "split": split,
            "relation": relation,
            "relation_id": rid,
            "events": len(part),
            "target_events": len(target),
            "distinct_heads": part[
                "head"
            ].nunique(),
            "distinct_tails": part[
                "tail"
            ].nunique(),
            "distinct_weeks": part[
                "time"
            ].nunique(),
        })

coverage = pd.DataFrame(all_coverage)

coverage.to_csv(
    OUT.parent / "relation_coverage.csv",
    index=False,
)

for split in ["train", "valid", "test"]:
    if all_split_data[split].empty:
        raise RuntimeError(
            f"No events in {split}; cannot train."
        )

train_counts = coverage[
    coverage["split"].eq("train")
]

if (train_counts["events"] == 0).any():
    raise RuntimeError(
        "At least one relation has zero "
        "training events."
    )

for split, df in all_split_data.items():
    if (
        (df["head"] < 0).any()
        or (df["tail"] < 0).any()
        or (df["head"] >= num_entities).any()
        or (df["tail"] >= num_entities).any()
    ):
        raise RuntimeError(
            f"Out-of-range entity IDs: {split}"
        )

    if not df["relation"].between(0, 3).all():
        raise RuntimeError(
            f"Invalid relation IDs: {split}"
        )

# Preserve the original timestamp count.
# Original DKG reads the first two stat fields.
(OUT / "stat.txt").write_text(
    f"{num_entities}\t4\t{stats[2] if len(stats)>2 else 0}\n"
)

print("\n=== RELATION COVERAGE ===")
print(
    coverage.to_string(index=False)
)

print("\n=== SPLIT COVERAGE ===")

for split, df in all_split_data.items():
    print(
        split,
        "events =", len(df),
        "weeks =", df["time"].nunique(),
        "min_time =", df["time"].min(),
        "max_time =", df["time"].max(),
    )

manifest = {
    "source": str(SOURCE),
    "dataset": "FinDKG-full",
    "entity_count": num_entities,
    "relation_count": 4,
    "relation_names": WANTED,
    "old_to_new_relation_ids": old_to_new,
    "target_ids": target_ids,
    "event_counts": {
        split: len(df)
        for split, df in all_split_data.items()
    },
    "dataset_built": True,
    "model_trained": False,
    "strict_pit_certified": False,
}

(OUT.parent / "manifest.json").write_text(
    json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    )
)

print("\n=== BUILD COMPLETE ===")
print("Dataset:", OUT)
print("Training ready: YES (coverage checks passed)")
print("Four-relation model trained: NO")
print("Strict PIT certified: NO")
