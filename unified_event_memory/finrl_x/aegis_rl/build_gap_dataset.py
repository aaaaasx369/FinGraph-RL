import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)

SRC = Path(
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)

OUT = Path("results/aegis_rl_v1/gap_dataset")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Load Unified Temporal KG
# ============================================================

df = pd.read_csv(SRC)

df["available_at"] = pd.to_datetime(
    df["available_at"],
    errors="raise"
)

df = df[
    (df["available_at"] >= "2018-01-01") &
    (df["available_at"] < "2024-01-01")
].copy()

df = df.sort_values(
    ["available_at", "event_id"]
).reset_index(drop=True)

print("=" * 65)
print("AEGIS-RL TEMPORAL GAP DATASET")
print("=" * 65)
print("Original historical edges:", len(df))

# ============================================================
# 2. Unique triple first-occurrence dataset
# ============================================================

df = df.drop_duplicates(
    ["head_id", "relation_id", "tail_id"],
    keep="first"
).reset_index(drop=True)

print("Unique historical triples:", len(df))

# ============================================================
# 3. New temporal split
# ============================================================

splits = {
    "train": df[
        df["available_at"] < "2021-01-01"
    ].copy(),

    "val": df[
        (df["available_at"] >= "2021-01-01") &
        (df["available_at"] < "2022-01-01")
    ].copy(),

    "test": df[
        (df["available_at"] >= "2022-01-01") &
        (df["available_at"] < "2024-01-01")
    ].copy(),
}

# ============================================================
# 4. Entity first-seen timestamps
# ============================================================

head_seen = df[
    ["head_id", "available_at"]
].rename(columns={"head_id": "entity_id"})

tail_seen = df[
    ["tail_id", "available_at"]
].rename(columns={"tail_id": "entity_id"})

entity_seen = (
    pd.concat([head_seen, tail_seen])
    .groupby("entity_id")["available_at"]
    .min()
    .sort_values()
)

entity_ids = entity_seen.index.to_numpy(
    dtype=np.int64
)

entity_times = entity_seen.to_numpy()

# All observed historical triples are used only to
# exclude known positives from negative candidates.
# Do not use future entities as negative replacements.
all_known_triples = set(
    zip(
        df["head_id"].astype(int),
        df["relation_id"].astype(int),
        df["tail_id"].astype(int)
    )
)

# ============================================================
# 5. PIT negative sampling
# ============================================================

def sample_negative(row, max_attempts=200):
    t = row.available_at

    # Only entities already observed at this timestamp.
    cutoff = np.searchsorted(
        entity_times,
        np.datetime64(t),
        side="right"
    )

    candidates = entity_ids[:cutoff]

    if len(candidates) < 2:
        return None

    h = int(row.head_id)
    r = int(row.relation_id)
    original_tail = int(row.tail_id)

    for _ in range(max_attempts):
        new_tail = int(
            candidates[rng.integers(len(candidates))]
        )

        if new_tail == original_tail:
            continue

        triple = (h, r, new_tail)

        if triple in all_known_triples:
            continue

        return {
            "head_id": h,
            "relation_id": r,
            "tail_id": new_tail,
            "available_at": t,
            "label": 0
        }

    return None

# ============================================================
# 6. Build dataset
# ============================================================

def build_split(name, frame):
    positives = frame[
        [
            "head_id",
            "relation_id",
            "tail_id",
            "available_at"
        ]
    ].copy()

    negatives = []
    valid_positive_indices = []

    for idx, row in enumerate(
        positives.itertuples(index=False)
    ):
        neg = sample_negative(row)

        if neg is None:
            continue

        negatives.append(neg)
        valid_positive_indices.append(idx)

    positives = positives.iloc[
        valid_positive_indices
    ].copy()

    positives["label"] = 1

    negatives = pd.DataFrame(negatives)

    if len(positives) != len(negatives):
        raise RuntimeError(
            "Positive/negative count mismatch"
        )

    result = pd.concat(
        [positives, negatives],
        ignore_index=True
    )

    result = result.sample(
        frac=1,
        random_state=SEED
    ).reset_index(drop=True)

    result["available_at"] = (
        result["available_at"].dt.strftime("%Y-%m-%d")
    )

    path = OUT / f"{name}.csv"
    result.to_csv(path, index=False)

    print(
        f"{name.upper():5s} | "
        f"Positive: {(result.label == 1).sum():7d} | "
        f"Negative: {(result.label == 0).sum():7d} | "
        f"Total: {len(result):7d}"
    )

    return result

print("\n" + "=" * 65)
print("DATASET SPLITS")
print("=" * 65)

train = build_split("train", splits["train"])
val = build_split("val", splits["val"])
test = build_split("test", splits["test"])

# ============================================================
# 7. Validate years and class balance
# ============================================================

expected = {
    "train": (2018, 2020),
    "val": (2021, 2021),
    "test": (2022, 2023),
}

for name, data in [
    ("train", train),
    ("val", val),
    ("test", test)
]:
    years = pd.to_datetime(
        data["available_at"]
    ).dt.year

    start, end = expected[name]

    assert years.min() >= start
    assert years.max() <= end

    assert (
        (data.label == 1).sum()
        == (data.label == 0).sum()
    )

print("\nAll split checks passed.")
print("Saved to:", OUT.resolve())
