import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

SRC = Path(
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)
OUT = Path("results/aegis_rl_v1/gap_dataset_pit_v2")
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "head_degree",
    "tail_degree",
    "head_relation_degree",
    "tail_relation_degree",
    "pair_degree",
    "relation_count",
    "head_age_days",
    "tail_age_days",
    "head_relation_seen",
    "tail_relation_seen",
]

FIELDS = [
    "head_id", "relation_id", "tail_id",
    "available_at", "label",
] + FEATURES

print("Loading Unified KG...", flush=True)

df = pd.read_csv(
    SRC,
    usecols=[
        "head", "tail", "head_id", "tail_id",
        "relation_id", "available_at", "event_id",
    ],
)

df["available_at"] = pd.to_datetime(
    df["available_at"], errors="raise"
)

df = df[
    (df.available_at >= "2018-01-01") &
    (df.available_at < "2024-01-01")
].copy()

df = df.sort_values(
    ["available_at", "event_id"]
).reset_index(drop=True)

# Entity type derives only from the identifier namespace.
# For example COMPANY:NVDA and SEC_SIGNAL:xxx.
def entity_type(name):
    name = str(name)
    return name.split(":", 1)[0] if ":" in name else "OTHER"

entity_types = {}
for row in df.itertuples(index=False):
    h, t = int(row.head_id), int(row.tail_id)
    ht, tt = entity_type(row.head), entity_type(row.tail)
    if h in entity_types and entity_types[h] != ht:
        raise ValueError("Conflicting entity type: head")
    if t in entity_types and entity_types[t] != tt:
        raise ValueError("Conflicting entity type: tail")
    entity_types[h] = ht
    entity_types[t] = tt

# Only first occurrences are positive prediction targets.
first = df.drop_duplicates(
    ["head_id", "relation_id", "tail_id"],
    keep="first"
).copy()

targets_by_date = {
    date: group
    for date, group in first.groupby("available_at", sort=True)
}

events_by_date = df.groupby("available_at", sort=True)

# Historical state: strictly before the current date.
seen_entities = set()
first_seen = {}
type_entities = defaultdict(list)

known = set()

head_degree = defaultdict(int)
tail_degree = defaultdict(int)
head_rel = defaultdict(int)
tail_rel = defaultdict(int)
pair_degree = defaultdict(int)
relation_count = defaultdict(int)

outputs = {"train": [], "val": [], "test": []}
stats = defaultdict(int)

def split_for(t):
    if t.year <= 2020:
        return "train"
    if t.year == 2021:
        return "val"
    return "test"

def get_features(h, r, t, day):
    h_age = max(0, (day - first_seen[h]).days)
    t_age = max(0, (day - first_seen[t]).days)

    return [
        np.log1p(head_degree[h]),
        np.log1p(tail_degree[t]),
        np.log1p(head_rel[(h, r)]),
        np.log1p(tail_rel[(t, r)]),
        np.log1p(pair_degree[(h, t)]),
        np.log1p(relation_count[r]),
        np.log1p(h_age),
        np.log1p(t_age),
        int(head_rel[(h, r)] > 0),
        int(tail_rel[(t, r)] > 0),
    ]

def make_record(h, r, t, day, label):
    values = get_features(h, r, t, day)
    result = {
        "head_id": h,
        "relation_id": r,
        "tail_id": t,
        "available_at": day.strftime("%Y-%m-%d"),
        "label": label,
    }
    result.update(dict(zip(FEATURES, values)))
    return result

for day, group in events_by_date:

    # All target examples use the graph available BEFORE day.
    targets = targets_by_date.get(day)

    if targets is not None:
        for row in targets.itertuples(index=False):
            h = int(row.head_id)
            r = int(row.relation_id)
            t = int(row.tail_id)

            # Restrict this first benchmark to previously
            # observed endpoints.
            # This avoids trivial new-entity detection.
            if h not in seen_entities or t not in seen_entities:
                stats["skipped_new_entity"] += 1
                continue

            tail_type = entity_types[t]
            candidates = type_entities[tail_type]

            if len(candidates) < 2:
                stats["skipped_small_pool"] += 1
                continue

            # Historical-only negative sampling:
            # do not consult future triple observations.
            neg_tail = None

            for _ in range(100):
                candidate = int(
                    candidates[rng.integers(len(candidates))]
                )

                if candidate == t:
                    continue

                if (h, r, candidate) in known:
                    continue

                neg_tail = candidate
                break

            if neg_tail is None:
                stats["skipped_no_negative"] += 1
                continue

            split = split_for(day)

            outputs[split].append(
                make_record(h, r, t, day, 1)
            )
            outputs[split].append(
                make_record(h, r, neg_tail, day, 0)
            )

            stats["pairs_" + split] += 1

    # Update history AFTER generating this day's examples.
    for row in group.itertuples(index=False):
        h = int(row.head_id)
        r = int(row.relation_id)
        t = int(row.tail_id)

        for entity in (h, t):
            if entity not in seen_entities:
                seen_entities.add(entity)
                first_seen[entity] = day
                typ = entity_types[entity]
                type_entities[typ].append(entity)

        triple = (h, r, t)

        # Structural features represent unique prior triples.
        if triple not in known:
            known.add(triple)
            head_degree[h] += 1
            tail_degree[t] += 1
            head_rel[(h, r)] += 1
            tail_rel[(t, r)] += 1
            pair_degree[(h, t)] += 1
            relation_count[r] += 1

print("\nSaving dataset...", flush=True)

summary = {}

for split, records in outputs.items():

    if not records:
        raise RuntimeError(f"Empty split: {split}")

    rng.shuffle(records)

    path = OUT / f"{split}.csv"

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    labels = [x["label"] for x in records]
    positives = labels.count(1)
    negatives = labels.count(0)

    assert positives == negatives

    dates = pd.to_datetime(
        [x["available_at"] for x in records]
    )

    expected = {
        "train": (2018, 2020),
        "val": (2021, 2021),
        "test": (2022, 2023),
    }[split]

    assert dates.year.min() >= expected[0]
    assert dates.year.max() <= expected[1]

    summary[split] = {
        "positive": positives,
        "negative": negatives,
        "total": len(records),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
    }

    print(
        f"{split.upper():5s} | "
        f"positive={positives:,} "
        f"negative={negatives:,} "
        f"total={len(records):,}"
    )

metadata = {
    "seed": SEED,
    "features": FEATURES,
    "splits": summary,
    "sampling": (
        "Historical-only, type-matched tail corruption; "
        "endpoints must pre-exist the candidate day"
    ),
    "limitations": [
        "Unobserved triples are not proven false",
        "Entity type matching does not guarantee semantic validity",
        "Relations are predicted from historical graph features",
        "Dataset is not a portfolio backtest",
        "Availability dates are assumed to be conservative PIT dates",
    ],
    "statistics": dict(stats),
}

with (OUT / "metadata.json").open("w") as f:
    json.dump(metadata, f, indent=2)

print("\nSkipped new entities:",
      stats["skipped_new_entity"])
print("Skipped small pool:",
      stats["skipped_small_pool"])
print("Skipped no negative:",
      stats["skipped_no_negative"])

print("\nAll checks passed.")
print("Saved:", OUT.resolve())
