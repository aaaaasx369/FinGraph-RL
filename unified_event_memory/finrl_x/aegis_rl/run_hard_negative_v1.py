import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
)
from joblib import dump

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path("results/aegis_rl_v1")
EASY = ROOT / "gap_dataset_pit_v2"
HARD = ROOT / "gap_dataset_hard_v1"
OUT = ROOT / "hard_negative_benchmark_v1"

HARD.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

SRC = Path(
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)

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

print("=" * 70)
print("AEGIS-RL HARD NEGATIVE BENCHMARK")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load original datasets
# ------------------------------------------------------------

easy = {}

for split in ["train", "val", "test"]:
    easy[split] = pd.read_csv(
        EASY / f"{split}.csv"
    )

    easy[split]["available_at"] = pd.to_datetime(
        easy[split]["available_at"]
    )

    print(
        split,
        "easy rows:",
        len(easy[split])
    )

# Use exactly the same positive examples.
positive_by_date = defaultdict(list)

for split, frame in easy.items():
    positive = frame[frame.label == 1]

    for row in positive.itertuples(index=False):
        positive_by_date[row.available_at].append(
            (split, row)
        )

# ------------------------------------------------------------
# 2. Load Unified Temporal KG
# ------------------------------------------------------------

kg = pd.read_csv(
    SRC,
    usecols=[
        "head", "tail",
        "head_id", "tail_id",
        "relation_id",
        "available_at",
        "event_id",
    ],
)

kg["available_at"] = pd.to_datetime(
    kg["available_at"]
)

kg = kg[
    (kg.available_at >= "2018-01-01") &
    (kg.available_at < "2024-01-01")
].copy()

kg = kg.sort_values(
    ["available_at", "event_id"]
)

# Entity namespace type:
# COMPANY, FINDKG, SEC_SIGNAL, etc.
entity_types = {}

def entity_type(name):
    name = str(name)
    return name.split(":", 1)[0] if ":" in name else "OTHER"

for row in kg.itertuples(index=False):
    entity_types[int(row.head_id)] = entity_type(row.head)
    entity_types[int(row.tail_id)] = entity_type(row.tail)

# ------------------------------------------------------------
# 3. Historical graph state
# ------------------------------------------------------------

known = set()
first_seen = {}

type_tails = defaultdict(list)
registered_tails = set()

head_degree = defaultdict(int)
tail_degree = defaultdict(int)
head_rel = defaultdict(int)
tail_rel = defaultdict(int)
pair_degree = defaultdict(int)
relation_count = defaultdict(int)

hard_rows = {
    "train": [],
    "val": [],
    "test": [],
}

stats = defaultdict(int)

def structural_features(h, r, t, day):
    return {
        "head_degree": np.log1p(head_degree[h]),
        "tail_degree": np.log1p(tail_degree[t]),
        "head_relation_degree": np.log1p(head_rel[(h, r)]),
        "tail_relation_degree": np.log1p(tail_rel[(t, r)]),
        "pair_degree": np.log1p(pair_degree[(h, t)]),
        "relation_count": np.log1p(relation_count[r]),
        "head_age_days": np.log1p(
            max(0, (day - first_seen[h]).days)
        ),
        "tail_age_days": np.log1p(
            max(0, (day - first_seen[t]).days)
        ),
        "head_relation_seen": int(head_rel[(h, r)] > 0),
        "tail_relation_seen": int(tail_rel[(t, r)] > 0),
    }

def make_record(h, r, t, day, label):
    result = {
        "head_id": h,
        "relation_id": r,
        "tail_id": t,
        "available_at": day.strftime("%Y-%m-%d"),
        "label": label,
    }
    result.update(structural_features(h, r, t, day))
    return result

def choose_hard_tail(h, r, true_tail):
    """
    Same historical tail type.
    Prefer similar historical degree and relation activity.

    Crucially, do not inspect future graph edges.
    """
    typ = entity_types[true_tail]
    pool = type_tails[typ]

    if not pool:
        return None

    true_degree = np.log1p(tail_degree[true_tail])
    true_relation = np.log1p(
        tail_rel[(true_tail, r)]
    )

    # Deterministic seeded candidate sampling.
    count = min(128, len(pool))

    if count == len(pool):
        candidates = pool
    else:
        indices = rng.choice(
            len(pool), size=count, replace=False
        )
        candidates = [pool[i] for i in indices]

    best = None
    best_score = float("inf")

    for candidate in candidates:
        t = int(candidate)

        if t == true_tail:
            continue

        if (h, r, t) in known:
            continue

        degree_diff = abs(
            np.log1p(tail_degree[t]) - true_degree
        )

        relation_diff = abs(
            np.log1p(tail_rel[(t, r)]) -
            true_relation
        )

        # Prefer candidates with similar history.
        score = (
            degree_diff +
            1.5 * relation_diff
        )

        if score < best_score:
            best_score = score
            best = t

    return best

# ------------------------------------------------------------
# 4. Generate hard negatives using strictly prior history
# ------------------------------------------------------------

for day, group in kg.groupby("available_at", sort=True):

    targets = positive_by_date.get(day, [])

    for split, row in targets:

        h = int(row.head_id)
        r = int(row.relation_id)
        t = int(row.tail_id)

        if h not in first_seen or t not in first_seen:
            stats["missing_prior_entity"] += 1
            continue

        neg_tail = choose_hard_tail(h, r, t)

        if neg_tail is None:
            stats["no_candidate"] += 1
            continue

        positive_record = {
            col: getattr(row, col)
            for col in FEATURES
        }

        positive_record.update({
            "head_id": h,
            "relation_id": r,
            "tail_id": t,
            "available_at": day.strftime("%Y-%m-%d"),
            "label": 1,
        })

        negative_record = make_record(
            h, r, neg_tail, day, 0
        )

        hard_rows[split].extend([
            positive_record,
            negative_record,
        ])

        stats[f"{split}_pairs"] += 1

    # Update graph AFTER processing all decisions
    # on this date.
    for row in group.itertuples(index=False):

        h = int(row.head_id)
        r = int(row.relation_id)
        t = int(row.tail_id)

        for entity in [h, t]:
            if entity not in first_seen:
                first_seen[entity] = day

        # Register tails observed up to current day.
        if t not in registered_tails:
            registered_tails.add(t)
            type_tails[entity_types[t]].append(t)

        triple = (h, r, t)

        if triple in known:
            continue

        known.add(triple)

        head_degree[h] += 1
        tail_degree[t] += 1
        head_rel[(h, r)] += 1
        tail_rel[(t, r)] += 1
        pair_degree[(h, t)] += 1
        relation_count[r] += 1

print("\n===== HARD DATASET =====")

for split, rows in hard_rows.items():

    if not rows:
        raise RuntimeError(
            f"No hard-negative examples: {split}"
        )

    rng.shuffle(rows)

    df = pd.DataFrame(rows)

    # Ensure consistent column ordering.
    columns = [
        "head_id", "relation_id",
        "tail_id", "available_at", "label",
    ] + FEATURES

    df = df[columns]

    assert int((df.label == 1).sum()) == int(
        (df.label == 0).sum()
    )

    df.to_csv(
        HARD / f"{split}.csv",
        index=False
    )

    print(
        f"{split.upper():5s}: "
        f"positive={(df.label == 1).sum():,} "
        f"negative={(df.label == 0).sum():,}"
    )

print("Sampling stats:", dict(stats))

# ------------------------------------------------------------
# 5. Compare easy vs hard training
# ------------------------------------------------------------

def load_dataset(folder, split):
    frame = pd.read_csv(
        folder / f"{split}.csv"
    )
    X = frame[FEATURES].to_numpy(
        dtype=np.float32
    )
    y = frame.label.to_numpy(
        dtype=np.int32
    )
    return X, y

def metrics(y, probability):
    pred = (probability >= 0.5).astype(int)
    return {
        "AUC": roc_auc_score(y, probability),
        "AP": average_precision_score(y, probability),
        "F1": f1_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
    }

results = []
models = {}

for train_type, train_folder in [
    ("Easy", EASY),
    ("Hard", HARD),
]:
    print("\nTraining model:", train_type, flush=True)

    X_train, y_train = load_dataset(
        train_folder, "train"
    )

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=160,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=SEED,
    )

    model.fit(X_train, y_train)

    models[train_type] = model

    dump(
        model,
        OUT / f"{train_type.lower()}_model.joblib"
    )

    for test_type, test_folder in [
        ("Easy", EASY),
        ("Hard", HARD),
    ]:
        for split in ["val", "test"]:
            X, y = load_dataset(
                test_folder, split
            )

            probability = model.predict_proba(X)[:, 1]

            score = metrics(y, probability)

            results.append({
                "training": train_type,
                "evaluation": test_type,
                "split": split,
                **score,
            })

            print(
                f"Train={train_type:4s} "
                f"Eval={test_type:4s} "
                f"Split={split:4s} "
                f"AUC={score['AUC']:.6f} "
                f"AP={score['AP']:.6f} "
                f"F1={score['F1']:.6f}"
            )

result = pd.DataFrame(results)

result.to_csv(
    OUT / "benchmark.csv",
    index=False
)

with (OUT / "sampling_stats.json").open("w") as f:
    json.dump(dict(stats), f, indent=2)

print("\n" + "=" * 70)
print("TEST AUC COMPARISON")
print("=" * 70)

test_result = result[result.split == "test"]

print(
    test_result.pivot(
        index="training",
        columns="evaluation",
        values="AUC"
    ).round(6).to_string()
)

print("\nSaved to:", OUT.resolve())
print("Benchmark complete.")
