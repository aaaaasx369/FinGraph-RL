import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MASK_RATE = 0.30

SRC = Path(
    "outputs/unified_temporal_kg_v3/sec_signal_nodes.csv"
)

OUT = Path(
    "results/aegis_rl_v1/sec_masking_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)

required = [
    "node", "company", "available_at",
    "impact", "risk", "confidence", "reason"
]

assert all(c in df.columns for c in required)
assert df["node"].is_unique

df["available_at"] = pd.to_datetime(
    df["available_at"], errors="raise"
)

df = df[
    (df.available_at >= "2018-01-01") &
    (df.available_at < "2024-01-01")
].copy()

df["ticker"] = df["company"].str.replace(
    "COMPANY:", "", regex=False
)

# Conservative daily availability:
# a signal dated D is first usable on a later
# business day, never on D itself.
df["decision_date"] = (
    df["available_at"] + pd.offsets.BDay(1)
)

df["year"] = df["decision_date"].dt.year

def split_for(year):
    if year <= 2020:
        return "train"
    if year == 2021:
        return "val"
    if year in (2022, 2023):
        return "test"
    return None

df["split"] = df["year"].map(split_for)
df = df.dropna(subset=["split"]).copy()

# Stable deterministic masking within company-year.
rng = np.random.default_rng(SEED)
df["masked"] = False

for (_, _), idx in df.groupby(
    ["ticker", "year"]
).groups.items():
    idx = np.array(list(idx))
    n = int(round(len(idx) * MASK_RATE))

    if len(idx) > 1:
        n = min(max(1, n), len(idx) - 1)
    else:
        n = 0

    selected = rng.choice(
        idx, size=n, replace=False
    )
    df.loc[selected, "masked"] = True

# Publicly observable KG information.
observed = df[~df.masked].copy()

# Hidden ground truth, kept separate.
hidden = df[df.masked].copy()

# Create detection examples.
# Critical: no hidden signal values, IDs, or
# text are included in detector inputs.
cases = df[
    ["ticker", "decision_date", "split", "masked"]
].copy()

cases["label_missing"] = cases["masked"].astype(int)
cases = cases.drop(columns=["masked"])

# The label is held separately from model inputs.
labels = cases[
    ["ticker", "decision_date",
     "split", "label_missing"]
].copy()

queries = cases.drop(
    columns=["label_missing"]
).copy()

# Duplicate company-date cases may be ambiguous.
# Exclude duplicates from the detector benchmark.
case_counts = queries.groupby(
    ["ticker", "decision_date"]
).size()

duplicates = set(
    case_counts[case_counts > 1].index
)

def unique_case(frame):
    keep = [
        (r.ticker, r.decision_date)
        not in duplicates
        for r in frame.itertuples(index=False)
    ]
    return frame.loc[keep].copy()

queries = unique_case(queries)
labels = unique_case(labels)

# Export:
# observed = visible knowledge
# hidden = held-out truth
# queries = detector input keys
# labels = evaluation only
observed.to_csv(
    OUT / "observed_signals.csv",
    index=False
)

hidden.to_csv(
    OUT / "hidden_signals_ORACLE_ONLY.csv",
    index=False
)

queries.to_csv(
    OUT / "queries.csv",
    index=False
)

labels.to_csv(
    OUT / "labels_ORACLE_ONLY.csv",
    index=False
)

summary = {
    "seed": SEED,
    "mask_rate": MASK_RATE,
    "total": len(df),
    "observed": len(observed),
    "hidden": len(hidden),
    "queries": len(queries),
    "duplicate_company_date_groups": len(duplicates),
    "split": {
        name: int((df["split"] == name).sum())
        for name in ["train", "val", "test"]
    },
    "warning": (
        "Synthetic masking benchmark. "
        "Labels are not independently verified "
        "real-world missing knowledge. "
        "Never expose ORACLE_ONLY files to "
        "detector training inputs or retrieval."
    ),
}

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

print("=" * 60)
print("AEGIS SEC MASKING BENCHMARK")
print("=" * 60)

for k, v in summary.items():
    print(k, ":", v)

print("\nSaved:", OUT.resolve())
print("PASS")
