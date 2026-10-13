#!/usr/bin/env python3
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
OUT = BASE / "research_audit"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["ORIGINAL_KGTRANSFORMER", "ORIGINAL_KG_PLACEBO"]

KEYS = ["tier", "agent", "seed", "split", "condition"]
METRICS = ["total_return", "sharpe", "max_drawdown"]
SLOTS = [
    "dyn_sec_risk_category_count",
    "dyn_fin_relation_count",
    "dyn_fin_produce_count",
    "dyn_fin_operate_in_count",
    "dyn_frkg_triple_count",
    "dyn_frkg_competition",
    "dyn_frkg_dependency",
    "dyn_frkg_negative",
    "dyn_sec_changed",
    "dyn_findkg_changed",
    "dyn_frkg_filing_changed",
]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

tables = {}
diagnostics = []

print("=== A. RAW RUNS AUDIT ===")

for arm in ARMS:
    path = BASE / "runs_multiseed" / arm / "runs.csv"
    if not path.exists():
        raise RuntimeError(f"Missing {path}")

    df = pd.read_csv(path)
    df["_row_number"] = np.arange(len(df))

    for key in KEYS + METRICS:
        if key not in df.columns:
            raise RuntimeError(f"Missing column {key}")

    df["seed"] = pd.to_numeric(df["seed"]).astype(int)

    tables[arm] = df

    duplicate = df.duplicated(KEYS, keep=False)

    grouped = (
        df.groupby(KEYS, dropna=False)[METRICS]
        .agg(["size", "nunique"])
    )

    conflicted_keys = 0

    for _, group in df.groupby(KEYS, dropna=False):
        if len(group) > 1 and any(
            group[c].nunique(dropna=False) > 1
            for c in METRICS
        ):
            conflicted_keys += 1

    print("\nARM:", arm)
    print("CSV rows:", len(df))
    print("Unique experiment keys:", df[KEYS].drop_duplicates().shape[0])
    print("Duplicate rows:", int(duplicate.sum()))
    print("Conflicting keys:", conflicted_keys)
    print("SHA256:", sha(path))

    df.loc[duplicate].to_csv(
        OUT / f"{arm}_raw_duplicates.csv",
        index=False
    )

    diagnostics.append({
        "arm": arm,
        "rows": len(df),
        "unique_keys": df[KEYS].drop_duplicates().shape[0],
        "duplicate_rows": int(duplicate.sum()),
        "conflicting_keys": conflicted_keys,
        "sha256": sha(path),
    })

pd.DataFrame(diagnostics).to_csv(
    OUT / "source_runs_audit.csv", index=False
)

print("\n=== B. INPUT FEATURE COMPARISON ===")

for tier in ["conservative", "inclusive"]:
    a = BASE / "inputs" / ARMS[0] / f"dataset_{tier}.csv"
    b = BASE / "inputs" / ARMS[1] / f"dataset_{tier}.csv"

    if not a.exists() or not b.exists():
        print("MISSING INPUT:", tier)
        continue

    left = pd.read_csv(a, low_memory=False)
    right = pd.read_csv(b, low_memory=False)

    print("\nTier:", tier)
    print("KG SHA256:", sha(a))
    print("Placebo SHA256:", sha(b))

    if len(left) != len(right):
        print("FAIL: Different row counts")
        continue

    missing = [
        c for c in SLOTS
        if c not in left.columns or c not in right.columns
    ]

    if missing:
        print("FAIL: Missing dynamic slots:", missing)
        continue

    x = left[SLOTS].to_numpy(float)
    y = right[SLOTS].to_numpy(float)

    difference = np.abs(x-y)

    print("Rows:", len(left))
    print("Changed dynamic cells:", int(np.count_nonzero(difference > 1e-12)))
    print("Total dynamic cells:", difference.size)
    print("Max absolute difference:", float(np.max(difference)))
    print("Mean absolute difference:", float(np.mean(difference)))

    if np.allclose(x, y, rtol=0, atol=1e-12):
        print("WARNING: KG and Placebo dynamic inputs are identical")
    else:
        print("PASS: KG and Placebo dynamic inputs differ")

print("\n=== C. KG VS PLACEBO RAW RUN DIFFERENCES ===")

left = tables[ARMS[0]]
right = tables[ARMS[1]]

key = KEYS

left = left[left["condition"].eq("FULL_DYNAMIC")]
right = right[right["condition"].eq("FULL_DYNAMIC")]

# Compare first-to-first and last-to-last, without
# concealing conflicts in repeated source runs.
records = []

for policy in ["first", "last"]:
    l = left.drop_duplicates(key, keep=policy)
    r = right.drop_duplicates(key, keep=policy)

    merged = l.merge(
        r,
        on=key,
        suffixes=("_kg", "_placebo"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    for _, row in merged.iterrows():
        record = {
            "policy": policy,
            **{k: row[k] for k in key},
            "merge_status": row["_merge"],
        }

        for metric in METRICS:
            if row["_merge"] == "both":
                record[f"{metric}_difference"] = (
                    row[f"{metric}_kg"]
                    - row[f"{metric}_placebo"]
                )
            else:
                record[f"{metric}_difference"] = np.nan

        records.append(record)

report = pd.DataFrame(records)
report.to_csv(
    OUT / "raw_kg_vs_placebo_diagnostics.csv",
    index=False
)

for policy, group in report.groupby("policy"):
    print("\nSelection policy:", policy)
    print("Compared keys:", int(group["merge_status"].eq("both").sum()))

    for metric in METRICS:
        values = group[
            f"{metric}_difference"
        ].dropna().to_numpy(float)

        print(
            metric,
            "exactly_zero:",
            int(np.count_nonzero(np.isclose(
                values, 0, atol=1e-12, rtol=0
            ))),
            "/",
            len(values),
        )

print("\n=== DIAGNOSTICS COMPLETE ===")
print("Reports:", OUT)
