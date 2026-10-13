from pathlib import Path
import pandas as pd
import numpy as np

OLD = Path("rag/data/features/sec_rag_events.csv")
NEW = Path("rag/data/features/sec_rag_events_2024_2026.csv")
OUT = Path("rag/data/features/sec_rag_events_2018_2026.csv")

old = pd.read_csv(OLD)
new = pd.read_csv(NEW)

print("OLD columns:")
print(old.columns.tolist())
print()
print("NEW columns:")
print(new.columns.tolist())
print()

# ------------------------------------------------------------
# Normalize ticker
# ------------------------------------------------------------

if "ticker" not in old.columns and "tic" in old.columns:
    old = old.rename(columns={"tic": "ticker"})

if "ticker" not in new.columns and "tic" in new.columns:
    new = new.rename(columns={"tic": "ticker"})

# ------------------------------------------------------------
# Detect old date column
# ------------------------------------------------------------

date_candidates = [
    "available_date",
    "filing_date",
    "published_at",
    "date",
    "event_date",
    "signal_date",
    "filingDate",
    "published_date",
    "report_date",
]

old_date_col = None

for c in date_candidates:
    if c in old.columns:
        old_date_col = c
        break

if old_date_col is None:
    # Fall back: try any column containing "date" or "time"
    candidates = [
        c for c in old.columns
        if "date" in c.lower() or "time" in c.lower()
    ]

    print("Possible old date columns:", candidates)

    for c in candidates:
        parsed = pd.to_datetime(
            old[c],
            errors="coerce"
        )

        if parsed.notna().mean() > 0.8:
            old_date_col = c
            break

if old_date_col is None:
    raise ValueError(
        "Could not identify date column in old sec_rag_events.csv"
    )

print("Detected OLD date column:", old_date_col)

if old_date_col != "available_date":
    old = old.rename(
        columns={
            old_date_col: "available_date"
        }
    )

# ------------------------------------------------------------
# Normalize scores
# ------------------------------------------------------------

rename_map = {}

if "impact_score" in old.columns and "impact" not in old.columns:
    rename_map["impact_score"] = "impact"

if "risk_score" in old.columns and "risk" not in old.columns:
    rename_map["risk_score"] = "risk"

if "confidence_score" in old.columns and "confidence" not in old.columns:
    rename_map["confidence_score"] = "confidence"

old = old.rename(columns=rename_map)

# ------------------------------------------------------------
# Parse dates
# ------------------------------------------------------------

old["available_date"] = pd.to_datetime(
    old["available_date"],
    errors="coerce"
)

new["available_date"] = pd.to_datetime(
    new["available_date"],
    errors="coerce"
)

old = old.dropna(
    subset=[
        "ticker",
        "available_date"
    ]
)

new = new.dropna(
    subset=[
        "ticker",
        "available_date"
    ]
)

old["ticker"] = (
    old["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

new["ticker"] = (
    new["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

# ------------------------------------------------------------
# Keep old baseline only before 2024
# ------------------------------------------------------------

old = old[
    old["available_date"] < pd.Timestamp("2024-01-01")
].copy()

new = new[
    new["available_date"] >= pd.Timestamp("2024-01-01")
].copy()

print()
print("OLD rows before 2024:", len(old))
print("NEW rows 2024+:", len(new))

# ------------------------------------------------------------
# Align columns
# ------------------------------------------------------------

all_cols = sorted(
    set(old.columns) | set(new.columns)
)

for c in all_cols:
    if c not in old.columns:
        old[c] = np.nan

    if c not in new.columns:
        new[c] = np.nan

merged = pd.concat(
    [
        old[all_cols],
        new[all_cols]
    ],
    ignore_index=True
)

merged = (
    merged
    .sort_values(
        [
            "available_date",
            "ticker"
        ]
    )
    .drop_duplicates(
        subset=[
            "ticker",
            "available_date"
        ],
        keep="last"
    )
    .reset_index(drop=True)
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

merged.to_csv(
    OUT,
    index=False
)

print()
print("=" * 70)
print("MERGE COMPLETE")
print("=" * 70)

print("rows:", len(merged))
print(
    "range:",
    merged["available_date"].min(),
    "->",
    merged["available_date"].max()
)

print()
print(
    merged.groupby("ticker")["available_date"]
    .agg(["min", "max", "count"])
)

print()
print("Saved ->", OUT)
