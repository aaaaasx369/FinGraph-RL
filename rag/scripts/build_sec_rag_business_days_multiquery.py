from pathlib import Path

import numpy as np
import pandas as pd


BACKBONE = Path(
    "rag/data/features/"
    "sec_rag_features_business_days_2018_2026.csv"
)

EVENTS = Path(
    "rag/data/features/"
    "sec_rag_events_2018_2026_multiquery.csv"
)

OUTPUT = Path(
    "rag/data/features/"
    "sec_rag_features_business_days_multiquery_2018_2026.csv"
)


# ============================================================
# Load
# ============================================================

backbone = pd.read_csv(BACKBONE)
events = pd.read_csv(EVENTS)


# ============================================================
# Normalize columns
# ============================================================

if "ticker" not in backbone.columns and "tic" in backbone.columns:
    backbone = backbone.rename(
        columns={"tic": "ticker"}
    )

if "date" not in backbone.columns:
    raise ValueError(
        f"Backbone missing date column. "
        f"Columns: {backbone.columns.tolist()}"
    )

required_event_cols = [
    "ticker",
    "available_date",
    "impact",
    "risk",
    "confidence",
]

missing = [
    c
    for c in required_event_cols
    if c not in events.columns
]

if missing:
    raise ValueError(
        f"Missing event columns: {missing}"
    )


# ============================================================
# Parse
# ============================================================

backbone["date"] = pd.to_datetime(
    backbone["date"],
    errors="coerce",
)

events["available_date"] = pd.to_datetime(
    events["available_date"],
    errors="coerce",
)

backbone["ticker"] = (
    backbone["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

events["ticker"] = (
    events["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)


# ============================================================
# Clean old scaffold SEC columns
# ============================================================

drop_cols = [
    "impact",
    "risk",
    "confidence",
    "signal_available_date",
    "latest_sec_age_days",
    "sec_chunks_today",
    "sec_docs_today",
]

backbone = backbone.drop(
    columns=[
        c
        for c in drop_cols
        if c in backbone.columns
    ],
    errors="ignore",
)


# ============================================================
# Keep only columns needed from SEC events
# ============================================================

event_keep = [
    "ticker",
    "available_date",
    "impact",
    "risk",
    "confidence",
    "filing_type",
    "pipeline_version",
    "llm_model",
]

event_keep = [
    c
    for c in event_keep
    if c in events.columns
]

events_small = events[
    event_keep
].copy()


# ============================================================
# AS-OF merge separately per ticker
#
# Critical rule:
#
# available_date <= trading date
#
# direction="backward"
#
# No future information.
# ============================================================

pieces = []

for ticker in sorted(
    backbone["ticker"].unique()
):

    daily = (
        backbone[
            backbone["ticker"] == ticker
        ]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    ev = (
        events_small[
            events_small["ticker"] == ticker
        ]
        .copy()
        .sort_values("available_date")
        .reset_index(drop=True)
    )

    if len(ev) == 0:

        daily["signal_available_date"] = pd.NaT
        daily["impact"] = 0.0
        daily["risk"] = 0.0
        daily["confidence"] = 0.0

        pieces.append(daily)
        continue

    ev = ev.drop(
        columns=["ticker"]
    )

    merged = pd.merge_asof(
        daily,
        ev,
        left_on="date",
        right_on="available_date",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.rename(
        columns={
            "available_date":
                "signal_available_date"
        }
    )

    # Before the first SEC signal:
    # no future-fill; use neutral zero features.
    merged["impact"] = (
        merged["impact"]
        .fillna(0.0)
        .astype(float)
    )

    merged["risk"] = (
        merged["risk"]
        .fillna(0.0)
        .astype(float)
    )

    merged["confidence"] = (
        merged["confidence"]
        .fillna(0.0)
        .astype(float)
    )

    pieces.append(merged)


# ============================================================
# Combine
# ============================================================

out = pd.concat(
    pieces,
    ignore_index=True,
)

out = out.sort_values(
    ["date", "ticker"]
).reset_index(drop=True)


# ============================================================
# Leakage checks
# ============================================================

valid_signal = out[
    out["signal_available_date"].notna()
].copy()

leakage = valid_signal[
    valid_signal["signal_available_date"]
    >
    valid_signal["date"]
]

if len(leakage) > 0:

    print(
        leakage[
            [
                "ticker",
                "date",
                "signal_available_date",
            ]
        ].head(20)
    )

    raise AssertionError(
        f"Temporal leakage detected: "
        f"{len(leakage)} rows"
    )


# ============================================================
# Split labels
# ============================================================

def split_name(date):

    year = date.year

    if 2018 <= year <= 2023:
        return "train"

    if year == 2024:
        return "val"

    if year == 2025:
        return "test"

    if year == 2026:
        return "holdout"

    return "other"


out["split"] = out["date"].apply(
    split_name
)


# ============================================================
# SEC signal age
# ============================================================

out["sec_signal_age_days"] = (
    out["date"]
    -
    out["signal_available_date"]
).dt.days


# ============================================================
# Save
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# QA
# ============================================================

print("=" * 70)
print("SEC-RAG BUSINESS-DAY AS-OF FEATURES")
print("=" * 70)

print(
    "ROWS:",
    len(out),
)

print(
    "DATE RANGE:",
    out["date"].min(),
    "->",
    out["date"].max(),
)

print()

print("SPLITS")
print(
    out.groupby("split").size()
)

print()

print("BY TICKER")
print(
    out.groupby("ticker").size()
)

print()

print("MISSING SCORES")
print(
    out[
        [
            "impact",
            "risk",
            "confidence",
        ]
    ]
    .isna()
    .sum()
)

print()

print("SCORE RANGE")
print(
    out[
        [
            "impact",
            "risk",
            "confidence",
        ]
    ].agg(
        ["min", "max"]
    )
)

print()

print(
    "ROWS WITH A REAL SEC SIGNAL:",
    out[
        "signal_available_date"
    ]
    .notna()
    .sum(),
)

print(
    "LEAKAGE ROWS:",
    len(leakage),
)

print()

print("FIRST/LAST SIGNAL BY TICKER")

print(
    valid_signal.groupby(
        "ticker"
    )[
        "signal_available_date"
    ]
    .agg(
        ["min", "max"]
    )
)

print()

print(
    "Saved ->",
    OUTPUT,
)
