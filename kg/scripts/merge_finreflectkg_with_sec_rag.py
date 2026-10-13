import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# Paths
# =========================================================

SEC_RAG_PATH = Path(
    "rag/data/features/sec_rag_features_business_days.csv"
)

KG_PATH = Path(
    "kg/output/finreflectkg_relation_embeddings.csv"
)

OUTPUT_PATH = Path(
    "kg/output/sec_rag_plus_finreflectkg.csv"
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# Load
# =========================================================

print("Loading SEC-RAG:")
print(SEC_RAG_PATH)

sec = pd.read_csv(SEC_RAG_PATH)

print("SEC-RAG shape:", sec.shape)
print("SEC-RAG columns:")
print(sec.columns.tolist())


print("\nLoading FinReflectKG embedding:")
print(KG_PATH)

kg = pd.read_csv(KG_PATH)

print("KG shape:", kg.shape)


# =========================================================
# Detect column names
# =========================================================

ticker_candidates = [
    "ticker",
    "tic",
    "symbol",
]

date_candidates = [
    "date",
    "datetime",
    "timestamp",
]


ticker_col = None

for c in ticker_candidates:
    if c in sec.columns:
        ticker_col = c
        break


date_col = None

for c in date_candidates:
    if c in sec.columns:
        date_col = c
        break


if ticker_col is None:
    raise ValueError(
        f"Cannot find ticker column. "
        f"Available columns: {sec.columns.tolist()}"
    )


if date_col is None:
    raise ValueError(
        f"Cannot find date column. "
        f"Available columns: {sec.columns.tolist()}"
    )


print("\nDetected:")
print("ticker column =", ticker_col)
print("date column   =", date_col)


# =========================================================
# Normalize
# =========================================================

sec[ticker_col] = (
    sec[ticker_col]
    .astype(str)
    .str.upper()
    .str.strip()
)

kg["ticker"] = (
    kg["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

sec[date_col] = pd.to_datetime(
    sec[date_col],
    errors="coerce"
)

if sec[date_col].isna().any():
    bad = sec[sec[date_col].isna()]

    print("\nWARNING: invalid dates:")
    print(bad.head())


# =========================================================
# Leakage-safe temporal alignment
#
# trading year Y uses KG from year Y-1
#
# Example:
# 2022 validation -> 2021 KG
# 2023 test       -> 2022 KG
# =========================================================

sec["trading_year"] = sec[date_col].dt.year

sec["kg_year"] = (
    sec["trading_year"] - 1
)


# =========================================================
# Prepare KG
# =========================================================

kg = kg.rename(
    columns={
        "year": "kg_year"
    }
)

kg_feature_cols = [
    c for c in kg.columns
    if c.startswith("kg_")
    and c != "kg_year"
]


print("\nNumber of KG features:")
print(len(kg_feature_cols))

print("\nFirst KG features:")
print(kg_feature_cols[:20])


# =========================================================
# Merge
# =========================================================

merged = sec.merge(
    kg,
    how="left",
    left_on=[
        ticker_col,
        "kg_year",
    ],
    right_on=[
        "ticker",
        "kg_year",
    ],
    validate="many_to_one",
)


# Remove duplicate ticker column from KG side
if ticker_col != "ticker":
    if "ticker" in merged.columns:
        merged = merged.drop(
            columns=["ticker"]
        )


# =========================================================
# Availability flag
# =========================================================

if "kg_total_triples" in merged.columns:

    merged["kg_available"] = (
        merged["kg_total_triples"]
        .notna()
        .astype(int)
    )

else:

    merged["kg_available"] = (
        merged[kg_feature_cols]
        .notna()
        .any(axis=1)
        .astype(int)
    )


# =========================================================
# Missing KG
#
# 2018 will normally have no KG because current dataset
# starts at 2018 -> needs 2017 KG.
#
# For first smoke experiment:
# missing KG = zero vector
# + kg_available flag tells the model it was unavailable.
# =========================================================

missing_before = (
    merged[kg_feature_cols]
    .isna()
    .any(axis=1)
    .sum()
)

print(
    "\nRows with unavailable KG before filling:",
    missing_before
)


merged[
    kg_feature_cols
] = merged[
    kg_feature_cols
].fillna(0.0)


# =========================================================
# Diagnostics
# =========================================================

print("\n===== KG AVAILABILITY =====")

print(
    merged.groupby(
        [
            ticker_col,
            "trading_year",
        ]
    )["kg_available"]
    .mean()
)


print("\n===== FINAL DATA =====")

print(
    "Shape:",
    merged.shape
)

print(
    "Date range:",
    merged[date_col].min(),
    "to",
    merged[date_col].max(),
)

print(
    "Tickers:",
    sorted(
        merged[ticker_col]
        .unique()
        .tolist()
    )
)


print(
    "\nTotal missing values "
    "inside KG feature columns:"
)

print(
    merged[
        kg_feature_cols
    ]
    .isna()
    .sum()
    .sum()
)


# =========================================================
# Save
# =========================================================

merged.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nDone.")
