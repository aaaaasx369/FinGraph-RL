#!/usr/bin/env python3
"""
V16: Filing-date-aligned FinReflectKG feature store.

Uses:
  kg/data/finreflectkg_5stocks_2018_2023.parquet
  kg/output/finreflectkg_dynamic_filing_map.csv
  kg/data/market_5stocks_2017_2026.csv

Rules:
- Full key mapping: ticker + source_file + year
- available_at < market date (one-day conservative lag)
- Never use triple start_date/end_date for availability
- Exclude unverified extraction types from conservative tier
- Missing source coverage stays missing, not zero
- Never claim externally verified public availability
- No training or return prediction
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
import numpy as np

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

RELATIONS = {
    "discloses": "disclosure",
    "depends_on": "dependency",
    "negatively_impacts": "negative",
    "subject_to": "exposure",
    "impacted_by": "impact",
    "competes_with": "competition",
    "partners_with": "partnership",
    "invests_in": "investment",
    "operates_in": "operations",
    "produce": "production",
    "supply": "supply",
}

# 'default' often represents fallback extraction; do not
# conflate it with high-confidence extracted content.
TIERS = {
    "conservative": {"extracted"},
    "inclusive": {"extracted", "default"},
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--kg",
        default="kg/data/finreflectkg_5stocks_2018_2023.parquet"
    )
    p.add_argument(
        "--filing-map",
        default="kg/output/finreflectkg_dynamic_filing_map.csv"
    )
    p.add_argument(
        "--market",
        default="kg/data/market_5stocks_2017_2026.csv"
    )
    p.add_argument("--tier", choices=TIERS,
                   default="conservative")
    p.add_argument("--out-dir",
                   default="outputs/v16/filing_aligned")
    args = p.parse_args()

    kg = pd.read_parquet(args.kg)
    mapping = pd.read_csv(args.filing_map)
    market = pd.read_csv(args.market)

    required_kg = {
        "ticker", "source_file", "year",
        "triplet_id", "relationship",
        "extraction_type",
    }
    required_map = {
        "source_ticker", "source_year",
        "source_file", "available_at",
    }
    required_market = {"date", "ticker"}

    for name, df, needed in (
        ("KG", kg, required_kg),
        ("MAP", mapping, required_map),
        ("MARKET", market, required_market),
    ):
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    kg["ticker"] = kg["ticker"].astype(str).str.upper()
    kg["year"] = pd.to_numeric(
        kg["year"], errors="raise"
    ).astype(int)
    kg["source_file"] = kg["source_file"].astype(str).str.strip()

    mapping["source_ticker"] = (
        mapping["source_ticker"].astype(str).str.upper()
    )
    mapping["source_year"] = pd.to_numeric(
        mapping["source_year"], errors="raise"
    ).astype(int)
    mapping["source_file"] = (
        mapping["source_file"].astype(str).str.strip()
    )
    mapping["available_at"] = pd.to_datetime(
        mapping["available_at"], errors="raise"
    ).dt.normalize()

    if mapping.duplicated(
        ["source_ticker", "source_year", "source_file"]
    ).any():
        raise ValueError("Duplicate filing map key")

    if kg["triplet_id"].isna().any():
        raise ValueError("Missing triplet IDs")

    merged = kg.merge(
        mapping,
        left_on=["ticker", "year", "source_file"],
        right_on=[
            "source_ticker", "source_year", "source_file"
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    missing = merged["_merge"].ne("both")
    if missing.any():
        raise ValueError(
            f"Unmapped KG rows: {int(missing.sum())}"
        )
    if merged["available_at"].isna().any():
        raise ValueError("Missing available_at")

    # Do not infer point-in-time availability from KG
    # start_date or end_date.
    merged["extraction_type"] = (
        merged["extraction_type"]
        .astype(str).str.lower().str.strip()
    )
    merged["relationship"] = (
        merged["relationship"]
        .astype(str).str.lower().str.strip()
    )

    excluded = merged.loc[
        ~merged["extraction_type"].isin(TIERS[args.tier])
    ].copy()

    selected = merged.loc[
        merged["extraction_type"].isin(TIERS[args.tier])
    ].copy()

    selected["relation_group"] = selected[
        "relationship"
    ].map(RELATIONS).fillna("other")

    # Relationship counts are observations, not confirmed
    # current relationships or realized financial events.
    filings = (
        selected.groupby(
            ["ticker", "source_file", "available_at"],
            as_index=False
        )
        .agg(
            triple_count=("triplet_id", "size"),
            unique_relation_count=(
                "relationship", "nunique"
            ),
            unique_target_count=("target", "nunique"),
        )
    )

    by_group = (
        selected.groupby(
            ["ticker", "source_file", "available_at",
             "relation_group"]
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    filings = filings.merge(
        by_group,
        on=["ticker", "source_file", "available_at"],
        validate="one_to_one",
    )

    feature_cols = [
        "triple_count", "unique_relation_count",
        "unique_target_count"
    ] + sorted(set(RELATIONS.values()) | {"other"})

    for c in feature_cols:
        if c not in filings:
            filings[c] = 0

    filing_features = filings[
        ["ticker", "source_file", "available_at"]
        + feature_cols
    ].copy()

    filing_features = filing_features.rename(
        columns={
            c: "frkg_" + c for c in feature_cols
        }
    )

    market = market.loc[
        market["ticker"].isin(TICKERS)
    ].copy()

    market["ticker"] = market["ticker"].str.upper()
    market["date"] = pd.to_datetime(
        market["date"], errors="raise"
    ).dt.normalize()

    if market.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate market keys")

    # A source is eligible only on later market dates.
    filing_features["usable_from"] = (
        filing_features["available_at"]
        + pd.Timedelta(days=1)
    )

    # Earliest filing visible is retained until the next
    # available filing; no future-backfill.
    ordered_market = market.sort_values(
        ["date", "ticker"]
    )
    ordered_filings = filing_features.sort_values(
        ["usable_from", "ticker"]
    )

    all_frames = []

    for ticker in TICKERS:
        mk = ordered_market.loc[
            ordered_market["ticker"] == ticker
        ].copy().sort_values("date")

        fl = ordered_filings.loc[
            ordered_filings["ticker"] == ticker
        ].copy().sort_values("usable_from")

        if fl.empty:
            for c in feature_cols:
                mk["frkg_" + c] = np.nan
            mk["source_file"] = None
            mk["available_at"] = pd.NaT
            mk["usable_from"] = pd.NaT
            all_frames.append(mk)
            continue

        aligned = pd.merge_asof(
            mk,
            fl.drop(columns=["ticker"]),
            left_on="date",
            right_on="usable_from",
            direction="backward",
            allow_exact_matches=True,
        )
        all_frames.append(aligned)

    output = pd.concat(
        all_frames, ignore_index=True
    ).sort_values(["date", "ticker"])

    output["frkg_observed"] = (
        output["source_file"].notna().astype(int)
    )
    output["frkg_external_availability_verified"] = 0
    output["frkg_formal_finrl_approved"] = 0

    # Hard guard: no filing may be introduced on its own
    # available_at day or earlier.
    valid = output["available_at"].notna()
    if not (
        output.loc[valid, "available_at"]
        < output.loc[valid, "date"]
    ).all():
        raise AssertionError("Future/same-day filing leakage")

    if not (
        output.loc[valid, "usable_from"]
        <= output.loc[valid, "date"]
    ).all():
        raise AssertionError("Invalid usable_from alignment")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    filing_features.to_csv(
        out / "filing_features.csv", index=False
    )
    excluded[
        ["ticker", "source_file", "triplet_id",
         "relationship", "extraction_type"]
    ].to_csv(
        out / "excluded_triples.csv", index=False
    )
    output.to_csv(
        out / "daily_filing_aligned.csv",
        index=False
    )

    report = {
        "tier": args.tier,
        "input_triples": int(len(kg)),
        "selected_triples": int(len(selected)),
        "excluded_triples": int(len(excluded)),
        "mapped_documents": int(
            merged["source_file"].nunique()
        ),
        "filings_with_selected_triples": int(
            len(filing_features)
        ),
        "market_rows": int(len(market)),
        "aligned_rows": int(len(output)),
        "observed_rows": int(
            output["frkg_observed"].sum()
        ),
        "no_snapshot_rows": int(
            (output["frkg_observed"] == 0).sum()
        ),
        "external_availability_verified": False,
        "formal_finrl_approved": False,
        "limitations": [
            "available_at has not been independently "
            "verified against EDGAR acceptance timestamps",
            "Triple semantics not fact-checked",
            "Latest eligible filing is carried forward",
            "Feature magnitudes not normalized",
            "FinRL training not performed",
        ],
    }

    (out / "report.json").write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))
    print("Saved:", out)

if __name__ == "__main__":
    main()
