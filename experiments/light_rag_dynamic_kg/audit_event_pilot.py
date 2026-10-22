#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PILOT = ROOT / (
    "results/light_rag_dynamic_kg/"
    "event_driven_v3/pilot_10_jobs.csv"
)

KG_PATH = ROOT / (
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)

SEC_PATH = ROOT / (
    "results/aegis_rl_v1/v11f_sec_pit/"
    "sec_pit_retrieval.parquet"
)

OUT = ROOT / (
    "results/light_rag_dynamic_kg/"
    "event_driven_v3/pilot_source_audit.csv"
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]


def is_true(series):
    return (
        series.fillna(False)
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1"])
    )


def main():

    pilot = pd.read_csv(PILOT)

    pilot["date"] = pd.to_datetime(
        pilot["date"], errors="raise"
    ).dt.normalize()

    assert len(pilot) == 10
    assert set(pilot["ticker"]) == set(TICKERS)

    kg = pd.read_csv(
        KG_PATH,
        usecols=[
            "event_id",
            "head",
            "relation",
            "tail",
            "available_at",
            "source",
            "time_precision",
        ],
        dtype=str,
        low_memory=False,
    )

    kg["available_at"] = pd.to_datetime(
        kg["available_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None).dt.normalize()

    sec = pd.read_parquet(
        SEC_PATH,
        columns=[
            "ticker",
            "form",
            "current_accession",
            "pit_usable_from",
            "pit_metadata_status",
            "pit_failure_reason",
            "v11d_status",
            "excerpt_content_verified",
        ],
    )

    sec["pit_usable_from"] = pd.to_datetime(
        sec["pit_usable_from"],
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None).dt.normalize()

    sec["ticker"] = sec["ticker"].astype(str).str.upper()

    meta_ok = (
        sec["pit_metadata_status"].astype(str)
        .eq("METADATA_MATCHED")
        & sec["v11d_status"].astype(str)
        .eq("VERIFIED_ACCESSION")
        & sec["pit_failure_reason"].fillna("")
        .astype(str).str.strip().eq("")
    )

    content_ok = is_true(
        sec["excerpt_content_verified"]
    )

    results = []

    for row in pilot.itertuples(index=False):

        ticker = str(row.ticker)
        date = pd.Timestamp(row.date)

        company_key = f"COMPANY:{ticker}"

        focal_kg = kg[
            kg["head"].eq(company_key)
            | kg["tail"].eq(company_key)
        ]

        # Strictly earlier than the pilot trading date.
        asof_kg = focal_kg[
            focal_kg["available_at"] < date
        ]

        sec_company = sec[
            sec["ticker"].eq(ticker)
        ]

        asof_sec = sec_company[
            (sec_company["pit_usable_from"] < date)
            & meta_ok.loc[sec_company.index]
        ]

        verified_sec = asof_sec[
            content_ok.loc[asof_sec.index]
        ]

        # Count accessions, not evidence chunks.
        sec_accessions = (
            asof_sec["current_accession"]
            .dropna().astype(str)
            .nunique()
        )

        audited_accessions = (
            verified_sec["current_accession"]
            .dropna().astype(str)
            .nunique()
        )

        kg_source_counts = (
            asof_kg["source"].value_counts().to_dict()
        )

        results.append({
            "ticker": ticker,
            "date": date.strftime("%Y-%m-%d"),
            "trigger": str(row.trigger),
            "kg_focal_historical_edges": int(len(asof_kg)),
            "kg_unique_relations": int(
                asof_kg["relation"].nunique()
            ),
            "kg_findkg_edges": int(
                kg_source_counts.get("FINDKG", 0)
            ),
            "kg_sec_relation_edges": int(
                kg_source_counts.get("SEC_RELATION", 0)
            ),
            "sec_metadata_filings": int(sec_accessions),
            "sec_content_verified_filings": int(
                audited_accessions
            ),
            "sec_content_verified_excerpts": int(
                len(verified_sec)
            ),
            "has_any_historical_kg": bool(
                len(asof_kg) > 0
            ),
            "historical_pit_certified": False,
            "retrieval_grounding_certified": False,
        })

    result = pd.DataFrame(results)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)

    print("=" * 100)
    print("EVENT-DRIVEN PILOT SOURCE AUDIT")
    print("=" * 100)

    columns = [
        "ticker",
        "date",
        "trigger",
        "kg_focal_historical_edges",
        "kg_unique_relations",
        "sec_metadata_filings",
        "sec_content_verified_filings",
    ]

    print(result[columns].to_string(index=False))

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    print(
        "Pilots with historical KG:",
        int(result["has_any_historical_kg"].sum()),
        "/", len(result)
    )

    print(
        "Total content-verified SEC excerpts:",
        int(result["sec_content_verified_excerpts"].sum())
    )

    print("\nSaved:", OUT)

    print("\nIMPORTANT:")
    print("KG available_at is not independently PIT certified.")
    print("Candidate-source availability does not prove")
    print("the LightRAG retrieval itself is leakage-free.")


if __name__ == "__main__":
    main()
