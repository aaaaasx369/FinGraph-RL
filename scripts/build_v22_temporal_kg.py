#!/usr/bin/env python3
"""
V22-A Temporal Financial Knowledge Graph.

Inputs:
 - FinReflectKG triple parquet
 - FinReflectKG filing map
 - SEC EDGAR manifest

Outputs:
 - filing_link_audit.csv
 - relation_observations.parquet
 - relation_events.parquet
 - daily_temporal_features.csv
 - report.json

Rules:
 - Never use extracted start_date/end_date as public availability.
 - A date match is CANDIDATE, not verified document identity.
 - Require filing availability strictly earlier than trade date.
 - No automatic deletion when a relation disappears.
 - Keep unverified FinDKG separate.
 - No future backfill.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]


def normalized(value):
    if pd.isna(value):
        return ""
    return " ".join(str(value).lower().split())


def digest(*parts):
    text = "\x1f".join(str(p) for p in parts)
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:24]


def load_sources(args):
    triples = pd.read_parquet(args.triples)
    mapping = pd.read_csv(args.mapping)
    manifest = pd.read_csv(args.manifest)

    needed_triples = [
        "ticker", "source_file",
        "entity", "relationship", "target",
        "extraction_type", "triplet_id",
    ]
    needed_mapping = [
        "source_ticker", "source_file",
        "available_at",
    ]
    needed_manifest = [
        "ticker", "form", "accession",
        "filing_date", "acceptance_utc",
        "text_verified", "source_sha256",
        "document_url",
    ]

    for label, frame, cols in [
        ("triples", triples, needed_triples),
        ("mapping", mapping, needed_mapping),
        ("manifest", manifest, needed_manifest),
    ]:
        missing = [c for c in cols if c not in frame]
        if missing:
            raise ValueError(
                f"{label} missing columns: {missing}"
            )

    return triples, mapping, manifest


def make_links(mapping, manifest):
    mapping = mapping.copy()
    manifest = manifest.copy()

    mapping["source_ticker"] = (
        mapping["source_ticker"]
        .astype(str).str.upper()
    )
    manifest["ticker"] = (
        manifest["ticker"].astype(str).str.upper()
    )

    mapping["available_at"] = pd.to_datetime(
        mapping["available_at"], errors="coerce"
    ).dt.normalize()

    manifest["filing_date"] = pd.to_datetime(
        manifest["filing_date"], errors="coerce"
    ).dt.normalize()

    manifest["acceptance_utc"] = pd.to_datetime(
        manifest["acceptance_utc"],
        errors="coerce",
        utc=True,
    )

    manifest = manifest[
        manifest["form"].eq("10-K")
    ].copy()

    if mapping.duplicated(
        ["source_ticker", "source_file"]
    ).any():
        raise ValueError(
            "Duplicate FinReflectKG source mapping"
        )

    results = []

    for _, row in mapping.iterrows():
        ticker = row["source_ticker"]
        mapped = row["available_at"]

        # Exact filing-date candidate only.
        candidates = manifest[
            manifest["ticker"].eq(ticker)
            & manifest["filing_date"].eq(mapped)
            & manifest["acceptance_utc"].notna()
        ]

        result = {
            "ticker": ticker,
            "source_file": row["source_file"],
            "mapping_available_at": mapped,
            "candidate_count": len(candidates),
            "accession": "",
            "acceptance_utc": pd.NaT,
            "filing_date": pd.NaT,
            "document_url": "",
            "source_sha256": "",
            "link_status": "UNMATCHED",
            "pdf_identity_verified": False,
        }

        if len(candidates) == 1:
            candidate = candidates.iloc[0]

            result.update({
                "accession": candidate["accession"],
                "acceptance_utc": candidate["acceptance_utc"],
                "filing_date": candidate["filing_date"],
                "document_url": candidate["document_url"],
                "source_sha256": candidate["source_sha256"],
                "link_status": "CANDIDATE_MATCH",
            })

        elif len(candidates) > 1:
            result["link_status"] = "AMBIGUOUS"

        results.append(result)

    links = pd.DataFrame(results)

    # A 10-K accepted on date d is only allowed
    # from the following calendar date onward.
    links["safe_date"] = pd.to_datetime(
        links["filing_date"], errors="coerce"
    ) + pd.Timedelta(days=1)

    return links


def prepare_observations(triples, links, args):
    triples = triples.copy()

    triples["ticker"] = (
        triples["ticker"].astype(str).str.upper()
    )

    # Only allowed extraction tiers.
    allowed = (
        ["extracted"]
        if args.tier == "conservative"
        else ["extracted", "default"]
    )

    triples = triples[
        triples["ticker"].isin(TICKERS)
        & triples["extraction_type"].isin(allowed)
    ].copy()

    # Exact source key. Never join on year alone.
    linkcols = [
        "ticker", "source_file", "accession",
        "acceptance_utc", "filing_date",
        "safe_date", "link_status",
        "pdf_identity_verified",
        "document_url",
    ]

    observations = triples.merge(
        links[linkcols],
        on=["ticker", "source_file"],
        how="left",
        validate="many_to_one",
    )

    observations["safe_date"] = pd.to_datetime(
        observations["safe_date"], errors="coerce"
    )

    observations["mapped_candidate"] = (
        observations["link_status"].eq(
            "CANDIDATE_MATCH"
        )
    )

    # Names should not be empty.
    for c in ["entity", "relationship", "target"]:
        observations[c] = (
            observations[c].fillna("").astype(str)
        )

    observations["usable_candidate"] = (
        observations["mapped_candidate"]
        & observations["safe_date"].notna()
        & observations["entity"].str.strip().ne("")
        & observations["relationship"].str.strip().ne("")
        & observations["target"].str.strip().ne("")
    )

    observations["source_level"] = np.where(
        observations["pdf_identity_verified"],
        "VERIFIED_PDF_IDENTITY",
        np.where(
            observations["usable_candidate"],
            "EDGAR_DATE_CANDIDATE",
            "UNVERIFIED",
        )
    )

    observations["relation_key"] = [
        digest(
            t,
            normalized(e),
            normalized(r),
            normalized(o),
        )
        for t, e, r, o in zip(
            observations["ticker"],
            observations["entity"],
            observations["relationship"],
            observations["target"],
        )
    ]

    # Same relation mentioned repeatedly in one filing
    # should not count as multiple relation events.
    observations["observation_id"] = [
        digest(
            key, file
        )
        for key, file in zip(
            observations["relation_key"],
            observations["source_file"],
        )
    ]

    return observations


def build_events(observations):
    valid = observations[
        observations["usable_candidate"]
    ].copy()

    valid = valid.sort_values(
        [
            "ticker", "relation_key",
            "safe_date", "source_file",
            "observation_id",
        ]
    )

    events = valid.drop_duplicates(
        "observation_id",
        keep="first",
    ).copy()

    events = events.sort_values(
        [
            "ticker", "relation_key",
            "safe_date", "source_file",
        ]
    )

    events["observation_number"] = (
        events.groupby("relation_key").cumcount() + 1
    )

    events["event_type"] = np.where(
        events["observation_number"].eq(1),
        "FIRST_OBSERVED",
        "RECONFIRMED",
    )

    events["event_id"] = [
        digest("EVENT", x)
        for x in events["observation_id"]
    ]

    events["event_date"] = events["safe_date"]

    # This is an observation log.
    # Do not infer termination or economic impact.
    events["relation_terminated"] = False
    events["economic_event_verified"] = False

    return events


def build_daily(events, args):
    market = pd.read_csv(args.market)

    if "ticker" not in market and "tic" in market:
        market = market.rename(
            columns={"tic": "ticker"}
        )

    market["date"] = pd.to_datetime(
        market["date"], errors="raise"
    ).dt.normalize()

    market["ticker"] = (
        market["ticker"].astype(str).str.upper()
    )

    market = market[
        market["ticker"].isin(TICKERS)
        & market["date"].between(
            "2018-01-01", "2023-12-31"
        )
    ].copy()

    market = market[
        ["date", "ticker"]
    ].drop_duplicates()

    if market.duplicated(
        ["date", "ticker"]
    ).any():
        raise ValueError("Market key duplicates")

    output = []

    for ticker in TICKERS:
        dates = sorted(
            market.loc[
                market["ticker"].eq(ticker),
                "date"
            ].unique()
        )

        company_events = events[
            events["ticker"].eq(ticker)
        ].sort_values("event_date")

        seen = set()
        confirmed = set()
        index = 0
        rows = company_events.to_dict("records")

        for date in dates:
            today_new = 0
            today_confirmed = 0

            while (
                index < len(rows)
                and rows[index]["event_date"]
                <= date
            ):
                row = rows[index]
                key = row["relation_key"]

                if row["event_type"] == "FIRST_OBSERVED":
                    seen.add(key)
                    today_new += 1
                else:
                    confirmed.add(key)
                    today_confirmed += 1

                index += 1

            output.append({
                "date": date,
                "ticker": ticker,
                "tkg_active_observed_relations": len(seen),
                "tkg_reconfirmed_relations": len(confirmed),
                "tkg_new_relations_today": today_new,
                "tkg_reconfirmations_today": today_confirmed,
                "tkg_has_candidate_evidence": int(
                    len(seen) > 0
                ),
            })

    return pd.DataFrame(output).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--triples",
        default=(
            "kg/data/"
            "finreflectkg_5stocks_2018_2023.parquet"
        )
    )
    p.add_argument(
        "--mapping",
        default=(
            "kg/output/"
            "finreflectkg_dynamic_filing_map.csv"
        )
    )
    p.add_argument(
        "--manifest",
        default="outputs/v20/sec_filings_manifest.csv"
    )
    p.add_argument(
        "--market",
        default="kg/data/market_5stocks_2017_2026.csv"
    )
    p.add_argument(
        "--tier",
        choices=["conservative", "inclusive"],
        default="conservative"
    )
    p.add_argument(
        "--out-dir",
        default="outputs/v22/conservative"
    )
    args = p.parse_args()

    triples, mapping, manifest = load_sources(args)

    links = make_links(mapping, manifest)

    observations = prepare_observations(
        triples, links, args
    )

    events = build_events(observations)
    daily = build_daily(events, args)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    links.to_csv(
        out / "filing_link_audit.csv",
        index=False
    )

    observations.to_parquet(
        out / "relation_observations.parquet",
        index=False
    )

    events.to_parquet(
        out / "relation_events.parquet",
        index=False
    )

    daily.to_csv(
        out / "daily_temporal_features.csv",
        index=False
    )

    report = {
        "tier": args.tier,
        "source_pdf_count": len(links),
        "candidate_filing_matches": int(
            links["link_status"]
            .eq("CANDIDATE_MATCH").sum()
        ),
        "verified_pdf_identity_count": int(
            links["pdf_identity_verified"].sum()
        ),
        "raw_triples": len(triples),
        "candidate_events": len(events),
        "first_observed": int(
            events["event_type"]
            .eq("FIRST_OBSERVED").sum()
        ),
        "reconfirmed": int(
            events["event_type"]
            .eq("RECONFIRMED").sum()
        ),
        "daily_rows": len(daily),
        "strict_pit_approved": False,
        "finDKG_publication_verified": False,
        "limitations": [
            "PDF identity has not been independently verified",
            "Candidate date match is not strict PIT certification",
            "Relationship counts are not economic ground truth",
            "No relation termination is inferred",
            "No future event-time columns are used",
        ],
    }

    (out / "report.json").write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V22 TEMPORAL KG ===")
    print(json.dumps(report, indent=2))
    print("Output:", out)


if __name__ == "__main__":
    main()
