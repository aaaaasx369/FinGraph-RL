#!/usr/bin/env python3
import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RSR = ROOT / "references/Temporal_Relational_Stock_Ranking"
V35 = ROOT / "outputs/v35/evidence_ledger.csv"

OUT = (
    ROOT
    / "outputs/finrl_dkg_original_full/four_relation_kg_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://api.sampoq.com/datasets/supply-chain.csv"

STOCKS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

FIELDS = [
    "head",
    "relation",
    "tail",
    "filing_date",
    "available_at",
    "source",
    "accession",
    "source_url",
    "evidence",
    "human_verified",
    "pit_verified",
    "status",
]


def ticker(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def txt(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def flag(value):
    return txt(value).lower() in {
        "true", "1", "yes"
    }


def inspect_rsr():
    print("\n=== RSR OFFICIAL RELATIONS ===")

    archives = [
        RSR / "data/relation.tar.gz",
        RSR / "relation.tar.gz",
    ]

    found = next(
        (p for p in archives if p.exists()),
        None,
    )

    rows = []

    if found is None:
        print("RSR archive not found")
    else:
        print("Archive:", found)

        with tarfile.open(found, "r:gz") as tar:
            members = [
                m for m in tar.getmembers()
                if m.isfile()
            ]

            for m in members:
                if (
                    "sector" in m.name.lower()
                    or "industry" in m.name.lower()
                    or "wikidata" in m.name.lower()
                ):
                    rows.append({
                        "file": m.name,
                        "bytes": m.size,
                    })

        print("Relation files:", len(rows))

        for item in rows[:60]:
            print(
                item["file"],
                item["bytes"],
            )

    pd.DataFrame(
        rows, columns=["file", "bytes"]
    ).to_csv(
        OUT / "rsr_archive_inventory.csv",
        index=False,
    )

    print(
        "RSR historical PIT: NOT CERTIFIED. "
        "Archive inventory is not evidence of "
        "historical sector validity."
    )


def get_supply_chain():
    print("\n=== SAMPOQ SUPPLY CHAIN ===")

    cache = OUT / "sampoq_supply_chain_snapshot.csv"

    if not cache.exists():
        print("Downloading:", URL, flush=True)

        request = urllib.request.Request(
            URL,
            headers={
                "User-Agent": (
                    "Academic-Research-Financial-KG/1.0"
                )
            },
        )

        with urllib.request.urlopen(
            request, timeout=120
        ) as response:
            content = response.read()

        if not content:
            raise RuntimeError("Empty supply-chain download")

        cache.write_bytes(content)

    df = pd.read_csv(
        cache, low_memory=False,
        dtype={
            "source_cik": "string",
            "filing_accession": "string",
        },
    )

    required = {
        "source_ticker",
        "related_ticker",
        "relation_type",
        "filing_date",
        "filing_accession",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "SAMPOQ schema mismatch: "
            + str(sorted(missing))
        )

    df["source_ticker"] = (
        df["source_ticker"].map(ticker)
    )
    df["related_ticker"] = (
        df["related_ticker"].map(ticker)
    )
    df["relation_type"] = (
        df["relation_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["filing_date"] = pd.to_datetime(
        df["filing_date"], errors="coerce"
    )

    df = df[
        df["filing_date"].between(
            "2018-01-01", "2023-12-31"
        )
    ].copy()

    df = df[
        df["source_ticker"].isin(STOCKS)
        | df["related_ticker"].isin(STOCKS)
    ].copy()

    records = []

    for _, r in df.iterrows():
        filer = r["source_ticker"]
        other = r["related_ticker"]

        if not filer or not other or filer == other:
            continue

        relation = r["relation_type"]

        if relation == "customer":
            # Filer sells to the counterparty.
            head, rel, tail = (
                filer, "SUPPLIES_TO", other
            )
        elif relation == "supplier":
            # Filer purchases from the counterparty.
            head, rel, tail = (
                filer, "CUSTOMER_OF", other
            )
        else:
            continue

        accession = txt(
            r.get("filing_accession")
        )

        cik = txt(r.get("source_cik"))
        if cik.endswith(".0"):
            cik = cik[:-2]

        accession_number = accession.replace("-", "")

        source_url = ""

        if cik.isdigit() and accession_number.isdigit():
            source_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession_number}/"
            )

        records.append({
            "head": head,
            "relation": rel,
            "tail": tail,
            "filing_date": r["filing_date"].date().isoformat(),
            "available_at": "",
            "source": "SAMPOQ_10K",
            "accession": accession,
            "source_url": source_url,
            "evidence": txt(
                r.get("related_name")
            ),
            "human_verified": False,
            "pit_verified": False,
            "status": "CANDIDATE",
        })

    print("Historical candidate edges:", len(records))

    return records


def get_v35():
    print("\n=== V35 COMPETITION ===")

    if not V35.exists():
        print("V35 file missing")
        return []

    df = pd.read_csv(V35, low_memory=False)

    required = {
        "ticker",
        "target_ticker",
        "relation",
        "filing_date",
        "accession",
    }

    if not required.issubset(df.columns):
        raise RuntimeError(
            "V35 schema mismatch"
        )

    records = []

    for _, r in df.iterrows():
        if txt(r["relation"]).upper() != "COMPETES_WITH":
            continue

        head = ticker(r["ticker"])
        tail = ticker(r["target_ticker"])

        if not head or not tail or head == tail:
            continue

        if head not in STOCKS and tail not in STOCKS:
            continue

        date = pd.to_datetime(
            r["filing_date"],
            errors="coerce",
        )

        if pd.isna(date) or not (
            2018 <= date.year <= 2023
        ):
            continue

        # A V35 silver label is not independently
        # source-verified historical evidence.
        human = flag(r.get("human_verified", False))
        pit = flag(r.get("strict_pit_approved", False))

        records.append({
            "head": head,
            "relation": "COMPETES_WITH",
            "tail": tail,
            "filing_date": date.date().isoformat(),
            "available_at": "",
            "source": "V35_SEC",
            "accession": txt(r.get("accession")),
            "source_url": "",
            "evidence": txt(
                r.get("evidence_quote")
            ),
            "human_verified": human,
            "pit_verified": pit,
            "status": "CANDIDATE",
        })

    print("Historical competition candidates:", len(records))

    return records


def build():
    inspect_rsr()

    all_rows = (
        get_supply_chain()
        + get_v35()
    )

    df = pd.DataFrame(
        all_rows, columns=FIELDS
    )

    if len(df):
        df = df.drop_duplicates(
            subset=[
                "head",
                "relation",
                "tail",
                "filing_date",
                "accession",
                "source",
            ]
        ).sort_values([
            "filing_date",
            "head",
            "relation",
            "tail",
        ])

    df.to_csv(
        OUT / "four_relation_candidates.csv",
        index=False,
    )

    # RSR SAME_SECTOR remains a separate
    # undated candidate source, not a 2018 event.
    sector = pd.DataFrame(columns=[
        "head",
        "relation",
        "tail",
        "sector_name",
        "valid_from",
        "source",
        "status",
    ])

    sector.to_csv(
        OUT / "same_sector_historical_candidates.csv",
        index=False,
    )

    companies = set(STOCKS)

    for row in all_rows:
        companies.add(row["head"])
        companies.add(row["tail"])

    companies = sorted(
        c for c in companies if c
    )

    ids = {
        ticker: index
        for index, ticker in enumerate(companies)
    }

    company_df = pd.DataFrame([
        {
            "ticker": ticker,
            "entity_id": entity_id,
            "tradable": ticker in STOCKS,
        }
        for ticker, entity_id in ids.items()
    ])

    company_df.to_csv(
        OUT / "company_entity_ids.csv",
        index=False,
    )

    relation_ids = pd.DataFrame([
        {
            "relation": relation,
            "relation_id": index,
        }
        for index, relation in enumerate(RELATIONS)
    ])

    relation_ids.to_csv(
        OUT / "relation_ids.csv",
        index=False,
    )

    counts = []

    for rel in RELATIONS:
        part = df[
            df["relation"].eq(rel)
        ]

        counts.append({
            "relation": rel,
            "historical_candidates": len(part),
            "human_verified": int(
                part["human_verified"].sum()
            ),
            "strict_pit_verified": int(
                part["pit_verified"].sum()
                & part["human_verified"].sum()
            ) if False else int(
                (
                    part["pit_verified"]
                    & part["human_verified"]
                ).sum()
            ),
        })

    coverage = pd.DataFrame(counts)

    coverage.to_csv(
        OUT / "candidate_coverage.csv",
        index=False,
    )

    manifest = {
        "candidate_edges": len(df),
        "company_entities": len(companies),
        "relation_types": RELATIONS,
        "training_completed": False,
        "strict_pit_certified": False,
        "same_sector_historical_source_verified": False,
        "note": (
            "Candidate graph only. SAMPOQ is machine "
            "extracted. V35 has silver competition evidence. "
            "Historical SEC acceptance and human review "
            "are still needed."
        ),
    }

    (OUT / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== FOUR RELATIONS CANDIDATE COVERAGE ===")
    print(coverage.to_string(index=False))

    print("\nCompany nodes:", len(companies))
    print("Historical candidate rows:", len(df))
    print("Training completed: False")
    print("Strict PIT certified: False")
    print("\nSAVED:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build",
        action="store_true",
    )

    args = parser.parse_args()

    if args.build:
        build()
    else:
        inspect_rsr()
