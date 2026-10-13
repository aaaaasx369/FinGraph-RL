#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "rag/data/processed/sec/sec_chunks.jsonl"
MANIFEST = ROOT / "outputs/v20/sec_filings_manifest.csv"

OUT = (
    ROOT
    / "outputs/finrl_dkg_original_full/"
    "sec_relation_expansion/historical"
)
OUT.mkdir(parents=True, exist_ok=True)

ACC_DASHED = re.compile(
    r"(?<!\d)(\d{10})-(\d{2})-(\d{6})(?!\d)"
)
ACC_BARE = re.compile(
    r"(?<!\d)(\d{10})(\d{2})(\d{6})(?!\d)"
)


def accession(value):
    value = str(value or "")

    m = ACC_DASHED.search(value)
    if m:
        return "-".join(m.groups())

    m = ACC_BARE.search(value)
    if m:
        return "-".join(m.groups())

    return ""


def flatten(obj):
    if not isinstance(obj, dict):
        return []

    result = []
    for k, v in obj.items():
        if isinstance(v, dict):
            result.extend(flatten(v))
        elif isinstance(v, (str, int)):
            result.append((k.lower(), str(v)))
    return result


manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")

manifest["ticker"] = (
    manifest["ticker"].str.upper().str.strip()
)

manifest["accession_norm"] = (
    manifest["accession"].map(accession)
)

lookup = {}

for _, row in manifest.iterrows():
    key = (row["ticker"], row["accession_norm"])
    lookup.setdefault(key, []).append(row.to_dict())

kept = 0
excluded = Counter()
year_counts = Counter()
issuer_counts = Counter()

kept_path = OUT / "sec_chunks_2018_2023.jsonl"
excluded_path = OUT / "excluded_chunks.csv"

excluded_rows = []

with SOURCE.open(encoding="utf-8", errors="replace") as inp, \
     kept_path.open("w", encoding="utf-8") as out:

    for line_no, line in enumerate(inp, 1):
        if not line.strip():
            continue

        obj = json.loads(line)

        fields = flatten(obj)

        ticker = ""
        candidates = set()

        for key, value in fields:
            if key in {
                "ticker", "tic", "symbol",
                "company_ticker",
                "metadata.ticker"
            }:
                ticker = value.upper().strip()

            if any(x in key for x in (
                "accession", "chunk_id",
                "doc_id", "url", "document"
            )):
                acc = accession(value)
                if acc:
                    candidates.add(acc)

        matches = []

        for acc in candidates:
            matches.extend(
                lookup.get((ticker, acc), [])
            )

        # Unique manifest filing required.
        identities = {
            (m["ticker"], m["accession"])
            for m in matches
        }

        if len(identities) != 1:
            reason = (
                "NO_MANIFEST_MATCH"
                if not identities
                else "AMBIGUOUS_MANIFEST_MATCH"
            )
            excluded[reason] += 1
            excluded_rows.append({
                "line_no": line_no,
                "ticker": ticker,
                "accessions": ";".join(sorted(candidates)),
                "reason": reason,
            })
            continue

        record = matches[0]

        accepted = pd.to_datetime(
            record.get("acceptance_utc", ""),
            utc=True,
            errors="coerce",
        )

        if pd.isna(accepted):
            excluded["NO_ACCEPTANCE_TIME"] += 1
            continue

        if not (2018 <= accepted.year <= 2023):
            excluded["OUTSIDE_PERIOD"] += 1
            continue

        if (
            str(record.get("metadata_verified", "")) != "1"
            or str(record.get("text_verified", "")) != "1"
        ):
            excluded["UNVERIFIED_SOURCE"] += 1
            continue

        obj["verified_sec_metadata"] = {
            "accession": record["accession"],
            "filing_date": record["filing_date"],
            "acceptance_utc": record["acceptance_utc"],
            "available_date": record["available_date"],
            "document_url": record["document_url"],
            "source_sha256": record["source_sha256"],
            "metadata_verified": True,
            "text_verified": True,
            "original_jsonl_line": line_no,
        }

        out.write(
            json.dumps(obj, ensure_ascii=False)
            + "\n"
        )

        kept += 1
        year_counts[accepted.year] += 1
        issuer_counts[ticker] += 1

pd.DataFrame(
    excluded_rows,
    columns=[
        "line_no", "ticker",
        "accessions", "reason"
    ],
).to_csv(excluded_path, index=False)

print("\n=== HISTORICAL SEC CORPUS ===")
print("Kept chunks:", kept)

print("\n=== YEARS ===")
for y in range(2018, 2024):
    print(y, year_counts[y])

print("\n=== TICKERS ===")
for t in ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]:
    print(t, issuer_counts[t])

print("\n=== EXCLUSIONS ===")
for reason, count in sorted(excluded.items()):
    print(reason, count)

print("\nOutput:", kept_path)
print("Strict PIT: NOT YET CERTIFIED")
