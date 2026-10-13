#!/usr/bin/env python3
"""
V26 SEC-FIRST TEMPORAL CONCEPT KG

Universe: AAPL MSFT NVDA AMZN GOOGL ONLY
Source: locally cached SEC 10-K / 10-Q documents

Creates:
  filing_evidence.parquet
  filing_features.csv
  daily_features.csv
  source_audit.csv
  report.json

A known filing's evidence becomes usable only on
calendar days AFTER its filing date.

This is intentionally conservative and does not
certify intraday PIT, semantic truth, or execution PIT.

No LLM/API key needed.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

# Terms are only evidence anchors. They do NOT
# automatically prove an economic relationship.
TOPICS = {
    "supply_chain": [
        r"\bsupply chain\b",
        r"\bsuppliers?\b",
        r"\bsupply constraints?\b",
        r"\bprocurement\b",
        r"\bcomponent shortages?\b",
    ],
    "manufacturing": [
        r"\bmanufactur(?:e|es|ed|ing|er|ers)\b",
        r"\bfoundr(?:y|ies)\b",
        r"\bfabricat(?:e|ion|ing)\b",
        r"\bsemiconductor wafers?\b",
    ],
    "cloud": [
        r"\bcloud services?\b",
        r"\bcloud computing\b",
        r"\bdata cent(?:er|re)s?\b",
        r"\bazure\b",
    ],
    "competition": [
        r"\bcompetition\b",
        r"\bcompetitors?\b",
        r"\bcompetitive\b",
        r"\bmarket share\b",
    ],
    "regulation": [
        r"\bregulat(?:ion|ions|ory)\b",
        r"\bantitrust\b",
        r"\bexport controls?\b",
        r"\bsanctions?\b",
    ],
    "litigation": [
        r"\blitigation\b",
        r"\blawsuits?\b",
        r"\blegal proceedings?\b",
    ],
    "finance": [
        r"\bcash flows?\b",
        r"\boperating income\b",
        r"\baccounts receivable\b",
        r"\brevenue\b",
    ],
    "research": [
        r"\bresearch and development\b",
        r"\br&d\b",
        r"\bproduct development\b",
    ],
}
TOPIC_NAMES = list(TOPICS)
PATTERNS = {
    topic: re.compile("|".join(patterns), re.I)
    for topic, patterns in TOPICS.items()
}


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def read_document(data):
    text = data.decode("utf-8", errors="replace")
    parser = HTMLText()
    parser.feed(text)
    return " ".join(" ".join(parser.parts).split())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def id_for(*parts):
    return sha256(
        "\x1f".join(map(str, parts)).encode()
    )[:24]


def cache_path(record):
    filename = (
        f"{record['ticker']}_{record['accession']}_"
        + re.sub(
            r"[^A-Za-z0-9_.-]", "_",
            str(record["primary_document"])
        )
    )
    return (
        ROOT / "outputs/v20/sec_cache/documents"
        / filename
    )


def read_manifest():
    df = pd.read_csv(
        ROOT / "outputs/v20/sec_filings_manifest.csv",
        dtype={"accession": str},
    )
    needed = [
        "ticker", "accession", "form",
        "filing_date", "primary_document",
        "source_sha256", "text_verified",
    ]
    missing = [c for c in needed if c not in df]
    if missing:
        raise ValueError(f"Missing manifest columns: {missing}")

    df["ticker"] = df["ticker"].str.upper()
    df["filing_date"] = pd.to_datetime(
        df["filing_date"], errors="raise"
    ).dt.normalize()

    df = df[
        df["ticker"].isin(STOCKS)
        & df["form"].isin(["10-K", "10-Q"])
    ].copy()

    if df.duplicated(["ticker", "accession"]).any():
        raise ValueError("Duplicate SEC accession in manifest")

    return df.sort_values(
        ["ticker", "filing_date", "accession"]
    )


def read_market():
    df = pd.read_csv(
        ROOT / "kg/data/market_5stocks_2017_2026.csv"
    )

    if "ticker" not in df and "tic" in df:
        df = df.rename(columns={"tic": "ticker"})

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()
    df["ticker"] = df["ticker"].str.upper()

    df = df[
        df["ticker"].isin(STOCKS)
        & df["date"].between(
            "2018-01-01", "2023-12-31"
        )
    ][["date", "ticker"]].drop_duplicates()

    if len(df) != 7545:
        raise ValueError(
            f"Expected 7545 stock-days, got {len(df)}"
        )

    for ticker in STOCKS:
        if (df["ticker"] == ticker).sum() != 1509:
            raise ValueError(f"Incomplete market data: {ticker}")

    return df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def extract_filing(record, raw, chunk_size=1200):
    """
    Store document-local evidence windows.
    A matched word is a concept mention, not a
    verified financial KG relation.

    Use non-overlapping windows to avoid counting
    the same phrase repeatedly across overlaps.
    """
    text = read_document(raw)
    evidence = []
    topic_counts = Counter()

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        if len(chunk) < 80:
            continue

        for topic, pattern in PATTERNS.items():
            hits = list(pattern.finditer(chunk))
            if not hits:
                continue

            topic_counts[topic] += len(hits)

            # Preserve one excerpt per topic/window.
            first = hits[0]
            excerpt_start = max(0, first.start() - 160)
            excerpt_end = min(
                len(chunk), first.end() + 320
            )

            evidence.append({
                "event_id": id_for(
                    record["ticker"], record["accession"],
                    topic, start
                ),
                "ticker": record["ticker"],
                "accession": record["accession"],
                "form": record["form"],
                "filing_date": record["filing_date"],
                "available_from": (
                    record["filing_date"]
                    + pd.Timedelta(days=1)
                ),
                "source_node": (
                    "COMPANY:" + record["ticker"]
                ),
                "target_node": (
                    "CONCEPT:" + topic
                ),
                "relation": "MENTIONS_CONCEPT",
                "concept": topic,
                "document_node": (
                    "DOCUMENT:" + record["accession"]
                ),
                "offset": int(start + excerpt_start),
                "matched_term": first.group(0),
                "excerpt": chunk[
                    excerpt_start:excerpt_end
                ],
                "source_sha256": record["source_sha256"],
                "source_hash_verified": True,
                "economic_relationship_verified": False,
            })

    return evidence, topic_counts


def build(args):
    manifest = read_manifest()
    market = read_market()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    evidence_rows = []
    filing_rows = []
    audit_rows = []

    for _, record in manifest.iterrows():
        path = cache_path(record)
        raw = path.read_bytes() if path.is_file() else None

        actual_hash = sha256(raw) if raw is not None else ""
        expected_hash = str(record["source_sha256"])

        verified = (
            raw is not None
            and actual_hash == expected_hash
            and str(record["text_verified"]).lower()
            in {"true", "1", "1.0"}
        )

        audit_rows.append({
            "ticker": record["ticker"],
            "accession": record["accession"],
            "filing_date": record["filing_date"],
            "cache_found": raw is not None,
            "sha256_matches_manifest": (
                raw is not None and actual_hash == expected_hash
            ),
            "included": bool(verified),
        })

        # Fail closed. Don't silently reduce corpus.
        if not verified:
            raise RuntimeError(
                "Missing/unverified SEC document: "
                f"{record['ticker']} {record['accession']}"
            )

        evidence, counts = extract_filing(record, raw)
        evidence_rows.extend(evidence)

        row = {
            "ticker": record["ticker"],
            "accession": record["accession"],
            "form": record["form"],
            "filing_date": record["filing_date"],
            "available_from": (
                record["filing_date"]
                + pd.Timedelta(days=1)
            ),
            "evidence_windows": len(evidence),
        }

        for topic in TOPIC_NAMES:
            row["sec_" + topic] = int(
                counts.get(topic, 0)
            )

        filing_rows.append(row)

    evidence = pd.DataFrame(evidence_rows)
    filings = pd.DataFrame(filing_rows)
    audit = pd.DataFrame(audit_rows)

    if evidence.empty:
        raise ValueError("No SEC evidence extracted")

    if evidence["event_id"].duplicated().any():
        raise ValueError("Duplicate evidence IDs")

    # Each filing is a snapshot of evidence counts.
    # Latest-known filing is carried forward.
    daily_rows = []

    for ticker in STOCKS:
        dates = (
            market.loc[
                market["ticker"] == ticker, "date"
            ]
            .sort_values().tolist()
        )

        known_filings = (
            filings[filings["ticker"] == ticker]
            .sort_values(
                ["available_from", "accession"]
            )
            .to_dict("records")
        )

        position = 0
        latest = None
        updates = 0

        for date in dates:
            new_filings = 0

            while (
                position < len(known_filings)
                and known_filings[position][
                    "available_from"
                ] <= date
            ):
                latest = known_filings[position]
                position += 1
                updates += 1
                new_filings += 1

            row = {
                "date": date,
                "ticker": ticker,
                "sec_has_past_filing": int(
                    latest is not None
                ),
                "sec_new_filings_since_previous_market_day":
                    new_filings,
                "sec_known_filing_count": updates,
                "sec_latest_accession": (
                    latest["accession"]
                    if latest else ""
                ),
                "sec_latest_available_from": (
                    latest["available_from"]
                    if latest else pd.NaT
                ),
            }

            for topic in TOPIC_NAMES:
                row["sec_" + topic] = (
                    latest["sec_" + topic]
                    if latest else 0
                )

            daily_rows.append(row)

    daily = pd.DataFrame(daily_rows)

    if len(daily) != 7545:
        raise RuntimeError("Daily panel length mismatch")

    if daily.duplicated(
        ["date", "ticker"]
    ).any():
        raise RuntimeError("Duplicate daily keys")

    dates = pd.to_datetime(daily["date"])
    known_dates = pd.to_datetime(
        daily["sec_latest_available_from"]
    )
    known_mask = known_dates.notna()

    if not (
        known_dates[known_mask].to_numpy()
        <= dates[known_mask].to_numpy()
    ).all():
        raise RuntimeError("Future SEC filing leakage")

    evidence.to_parquet(
        output / "filing_evidence.parquet",
        index=False
    )
    filings.to_csv(
        output / "filing_features.csv",
        index=False
    )
    daily.to_csv(
        output / "daily_features.csv",
        index=False
    )
    audit.to_csv(
        output / "source_audit.csv",
        index=False
    )

    report = {
        "universe": STOCKS,
        "source_filings": int(len(filings)),
        "source_hash_matched": int(
            audit["sha256_matches_manifest"].sum()
        ),
        "evidence_windows": int(len(evidence)),
        "daily_stock_rows": int(len(daily)),
        "filing_updates_by_ticker": {
            t: int(
                daily.loc[
                    daily["ticker"] == t,
                    "sec_new_filings_since_previous_market_day"
                ].sum()
            )
            for t in STOCKS
        },
        "updated_stock_days": int(
            (
                daily[
                    "sec_new_filings_since_previous_market_day"
                ] > 0
            ).sum()
        ),
        "strict_pit_approved": False,
        "graph_claims_verified": False,
        "note": (
            "SEC text-supported concept mentions only; "
            "no directed company relations inferred."
        ),
    }

    (output / "report.json").write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V26 SEC TEMPORAL KG ===")
    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))
    print("Saved:", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="outputs/v26"
    )
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
