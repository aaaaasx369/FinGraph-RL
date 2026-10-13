#!/usr/bin/env python3
"""
V22-B Offline As-of SEC + Temporal KG Retriever.

Strict as-of document date gate:
    source filing date < query/trade date

SEC:
    original downloaded filing documents
    verified SHA256 when supplied by V20 manifest

KG:
    historical first-observed/reconfirmed relation events
    with date-candidate provenance

No paid API, no LLM predictions, no future retrieval.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd


def norm(x):
    return re.sub(r"\s+", " ", str(x).lower()).strip()


def tokens(s):
    return re.findall(
        r"[a-z][a-z0-9_-]{2,}",
        norm(s),
    )


def score(query, text):
    q = set(tokens(query))
    if not q:
        return 0.0

    t = tokens(text)
    frequencies = Counter(t)

    overlap = sum(
        math.log1p(frequencies[w])
        for w in q if w in frequencies
    )

    return float(
        overlap / math.sqrt(1 + len(t) / 100)
    )


class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skipping = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skipping += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skipping = max(
                0, self.skipping - 1
            )

    def handle_data(self, data):
        if not self.skipping:
            self.parts.append(data)


def read_text(raw):
    source = raw.decode(
        "utf-8", errors="replace"
    )

    parser = HTMLText()
    try:
        parser.feed(source)
        txt = " ".join(parser.parts)
    except Exception:
        txt = re.sub(
            r"<[^>]+>", " ", source
        )

    return re.sub(
        r"\s+", " ", txt
    )


def chunks(text, size=1300, stride=1100):
    for start in range(0, len(text), stride):
        segment = text[start:start + size]
        if len(segment) < 100:
            break
        yield start, segment


def get_kg(args, cutoff):
    file = (
        Path(args.kg_dir)
        / "relation_events.parquet"
    )

    if not file.exists():
        raise FileNotFoundError(file)

    df = pd.read_parquet(file)

    df["event_date"] = pd.to_datetime(
        df["event_date"],
        errors="coerce"
    ).dt.normalize()

    df = df[
        df["ticker"].eq(args.ticker)
        & df["event_date"].lt(cutoff)
    ].copy()

    # Represent latest observed state of each relation.
    df = df.sort_values(
        ["event_date", "observation_number"]
    ).drop_duplicates(
        "relation_key", keep="last"
    )

    matches = []

    for _, row in df.iterrows():
        description = (
            f"{row['entity']} "
            f"{row['relationship']} "
            f"{row['target']}"
        )
        value = score(
            args.question, description
        )
        if value <= 0:
            continue

        matches.append({
            "type": "KG_RELATION",
            "score": value,
            "entity": row["entity"],
            "relation": row["relationship"],
            "target": row["target"],
            "observed_at": str(
                row["event_date"].date()
            ),
            "source_file": row["source_file"],
            "accession_candidate": row["accession"],
            "event_id": row["event_id"],
            "link_status": row["link_status"],
            "provenance_verified": bool(
                row["pdf_identity_verified"]
            ),
        })

    return sorted(
        matches,
        key=lambda x: x["score"],
        reverse=True
    )[:args.top_k]


def get_sec(args, cutoff):
    manifest = pd.read_csv(args.manifest)

    manifest["filing_date"] = pd.to_datetime(
        manifest["filing_date"],
        errors="coerce"
    ).dt.normalize()

    manifest = manifest[
        manifest["ticker"].eq(args.ticker)
        & manifest["filing_date"].lt(cutoff)
        & manifest["text_verified"].eq(1)
    ].copy()

    if args.form:
        manifest = manifest[
            manifest["form"].isin(args.form)
        ]

    result = []

    for _, filing in manifest.iterrows():
        name = (
            f"{filing['ticker']}_"
            f"{filing['accession']}_"
            + re.sub(
                r"[^A-Za-z0-9_.-]",
                "_",
                str(filing["primary_document"])
            )
        )

        path = (
            Path(args.cache)
            / "documents"
            / name
        )

        if not path.exists():
            continue

        raw = path.read_bytes()

        expected_sha = str(
            filing["source_sha256"]
        )

        actual_sha = hashlib.sha256(
            raw
        ).hexdigest()

        if actual_sha != expected_sha:
            raise ValueError(
                "SEC document hash mismatch: "
                + str(path)
            )

        text = read_text(raw)

        for offset, segment in chunks(text):
            relevance = score(
                args.question, segment
            )

            if relevance <= 0:
                continue

            result.append({
                "type": "SEC_TEXT",
                "score": relevance,
                "ticker": filing["ticker"],
                "filing_date": str(
                    filing["filing_date"].date()
                ),
                "accession": filing["accession"],
                "document_url": filing["document_url"],
                "source_sha256": actual_sha,
                "character_offset": offset,
                "excerpt": segment[:900],
                "source_text_hash_verified": True,
            })

    return sorted(
        result,
        key=lambda x: x["score"],
        reverse=True
    )[:args.top_k]


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ticker", required=True,
        choices=[
            "AAPL", "MSFT", "NVDA",
            "AMZN", "GOOGL",
        ]
    )
    p.add_argument(
        "--as-of", required=True
    )
    p.add_argument(
        "--question", required=True
    )
    p.add_argument(
        "--kg-dir",
        default="outputs/v22/conservative"
    )
    p.add_argument(
        "--manifest",
        default="outputs/v20/sec_filings_manifest.csv"
    )
    p.add_argument(
        "--cache",
        default="outputs/v20/sec_cache"
    )
    p.add_argument(
        "--form", nargs="+",
        default=["10-K", "10-Q"]
    )
    p.add_argument(
        "--top-k", type=int,
        default=5
    )
    p.add_argument(
        "--output",
        default="outputs/v22/retrieval_result.json"
    )

    args = p.parse_args()

    cutoff = pd.Timestamp(
        args.as_of
    ).normalize()

    if args.top_k < 1:
        p.error("--top-k must be positive")

    kg = get_kg(args, cutoff)
    sec = get_sec(args, cutoff)

    output = {
        "ticker": args.ticker,
        "as_of": args.as_of,
        "question": args.question,
        "SEC_evidence": sec,
        "KG_relation_evidence": kg,
        "source_policy": (
            "Prior filing dates only. "
            "SEC downloaded bytes SHA256 checked. "
            "KG PDF identity remains candidate-level."
        ),
        "LLM_answer_generated": False,
        "FinDKG_historical_publication_verified": False,
    }

    path = Path(args.output)
    path.parent.mkdir(
        parents=True, exist_ok=True
    )
    path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V22 AS-OF GRAPH-RAG ===")
    print("Ticker:", args.ticker)
    print("As-of:", args.as_of)
    print("SEC evidence:", len(sec))
    print("KG evidence:", len(kg))

    for i, item in enumerate(sec, 1):
        print(
            f"\nSEC {i}:",
            item["filing_date"],
            item["accession"],
            "score=",
            round(item["score"], 4),
        )
        print(item["excerpt"][:300])

    for i, item in enumerate(kg, 1):
        print(
            f"\nKG {i}:",
            item["entity"],
            "--",
            item["relation"],
            "-->",
            item["target"],
        )
        print(
            "SOURCE:",
            item["source_file"],
            "DATE:",
            item["observed_at"],
        )

    if not sec and not kg:
        print(
            "ABSTAIN: No matching historical evidence."
        )

    print("\nSaved:", path)


if __name__ == "__main__":
    main()
