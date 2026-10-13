#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT /
    "outputs/finrl_dkg_original_full/"
    "sec_relation_expansion/historical/"
    "sec_chunks_2018_2023.jsonl"
)

COMPANIES = (
    ROOT / "kg/data/reference/company_tickers.json"
)

OUT = (
    ROOT /
    "outputs/finrl_dkg_original_full/"
    "sec_relation_expansion/historical"
)

OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
}

KEYWORDS = re.compile(
    r"\b(supplier|suppliers|supply|supplies|"
    r"customer|customers|purchase|purchased|"
    r"competitor|competitors|compete|competition|"
    r"industry|sector)\b",
    re.I,
)

TEXT_KEYS = [
    "text", "chunk_text", "content", "page_content"
]


def load_aliases():
    raw = json.loads(COMPANIES.read_text())

    entries = (
        raw.values() if isinstance(raw, dict)
        else raw
    )

    aliases = defaultdict(set)
    cik_to_tickers = defaultdict(set)

    for company in entries:
        if not isinstance(company, dict):
            continue

        ticker = str(
            company.get("ticker", "")
        ).upper().strip()

        name = str(
            company.get(
                "title",
                company.get("name", "")
            )
        ).strip()

        cik = str(
            company.get(
                "cik_str",
                company.get("cik", "")
            )
        ).strip()

        if not ticker or len(name) < 8:
            continue

        aliases[name].add(ticker)

        if cik.isdigit():
            cik_to_tickers[int(cik)].add(ticker)

    # Only exact/full corporate names. Avoid
    # short or ambiguous aliases such as "Google".
    additions = {
        "Apple Inc.": "AAPL",
        "Microsoft Corporation": "MSFT",
        "NVIDIA Corporation": "NVDA",
        "Amazon.com, Inc.": "AMZN",
        "Alphabet Inc.": "GOOGL",
    }

    for name, ticker in additions.items():
        aliases[name].add(ticker)

    patterns = []

    for name, tickers in aliases.items():
        pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            + re.escape(name)
            + r"(?![A-Za-z0-9])",
            re.I
        )
        patterns.append(
            (pattern, name, tuple(sorted(tickers)))
        )

    return patterns, cik_to_tickers


def get_text(obj):
    for key in TEXT_KEYS:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    return ""


def sentences(text):
    return [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+|\n+",
            text
        )
        if s.strip()
    ]


def main():
    patterns, cik_to_tickers = load_aliases()

    rows = []
    chunks = 0
    companies_found = Counter()

    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue

            obj = json.loads(line)
            chunks += 1

            issuer = str(
                obj.get(
                    "ticker",
                    obj.get("metadata", {}).get(
                        "ticker", ""
                    )
                )
            ).upper().strip()

            if issuer not in TARGETS:
                continue

            meta = obj.get(
                "verified_sec_metadata", {}
            )

            text = get_text(obj)

            for sentence in sentences(text):
                if not KEYWORDS.search(sentence):
                    continue

                matches = []

                for pattern, name, tickers in patterns:
                    if pattern.search(sentence):
                        for ticker in tickers:
                            if ticker != issuer:
                                matches.append(
                                    (ticker, name)
                                )

                for ticker, name in set(matches):
                    rows.append({
                        "issuer": issuer,
                        "counterparty": ticker,
                        "matched_name": name,
                        "sentence": sentence,
                        "accession": meta.get(
                            "accession", ""
                        ),
                        "filing_date": meta.get(
                            "filing_date", ""
                        ),
                        "acceptance_utc": meta.get(
                            "acceptance_utc", ""
                        ),
                        "available_date": meta.get(
                            "available_date", ""
                        ),
                        "source_url": meta.get(
                            "document_url", ""
                        ),
                        "source_line": meta.get(
                            "original_jsonl_line", ""
                        ),
                        "status": "CO_MENTION_ONLY",
                    })

                    companies_found[ticker] += 1

            if chunks % 3000 == 0:
                print(
                    f"Scanned {chunks:,} historical chunks",
                    flush=True,
                )

    cols = [
        "issuer", "counterparty",
        "matched_name", "sentence",
        "accession", "filing_date",
        "acceptance_utc", "available_date",
        "source_url", "source_line", "status"
    ]

    df = pd.DataFrame(rows, columns=cols)

    # Retain original source rows, but remove
    # duplicate pattern matches in a sentence.
    df = df.drop_duplicates(
        subset=[
            "issuer",
            "counterparty",
            "accession",
            "source_line",
            "sentence",
        ]
    )

    df.to_csv(
        OUT / "historical_named_company_candidates.csv",
        index=False
    )

    print("\n=== HISTORICAL NAMED ENTITIES ===")
    print("Historical chunks:", chunks)
    print("Company co-mentions:", len(df))
    print("Distinct counterparties:", df["counterparty"].nunique())

    print("\n=== TOP COUNTERPARTIES ===")
    for ticker, count in companies_found.most_common(25):
        print(ticker, count)

    print(
        "\nSTATUS: COMPANY CO-MENTIONS ONLY. "
        "No semantic relationship verified."
    )


if __name__ == "__main__":
    main()
