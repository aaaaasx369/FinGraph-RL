#!/usr/bin/env python3
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INFILE = (
    ROOT
    / "outputs/finrl_dkg_original_full/"
    "sec_relation_expansion/"
    "sec_four_relation_sentence_candidates.csv"
)

OUT = (
    ROOT
    / "outputs/finrl_dkg_original_full/"
    "sec_relation_expansion"
)

# High-precision alias set for the initial five-stock
# experiment. Avoid ambiguous one-word mentions
# such as "apple", "amazon", or "alphabet".
ALIASES = {
    "AAPL": [
        r"\bApple Inc\.?\b",
        r"\bApple Incorporated\b",
    ],
    "MSFT": [
        r"\bMicrosoft Corporation\b",
        r"\bMicrosoft Corp\.?\b",
        r"\bMicrosoft\b",
    ],
    "NVDA": [
        r"\bNVIDIA Corporation\b",
        r"\bNVIDIA Corp\.?\b",
        r"\bNVIDIA\b",
    ],
    "AMZN": [
        r"\bAmazon\.com,?\s*Inc\.?\b",
        r"\bAmazon Web Services\b",
    ],
    "GOOGL": [
        r"\bAlphabet Inc\.?\b",
        r"\bGoogle LLC\b",
        r"\bGoogle Inc\.?\b",
        r"\bGoogle Cloud\b",
    ],
}

PATTERNS = {
    ticker: [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]
    for ticker, patterns in ALIASES.items()
}


def identify(text):
    result = []

    for ticker, expressions in PATTERNS.items():
        if any(
            pattern.search(text)
            for pattern in expressions
        ):
            result.append(ticker)

    return result


def main():
    if not INFILE.exists():
        raise RuntimeError(f"Missing: {INFILE}")

    df = pd.read_csv(
        INFILE,
        low_memory=False,
    )

    results = []

    for row in df.itertuples(index=False):
        sentence = str(row.sentence)

        issuer = str(row.ticker).upper().strip()
        mentioned = identify(sentence)

        # The SEC document's issuer is the possible
        # source entity, but being the filer alone
        # does not prove any relation.
        targets = [
            t for t in mentioned
            if t != issuer
        ]

        if issuer not in ALIASES:
            continue

        for target in targets:
            results.append({
                "issuer_ticker": issuer,
                "target_ticker": target,
                "candidate_relation": (
                    row.candidate_relation
                ),
                "sentence": sentence,
                "filing_date": row.filing_date,
                "accession": row.accession,
                "filing_id": row.filing_id,
                "source_line": row.source_line,
                "sentence_index": row.sentence_index,
                "entity_match": "EXPLICIT_NAME",
                "relation_verified": False,
                "direction_verified": False,
                "strict_pit_approved": False,
                "status": "CO_MENTION_ONLY",
            })

    columns = [
        "issuer_ticker",
        "target_ticker",
        "candidate_relation",
        "sentence",
        "filing_date",
        "accession",
        "filing_id",
        "source_line",
        "sentence_index",
        "entity_match",
        "relation_verified",
        "direction_verified",
        "strict_pit_approved",
        "status",
    ]

    out = pd.DataFrame(
        results, columns=columns
    )

    out = out.drop_duplicates(
        subset=[
            "issuer_ticker",
            "target_ticker",
            "candidate_relation",
            "source_line",
            "sentence_index",
        ]
    )

    path = (
        OUT / "sec_named_relation_candidates.csv"
    )

    out.to_csv(path, index=False)

    print("\n=== SEC NAMED RELATION CANDIDATES ===")
    print("Input sentence matches:", len(df))
    print("Named-company co-mentions:", len(out))

    if len(out):
        print("\n=== RELATION COVERAGE ===")
        print(
            out.groupby("candidate_relation")
            .size()
            .to_string()
        )

        print("\n=== COMPANY PAIRS ===")
        print(
            out.groupby([
                "issuer_ticker",
                "target_ticker",
            ])
            .size()
            .sort_values(ascending=False)
            .head(30)
            .to_string()
        )

    print("\nSaved:", path)
    print(
        "Status: Named-company co-mentions, "
        "NOT verified semantic relations."
    )


if __name__ == "__main__":
    main()
