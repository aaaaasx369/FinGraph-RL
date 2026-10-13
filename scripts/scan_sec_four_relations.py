#!/usr/bin/env python3
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "rag/data/processed/sec/sec_chunks.jsonl"
)

MANIFEST = (
    ROOT / "outputs/v20/sec_filings_manifest.csv"
)

OUT = (
    ROOT
    / "outputs/finrl_dkg_original_full/sec_relation_expansion"
)

OUT.mkdir(parents=True, exist_ok=True)

PATTERNS = {
    "SUPPLIES_TO": [
        r"\bsuppl(?:y|ies|ied|ier|iers)\b",
        r"\bcomponent suppliers?\b",
        r"\bmanufactur(?:e|er|ers|ing)\b",
        r"\bvendors?\b",
    ],
    "CUSTOMER_OF": [
        r"\bcustomers?\b",
        r"\bmajor customers?\b",
        r"\bpurchas(?:e|es|ed|ing)\b",
        r"\bclients?\b",
    ],
    "COMPETES_WITH": [
        r"\bcompet(?:e|es|ed|ing|ition|itor|itors)\b",
        r"\bcompetitive\b",
        r"\brivals?\b",
    ],
    "SAME_SECTOR": [
        r"\bindustr(?:y|ies)\b",
        r"\bsector\b",
        r"\bSIC\b",
    ],
}

COMPILED = {
    relation: [
        re.compile(pattern, re.IGNORECASE)
        for pattern in patterns
    ]
    for relation, patterns in PATTERNS.items()
}

TEXT_KEYS = [
    "text",
    "chunk_text",
    "content",
    "page_content",
    "sentence",
]

DATE_KEYS = [
    "filing_date",
    "filed_at",
    "date",
    "available_from",
]

TICKER_KEYS = [
    "ticker",
    "tic",
    "symbol",
    "company_ticker",
]


def find_field(obj, keys):
    if not isinstance(obj, dict):
        return None

    for key in keys:
        value = obj.get(key)

        if value is not None and value != "":
            return value

    for nested in ["metadata", "meta", "filing"]:
        value = obj.get(nested)

        if isinstance(value, dict):
            found = find_field(value, keys)
            if found is not None:
                return found

    return None


def stringify(value):
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        return str(value)

    return ""


def sentences(text):
    # Avoid splitting on abbreviations in a way
    # that destroys the original sentence entirely.
    return [
        piece.strip()
        for piece in re.split(
            r"(?<=[.!?])\s+|\n+",
            text,
        )
        if piece.strip()
    ]


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Missing SEC chunks: {SOURCE}")

    print("=== SEC FOUR-RELATION SCAN ===")
    print("Source:", SOURCE)

    total = 0
    invalid_json = 0
    missing_text = 0
    sentence_hits = Counter()
    rows = []
    examples = []
    fields_seen = Counter()

    with SOURCE.open(
        "r", encoding="utf-8", errors="replace"
    ) as handle:
        for line_number, line in enumerate(
            handle, start=1
        ):
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue

            if not isinstance(obj, dict):
                continue

            total += 1
            fields_seen.update(obj.keys())

            raw = find_field(obj, TEXT_KEYS)
            text = stringify(raw)

            if not text.strip():
                missing_text += 1
                if len(examples) < 3:
                    examples.append({
                        "line": line_number,
                        "keys": list(obj.keys()),
                    })
                continue

            filing_date = stringify(
                find_field(obj, DATE_KEYS)
            )

            ticker = stringify(
                find_field(obj, TICKER_KEYS)
            ).upper().strip()

            accession = stringify(
                find_field(
                    obj,
                    [
                        "accession",
                        "accession_number",
                        "filing_accession",
                    ],
                )
            )

            filing_id = stringify(
                find_field(
                    obj,
                    [
                        "doc_id",
                        "document_id",
                        "filing_id",
                        "id",
                    ],
                )
            )

            parsed_date = pd.to_datetime(
                filing_date,
                errors="coerce",
                utc=True,
            )

            in_scope = (
                pd.notna(parsed_date)
                and 2018 <= parsed_date.year <= 2023
            )

            # Keep records with missing dates as candidates.
            # They cannot be PIT-certified yet.
            if pd.notna(parsed_date) and not in_scope:
                continue

            for sentence_index, sentence in enumerate(
                sentences(text)
            ):
                if not (25 <= len(sentence) <= 2500):
                    continue

                matched = [
                    relation
                    for relation, expressions
                    in COMPILED.items()
                    if any(
                        expression.search(sentence)
                        for expression in expressions
                    )
                ]

                for relation in matched:
                    sentence_hits[relation] += 1

                    rows.append({
                        "source_line": line_number,
                        "sentence_index": sentence_index,
                        "ticker": ticker,
                        "filing_date": filing_date,
                        "accession": accession,
                        "filing_id": filing_id,
                        "candidate_relation": relation,
                        "sentence": sentence,
                        "source_file": str(
                            SOURCE.relative_to(ROOT)
                        ),
                        "date_in_scope": in_scope,
                        "human_verified": False,
                        "strict_pit_approved": False,
                        "status": "KEYWORD_CANDIDATE",
                    })

            if total % 3000 == 0:
                print(
                    f"Scanned {total:,} chunks",
                    flush=True,
                )

    columns = [
        "source_line",
        "sentence_index",
        "ticker",
        "filing_date",
        "accession",
        "filing_id",
        "candidate_relation",
        "sentence",
        "source_file",
        "date_in_scope",
        "human_verified",
        "strict_pit_approved",
        "status",
    ]

    df = pd.DataFrame(rows, columns=columns)

    # Do not lose source-level occurrences; retain
    # each sentence's original chunk location.
    df.to_csv(
        OUT / "sec_four_relation_sentence_candidates.csv",
        index=False,
    )

    if len(df):
        counts = (
            df.groupby(
                ["candidate_relation", "ticker"],
                dropna=False,
            )
            .size()
            .reset_index(name="sentence_count")
        )
    else:
        counts = pd.DataFrame(columns=[
            "candidate_relation",
            "ticker",
            "sentence_count",
        ])

    counts.to_csv(
        OUT / "sec_relation_ticker_counts.csv",
        index=False,
    )

    report = {
        "source": str(SOURCE),
        "chunks_read": total,
        "invalid_json_lines": invalid_json,
        "missing_text_chunks": missing_text,
        "sentence_candidates": len(df),
        "relation_hits": dict(sentence_hits),
        "source_top_level_fields": dict(
            fields_seen.most_common(30)
        ),
        "examples_missing_text": examples,
        "candidate_status": "NOT_VERIFIED",
        "strict_pit_certified": False,
    }

    (
        OUT / "sec_scan_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== SCAN SUMMARY ===")
    print("Chunks:", total)
    print("Missing text:", missing_text)
    print("Invalid JSON:", invalid_json)
    print("Candidate sentences:", len(df))

    for relation in PATTERNS:
        print(
            f"{relation:16s}",
            sentence_hits[relation],
        )

    print("\n=== TICKER COVERAGE ===")
    print(
        counts.to_string(index=False)
        if len(counts) else "No matches"
    )

    print("\nSAVED:", OUT)
    print(
        "STATUS: Candidate extraction only. "
        "Relation direction, companies and actual "
        "publication timestamps are not verified."
    )


if __name__ == "__main__":
    main()
