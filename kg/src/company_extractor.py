from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


# ============================================================
# Relationship language
# ============================================================

RELATION_PATTERNS = [
    # supplier
    r"\bsupplier(?:s)?\b",
    r"\bsupplies\b",
    r"\bsupply\b",
    r"\bmanufactur(?:e|es|ed|er|ers|ing)\b",
    r"\bfoundry\b",
    r"\brely(?:ies|ied)? on\b",
    r"\bdepend(?:s|ed)? on\b",
    r"\bsource(?:d)? from\b",
    r"\bprocure(?:d)? from\b",
    r"\bpurchase(?:d|s)? from\b",
    r"\bcomponents? from\b",

    # customer
    r"\bcustomer(?:s)?\b",
    r"\bmajor customer\b",
    r"\blargest customer\b",
    r"\bsales to\b",
    r"\bsold to\b",
    r"\brevenue from\b",

    # competitor
    r"\bcompetitor(?:s)?\b",
    r"\bcompete(?:s|d)? with\b",
    r"\bcompetition\b",
]


# ============================================================
# Unsafe / generic aliases
# ============================================================

GENERIC_SINGLE_WORDS = {
    "seek",
    "team",
    "paid",
    "popular",
    "post",
    "strategy",
    "pattern",
    "frequency",
    "interface",
    "senior",
    "people",
    "southern",
    "eastern",
    "frontline",
    "enhanced",
    "emerging",
    "track",
    "match",
    "freedom",
    "sound",
    "equitable",
    "reliability",
    "intelligent",
    "innovation",
    "innovate",
    "company",
    "group",
    "holding",
    "holdings",
    "systems",
    "services",
    "technology",
    "technologies",
    "financial",
    "institutions",
}


# Safe famous/unique single-word company aliases.
SAFE_SINGLE_WORD_ALIASES = {
    "adobe",
    "salesforce",
    "broadcom",
    "qualcomm",
    "nvidia",
    "microsoft",
    "google",
    "amazon",
    "apple",
    "netflix",
    "intel",
    "amd",
    "oracle",
    "cisco",
    "paypal",
    "spotify",
    "tesla",
    "uber",
}


LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    r"incorporated|inc\.?|"
    r"corporation|corp\.?|"
    r"company|co\.?|"
    r"limited|ltd\.?|"
    r"plc|"
    r"holdings?|"
    r"group"
    r")\b",
    re.I,
)


# ============================================================
# Helpers
# ============================================================

def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def relation_chunk(text: str) -> bool:
    text = text.lower()

    return any(
        re.search(pattern, text, re.I)
        for pattern in RELATION_PATTERNS
    )


def normalize_for_matching(name: str) -> str:
    name = clean_text(name).lower()

    name = name.replace("&", " and ")

    name = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        name,
    )

    name = re.sub(
        r"\s+",
        " ",
        name,
    ).strip()

    return name


def strip_legal_suffixes(name: str) -> str:
    name = normalize_for_matching(name)

    name = LEGAL_SUFFIX_RE.sub(
        " ",
        name,
    )

    return re.sub(
        r"\s+",
        " ",
        name,
    ).strip()


def valid_alias(alias: str) -> bool:
    alias = normalize_for_matching(alias)

    if not alias:
        return False

    tokens = alias.split()

    # Single word aliases are dangerous.
    if len(tokens) == 1:
        word = tokens[0]

        if word in SAFE_SINGLE_WORD_ALIASES:
            return True

        return False

    # Require multiword aliases to have reasonable length.
    if len(alias) < 6:
        return False

    # Reject aliases composed mostly of generic words.
    if all(
        token in GENERIC_SINGLE_WORDS
        for token in tokens
    ):
        return False

    return True


def make_aliases(
    title: str,
) -> List[str]:

    aliases = set()

    full = normalize_for_matching(title)

    # Full SEC title is safest.
    if len(full) >= 5:
        aliases.add(full)

    stripped = strip_legal_suffixes(title)

    # Add legal-suffix-stripped alias only when safe.
    if (
        stripped != full
        and valid_alias(stripped)
    ):
        aliases.add(stripped)

    return sorted(
        aliases,
        key=len,
        reverse=True,
    )


def company_pattern(alias: str) -> re.Pattern:

    parts = alias.split()

    escaped = r"\s+".join(
        re.escape(part)
        for part in parts
    )

    return re.compile(
        rf"(?<![A-Za-z0-9])"
        rf"{escaped}"
        rf"(?![A-Za-z0-9])",
        re.I,
    )


def get_context(
    text: str,
    start: int,
    end: int,
    radius: int = 250,
) -> str:

    left = max(
        0,
        start - radius,
    )

    right = min(
        len(text),
        end + radius,
    )

    return clean_text(
        text[left:right]
    )


# ============================================================
# SEC company master
# ============================================================

def load_sec_companies(
    path: str,
) -> List[Dict]:

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        raw = json.load(f)

    # Dedupe by CIK:
    # one company can have multiple securities/tickers.
    by_cik = {}

    for item in raw.values():

        ticker = clean_text(
            item.get("ticker", "")
        ).upper()

        title = clean_text(
            item.get("title", "")
        )

        cik = str(
            item.get("cik_str", "")
        ).strip()

        if not cik or not title:
            continue

        if cik not in by_cik:

            by_cik[cik] = {
                "cik": cik,
                "company_name": title,
                "tickers": set(),
            }

        if ticker:
            by_cik[cik]["tickers"].add(
                ticker
            )

    companies = []

    for company in by_cik.values():

        aliases = make_aliases(
            company["company_name"]
        )

        if not aliases:
            continue

        companies.append(
            {
                "cik":
                    company["cik"],

                "company_name":
                    company["company_name"],

                "tickers":
                    sorted(
                        company["tickers"]
                    ),

                "aliases":
                    aliases,
            }
        )

    return companies


# ============================================================
# Alias index
# ============================================================

def build_index(
    companies: List[Dict],
):

    index = defaultdict(list)

    for company in companies:

        for alias in company["aliases"]:

            tokens = alias.split()

            if not tokens:
                continue

            first = tokens[0]

            index[first].append(
                (
                    alias,
                    company,
                    company_pattern(alias),
                )
            )

    return index


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--sec-companies",
        default=(
            "kg/data/reference/"
            "company_tickers.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "kg/data/"
            "company_candidates_sec.csv"
        ),
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    print(
        "Loading SEC company dictionary..."
    )

    companies = load_sec_companies(
        args.sec_companies
    )

    print(
        f"Unique SEC companies by CIK: "
        f"{len(companies)}"
    )

    company_index = build_index(
        companies
    )

    counts = Counter()

    filing_tickers = defaultdict(set)
    evidence = defaultdict(list)
    chunks_seen = defaultdict(list)

    total = 0
    relation_total = 0

    with open(
        args.chunks,
        "r",
        encoding="utf-8",
    ) as f:

        for i, line in enumerate(f):

            if (
                args.max_chunks is not None
                and i >= args.max_chunks
            ):
                break

            row = json.loads(line)

            total += 1

            text = clean_text(
                row.get("text", "")
            )

            if not text:
                continue

            if not relation_chunk(text):
                continue

            relation_total += 1

            filing_ticker = clean_text(
                row.get("ticker", "")
            ).upper()

            chunk_id = clean_text(
                row.get("chunk_id", "")
            )

            text_lower = (
                normalize_for_matching(text)
            )

            words = set(
                re.findall(
                    r"\b[a-z0-9]+\b",
                    text_lower,
                )
            )

            matched_ciks = set()

            for first_word in words:

                candidates = (
                    company_index.get(
                        first_word,
                        []
                    )
                )

                for (
                    alias,
                    company,
                    pattern,
                ) in candidates:

                    # Skip if filing company ticker
                    # belongs to this same CIK.
                    if (
                        filing_ticker
                        in company["tickers"]
                    ):
                        continue

                    match = pattern.search(
                        text_lower
                    )

                    if not match:
                        continue

                    cik = company["cik"]

                    if cik in matched_ciks:
                        continue

                    context = get_context(
                        text_lower,
                        match.start(),
                        match.end(),
                    )

                    if not relation_chunk(
                        context
                    ):
                        continue

                    matched_ciks.add(
                        cik
                    )

                    counts[cik] += 1

                    filing_tickers[cik].add(
                        filing_ticker
                    )

                    if (
                        len(
                            chunks_seen[cik]
                        )
                        < 5
                    ):
                        chunks_seen[
                            cik
                        ].append(
                            chunk_id
                        )

                    if (
                        len(evidence[cik])
                        < 3
                    ):
                        evidence[cik].append(
                            context
                        )

            if (
                relation_total > 0
                and
                relation_total % 500 == 0
            ):
                print(
                    f"relation chunks="
                    f"{relation_total}, "
                    f"unique companies="
                    f"{len(counts)}"
                )

    company_lookup = {
        company["cik"]: company
        for company in companies
    }

    rows = []

    for cik, count in (
        counts.most_common()
    ):

        if count < args.min_count:
            continue

        company = company_lookup[cik]

        rows.append(
            {
                "cik":
                    cik,

                "company_name":
                    company[
                        "company_name"
                    ],

                "tickers":
                    "|".join(
                        company["tickers"]
                    ),

                "mention_count":
                    count,

                "filing_tickers":
                    "|".join(
                        sorted(
                            filing_tickers[
                                cik
                            ]
                        )
                    ),

                "example_chunks":
                    "|".join(
                        chunks_seen[cik]
                    ),

                "evidence_example":
                    (
                        evidence[cik][0]
                        if evidence[cik]
                        else ""
                    ),
            }
        )

    import pandas as pd

    df = pd.DataFrame(
        rows
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print(
        "STRICT SEC COMPANY DISCOVERY"
    )
    print("=" * 70)

    print(
        f"Chunks read: {total}"
    )

    print(
        f"Relation-bearing chunks: "
        f"{relation_total}"
    )

    print(
        f"Matched unique companies: "
        f"{len(df)}"
    )

    if len(df):

        print()

        print(
            df[
                [
                    "cik",
                    "company_name",
                    "tickers",
                    "mention_count",
                    "filing_tickers",
                ]
            ]
            .head(50)
            .to_string(
                index=False
            )
        )

    print()

    print(
        f"Saved: {output}"
    )


if __name__ == "__main__":
    main()
