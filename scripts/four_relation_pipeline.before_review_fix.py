#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
OUT = BASE / "four_relation_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

SENTENCES = (
    BASE / "sec_relation_expansion/"
    "sec_four_relation_sentence_candidates.csv"
)

COMPANIES = (
    ROOT / "kg/data/reference/company_tickers.json"
)

MANIFEST = (
    ROOT / "outputs/v20/sec_filings_manifest.csv"
)

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

TARGETS = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
}

NY = ZoneInfo("America/New_York")

STOPWORDS = {
    "the company", "the group", "holdings inc",
    "international inc", "global inc",
    "financial services", "technologies inc",
    "corporation", "company", "industries inc",
}

GENERIC = {
    "our customers", "our suppliers",
    "our competitors", "third parties",
    "other companies",
}

SUPPLY_RE = re.compile(
    r"\b(supplies|supplied|sells|sold|provides|"
    r"provided|manufactures|manufactured|"
    r"delivers|delivered)\b",
    re.I
)

PURCHASE_RE = re.compile(
    r"\b(purchases|purchased|buys|bought|"
    r"procures|procured|sources|sourced)\b",
    re.I
)

COMPETE_RE = re.compile(
    r"\b(competes? with|competing with|"
    r"competitors? (?:include|such as|are)|"
    r"competition from)\b",
    re.I
)

def field(row, *names):
    for name in names:
        if name in row and pd.notna(row[name]):
            value = str(row[name]).strip()
            if value:
                return value
    return ""

def load_companies():
    obj = json.loads(COMPANIES.read_text())
    if isinstance(obj, dict):
        entries = list(obj.values())
    elif isinstance(obj, list):
        entries = obj
    else:
        raise RuntimeError("Unsupported company_tickers.json")

    ticker_to_cik = {}
    aliases = []

    for item in entries:
        if not isinstance(item, dict):
            continue

        tic = str(item.get("ticker", "")).upper().strip()
        title = str(
            item.get("title", item.get("name", ""))
        ).strip()

        cik = item.get("cik_str", item.get("cik", ""))

        if not tic:
            continue

        if str(cik).isdigit():
            ticker_to_cik[tic] = int(cik)

        if len(title) < 7:
            continue

        normalized = title.lower().strip()
        if normalized in STOPWORDS:
            continue

        aliases.append((title, tic))

    # Additional established aliases. They do not
    # independently prove a financial relationship.
    extras = {
        "AAPL": ["Apple Inc.", "Apple Computer Inc."],
        "MSFT": ["Microsoft Corporation", "Microsoft Corp."],
        "NVDA": ["NVIDIA Corporation", "Nvidia Corp."],
        "AMZN": ["Amazon.com Inc.", "Amazon Web Services"],
        "GOOGL": ["Alphabet Inc.", "Google LLC", "Google Inc."],
    }

    for ticker, names in extras.items():
        for name in names:
            aliases.append((name, ticker))

    # Index by the first distinctive word to avoid
    # searching thousands of names separately.
    index = defaultdict(list)
    seen = set()

    for name, ticker in aliases:
        key = (name.casefold(), ticker)

        if key in seen:
            continue
        seen.add(key)

        first = re.search(
            r"[A-Za-z][A-Za-z0-9.-]*",
            name
        )
        if not first:
            continue

        token = first.group().casefold()

        if token in {"the", "a", "an"}:
            continue

        pattern = re.compile(
            r"(?<![A-Za-z0-9])"
            + re.escape(name)
            + r"(?![A-Za-z0-9])",
            re.I,
        )

        index[token].append(
            (pattern, ticker, name)
        )

    return ticker_to_cik, index


def companies_in_sentence(sentence, index):
    tokens = set(
        word.casefold()
        for word in re.findall(
            r"[A-Za-z][A-Za-z0-9.-]*",
            sentence
        )
    )

    found = defaultdict(list)

    for token in tokens:
        for pattern, ticker, name in index.get(
            token, []
        ):
            match = pattern.search(sentence)

            if match:
                found[ticker].append({
                    "start": match.start(),
                    "end": match.end(),
                    "name": name,
                })

    return dict(found)


def classify(issuer, counterparty, sentence, mentions):
    """
    Conservative rule baseline.

    Only explicit verbs between identified company
    mentions can become direction-labeled candidates.
    Otherwise the relation remains unclassified.
    """

    a_mentions = mentions.get(issuer, [])
    b_mentions = mentions.get(counterparty, [])

    if not a_mentions or not b_mentions:
        return None

    best = None

    for a in a_mentions:
        for b in b_mentions:
            if a["end"] <= b["start"]:
                between = sentence[
                    a["end"]:b["start"]
                ]
                head, tail = issuer, counterparty

            elif b["end"] <= a["start"]:
                between = sentence[
                    b["end"]:a["start"]
                ]
                head, tail = counterparty, issuer

            else:
                continue

            if len(between) > 120:
                continue

            # Avoid classifying clauses separated by
            # sentence-ending punctuation.
            if re.search(r"[.;!?]", between):
                continue

            if SUPPLY_RE.search(between):
                best = (
                    head, "SUPPLIES_TO", tail,
                    "EXPLICIT_SUPPLY_VERB"
                )

            elif PURCHASE_RE.search(between):
                best = (
                    head, "CUSTOMER_OF", tail,
                    "EXPLICIT_PURCHASE_VERB"
                )

            elif COMPETE_RE.search(between):
                best = (
                    issuer, "COMPETES_WITH",
                    counterparty,
                    "EXPLICIT_COMPETITION_PHRASE"
                )

            if best:
                return best

    return None



def extract():
    if not SENTENCES.exists():
        raise RuntimeError(f"Missing {SENTENCES}")

    cik_map, index = load_companies()

    df = pd.read_csv(
        SENTENCES,
        low_memory=False,
    )

    candidate_columns = [
        "issuer", "counterparty",
        "filing_date", "accession",
        "source_line", "sentence", "cik",
        "head", "relation", "tail",
        "extraction_method", "review_status",
        "human_verified", "accepted_at_et",
        "available_at_et", "pit_verified",
    ]

    unresolved_columns = [
        "issuer", "counterparty",
        "filing_date", "accession",
        "source_line", "sentence", "cik",
        "reason",
    ]

    candidates = []
    unresolved = []
    seen = set()

    # Explicit entity name or filing-issuer pronoun.
    issuer_pronouns = re.compile(
        r"\b(we|our|us|the company)\b",
        re.I,
    )

    # Constructions requiring human semantic review.
    supply_to = re.compile(
        r"\b(supply|supplies|supplied|"
        r"sell|sells|sold|provide|provides|"
        r"provided|deliver|delivers|delivered)\b",
        re.I,
    )
    buy_from = re.compile(
        r"\b(purchase|purchases|purchased|"
        r"buy|buys|bought|procure|procures|"
        r"procured|source|sources|sourced)\b",
        re.I,
    )
    compete = re.compile(
        r"\b(compete|competes|competing|"
        r"competitor|competitors|competition)\b",
        re.I,
    )

    # Negative or hypothetical wording is not
    # suitable for automatic direction assignment.
    uncertain = re.compile(
        r"\b(not|never|no longer|may|might|"
        r"could|would|potential|possible|"
        r"formerly|previously)\b",
        re.I,
    )

    for row in df.to_dict("records"):

        issuer = field(row, "ticker").upper()
        sentence = field(row, "sentence")

        if issuer not in TARGETS or not sentence:
            continue

        mentions = companies_in_sentence(
            sentence, index
        )

        counterparties = sorted(
            t for t in mentions
            if t != issuer
        )

        if not counterparties:
            continue

        for other in counterparties:
            common = {
                "issuer": issuer,
                "counterparty": other,
                "filing_date": field(
                    row, "filing_date"
                ),
                "accession": field(
                    row, "accession"
                ),
                "source_line": field(
                    row, "source_line"
                ),
                "sentence": sentence,
                "cik": cik_map.get(issuer, ""),
            }

            key_base = (
                issuer, other,
                common["accession"],
                common["source_line"],
                sentence,
            )

            if key_base in seen:
                continue

            seen.add(key_base)

            # Rule 1: try the original explicit
            # company-to-company classifier.
            explicit = classify(
                issuer, other,
                sentence, mentions,
            )

            if explicit:
                head, relation, tail, method = explicit
            else:
                head = relation = tail = method = None

            # Rule 2: filing-company pronoun
            # followed by an explicit business verb
            # and a named counterparty.
            #
            # Important: only direction-CANDIDATE.
            # No relation becomes VERIFIED here.
            if relation is None and not uncertain.search(
                sentence
            ):
                occurrences = mentions[other]

                for occurrence in occurrences:
                    before = sentence[
                        max(0, occurrence["start"] - 180):
                        occurrence["start"]
                    ]

                    pronoun_matches = list(
                        issuer_pronouns.finditer(before)
                    )

                    if not pronoun_matches:
                        continue

                    context = before[
                        pronoun_matches[-1].start():
                    ]

                    # Limit to one clause.
                    if re.search(r"[.;!?]", context):
                        continue

                    if supply_to.search(context):
                        head = issuer
                        relation = "SUPPLIES_TO"
                        tail = other
                        method = "ISSUER_PRONOUN_SUPPLY"
                        break

                    if buy_from.search(context):
                        head = issuer
                        relation = "CUSTOMER_OF"
                        tail = other
                        method = "ISSUER_PRONOUN_PURCHASE"
                        break

                    if compete.search(context):
                        head = issuer
                        relation = "COMPETES_WITH"
                        tail = other
                        method = "ISSUER_PRONOUN_COMPETITION"
                        break

            if relation is None:
                unresolved.append({
                    **common,
                    "reason": "CO_MENTION_NO_DIRECTION",
                })
                continue

            candidates.append({
                **common,
                "head": head,
                "relation": relation,
                "tail": tail,
                "extraction_method": method,
                "review_status": "AUTO_CANDIDATE",
                "human_verified": False,
                "accepted_at_et": "",
                "available_at_et": "",
                "pit_verified": False,
            })

    output = pd.DataFrame(
        candidates,
        columns=candidate_columns,
    )

    unresolved_df = pd.DataFrame(
        unresolved,
        columns=unresolved_columns,
    )

    output.to_csv(
        OUT / "relation_candidates.csv",
        index=False,
    )

    unresolved_df.to_csv(
        OUT / "unclassified_mention_pairs.csv",
        index=False,
    )

    print("\n=== MULTI-COMPANY EXTRACTION V2 ===")
    print("Input sentences:", len(df))
    print("Directed candidates:", len(output))
    print("Unclassified pairs:", len(unresolved_df))

    if len(output):
        print("\n=== RELATION COUNTS ===")
        print(
            output.groupby(
                ["relation", "extraction_method"]
            ).size().to_string()
        )

    print("\nOutput:", OUT / "relation_candidates.csv")
    print(
        "STATUS: AUTO_CANDIDATE, not verified. "
        "Negation, entity ambiguity, and semantic "
        "direction require original-source review."
    )

def acceptance_header(cik, accession, agent):
    accession = accession.strip()
    bare = accession.replace("-", "")

    if not (
        str(cik).isdigit()
        and re.fullmatch(r"\d{18}", bare)
    ):
        raise ValueError("Invalid CIK/accession")

    url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{bare}/{accession}"
        "-index-headers.html"
    )

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": agent,
            "Accept": "text/html",
        },
    )

    with urllib.request.urlopen(
        req, timeout=25
    ) as resp:
        data = resp.read().decode(
            "utf-8", errors="replace"
        )

    # SEC filing headers use local Eastern time.
    match = re.search(
        r"ACCEPTANCE-DATETIME"
        r"\s*[:>\s]\s*(\d{14})",
        data,
        re.I,
    )

    if not match:
        # Do not guess from filing date.
        raise ValueError(
            "ACCEPTANCE-DATETIME not located"
        )

    value = datetime.strptime(
        match.group(1), "%Y%m%d%H%M%S"
    ).replace(tzinfo=NY)

    return value, url


def pit():
    source = OUT / "relation_candidates.csv"

    if not source.exists():
        raise RuntimeError("Run extract first")

    agent = os.environ.get(
        "SEC_USER_AGENT", ""
    ).strip()

    if (
        "@" not in agent
        or "example.com" in agent.lower()
    ):
        raise RuntimeError(
            "Set SEC_USER_AGENT to your real "
            "research contact name and email."
        )

    df = pd.read_csv(
        source, dtype=str
    ).fillna("")

    cache_path = OUT / "sec_acceptance_cache.json"

    cache = (
        json.loads(cache_path.read_text())
        if cache_path.exists() else {}
    )

    failures = 0

    for i, row in df.iterrows():
        accession = row["accession"]
        cik = row["cik"]

        if not accession or not cik:
            failures += 1
            continue

        key = f"{cik}:{accession}"

        if key not in cache:
            try:
                accepted, url = acceptance_header(
                    cik, accession, agent
                )

                cache[key] = {
                    "accepted_at_et": (
                        accepted.isoformat()
                    ),
                    "source_url": url,
                    "status": "HEADER_CONFIRMED",
                }

            except Exception as exc:
                cache[key] = {
                    "accepted_at_et": "",
                    "source_url": "",
                    "status": (
                        "UNVERIFIED: "
                        + str(exc)[:180]
                    ),
                }

            cache_path.write_text(
                json.dumps(cache, indent=2)
            )

            # Stay well below SEC rate limits.
            time.sleep(0.25)

        info = cache[key]

        if info["status"] != "HEADER_CONFIRMED":
            failures += 1
            continue

        df.at[
            i, "accepted_at_et"
        ] = info["accepted_at_et"]

        # Conservative next-market-day policy:
        # never use a same-day filing in the
        # day's trading observation.
        accepted = pd.Timestamp(
            info["accepted_at_et"]
        )

        next_day = (
            accepted.tz_convert(NY)
            .normalize()
            + pd.Timedelta(days=1)
        )

        # Weekend / holiday handling still needs
        # exchange-session calendar alignment.
        df.at[
            i, "available_at_et"
        ] = next_day.isoformat()

        df.at[
            i, "source_url"
        ] = info["source_url"]

        df.at[
            i, "pit_verified"
        ] = "HEADER_CONFIRMED_DATE_ONLY"

    df.to_csv(
        OUT / "relation_candidates_with_sec_time.csv",
        index=False,
    )

    print("\n=== SEC PIT MATCHING ===")
    print("Candidates:", len(df))
    print(
        "Header-confirmed:",
        len(df) - failures
    )
    print("Unconfirmed:", failures)
    print(
        "IMPORTANT: timestamp verification "
        "does not verify relation semantics."
    )


def prepare_review():
    source = OUT / "relation_candidates_with_sec_time.csv"

    if not source.exists():
        raise RuntimeError("Run pit first")

    df = pd.read_csv(
        source, dtype=str
    ).fillna("")

    review = df[
        [
            "head", "relation", "tail",
            "accession", "accepted_at_et",
            "available_at_et", "source_url",
            "sentence", "extraction_method",
            "pit_verified",
        ]
    ].copy()

    review["review_status"] = "PENDING"
    review["reviewer_notes"] = ""

    path = OUT / "relation_review.csv"

    if not path.exists():
        review.to_csv(path, index=False)
    else:
        print(
            "Existing review preserved:",
            path
        )

    print("Review file:", path)
    print(
        "Only set review_status=VERIFIED "
        "after checking source, company identities, "
        "direction and negation."
    )


def summary():
    print("\n=== FOUR-RELATION PIPELINE ===")

    for name in [
        "relation_candidates.csv",
        "relation_candidates_with_sec_time.csv",
        "relation_review.csv",
    ]:
        path = OUT / name

        if not path.exists():
            print(name, "MISSING")
            continue

        df = pd.read_csv(path)

        print(name, len(df))

        if (
            name == "relation_review.csv"
            and len(df)
        ):
            confirmed = (
                df["review_status"]
                .astype(str).str.upper()
                .eq("VERIFIED")
            )

            print(
                "Semantic VERIFIED:",
                int(confirmed.sum())
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "task",
        choices=[
            "extract", "pit",
            "review", "summary",
        ]
    )
    args = p.parse_args()

    {
        "extract": extract,
        "pit": pit,
        "review": prepare_review,
        "summary": summary,
    }[args.task]()


if __name__ == "__main__":
    main()
