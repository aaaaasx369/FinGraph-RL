#!/usr/bin/env python3
import json
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full/sec_relation_expansion/historical"

SOURCE = BASE / "historical_named_company_candidates.csv"
COMPANY_FILE = ROOT / "kg/data/reference/company_tickers.json"

TARGET_CIK = {
    "AAPL": "320193",
    "MSFT": "789019",
    "NVDA": "1045810",
    "AMZN": "1018724",
    "GOOGL": "1652044",
}

PATTERNS = {
    "SUPPLY_LANGUAGE": re.compile(
        r"\b(supply|supplies|supplied|supplier|"
        r"suppliers|manufactures?|manufactured|"
        r"provides?|provided|sells?|sold)\b", re.I
    ),
    "PURCHASE_LANGUAGE": re.compile(
        r"\b(purchas(?:e|es|ed|ing)|"
        r"procure[sd]?|sourc(?:e|es|ed|ing)|"
        r"buy|buys|bought|customers?|clients?)\b", re.I
    ),
    "COMPETITION_LANGUAGE": re.compile(
        r"\b(compete|competes|competing|"
        r"competitor|competitors|competition|rivals?)\b",
        re.I
    ),
}

UNCERTAIN = re.compile(
    r"\b(may|might|could|potential|hypothetical|"
    r"not|never|formerly|previously)\b", re.I
)


def canonical_cik(value):
    value = str(value or "").strip()

    if value.endswith(".0"):
        value = value[:-2]

    return str(int(value)) if value.isdigit() else ""


def load_companies():
    raw = json.loads(COMPANY_FILE.read_text())

    entries = raw.values() if isinstance(raw, dict) else raw

    ticker_ciks = defaultdict(set)
    cik_tickers = defaultdict(set)

    for item in entries:
        if not isinstance(item, dict):
            continue

        ticker = str(item.get("ticker", "")).upper().strip()
        cik = canonical_cik(
            item.get("cik_str", item.get("cik", ""))
        )

        if ticker and cik:
            ticker_ciks[ticker].add(cik)
            cik_tickers[cik].add(ticker)

    return ticker_ciks, cik_tickers


def main():
    df = pd.read_csv(SOURCE, dtype=str).fillna("")
    ticker_ciks, cik_tickers = load_companies()

    output = []

    for _, r in df.iterrows():
        issuer = r["issuer"].strip().upper()
        counterparty = r["counterparty"].strip().upper()

        source_cik = TARGET_CIK.get(issuer, "")
        other_ciks = ticker_ciks.get(counterparty, set())

        if len(other_ciks) != 1:
            status = "AMBIGUOUS_OR_MISSING_CIK"
            other_cik = ""
        else:
            other_cik = next(iter(other_ciks))

            if source_cik == other_cik:
                status = "SAME_ISSUER_EXCLUDED"
            else:
                status = "DIFFERENT_CIK_CANDIDATE"

        sentence = r["sentence"]

        signals = [
            name
            for name, pattern in PATTERNS.items()
            if pattern.search(sentence)
        ]

        if not signals:
            priority = "LOW"
        elif UNCERTAIN.search(sentence):
            priority = "REVIEW_UNCERTAIN"
        else:
            priority = "REVIEW_HIGH"

        record = r.to_dict()

        record.update({
            "issuer_cik": source_cik,
            "counterparty_cik": other_cik,
            "entity_status": status,
            "canonical_entity_id": (
                "CIK:" + other_cik if other_cik else ""
            ),
            "relation_language_signals": ";".join(signals),
            "review_priority": priority,
            "semantic_verified": False,
            "relationship_direction_verified": False,
            "strict_pit_approved": False,
        })

        output.append(record)

    result = pd.DataFrame(output)

    # Same source evidence / same company / same sentence
    # should not be counted once per share-class ticker.
    eligible = result[
        result["entity_status"].eq(
            "DIFFERENT_CIK_CANDIDATE"
        )
    ].copy()

    eligible = eligible.drop_duplicates(
        subset=[
            "issuer_cik",
            "counterparty_cik",
            "accession",
            "source_line",
            "sentence",
        ]
    )

    excluded = result[
        ~result["entity_status"].eq(
            "DIFFERENT_CIK_CANDIDATE"
        )
    ].copy()

    result.to_csv(
        BASE / "company_entity_resolution_audit.csv",
        index=False,
    )

    eligible.to_csv(
        BASE / "company_relation_review_queue.csv",
        index=False,
    )

    excluded.to_csv(
        BASE / "company_entity_quarantine.csv",
        index=False,
    )

    print("\n=== CIK ENTITY RESOLUTION ===")
    print("Original co-mentions:", len(result))

    print("\nEntity status:")
    print(result["entity_status"].value_counts().to_string())

    print("\n=== REVIEW QUEUE ===")
    print("Different-CIK candidate rows:", len(eligible))
    print(
        "Distinct company pairs:",
        eligible[
            ["issuer_cik", "counterparty_cik"]
        ].drop_duplicates().shape[0],
    )

    if len(eligible):
        print("\nReview priority:")
        print(
            eligible["review_priority"]
            .value_counts().to_string()
        )

        print("\nCompany pairs:")
        print(
            eligible.groupby(
                ["issuer", "counterparty", "counterparty_cik"]
            ).size()
            .sort_values(ascending=False)
            .to_string()
        )

    print("\nVerified semantic relations: 0")
    print("Strict PIT: NOT CERTIFIED")
    print("\nOutput:", BASE)


if __name__ == "__main__":
    main()
