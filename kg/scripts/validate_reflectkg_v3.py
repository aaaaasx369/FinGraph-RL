#!/usr/bin/env python3
import csv
import json
import re
import argparse
from pathlib import Path
from collections import Counter

ALIASES = {
    "MSFT": ["Microsoft", "Microsoft Corporation"],
    "AAPL": ["Apple", "Apple Inc."],
    "NVDA": ["NVIDIA", "NVIDIA Corporation"],
    "GOOGL": ["Google", "Alphabet", "Alphabet Inc."],
    "GOOG": ["Google", "Alphabet"],
    "AMZN": ["Amazon", "Amazon.com"],
    "TSM": ["TSMC", "Taiwan Semiconductor Manufacturing Company"],
    "FLEX": ["Flex Ltd.", "Flex Ltd", "Flextronics"],
    "FN": ["Fabrinet"],
    "DELL": ["Dell", "Dell Technologies"],
    "INTC": ["Intel", "Intel Corporation"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "TSLA": ["Tesla", "Tesla Inc."],
    "HPE": ["Hewlett Packard Enterprise"],
}

AMBIGUOUS = [
    r"\bcurrent or potential\b",
    r"\bpotential (?:partners?|customers?|suppliers?)\b",
    r"\bmany of which\b",
    r"\bpartners? and customers?\b",
]

LICENSE = [
    r"\bpatent licens(?:e|es|ing)\b",
    r"\blicens(?:e|es|ing) agreement\b",
    r"\blicensing revenue\b",
    r"\broyalt(?:y|ies)\b",
]

COMPETITION = [
    r"\bcompet(?:e|es|ed|ing|ition|itive|itor|itors)\b",
    r"\brivals?\b",
]

SUPPLY = [
    r"\bdepend(?:ent|s|ing)? on\b.{0,120}\bfoundr(?:y|ies)\b",
    r"\butiliz(?:e|es|ing)\b.{0,120}\bsuppliers?\b",
    r"\butiliz(?:e|es|ing)\b.{0,120}\bfoundr(?:y|ies)\b",
    r"\buse\b.{0,120}\bcontract manufacturers?\b",
    r"\butiliz(?:e|es|ing)\b.{0,120}\bcontract manufacturers?\b",
    r"\butiliz(?:e|es|ing)\b.{0,120}\bcontract manufacturers?,?\s+or CMs\b",
    r"\bdepend(?:ent|s|ing)? on\b.{0,120}\bsuppliers?\b",
    r"\buse\b.{0,120}\bsuppliers?\b",
]

CUSTOMER = [
    r"\bsales to\b",
    r"\bsold to\b",
    r"\brevenue from (?:sales to|customers? such as)\b",
    r"\b(?:is|are) (?:one of )?our customers?\b",
]

def norm(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def hit(text, patterns):
    return any(re.search(p, text, re.I | re.S) for p in patterns)

def mentioned(text, ticker, display=""):
    names = ALIASES.get(ticker, []) + [display]

    for name in names:
        name = norm(name)
        if len(name) < 3:
            continue

        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        if re.search(pattern, text, re.I):
            return True

    return False

def filing_ticker(row):
    ticker = norm(row.get("filing_ticker")).upper()
    if ticker:
        return ticker

    doc = norm(row.get("doc_id"))
    m = re.match(r"SEC_([A-Z0-9]+)_", doc)
    return m.group(1) if m else ""

def decide(row):
    relation = norm(row.get("relation")).upper()
    subject = norm(row.get("source")).upper()
    obj = norm(row.get("target")).upper()
    filing = filing_ticker(row)
    evidence = norm(row.get("evidence"))

    # V2 already joined evidence to the original SEC chunk.
    # Do not promote rows that failed source/date grounding.
    if row.get("date_semantics") != "sec_published_at_matched":
        return "REVIEW", "unverified_publication_or_chunk"

    if not norm(row.get("published_at")):
        return "REVIEW", "missing_published_at"

    if not evidence or subject == obj:
        return "REJECT_AS_PROPOSED", "missing_evidence_or_self_relation"

    if hit(evidence, AMBIGUOUS):
        return "REJECT_AS_PROPOSED", "ambiguous_roles"

    subject_present = mentioned(
        evidence, subject, row.get("source_name", "")
    )
    object_present = mentioned(
        evidence, obj, row.get("target_name", "")
    )

    if relation == "COMPETES_WITH":
        # Filing company's products can be referred to as
        # "we", "our products", "Windows", "Xbox", etc.
        other = obj if subject == filing else subject
        other_name = (
            row.get("target_name", "")
            if subject == filing else row.get("source_name", "")
        )

        if filing not in (subject, obj):
            return "REVIEW", "competition_not_filing_party"

        if not mentioned(evidence, other, other_name):
            return "REVIEW", "competitor_not_named"

        if hit(evidence, COMPETITION):
            return "ACCEPT", "explicit_competition_context"

        return "REVIEW", "competition_phrase_missing"

    if relation == "SUPPLIES_TO":
        # supplier -> filing company
        if obj != filing:
            return "REVIEW", "supply_direction_unverified"

        if not subject_present:
            return "REVIEW", "supplier_not_named"

        if hit(evidence, SUPPLY):
            return "ACCEPT", "explicit_supply_or_manufacturing"

        return "REVIEW", "no_explicit_supply_relation"

    if relation == "CUSTOMER_OF":
        # customer -> filing company
        if obj != filing:
            return "REVIEW", "customer_direction_unverified"

        if not subject_present:
            return "REVIEW", "customer_not_named"

        # Patent-license income does not automatically prove
        # a product/service purchasing relationship.
        if hit(evidence, LICENSE):
            return "REVIEW", "license_revenue_not_product_sales"

        if hit(evidence, CUSTOMER):
            return "ACCEPT", "explicit_customer_sales"

        return "REVIEW", "no_explicit_customer_relation"

    return "REVIEW", "unsupported_relation"

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def run(input_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    reviewed = []

    for row in rows:
        row = dict(row)
        decision, reason = decide(row)

        row["v3_decision"] = decision
        row["v3_reason"] = reason
        reviewed.append(row)

    accepted = [
        r for r in reviewed
        if r["v3_decision"] == "ACCEPT"
    ]

    manual = [
        r for r in reviewed
        if r["v3_decision"] != "ACCEPT"
    ]

    # One relation assertion event per relation/document.
    accepted.sort(key=lambda r: (
        r["published_at"],
        r.get("doc_id", ""),
        r.get("resolved_chunk_id", ""),
    ))

    seen_relations = set()
    seen_documents = set()
    events = []

    for r in accepted:
        key = (
            r["source"],
            r["relation"],
            r["target"]
        )
        doc_key = key + (r.get("doc_id", ""),)

        if doc_key in seen_documents:
            continue

        seen_documents.add(doc_key)

        event_type = (
            "REASSERTED"
            if key in seen_relations
            else "FIRST_OBSERVED"
        )
        seen_relations.add(key)

        events.append({
            "event_type": event_type,
            "subject": r["source"],
            "relation": r["relation"],
            "object": r["target"],
            "published_at": r["published_at"],
            "doc_id": r.get("doc_id", ""),
            "chunk_id": (
                r.get("resolved_chunk_id")
                or r.get("chunk_id", "")
            ),
            "evidence": r.get("evidence", ""),
            "confidence": r.get("confidence", ""),
            "validation": "RULE_ACCEPT_V3",
        })

    write_csv(out / "all_reviewed_v3.csv", reviewed)
    write_csv(out / "accepted_relations_v3.csv", accepted)
    write_csv(out / "manual_review_v3.csv", manual)
    write_csv(out / "dynamic_events_v3.csv", events)

    report = {
        "input_rows": len(rows),
        "decisions": dict(Counter(
            r["v3_decision"] for r in reviewed
        )),
        "accepted_by_relation": dict(Counter(
            r["relation"] for r in accepted
        )),
        "review_reasons": dict(Counter(
            r["v3_reason"] for r in manual
        )),
        "events": dict(Counter(
            e["event_type"] for e in events
        )),
        "unique_accepted_relations": len({
            (r["source"], r["relation"], r["target"])
            for r in accepted
        }),
        "notes": [
            "Rule ACCEPT is not human ground truth.",
            "FIRST_OBSERVED is first observed in dataset.",
            "REASSERTED means another supporting filing.",
            "No relation termination inferred from absence.",
            "Validate source availability before FinRL backtesting."
        ]
    }

    (out / "report_v3.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nSaved:", out)

def self_test():
    base = {
        "date_semantics": "sec_published_at_matched",
        "published_at": "2020-01-01",
        "doc_id": "SEC_NVDA_2020-01-01_10-K_test",
        "filing_ticker": "NVDA",
        "source": "INTC",
        "target": "NVDA",
        "relation": "CUSTOMER_OF",
        "evidence": (
            "Revenue from our patent license agreement "
            "with Intel concluded."
        )
    }

    assert decide(base)[0] == "REVIEW"

    dell = dict(base, source="DELL", evidence=(
        "Sales to Dell Technologies Inc. accounted "
        "for 11% of our revenue."
    ))
    assert decide(dell)[0] == "ACCEPT"

    msft = dict(
        base,
        source="MSFT",
        target="AAPL",
        filing_ticker="MSFT",
        relation="COMPETES_WITH",
        evidence=(
            "Windows faces competition from platforms "
            "mainly from Apple and Google."
        )
    )
    assert decide(msft)[0] == "ACCEPT"

    supplier = dict(
        base,
        source="TSM",
        relation="SUPPLIES_TO",
        evidence=(
            "We are dependent on industry-leading "
            "foundries, such as Taiwan Semiconductor "
            "Manufacturing Company Limited."
        )
    )
    assert decide(supplier)[0] == "ACCEPT"

    vague = dict(base, source="AAPL", target="MSFT",
                filing_ticker="MSFT", evidence=(
        "These manufacturers, many of which are also "
        "current or potential partners and customers, "
        "include Apple."
    ))
    assert decide(vague)[0] == "REJECT_AS_PROPOSED"

    print("SELF-TEST PASSED: 5 cases")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default="outputs/reflectkg_validation_v2/all_reviewed.csv"
    )
    p.add_argument(
        "--out-dir",
        default="outputs/reflectkg_validation_v3"
    )
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args.input, args.out_dir)
