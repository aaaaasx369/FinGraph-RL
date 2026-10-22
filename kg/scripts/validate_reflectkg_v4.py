#!/usr/bin/env python3
"""ReflectKG V4: conservative role grounding and event rebuild."""
import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter

ALIAS = {"GOOG": "GOOGL"}

NAMES = {
    "TSM": ["Taiwan Semiconductor Manufacturing Company", "TSMC"],
    "FLEX": ["Flex Ltd", "Flextronics"],
    "FN": ["Fabrinet"],
    "DELL": ["Dell Technologies", "Dell"],
    "MSFT": ["Microsoft"],
    "AAPL": ["Apple"],
    "NVDA": ["NVIDIA"],
    "GOOGL": ["Alphabet", "Google"],
    "INTC": ["Intel"],
    "AMD": ["Advanced Micro Devices"],
    "AMZN": ["Amazon"],
}

# Phrases that explicitly describe a supplier or manufacturer
# providing goods/services TO the filing company.
DIRECT_SUPPLY = [
    r"\b(?:we|our company)\s+(?:are|is)\s+dependent on\b.{0,100}\b(?:foundries|suppliers)\b",
    r"\b(?:we|our company)\s+(?:use|uses|utilize|utilizes)\b.{0,90}\b(?:suppliers|foundries|contract manufacturers)\b",
    r"\bwe\s+(?:purchase|procure|source|obtain)\b.{0,90}\bfrom\b",
    r"\bwe\s+(?:outsource|contract)\b.{0,90}\bmanufactur",
    r"\b(?:supplies|provides|manufactures)\b.{0,100}\b(?:for|to)\s+(?:us|our company)\b",
]

# Competitive landscape descriptions: "supplier of X products"
# does not mean that X supplies the SEC filing company.
MARKET_CATEGORIES = [
    r"\bsuppliers?\s+of\s+(?:soc|interconnect|switch|cable|hardware|software)\b",
    r"\bcompanies?\s+with\s+internal\s+teams?\s+designing\b",
    r"\blarge\s+(?:internet|cloud)\s+services?\s+companies?\b",
]

AMBIGUOUS = [
    r"\bcurrent or potential\b",
    r"\bpotential\s+(?:customers?|partners?|suppliers?)\b",
    r"\bmany of which\b",
    r"\bpartners?\s+and\s+customers?\b",
]

LICENSE = [
    r"\bpatent licens",
    r"\blicens(?:e|ing)\s+agreement\b",
    r"\blicensing revenue\b",
    r"\broyalt(?:y|ies)\b",
]

def clean(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def canon(x):
    s = clean(x).upper()
    return ALIAS.get(s, s)

def match(text, term):
    return bool(re.search(
        r"(?<!\w)" + re.escape(term) + r"(?!\w)",
        text, re.I
    ))

def entity_mentions(text, ticker, display=""):
    choices = [display, ticker] + NAMES.get(ticker, [])
    return any(
        match(text, clean(name))
        for name in choices
        if len(clean(name)) >= 3
    )

def hit(text, patterns):
    return any(re.search(p, text, re.I) for p in patterns)

def filing_ticker(row):
    ticker = clean(row.get("filing_ticker")).upper()
    if ticker:
        return canon(ticker)
    m = re.match(r"SEC_([A-Z0-9]+)_", clean(row.get("doc_id")))
    return canon(m.group(1)) if m else ""

def examine(row):
    """Return decision and reason; never infer new relations."""
    relation = clean(row.get("relation")).upper()
    subject = canon(row.get("source"))
    obj = canon(row.get("target"))
    filing = filing_ticker(row)
    evidence = clean(row.get("evidence"))

    if (row.get("date_semantics") != "sec_published_at_matched"
            or not row.get("published_at")
            or not row.get("doc_id")
            or not row.get("resolved_chunk_id")):
        return "REVIEW", "source_date_or_chunk_unverified"

    if not evidence or subject == obj:
        return "REJECT_AS_PROPOSED", "missing_or_self_relation"

    if hit(evidence, AMBIGUOUS):
        return "REJECT_AS_PROPOSED", "ambiguous_commercial_roles"

    if relation == "SUPPLIES_TO":
        if obj != filing:
            return "REVIEW", "supply_direction_unverified"

        if not entity_mentions(
            evidence, subject, row.get("source_name", "")
        ):
            return "REVIEW", "supplier_not_named"

        # Important: matching a broad supplier category in a
        # competitive-landscape paragraph is NOT direct evidence
        # of supply to the filing company.
        if hit(evidence, MARKET_CATEGORIES) and not hit(
            evidence, DIRECT_SUPPLY
        ):
            return (
                "REJECT_AS_PROPOSED",
                "market_supplier_list_not_supply_to_filer"
            )

        if hit(evidence, DIRECT_SUPPLY):
            # If multiple commercial roles appear within the same
            # text span, keep for human review rather than auto-accept.
            if hit(evidence, MARKET_CATEGORIES):
                return "REVIEW", "mixed_market_and_direct_supply"
            return "ACCEPT", "direct_supply_to_filing_company"

        return "REVIEW", "no_direct_supply_to_filing_company"

    if relation == "CUSTOMER_OF":
        if obj != filing:
            return "REVIEW", "customer_direction_unverified"

        if not entity_mentions(
            evidence, subject, row.get("source_name", "")
        ):
            return "REVIEW", "customer_not_named"

        if hit(evidence, LICENSE):
            return "REVIEW", "license_not_direct_product_sales"

        # Require explicitly named buyer in a seller-side statement.
        names = [
            clean(row.get("source_name")),
            *NAMES.get(subject, []),
            subject
        ]
        buyer_pattern = "|".join(
            re.escape(n) for n in names if len(n) >= 3
        )
        if buyer_pattern:
            if re.search(
                r"\b(?:sales to|sold to)\s+(?:" +
                buyer_pattern + r")\b", evidence, re.I
            ):
                return "ACCEPT", "named_buyer_direct_sales"

        return "REVIEW", "buyer_not_directly_grounded"

    if relation == "COMPETES_WITH":
        if filing not in (subject, obj):
            return "REVIEW", "filing_not_a_competitor"

        other = obj if subject == filing else subject
        display = (
            row.get("target_name", "")
            if subject == filing else row.get("source_name", "")
        )
        if not entity_mentions(evidence, other, display):
            return "REVIEW", "competitor_not_named"

        if re.search(
            r"\b(?:competition|compete|competes|competing|competitors?|rivals?)\b",
            evidence, re.I
        ):
            return "ACCEPT", "explicit_competition"

        return "REVIEW", "competition_context_missing"

    return "REVIEW", "unsupported_relation"

def save_csv(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(
            k for r in rows for k in r
        ))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="outputs/reflectkg_validation_v3/all_reviewed_v3.csv"
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/reflectkg_validation_v4"
    )
    args = ap.parse_args()

    out = Path(args.out_dir)
    with open(args.input, newline="", encoding="utf-8-sig") as f:
        raw = list(csv.DictReader(f))

    reviewed = []
    for r in raw:
        row = dict(r)
        row["source"] = canon(row.get("source"))
        row["target"] = canon(row.get("target"))
        row["relation"] = clean(row.get("relation")).upper()
        row["filing_ticker"] = filing_ticker(row)
        decision, reason = examine(row)
        row["v4_decision"] = decision
        row["v4_reason"] = reason
        reviewed.append(row)

    # Dedupe exact evidence claims after entity normalization.
    unique = {}
    for r in reviewed:
        key = (
            r["source"], r["relation"], r["target"],
            r.get("doc_id", ""),
            r.get("resolved_chunk_id", ""),
            clean(r.get("evidence")).casefold()
        )
        if key not in unique:
            unique[key] = r
        elif (unique[key]["v4_decision"] != "ACCEPT"
              and r["v4_decision"] == "ACCEPT"):
            unique[key] = r

    rows = list(unique.values())
    accepted = [r for r in rows if r["v4_decision"] == "ACCEPT"]
    manual = [r for r in rows if r["v4_decision"] != "ACCEPT"]

    accepted.sort(key=lambda r: (
        r["published_at"], r.get("doc_id", ""),
        r.get("resolved_chunk_id", "")
    ))

    seen_relations = set()
    seen_relation_docs = set()
    events = []

    for r in accepted:
        key = (r["source"], r["relation"], r["target"])
        doc_key = key + (r["doc_id"],)

        if doc_key in seen_relation_docs:
            continue
        seen_relation_docs.add(doc_key)

        event_type = (
            "REASSERTED" if key in seen_relations
            else "FIRST_OBSERVED"
        )
        seen_relations.add(key)

        events.append({
            "event_type": event_type,
            "subject": r["source"],
            "relation": r["relation"],
            "object": r["target"],
            "published_at": r["published_at"],
            "doc_id": r["doc_id"],
            "chunk_id": r.get("resolved_chunk_id", ""),
            "evidence": r.get("evidence", ""),
            "confidence": r.get("confidence", ""),
            "validation": "RULE_ACCEPT_V4",
        })

    save_csv(out / "all_reviewed_v4.csv", rows)
    save_csv(out / "accepted_relations_v4.csv", accepted)
    save_csv(out / "manual_review_v4.csv", manual)
    save_csv(out / "dynamic_events_v4.csv", events, [
        "event_type", "subject", "relation", "object",
        "published_at", "doc_id", "chunk_id", "evidence",
        "confidence", "validation"
    ])

    report = {
        "input_rows": len(raw),
        "unique_evidence_rows": len(rows),
        "decisions": dict(Counter(r["v4_decision"] for r in rows)),
        "accepted_by_relation": dict(Counter(
            r["relation"] for r in accepted
        )),
        "review_reasons": dict(Counter(
            r["v4_reason"] for r in manual
        )),
        "events": dict(Counter(e["event_type"] for e in events)),
        "unique_relations": len(seen_relations),
        "warnings": [
            "Rule ACCEPT is not human-verified ground truth",
            "Broad multi-clause supply descriptions are not automatically accepted",
            "FIRST_OBSERVED is first in this dataset",
            "No REMOVE inferred from a missing filing",
            "As-of queries must strictly filter SEC published_at"
        ]
    }
    (out / "report_v4.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
