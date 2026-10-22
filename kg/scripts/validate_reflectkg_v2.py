#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

RELATIONS = {
    "SUPPLIES_TO", "CUSTOMER_OF", "COMPETES_WITH"
}

ALIASES = {
    "TSM": ["Taiwan Semiconductor Manufacturing Company", "TSMC"],
    "FN": ["Fabrinet"],
    "FLEX": ["Flex Ltd", "Flextronics"],
    "DELL": ["Dell Technologies", "Dell"],
    "MSFT": ["Microsoft"],
    "AAPL": ["Apple"],
    "NVDA": ["NVIDIA"],
    "GOOGL": ["Alphabet", "Google"],
    "GOOG": ["Alphabet", "Google"],
    "AMZN": ["Amazon"],
    "TSLA": ["Tesla"],
    "AMD": ["Advanced Micro Devices", "AMD"],
    "INTC": ["Intel"],
    "HPE": ["Hewlett Packard Enterprise"],
    "AVGO": ["Broadcom"],
}

SUPPLY_PATTERNS = [
    r"\bdepend(?:ent|s|ing)?\s+on\b.{0,120}\bfoundr(?:y|ies)\b",
    r"\bdepend(?:ent|s|ing)?\s+on\b.{0,120}\bsuppliers?\b",
    r"\butiliz(?:e|es|ing)\b.{0,90}\bsuppliers?\b",
    r"\butiliz(?:e|es|ing)\b.{0,90}\bfoundr(?:y|ies)\b",
    r"\buse\b.{0,90}\bcontract manufacturers?\b",
    r"\butiliz(?:e|es|ing)\b.{0,90}\bcontract manufacturers?\b",
    r"\bmanufactur(?:e|es|ing)\b.{0,90}\bfor\b",
]

CUSTOMER_PATTERNS = [
    r"\bsales to\b",
    r"\brevenue from\b",
    r"\bsold to\b",
    r"\bpurchases? from\b",
    r"\bis (?:one of )?our customers?\b",
    r"\b(?:major|significant|largest|key) customers?\b",
]

AMBIGUOUS_PATTERNS = [
    r"\bpotential (?:partners?|customers?|suppliers?)\b",
    r"\bcurrent or potential\b",
    r"\bmany of which\b",
    r"\bmay (?:be|become|include)\b",
    r"\bcould (?:be|become|include)\b",
    r"\bpartners? and customers?\b",
]

COMPETITION_PATTERNS = [
    r"\bcompet(?:e|es|ing|ition|itors?)\b",
    r"\brivals?\b",
    r"\bcompetitive alternatives?\b",
]

def norm(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def canonical(value):
    v = norm(value).upper()
    for suffix in (" INC.", " INC", " CORPORATION", " CORP.",
                   " CORP", " LTD.", " LTD"):
        if v.endswith(suffix):
            v = v[:-len(suffix)]
    for ticker, names in ALIASES.items():
        if v == ticker or any(v == name.upper() for name in names):
            return ticker
    return v

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def available_date(row):
    return norm(row.get("published_at"))[:10]

def load_sec(path):
    by_chunk = {}
    by_doc = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            chunk = norm(r.get("chunk_id"))
            doc = norm(r.get("doc_id"))
            if chunk:
                by_chunk[chunk] = r
            if doc:
                by_doc[doc].append(r)
    return by_chunk, by_doc

def match_name(text, entity, display_name=""):
    choices = [entity, display_name] + ALIASES.get(
        canonical(entity), []
    )
    for name in choices:
        name = norm(name)
        if len(name) < 3:
            continue
        pattern = r"(?<!\w)" + re.escape(name) + r"(?!\w)"
        if re.search(pattern, text, re.I):
            return True
    return False

def has_pattern(text, patterns):
    return any(re.search(p, text, re.I | re.S) for p in patterns)

def filing_company(row):
    ticker = norm(row.get("filing_ticker")).upper()
    if ticker:
        return canonical(ticker)
    doc = norm(row.get("doc_id"))
    m = re.match(r"SEC_([A-Z0-9]+)_", doc)
    return canonical(m.group(1)) if m else ""

def classify(row, sec_chunk):
    relation = norm(row.get("relation")).upper()
    subject = canonical(row.get("source"))
    obj = canonical(row.get("target"))
    filing = filing_company(row)
    evidence = norm(row.get("evidence"))
    status = norm(row.get("validation_status")).upper()

    result = {
        "decision": "REVIEW",
        "reason": "",
        "evidence_type": "",
    }

    if not evidence:
        result.update(
            decision="REVIEW",
            reason="missing_evidence"
        )
        return result

    if relation not in RELATIONS:
        result.update(
            decision="REVIEW",
            reason="unsupported_relation"
        )
        return result

    if sec_chunk is None:
        result.update(
            decision="REVIEW",
            reason="sec_chunk_not_matched"
        )
        return result

    sec_text = norm(sec_chunk.get("text"))
    if evidence.lower() not in sec_text.lower():
        result.update(
            decision="REVIEW",
            reason="evidence_not_exactly_grounded"
        )
        return result

    if not available_date(sec_chunk):
        result.update(
            decision="REVIEW",
            reason="missing_sec_publication_date"
        )
        return result

    if subject == obj:
        result.update(
            decision="REJECT_AS_PROPOSED",
            reason="self_relation"
        )
        return result

    if has_pattern(evidence, AMBIGUOUS_PATTERNS):
        result.update(
            decision="REJECT_AS_PROPOSED",
            reason="ambiguous_or_potential_relation"
        )
        return result

    subject_found = match_name(
        evidence, row.get("source"), row.get("source_name")
    )
    object_found = match_name(
        evidence, row.get("target"), row.get("target_name")
    )

    if relation == "SUPPLIES_TO":
        # For "We use suppliers such as X", the
        # filing company is the buyer, X is supplier.
        if obj != filing or not subject_found:
            result.update(
                decision="REVIEW",
                reason="supply_direction_or_entity_unverified"
            )
        elif has_pattern(evidence, SUPPLY_PATTERNS):
            result.update(
                decision="ACCEPT",
                reason="explicit_supplier_or_manufacturer",
                evidence_type="commercial_supply"
            )
        else:
            result.update(
                decision="REVIEW",
                reason="no_explicit_supply_relation"
            )

    elif relation == "CUSTOMER_OF":
        # "Sales to Dell" in NVDA filing:
        # DELL CUSTOMER_OF NVDA.
        # Customer=subject, Seller=object.
        if obj != filing or not subject_found:
            result.update(
                decision="REVIEW",
                reason="customer_direction_or_entity_unverified"
            )
        elif has_pattern(evidence, CUSTOMER_PATTERNS):
            # Exclude text where the named entity is
            # not clearly the actual customer.
            if re.search(r"\bpurchases? from\b", evidence, re.I):
                result.update(
                    decision="REVIEW",
                    reason="purchase_direction_requires_context"
                )
            else:
                result.update(
                    decision="ACCEPT",
                    reason="explicit_sales_or_customer_evidence",
                    evidence_type="commercial_customer"
                )
        else:
            result.update(
                decision="REVIEW",
                reason="no_explicit_customer_relation"
            )

    elif relation == "COMPETES_WITH":
        if (subject_found and object_found
                and has_pattern(evidence, COMPETITION_PATTERNS)):
            result.update(
                decision="ACCEPT",
                reason="explicit_competition_evidence",
                evidence_type="competition"
            )
        else:
            result.update(
                decision="REVIEW",
                reason="competition_entities_or_phrase_unverified"
            )

    if status in {"REJECT", "REJECTED"}:
        result.update(
            decision="REVIEW",
            reason="previously_rejected_requires_manual_review"
        )

    return result

def validate_date(value):
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(
        k for row in rows for k in row.keys()
    ))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v2", default="kg/data/edges_v2_full.csv"
    )
    p.add_argument(
        "--v4", default="kg/data/edges_v4_full.csv"
    )
    p.add_argument(
        "--sec",
        default="rag/data/processed/sec/sec_chunks.jsonl"
    )
    p.add_argument(
        "--out-dir",
        default="outputs/reflectkg_validation_v2"
    )
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by_chunk, by_doc = load_sec(args.sec)

    raw = []
    for version, path in [
        ("v2", args.v2), ("v4", args.v4)
    ]:
        for r in read_csv(path):
            r = dict(r)
            r["candidate_version"] = version
            raw.append(r)

    # Deduplicate identical claims from different versions.
    # Keep V4 first when the two versions have the same claim.
    raw.sort(
        key=lambda r: 0 if r["candidate_version"] == "v4" else 1
    )

    seen = set()
    reviewed = []

    for r in raw:
        k = (
            canonical(r.get("source")),
            norm(r.get("relation")).upper(),
            canonical(r.get("target")),
            norm(r.get("doc_id")),
            norm(r.get("evidence")).lower()
        )

        if k in seen:
            continue
        seen.add(k)

        chunk_id = norm(r.get("chunk_id"))
        sec_chunk = by_chunk.get(chunk_id)

        # Exact doc-level evidence matching only; no invented
        # text or date when a matching chunk is unavailable.
        if sec_chunk is None:
            for c in by_doc.get(norm(r.get("doc_id")), []):
                evidence = norm(r.get("evidence"))
                if evidence and evidence.lower() in norm(
                    c.get("text")
                ).lower():
                    sec_chunk = c
                    break

        classification = classify(r, sec_chunk)

        new = dict(r)
        new.update(classification)

        new["source"] = canonical(r.get("source"))
        new["target"] = canonical(r.get("target"))
        new["relation"] = norm(r.get("relation")).upper()

        new["published_at"] = (
            available_date(sec_chunk) if sec_chunk else ""
        )
        new["date_semantics"] = (
            "sec_published_at_matched"
            if new["published_at"] else "unverified"
        )
        new["resolved_chunk_id"] = (
            norm(sec_chunk.get("chunk_id"))
            if sec_chunk else ""
        )

        reviewed.append(new)

    accepted = [
        r for r in reviewed
        if r["decision"] == "ACCEPT"
        and validate_date(r["published_at"])
    ]

    review = [
        r for r in reviewed
        if r["decision"] != "ACCEPT"
    ]

    # Event generation: preserve every SEC disclosure.
    # FIRST_OBSERVED and REASSERTED refer to the dataset,
    # not necessarily to real-world relationship creation.
    accepted.sort(
        key=lambda r: (
            r["published_at"],
            r.get("doc_id", ""),
            r.get("resolved_chunk_id", "")
        )
    )

    first_seen = set()
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
            "REASSERTED" if key in first_seen
            else "FIRST_OBSERVED"
        )
        first_seen.add(key)

        events.append({
            "event_type": event_type,
            "subject": r["source"],
            "relation": r["relation"],
            "object": r["target"],
            "published_at": r["published_at"],
            "doc_id": r.get("doc_id", ""),
            "chunk_id": r.get("resolved_chunk_id", ""),
            "evidence": r.get("evidence", ""),
            "confidence": r.get("confidence", ""),
            "evidence_type": r["evidence_type"],
            "date_semantics": r["date_semantics"],
        })

    write_csv(out / "all_reviewed.csv", reviewed)
    write_csv(out / "accepted_relations.csv", accepted)
    write_csv(out / "manual_review.csv", review)
    write_csv(out / "dynamic_events.csv", events)

    report = {
        "input_rows": len(raw),
        "unique_candidates": len(reviewed),
        "decisions": dict(Counter(
            r["decision"] for r in reviewed
        )),
        "accepted_by_relation": dict(Counter(
            r["relation"] for r in accepted
        )),
        "review_reasons": dict(Counter(
            r["reason"] for r in review
        )),
        "events": dict(Counter(
            r["event_type"] for r in events
        )),
        "notes": [
            "ACCEPT is rule-supported, not human verified",
            "FIRST_OBSERVED is first in available dataset",
            "No REMOVE events inferred from absence",
            "SAME_SECTOR requires historical SIC validity",
            "Document dates alone may lack intraday precision"
        ]
    }

    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))
    print("\nSaved to:", out)

if __name__ == "__main__":
    main()
