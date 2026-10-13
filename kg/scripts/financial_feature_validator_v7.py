#!/usr/bin/env python3
"""V7: Conservative financial feature validation.

Input:  Late Fusion V6 JSON
Output: validation JSON and optional feature dataset row.

This validator detects selected errors; it does not prove factual correctness.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

FEATURES = (
    "impact", "risk", "confidence",
    "graph_change", "event_intensity"
)

COMPANIES = {
    "AMD": ("Advanced Micro Devices",),
    "AVGO": ("Broadcom", "Broadcom Inc."),
    "NVDA": ("NVIDIA",),
    "INTC": ("Intel", "Intel Corporation"),
    "TSM": ("TSMC", "Taiwan Semiconductor Manufacturing Company"),
    "FLEX": ("Flex", "Flex Ltd."),
    "FN": ("Fabrinet",),
    "DELL": ("Dell", "Dell Technologies"),
    "MSFT": ("Microsoft",),
    "AAPL": ("Apple",),
    "GOOGL": ("Alphabet", "Google"),
    "GOOG": ("Alphabet", "Google"),
    "AMZN": ("Amazon",),
}

CANONICAL = {
    "GOOG": "GOOGL",
}

# Statements about actual problems require evidence beyond generic
# forward-looking disclaimers.
ACTUAL_PROBLEM_PATTERNS = [
    r"\bactual fraud\b",
    r"\bfraud (?:occurred|was detected|was discovered)\b",
    r"\bunderlying (?:financial(?:\s+or\s+operational)?|operational) issues\b",
    r"\bmaterial weakness(?:es)? (?:was|were|has been) identified\b",
    r"\bconfirmed (?:fraud|accounting irregularit(?:y|ies))\b",
]

DISCLAIMER_PATTERNS = [
    r"\bno (?:system of )?controls? can provide absolute assurance\b",
    r"\bno (?:absolute|complete) assurance\b",
    r"\bif any\b.{0,80}\bdetected\b",
]

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def write_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def add_flag(flags, severity, code, detail):
    flags.append({
        "severity": severity,
        "code": code,
        "detail": detail,
    })

def valid_number(x):
    return (
        isinstance(x, (int, float))
        and not isinstance(x, bool)
        and math.isfinite(x)
        and -1 <= x <= 1
    )

def entity_consistency(text, flags):
    # Detect explicit "Company Name (TICKER)" contradictions.
    pattern = re.compile(
        r"\b([A-Z][A-Za-z&., ]{3,65}?)\s*\(([A-Z]{2,5})\)"
    )

    for match in pattern.finditer(text):
        name = match.group(1).strip()
        ticker = match.group(2)

        if ticker not in COMPANIES:
            continue

        canonical_ticker = CANONICAL.get(ticker, ticker)
        known = COMPANIES[canonical_ticker]

        # Use the final company-name phrase, allowing prose prefixes.
        aliases = [a.lower() for a in known]
        name_low = name.lower()

        recognized_other = [
            (code, alias)
            for code, names in COMPANIES.items()
            if CANONICAL.get(code, code) != canonical_ticker
            for alias in names
            if name_low.endswith(alias.lower())
        ]

        if recognized_other and not any(
            name_low.endswith(a) for a in aliases
        ):
            other = recognized_other[0][0]
            add_flag(
                flags, "HIGH", "ENTITY_MISMATCH",
                f"{name} ({ticker}) conflicts with {other}"
            )

def citation_check(data, features, flags):
    evidence = {}

    for section in (
        "sec_evidence", "findkg_evidence",
        "reflectkg_snapshot", "reflectkg_changes"
    ):
        for item in data.get(section, []):
            if isinstance(item, dict) and item.get("evidence_id"):
                eid = item["evidence_id"]
                if eid in evidence:
                    add_flag(
                        flags, "HIGH", "DUPLICATE_EVIDENCE_ID",
                        f"Duplicate ID: {eid}"
                    )
                evidence[eid] = section

    citations = features.get("citations", [])
    if not isinstance(citations, list):
        add_flag(
            flags, "HIGH", "INVALID_CITATIONS",
            "citations is not a list"
        )
        return

    if not citations:
        add_flag(
            flags, "HIGH", "NO_CITATIONS",
            "No evidence citations were returned"
        )

    for c in citations:
        if not isinstance(c, str) or c not in evidence:
            add_flag(
                flags, "HIGH", "UNKNOWN_CITATION",
                f"Unknown evidence ID: {c}"
            )

    # Missing FinDKG references are an audit flag, not proof that
    # Qwen ignored graph evidence.
    if data.get("findkg_evidence") and not any(
        isinstance(c, str) and c.startswith("FIN_")
        for c in citations
    ):
        add_flag(
            flags, "INFO", "NO_FINDKG_CITATION",
            "FinDKG evidence was available but not cited"
        )

def temporal_check(data, flags):
    from datetime import date

    try:
        cutoff = date.fromisoformat(data["as_of"])
    except (KeyError, ValueError, TypeError):
        add_flag(
            flags, "HIGH", "INVALID_AS_OF",
            "Missing or invalid as_of date"
        )
        return

    fields = (
        ("sec_evidence", "published_at"),
        ("findkg_evidence", "observed_at"),
        ("reflectkg_snapshot", "last_observed_at"),
        ("reflectkg_changes", "published_at"),
    )

    for section, field in fields:
        for item in data.get(section, []):
            if not isinstance(item, dict):
                continue
            value = item.get(field)
            try:
                d = date.fromisoformat(str(value)[:10])
                if d >= cutoff:
                    raise ValueError("Same-day or future data")
            except (ValueError, TypeError):
                add_flag(
                    flags, "HIGH", "TEMPORAL_VIOLATION",
                    f"{section} {item.get('evidence_id')} "
                    f"has invalid or nonhistorical {field}: {value}"
                )

def grounding_heuristics(data, features, flags):
    reasoning = str(features.get("reasoning", ""))

    if any(re.search(p, reasoning, re.I)
           for p in ACTUAL_PROBLEM_PATTERNS):
        add_flag(
            flags, "HIGH", "GROUNDING_REVIEW",
            "Reasoning potentially asserts actual internal problems; "
            "requires manual verification against SEC evidence"
        )

    if re.search(r"\bfraud\b", reasoning, re.I):
        sec_text = " ".join(
            str(e.get("text", ""))
            for e in data.get("sec_evidence", [])
            if isinstance(e, dict)
        )
        if any(re.search(p, sec_text, re.I)
               for p in DISCLAIMER_PATTERNS):
            add_flag(
                flags, "MEDIUM", "BOILERPLATE_RISK",
                "Fraud-related inference accompanies generic "
                "internal-control disclaimer language"
            )

    impact = features.get("impact")
    confidence = features.get("confidence")

    if valid_number(impact) and valid_number(confidence):
        if abs(impact) >= 0.9 and confidence >= 0.9:
            add_flag(
                flags, "MEDIUM", "EXTREME_SCORE_REVIEW",
                "Extreme impact combined with high self-rated confidence"
            )

    if len(data.get("sec_evidence", [])) <= 2:
        add_flag(
            flags, "INFO", "LIMITED_SEC_COVERAGE",
            "Two or fewer SEC evidence chunks were provided"
        )

def validate(data):
    flags = []
    features = data.get("financial_features")

    if not isinstance(features, dict):
        add_flag(
            flags, "HIGH", "MISSING_FINANCIAL_FEATURES",
            "No Qwen financial features found"
        )
        features = {}

    for key in FEATURES:
        if not valid_number(features.get(key)):
            add_flag(
                flags, "HIGH", "INVALID_FEATURE_VALUE",
                f"{key} is missing, nonnumeric or outside [-1,1]"
            )

    reasoning = features.get("reasoning", "")
    if not isinstance(reasoning, str) or not reasoning.strip():
        add_flag(
            flags, "HIGH", "MISSING_REASONING",
            "Qwen reasoning is missing"
        )
        reasoning = ""

    entity_consistency(reasoning, flags)
    citation_check(data, features, flags)
    temporal_check(data, flags)
    grounding_heuristics(data, features, flags)

    blocking = any(
        x["severity"] in ("HIGH", "MEDIUM")
        for x in flags
    )

    # This flag is intentionally conservative. No rule flags
    # means "passed these checks", not "factually verified".
    status = "NEEDS_REVIEW" if blocking else "RULE_CHECK_PASS"

    return {
        "schema_version": "financial_feature_validator_v7",
        "ticker": data.get("ticker"),
        "as_of": data.get("as_of"),
        "raw_features": features,
        "validation_status": status,
        "usable_for_finrl": not blocking,
        "validation_flags": flags,
        "flag_count": len(flags),
        "high_flags": sum(
            x["severity"] == "HIGH" for x in flags
        ),
        "medium_flags": sum(
            x["severity"] == "MEDIUM" for x in flags
        ),
        "notes": [
            "Raw Qwen values are never silently modified.",
            "Rule-check pass does not establish factual accuracy.",
            "Model confidence is not empirically calibrated.",
            "Human review is required for high-stakes research claims."
        ],
    }

def append_csv(path, report):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "ticker", "as_of", *FEATURES,
        "validation_status", "usable_for_finrl",
        "flag_count", "flag_codes"
    ]

    row = {
        "ticker": report["ticker"],
        "as_of": report["as_of"],
        "validation_status": report["validation_status"],
        "usable_for_finrl": report["usable_for_finrl"],
        "flag_count": report["flag_count"],
        "flag_codes": "|".join(
            x["code"] for x in report["validation_flags"]
        ),
    }

    for k in FEATURES:
        row[k] = report["raw_features"].get(k, "")

    # Replace same ticker/date instead of silently accumulating
    # duplicate training rows.
    old = []
    if p.exists():
        with p.open(newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))

    old = [
        x for x in old
        if not (
            x["ticker"] == str(row["ticker"])
            and x["as_of"] == str(row["as_of"])
        )
    ]
    old.append(row)

    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(old)

def self_test():
    base = {
        "ticker": "NVDA",
        "as_of": "2021-06-01",
        "sec_evidence": [{
            "evidence_id": "SEC_001",
            "published_at": "2021-05-26",
            "text": "No controls can provide absolute assurance."
        }],
        "findkg_evidence": [],
        "reflectkg_snapshot": [],
        "reflectkg_changes": [],
        "financial_features": {
            "impact": -1.0,
            "risk": 0.85,
            "confidence": 0.95,
            "graph_change": 0.7,
            "event_intensity": 0.9,
            "citations": ["SEC_001"],
            "reasoning": (
                "Advanced Micro Devices (AVGO) competes. "
                "There could be underlying financial or operational issues."
            )
        }
    }
    report = validate(base)
    codes = {x["code"] for x in report["validation_flags"]}
    assert "ENTITY_MISMATCH" in codes
    assert "GROUNDING_REVIEW" in codes
    assert "EXTREME_SCORE_REVIEW" in codes
    assert report["usable_for_finrl"] is False

    valid = json.loads(json.dumps(base))
    valid["financial_features"].update({
        "impact": -0.2,
        "risk": 0.3,
        "confidence": 0.5,
        "reasoning": "The cited SEC text discusses potential risks."
    })
    assert validate(valid)["validation_status"] == "RULE_CHECK_PASS"

    future = json.loads(json.dumps(valid))
    future["sec_evidence"][0]["published_at"] = "2022-01-01"
    assert any(
        x["code"] == "TEMPORAL_VIOLATION"
        for x in validate(future)["validation_flags"]
    )

    print("SELF-TEST PASSED: mismatch, grounding, scores, clean, future")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input")
    ap.add_argument("--out")
    ap.add_argument("--csv")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.input:
        ap.error("--input is required")

    data = read_json(args.input)
    report = validate(data)

    output = args.out or str(
        Path(args.input).with_suffix(".validated_v7.json")
    )
    write_json(output, report)

    if args.csv:
        append_csv(args.csv, report)

    print(json.dumps({
        "output": output,
        "ticker": report["ticker"],
        "as_of": report["as_of"],
        "status": report["validation_status"],
        "usable_for_finrl": report["usable_for_finrl"],
        "high_flags": report["high_flags"],
        "medium_flags": report["medium_flags"],
        "flags": [
            x["code"] for x in report["validation_flags"]
        ]
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
