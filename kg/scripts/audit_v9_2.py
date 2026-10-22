#!/usr/bin/env python3
"""
V9.2: Claim-to-Evidence Audit and Human Annotation Queue.

Input:
  Existing V6 fused JSON and V9 run JSON files.

Outputs:
  claims_review_queue.csv
  run_quality_gate.csv
  grounding_report.json
  human_labels.csv (created only when absent)

No Ollama calls. No source files overwritten.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter, defaultdict

LABELS = {
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "UNSUPPORTED",
    "CONTRADICTED",
    "NOT_ASSESSABLE",
}

SOURCES = {
    "sec_evidence": "SEC",
    "findkg_evidence": "FinDKG",
    "reflectkg_snapshot": "ReflectKG",
    "reflectkg_changes": "ReflectKG",
}

LABEL_FIELDS = [
    "run_id", "claim_id", "label",
    "reviewer_note"
]

QUEUE_FIELDS = [
    "run_id", "condition", "repeat", "claim_id",
    "claim", "certainty", "evidence_ids",
    "evidence_sources", "evidence_excerpts",
    "auto_flags", "human_label", "reviewer_note"
]

GATE_FIELDS = [
    "run_id", "condition", "repeat",
    "impact", "risk", "confidence",
    "claims", "labeled_claims",
    "unsupported_claims", "rule_flags",
    "review_status", "usable_for_finrl"
]

COMPANY_ALIASES = {
    "NVDA": ("NVIDIA",),
    "AAPL": ("Apple",),
    "MSFT": ("Microsoft",),
    "AMZN": ("Amazon",),
    "GOOGL": ("Alphabet", "Google"),
}

OTHER_COMPANIES = (
    "Advanced Micro Devices", "AMD", "Apple",
    "Microsoft", "Intel", "Broadcom",
    "Alphabet", "Google", "Amazon",
)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_evidence_map(v6):
    mapping = {}
    for field, source in SOURCES.items():
        for item in v6.get(field, []):
            eid = item.get("evidence_id")
            if not eid:
                continue
            if eid in mapping:
                raise ValueError("Duplicate evidence ID " + eid)
            mapping[eid] = {
                "source": source,
                "section": field,
                "item": item,
            }
    return mapping


def excerpt(item, section, limit=700):
    if section == "sec_evidence":
        value = item.get("text", "")
    elif section == "findkg_evidence":
        value = json.dumps(
            item.get("record", {}),
            ensure_ascii=False,
        )
    elif section == "reflectkg_snapshot":
        value = (
            f"{item.get('subject')} "
            f"{item.get('relation')} "
            f"{item.get('object')}. "
            f"{item.get('latest_evidence', '')}"
        )
    else:
        value = (
            f"{item.get('subject')} "
            f"{item.get('relation')} "
            f"{item.get('object')}. "
            f"Event: {item.get('event_type')}. "
            f"{item.get('evidence', '')}"
        )
    return re.sub(r"\s+", " ", str(value)).strip()[:limit]


def claim_flags(claim, evidence_map, ticker):
    flags = []
    text = str(claim.get("claim", ""))
    ids = claim.get("evidence_ids", [])

    if not ids:
        flags.append("UNCITED_CLAIM")

    if len(ids) > 5:
        flags.append("CITATION_SATURATION")

    for eid in ids:
        if eid not in evidence_map:
            flags.append("UNKNOWN_CITATION")

    if re.search(
        r"Advanced Micro Devices\s*\(AVGO\)"
        r"|Broadcom\s*\(AMD\)",
        text, re.I
    ):
        flags.append("ENTITY_MISMATCH")

    # Detect a claim whose assessed subject is a different
    # company, without rejecting normal competitor mentions.
    head = text[:130]
    pattern = (
        r"^\s*(" +
        "|".join(re.escape(x) for x in OTHER_COMPANIES) +
        r")(?:\s*\([A-Z]{2,5}\))?"
        r"\s+(?:faces?|suffers?|experiences?|"
        r"has|is|was|reports?|reported)\b"
    )
    m = re.search(pattern, head, re.I)

    if m:
        company = m.group(1).lower()
        names = COMPANY_ALIASES.get(ticker, (ticker,))
        if not any(company == x.lower() for x in names):
            flags.append("TARGET_COMPANY_MISMATCH")

    covid_claim = bool(re.search(
        r"\bCOVID[- ]?19\b|\bpandemic\b",
        text, re.I
    ))
    if covid_claim and ids:
        sources = {
            evidence_map[eid]["source"]
            for eid in ids if eid in evidence_map
        }
        if sources == {"FinDKG"}:
            flags.append("COVID_GRAPH_ONLY_REVIEW")

    # A citation verifies identity, not entailment.
    if claim.get("certainty") == "explicit":
        if not ids:
            flags.append("EXPLICIT_UNCITED")

    return sorted(set(flags))


def run_flags(run):
    flags = []
    scores = run.get("scores", {})
    impact = scores.get("impact")
    risk = scores.get("risk")
    confidence = scores.get("confidence")

    for name, val in (
        ("impact", impact),
        ("risk", risk),
        ("confidence", confidence)
    ):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            flags.append("INVALID_" + name.upper())

    if isinstance(impact, (int, float)) and not isinstance(impact, bool):
        if not -1 <= impact <= 1:
            flags.append("INVALID_IMPACT_RANGE")

    for name, val in (("risk", risk), ("confidence", confidence)):
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if not 0 <= val <= 1:
                flags.append("INVALID_" + name.upper() + "_RANGE")

    return flags


def make_run_id(run):
    return (
        run["condition"].replace("+", "_").replace("-", "_")
        + "_repeat"
        + f"{int(run['repeat']):02d}"
    )


def audit(v6, runs, labels):
    evidence_map = build_evidence_map(v6)
    ticker = v6["ticker"]

    queue = []
    gates = []
    seen_claims = set()

    for run in runs:
        if run.get("status") != "OK":
            continue

        run_id = make_run_id(run)
        answer = run.get("answer", {})
        claims = answer.get("claims", [])
        flags = set(run_flags(run))
        count_labeled = 0
        count_bad = 0

        for claim in claims:
            claim_id = claim["claim_id"]
            key = (run_id, claim_id)
            if key in seen_claims:
                raise ValueError("Duplicate run/claim: " + str(key))
            seen_claims.add(key)

            ids = claim.get("evidence_ids", [])
            cflags = claim_flags(claim, evidence_map, ticker)
            flags.update(cflags)

            refs = []
            source_names = []

            for eid in ids:
                found = evidence_map.get(eid)
                if not found:
                    refs.append(f"{eid}: UNKNOWN")
                    continue

                source_names.append(found["source"])
                refs.append(
                    f"{eid} [{found['source']}] "
                    + excerpt(
                        found["item"], found["section"]
                    )
                )

            manual = labels.get(key, {})
            label = manual.get("label", "")
            note = manual.get("reviewer_note", "")

            if label in LABELS:
                count_labeled += 1
            if label in ("UNSUPPORTED", "CONTRADICTED"):
                count_bad += 1

            queue.append({
                "run_id": run_id,
                "condition": run["condition"],
                "repeat": run["repeat"],
                "claim_id": claim_id,
                "claim": claim["claim"],
                "certainty": claim.get("certainty", ""),
                "evidence_ids": "|".join(ids),
                "evidence_sources": "|".join(
                    sorted(set(source_names))
                ),
                "evidence_excerpts": "\n".join(refs),
                "auto_flags": "|".join(cflags),
                "human_label": label,
                "reviewer_note": note,
            })

        if flags:
            status = "NEEDS_REVIEW"
        elif count_labeled < len(claims) or not claims:
            status = "AWAITING_HUMAN_LABELS"
        elif count_bad:
            status = "NEEDS_REVIEW"
        elif any(
            labels.get((run_id, c["claim_id"]), {}).get("label")
            == "NOT_ASSESSABLE"
            for c in claims
        ):
            status = "NEEDS_REVIEW"
        else:
            status = "HUMAN_REVIEW_COMPLETED"

        # Preserve the distinction between a completed review and
        # a defensible positive label. PARTIALLY_SUPPORTED is not
        # approved for formal finance feature consumption here.
        strictly_supported = (
            bool(claims)
            and count_labeled == len(claims)
            and all(
                labels.get((run_id, c["claim_id"]), {}).get("label")
                == "SUPPORTED"
                for c in claims
            )
        )
        usable = (
            not flags
            and strictly_supported
        )

        if usable:
            status = "HUMAN_REVIEW_COMPLETED"
        elif status == "HUMAN_REVIEW_COMPLETED":
            status = "NEEDS_REVIEW"

        scores = run.get("scores", {})
        gates.append({
            "run_id": run_id,
            "condition": run["condition"],
            "repeat": run["repeat"],
            "impact": scores.get("impact"),
            "risk": scores.get("risk"),
            "confidence": scores.get("confidence"),
            "claims": len(claims),
            "labeled_claims": count_labeled,
            "unsupported_claims": count_bad,
            "rule_flags": "|".join(sorted(flags)),
            "review_status": status,
            "usable_for_finrl": str(usable).lower(),
        })

    return queue, gates


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v6",
        default="outputs/v6/nvda_20210601_qwen.json"
    )
    p.add_argument(
        "--runs-dir",
        default="outputs/v9/nvda_20210601"
    )
    args = p.parse_args()

    root = Path(args.runs_dir)
    v6 = read_json(args.v6)

    run_paths = sorted(root.glob("*_repeat*.json"))
    if not run_paths:
        raise SystemExit("No V9 run JSON files found")

    runs = [read_json(path) for path in run_paths]
    runs = [r for r in runs if r.get("status") == "OK"]

    label_path = root / "human_labels.csv"
    labels = {}

    if label_path.exists():
        for r in read_csv(label_path):
            label = r.get("label", "").strip().upper()
            if label and label not in LABELS:
                raise ValueError(
                    "Invalid human label: " + label
                )
            key = (r["run_id"], r["claim_id"])
            labels[key] = {
                "label": label,
                "reviewer_note": r.get("reviewer_note", ""),
            }

    queue, gates = audit(v6, runs, labels)

    write_csv(
        root / "claims_review_queue.csv",
        QUEUE_FIELDS, queue
    )
    write_csv(
        root / "run_quality_gate.csv",
        GATE_FIELDS, gates
    )

    # Do not overwrite annotation work.
    if not label_path.exists():
        write_csv(
            label_path,
            LABEL_FIELDS,
            [{
                "run_id": r["run_id"],
                "claim_id": r["claim_id"],
                "label": "",
                "reviewer_note": "",
            } for r in queue]
        )

    report = {
        "runs_audited": len(gates),
        "claims_audited": len(queue),
        "status_counts": dict(Counter(
            r["review_status"] for r in gates
        )),
        "usable_for_finrl": sum(
            r["usable_for_finrl"] == "true"
            for r in gates
        ),
        "claim_flag_counts": dict(Counter(
            flag
            for r in queue
            for flag in r["auto_flags"].split("|")
            if flag
        )),
        "notes": [
            "RULE_SCREEN_PASS does not mean factual correctness.",
            "Human labels are not generated automatically.",
            "Evidence snippets can be truncated; inspect full source.",
            "No model confidence calibration is performed.",
            "No feature values are changed.",
        ]
    }

    write_json(root / "grounding_report.json", report)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nReview queue:", root / "claims_review_queue.csv")
    print("Human labels:", label_path)
    print("Run gate:", root / "run_quality_gate.csv")


if __name__ == "__main__":
    main()
