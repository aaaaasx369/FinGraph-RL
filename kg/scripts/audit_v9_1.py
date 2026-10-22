#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

ROOT = Path("outputs/v9/nvda_20210601")
SOURCE = Path("outputs/v6/nvda_20210601_qwen.json")

data = json.loads(SOURCE.read_text(encoding="utf-8"))
ticker = data["ticker"]

evidence = {}
for section in (
    "sec_evidence", "findkg_evidence",
    "reflectkg_snapshot", "reflectkg_changes"
):
    for item in data.get(section, []):
        evidence[item["evidence_id"]] = {
            "section": section,
            "content": json.dumps(item, ensure_ascii=False).lower()
        }

rows = []
report = []

for path in sorted(ROOT.glob("*_repeat*.json")):
    run = json.loads(path.read_text(encoding="utf-8"))

    if run.get("status") != "OK":
        continue

    scores = run["scores"]
    claims = run["answer"]["claims"]
    flags = []

    risk = scores["risk"]
    confidence = scores["confidence"]

    if not 0 <= risk <= 1:
        flags.append("INVALID_RISK_SEMANTICS")

    if not 0 <= confidence <= 1:
        flags.append("INVALID_CONFIDENCE_SEMANTICS")

    cited_all = set()

    for claim in claims:
        text = claim["claim"]
        ids = claim["evidence_ids"]
        cited_all.update(ids)

        if not ids:
            flags.append("UNCITED_CLAIM")

        # Flag only when the claim begins by assessing a different
        # company's risk/operations. Merely mentioning a competitor
        # does not trigger this rule.
        wrong_subject = re.search(
            r"^\s*(?:Advanced Micro Devices|AMD|"
            r"Apple|Microsoft|Intel|Broadcom|Alphabet)"
            r"(?:\s*\([A-Z]+\))?\s+"
            r"(?:faces?|suffers?|experiences?|has|"
            r"is|was|reports?|reported)\b",
            text, re.I
        )
        if ticker == "NVDA" and wrong_subject:
            flags.append("TARGET_COMPANY_MISMATCH")

        # COVID claims supported exclusively by graph edge records
        # are not established by graph topology alone.
        covid = re.search(
            r"\bCOVID[- ]?19\b|\bpandemic\b",
            text, re.I
        )
        if covid and ids:
            sections = {
                evidence[eid]["section"]
                for eid in ids if eid in evidence
            }
            if sections == {"findkg_evidence"}:
                flags.append("COVID_CLAIM_GRAPH_ONLY_REVIEW")

        if len(ids) > 5:
            flags.append("CITATION_SATURATION")

    if claims and not cited_all:
        flags.append("NO_SUPPORTED_CLAIM_CITATIONS")

    # Existing citations alone do not validate score grounding.
    if abs(scores["impact"]) >= 0.4 and not cited_all:
        flags.append("IMPACT_WITHOUT_CITED_CLAIMS")

    flags = sorted(set(flags))
    status = "NEEDS_REVIEW" if flags else "RULE_SCREEN_PASS"

    record = {
        "condition": run["condition"],
        "repeat": run["repeat"],
        "impact": scores["impact"],
        "risk": risk,
        "confidence": confidence,
        "status": status,
        "flags": flags
    }
    report.append(record)

    rows.append({
        **{k: v for k, v in record.items() if k != "flags"},
        "flags": "|".join(flags)
    })

    print(
        f"{run['condition']:18s}",
        f"repeat={run['repeat']}",
        status,
        flags
    )

out = ROOT / "audit_v9_1.csv"

with out.open("w", encoding="utf-8", newline="") as f:
    fields = [
        "condition", "repeat", "impact", "risk",
        "confidence", "status", "flags"
    ]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

(ROOT / "audit_v9_1.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nSaved:", out)
print(
    "NEEDS_REVIEW:",
    sum(r["status"] == "NEEDS_REVIEW" for r in report)
)
