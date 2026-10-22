#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from collections import Counter

def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_csv(path, fields, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().lower()

def company_alias(name):
    n = norm(name)
    aliases = {
        "nvidia corp.": "nvidia",
        "nvidia": "nvidia",
        "nvda": "nvidia",
        "advanced micro devices": "amd",
        "amd": "amd",
        "broadcom": "broadcom",
        "avgo": "broadcom",
        "alphabet inc.": "alphabet",
        "google": "alphabet",
        "googl": "alphabet",
        "intel corporation": "intel",
        "intc": "intel",
        "flex ltd.": "flex",
        "dell technologies": "dell",
    }
    return aliases.get(n, n)

def build_lookup(v6):
    lookup = {}
    for section in (
        "sec_evidence", "findkg_evidence",
        "reflectkg_snapshot", "reflectkg_changes"
    ):
        for item in v6.get(section, []):
            eid = item.get("evidence_id")
            if eid:
                lookup[eid] = (section, item)
    return lookup

def check_claim(claim, lookup):
    text = claim.get("claim", "")
    ids = claim.get("evidence_ids", [])
    flags = []

    if not ids:
        flags.append("UNCITED_CLAIM")

    if len(ids) > 4:
        flags.append("CITATION_SATURATION")

    for eid in ids:
        if eid not in lookup:
            flags.append("UNKNOWN_CITATION")
            continue

        section, item = lookup[eid]

        if section == "findkg_evidence":
            record = item.get("record", {})
            subject = company_alias(record.get("subject"))
            obj = company_alias(record.get("object"))
            relation = norm(record.get("relation"))

            if not record.get("mapping_resolved"):
                flags.append("UNRESOLVED_FINDKG_MAPPING")

            # Investment direction in a KG must be preserved.
            if relation == "invests_in":
                lower = norm(text)
                if (
                    subject == "technology companies"
                    and obj == "nvidia"
                    and re.search(
                        r"nvidia.*invest(?:ed|s|ing)? in "
                        r"technology companies", lower
                    )
                ):
                    flags.append("REVERSED_INVESTMENT_DIRECTION")

            if relation == "control":
                flags.append("OWNERSHIP_CONTROL_REQUIRES_REVIEW")

            # Broad relation categories alone do not establish
            # realized financial impact.
            if relation in (
                "relate_to", "operate_in", "has"
            ) and re.search(
                r"\b(?:confirmed|caused|resulted in)\b"
                r".*\b(?:loss|revenue|profit|decline)\b",
                text, re.I
            ):
                flags.append("UNSUPPORTED_FINANCIAL_CAUSALITY")

        if section in ("reflectkg_snapshot", "reflectkg_changes"):
            relation = item.get("relation")
            if relation == "COMPETES_WITH":
                # Historical observations do not establish
                # a relation remains active at query time.
                if re.search(
                    r"\bcurrent competitors?\b"
                    r"|\bcurrently competes?\b",
                    text, re.I
                ):
                    flags.append("CURRENT_STATUS_UNVERIFIED")

    # Target-specific issue previously observed in SEC-only.
    if re.search(
        r"^\s*Advanced Micro Devices\s*\(AMD\)\s+faces\b",
        text, re.I
    ):
        flags.append("TARGET_COMPANY_MISMATCH")

    # SEC and KG may describe potential, not realized, risks.
    if re.search(
        r"\bconfirmed fraud\b|\bactual fraud\b",
        text, re.I
    ):
        flags.append("REALIZED_EVENT_NEEDS_VERIFICATION")

    # Citation IDs might exist yet support unrelated claims.
    # We intentionally do not auto-certify any claim.
    return sorted(set(flags))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--v6",
        default="outputs/v6/nvda_20210601_focal_findkg.json"
    )
    ap.add_argument(
        "--runs",
        default="outputs/v9/nvda_focal_findkg_v2"
    )
    ap.add_argument(
        "--out",
        default="outputs/v10/directional_audit.csv"
    )
    args = ap.parse_args()

    v6 = load(args.v6)
    lookup = build_lookup(v6)

    rows = []
    for path in sorted(Path(args.runs).glob("*_repeat*.json")):
        run = load(path)
        if run.get("status") != "OK":
            continue

        claims = run.get("answer", {}).get("claims", [])

        for c in claims:
            flags = check_claim(c, lookup)
            rows.append({
                "condition": run["condition"],
                "repeat": run["repeat"],
                "claim_id": c.get("claim_id"),
                "claim": c.get("claim"),
                "evidence_ids": "|".join(
                    c.get("evidence_ids", [])
                ),
                "flags": "|".join(flags),
                "status": (
                    "NEEDS_REVIEW"
                    if flags else "NOT_SEMANTICALLY_VERIFIED"
                )
            })

    fields = [
        "condition", "repeat", "claim_id", "claim",
        "evidence_ids", "flags", "status"
    ]
    write_csv(args.out, fields, rows)

    counts = Counter()
    for r in rows:
        for flag in r["flags"].split("|"):
            if flag:
                counts[flag] += 1

    print("Claims audited:", len(rows))
    print("Flag counts:", dict(counts))
    print("Saved:", args.out)

    print("\n=== FLAGGED CLAIMS ===")
    for r in rows:
        if r["flags"]:
            print(
                r["condition"],
                "repeat", r["repeat"],
                r["flags"]
            )
            print(" ", r["claim"])

if __name__ == "__main__":
    main()
