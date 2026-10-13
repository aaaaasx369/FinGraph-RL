#!/usr/bin/env python3
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full/four_relation_pipeline"
RAW = ROOT / "rag/data/processed/sec/sec_chunks.jsonl"
MANIFEST = ROOT / "outputs/v20/sec_filings_manifest.csv"
SOURCE = BASE / "relation_provenance_v2.csv"

ACC_RE = re.compile(r"\d{10}-\d{2}-\d{6}")
BARE_RE = re.compile(r"(?<!\d)\d{18}(?!\d)")

REVIEW_ONLY = {
    "GOOGM", "GOOGN", "HXSCL", "SKHY",
    "BRK-A", "BRK-B"
}

def normalize_accession(value):
    s = str(value or "")
    m = ACC_RE.search(s)
    if m:
        return m.group()

    m = BARE_RE.search(s)
    if m:
        v = m.group()
        return f"{v[:10]}-{v[10:12]}-{v[12:]}"

    return ""

def flatten(obj):
    if not isinstance(obj, dict):
        return []
    values = []
    for key, value in obj.items():
        if isinstance(value, dict):
            values.extend(flatten(value))
        elif isinstance(value, (str, int)):
            values.append((key, str(value)))
    return values

def main():
    df = pd.read_csv(SOURCE, dtype=str).fillna("")
    manifest = pd.read_csv(MANIFEST, dtype=str).fillna("")

    manifest["ticker"] = (
        manifest["ticker"].str.upper().str.strip()
    )
    manifest["acc_normalized"] = (
        manifest["accession"].map(normalize_accession)
    )

    index = {}
    for _, row in manifest.iterrows():
        key = (row["ticker"], row["acc_normalized"])
        if key[1]:
            index.setdefault(key, []).append(row)

    source_lines = set(df["source_line"].astype(int))
    original = {}

    with RAW.open(encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            if n in source_lines:
                original[n] = json.loads(line)

    output = []

    for _, row in df.iterrows():
        r = row.to_dict()
        raw = original.get(int(r["source_line"]), {})
        candidates = set()

        for key, value in flatten(raw):
            k = key.lower()
            if any(x in k for x in (
                "accession", "chunk_id",
                "doc_id", "url", "filing_id",
                "document"
            )):
                a = normalize_accession(value)
                if a:
                    candidates.add(a)

        # Include v2's previously matched accession,
        # but never arbitrarily choose among conflicts.
        old = normalize_accession(
            r.get("matched_accession", "")
        )
        if old:
            candidates.add(old)

        matches = {}
        for acc in candidates:
            for m in index.get(
                (r["issuer"].upper(), acc), []
            ):
                matches[(m["ticker"], m["accession"])] = m

        r["extracted_accessions"] = ";".join(
            sorted(candidates)
        )

        if len(matches) == 1:
            m = next(iter(matches.values()))
            r["provenance_status_v3"] = "EXACT_FILING"
            r["matched_accession"] = m["accession"]
            r["matched_filing_date"] = m["filing_date"]
            r["acceptance_utc"] = m["acceptance_utc"]
            r["available_date"] = m["available_date"]
            r["source_url"] = m["document_url"]
            r["metadata_verified"] = m["metadata_verified"]
            r["text_verified"] = m["text_verified"]
        elif len(matches) > 1:
            r["provenance_status_v3"] = "AMBIGUOUS"
        else:
            r["provenance_status_v3"] = "UNRESOLVED"

        head, tail = r["head"], r["tail"]

        if {head, tail} == {"GOOGL", "GOOG"}:
            r["entity_status_v3"] = "SAME_ISSUER_REJECT"
        elif head in REVIEW_ONLY or tail in REVIEW_ONLY:
            r["entity_status_v3"] = "ENTITY_REVIEW_REQUIRED"
        else:
            r["entity_status_v3"] = "UNREVIEWED"

        r["review_status_v3"] = "PENDING"
        r["strict_pit_approved_v3"] = False
        output.append(r)

    result = pd.DataFrame(output)

    target = BASE / "relation_provenance_v3.csv"
    result.to_csv(target, index=False)

    print("\n=== PROVENANCE V3 ===")
    print(result["provenance_status_v3"].value_counts())

    print("\n=== ENTITY STATUS ===")
    print(result["entity_status_v3"].value_counts())

    print("\n=== TIME COVERAGE ===")
    confirmed = (
        result["provenance_status_v3"].eq("EXACT_FILING")
        & result["acceptance_utc"].ne("")
        & result["metadata_verified"].eq("1")
        & result["text_verified"].eq("1")
    )
    print("Verified filing metadata:", int(confirmed.sum()), "/", len(result))

    print("\n=== UNRESOLVED DETAILS ===")
    unresolved = result[
        result["provenance_status_v3"].ne("EXACT_FILING")
    ]
    if len(unresolved):
        print(unresolved[
            ["source_line", "issuer", "extracted_accessions"]
        ].to_string(index=False))
    else:
        print("None")

    print("\nOutput:", target)
    print("Strict PIT: NOT CERTIFIED")
    print("KGTransformer: NOT STARTED")

if __name__ == "__main__":
    main()
