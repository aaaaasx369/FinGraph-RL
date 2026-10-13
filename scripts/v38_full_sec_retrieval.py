#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v38"
sys.path.insert(0, str(ROOT / "scripts"))

import v34_sec_relation_retrieval as v34

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

CUES = {
    "SUPPLIES_TO": [
        r"\bsupplier(?:s)?\b",
        r"\bsupply(?:ing|ies|ied)?\b",
        r"\bsupplied by\b",
        r"\bmanufactur(?:e|er|ers|ing|ed)\b",
        r"\bfoundr(?:y|ies)\b",
        r"\bsource(?:d|s|ing)? from\b",
        r"\bsubcontract(?:or|ors|ed|ing)?\b",
        r"\bprovide(?:s|d)? .{0,100}?\bto\b",
    ],
    "CUSTOMER_OF": [
        r"\bcustomer(?:s)?\b",
        r"\bclient(?:s)?\b",
        r"\bpurchas(?:e|es|ed|ing)\b",
        r"\bprocure(?:ment|d)?\b",
        r"\bbuy(?:s|ing)? from\b",
        r"\brevenue from\b",
        r"\bsales to\b",
    ],
    "COMPETES_WITH": [
        r"\bcompetitor(?:s)?\b",
        r"\bcompet(?:e|es|ed|ing|ition)\b",
        r"\brival(?:s)?\b",
    ],
    "SAME_SECTOR": [
        r"\bsame industry\b",
        r"\bsame sector\b",
        r"\bindustry peers?\b",
        r"\bsector peers?\b",
        r"\bpeer companies\b",
        r"\bindustry participants\b",
    ],
}

def h(*parts):
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode()
    ).hexdigest()[:24]

def cue_matches(text):
    found = {}
    for rel, patterns in CUES.items():
        matches = []
        for pattern in patterns:
            matches.extend(
                m.group()
                for m in re.finditer(pattern, text, re.I)
            )
        if matches:
            found[rel] = sorted(set(matches))[:12]
    return found

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    aliases = v34.load_aliases()
    documents = v34.prepare_documents()

    output = OUT / "full_sec_relation_evidence.csv"
    audit_rows = []
    total = Counter()
    seen = set()

    cols = [
        "evidence_id", "ticker", "form", "accession",
        "filing_date", "available_from", "source_sha256",
        "offset_start", "offset_end",
        "context_start", "sentence", "context",
        "named_entities", "relation_cues", "relation_hint",
        "candidate_only", "human_verified",
    ]

    with output.open("w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()

        for index, doc in enumerate(documents, 1):
            file = v34.cache_path(doc)

            if not file.is_file():
                raise FileNotFoundError(file)

            raw = file.read_bytes()
            actual_sha = hashlib.sha256(raw).hexdigest()
            expected_sha = str(doc["source_sha256"])

            if actual_sha != expected_sha:
                raise ValueError(
                    "SEC SHA256 mismatch: " + str(file)
                )

            text = v34.extract_text(raw)
            spans = list(v34.sentence_spans(text))

            file_count = Counter()

            for i, (start, end, sentence) in enumerate(spans):
                if len(sentence.split()) < 7:
                    continue

                # Reject extremely long parser artifacts.
                # Do not silently truncate quoted evidence.
                if len(sentence) > 3000:
                    continue

                entities = v34.find_entities(
                    sentence, aliases
                )

                # The reporting issuer is known even when
                # the sentence uses "we" or "our".
                if not entities:
                    continue

                cues = cue_matches(sentence)
                if not cues:
                    continue

                left = spans[max(0, i - 1)][0]
                right = spans[
                    min(len(spans) - 1, i + 1)
                ][1]

                context = text[left:right]

                if len(context) > 6500:
                    context = sentence
                    left = start

                eid = h(
                    doc["ticker"],
                    doc["accession"],
                    start,
                    end,
                )

                if eid in seen:
                    continue

                seen.add(eid)

                for rel in cues:
                    file_count[rel] += 1
                    total[rel] += 1

                writer.writerow({
                    "evidence_id": eid,
                    "ticker": doc["ticker"],
                    "form": doc["form"],
                    "accession": doc["accession"],
                    "filing_date": str(
                        doc["filing_date"].date()
                    ),
                    "available_from": str(
                        doc["available_from"].date()
                    ),
                    "source_sha256": expected_sha,
                    "offset_start": start,
                    "offset_end": end,
                    "context_start": left,
                    "sentence": sentence,
                    "context": context,
                    "named_entities": json.dumps(
                        entities, ensure_ascii=False
                    ),
                    "relation_cues": json.dumps(
                        cues, ensure_ascii=False
                    ),
                    "relation_hint": "|".join(cues),
                    "candidate_only": True,
                    "human_verified": False,
                })

            audit_rows.append({
                "ticker": doc["ticker"],
                "accession": doc["accession"],
                "sha256_pass": True,
                **{
                    rel: file_count[rel]
                    for rel in RELATIONS
                },
            })

            print(
                f"[{index}/{len(documents)}]"
                f" {doc['ticker']} "
                + " ".join(
                    f"{r}={file_count[r]}"
                    for r in RELATIONS
                ),
                flush=True,
            )

    pd.DataFrame(audit_rows).to_csv(
        OUT / "document_audit.csv",
        index=False,
    )

    result = {
        "documents_checked": len(documents),
        "unique_evidence": len(seen),
        "evidence_counts_by_relation_hint": dict(total),
        "four_relations_supported_by_pipeline": RELATIONS,
        "verified_silver_edges": 0,
        "strict_pit_approved": False,
        "human_verified": False,
        "source": "full_140_cached_SEC_documents",
    }

    (OUT / "report.json").write_text(
        json.dumps(
            result, indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print("\n=== V38 COMPLETE ===")
    print(json.dumps(
        result, indent=2, ensure_ascii=False
    ))

def audit():
    path = OUT / "full_sec_relation_evidence.csv"
    df = pd.read_csv(path, dtype={"accession": str})
    source = v34.prepare_documents()

    source_index = {
        (d["ticker"], str(d["accession"])): d
        for d in source
    }

    cache = {}
    for key, doc in source_index.items():
        raw = v34.cache_path(doc).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != str(doc["source_sha256"]):
            raise ValueError("SHA mismatch: " + str(key))
        cache[key] = v34.extract_text(raw)

    if df["evidence_id"].duplicated().any():
        raise ValueError("Duplicate evidence ID")

    for r in df.itertuples(index=False):
        key = (r.ticker, str(r.accession))
        doc = source_index[key]
        text = cache[key]

        a = int(r.offset_start)
        b = int(r.offset_end)
        c = int(r.context_start)

        if text[a:b] != r.sentence:
            raise ValueError("Sentence offset mismatch")

        if text[c:c + len(r.context)] != r.context:
            raise ValueError("Context offset mismatch")

        if r.source_sha256 != doc["source_sha256"]:
            raise ValueError("Provenance mismatch")

        if not (
            pd.Timestamp(r.available_from)
            > pd.Timestamp(r.filing_date)
        ):
            raise ValueError("Date chronology mismatch")

    print("\n=== V38 AUDIT PASS ===")
    print("Verified SEC documents:", len(source))
    print("Evidence rows checked:", len(df))
    print("All normalized-text offsets: PASS")
    print("Source SHA256: PASS")
    print("Availability chronology: PASS")
    print("Semantic relations: NOT YET VERIFIED")
    print("Strict PIT: NOT CERTIFIED")

def report():
    df = pd.read_csv(
        OUT / "full_sec_relation_evidence.csv"
    )

    print("\n=== V38 RELATION RETRIEVAL ===")
    print("Evidence:", len(df))

    for relation in RELATIONS:
        subset = df[
            df["relation_hint"].str.contains(
                relation, regex=False, na=False
            )
        ]

        print(
            f"\n{relation}: "
            f"{len(subset)} evidence candidates"
        )

        for r in subset.head(2).itertuples(index=False):
            print(" Ticker:", r.ticker)
            print(" Entities:", r.named_entities)
            print(" Sentence:", str(r.sentence)[:600])

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["build", "audit", "report"]
    )
    args = p.parse_args()

    if args.command == "build":
        build()
    elif args.command == "audit":
        audit()
    else:
        report()

if __name__ == "__main__":
    main()
