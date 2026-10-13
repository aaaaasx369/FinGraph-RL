#!/usr/bin/env python3
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASE = (
    ROOT
    / "outputs/finrl_dkg_original_full/four_relation_pipeline"
)

CHUNKS = (
    ROOT / "rag/data/processed/sec/sec_chunks.jsonl"
)

MANIFEST = (
    ROOT / "outputs/v20/sec_filings_manifest.csv"
)

CANDIDATES = BASE / "relation_candidates.csv"

ACCESSION_RE = re.compile(
    r"(?<!\d)\d{10}-\d{2}-\d{6}(?!\d)"
)

BAD_ENTITY_TICKERS = {
    "GOOGM", "GOOGN", "HXSCL", "SKHY",
    "BRK-A", "BRK-B",
}

def flat(obj, prefix="", depth=0):
    if depth > 5 or not isinstance(obj, dict):
        return {}

    result = {}

    for key, value in obj.items():
        name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flat(value, name, depth + 1))
        elif isinstance(value, (str, int, float)):
            result[name] = str(value)

    return result


def accession_from(value):
    match = ACCESSION_RE.search(str(value))
    return match.group() if match else ""


def find_identifiers(obj):
    fields = flat(obj)

    accessions = set()
    evidence_ids = set()
    urls = set()

    for key, value in fields.items():
        lower = key.lower()

        if any(x in lower for x in [
            "accession", "evidence_id", "filing",
            "document", "source", "url", "path",
            "id",
        ]):
            acc = accession_from(value)

            if acc:
                accessions.add(acc)

        if (
            "evidence_id" in lower
            and value.startswith("EDGAR:")
        ):
            evidence_ids.add(value)

        if (
            ("url" in lower or "source" in lower)
            and value.startswith("https://")
        ):
            urls.add(value)

    return accessions, evidence_ids, urls, fields


def index_manifest(df):
    by_acc = defaultdict(list)
    by_evidence = defaultdict(list)
    by_url = defaultdict(list)

    for idx, row in df.iterrows():
        acc = str(row["accession"]).strip()
        evidence = str(row["evidence_id"]).strip()
        url = str(row["document_url"]).strip()

        if acc:
            by_acc[acc].append(idx)

        if evidence:
            by_evidence[evidence].append(idx)

        if url:
            by_url[url].append(idx)

    return by_acc, by_evidence, by_url


def main():
    BASE.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(
        CANDIDATES, dtype=str
    ).fillna("")

    manifest = pd.read_csv(
        MANIFEST, dtype=str
    ).fillna("")

    # Prevent ambiguous manifest matches.
    manifest["ticker"] = (
        manifest["ticker"]
        .str.upper()
        .str.strip()
    )

    by_acc, by_evidence, by_url = index_manifest(
        manifest
    )

    candidate_lines = set(
        pd.to_numeric(
            candidates["source_line"],
            errors="coerce",
        ).dropna().astype(int)
    )

    raw_chunks = {}

    with CHUNKS.open(
        encoding="utf-8",
        errors="replace",
    ) as stream:
        for line_no, line in enumerate(
            stream, start=1
        ):
            if line_no in candidate_lines:
                raw_chunks[line_no] = json.loads(line)

    records = []
    diagnostics = []

    for row in candidates.to_dict("records"):
        line_no = int(row["source_line"])
        chunk = raw_chunks.get(line_no)

        if chunk is None:
            raise RuntimeError(
                f"Missing source line {line_no}"
            )

        accessions, evidence_ids, urls, fields = (
            find_identifiers(chunk)
        )

        # Exact identifiers can resolve a filing.
        matches = set()

        for acc in accessions:
            matches.update(by_acc.get(acc, []))

        for evidence in evidence_ids:
            matches.update(
                by_evidence.get(evidence, [])
            )

        for url in urls:
            matches.update(by_url.get(url, []))

        issuer = row["issuer"].upper()

        # Reject contradicting identifying fields:
        # never select an arbitrary matching filing.
        matches = {
            idx for idx in matches
            if manifest.at[idx, "ticker"] == issuer
        }

        status = "UNRESOLVED"
        matched = None

        if len(matches) == 1:
            matched = manifest.loc[
                next(iter(matches))
            ]

            status = "EXACT_IDENTIFIER_MATCH"

        elif len(matches) > 1:
            status = "AMBIGUOUS_MULTIPLE_FILINGS"

        if not accessions and not evidence_ids and not urls:
            status = "NO_FILING_IDENTIFIER"

        enriched = dict(row)
        enriched["provenance_status"] = status
        enriched["source_accessions"] = ";".join(
            sorted(accessions)
        )
        enriched["source_evidence_ids"] = ";".join(
            sorted(evidence_ids)
        )

        for name in [
            "matched_accession",
            "matched_filing_date",
            "acceptance_utc",
            "available_date",
            "source_url",
            "metadata_verified",
            "text_verified",
        ]:
            enriched[name] = ""

        if matched is not None:
            enriched["matched_accession"] = (
                matched["accession"]
            )
            enriched["matched_filing_date"] = (
                matched["filing_date"]
            )
            enriched["acceptance_utc"] = (
                matched["acceptance_utc"]
            )
            enriched["available_date"] = (
                matched["available_date"]
            )
            enriched["source_url"] = (
                matched["document_url"]
            )
            enriched["metadata_verified"] = (
                matched["metadata_verified"]
            )
            enriched["text_verified"] = (
                matched["text_verified"]
            )

        # Share-class aliases are the same issuer.
        # Other suspicious ticker matches remain
        # quarantined, not automatically discarded.
        if (
            enriched["head"] == "GOOGL"
            and enriched["tail"] == "GOOG"
        ):
            enriched["entity_status"] = (
                "SELF_COMPANY_SHARE_CLASS"
            )
        elif (
            enriched["head"] in BAD_ENTITY_TICKERS
            or enriched["tail"] in BAD_ENTITY_TICKERS
        ):
            enriched["entity_status"] = (
                "IDENTITY_REVIEW_REQUIRED"
            )
        else:
            enriched["entity_status"] = (
                "UNREVIEWED"
            )

        enriched["semantic_status"] = "PENDING"

        records.append(enriched)

        if len(diagnostics) < 8:
            diagnostics.append({
                "source_line": line_no,
                "metadata_fields": {
                    k: v[:120]
                    for k, v in fields.items()
                    if any(x in k.lower() for x in [
                        "filing", "accession",
                        "document", "source",
                        "url", "date", "id",
                    ])
                },
            })

    result = pd.DataFrame(records)

    result.to_csv(
        BASE / "relation_provenance_v2.csv",
        index=False,
    )

    result["review_status"] = "PENDING"

    result.to_csv(
        BASE / "relation_review_v2.csv",
        index=False,
    )

    report = {
        "total_candidates": len(result),
        "provenance_counts": (
            result["provenance_status"]
            .value_counts()
            .to_dict()
        ),
        "entity_counts": (
            result["entity_status"]
            .value_counts()
            .to_dict()
        ),
        "source_examples": diagnostics,
        "strict_pit_approved": False,
    }

    (BASE / "provenance_repair_report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== PROVENANCE REPAIR ===")

    print("Candidates:", len(result))

    print("\nPROVENANCE")
    print(
        result["provenance_status"]
        .value_counts()
        .to_string()
    )

    print("\nENTITY REVIEW")
    print(
        result["entity_status"]
        .value_counts()
        .to_string()
    )

    confirmed = (
        result["provenance_status"]
        .eq("EXACT_IDENTIFIER_MATCH")
        & result["acceptance_utc"].ne("")
        & result["metadata_verified"].eq("1")
        & result["text_verified"].eq("1")
    )

    print(
        "\nFiling metadata verified:",
        int(confirmed.sum()),
        "/", len(result)
    )

    print("\nSaved:")
    print(
        BASE / "relation_provenance_v2.csv"
    )
    print(
        BASE / "relation_review_v2.csv"
    )
    print(
        BASE / "provenance_repair_report.json"
    )

    print("\nStrict PIT: NOT CERTIFIED")
    print("KGTransformer training: NOT STARTED")


if __name__ == "__main__":
    main()
