#!/usr/bin/env python3
"""
V34.1 JSON compatibility repair.

Handles:
- Markdown ```json fences
- Bare JSON arrays
- {"decisions": [...]} objects
- quote / evidence / evidence_quote
- Missing optional fields
- Reprocessing existing raw Ollama responses
- Persistent repair of future V34.1 parsing

Does NOT fabricate SEC relationships.
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import v341_relation_extractor as v

MAIN = ROOT / "scripts/v341_relation_extractor.py"


def parse_llm_json(content):
    text = (content or "").strip()

    # Strip Markdown JSON fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    ).strip()

    obj = json.loads(text)

    # Accept bare JSON arrays.
    if isinstance(obj, list):
        obj = {"decisions": obj}

    if not isinstance(obj, dict):
        raise ValueError("JSON_ROOT_NOT_OBJECT_OR_ARRAY")

    decisions = obj.get("decisions")

    if not isinstance(decisions, list):
        raise ValueError("JSON_MISSING_DECISIONS_ARRAY")

    fixed = []

    for item in decisions:
        if not isinstance(item, dict):
            continue

        quote = item.get("evidence_quote")

        if quote is None:
            quote = item.get("quote")

        if quote is None:
            quote = item.get("evidence")

        fixed.append({
            "candidate_id": str(
                item.get("candidate_id", "")
            ),
            "supported": item.get("supported"),
            "evidence_quote": quote if quote is not None else "",
            "market_scope": (
                item.get("market_scope")
                if isinstance(item.get("market_scope"), str)
                else ""
            ),
            "reason": (
                item.get("reason")
                if isinstance(item.get("reason"), str)
                else ""
            ),
        })

    return {"decisions": fixed}


def repair_source():
    src = MAIN.read_text(encoding="utf-8")

    old = """parsed = json.loads(content)
            decisions = parsed.get("decisions")"""

    new = """from v341_repair import parse_llm_json
            parsed = parse_llm_json(content)
            decisions = parsed.get("decisions")"""

    if old in src:
        src = src.replace(old, new, 1)
        MAIN.write_text(src, encoding="utf-8")
        print("Future V34.1 JSON parsing: PATCHED")
    elif new in src:
        print("Future V34.1 JSON parsing: ALREADY PATCHED")
    else:
        raise RuntimeError(
            "Cannot locate original parser. "
            "Source code may have changed."
        )


def recover(run_name, max_candidates):
    run_dir = ROOT / "outputs/v341/runs" / run_name
    log_path = run_dir / "responses.jsonl"

    if not log_path.exists():
        raise FileNotFoundError(log_path)

    # Preserve the original log before modifying anything.
    backup = run_dir / "responses_before_repair.jsonl"

    if not backup.exists():
        backup.write_bytes(log_path.read_bytes())
        print("Original log backed up:", backup)

    groups, _ = v.make_batches(0, max_candidates)

    group_lookup = {
        str(g["evidence_id"].iloc[0]): g
        for g in groups
    }

    aliases = v.load_aliases()
    records = v.read_log(log_path)

    recovered_requests = 0
    recovered_edges = 0
    skipped = 0

    for eid, record in list(records.items()):

        if record.get("status") == "ok":
            skipped += 1
            continue

        content = record.get("raw_content", "")

        if not content.strip():
            print("SKIP EMPTY:", eid)
            continue

        group = group_lookup.get(eid)

        if group is None:
            print("SKIP UNKNOWN EVIDENCE:", eid)
            continue

        try:
            parsed = parse_llm_json(content)

            decisions = parsed["decisions"]

            by_id = {}
            duplicate_ids = set()

            for d in decisions:
                cid = d["candidate_id"]

                if cid in by_id:
                    duplicate_ids.add(cid)

                by_id[cid] = d

            expected = {
                str(r["candidate_id"])
                for r in group.to_dict("records")
            }

            unknown_ids = [
                cid for cid in by_id
                if cid not in expected
            ]

            verified = []
            missing_ids = []

            for r in group.to_dict("records"):

                cid = str(r["candidate_id"])
                d = by_id.get(cid)

                if d is None:
                    missing_ids.append(cid)
                    continue

                if cid in duplicate_ids:
                    accepted = False
                    validation_error = "DUPLICATE_MODEL_DECISION"
                else:
                    accepted, validation_error = (
                        v.validate_decision(
                            r, d, aliases
                        )
                    )

                # A missing field is recorded, never invented
                # as a positive relationship.
                verified.append({
                    "candidate_id": cid,
                    "evidence_id": eid,
                    "ticker": r["ticker"],
                    "subject": r["subject"],
                    "relation": r["relation"],
                    "object": r["object"],
                    "filing_date": r["filing_date"],
                    "available_from": r["available_from"],
                    "accession": r["accession"],
                    "source_sha256": r["source_sha256"],
                    "offset_start": r["offset_start"],
                    "offset_end": r["offset_end"],
                    "matched_alias": r["matched_alias"],
                    "supported": d["supported"],
                    "evidence_quote": d["evidence_quote"],
                    "market_scope": d["market_scope"],
                    "reason": d["reason"],
                    "validation_error": validation_error,
                    "accepted_silver": accepted,
                    "model": record["model"],
                    "label_source": "AI_SILVER",
                    "human_verified": False,
                    "strict_pit_approved": False,
                })

            accepted_count = sum(
                row["accepted_silver"]
                for row in verified
            )

            # An incomplete set stays retryable. Do not
            # silently mark missing candidate decisions
            # as successful model verifications.
            if missing_ids or duplicate_ids or unknown_ids:
                record["status"] = "error"
                record["error"] = (
                    "INCOMPLETE_MODEL_DECISIONS "
                    f"missing={len(missing_ids)} "
                    f"duplicates={len(duplicate_ids)} "
                    f"unknown={len(unknown_ids)}"
                )
                record["recovered_partial_decisions"] = verified
                print(
                    "PARTIAL:",
                    eid,
                    "accepted=", accepted_count,
                    "missing=", len(missing_ids),
                    "duplicates=", len(duplicate_ids),
                    "unknown=", len(unknown_ids),
                )
            else:
                record["status"] = "ok"
                record.pop("error", None)
                record["verified_decisions"] = verified
                record["recovered_from_raw_json"] = True

                recovered_requests += 1
                recovered_edges += accepted_count

                print(
                    "RECOVERED:",
                    eid,
                    "silver=", accepted_count,
                    "decisions=", len(verified),
                )

            # Append rather than overwrite original history.
            with log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        default=str,
                    ) + "\n"
                )
                f.flush()

            records[eid] = record

        except Exception as e:
            print(
                "RECOVERY ERROR:",
                eid,
                type(e).__name__,
                str(e),
            )

    v.export_results(records, run_dir)

    print("\n=== V34.1 RECOVERY SUMMARY ===")
    print("Recovered requests:", recovered_requests)
    print("Recovered silver mentions:", recovered_edges)
    print("Already successful:", skipped)

    print("\n=== UPDATED REPORT ===")
    v.show_report(run_name)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="pilot_27b",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    repair_source()
    recover(args.run_name, args.max_candidates)


if __name__ == "__main__":
    main()
