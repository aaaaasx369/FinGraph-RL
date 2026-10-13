#!/usr/bin/env python3
"""
V33: Ollama SEC Silver Label Extraction

Input:
  outputs/v31/evidence_alignment.parquet

Output:
  outputs/v33/<run-name>/raw_responses.jsonl
  outputs/v33/<run-name>/silver_labels.csv
  outputs/v33/<run-name>/silver_relations.csv
  outputs/v33/<run-name>/report.json

Uses ONLY the five investment companies:
AAPL MSFT NVDA AMZN GOOGL.

External companies are allowed as graph entities,
not additional portfolio assets.

No automatic human verification.
No automatic strict-PIT certification.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from ollama import Client

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/v31/evidence_alignment.parquet"
OUT = ROOT / "outputs/v33"

STOCKS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}

RELATIONS = {
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
}

NOVELTY = {"NEW", "REPEAT", "UNCERTAIN"}
EVENT = {"TRUE_EVENT", "TEXT_ONLY", "UNCERTAIN"}

SCHEMA = {
    "type": "object",
    "properties": {
        "novelty_label": {
            "type": "string",
            "enum": sorted(NOVELTY),
        },
        "event_label": {
            "type": "string",
            "enum": sorted(EVENT),
        },
        "reason": {"type": "string"},
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "relation": {
                        "type": "string",
                        "enum": sorted(RELATIONS),
                    },
                    "object": {"type": "string"},
                    "evidence_quote": {"type": "string"},
                },
                "required": [
                    "subject",
                    "relation",
                    "object",
                    "evidence_quote",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "novelty_label",
        "event_label",
        "reason",
        "relations",
    ],
    "additionalProperties": False,
}

SYSTEM = """
You are an evidence-constrained annotator for
historical SEC 10-K and 10-Q disclosures.

Your job is financial relation extraction, NOT
investment advice.

Tasks:
1. Compare the CURRENT excerpt to the retrieved
   historical excerpt.
2. Distinguish wording novelty from an actual
   economic change.
3. Extract only explicit, directed, named relations.
4. Never infer a specific company from "suppliers",
   "customers", "foundries", "partners", or pronouns.
5. Do not convert a longstanding relationship into
   a new economic event simply because the text
   appears in a newer SEC filing.
6. If direction is ambiguous, output no relation.
7. Return only relations supported by CURRENT text.
8. evidence_quote must be an exact contiguous
   substring of CURRENT excerpt.
9. The reporting company's ticker alone is not
   evidence that it is the subject of every relation.
10. No outside knowledge, guesses or inferred names.
11. Do not treat a proposed, hypothetical or risk
    scenario as an actual completed event.
12. SAME_SECTOR should not be guessed from SEC text;
    omit it unless the text explicitly supports it.

Relation semantics:
 SUPPLIES_TO: subject supplies products/services
              to object.
 CUSTOMER_OF: subject buys products/services
              from object.
 COMPETES_WITH: subject competes with object.
 SAME_SECTOR: explicit shared sector classification.

Novelty:
 NEW: available comparison suggests genuinely
      new textual information.
 REPEAT: same disclosure is repeated/rephrased.
 UNCERTAIN: insufficient historical context.

Economic event:
 TRUE_EVENT: concrete change/action explicitly
             described in CURRENT text.
 TEXT_ONLY: description, risk, recurring disclosure,
            or mere wording/numeric update without
            an identifiable economic event.
 UNCERTAIN: cannot establish confidently.

NEW text does not automatically imply TRUE_EVENT.
TRUE_EVENT does not require a named-company relation.
Relations may describe existing relationships; their
presence does NOT prove the relationship just began.
Return concise reasons.
"""

def normalize(s):
    return " ".join(str(s).split())

def validate(obj, current):
    if not isinstance(obj, dict):
        raise ValueError("Model response is not an object")
    if obj.get("novelty_label") not in NOVELTY:
        raise ValueError("Invalid novelty_label")
    if obj.get("event_label") not in EVENT:
        raise ValueError("Invalid event_label")

    rels = obj.get("relations")
    if not isinstance(rels, list):
        raise ValueError("relations must be list")

    clean = []
    seen = set()
    for rel in rels:
        if not isinstance(rel, dict):
            continue
        subject = normalize(rel.get("subject", ""))
        relation = rel.get("relation", "")
        target = normalize(rel.get("object", ""))
        quote = normalize(rel.get("evidence_quote", ""))

        # Relation extraction is anchored to
        # evidence, not just JSON schema validity.
        valid = (
            subject
            and target
            and relation in RELATIONS
            and len(quote) >= 12
            and quote in normalize(current)
            and subject.casefold() != target.casefold()
        )
        if not valid:
            continue

        key = (
            subject.casefold(),
            relation,
            target.casefold(),
            quote,
        )
        if key in seen:
            continue
        seen.add(key)

        clean.append({
            "subject": subject,
            "relation": relation,
            "object": target,
            "evidence_quote": quote,
        })

    obj["relations"] = clean
    obj["reason"] = normalize(obj.get("reason", ""))[:500]
    return obj

def prompt_for(row):
    payload = {
        "ticker": str(row["ticker"]),
        "topic": str(row["topic"]),
        "form": str(row["form"]),
        "filing_date": str(row["current_filing_date"]),
        "accession": str(row["current_accession"]),
        "v31_alignment_label": str(row["v31_label"]),
        "prior_filing_date": str(
            row["best_prior_filing_date"]
        ),
        "previous_excerpt": str(
            row["best_prior_excerpt"]
        )[:2200],
        "current_excerpt": str(
            row["current_excerpt"]
        )[:2200],
    }

    return (
        "Classify this SEC evidence pair. "
        "Use only supplied text. "
        "Output the required JSON schema.\n\n"
        + json.dumps(payload, ensure_ascii=False)
    )

def load_source(scope):
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_parquet(SOURCE)
    if set(df["ticker"]) != STOCKS:
        raise ValueError("Investment universe changed")
    if df["pair_id"].duplicated().any():
        raise ValueError("Duplicate pair_id")

    if scope == "candidates":
        df = df[
            df["v31_label"].eq("NEW_TEXT_CANDIDATE")
        ]
    elif scope == "review":
        selected = ROOT / "outputs/v32/sample_master.csv"
        if not selected.exists():
            raise FileNotFoundError(selected)
        ids = set(pd.read_csv(selected)["pair_id"])
        df = df[df["pair_id"].isin(ids)]

    return df.sort_values(
        ["current_filing_date", "ticker", "pair_id"]
    ).reset_index(drop=True)

def load_existing(path):
    results = {}
    if not path.exists():
        return results
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            # Retry failed rows on next run.
            if r.get("status") == "ok":
                results[r["pair_id"]] = r
    return results

def export(records, folder):
    labels = []
    relations = []
    for rec in records.values():
        if rec.get("status") != "ok":
            continue
        obj = rec["annotation"]

        labels.append({
            "pair_id": rec["pair_id"],
            "ticker": rec["ticker"],
            "topic": rec["topic"],
            "filing_date": rec["filing_date"],
            "accession": rec["accession"],
            "available_from": rec["available_from"],
            "v31_label": rec["v31_label"],
            "novelty_label": obj["novelty_label"],
            "event_label": obj["event_label"],
            "reason": obj["reason"],
            "relation_count": len(obj["relations"]),
            "model": rec["model"],
            "label_source": "AI_SILVER",
            "human_verified": False,
            "strict_pit_verified": False,
        })

        for r in obj["relations"]:
            relations.append({
                "pair_id": rec["pair_id"],
                "ticker": rec["ticker"],
                "filing_date": rec["filing_date"],
                "available_from": rec["available_from"],
                "accession": rec["accession"],
                "topic": rec["topic"],
                **r,
                "model": rec["model"],
                "label_source": "AI_SILVER",
                "human_verified": False,
                "strict_pit_verified": False,
            })

    pd.DataFrame(
        labels,
        columns=[
            "pair_id", "ticker", "topic", "filing_date",
            "accession", "available_from", "v31_label",
            "novelty_label", "event_label", "reason",
            "relation_count", "model", "label_source",
            "human_verified", "strict_pit_verified",
        ],
    ).to_csv(folder / "silver_labels.csv", index=False)

    pd.DataFrame(
        relations,
        columns=[
            "pair_id", "ticker", "filing_date",
            "available_from", "accession", "topic",
            "subject", "relation", "object",
            "evidence_quote", "model", "label_source",
            "human_verified", "strict_pit_verified",
        ],
    ).to_csv(folder / "silver_relations.csv", index=False)

    report = {
        "labeled_rows": len(labels),
        "silver_relation_mentions": len(relations),
        "event_label_counts": pd.Series(
            [r["event_label"] for r in labels]
        ).value_counts().to_dict(),
        "model": labels[0]["model"] if labels else None,
        "human_verified": False,
        "strict_pit_approved": False,
        "warning": (
            "Silver relations are not verified facts; "
            "no KG deployment or PIT approval implied."
        ),
    }
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(
        report, ensure_ascii=False, indent=2
    ))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default=os.getenv(
            "OLLAMA_HOST", "http://140.113.87.164:11434"
        ),
    )
    parser.add_argument("--model", default="qwen3.5:9b")
    parser.add_argument(
        "--scope",
        choices=["candidates", "review", "all"],
        default="candidates",
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--run-name", default="smoke_9b")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    if args.limit < 0:
        raise ValueError("--limit must be >= 0")

    folder = OUT / args.run_name
    folder.mkdir(parents=True, exist_ok=True)

    manifest_path = folder / "run_config.json"
    config = {
        "model": args.model,
        "scope": args.scope,
        "host": args.host,
        "schema_version": "v33_1",
    }
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != config:
            raise ValueError(
                "Existing run-name has different configuration. "
                "Use a new --run-name."
            )
    else:
        manifest_path.write_text(json.dumps(config, indent=2))

    df = load_source(args.scope)
    if args.limit:
        df = df.head(args.limit)

    raw_path = folder / "raw_responses.jsonl"
    completed = load_existing(raw_path)
    all_good = dict(completed)

    client = Client(host=args.host, timeout=args.timeout)
    print(
        f"Model: {args.model} | scope={args.scope} "
        f"| selected={len(df)} | completed={len(completed)}",
        flush=True,
    )

    with raw_path.open("a", encoding="utf-8") as log:
        for index, row in enumerate(df.to_dict("records"), 1):
            pair_id = str(row["pair_id"])
            if pair_id in completed:
                continue

            record = {
                "pair_id": pair_id,
                "ticker": row["ticker"],
                "topic": row["topic"],
                "accession": row["current_accession"],
                "filing_date": str(row["current_filing_date"]),
                "available_from": str(row["available_from"]),
                "v31_label": row["v31_label"],
                "model": args.model,
            }

            try:
                response = client.chat(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": prompt_for(row),
                        },
                    ],
                    format=SCHEMA,
                    options={
                        "temperature": 0,
                        "num_ctx": 4096,
                        "num_predict": 2048,
                    },
                    stream=False,
                )
                content = response.message.content or ""

                if not content.strip():
                    thinking = getattr(
                        response.message,
                        "thinking",
                        None
                    )
                    done_reason = getattr(
                        response,
                        "done_reason",
                        None
                    )

                    raise ValueError(
                        "EMPTY_CONTENT "
                        f"done_reason={done_reason} "
                        f"thinking_chars={len(thinking or '')}"
                    )

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "INVALID_JSON "
                        f"done_reason={getattr(response, 'done_reason', None)} "
                        f"content_preview={content[:300]!r}"
                    ) from exc

                parsed = validate(
                    parsed, str(row["current_excerpt"])[:2200]
                )
                record["status"] = "ok"
                record["annotation"] = parsed
                all_good[pair_id] = record
                print(
                    f"[{index}/{len(df)}] OK "
                    f"{row['ticker']} "
                    f"{parsed['event_label']} "
                    f"relations={len(parsed['relations'])}",
                    flush=True,
                )
            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)[:500]
                print(
                    f"[{index}/{len(df)}] ERROR: {exc}",
                    flush=True,
                )

            log.write(
                json.dumps(record, ensure_ascii=False, default=str)
                + "\n"
            )
            log.flush()

            # Save partial usable results after each row.
            export(all_good, folder)

    print("\n=== V33 FINISHED ===")
    print(f"Results: {folder}")

if __name__ == "__main__":
    main()
