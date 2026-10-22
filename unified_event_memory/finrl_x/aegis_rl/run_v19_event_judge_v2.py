#!/usr/bin/env python3

import json
import os
import argparse
from pathlib import Path

import pandas as pd
import ollama

ROOT = Path("results/aegis_rl_v1")
OUT = ROOT / "v19_final/event_judge_v2"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_CSV = ROOT / "v18_2_audit/event_review.csv"
B7_JSONL = ROOT / "v18_b7_matched_20/audit.jsonl"
ORIGINAL_JSONL = ROOT / (
    "v17_joint_rag_ppo_smoke/joint_rag_cache.jsonl"
)

FIELDS = {
    "occurrence": [
        "confirmed", "historical", "planned",
        "hypothetical", "unknown"
    ],
    "outcome": [
        "realized", "not_established", "unknown"
    ],
    "potential": [
        "positive", "negative", "neutral", "unknown"
    ],
    "realized": [
        "positive", "negative", "neutral", "unknown"
    ],
}

SCHEMA = {
    "type": "object",
    "properties": {
        "occurrence": {
            "type": "string",
            "enum": FIELDS["occurrence"]
        },
        "outcome": {
            "type": "string",
            "enum": FIELDS["outcome"]
        },
        "potential": {
            "type": "string",
            "enum": FIELDS["potential"]
        },
        "realized": {
            "type": "string",
            "enum": FIELDS["realized"]
        },
        "reason": {
            "type": "string"
        },
        "evidence_quote": {
            "type": "string"
        },
    },
    "required": [
        "occurrence", "outcome",
        "potential", "realized",
        "reason", "evidence_quote"
    ],
    "additionalProperties": False
}


def load_jsonl(path):
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["date"]), str(row["ticker"]))
        if key in output:
            raise RuntimeError(f"Duplicate {key}")
        output[key] = row
    return output


def check_result(result):
    if not isinstance(result, dict):
        raise ValueError("JSON object required")

    for field, choices in FIELDS.items():
        if result.get(field) not in choices:
            raise ValueError(
                f"Invalid {field}: {result.get(field)!r}"
            )

    for field in ("reason", "evidence_quote"):
        if not isinstance(result.get(field), str):
            raise ValueError(f"Invalid {field}")

    return result


def evidence_for_claim(row, b7, original):
    key = (str(row["date"]), str(row["ticker"]))

    if key not in b7 or key not in original:
        raise RuntimeError(f"Missing source row: {key}")

    claim_id = str(row["claim_id"])
    claim_items = b7[key].get("case", {}).get("claims", [])

    matched_claims = [
        c for c in claim_items
        if str(c.get("claim_id")) == claim_id
    ]

    if len(matched_claims) != 1:
        raise RuntimeError(
            f"Claim ID cannot be resolved: {key} {claim_id}"
        )

    claim = matched_claims[0]
    evidence_id = str(claim.get("sec_evidence_id", ""))

    hits = original[key].get("sec_evidence", [])
    matching = [
        h for h in hits
        if str(h.get("id")) == evidence_id
    ]

    if len(matching) != 1:
        raise RuntimeError(
            f"SEC evidence cannot be resolved: {key} {evidence_id}"
        )

    hit = matching[0]
    day = pd.Timestamp(key[0])
    available = pd.Timestamp(hit["available_from"])

    if not available < day:
        raise RuntimeError(
            f"PIT violation: {key} {evidence_id}"
        )

    return claim, hit


def judge(client, model, row, claim, hit):
    evidence = str(hit.get("text", ""))

    if not evidence.strip():
        raise ValueError("Empty SEC evidence")

    prompt = {
        "decision_date": str(row["date"]),
        "ticker": str(row["ticker"]),
        "claim": str(claim.get("final_claim", "")),
        "sec_evidence": evidence,
        "task": (
            "Classify the SEC claim using ONLY the available "
            "evidence. Do not use financial knowledge outside "
            "the provided document."
        ),
        "definitions": {
            "occurrence": {
                "confirmed": "Explicitly confirmed as an event relevant to the statement",
                "historical": "Describes an event occurring in an earlier historical period",
                "planned": "A future plan or intention",
                "hypothetical": "A possible event or conditional risk",
                "unknown": "Cannot determine occurrence"
            },
            "outcome": {
                "realized": "The claimed outcome is explicitly documented as having occurred",
                "not_established": "An outcome has not been established by the evidence",
                "unknown": "Insufficient information to distinguish"
            },
            "potential": {
                "positive": "Explicit positive potential business impact",
                "negative": "Explicit negative potential business impact",
                "neutral": "Evidence explicitly establishes no meaningful impact",
                "unknown": "No supported impact direction"
            },
            "realized": {
                "positive": "Explicitly established positive realized business impact",
                "negative": "Explicitly established negative realized business impact",
                "neutral": "Explicitly established zero/neutral realized impact",
                "unknown": "No established realized impact direction"
            }
        },
        "rules": [
            "Return ONLY the JSON object required by the schema.",
            "Use exactly one allowed enum value per classification field.",
            "Never output boolean True or False.",
            "Do not replace enum values with descriptive sentences.",
            "Unknown is not neutral.",
            "A stated risk is not an observed loss.",
            "An investigation is not proof of an adverse outcome.",
            "Historical occurrence is not necessarily a realized business impact.",
            "Use unknown when the evidence is insufficient.",
            "Do not repeat the model's earlier B8 classifications.",
            "Give a short exact supporting quote from the evidence."
        ]
    }

    response = client.chat(
        model=model,
        format=SCHEMA,
        options={"temperature": 0},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a financial SEC disclosure evidence judge. "
                    "Return only the exact requested JSON schema. "
                    "Conclusions must be strictly supported by the text."
                )
            },
            {
                "role": "user",
                "content": json.dumps(
                    prompt, ensure_ascii=False
                )
            }
        ]
    )

    result = check_result(
        json.loads(response["message"]["content"])
    )

    quote = result["evidence_quote"].strip()

    if quote and quote not in evidence:
        raise ValueError(
            "Judge returned an evidence quote not found verbatim "
            "in the source SEC document"
        )

    if not quote:
        # No attributable quote -> abstain rather than asserting direction.
        result["occurrence"] = "unknown"
        result["outcome"] = "unknown"
        result["potential"] = "unknown"
        result["realized"] = "unknown"

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--host",
        default=os.getenv(
            "OLLAMA_HOST",
            "http://140.113.87.164:11434"
        )
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    events = pd.read_csv(
        EVENT_CSV, dtype=str
    ).fillna("")

    b7 = load_jsonl(B7_JSONL)
    original = load_jsonl(ORIGINAL_JSONL)

    client = ollama.Client(host=args.host, timeout=180)

    selected = (
        events.head(args.limit)
        if args.limit else events
    )

    records = []
    cache_path = OUT / "judge_cache.jsonl"
    cache = {}

    if cache_path.exists():
        for line in cache_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("model") == args.model:
                cache[record["key"]] = record

    for i, (_, row) in enumerate(selected.iterrows(), 1):
        key = "|".join([
            str(row["date"]),
            str(row["ticker"]),
            str(row["claim_id"])
        ])

        claim, hit = evidence_for_claim(
            row, b7, original
        )

        if key in cache:
            record = cache[key]
        else:
            try:
                result = judge(
                    client, args.model, row, claim, hit
                )
                status = "CLASSIFIED"
                error = ""
            except Exception as exc:
                result = {
                    "occurrence": "unknown",
                    "outcome": "unknown",
                    "potential": "unknown",
                    "realized": "unknown",
                    "reason": "",
                    "evidence_quote": ""
                }
                status = "REVIEW_REQUIRED"
                error = repr(exc)

            record = {
                "key": key,
                "model": args.model,
                "date": str(row["date"]),
                "ticker": str(row["ticker"]),
                "claim_id": str(row["claim_id"]),
                "sec_evidence_id": str(
                    claim.get("sec_evidence_id", "")
                ),
                "status": status,
                "error": error,
                "result": result
            }

            with cache_path.open(
                "a", encoding="utf-8"
            ) as f:
                f.write(
                    json.dumps(
                        record, ensure_ascii=False
                    ) + "\n"
                )

            cache[key] = record

        records.append(record)

        print(
            f"[{i}/{len(selected)}]",
            key,
            record["status"],
            record["result"]["occurrence"],
            record["result"]["potential"],
            flush=True
        )

    new_rows = []

    for record in records:
        day, ticker, claim_id = record["key"].split("|", 2)

        matches = events[
            (events.date == day)
            & (events.ticker == ticker)
            & (events.claim_id == claim_id)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Ambiguous event row: {record['key']}"
            )

        row = matches.iloc[0].to_dict()

        result = record["result"]

        row.update({
            "judge_status": record["status"],
            "judge_error": record["error"],
            "judge_occurrence": result["occurrence"],
            "judge_outcome": result["outcome"],
            "judge_potential": result["potential"],
            "judge_realized": result["realized"],
            "judge_reason": result["reason"],
            "judge_evidence_quote": result["evidence_quote"],
        })

        new_rows.append(row)

    output = OUT / (
        "event_judge_v2_partial.csv"
        if args.limit else "event_judge_v2.csv"
    )

    pd.DataFrame(new_rows).to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    if args.limit:
        print("\nPARTIAL TEST COMPLETE")
        print("File:", output)
        return

    df = pd.DataFrame(new_rows)
    counts = {}
    disputes = []

    for field in FIELDS:
        p = df[f"predicted_{field}"].astype(str).str.strip().str.lower()
        q = df[f"judge_{field}"].astype(str).str.strip().str.lower()

        valid = (
            df["judge_status"].eq("CLASSIFIED")
            & p.isin(FIELDS[field])
            & q.isin(FIELDS[field])
        )

        agree = p.eq(q)

        counts[field] = {
            "total": len(df),
            "comparable": int(valid.sum()),
            "agreement": (
                float(agree[valid].mean())
                if valid.any() else None
            ),
            "disagreements": int(
                (valid & ~agree).sum()
            ),
            "not_comparable": int((~valid).sum())
        }

        bad = df[valid & ~agree].copy()
        bad["comparison_field"] = field
        disputes.append(bad)

    if disputes:
        disagreements = pd.concat(
            disputes, ignore_index=True
        )
        disagreements.to_csv(
            OUT / "event_disagreements_v2.csv",
            index=False,
            encoding="utf-8-sig"
        )

    report = {
        "evaluation_type": "STRICT_SCHEMA_AI_JUDGE_AGREEMENT",
        "model": args.model,
        "n_event_rows": len(df),
        "status_counts": df.judge_status.value_counts().to_dict(),
        "field_agreement": counts,
        "human_ground_truth": False,
        "limitations": [
            "Qwen as judge is not an independent human reference.",
            "Agreement is not predictive accuracy.",
            "Only explicitly available SEC evidence was used.",
            "Failed or malformed outputs abstain.",
            "Classification category definitions may differ from B8."
        ]
    }

    (OUT / "event_quality_v2.json").write_text(
        json.dumps(
            report, indent=2, ensure_ascii=False
        )
    )

    print("\n========== V19.1 EVENT AGREEMENT ==========")
    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))

    print("\nV19.1 EVENT JUDGE: COMPLETE")
    print("Frozen PPO: UNCHANGED")
    print("2022-2023 Test: NOT ACCESSED")


if __name__ == "__main__":
    main()
