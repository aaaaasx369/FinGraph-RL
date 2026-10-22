#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17-B8: Event–Outcome Disentanglement

Input:
    V17-B7 AlignScore audit.jsonl

Reuses:
    B7 verified SEC claims and quotations
    B7 AlignScore checks
    B7 KG structural features

New:
    Per-claim event/outcome classification
    Strict Pydantic JSON schema
    Temporal consistency constraints
    One targeted classification retry
    Per-claim SQLite caching
    Full audit and B7/B8 comparison

This is an experimental classifier.
No PPO training.
No access to 2022–2023 test data.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import ollama
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


VERSION = "v17b8_event_outcome_1"

EVENT_TYPES = [
    "investigation",
    "capital_expenditure_plan",
    "acquisition_plan",
    "supply_chain_risk",
    "competition_risk",
    "regulatory_risk",
    "financial_disclosure",
    "cost_driver",
    "other",
    "unknown",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class B8Classification(Strict):
    event_type: Literal[
        "investigation",
        "capital_expenditure_plan",
        "acquisition_plan",
        "supply_chain_risk",
        "competition_risk",
        "regulatory_risk",
        "financial_disclosure",
        "cost_driver",
        "other",
        "unknown",
    ]

    event_occurrence: Literal[
        "confirmed",
        "announced_plan",
        "hypothetical",
        "historical",
        "unknown",
    ]

    outcome_status: Literal[
        "confirmed",
        "not_established",
        "not_applicable",
        "unknown",
    ]

    potential_business_impact: Literal[
        "positive",
        "negative",
        "neutral",
        "unknown",
    ]

    realized_business_impact: Literal[
        "positive",
        "negative",
        "neutral",
        "unknown",
    ]

    outcome_evidence: str = Field(
        description=(
            "Exact quote substring supporting a "
            "confirmed outcome; otherwise empty."
        )
    )

    reasoning: str = Field(min_length=15)


CLASSIFY_PROMPT = """
You classify ONE historical SEC evidence-supported claim.

CRUCIAL: Separate the existence of an event from
whether the event achieved a legal or financial outcome.

Event occurrence:
- confirmed: quote explicitly states that an
  event occurred, such as an investigation opened.
- announced_plan: the company states an intended
  or expected future action.
- hypothetical: a conditional possibility, such
  as something that may happen if a risk materializes.
- historical: past activities or historical financial
  statements, where current novelty is not implied.
- unknown: insufficient information.

Outcome status:
- confirmed: the quote explicitly establishes a
  specific realized legal or financial outcome.
- not_established: an outcome is relevant but the
  quote does not establish whether it happened.
- not_applicable: the claim does not concern a
  distinct outcome beyond the stated fact.
- unknown: unclear whether an outcome is involved.

Examples of correct distinctions:
1. "The Commission has opened an investigation."
   investigation existence = confirmed.
   violation/penalty outcome = not_established.
   realized business loss = unknown.

2. "We expect capital expenditures to increase."
   occurrence = announced_plan.
   future expenditure realization = not_established.
   realized impact = unknown.

3. "If supplies are delayed, sales may decline."
   occurrence = hypothetical.
   outcome = not_established.
   potential impact = negative.
   realized impact = unknown.

4. "Cost of revenues may be affected by mix."
   occurrence = hypothetical.
   potential impact = unknown if direction unspecified.
   realized impact = unknown.

5. "A prior tax audit was settled."
   occurrence = historical.
   outcome = confirmed only if the quote explicitly
   describes a settlement, and outcome_evidence
   quotes the exact settlement language.
   Do not infer savings or lower expenses.

Potential impact:
- Only assign positive/negative when the quote
  explicitly supports that direction.
- Never translate business impact into stock return.
- If a risk is mentioned without a directional
  consequence, choose unknown.

Realized impact:
- positive/negative/neutral only if the quoted
  text explicitly supports a realized effect on
  business or finances.
- Investigation opening does not prove a fine.
- Announced plans do not prove completed outcomes.
- "Has not had a material adverse effect" does not
  automatically prove a positive financial benefit.

Outcome evidence:
- If outcome_status is confirmed, return a short,
  EXACT contiguous substring from the quote which
  states that the outcome occurred.
- Otherwise return an empty string.
- Do not paraphrase the outcome evidence.

Use ONLY the supplied quote and verified claim.
Do not infer facts from outside knowledge.
Do not infer economic meaning from KG relation codes.
Return exactly the JSON schema provided.
"""


def js(obj):
    return json.dumps(
        obj, ensure_ascii=False,
        sort_keys=True, default=str
    )


def sha(obj):
    return hashlib.sha256(
        js(obj).encode("utf-8")
    ).hexdigest()[:24]


def atomic_write(path, content):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


class Cache:
    def __init__(self, path):
        self.conn = sqlite3.connect(str(path))
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS answers (
                key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
        """)
        self.conn.commit()

    def get(self, key):
        found = self.conn.execute(
            "SELECT payload FROM answers WHERE key=?",
            (key,),
        ).fetchone()
        return json.loads(found[0]) if found else None

    def put(self, key, result):
        self.conn.execute(
            "INSERT OR REPLACE INTO answers "
            "(key, payload) VALUES (?, ?)",
            (key, js(result)),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


def ask(client, model, quote, claim, feedback=None):
    prompt = CLASSIFY_PROMPT

    if feedback:
        prompt += (
            "\nYour previous classification failed "
            "a deterministic evidence check. "
            "Correct the error without adding facts.\n"
            "Failure: " + feedback[:500]
        )

    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": js({
                    "quote": quote,
                    "verified_claim": claim,
                }),
            },
        ],
        format=B8Classification.model_json_schema(),
        options={
            "temperature": 0,
            "num_predict": 900,
        },
    )

    return B8Classification.model_validate_json(
        response.message.content
    )


def validate_semantics(cls, quote, claim):
    q = " ".join(quote.lower().split())
    c = " ".join(claim.lower().split())
    all_text = q + " " + c

    if (
        cls.event_occurrence in [
            "announced_plan", "hypothetical"
        ]
        and cls.realized_business_impact != "unknown"
    ):
        raise ValueError(
            "NONREALIZED_EVENT_WITH_REALIZED_IMPACT"
        )

    if (
        cls.event_occurrence == "hypothetical"
        and cls.outcome_status == "confirmed"
    ):
        raise ValueError(
            "HYPOTHETICAL_WITH_CONFIRMED_OUTCOME"
        )

    if cls.outcome_status == "confirmed":
        evidence = cls.outcome_evidence.strip()

        if (
            len(evidence) < 12
            or evidence not in quote
        ):
            raise ValueError(
                "CONFIRMED_OUTCOME_MISSING_EXACT_EVIDENCE"
            )

    elif cls.outcome_evidence.strip():
        raise ValueError(
            "NONCONFIRMED_OUTCOME_HAS_EVIDENCE"
        )

    if (
        cls.realized_business_impact != "unknown"
        and cls.outcome_status != "confirmed"
    ):
        raise ValueError(
            "REALIZED_IMPACT_WITHOUT_CONFIRMED_OUTCOME"
        )

    # High-precision anchors. These are guards,
    # not a complete linguistic temporal parser.
    opened_investigation = bool(
        re.search(
            r"\b(opened|has opened|launched|"
            r"initiated|began)\s+"
            r"(?:an?\s+)?investigation\b",
            all_text,
        )
    )

    if opened_investigation:
        if cls.event_occurrence not in [
            "confirmed", "historical"
        ]:
            raise ValueError(
                "OPENED_INVESTIGATION_NOT_CONFIRMED"
            )

        if cls.event_type != "investigation":
            raise ValueError(
                "INVESTIGATION_EVENT_TYPE_MISMATCH"
            )

    planned_spending = bool(
        re.search(
            r"\b(expect|expects|plan|plans|intend|"
            r"intends|anticipate|anticipates)\b"
            r".{0,90}\b("
            r"capital expenditures|"
            r"capital spending|"
            r"expenditures|"
            r"acquisitions)\b",
            all_text,
        )
    )

    if (
        planned_spending
        and cls.event_occurrence == "hypothetical"
    ):
        raise ValueError(
            "ANNOUNCED_PLAN_AS_HYPOTHETICAL"
        )

    return True


def load_b7(path):
    if not path.exists():
        raise FileNotFoundError(path)

    records = {}

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        row = json.loads(line)
        key = (row["date"], row["ticker"])
        records[key] = row

    result = [
        records[key]
        for key in sorted(records)
    ]

    if not result:
        raise ValueError("B7 audit is empty")

    return result


def accepted_b7_claims(row, threshold):
    output = []

    for claim in row["case"].get("claims", []):
        final = claim.get("final_check", {})

        score = final.get("alignment_score")

        if (
            final.get("independent_verified") is not True
            or score is None
            or float(score) < threshold
            or claim.get("classification") is None
        ):
            continue

        output.append(claim)

    return output


def process_claim(client, args, claim):
    quote = claim["quote"]
    statement = claim["final_claim"]

    last_error = None
    attempts = []

    for attempt in range(2):
        try:
            result = ask(
                client,
                args.llm,
                quote,
                statement,
                feedback=last_error,
            )

            validate_semantics(
                result, quote, statement
            )

            return {
                "status": "CLASSIFIED",
                "classification": result.model_dump(),
                "attempts": attempts + [{
                    "attempt": attempt + 1,
                    "status": "PASS",
                }],
            }

        except Exception as exc:
            last_error = str(exc)

            attempts.append({
                "attempt": attempt + 1,
                "status": "REJECTED",
                "reason": last_error[:800],
            })

    return {
        "status": "REVIEW_REQUIRED",
        "classification": None,
        "attempts": attempts,
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--b7-audit",
        default=(
            "results/aegis_rl_v1/"
            "v17b7_alignscore_20/audit.jsonl"
        ),
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument(
        "--llm", default="qwen2.5:3b"
    )
    p.add_argument(
        "--b7-threshold",
        type=float, default=0.5
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v17b8_event_outcome_20"
        ),
    )

    args = p.parse_args()

    args.out.mkdir(
        parents=True, exist_ok=True
    )

    records = load_b7(
        Path(args.b7_audit)
    )

    for record in records:
        date = pd.Timestamp(record["date"])

        if not (
            pd.Timestamp("2018-01-01")
            <= date
            < pd.Timestamp("2022-01-01")
        ):
            raise ValueError(
                "Unexpected test-period date"
            )

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    models = {
        x.model for x in client.list().models
    }

    if args.llm not in models:
        raise RuntimeError(
            "Qwen model not available: " + args.llm
        )

    cache = Cache(
        args.out / "cache.sqlite"
    )

    rows = []
    audits = []
    cache_hits = 0
    model_calls = 0

    try:
        for record in records:
            date = record["date"]
            ticker = record["ticker"]

            claims = accepted_b7_claims(
                record,
                args.b7_threshold,
            )

            decisions = []

            for claim in claims:
                request = {
                    "version": VERSION,
                    "model": args.llm,
                    "host": args.ollama_host,
                    "quote": claim["quote"],
                    "final_claim": claim["final_claim"],
                    "sec_evidence_id": claim[
                        "sec_evidence_id"
                    ],
                    "schema": (
                        B8Classification.model_json_schema()
                    ),
                    "instruction": CLASSIFY_PROMPT,
                }

                key = sha(request)

                outcome = cache.get(key)

                if outcome is None:
                    outcome = process_claim(
                        client, args, claim
                    )
                    model_calls += len(
                        outcome["attempts"]
                    )
                    cache.put(key, outcome)
                else:
                    cache_hits += 1

                decisions.append({
                    "claim_id": claim["claim_id"],
                    "quote": claim["quote"],
                    "final_claim": claim["final_claim"],
                    "sec_evidence_id": claim[
                        "sec_evidence_id"
                    ],
                    "alignscore": claim[
                        "final_check"
                    ]["alignment_score"],
                    "b7_classification": claim.get(
                        "classification"
                    ),
                    "b8": outcome,
                })

            valid = [
                x for x in decisions
                if x["b8"]["status"] == "CLASSIFIED"
            ]

            failed = [
                x for x in decisions
                if x["b8"]["status"] != "CLASSIFIED"
            ]

            if not claims:
                status = "NO_VERIFIED_CLAIMS"
            elif failed:
                status = "REVIEW_REQUIRED"
            else:
                status = "CLASSIFIED"

            row = {
                "date": date,
                "ticker": ticker,
                "b7_verified_claims": len(claims),
                "b8_classified_claims": len(valid),
                "b8_review_claims": len(failed),
                "status": status,
                "kg_edge_count": record.get(
                    "kg_edge_count"
                ),
            }

            for category in [
                "event_occurrence",
                "event_type",
                "outcome_status",
                "potential_business_impact",
                "realized_business_impact",
            ]:
                counts = Counter(
                    item["b8"]["classification"][
                        category
                    ]
                    for item in valid
                )

                row[category + "_labels"] = (
                    js(dict(counts))
                )

            rows.append(row)

            audits.append({
                **row,
                "claim_decisions": decisions,
            })

            # Persist after each company-date.
            pd.DataFrame(rows).to_csv(
                args.out / "signals.csv",
                index=False,
            )

            atomic_write(
                args.out / "audit.jsonl",
                "".join(
                    js(x) + "\n"
                    for x in audits
                ),
            )

            print(
                "[B8]",
                date,
                ticker,
                "status=", status,
                "classified=", len(valid),
                "review=", len(failed),
                flush=True,
            )

    finally:
        cache.close()

    statuses = Counter(
        x["status"]
        for x in rows
    )

    manifest = {
        "version": VERSION,
        "input_b7_audit": args.b7_audit,
        "ollama_host": args.ollama_host,
        "llm": args.llm,
        "threshold_inherited_from_b7": (
            args.b7_threshold
        ),
        "company_dates": len(rows),
        "statuses": dict(statuses),
        "cache_hits": cache_hits,
        "model_calls": model_calls,
        "method": (
            "Separating SEC event occurrence, "
            "outcome status and business impact"
        ),
        "limitations": [
            "B7 AlignScore threshold remains uncalibrated.",
            "B8 classification is generated by Qwen.",
            "Deterministic temporal checks cover only "
            "selected high-precision patterns.",
            "No human event labels were used.",
            "No PPO experiment was run.",
        ],
    }

    atomic_write(
        args.out / "manifest.json",
        json.dumps(
            manifest, indent=2,
            ensure_ascii=False,
        ),
    )

    print()
    print("========== V17-B8 SUMMARY ==========")
    print("[CASES]", len(rows))
    print("[STATUSES]", dict(statuses))
    print(
        "[CLASSIFIED CLAIMS]",
        sum(x["b8_classified_claims"] for x in rows)
    )
    print(
        "[REVIEW CLAIMS]",
        sum(x["b8_review_claims"] for x in rows)
    )
    print("[CACHE HITS]", cache_hits)
    print("[MODEL CALLS]", model_calls)
    print("V17-B8: AUDIT COMPLETE")


if __name__ == "__main__":
    main()
