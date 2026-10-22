#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V17-B6
Evidence-grounded Financial Event Classification

Stages:
1. PIT SEC retrieval
2. Deterministic quotation candidates
3. Qwen atomic claim extraction
4. Qwen claim verification
5. Qwen supported-claim classification
6. Python validation and feature encoding
7. Dynamic KG structural summary

This pipeline DOES NOT:
- Generate arbitrary impact/risk/confidence scores
- Interpret opaque FinDKG relations economically
- Train PPO
- Read 2022-2023 test data

Models:
  qwen2.5:3b via remote Ollama

Important:
  LLM verification is not independent ground truth.
  KG observed dates are not certified disclosure dates.
"""

import argparse
import hashlib
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import ollama

from unified_event_memory.finrl_x.aegis_rl import (
    run_v17_unified_llm_rag as v17,
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_v17b2_grounded_rag as b2,
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_v17b4_quote_grounded as b4,
)


VERSION = "v17b61_events_schema_repair_1"

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]

RISK_CATEGORIES = [
    "supply_chain",
    "regulatory",
    "competition",
    "financial",
    "operational",
    "cybersecurity",
    "macroeconomic",
    "other",
    "none",
]

IMPACT_DIRECTIONS = [
    "positive",
    "negative",
    "neutral",
    "unknown",
]

EVENT_STATUSES = [
    "realized",
    "planned",
    "hypothetical",
    "historical",
    "unknown",
]

SEVERITIES = [
    "low",
    "medium",
    "high",
    "unknown",
]

EXTRACT_PROMPT = """
Extract up to three atomic financial facts from
the supplied historical SEC quotations.

Return a JSON object with a claims array.
Each claim must have:
  quote_id: an exact ID from the input
  claim: one specific factual assertion

Never copy these instructions into a claim.

A valid claim must identify a concrete subject,
action, disclosure, exposure, commitment, or risk
that is actually stated in the selected quotation.

Reject generic statements such as:
- a supported atomic fact
- the company has financial information
- one directly supported atomic fact
- a financial risk exists

Rules:
- Use ONLY the supplied candidate quotations.
- Cite quote IDs exactly.
- Do not rewrite the source quotations.
- No predictions of future stock returns.
- A hypothetical risk is not a realized event.
- No knowledge outside the provided source.
- No KG economic inference.
- Do not generate numeric impact/risk scores.
- If no factual claim is supported, claims=[].
"""


VERIFY_PROMPT = """
You are a strict SEC evidence verification module.

For each atomic claim, compare its claim text
ONLY to the quoted SEC passage.

Return JSON only:

{
  "judgments": [
    {
      "claim_id": "C01",
      "label": "supported",
      "reason": "Why the quote supports the claim"
    }
  ]
}

Labels:
supported, partial, unsupported

supported:
The quote directly establishes every material
part of the claim.

partial:
Some information is supported, but other claims
require unstated assumptions.

unsupported:
The claim is contradicted or not established
by its quoted evidence.

Rules:
- Do not use external knowledge.
- Do not infer realized events from hypothetical risks.
- Check amounts, dates, subjects, and causality.
- Do not assume financial outcomes not in the text.
- Return exactly one judgment for every claim_id.
"""

CLASSIFY_PROMPT = """
Classify the financial meaning of VERIFIED
historical SEC claims.

Return JSON only:

{
  "classifications": [
    {
      "claim_id": "C01",
      "risk_category": "supply_chain",
      "risk_severity": "medium",
      "impact_direction": "unknown",
      "event_status": "hypothetical",
      "reason": "Concise evidence-based explanation"
    }
  ]
}

Allowed risk_category:
supply_chain, regulatory, competition,
financial, operational, cybersecurity,
macroeconomic, other, none

Allowed risk_severity:
low, medium, high, unknown

Allowed impact_direction:
positive, negative, neutral, unknown

Allowed event_status:
realized, planned, hypothetical,
historical, unknown

Interpretation:
- risk_category identifies the topic.
- risk_severity is the severity of the disclosed
  exposure, not the probability of stock loss.
- impact_direction concerns evidence-supported
  BUSINESS or FINANCIAL consequences, not
  future stock-price direction.
- unknown means insufficient evidence.
- neutral means evidence supports no clear
  directional effect; do not substitute neutral
  for unknown.
- realized requires explicit proof that an
  event occurred.
- hypothetical means a conditional risk or
  possible future outcome.
- planned means an intention or stated plan.
- historical means a past event, without
  implying it is new on the decision date.

Rules:
- Use only the supplied verified claims.
- Do not invent outcomes, magnitude, or events.
- Do not classify a hypothetical risk as realized.
- If financial magnitude is unclear, severity may
  be unknown even if the risk category is clear.
- Use every supplied claim_id exactly once.
- Do not infer KG economic semantics.
- Do not output floating-point financial scores.
"""


def log(*items):
    print(*items, flush=True)


def json_text(obj):
    return json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )


def key_of(obj):
    return hashlib.sha256(
        json_text({
            "version": VERSION,
            "payload": obj,
            "prompts": [
                EXTRACT_PROMPT,
                VERIFY_PROMPT,
                CLASSIFY_PROMPT,
            ],
        }).encode("utf-8")
    ).hexdigest()[:24]


def save_csv(path, rows, cols):
    temp = path.with_suffix(".csv.tmp")
    pd.DataFrame(
        rows,
        columns=cols,
    ).to_csv(temp, index=False)
    temp.replace(path)


def save_jsonl(path, rows):
    temp = path.with_suffix(".jsonl.tmp")
    with open(temp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_text(row) + "\n")
    temp.replace(path)


class Cache:
    def __init__(self, path):
        self.db = sqlite3.connect(str(path))
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS results (
            k TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        )
        """)
        self.db.commit()

    def get(self, key):
        r = self.db.execute(
            "SELECT payload FROM results WHERE k=?",
            (key,),
        ).fetchone()
        return json.loads(r[0]) if r else None

    def put(self, key, result):
        self.db.execute(
            "INSERT OR REPLACE INTO results VALUES (?, ?)",
            (key, json_text(result)),
        )
        self.db.commit()

    def close(self):
        self.db.close()


def chat(client, model, system, payload):
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json_text(payload),
            },
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 1100,
        },
    )

    result = json.loads(
        response.message.content
    )

    if not isinstance(result, dict):
        raise ValueError("LLM JSON is not an object")

    return result


def extract(client, model, ticker, day, bank, candidates):
    raw = chat(
        client,
        model,
        EXTRACT_PROMPT,
        {
            "ticker": ticker,
            "date": str(day.date()),
            "candidate_quotes": [
                {
                    "quote_id": c["quote_id"],
                    "text": c["quote"],
                    "form": c["form"],
                    "topic": c["topic"],
                }
                for c in candidates
            ],
        },
    )

    claims = raw.get("claims")

    if not isinstance(claims, list):
        raise ValueError("Missing claims list")

    if len(claims) > 3:
        raise ValueError("Too many claims")

    output = []
    seen = set()

    for index, item in enumerate(claims, 1):
        if not isinstance(item, dict):
            raise ValueError("Malformed claim")

        qid = item.get("quote_id")
        statement = str(
            item.get("claim", "")
        ).strip()

        if qid not in bank or qid in seen:
            raise ValueError("Invalid quote ID")

        if not 8 <= len(statement) <= 500:
            raise ValueError("Invalid atomic claim")

        normalized_statement = " ".join(
            statement.lower().split()
        )

        placeholders = (
            "one directly supported atomic fact",
            "a supported atomic fact",
            "the company has financial information",
            "a financial risk exists",
        )

        if any(
            phrase in normalized_statement
            for phrase in placeholders
        ):
            raise ValueError(
                "PROMPT_PLACEHOLDER_COPIED"
            )

        seen.add(qid)
        source = bank[qid]

        output.append({
            "claim_id": f"C{index:02d}",
            "quote_id": qid,
            "sec_evidence_id": source["evidence_id"],
            "quote": source["quote"],
            "claim": statement,
        })

    return output


def verify(client, model, claims):
    if not claims:
        return []

    result = chat(
        client,
        model,
        VERIFY_PROMPT,
        {"claims": claims},
    )

    judgments = result.get("judgments")

    if not isinstance(judgments, list):
        raise ValueError("Missing judgments")

    if len(judgments) != len(claims):
        raise ValueError("Judgment count mismatch")

    allowed_ids = {
        c["claim_id"]
        for c in claims
    }

    by_id = {}

    for j in judgments:
        cid = j.get("claim_id")
        label = j.get("label")
        reason = str(
            j.get("reason", "")
        ).strip()

        if cid not in allowed_ids or cid in by_id:
            raise ValueError("Invalid judgment claim ID")

        if label not in {
            "supported", "partial", "unsupported"
        }:
            raise ValueError("Invalid judgment label")

        if not reason:
            raise ValueError("Missing verification reason")

        by_id[cid] = {
            "label": label,
            "validation_reason": reason,
        }

    if set(by_id) != allowed_ids:
        raise ValueError("Incomplete judgments")

    return [
        {**claim, **by_id[claim["claim_id"]]}
        for claim in claims
    ]


def classify(client, model, verified):
    supported = [
        c for c in verified
        if c["label"] == "supported"
    ]

    if not supported:
        return []

    raw = chat(
        client,
        model,
        CLASSIFY_PROMPT,
        {
            "supported_claims": [
                {
                    "claim_id": c["claim_id"],
                    "quote": c["quote"],
                    "claim": c["claim"],
                }
                for c in supported
            ],
        },
    )

    result = raw.get("classifications")

    if not isinstance(result, list):
        raise ValueError("Missing classifications")

    if len(result) != len(supported):
        raise ValueError(
            "Classification count mismatch"
        )

    by_claim = {
        c["claim_id"]: c
        for c in supported
    }

    expected = set(by_claim)
    by_id = {}

    for row in result:
        if not isinstance(row, dict):
            raise ValueError(
                "Malformed classification"
            )

        cid = row.get("claim_id")

        if cid not in expected or cid in by_id:
            raise ValueError(
                "Invalid classification ID"
            )

        category = row.get("risk_category")
        severity = row.get("risk_severity")
        direction = row.get("impact_direction")
        status = row.get("event_status")

        if category not in RISK_CATEGORIES:
            raise ValueError(
                "Unknown risk category"
            )

        if severity not in SEVERITIES:
            raise ValueError(
                "Unknown severity"
            )

        if direction not in IMPACT_DIRECTIONS:
            raise ValueError(
                "Unknown direction"
            )

        if status not in EVENT_STATUSES:
            raise ValueError(
                "Unknown event status"
            )

        reason = str(
            row.get("reason") or ""
        ).strip()

        repaired = False

        if not reason:
            original = by_claim[cid]

            repair_prompt = """
You are auditing ONE classification of a
historical SEC filing claim.

You receive the source quote, extracted claim,
and an existing classification.

Explain whether the classification is directly
supported by the supplied quote.

Return JSON only:
{
  "supported": true,
  "reason": "specific evidence-based explanation"
}

Set supported=false when a category, severity,
direction or event status makes an assertion
that the quote cannot establish.

Do not invent facts.
Unknown is permitted.
Hypothetical does not mean realized.
Do not predict stock returns.
"""

            repair = chat(
                client,
                model,
                repair_prompt,
                {
                    "quote": original["quote"],
                    "claim": original["claim"],
                    "classification": {
                        "risk_category": category,
                        "risk_severity": severity,
                        "impact_direction": direction,
                        "event_status": status,
                    },
                },
            )

            if repair.get("supported") is not True:
                raise ValueError(
                    "CLASSIFICATION_REPAIR_REJECTED: "
                    + str(
                        repair.get("reason", "")
                    )[:250]
                )

            reason = str(
                repair.get("reason") or ""
            ).strip()

            if not reason:
                raise ValueError(
                    "CLASSIFICATION_REPAIR_EMPTY"
                )

            repaired = True

        by_id[cid] = {
            "claim_id": cid,
            "risk_category": category,
            "risk_severity": severity,
            "impact_direction": direction,
            "event_status": status,
            "reason": reason,
            "reason_repaired": repaired,
        }

    if set(by_id) != expected:
        raise ValueError(
            "Incomplete classifications"
        )

    return [
        by_id[c["claim_id"]]
        for c in supported
    ]



def encode(case, kg):
    """
    Fixed-size numerical representation.

    Categorical counts rather than fabricated
    continuous finance scores.

    Unknown is retained explicitly.
    """
    claims = case["claims"]
    labels = case["classifications"]

    n = len(claims)
    supported = sum(
        c["label"] == "supported"
        for c in claims
    )

    out = {
        "claim_count": n,
        "supported_count": supported,
        "partial_count": sum(
            c["label"] == "partial"
            for c in claims
        ),
        "unsupported_count": sum(
            c["label"] == "unsupported"
            for c in claims
        ),
        "evidence_support_ratio": (
            supported / n if n else np.nan
        ),
        "has_usable_sec_signal": int(
            supported > 0
        ),
        "kg_edge_count": kg["count"],
        "kg_relation_types": len(
            kg["relation_counts"]
        ),
        "kg_in_count": kg[
            "direction_counts"
        ]["in"],
        "kg_out_count": kg[
            "direction_counts"
        ]["out"],
        "kg_min_age_days": (
            kg["min_age_days"]
            if kg["min_age_days"] is not None
            else np.nan
        ),
    }

    # All categories have explicit separate columns.
    for cat in RISK_CATEGORIES:
        out[f"risk_{cat}"] = sum(
            x["risk_category"] == cat
            for x in labels
        )

    for sev in SEVERITIES:
        out[f"severity_{sev}"] = sum(
            x["risk_severity"] == sev
            for x in labels
        )

    for direction in IMPACT_DIRECTIONS:
        out[f"impact_{direction}"] = sum(
            x["impact_direction"] == direction
            for x in labels
        )

    for event in EVENT_STATUSES:
        out[f"event_{event}"] = sum(
            x["event_status"] == event
            for x in labels
        )

    return out


def run_case(client, args, day, ticker, sec_hits, kg_hits):
    kg_summary = b2.kg_structural_summary(
        kg_hits, day
    )

    bank, candidates = b4.build_quote_bank(
        sec_hits,
        width=220,
        overlap=50,
    )

    if not candidates:
        return {
            "status": "NO_EVIDENCE",
            "claims": [],
            "classifications": [],
            "kg": kg_summary,
            "reason": "No SEC quotation candidates",
        }

    extracted = extract(
        client,
        args.llm,
        ticker,
        day,
        bank,
        candidates,
    )

    if not extracted:
        return {
            "status": "ABSTAINED",
            "claims": [],
            "classifications": [],
            "kg": kg_summary,
            "reason": "No atomic claims extracted",
        }

    verified = verify(
        client,
        args.llm,
        extracted,
    )

    classifications = classify(
        client,
        args.llm,
        verified,
    )

    return {
        "status": (
            "SUPPORTED"
            if classifications
            else "NO_SUPPORTED_CLAIMS"
        ),
        "claims": verified,
        "classifications": classifications,
        "kg": kg_summary,
        "reason": "",
    }


def get_dates(args):
    if args.dates:
        dates = pd.DatetimeIndex(
            pd.to_datetime(args.dates)
        )

    else:
        path = Path(args.dates_csv)

        if not path.exists():
            raise FileNotFoundError(path)

        raw = pd.read_csv(path)

        if "date" not in raw.columns:
            raise ValueError(
                "Dates CSV lacks date column"
            )

        dates = pd.DatetimeIndex(
            pd.to_datetime(
                raw["date"],
                errors="coerce",
            ).dropna()
        )

    dates = dates.normalize().unique().sort_values()

    dates = dates[
        (dates >= pd.Timestamp(args.start_date))
        & (dates <= pd.Timestamp(args.end_date))
    ]

    if len(dates) == 0:
        raise ValueError("No selected dates")

    if (
        dates.min() < pd.Timestamp("2018-01-01")
        or dates.max() >= pd.Timestamp("2022-01-01")
    ):
        raise ValueError(
            "B6 limited to 2018-2021"
        )

    return dates


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pit-sec",
        default=(
            "results/aegis_rl_v1/v11f_sec_pit/"
            "sec_pit_retrieval.parquet"
        ),
    )
    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )
    p.add_argument(
        "--dates-csv",
        default=(
            "results/aegis_rl_v1/"
            "v16_dual_rag_ppo/"
            "kg_retrieval_audit.csv"
        ),
    )
    p.add_argument(
        "--dates",
        nargs="+",
        default=None,
    )
    p.add_argument(
        "--start-date",
        default="2020-01-01",
    )
    p.add_argument(
        "--end-date",
        default="2020-12-31",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
    )
    p.add_argument(
        "--sample-count",
        type=int,
        default=20,
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--sec-top-k",
        type=int,
        default=5,
    )
    p.add_argument(
        "--kg-top-k",
        type=int,
        default=12,
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=2,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v17b6_events_20"
        ),
    )

    args = p.parse_args()

    if not set(args.tickers).issubset(TICKERS):
        raise ValueError("Invalid tickers")

    if min(
        args.sec_top_k,
        args.kg_top_k,
        args.sample_count,
        args.retries,
    ) < 1:
        raise ValueError("Invalid arguments")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    sec = v17.load_sec(args.pit_sec)
    kg = v17.load_kg(args.temporal_kg)

    dates = get_dates(args)

    all_pairs = [
        (pd.Timestamp(day), ticker)
        for day in dates
        for ticker in args.tickers
    ]

    n = min(
        args.sample_count,
        len(all_pairs),
    )

    rng = np.random.default_rng(
        args.sample_seed
    )

    chosen = rng.choice(
        len(all_pairs),
        size=n,
        replace=False,
    )

    pairs = sorted(
        [
            all_pairs[int(i)]
            for i in chosen
        ],
        key=lambda x: (x[0], x[1]),
    )

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    available = {
        m.model
        for m in client.list().models
    }

    if args.llm not in available:
        raise RuntimeError(
            f"Missing model {args.llm}; "
            f"available={sorted(available)}"
        )

    cache = Cache(
        args.out / "cache.sqlite"
    )

    signal_rows = []
    audit_rows = []
    cache_hits = 0
    new_cases = 0

    try:
        for day, ticker in pairs:
            sec_hits = v17.sec_retrieve(
                sec,
                ticker,
                day,
                args.sec_top_k,
            )

            kg_hits = v17.kg_retrieve(
                kg,
                ticker,
                day,
                args.kg_top_k,
            )

            for item in sec_hits:
                if not (
                    pd.Timestamp(
                        item["available_from"]
                    ) < day
                ):
                    raise RuntimeError(
                        "SEC temporal violation"
                    )

            for item in kg_hits:
                if not (
                    pd.Timestamp(
                        item["first_observed"]
                    ) < day
                ):
                    raise RuntimeError(
                        "KG temporal violation"
                    )

            payload = {
                "date": str(day.date()),
                "ticker": ticker,
                "llm": args.llm,
                "host": args.ollama_host,
                "sec_hits": sec_hits,
                "kg_hits": kg_hits,
            }

            key = key_of(payload)

            case = cache.get(key)

            if case is not None:
                # Cached inputs are addressed by
                # hash of full evidence and version.
                if case.get("status") not in {
                    "SUPPORTED",
                    "NO_SUPPORTED_CLAIMS",
                    "NO_EVIDENCE",
                    "ABSTAINED",
                }:
                    raise RuntimeError(
                        "Invalid cached status"
                    )

                cache_hits += 1

            else:
                last_error = None
                case = None

                for attempt in range(
                    args.retries
                ):
                    try:
                        case = run_case(
                            client,
                            args,
                            day,
                            ticker,
                            sec_hits,
                            kg_hits,
                        )
                        last_error = None
                        break

                    except Exception as exc:
                        last_error = exc

                        log(
                            "[RETRY]",
                            day.date(),
                            ticker,
                            attempt + 1,
                            str(exc)[:250],
                        )

                        time.sleep(
                            min(2 ** attempt, 8)
                        )

                if last_error is not None:
                    case = {
                        "status": "ERROR",
                        "claims": [],
                        "classifications": [],
                        "kg": b2.kg_structural_summary(
                            kg_hits, day
                        ),
                        "reason": str(
                            last_error
                        )[:1000],
                    }

                if case["status"] != "ERROR":
                    cache.put(key, case)

                new_cases += 1

            features = encode(
                case,
                case["kg"],
            )

            row = {
                "date": str(day.date()),
                "ticker": ticker,
                "status": case["status"],
                "cache_key": key,
                **features,
            }

            signal_rows.append(row)

            audit_rows.append({
                **row,
                "case": case,
                "sec_evidence": sec_hits,
                "kg_evidence": kg_hits,
            })

            # Persist on every company-date.
            cols = list(signal_rows[0].keys())

            save_csv(
                args.out / "signals.csv",
                signal_rows,
                cols,
            )

            save_jsonl(
                args.out / "audit.jsonl",
                audit_rows,
            )

            log(
                "[B6]",
                row["date"],
                ticker,
                "status=", row["status"],
                "supported=", row["supported_count"],
                "risk_classes=", {
                    c: row[f"risk_{c}"]
                    for c in RISK_CATEGORIES
                    if row[f"risk_{c}"] > 0
                },
            )

    finally:
        cache.close()

    statuses = Counter(
        x["status"]
        for x in signal_rows
    )

    manifest = {
        "experiment": "FinRL-X V17-B6",
        "version": VERSION,
        "ollama_host": args.ollama_host,
        "model": args.llm,
        "cases": len(signal_rows),
        "statuses": dict(statuses),
        "cache_hits": cache_hits,
        "new_cases": new_cases,
        "risk_categories": RISK_CATEGORIES,
        "impact_directions": IMPACT_DIRECTIONS,
        "event_statuses": EVENT_STATUSES,
        "severity_classes": SEVERITIES,
        "test_2022_2023_used": False,
        "limitations": [
            "Verification and classification use the same LLM.",
            "Classification is not independent ground truth.",
            "Risk severity is qualitative and uncalibrated.",
            "Unknown and abstained are not neutral.",
            "Only selected SEC excerpts are analyzed.",
            "KG relations with opaque labels remain structural.",
            "KG observation dates are not certified PIT.",
            "No PPO training in this stage.",
        ],
    }

    (
        args.out / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("========== V17-B6 SUMMARY ==========")
    log("[CASES]", len(signal_rows))
    log("[STATUSES]", dict(statuses))
    log("[CACHE HITS]", cache_hits)
    log("[NEW CASES]", new_cases)
    log("[OUTPUT]", args.out)
    log("V17-B6: AUDIT COMPLETE")


if __name__ == "__main__":
    main()
