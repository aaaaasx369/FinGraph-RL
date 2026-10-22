#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17-B5: Claim Validation + Financial Assessment

Pipeline:
  SEC PIT retrieval
      -> Quote-ID selection
      -> Claim-by-claim Qwen validation
      -> Qwen financial assessment
  Temporal KG 1-hop retrieval
      -> Deterministic structural summary

Outputs:
  signals.csv
  unsupported.csv
  claim_audit.jsonl
  manifest.json
  cache.sqlite

No PPO training.
No 2022-2023 test data.
Opaque FinDKG relations have no inferred
financial or economic semantics.

The same Qwen model performs separate stages:
this is not independent model validation.
"""

import argparse
import hashlib
import json
import re
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


VERSION = "v17b5_claim_validation_1"

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]


EXTRACTION_PROMPT = """
You extract financial claims from historical SEC excerpts.

Select 1-3 relevant candidate quote IDs.

Return JSON:
{
  "claims": [
    {
      "quote_id": "Q0001",
      "interpretation": "A short factual interpretation"
    }
  ]
}

RULES:
- Use ONLY supplied candidate quotes.
- Cite their exact quote_id values.
- Do not rewrite quotations.
- No KG reasoning.
- Do not predict returns.
- Distinguish actual events from hypothetical risks.
- Each interpretation must be a factual statement
  supported by its quote.
- Do not add numbers, dates, or outcomes absent
  from the selected quotation.
- If no reliable claim exists, use claims=[].
"""


VALIDATION_PROMPT = """
You are a STRICT evidence-grounding auditor.

Decide whether each interpretation is entailed
by its cited SEC quote.

Return JSON only:
{
  "judgments": [
    {
      "claim_id": "C01",
      "label": "supported",
      "reason": "Brief explanation"
    }
  ]
}

Valid labels:
supported
partial
unsupported

SUPPORTED:
Every material claim in the interpretation
follows directly from the quoted text.

PARTIAL:
The quotation supports some, but not all,
of the interpretation.

UNSUPPORTED:
The quotation does not support the claim,
or requires unstated facts or assumptions.

Important:
- Risk of an event is not proof that it occurred.
- Do not infer realized financial outcomes
  from conditional statements.
- Do not use outside information.
- Assess each claim independently.
- Do not make financial scores.
- Use every supplied claim_id exactly once.
"""


ASSESSMENT_PROMPT = """
You are a cautious SEC-grounded financial analyst.

You receive ONLY claims already judged supported.
Assess disclosed business implications as of the
historical decision date.

Return JSON only:
{
  "impact": 0.0,
  "risk": 0.0,
  "confidence": 0.0,
  "assessment": "Brief reasoning",
  "used_claim_ids": ["C01"],
  "impact_direction": "neutral"
}

impact_direction must be:
positive, neutral, or negative

Rules:
- Use ONLY the verified claims provided.
- Cite IDs from used_claim_ids.
- No knowledge of future stock returns.
- Do not assume a risk has materialized.
- Disclosed risks do not automatically
  imply a negative stock return.
- No unsupported company-specific facts.
- Confidence reflects directness and quality
  of evidence, NOT probability of stock gains.
- Scores may be zero.
- Neutral impact with nonzero disclosed risk
  is often appropriate.
- Do not assign high confidence merely because
  several quotations repeat the same fact.
- Do not interpret KG information.
- Explain why the impact direction is chosen.

Scales:
impact: -1 to +1
risk: 0 to 1
confidence: 0 to 1

These are qualitative and uncalibrated scores.
"""


def log(*args):
    print(*args, flush=True)


def serialize(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def make_key(payload):
    return hashlib.sha256(
        serialize({
            "version": VERSION,
            "payload": payload,
        }).encode("utf-8")
    ).hexdigest()[:24]


def write_jsonl(path, row):
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            serialize(row) + "\n"
        )


def atomic_csv(path, rows, columns):
    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    pd.DataFrame(
        rows,
        columns=columns,
    ).to_csv(
        tmp,
        index=False,
    )

    tmp.replace(path)


SIGNAL_COLUMNS = [
    "date", "ticker",
    "impact", "risk", "confidence",
    "impact_direction",
    "supported_claims",
    "partial_claims",
    "unsupported_claims",
    "sec_retrieved",
    "kg_retrieved",
    "kg_relations",
    "kg_in",
    "kg_out",
    "assessment",
    "status",
    "cache_key",
]

UNSUPPORTED_COLUMNS = [
    "date", "ticker", "status",
    "stage", "reason", "cache_key",
]


class CaseCache:
    def __init__(self, path):
        self.db = sqlite3.connect(
            str(path)
        )

        self.db.execute("""
          CREATE TABLE IF NOT EXISTS cases (
            cache_key TEXT PRIMARY KEY,
            result_json TEXT NOT NULL
          )
        """)

        self.db.commit()

    def get(self, key):
        row = self.db.execute(
            "SELECT result_json FROM cases "
            "WHERE cache_key=?",
            (key,),
        ).fetchone()

        return (
            json.loads(row[0])
            if row else None
        )

    def put(self, key, obj):
        self.db.execute(
            "INSERT OR REPLACE INTO cases "
            "(cache_key,result_json) VALUES (?,?)",
            (
                key,
                serialize(obj),
            ),
        )

        self.db.commit()

    def close(self):
        self.db.close()


def chat_json(
    client,
    model,
    system,
    payload,
):
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": serialize(payload),
            },
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 1000,
        },
    )

    data = json.loads(
        response.message.content
    )

    if not isinstance(data, dict):
        raise ValueError(
            "LLM did not produce a JSON object"
        )

    return data


def quote_bank(sec_hits):
    # Existing deterministic quote window generator.
    bank, candidates = b4.build_quote_bank(
        sec_hits,
        width=220,
        overlap=50,
    )

    return bank, candidates


def extract_claims(
    client,
    model,
    ticker,
    day,
    bank,
    candidates,
):
    raw = chat_json(
        client,
        model,
        EXTRACTION_PROMPT,
        {
            "ticker": ticker,
            "decision_date": str(day.date()),
            "candidate_quotes": [
                {
                    "quote_id": c["quote_id"],
                    "text": c["quote"],
                    "topic": c["topic"],
                    "form": c["form"],
                }
                for c in candidates
            ],
        },
    )

    claims = raw.get("claims")

    if not isinstance(claims, list):
        raise ValueError(
            "Extraction claims must be a list"
        )

    if not 1 <= len(claims) <= 3:
        raise ValueError(
            "No claims or more than three claims"
        )

    result = []
    seen = set()

    for i, claim in enumerate(claims):
        if not isinstance(claim, dict):
            raise ValueError(
                "Malformed extraction claim"
            )

        qid = claim.get("quote_id")

        if qid not in bank or qid in seen:
            raise ValueError(
                f"Invalid/repeated quote ID: {qid}"
            )

        seen.add(qid)

        interpretation = str(
            claim.get(
                "interpretation", ""
            )
        ).strip()

        if not (
            12 <= len(interpretation) <= 500
        ):
            raise ValueError(
                "Invalid interpretation length"
            )

        source = bank[qid]

        result.append({
            "claim_id": f"C{i+1:02d}",
            "quote_id": qid,
            "evidence_id": source[
                "evidence_id"
            ],
            "quote": source["quote"],
            "interpretation": interpretation,
        })

    return result


def validate_claims(
    client,
    model,
    claims,
):
    raw = chat_json(
        client,
        model,
        VALIDATION_PROMPT,
        {"claims": claims},
    )

    judgments = raw.get(
        "judgments"
    )

    if not isinstance(judgments, list):
        raise ValueError(
            "Missing judgments list"
        )

    required = {
        c["claim_id"]
        for c in claims
    }

    if len(judgments) != len(required):
        raise ValueError(
            "Validator returned wrong number of claims"
        )

    labels = {}
    reasons = {}

    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError(
                "Malformed judgment"
            )

        cid = judgment.get(
            "claim_id"
        )
        label = judgment.get(
            "label"
        )
        reason = str(
            judgment.get(
                "reason", ""
            )
        ).strip()

        if cid not in required or cid in labels:
            raise ValueError(
                f"Invalid judgment ID: {cid}"
            )

        if label not in {
            "supported",
            "partial",
            "unsupported",
        }:
            raise ValueError(
                f"Invalid label: {label}"
            )

        if len(reason) < 5:
            raise ValueError(
                "Validator omitted explanation"
            )

        labels[cid] = label
        reasons[cid] = reason

    if set(labels) != required:
        raise ValueError(
            "Incomplete validation"
        )

    output = []

    for claim in claims:
        cid = claim["claim_id"]

        output.append({
            **claim,
            "label": labels[cid],
            "validation_reason": reasons[cid],
        })

    return output


def assess(
    client,
    model,
    ticker,
    day,
    supported,
):
    if not supported:
        raise ValueError(
            "No supported claims to assess"
        )

    payload = {
        "ticker": ticker,
        "decision_date": str(day.date()),
        "verified_claims": [
            {
                "claim_id": x["claim_id"],
                "quote": x["quote"],
                "interpretation": x[
                    "interpretation"
                ],
            }
            for x in supported
        ],
    }

    raw = chat_json(
        client,
        model,
        ASSESSMENT_PROMPT,
        payload,
    )

    score = b2.numbers(raw)

    direction = raw.get(
        "impact_direction"
    )

    if direction not in {
        "positive",
        "neutral",
        "negative",
    }:
        raise ValueError(
            "Invalid impact direction"
        )

    impact = score["impact"]

    if direction == "positive" and impact <= 0:
        raise ValueError(
            "Positive direction/impact mismatch"
        )

    if direction == "negative" and impact >= 0:
        raise ValueError(
            "Negative direction/impact mismatch"
        )

    if direction == "neutral" and impact != 0:
        raise ValueError(
            "Neutral direction/impact mismatch"
        )

    used = raw.get(
        "used_claim_ids"
    )

    if not isinstance(used, list):
        raise ValueError(
            "Missing used_claim_ids"
        )

    valid_ids = {
        x["claim_id"]
        for x in supported
    }

    if (
        not used
        or len(used) != len(set(used))
        or not set(used).issubset(valid_ids)
    ):
        raise ValueError(
            "Invalid financial assessment citations"
        )

    explanation = str(
        raw.get(
            "assessment", ""
        )
    ).strip()

    if not (
        15 <= len(explanation) <= 1200
    ):
        raise ValueError(
            "Invalid financial assessment explanation"
        )

    # Conservative evidence-quality constraint.
    # This is a heuristic ceiling, NOT statistical
    # calibration. Repeated claims do not raise it.
    #
    # In this stage we cap maximum confidence
    # rather than treating it as a probability.
    capped_confidence = min(
        score["confidence"],
        0.70,
    )

    return {
        "impact": score["impact"],
        "risk": score["risk"],
        "confidence": capped_confidence,
        "raw_confidence": score[
            "confidence"
        ],
        "confidence_capped": (
            capped_confidence
            < score["confidence"]
        ),
        "impact_direction": direction,
        "assessment": explanation,
        "used_claim_ids": used,
    }


def run_case(
    client,
    args,
    day,
    ticker,
    sec_hits,
    kg_hits,
):
    structure = b2.kg_structural_summary(
        kg_hits,
        day,
    )

    bank, candidates = quote_bank(
        sec_hits
    )

    if not candidates:
        return {
            "status": "UNSUPPORTED",
            "stage": "retrieval",
            "reason": "No SEC candidate quotations",
            "kg_structure": structure,
        }

    # Three separate LLM calls.
    claims = extract_claims(
        client,
        args.llm,
        ticker,
        day,
        bank,
        candidates,
    )

    validated = validate_claims(
        client,
        args.llm,
        claims,
    )

    supported = [
        x for x in validated
        if x["label"] == "supported"
    ]

    partial = [
        x for x in validated
        if x["label"] == "partial"
    ]

    unsupported = [
        x for x in validated
        if x["label"] == "unsupported"
    ]

    # In B5, only clearly supported claims can
    # reach financial assessment. Other claims
    # remain visible in the audit.
    if not supported:
        return {
            "status": "UNSUPPORTED",
            "stage": "validation",
            "reason": "No supported SEC claims",
            "claims": validated,
            "kg_structure": structure,
        }

    financial = assess(
        client,
        args.llm,
        ticker,
        day,
        supported,
    )

    return {
        "status": "SUPPORTED",
        "claims": validated,
        "financial": financial,
        "kg_structure": structure,
        "sec_retrieved": len(sec_hits),
        "kg_retrieved": len(kg_hits),
        "supported_claims": len(supported),
        "partial_claims": len(partial),
        "unsupported_claims": len(unsupported),
    }


def validate_cached_result(result):
    if not isinstance(result, dict):
        raise ValueError(
            "Corrupt cached result"
        )

    if result.get("status") not in {
        "SUPPORTED",
        "UNSUPPORTED",
    }:
        raise ValueError(
            "Invalid cache status"
        )

    if result["status"] == "SUPPORTED":
        financial = result["financial"]

        b2.numbers(financial)

        if not result.get(
            "supported_claims", 0
        ) > 0:
            raise ValueError(
                "Cached result lacks supported claims"
            )

        if financial["confidence"] > 0.7:
            raise ValueError(
                "Cached confidence exceeds cap"
            )

    return result


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pit-sec",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/"
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
        "--tickers",
        nargs="+",
        default=TICKERS,
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
        default=(
            "http://140.113.87.164:11434"
        ),
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
            "v17b5_validation_20"
        ),
    )

    args = p.parse_args()

    if not set(args.tickers).issubset(TICKERS):
        raise ValueError("Invalid ticker")

    if min(
        args.sec_top_k,
        args.kg_top_k,
        args.retries,
    ) < 1:
        raise ValueError(
            "Invalid retrieval parameters"
        )

    if args.sample_count < 1:
        raise ValueError(
            "sample-count must be positive"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    sec = v17.load_sec(
        args.pit_sec
    )

    kg = v17.load_kg(
        args.temporal_kg
    )

    if args.dates:
        dates = pd.DatetimeIndex(
            pd.to_datetime(args.dates)
        ).normalize().unique()

    else:
        # Uses historical audited V16 dates.
        raw = pd.read_csv(
            args.dates_csv
        )

        if "date" not in raw.columns:
            raise RuntimeError(
                "dates-csv has no date column"
            )

        dates = pd.DatetimeIndex(
            pd.to_datetime(
                raw["date"],
                errors="coerce",
            ).dropna().dt.normalize().unique()
        )

    dates = dates[
        (dates >= pd.Timestamp(
            args.start_date
        ))
        & (dates <= pd.Timestamp(
            args.end_date
        ))
    ]

    if len(dates) == 0:
        raise RuntimeError(
            "No historical dates selected"
        )

    # No test data in V17-B5.
    if (
        dates.min() < pd.Timestamp("2018-01-01")
        or dates.max() >= pd.Timestamp("2022-01-01")
    ):
        raise RuntimeError(
            "V17-B5 only permits 2018-2021"
        )

    all_pairs = [
        (
            pd.Timestamp(day),
            ticker,
        )
        for day in sorted(dates)
        for ticker in args.tickers
    ]

    rng = np.random.default_rng(
        args.sample_seed
    )

    n = min(
        args.sample_count,
        len(all_pairs),
    )

    indices = rng.choice(
        len(all_pairs),
        size=n,
        replace=False,
    )

    selected = sorted(
        [all_pairs[int(i)] for i in indices],
        key=lambda x: (x[0], x[1]),
    )

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    models = {
        m.model
        for m in client.list().models
    }

    if args.llm not in models:
        raise RuntimeError(
            f"Model {args.llm} unavailable. "
            f"Found: {sorted(models)}"
        )

    cache = CaseCache(
        args.out / "cache.sqlite"
    )

    signal_rows = []
    unsupported_rows = []
    cache_hits = 0
    llm_attempts = 0

    audit_path = (
        args.out / "claim_audit.jsonl"
    )

    try:
        for day, ticker in selected:
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

            for hit in sec_hits:
                if not pd.Timestamp(
                    hit["available_from"]
                ) < day:
                    raise RuntimeError(
                        "SEC temporal leakage"
                    )

            for hit in kg_hits:
                if not pd.Timestamp(
                    hit["first_observed"]
                ) < day:
                    raise RuntimeError(
                        "KG temporal leakage"
                    )

            payload = {
                "date": str(day.date()),
                "ticker": ticker,
                "model": args.llm,
                "ollama_host": args.ollama_host,
                "sec_hits": sec_hits,
                "kg_hits": kg_hits,
            }

            key = make_key(payload)

            result = cache.get(key)

            if result is not None:
                result = validate_cached_result(
                    result
                )
                cache_hits += 1

            else:
                result = None
                error = None

                for attempt in range(
                    args.retries
                ):
                    try:
                        llm_attempts += 1

                        result = run_case(
                            client,
                            args,
                            day,
                            ticker,
                            sec_hits,
                            kg_hits,
                        )

                        error = None
                        break

                    except Exception as exc:
                        error = exc

                        log(
                            "[RETRY]",
                            day.date(),
                            ticker,
                            attempt + 1,
                            str(exc)[:300],
                        )

                        time.sleep(
                            min(2 ** attempt, 8)
                        )

                if error is not None:
                    result = {
                        "status": "UNSUPPORTED",
                        "stage": "exception",
                        "reason": str(error)[:1000],
                    }

                # Cache both supported and unsupported
                # outcomes, including their reasons.
                cache.put(
                    key,
                    result,
                )

            audit_record = {
                "date": str(day.date()),
                "ticker": ticker,
                "cache_key": key,
                "sec_hits": sec_hits,
                "kg_hits": kg_hits,
                **result,
            }

            # One audit record per selected case.
            # Existing log duplicates are avoided
            # by rewriting from deterministic cases
            # on each run, below.
            if result["status"] == "SUPPORTED":
                fin = result["financial"]
                kg_info = result["kg_structure"]

                row = {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "impact": fin["impact"],
                    "risk": fin["risk"],
                    "confidence": fin["confidence"],
                    "impact_direction": fin[
                        "impact_direction"
                    ],
                    "supported_claims": result[
                        "supported_claims"
                    ],
                    "partial_claims": result[
                        "partial_claims"
                    ],
                    "unsupported_claims": result[
                        "unsupported_claims"
                    ],
                    "sec_retrieved": result[
                        "sec_retrieved"
                    ],
                    "kg_retrieved": result[
                        "kg_retrieved"
                    ],
                    "kg_relations": len(
                        kg_info["relation_counts"]
                    ),
                    "kg_in": kg_info[
                        "direction_counts"
                    ]["in"],
                    "kg_out": kg_info[
                        "direction_counts"
                    ]["out"],
                    "assessment": fin["assessment"],
                    "status": "SUPPORTED",
                    "cache_key": key,
                }

                signal_rows.append(row)

                log(
                    "[SUPPORTED]",
                    row["date"],
                    ticker,
                    "impact=", row["impact"],
                    "risk=", row["risk"],
                    "conf=", row["confidence"],
                    "claims=", row[
                        "supported_claims"
                    ],
                )

            else:
                row = {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "status": "UNSUPPORTED",
                    "stage": result.get(
                        "stage", "unknown"
                    ),
                    "reason": result.get(
                        "reason", ""
                    ),
                    "cache_key": key,
                }

                unsupported_rows.append(row)

                log(
                    "[UNSUPPORTED]",
                    row["date"],
                    ticker,
                    row["stage"],
                )

            # Persist valid progress after each case.
            atomic_csv(
                args.out / "signals.csv",
                signal_rows,
                SIGNAL_COLUMNS,
            )

            atomic_csv(
                args.out / "unsupported.csv",
                unsupported_rows,
                UNSUPPORTED_COLUMNS,
            )

            # Append for diagnosis. Cache-key lets
            # downstream readers deduplicate.
            write_jsonl(
                audit_path,
                audit_record,
            )

    finally:
        cache.close()

    triples = [
        (
            float(row["impact"]),
            float(row["risk"]),
            float(row["confidence"]),
        )
        for row in signal_rows
    ]

    counter = Counter(triples)

    repeated = {
        str(k): v
        for k, v in counter.items()
        if v > 1
    }

    manifest = {
        "version": VERSION,
        "model": args.llm,
        "ollama_host": args.ollama_host,
        "requested_cases": len(selected),
        "supported_cases": len(signal_rows),
        "unsupported_cases": len(
            unsupported_rows
        ),
        "cache_hits": cache_hits,
        "llm_case_attempts": llm_attempts,
        "repeated_score_triplets": repeated,
        "kg_role": "structural_only",
        "test_2022_2023_used": False,
        "warnings": [
            "Validation uses the same Qwen model as extraction and assessment.",
            "Verbatim quote selection does not itself prove entailment.",
            "The confidence cap is heuristic, not statistical calibration.",
            "KG first-observed date is not certified public disclosure time.",
            "No PPO training is performed.",
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
    log("========== V17-B5 SUMMARY ==========")
    log("[REQUESTED]", len(selected))
    log("[SUPPORTED]", len(signal_rows))
    log("[UNSUPPORTED]", len(unsupported_rows))
    log("[CACHE HITS]", cache_hits)
    log("[LLM CASE ATTEMPTS]", llm_attempts)
    log("[REPEATED TRIPLETS]", repeated)
    log("[OUTPUT]", args.out)

    if len(signal_rows) + len(unsupported_rows) != len(
        selected
    ):
        raise RuntimeError(
            "Incomplete V17-B5 results"
        )

    log("V17-B5: AUDIT COMPLETE")
    log(
        "Manual semantic review is still required."
    )


if __name__ == "__main__":
    main()
