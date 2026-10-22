#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17-B2: Grounded SEC-RAG + Dynamic KG-RAG + Qwen

Reuses V17:
  SEC PIT loader / retrieval
  Temporal KG loader / retrieval
  Stable IDs and SQLite cache

Stages:
  1. SEC-only Qwen reasoning with verbatim quotes
  2. Deterministic KG structural representation
  3. Qwen Unified Fusion
  4. Strict validation and unsupported quarantine

No stock-return labels.
No PPO training.
No 2022-2023 data evaluated.

IMPORTANT:
Exact quotation verification is not a substitute
for semantic entailment or human evidence review.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ollama

from unified_event_memory.finrl_x.aegis_rl import (
    run_v17_unified_llm_rag as v17,
)

VERSION = "v17b3_grounded_2"

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]


def log(*args):
    print(*args, flush=True)


def dump(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def cache_key(payload):
    return v17.stable_id(
        "B2",
        {
            "version": VERSION,
            "payload": payload,
        },
    )


def append_jsonl(path, obj):
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(dump(obj) + "\n")


def numbers(obj):
    out = {}

    for key, lo, hi in [
        ("impact", -1.0, 1.0),
        ("risk", 0.0, 1.0),
        ("confidence", 0.0, 1.0),
    ]:
        if key not in obj:
            raise ValueError(
                f"Missing {key}"
            )

        x = float(obj[key])

        if not np.isfinite(x):
            raise ValueError(
                f"{key} is nonfinite"
            )

        if not lo <= x <= hi:
            raise ValueError(
                f"{key} out of range: {x}"
            )

        out[key] = x

    return out


def unique_ids(ids, allowed, field):
    if not isinstance(ids, list):
        raise ValueError(
            f"{field} must be list"
        )

    if any(
        not isinstance(x, str)
        for x in ids
    ):
        raise ValueError(
            f"{field} contains non-string ID"
        )

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Duplicate IDs in {field}"
        )

    if not set(ids).issubset(allowed):
        raise ValueError(
            f"Unsupported citation in {field}"
        )

    return ids


def sec_system_prompt():
    return """
You are a careful financial filing fact extractor.

You receive historical SEC filing excerpts.

Your FIRST task is to identify concrete facts explicitly
written in the provided excerpts.

Valid examples include:
- Financial commitments and purchase obligations
- Disclosed competition risks
- Capital expenditure plans
- Research and development expenditure
- Explicitly disclosed regulatory risks
- Explicitly described supply shortages

IMPORTANT:
- A filing mentioning a possible risk does NOT prove
  the risk has occurred.
- An excerpt may begin or end mid-sentence.
  Only quote complete meaningful spans visible there.
- Do not invent any numbers, events, or dates.
- Do not use your outside knowledge.
- Do not mention KG or graph relationships.
- Neutral impact is allowed even when facts are found.

Return exactly this JSON object:
{
  "impact": 0.0,
  "risk": 0.0,
  "confidence": 0.0,
  "claims": [
    {
      "evidence_id": "SEC_ID_FROM_INPUT",
      "quote": "EXACT substring copied from excerpt",
      "interpretation": "What the quote states, without speculation"
    }
  ]
}

For each clearly relevant excerpt:
1. Identify an explicit financial fact.
2. Copy an exact contiguous quote of 15-180 characters.
3. Explain that quote without adding outside facts.
4. Cite the evidence ID actually provided.

Prefer 1-3 strong grounded claims over many weak claims.

If excerpts contain clear financial facts, extract them,
even when their future price impact is unknown.

If no excerpt supports ANY factual claim,
return empty claims and zero impact, risk, confidence.

Scores:
impact [-1,1]: cautious directional financial assessment
risk [0,1]: disclosed risk severity assessment
confidence [0,1]: confidence supported by supplied text

These scores are qualitative judgments, not
calibrated probabilities or predictions.
"""


def fusion_system_prompt():
    return """
You combine verified historical SEC analysis with
a STRUCTURAL financial knowledge graph summary.

Return JSON only:
{
  "impact": 0.0,
  "risk": 0.0,
  "confidence": 0.0,
  "sec_evidence_ids": [],
  "kg_evidence_ids": [],
  "kg_role": "structural_only"
}

Rules:
- All financial interpretations must be grounded in
  the already verified SEC claims.
- KG data supplies graph structure, direction,
  relation IDs, and observed time ONLY.
- FINDKG_REL:n names have UNKNOWN economic meaning.
- Never interpret them as competition, supply chain,
  regulation, acquisition, sentiment, or causality.
- A KG observation is not automatically a new event.
- Never infer stock return from KG relation counts.
- You may cite supplied KG IDs to indicate structural
  context, but not as support for SEC financial claims.
- If there are no verified SEC claims, output neutral
  impact=0, risk=0, confidence=0.
- Preserve cautious confidence: do not increase
  confidence simply because there are more KG edges.
- kg_role must equal "structural_only".
"""


def ollama_json(client, model, system, user):
    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": dump(user),
            },
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 900,
        },
    )

    obj = json.loads(
        response.message.content
    )

    if not isinstance(obj, dict):
        raise ValueError(
            "LLM result is not a JSON object"
        )

    return obj


def validate_sec(obj, hits):
    score = numbers(obj)

    claims = obj.get("claims")

    if not isinstance(claims, list):
        raise ValueError(
            "SEC claims must be a list"
        )

    by_id = {
        h["id"]: h
        for h in hits
    }

    verified = []
    unsupported = []

    for claim in claims:
        if not isinstance(claim, dict):
            unsupported.append({
                "reason": "claim_not_object"
            })
            continue

        evidence_id = claim.get(
            "evidence_id"
        )
        quote = str(
            claim.get("quote", "")
        ).strip()
        interpretation = str(
            claim.get(
                "interpretation", ""
            )
        ).strip()

        if evidence_id not in by_id:
            unsupported.append({
                "reason": "unknown_sec_id",
                "claim": claim,
            })
            continue

        source = by_id[evidence_id]["text"]

        if (
            len(quote) < 15
            or quote not in source
        ):
            unsupported.append({
                "reason": "quote_not_in_excerpt",
                "claim": claim,
            })
            continue

        if not interpretation:
            unsupported.append({
                "reason": "empty_interpretation",
                "claim": claim,
            })
            continue

        verified.append({
            "evidence_id": evidence_id,
            "quote": quote,
            "interpretation": interpretation,
        })

    # A single unsupported claim invalidates the
    # financial signal. Do not silently discard it
    # and retain a possibly contaminated score.
    if unsupported:
        raise ValueError(
            "SEC grounding failed: "
            + dump(unsupported)
        )

    if not verified:
        if (
            score["impact"] != 0
            or score["risk"] != 0
            or score["confidence"] != 0
        ):
            raise ValueError(
                "No SEC claims but nonneutral score"
            )

    score["claims"] = verified
    score["unsupported"] = unsupported

    return score


def kg_structural_summary(hits, day):
    """
    All quantities are computed in Python.
    No LLM-generated KG interpretation.
    """
    relation_counts = {}
    directions = {
        "in": 0,
        "out": 0,
    }

    ages = []

    for hit in hits:
        relation = hit["relation"]

        relation_counts[relation] = (
            relation_counts.get(
                relation, 0
            ) + 1
        )

        direction = hit[
            "direction_from_company"
        ]

        if direction not in directions:
            raise ValueError(
                "Unknown traversal direction"
            )

        directions[direction] += 1

        observed = pd.Timestamp(
            hit["first_observed"]
        )

        if not observed < day:
            raise ValueError(
                "KG temporal violation"
            )

        ages.append(
            (day - observed).days
        )

    return {
        "count": len(hits),
        "relation_counts": relation_counts,
        "direction_counts": directions,
        "min_age_days": (
            min(ages) if ages else None
        ),
        "max_age_days": (
            max(ages) if ages else None
        ),
        "evidence_ids": [
            h["id"]
            for h in hits
        ],
        "semantic_interpretation": (
            "not inferred for opaque relations"
        ),
    }


def validate_fusion(
    obj,
    sec,
    sec_hits,
    kg_hits,
):
    score = numbers(obj)

    if obj.get("kg_role") != "structural_only":
        raise ValueError(
            "KG role must be structural_only"
        )

    allowed_sec = {
        c["evidence_id"]
        for c in sec["claims"]
    }

    allowed_kg = {
        h["id"]
        for h in kg_hits
    }

    sec_ids = unique_ids(
        obj.get("sec_evidence_ids"),
        allowed_sec,
        "sec_evidence_ids",
    )

    kg_ids = unique_ids(
        obj.get("kg_evidence_ids"),
        allowed_kg,
        "kg_evidence_ids",
    )

    if not sec["claims"]:
        if any(
            score[k] != 0
            for k in [
                "impact",
                "risk",
                "confidence",
            ]
        ):
            raise ValueError(
                "Unsupported KG-only financial score"
            )

        if sec_ids:
            raise ValueError(
                "No verified SEC claims to cite"
            )

    else:
        if not sec_ids:
            raise ValueError(
                "Fusion omitted verified SEC citation"
            )

        # No higher confidence merely from KG.
        if (
            score["confidence"]
            > sec["confidence"] + 1e-8
        ):
            raise ValueError(
                "Fusion confidence exceeds SEC grounding"
            )

    return {
        **score,
        "sec_evidence_ids": sec_ids,
        "kg_evidence_ids": kg_ids,
        "kg_role": "structural_only",
    }


def run_one(
    client,
    args,
    ticker,
    day,
    sec_hits,
    kg_hits,
):
    kg_summary = kg_structural_summary(
        kg_hits, day
    )

    if sec_hits:
        sec_raw = ollama_json(
            client,
            args.llm,
            sec_system_prompt(),
            {
                "ticker": ticker,
                "decision_date": str(
                    day.date()
                ),
                "sec_evidence": sec_hits,
            },
        )

        sec = validate_sec(
            sec_raw,
            sec_hits,
        )

        # Evidence-bearing SEC retrieval without a
        # single verifiable claim is NOT a successful
        # grounded analysis. Quarantine this case.
        if not sec["claims"]:
            raise ValueError(
                "SEC evidence retrieved but no verified "
                "claims extracted; abstention requires "
                "manual review before PPO."
            )

    else:
        sec = {
            "impact": 0.0,
            "risk": 0.0,
            "confidence": 0.0,
            "claims": [],
            "unsupported": [],
        }

    if not sec["claims"]:
        fusion = {
            "impact": 0.0,
            "risk": 0.0,
            "confidence": 0.0,
            "sec_evidence_ids": [],
            "kg_evidence_ids": [],
            "kg_role": "structural_only",
        }

    else:
        fusion_raw = ollama_json(
            client,
            args.llm,
            fusion_system_prompt(),
            {
                "ticker": ticker,
                "decision_date": str(
                    day.date()
                ),
                "verified_sec_analysis": sec,
                "kg_structural_summary": (
                    kg_summary
                ),
                "kg_evidence": kg_hits,
            },
        )

        fusion = validate_fusion(
            fusion_raw,
            sec,
            sec_hits,
            kg_hits,
        )

    return {
        "sec_analysis": sec,
        "kg_structure": kg_summary,
        "fusion": fusion,
        "status": "SUPPORTED",
    }


def make_parser():
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
        default=None,
    )

    p.add_argument(
        "--end-date",
        default=None,
    )

    p.add_argument(
        "--max-dates",
        type=int,
        default=0,
    )

    p.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
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
        "--retries",
        type=int,
        default=3,
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v17b2_grounded"
        ),
    )

    return p


def main():
    args = make_parser().parse_args()

    if not set(args.tickers).issubset(
        TICKERS
    ):
        raise ValueError(
            "Unexpected ticker"
        )

    if min(
        args.sec_top_k,
        args.kg_top_k,
        args.retries,
    ) < 1:
        raise ValueError(
            "Invalid retrieval/retry setting"
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

    dates = v17.load_decision_dates(
        args
    )

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    available = {
        item.model
        for item in client.list().models
    }

    if args.llm not in available:
        raise RuntimeError(
            f"Missing model {args.llm}. "
            f"Available: {sorted(available)}"
        )

    cache = v17.Cache(
        args.out / "cache.sqlite"
    )

    success = []
    unsupported = []
    cached = 0
    llm_runs = 0

    audit_path = (
        args.out / "grounding_audit.jsonl"
    )

    try:
        for day in dates:
            day = pd.Timestamp(
                day
            ).normalize()

            for ticker in args.tickers:
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

                # Verify evidence dates independently.
                for h in sec_hits:
                    if not (
                        pd.Timestamp(
                            h["available_from"]
                        ) < day
                    ):
                        raise RuntimeError(
                            "SEC temporal violation"
                        )

                for h in kg_hits:
                    if not (
                        pd.Timestamp(
                            h["first_observed"]
                        ) < day
                    ):
                        raise RuntimeError(
                            "KG temporal violation"
                        )

                payload = {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "model": args.llm,
                    "sec_hits": sec_hits,
                    "kg_hits": kg_hits,
                    "prompt_version": VERSION,
                }

                key = cache_key(payload)

                result = cache.get(key)

                if result is not None:
                    # Revalidate cached result.
                    validated_sec = validate_sec(
                        result["sec_analysis"],
                        sec_hits,
                    )

                    kg_structural_summary(
                        kg_hits, day
                    )

                    validate_fusion(
                        result["fusion"],
                        validated_sec,
                        sec_hits,
                        kg_hits,
                    )

                    cached += 1

                else:
                    last_error = None

                    for attempt in range(
                        args.retries
                    ):
                        try:
                            result = run_one(
                                client,
                                args,
                                ticker,
                                day,
                                sec_hits,
                                kg_hits,
                            )
                            last_error = None
                            llm_runs += 1
                            break

                        except Exception as exc:
                            last_error = exc

                            log(
                                "[RETRY]",
                                day.date(),
                                ticker,
                                attempt+1,
                                str(exc)[:300],
                            )

                            time.sleep(
                                min(2**attempt, 8)
                            )

                    if last_error is not None:
                        record = {
                            "date": str(day.date()),
                            "ticker": ticker,
                            "status": "UNSUPPORTED",
                            "error": str(
                                last_error
                            )[:1000],
                            "cache_key": key,
                        }

                        unsupported.append(
                            record
                        )

                        append_jsonl(
                            audit_path,
                            record,
                        )

                        log(
                            "[UNSUPPORTED]",
                            day.date(),
                            ticker,
                        )

                        continue

                    cache.put(
                        key, result
                    )

                fusion = result["fusion"]
                sec_analysis = (
                    result["sec_analysis"]
                )
                kg_structure = (
                    result["kg_structure"]
                )

                row = {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "impact": fusion["impact"],
                    "risk": fusion["risk"],
                    "confidence": (
                        fusion["confidence"]
                    ),
                    "sec_impact": (
                        sec_analysis["impact"]
                    ),
                    "sec_risk": (
                        sec_analysis["risk"]
                    ),
                    "sec_confidence": (
                        sec_analysis["confidence"]
                    ),
                    "sec_verified_claims": len(
                        sec_analysis["claims"]
                    ),
                    "kg_structural_edges": (
                        kg_structure["count"]
                    ),
                    "sec_cited": len(
                        fusion[
                            "sec_evidence_ids"
                        ]
                    ),
                    "kg_cited": len(
                        fusion[
                            "kg_evidence_ids"
                        ]
                    ),
                    "status": "SUPPORTED",
                    "cache_key": key,
                }

                success.append(row)

                append_jsonl(
                    audit_path,
                    {
                        **row,
                        "sec_analysis": (
                            sec_analysis
                        ),
                        "kg_structure": (
                            kg_structure
                        ),
                        "fusion": fusion,
                    },
                )

                log(
                    "[GROUNDED]",
                    row["date"],
                    ticker,
                    "impact=",
                    row["impact"],
                    "risk=",
                    row["risk"],
                    "confidence=",
                    row["confidence"],
                    "SEC claims=",
                    row[
                        "sec_verified_claims"
                    ],
                    "KG edges=",
                    row[
                        "kg_structural_edges"
                    ],
                )

    finally:
        cache.close()

        pd.DataFrame(
            success,
            columns=[
                "date", "ticker",
                "impact", "risk",
                "confidence",
                "sec_impact",
                "sec_risk",
                "sec_confidence",
                "sec_verified_claims",
                "kg_structural_edges",
                "sec_cited",
                "kg_cited",
                "status",
                "cache_key",
            ],
        ).to_csv(
            args.out / "signals.csv",
            index=False,
        )

        pd.DataFrame(
            unsupported,
            columns=[
                "date", "ticker",
                "status", "error",
                "cache_key",
            ],
        ).to_csv(
            args.out / "unsupported.csv",
            index=False,
        )

    expected = (
        len(dates)
        * len(args.tickers)
    )

    manifest = {
        "experiment": "FinRL-X V17-B2",
        "version": VERSION,
        "architecture": (
            "SEC grounded reasoning + "
            "structural KG + Qwen fusion"
        ),
        "decision_dates": len(dates),
        "tickers": args.tickers,
        "expected_rows": expected,
        "supported_rows": len(success),
        "unsupported_rows": len(
            unsupported
        ),
        "cache_hits": cached,
        "new_successful_cases": (
            llm_runs
        ),
        "model": args.llm,
        "kg_relation_types": int(
            kg["relation"].nunique()
        ),
        "test_2022_2023_used": False,
        "limitations": [
            "Quoted excerpts are checked verbatim, not semantically entailed.",
            "KG first-observed time is not independently certified.",
            "Unknown FinDKG relations are structural only.",
            "LLM numeric scores are not calibrated.",
            "No unsupported signals may enter PPO.",
            "No RL training in V17-B2.",
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
    log("========== V17-B2 SUMMARY ==========")
    log(
        pd.DataFrame(success).to_string(
            index=False
        ) if success else "No valid signals"
    )

    log("[EXPECTED]", expected)
    log("[SUPPORTED]", len(success))
    log("[UNSUPPORTED]", len(unsupported))
    log("[CACHE HITS]", cached)
    log("[NEW SUCCESSFUL CASES]", llm_runs)

    if len(success) != expected:
        raise RuntimeError(
            "V17-B2 incomplete. Inspect unsupported.csv "
            "before using signals in PPO."
        )

    log("V17-B2: PASS")


if __name__ == "__main__":
    main()
