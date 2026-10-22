#!/usr/bin/env python3

import argparse
import asyncio
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import ollama

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import (
    ollama_model_complete,
    ollama_embed,
)
from lightrag.utils import EmbeddingFunc


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "results/light_rag_dynamic_kg"

OUTPUT_DIR = BASE / "grounded"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

HOST = "http://140.113.87.164:11434"

LLM_MODEL = "qwen3.5:27b"

EMBED_MODEL = "bge-m3:latest"

TICKERS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
]


# ============================================================
# UTILITIES
# ============================================================

def normalize(value):
    return re.sub(
        r"\s+",
        " ",
        str(value).strip()
    ).casefold()


def parse_json_object(content):
    text = str(content).strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
        )
        text = re.sub(
            r"\s*```$",
            "",
            text,
        )

    obj = json.loads(text)

    if not isinstance(obj, dict):
        raise ValueError(
            "Expected a JSON object."
        )

    return obj


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# LOAD EXISTING LIGHTRAG SOURCE MODULE
# ============================================================

def load_existing_module():
    path = (
        ROOT /
        "experiments/light_rag_dynamic_kg/"
        "run_official_lightrag.py"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing original LightRAG script: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "existing_lightrag_pipeline",
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load original LightRAG script."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# LOAD EXISTING HISTORICAL SNAPSHOT
# ============================================================

def load_existing_snapshot(ticker, date):
    result_file = (
        BASE /
        f"lightrag_{ticker}_{date}.json"
    )

    if not result_file.exists():
        raise FileNotFoundError(
            f"LightRAG result does not exist: "
            f"{result_file}"
        )

    result = json.loads(
        result_file.read_text(
            encoding="utf-8"
        )
    )

    if result.get("ticker") != ticker:
        raise RuntimeError(
            "Snapshot ticker mismatch."
        )

    if result.get("decision_date") != date:
        raise RuntimeError(
            "Snapshot date mismatch."
        )

    fingerprint = result.get(
        "snapshot_fingerprint"
    )

    if not fingerprint:
        raise RuntimeError(
            "Snapshot fingerprint missing."
        )

    snapshot = (
        BASE /
        "snapshots" /
        f"{ticker}_{date}_{fingerprint}"
    )

    if not snapshot.exists():
        raise FileNotFoundError(
            f"Snapshot directory missing: {snapshot}"
        )

    ready_marker = (
        snapshot /
        "index_ready.json"
    )

    if not ready_marker.exists():
        raise RuntimeError(
            "Existing snapshot is not marked ready."
        )

    return snapshot, result


# ============================================================
# RECONSTRUCT ORIGINAL EVIDENCE REGISTRY
# ============================================================

def build_evidence_registry(
    module,
    ticker,
    date_string,
    kg_k,
    sec_k,
):
    """
    Read original historical KG and SEC inputs.

    Do not treat a generated LLM answer as
    original evidence.
    """

    date = pd.Timestamp(
        date_string
    ).normalize()

    companies = (
        module.load_company_mapping()
    )

    entity_names = (
        module.load_entity_names()
    )

    relation_names = (
        module.load_relation_names()
    )

    kg_rows = module.load_historical_kg(
        ticker,
        date,
        kg_k,
        entity_names,
        relation_names,
        companies,
    )

    sec_rows = module.load_historical_sec(
        ticker,
        date,
        sec_k,
    )

    registry = {}

    cutoff = pd.Timestamp(
        date_string
    )

    for row in kg_rows:
        observed = pd.to_datetime(
            row["available_at"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(observed):
            continue

        if (
            observed.tz_localize(None)
            >= cutoff
        ):
            continue

        source_id = (
            "KG_" +
            str(row["event_id"])
        )

        statement = (
            f"{row['head']} "
            f"--[{row['relation']}]--> "
            f"{row['tail']}"
        )

        registry[source_id] = {
            "id": source_id,
            "type": "KG",
            "available_at": (
                row["available_at"]
            ),
            "statement": statement,
            "source": row["source"],
            "relation": row["relation"],
            "original_event_id": (
                row["event_id"]
            ),
        }

    for row in sec_rows:
        observed = pd.to_datetime(
            row["available_at"],
            utc=True,
            errors="coerce",
        )

        if pd.isna(observed):
            continue

        if (
            observed.tz_localize(None)
            >= cutoff
        ):
            continue

        source_id = (
            "SEC_" +
            str(row["evidence_id"])
        )

        registry[source_id] = {
            "id": source_id,
            "type": "SEC",
            "available_at": (
                row["available_at"]
            ),
            "form": row["form"],
            "statement": row["text"],
            "original_evidence_id": (
                row["evidence_id"]
            ),
        }

    return registry


# ============================================================
# RETRIEVED SOURCE MATCHING
# ============================================================

def filter_retrieved_sources(
    context,
    registry,
):
    """
    Only allow source records that are
    actually identifiable in LightRAG context.

    KG:
      Original event ID must occur in context.

    SEC:
      Original evidence ID or a substantial
      verbatim excerpt must occur in context.

    This is a conservative retrieval linkage,
    not a semantic entailment verifier.
    """

    context_text = str(context)

    normalized_context = normalize(
        context_text
    )

    eligible = {}

    for source_id, item in registry.items():

        if item["type"] == "KG":

            event_id = (
                item["original_event_id"]
            )

            if event_id in context_text:
                eligible[source_id] = item

        elif item["type"] == "SEC":

            evidence_id = (
                item["original_evidence_id"]
            )

            statement = normalize(
                item["statement"]
            )

            matched_by_id = (
                evidence_id in context_text
            )

            matched_by_text = False

            if len(statement) >= 100:
                matched_by_text = (
                    statement[:100]
                    in normalized_context
                )

            if (
                matched_by_id
                or matched_by_text
            ):
                eligible[source_id] = item

    return eligible


# ============================================================
# GENERATION VALIDATION
# ============================================================

def validate_generation(
    generated,
    retrieved_registry,
):
    """
    Verify JSON structure, source IDs
    and verbatim support quotations.

    IMPORTANT:
    This is not full semantic entailment.
    """

    valid_status = {
        "supported",
        "partial",
        "insufficient",
    }

    status = generated.get(
        "grounding_status"
    )

    if status not in valid_status:
        raise ValueError(
            f"Invalid grounding status: {status}"
        )

    claims = generated.get(
        "claims",
        []
    )

    if not isinstance(claims, list):
        raise ValueError(
            "claims must be a list"
        )

    accepted = []
    rejected = []

    for claim in claims:

        if not isinstance(
            claim, dict
        ):
            rejected.append({
                "claim": str(claim),
                "reason": (
                    "claim_not_an_object"
                ),
            })
            continue

        evidence_ids = claim.get(
            "evidence_ids",
            []
        )

        support_quote = claim.get(
            "support_quote",
            ""
        )

        reason = None

        if (
            not isinstance(
                evidence_ids,
                list
            )
            or len(evidence_ids) != 1
        ):
            reason = (
                "exactly_one_source_required"
            )

        else:
            source_id = (
                evidence_ids[0]
            )

            if (
                source_id
                not in retrieved_registry
            ):
                reason = (
                    "source_not_retrieved"
                )

            elif (
                not isinstance(
                    support_quote,
                    str
                )
                or len(
                    support_quote.strip()
                ) < 15
            ):
                reason = (
                    "support_quote_missing"
                )

            else:
                original = normalize(
                    retrieved_registry[
                        source_id
                    ]["statement"]
                )

                quote = normalize(
                    support_quote
                )

                if quote not in original:
                    reason = (
                        "quote_not_verbatim"
                    )

        if reason is None:
            accepted.append(claim)
        else:
            rejected.append({
                "claim": claim,
                "reason": reason,
            })

    numeric = {}

    for field, low, high in [
        ("impact", -1.0, 1.0),
        ("risk", 0.0, 1.0),
        ("confidence", 0.0, 1.0),
    ]:

        value = generated.get(
            field
        )

        if value is None:
            numeric[field] = None
            continue

        if isinstance(value, bool):
            numeric[field] = None
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            number = float("nan")

        if (
            np.isfinite(number)
            and low <= number <= high
        ):
            numeric[field] = number
        else:
            numeric[field] = None

    format_quote_passed = bool(
        accepted
        and not rejected
    )

    if not format_quote_passed:
        numeric = {
            "impact": None,
            "risk": None,
            "confidence": None,
        }

    return {
        "accepted_claims": accepted,
        "rejected_claims": rejected,
        "candidate_numeric_signals": numeric,
        "format_and_quote_checks_passed": (
            format_quote_passed
        ),
        "semantic_grounding_verified": False,
        "ready_for_ppo": False,
    }


# ============================================================
# MAIN
# ============================================================

async def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker",
        choices=TICKERS,
        default="AAPL",
    )

    parser.add_argument(
        "--date",
        default="2021-12-30",
    )

    parser.add_argument(
        "--kg-k",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--sec-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    ticker = args.ticker

    date_string = args.date

    date = pd.Timestamp(
        date_string
    ).normalize()

    if not (
        pd.Timestamp("2018-01-01")
        <= date
        < pd.Timestamp("2022-01-01")
    ):
        raise RuntimeError(
            "Only 2018–2021 dates are allowed."
        )

    # --------------------------------------------------------
    # LOAD EXISTING SNAPSHOT
    # --------------------------------------------------------

    module = load_existing_module()

    snapshot, previous_result = (
        load_existing_snapshot(
            ticker,
            date_string
        )
    )

    print()
    print("=" * 78)
    print("OFFICIAL LIGHTRAG GROUNDED GENERATION")
    print("=" * 78)

    print("[TICKER]", ticker)
    print("[DATE]", date_string)
    print("[SNAPSHOT]", snapshot)
    print("[OLLAMA HOST]", HOST)
    print("[QWEN MODEL]", LLM_MODEL)
    print("[EMBEDDING MODEL]", EMBED_MODEL)

    # --------------------------------------------------------
    # REBUILD ORIGINAL EVIDENCE REGISTRY
    # --------------------------------------------------------

    registry = build_evidence_registry(
        module,
        ticker,
        date_string,
        args.kg_k,
        args.sec_k,
    )

    print(
        "[REGISTERED EVIDENCE]",
        len(registry)
    )

    # --------------------------------------------------------
    # LOAD LIGHTRAG
    # --------------------------------------------------------

    rag = LightRAG(
        working_dir=str(snapshot),

        llm_model_name=LLM_MODEL,

        llm_model_func=(
            ollama_model_complete
        ),

        llm_model_kwargs={
            "host": HOST,
            "options": {
                "num_ctx": 8192,
            },
            "timeout": 1200,
        },

        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=partial(
                ollama_embed.func,
                embed_model=EMBED_MODEL,
                host=HOST,
            ),
        ),
    )

    await rag.initialize_storages()

    # --------------------------------------------------------
    # OFFICIAL LIGHTRAG RETRIEVAL
    # --------------------------------------------------------

    query = (
        f"Apple Inc. SEC 10-K and Knowledge Graph "
        f"financial business risks as of {date_string}. "
        f"Retrieve SEC filing excerpts and relevant "
        f"historical KG relationships. "
        f"Do not infer causality from generic relations."
    )

    try:

        print()
        print("[LIGHTRAG] Hybrid retrieval started")

        context = await rag.aquery(
            query,
            param=QueryParam(
                mode="mix",
                only_need_context=True,
                enable_rerank=False,
            ),
        )

    finally:

        await rag.finalize_storages()

    if isinstance(context, str):
        context_text = context
    else:
        context_text = json.dumps(
            context,
            ensure_ascii=False,
            default=str,
        )

    context_file = (
        OUTPUT_DIR /
        f"{ticker}_{date_string}_retrieved_context.txt"
    )

    context_file.write_text(
        context_text,
        encoding="utf-8",
    )

    print(
        "[RETRIEVED CONTEXT LENGTH]",
        len(context_text)
    )

    # --------------------------------------------------------
    # MATCH RETRIEVED CONTEXT TO ORIGINAL EVIDENCE
    # --------------------------------------------------------

    retrieved = (
        filter_retrieved_sources(
            context_text,
            registry,
        )
    )

    kg_count = sum(
        item["type"] == "KG"
        for item in retrieved.values()
    )

    sec_count = sum(
        item["type"] == "SEC"
        for item in retrieved.values()
    )

    print(
        "[RETRIEVED KG SOURCES]",
        kg_count
    )

    print(
        "[RETRIEVED SEC SOURCES]",
        sec_count
    )

    if not retrieved:
        raise RuntimeError(
            "No original source IDs could be "
            "linked to LightRAG retrieved context. "
            "Inspect the saved context file."
        )

    # --------------------------------------------------------
    # GROUNDED QWEN GENERATION
    # --------------------------------------------------------

    client = ollama.Client(
        host=HOST,
        timeout=1200,
    )

    instructions = """
You are an evidence-grounded financial research assistant.

Analyze only the historical evidence supplied
in the user message.

Rules:

1. Do not rely on external knowledge.
2. Do not invent companies, events, financial values,
   business relationships, or dates.
3. Do not infer causality from generic KG relations.
4. A KG triple alone may not justify a financial risk.
5. Every claim must cite exactly ONE retrieved source ID.
6. Every claim must include a verbatim support_quote
   copied from that source statement.
7. Never interpret a citation as proof of causality.
8. Prefer abstention rather than unsupported analysis.
9. If the available evidence is insufficient,
   return null for numerical scores.
10. Return JSON only.

Required JSON format:

{
  "summary": "string",
  "claims": [
    {
      "claim": "string",
      "evidence_ids": ["KG_..."],
      "support_quote": "verbatim source substring"
    }
  ],
  "impact": null,
  "risk": null,
  "confidence": null,
  "grounding_status": "insufficient"
}

Allowed score ranges:

impact: -1 to 1, or null
risk: 0 to 1, or null
confidence: 0 to 1, or null

grounding_status:
supported / partial / insufficient
"""

    payload = {
        "ticker": ticker,
        "decision_date": date_string,
        "retrieved_evidence": (
            list(retrieved.values())
        ),
    }

    print()
    print("[QWEN] Generating grounded financial signals")


    # --------------------------------------------------------
    # ROBUST QWEN JSON GENERATION
    # --------------------------------------------------------

    # Keep generation bounded on shared GPU servers.
    # Reduce retrieved evidence to avoid overly long prompts.
    evidence_for_qwen = list(
        retrieved.values()
    )

    # Prefer some SEC evidence whenever retrieved.
    sec_evidence = [
        x for x in evidence_for_qwen
        if x["type"] == "SEC"
    ][:5]

    kg_evidence = [
        x for x in evidence_for_qwen
        if x["type"] == "KG"
    ][:15]

    payload["retrieved_evidence"] = (
        sec_evidence + kg_evidence
    )

    print(
        "[QWEN EVIDENCE] KG=",
        len(kg_evidence),
        "SEC=",
        len(sec_evidence),
        flush=True,
    )

    # Avoid model wasting output budget on long reports.
    instructions += """
Additional requirements:
- Return at most THREE claims.
- Keep summary under 80 words.
- Keep each claim under 50 words.
- JSON must be a single object.
- Do not output markdown code fences.
- For generic graph relations, state only
  the literal relationship, not financial causality.
"""

    raw_dir = OUTPUT_DIR / "raw_qwen"
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated = None
    generation_errors = []

    attempts = [
        {
            "name": "thinking_json",
            "think": True,
            "format": "json",
            "num_predict": 4096,
        },
        {
            "name": "nonthinking_prompt_json",
            "think": False,
            "format": None,
            "num_predict": 1536,
        },
    ]

    for attempt in attempts:

        print(
            "[QWEN ATTEMPT]",
            attempt["name"],
            flush=True,
        )

        kwargs = {
            "model": LLM_MODEL,
            "think": attempt["think"],
            "messages": [
                {
                    "role": "system",
                    "content": instructions,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        payload,
                        ensure_ascii=False,
                    ),
                },
            ],
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": attempt[
                    "num_predict"
                ],
            },
        }

        if attempt["format"] is not None:
            kwargs["format"] = attempt["format"]

        try:
            response = client.chat(**kwargs)

            message = response.message

            raw_content = (
                getattr(message, "content", None)
                or ""
            )

            raw_thinking = (
                getattr(message, "thinking", None)
                or ""
            )

            done_reason = getattr(
                response,
                "done_reason",
                None,
            )

            debug_path = (
                raw_dir /
                f"{ticker}_{date_string}_"
                f"{attempt['name']}.json"
            )

            save_json(
                debug_path,
                {
                    "attempt": attempt,
                    "done_reason": done_reason,
                    "content": raw_content,
                    "thinking_length": len(
                        raw_thinking
                    ),
                    "eval_count": getattr(
                        response,
                        "eval_count",
                        None,
                    ),
                },
            )

            print(
                "[QWEN DONE REASON]",
                done_reason,
                flush=True,
            )

            print(
                "[QWEN CONTENT LENGTH]",
                len(raw_content),
                flush=True,
            )

            print(
                "[QWEN THINKING LENGTH]",
                len(raw_thinking),
                flush=True,
            )

            print(
                "[QWEN RAW PREVIEW]",
                repr(raw_content[:300]),
                flush=True,
            )

            if not raw_content.strip():
                raise ValueError(
                    "Empty Qwen content. "
                    f"done_reason={done_reason}; "
                    f"thinking_length={len(raw_thinking)}"
                )

            generated = parse_json_object(
                raw_content
            )

            required = {
                "summary",
                "claims",
                "impact",
                "risk",
                "confidence",
                "grounding_status",
            }

            missing = (
                required - set(generated)
            )

            if missing:
                raise ValueError(
                    f"Missing JSON fields: "
                    f"{sorted(missing)}"
                )

            print(
                "[QWEN JSON] VALID",
                flush=True,
            )

            break

        except Exception as exc:

            generation_errors.append(
                {
                    "attempt": attempt["name"],
                    "error": repr(exc),
                }
            )

            print(
                "[QWEN WARNING]",
                repr(exc),
                flush=True,
            )

    if generated is None:
        failure_path = (
            raw_dir /
            f"{ticker}_{date_string}_"
            "generation_failures.json"
        )

        save_json(
            failure_path,
            generation_errors,
        )

        raise RuntimeError(
            "Both Qwen JSON attempts failed. "
            f"Diagnostics saved to {failure_path}"
        )

    # --------------------------------------------------------
    # EVIDENCE VALIDATION
    # --------------------------------------------------------

    validation = (
        validate_generation(
            generated,
            retrieved,
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    result = {
        "ticker": ticker,
        "decision_date": date_string,

        "method": (
            "Official LightRAG Hybrid "
            "+ Grounded Qwen Generation"
        ),

        "ollama_host": HOST,
        "llm_model": LLM_MODEL,
        "embedding_model": EMBED_MODEL,

        "snapshot_fingerprint": (
            previous_result[
                "snapshot_fingerprint"
            ]
        ),

        "retrieved_kg_sources": kg_count,
        "retrieved_sec_sources": sec_count,

        "retrieved_source_ids": (
            list(retrieved.keys())
        ),

        "generation": generated,
        "validation": validation,

        "strict_pit_certified": False,
        "semantic_grounding_verified": False,
        "ppo_integrated": False,
    }

    output = (
        OUTPUT_DIR /
        f"joint_signal_v1_{ticker}_{date_string}.json"
    )

    save_json(
        output,
        result,
    )

    print()
    print("=" * 78)
    print("GROUNDED LIGHTRAG RESULT")
    print("=" * 78)

    print(
        json.dumps(
            {
                "ticker": ticker,
                "date": date_string,
                "retrieved_kg": kg_count,
                "retrieved_sec": sec_count,

                "accepted_claims": len(
                    validation[
                        "accepted_claims"
                    ]
                ),

                "rejected_claims": len(
                    validation[
                        "rejected_claims"
                    ]
                ),

                "candidate_numeric_signals": (
                    validation[
                        "candidate_numeric_signals"
                    ]
                ),

                "ready_for_ppo": (
                    validation[
                        "ready_for_ppo"
                    ]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("[SAVED RESULT]", output)
    print("[SAVED RETRIEVED CONTEXT]", context_file)


if __name__ == "__main__":
    asyncio.run(main())
