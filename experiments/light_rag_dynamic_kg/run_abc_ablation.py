#!/usr/bin/env python3

import argparse
import asyncio
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
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "results/light_rag_dynamic_kg"
OUT = BASE / "ablation_v1"
OUT.mkdir(parents=True, exist_ok=True)

HOST = "http://140.113.87.164:11434"
MODEL = "qwen3.5:27b"
EMBED_MODEL = "bge-m3:latest"

CONFIGS = {
    "A_KG_RAG": {
        "mode": "hybrid",
        "allowed": {"KG"},
    },
    "B_SEC_RAG": {
        "mode": "naive",
        "allowed": {"SEC"},
    },
    "C_KG_SEC_RAG": {
        "mode": "mix",
        "allowed": {"KG", "SEC"},
    },
}

# Same evidence budgets across comparable branches.
MAX_KG = 12
MAX_SEC = 5


# ============================================================
# HELPERS
# ============================================================

def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def normalize(x):
    return re.sub(
        r"\s+",
        " ",
        str(x).strip(),
    ).casefold()


def parse_json(text):
    text = str(text).strip()

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

    result = json.loads(text)

    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")

    return result


def load_module():
    path = (
        ROOT /
        "experiments/light_rag_dynamic_kg/"
        "run_official_lightrag.py"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "existing_official_lightrag",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def load_snapshot(ticker, date):
    result_path = (
        BASE / f"lightrag_{ticker}_{date}.json"
    )

    if not result_path.exists():
        raise FileNotFoundError(result_path)

    previous = json.loads(result_path.read_text())

    if (
        previous.get("ticker") != ticker
        or previous.get("decision_date") != date
    ):
        raise RuntimeError("Snapshot metadata mismatch")

    fingerprint = previous["snapshot_fingerprint"]

    snapshot = (
        BASE / "snapshots" /
        f"{ticker}_{date}_{fingerprint}"
    )

    ready = snapshot / "index_ready.json"

    if not ready.exists():
        raise RuntimeError(
            f"Snapshot is not ready: {snapshot}"
        )

    marker = json.loads(ready.read_text())

    if marker.get("fingerprint") != fingerprint:
        raise RuntimeError(
            "Index marker fingerprint mismatch"
        )

    return snapshot, marker


# ============================================================
# RECONSTRUCT THE INDEXED SOURCE EVIDENCE
# ============================================================

def build_registry(
    module,
    ticker,
    date,
    kg_k,
    sec_k,
):
    companies = module.load_company_mapping()
    entities = module.load_entity_names()
    relations = module.load_relation_names()

    cutoff = pd.Timestamp(date).normalize()

    kg_rows = module.load_historical_kg(
        ticker,
        cutoff,
        kg_k,
        entities,
        relations,
        companies,
    )

    sec_rows = module.load_historical_sec(
        ticker,
        cutoff,
        sec_k,
    )

    registry = {}

    for row in kg_rows:
        head = str(row["head"]).strip()
        tail = str(row["tail"]).strip()

        # Matches the original custom KG importer:
        # self-loops were excluded from LightRAG.
        if head == tail:
            continue

        source_id = "KG_" + str(row["event_id"])

        available = pd.to_datetime(
            row["available_at"],
            utc=True,
            errors="coerce",
        )

        if (
            pd.isna(available)
            or available.tz_localize(None) >= cutoff
        ):
            continue

        registry[source_id] = {
            "id": source_id,
            "type": "KG",
            "available_at": row["available_at"],
            "statement": (
                f"{head} --[{row['relation']}]--> {tail}"
            ),
            "relation": row["relation"],
            "event_id": str(row["event_id"]),
        }

    for row in sec_rows:
        available = pd.to_datetime(
            row["available_at"],
            utc=True,
            errors="coerce",
        )

        if (
            pd.isna(available)
            or available.tz_localize(None) >= cutoff
        ):
            continue

        source_id = "SEC_" + str(row["evidence_id"])

        registry[source_id] = {
            "id": source_id,
            "type": "SEC",
            "available_at": row["available_at"],
            "form": row["form"],
            "statement": row["text"],
            "evidence_id": str(row["evidence_id"]),
        }

    return registry


# ============================================================
# LINK LIGHTRAG CONTEXT BACK TO ORIGINAL EVIDENCE
# ============================================================

def match_sources(context, registry, allowed):
    context_text = str(context)
    normalized_context = normalize(context_text)

    results = {}

    for source_id, item in registry.items():
        if item["type"] not in allowed:
            continue

        if item["type"] == "KG":
            matched = item["event_id"] in context_text

        else:
            matched = item["evidence_id"] in context_text

            if not matched:
                statement = normalize(item["statement"])

                matched = (
                    len(statement) >= 100
                    and statement[:100]
                    in normalized_context
                )

        if matched:
            results[source_id] = item

    return results


def select_evidence(registry):
    kg = [
        x for x in registry.values()
        if x["type"] == "KG"
    ][:MAX_KG]

    sec = [
        x for x in registry.values()
        if x["type"] == "SEC"
    ][:MAX_SEC]

    return kg + sec


# ============================================================
# UNIFIED GENERATION PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an evidence-grounded financial research model.

Your task is to analyze the financial and operational
implications of historical evidence for one company.

Use ONLY the evidence in the user message.

All configurations use identical scoring definitions.

SCORING DEFINITIONS:

impact:
  Estimate the directional effect of the supplied
  evidence on the company's future operating or
  financial outlook.

  Range -1 to +1:
  -1 = strongly adverse
   0 = approximately neutral effect
  +1 = strongly favorable

  This is NOT a prediction of stock return.

risk:
  Estimate the severity of adverse business or
  financial risks supported by the evidence.

  Range 0 to 1:
  0 = minimal indicated risk
  1 = severe indicated risk

confidence:
  Subjective assessment of how sufficient and
  specific the evidence is for the scores.

  Range 0 to 1.
  This is NOT a calibrated probability.

IMPORTANT:

- Unknown is NOT neutral.
- If evidence cannot support a directional estimate,
  impact must be null, not 0.
- If evidence cannot support a risk estimate,
  risk must be null, not 0.
- If scores cannot be supported, confidence may
  also be null.
- Do not invent company events, relationships,
  dates, revenues, or financial outcomes.
- Do not infer causality from Relate_To or Has.
- A relation such as Control does not automatically
  establish a regulatory or financial risk.
- Do not assume all KG triples are factually correct.
- No information after the decision date is allowed.
- Do not recommend buying or selling shares.

CLAIM REQUIREMENTS:

- At most three claims.
- Each claim must cite exactly one source ID.
- Each claim must include a verbatim support_quote
  copied from that source statement.
- The claim must not assert more than the quote supports.
- If no useful claim is supported, return an empty list.

OUTPUT:

Return one valid JSON object only.
No markdown, no commentary.

{
  "summary": "",
  "claims": [
    {
      "claim": "",
      "evidence_ids": ["KG_..."],
      "support_quote": ""
    }
  ],
  "impact": null,
  "risk": null,
  "confidence": null,
  "grounding_status": "insufficient"
}

grounding_status must be one of:
supported, partial, insufficient

Keep summary under 80 words.
"""


# ============================================================
# QWEN GENERATION
# ============================================================

def generate(client, ticker, date, group, evidence):
    payload = {
        "ticker": ticker,
        "decision_date": date,
        "configuration": group,
        "retrieved_evidence": evidence,
    }

    response = client.chat(
        model=MODEL,
        think=False,
        format="json",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                ),
            },
        ],
        options={
            "temperature": 0,
            "num_ctx": 8192,
            "num_predict": 2048,
        },
    )

    raw = response.message.content or ""

    diagnostics = {
        "done_reason": getattr(
            response, "done_reason", None
        ),
        "content_length": len(raw),
        "raw_content": raw,
    }

    if not raw.strip():
        return None, diagnostics

    try:
        generated = parse_json(raw)
    except Exception as exc:
        diagnostics["parse_error"] = repr(exc)
        return None, diagnostics

    required = {
        "summary",
        "claims",
        "impact",
        "risk",
        "confidence",
        "grounding_status",
    }

    if not required.issubset(generated):
        diagnostics["missing_fields"] = sorted(
            required - set(generated)
        )
        return None, diagnostics

    return generated, diagnostics


# ============================================================
# VERIFY CITATIONS / QUOTES / NUMERIC FIELDS
# ============================================================

def validate(generated, evidence):
    if generated is None:
        return {
            "accepted": [],
            "rejected": [],
            "impact": None,
            "risk": None,
            "confidence": None,
            "missing": True,
            "status": "generation_failed",
            "semantic_verified": False,
            "ready_for_ppo": False,
        }

    lookup = {
        x["id"]: x
        for x in evidence
    }

    accepted = []
    rejected = []

    claims = generated.get("claims", [])

    if not isinstance(claims, list):
        claims = []
        rejected.append({
            "reason": "invalid_claims_field"
        })

    for claim in claims:
        reason = None

        if not isinstance(claim, dict):
            reason = "not_object"
            ids = []
            quote = ""
        else:
            ids = claim.get("evidence_ids", [])
            quote = claim.get("support_quote", "")

        if reason is None:
            if (
                not isinstance(ids, list)
                or len(ids) != 1
            ):
                reason = "one_source_required"

            elif ids[0] not in lookup:
                reason = "unknown_source"

            elif not isinstance(quote, str):
                reason = "invalid_quote"

            elif len(quote.strip()) < 15:
                reason = "short_quote"

            elif (
                normalize(quote)
                not in normalize(
                    lookup[ids[0]]["statement"]
                )
            ):
                reason = "quote_not_in_source"

        if reason:
            rejected.append({
                "claim": claim,
                "reason": reason,
            })
        else:
            accepted.append(claim)

    numeric = {}

    for key, lo, hi in [
        ("impact", -1, 1),
        ("risk", 0, 1),
        ("confidence", 0, 1),
    ]:
        value = generated.get(key)

        if value is None or isinstance(value, bool):
            numeric[key] = None
            continue

        try:
            value = float(value)
        except (ValueError, TypeError):
            value = float("nan")

        numeric[key] = (
            value
            if np.isfinite(value) and lo <= value <= hi
            else None
        )

    status = generated.get("grounding_status")

    if status not in {
        "supported",
        "partial",
        "insufficient",
    }:
        status = "invalid"

    quote_valid = (
        bool(accepted)
        and not rejected
    )

    # Avoid treating unsupported numeric outputs
    # as usable financial features.
    if not quote_valid or status == "invalid":
        numeric = {
            "impact": None,
            "risk": None,
            "confidence": None,
        }

    missing = any(
        value is None
        for value in numeric.values()
    )

    return {
        "accepted": accepted,
        "rejected": rejected,
        "impact": numeric["impact"],
        "risk": numeric["risk"],
        "confidence": numeric["confidence"],
        "missing": missing,
        "status": status,
        "quote_checks_passed": quote_valid,
        "semantic_verified": False,
        "ready_for_ppo": False,
    }


# ============================================================
# MAIN A/B/C EXPERIMENT
# ============================================================

async def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ticker", default="AAPL"
    )
    parser.add_argument(
        "--date", default="2021-12-30"
    )
    parser.add_argument(
        "--kg-k", type=int, default=80
    )
    parser.add_argument(
        "--sec-k", type=int, default=5
    )

    args = parser.parse_args()

    ticker = args.ticker.upper()
    date = args.date

    date_value = pd.Timestamp(date).normalize()

    if not (
        pd.Timestamp("2018-01-01")
        <= date_value
        < pd.Timestamp("2022-01-01")
    ):
        raise RuntimeError(
            "Ablation restricted to 2018–2021."
        )

    module = load_module()
    snapshot, marker = load_snapshot(
        ticker, date
    )

    # IMPORTANT: Reconstruct using the same
    # source counts used by the existing index.
    kg_k = int(marker["kg_events"])
    sec_k = int(marker["sec_chunks"])

    if args.kg_k != kg_k or args.sec_k != sec_k:
        print(
            "[NOTICE] Using snapshot source limits:",
            kg_k, sec_k,
        )

    registry = build_registry(
        module,
        ticker,
        date,
        kg_k,
        sec_k,
    )

    print("=" * 72)
    print("A/B/C OFFICIAL LIGHTRAG ABLATION")
    print("=" * 72)
    print("[TICKER]", ticker)
    print("[DATE]", date)
    print("[SNAPSHOT]", snapshot)
    print("[REGISTERED EVIDENCE]", len(registry))
    print("[OLLAMA]", HOST)
    print("[QWEN]", MODEL)

    client = ollama.Client(
        host=HOST,
        timeout=1200,
    )

    rag = LightRAG(
        working_dir=str(snapshot),

        llm_model_name=MODEL,

        llm_model_func=ollama_model_complete,

        llm_model_kwargs={
            "host": HOST,
            "timeout": 600,
            "options": {
                "num_ctx": 8192,
            },
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

    summary_rows = []

    try:
        for group, cfg in CONFIGS.items():
            print("\n" + "=" * 72)
            print("[GROUP]", group)
            print("[MODE]", cfg["mode"])
            print("=" * 72)

            company_names = {
                "AAPL": "Apple Inc.",
                "AMZN": "Amazon.com Inc.",
                "GOOGL": "Alphabet Inc.",
                "MSFT": "Microsoft Corporation",
                "NVDA": "NVIDIA Corporation",
            }

            company_name = company_names[ticker]

            query = (
                f"{company_name} SEC 10-K and Knowledge "
                f"Graph financial business risks "
                f"as of {date}. "
                f"Retrieve historical operating risks, "
                f"financial disclosures and company "
                f"relationships. Do not infer causality "
                f"from generic KG relations."
            )

            try:
                context = await rag.aquery(
                    query,
                    param=QueryParam(
                        mode=cfg["mode"],
                        only_need_context=True,
                        enable_rerank=False,
                    ),
                )

                context_text = str(context)

                context_file = (
                    OUT /
                    f"{ticker}_{date}_{group}_context.txt"
                )

                context_file.write_text(
                    context_text,
                    encoding="utf-8",
                )

                retrieved = match_sources(
                    context_text,
                    registry,
                    cfg["allowed"],
                )

                evidence = select_evidence(
                    retrieved
                )

                kg_count = sum(
                    x["type"] == "KG"
                    for x in evidence
                )

                sec_count = sum(
                    x["type"] == "SEC"
                    for x in evidence
                )

                print("[KG TO QWEN]", kg_count)
                print("[SEC TO QWEN]", sec_count)

                # Strict source-isolation invariant.
                assert all(
                    x["type"] in cfg["allowed"]
                    for x in evidence
                )

                if evidence:
                    generated, diagnostics = generate(
                        client,
                        ticker,
                        date,
                        group,
                        evidence,
                    )

                    validation = validate(
                        generated, evidence
                    )
                else:
                    generated = None

                    diagnostics = {
                        "error": "no_retrieved_evidence"
                    }

                    validation = validate(
                        None, []
                    )

                result = {
                    "ticker": ticker,
                    "date": date,
                    "group": group,
                    "retrieval_mode": cfg["mode"],
                    "allowed_sources": sorted(
                        cfg["allowed"]
                    ),
                    "kg_count": kg_count,
                    "sec_count": sec_count,
                    "source_ids": [
                        x["id"]
                        for x in evidence
                    ],
                    "evidence": evidence,
                    "generated": generated,
                    "validation": validation,
                    "diagnostics": diagnostics,
                    "strict_pit_certified": False,
                    "ppo_integrated": False,
                }

                save_json(
                    OUT /
                    f"{ticker}_{date}_{group}.json",
                    result,
                )

                summary_rows.append({
                    "ticker": ticker,
                    "date": date,
                    "group": group,
                    "retrieval_mode": cfg["mode"],
                    "kg_count": kg_count,
                    "sec_count": sec_count,
                    "accepted_claims": len(
                        validation["accepted"]
                    ),
                    "rejected_claims": len(
                        validation["rejected"]
                    ),
                    "impact": validation["impact"],
                    "risk": validation["risk"],
                    "confidence": validation["confidence"],
                    "missing": validation["missing"],
                    "grounding_status": validation["status"],
                    "semantic_verified": False,
                    "ready_for_ppo": False,
                })

                print(
                    "[SIGNALS]",
                    {
                        "impact": validation["impact"],
                        "risk": validation["risk"],
                        "confidence": validation["confidence"],
                    }
                )

                print(
                    "[CLAIMS]",
                    len(validation["accepted"]),
                    "accepted,",
                    len(validation["rejected"]),
                    "rejected",
                )

            except Exception as exc:
                print(
                    "[GROUP ERROR]",
                    group,
                    repr(exc),
                )

                save_json(
                    OUT /
                    f"{ticker}_{date}_{group}_error.json",
                    {
                        "group": group,
                        "error": repr(exc),
                    },
                )

                summary_rows.append({
                    "ticker": ticker,
                    "date": date,
                    "group": group,
                    "retrieval_mode": cfg["mode"],
                    "kg_count": 0,
                    "sec_count": 0,
                    "accepted_claims": 0,
                    "rejected_claims": 0,
                    "impact": None,
                    "risk": None,
                    "confidence": None,
                    "missing": True,
                    "grounding_status": "error",
                    "semantic_verified": False,
                    "ready_for_ppo": False,
                })

    finally:
        await rag.finalize_storages()

    summary = pd.DataFrame(summary_rows)

    output_csv = (
        OUT /
        f"ablation_{ticker}_{date}.csv"
    )

    summary.to_csv(
        output_csv,
        index=False,
    )

    print("\n" + "=" * 72)
    print("FINAL A/B/C ABLATION COMPARISON")
    print("=" * 72)

    print(
        summary.to_string(index=False)
    )

    print("\n[SAVED CSV]", output_csv)
    print("[SAVED JSON DIRECTORY]", OUT)

    print(
        "\n[NOTE] These are candidate RAG signals, "
        "not semantically certified PPO features."
    )


if __name__ == "__main__":
    asyncio.run(main())
