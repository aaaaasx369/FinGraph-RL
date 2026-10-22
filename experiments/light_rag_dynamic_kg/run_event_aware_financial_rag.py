#!/usr/bin/env python3
"""
Event-Aware Financial RAG v1

Inspired by:
- FinMem: short/long-term memory
- CFinRAG: event-centric evidence and generation verification

Uses existing Unified Temporal KG V3.
No modification of original LightRAG or PPO.
Default = retrieval comparison only.
--generate = Qwen structured event analysis.

Research status: exploratory, PIT source timestamps not certified.
"""

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KG_PATH = ROOT / (
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)
MAP_PATH = ROOT / (
    "outputs/unified_temporal_kg_v3/"
    "entity_mapping_report.csv"
)
ENTITY_MAP_PATH = ROOT / (
    "outputs/unified_temporal_kg_v3/entity2id.json"
)
OUT_ROOT = ROOT / (
    "results/light_rag_dynamic_kg/"
    "event_aware_financial_rag_v1"
)

TICKERS = ("AAPL", "AMZN", "GOOGL", "MSFT", "NVDA")

GENERIC_RELATIONS = {
    "RELATE_TO", "RELATES_TO", "RELATED_TO",
    "HAS", "IS", "ASSOCIATED_WITH", "LINKED_TO",
}

STATIC_RELATIONS = {
    "IS_MEMBER_OF", "MEMBER_OF",
    "OPERATE_IN", "OPERATES_IN",
    "LOCATED_IN", "BELONGS_TO",
}

# Event tags are search priorities, NOT verified financial impacts.
FINANCIAL_TERMS = {
    "ACQUIRE", "ACQUISITION", "MERGER",
    "INVEST", "SUPPLY", "CUSTOMER",
    "COMPETE", "LITIGATION", "LAWSUIT",
    "REGULATORY", "FINANCING", "DEBT",
    "EARNINGS", "REVENUE", "BANKRUPTCY",
    "CONTRACT", "PRODUCTION",
}


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {path}"
        )
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def fingerprint(obj):
    raw = json.dumps(
        obj, sort_keys=True, ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def normalize_relation(relation):
    return (
        str(relation).upper()
        .strip()
        .replace("-", "_")
        .replace(" ", "_")
    )


def is_generic(relation):
    return normalize_relation(relation) in GENERIC_RELATIONS


def is_static(relation):
    return normalize_relation(relation) in STATIC_RELATIONS


def materiality_hint(relation, target):
    """
    Only a retrieval PRIORITY heuristic.
    Not a verified material event or impact label.
    """
    content = (
        normalize_relation(relation) + " "
        + str(target).upper()
    )
    return int(any(term in content for term in FINANCIAL_TERMS))


def load_mapping(ticker):
    report = pd.read_csv(MAP_PATH)

    row = report.loc[
        report["ticker"].astype(str).str.upper().eq(ticker)
    ]

    if len(row) != 1:
        raise RuntimeError("Company mapping missing or ambiguous")

    r = row.iloc[0]
    verified = str(r["verified"]).strip().lower() in (
        "true", "1",
    )

    if not verified:
        raise RuntimeError("Company mapping not verified")

    canonical = f"COMPANY:{ticker}"

    if str(r["canonical_id"]) != canonical:
        raise RuntimeError("Canonical company mapping mismatch")

    entity2id = json.loads(ENTITY_MAP_PATH.read_text())

    if canonical not in entity2id:
        raise RuntimeError("Company absent from Unified KG")

    return canonical


def load_relation_labels():
    """
    Reuse the existing official LightRAG decoder.
    Avoid independently inventing FINDKG relation meanings.
    """
    from experiments.light_rag_dynamic_kg import (
        run_official_lightrag as official,
    )
    return official.load_relation_names()


def load_entity_labels():
    from experiments.light_rag_dynamic_kg import (
        run_official_lightrag as official,
    )
    return (
        official.load_entity_names(),
        official.load_company_mapping(),
        official.decode_entity,
    )


def safe_label(value, names, companies, decode):
    try:
        return str(decode(value, names, companies))
    except Exception:
        return str(value)


def load_evidence(ticker, decision_date, source):
    canonical = load_mapping(ticker)

    relation_labels = load_relation_labels()
    entity_names, companies, decoder = load_entity_labels()

    df = pd.read_csv(
        KG_PATH,
        usecols=[
            "event_id", "head", "relation", "tail",
            "available_at", "source", "time_precision",
        ],
        dtype=str,
        low_memory=False,
    )

    available = pd.to_datetime(
        df["available_at"],
        utc=True, errors="coerce",
    ).dt.tz_localize(None).dt.normalize()

    # Strictly before decision date.
    eligible = (
        available.notna()
        & (available < decision_date)
        & (
            df["head"].eq(canonical)
            | df["tail"].eq(canonical)
        )
    )

    if source != "ALL":
        eligible &= df["source"].eq(source)

    selected = df.loc[eligible].copy()
    selected["_at"] = available.loc[selected.index]

    # All eligible evidence, not merely the newest 80.
    selected = selected.sort_values(
        ["_at", "event_id"],
        ascending=[True, True],
        kind="stable",
    )

    records = []

    for row in selected.itertuples(index=False):
        raw_rel = str(row.relation)
        readable_relation = str(
            relation_labels.get(raw_rel, raw_rel)
        )

        raw_head = str(row.head)
        raw_tail = str(row.tail)

        head = safe_label(
            raw_head, entity_names, companies, decoder
        )
        tail = safe_label(
            raw_tail, entity_names, companies, decoder
        )

        records.append({
            "event_id": str(row.event_id),
            "head_id": raw_head,
            "relation_id": raw_rel,
            "tail_id": raw_tail,
            "head": head,
            "relation": readable_relation,
            "tail": tail,
            "available_at": pd.Timestamp(
                row._at if hasattr(row, "_at")
                else row.available_at
            ).strftime("%Y-%m-%d"),
            "source": str(row.source),
            "time_precision": str(row.time_precision),
        })

    # itertuples may rename underscore columns.
    for rec, date in zip(records, selected["_at"]):
        rec["available_at"] = pd.Timestamp(
            date
        ).strftime("%Y-%m-%d")

    return records


def build_unique_memory(records, decision_date, recent_days):
    """
    First-observed relationship in the available historical
    dataset. Does NOT assert real-world occurrence date.
    """
    first_seen = {}

    for r in records:
        key = (
            r["head_id"],
            r["relation_id"],
            r["tail_id"],
        )

        if key not in first_seen:
            first_seen[key] = dict(r)

    recent = []
    long_term = []

    for r in first_seen.values():
        at = pd.Timestamp(r["available_at"])
        age = (decision_date - at).days
        r["age_days"] = int(age)
        r["first_observed_in_dataset"] = True

        if age <= recent_days:
            r["memory_layer"] = "SHORT_TERM"
            recent.append(r)
        else:
            r["memory_layer"] = "LONG_TERM"
            long_term.append(r)

    return recent, long_term


def evidence_score(r, recent_days):
    """
    Deterministic ranking, no future returns.

    This is a research heuristic, not a learned
    financial materiality estimate.
    """
    age = r["age_days"]
    relation = r["relation"]

    freshness = np.exp(-age / 30.0)
    recent = int(age <= recent_days)
    specific = int(
        not is_generic(relation)
        and not is_static(relation)
    )
    finance_hint = materiality_hint(
        relation, r["tail"]
    )

    # Prefer documented relation specificity and recency.
    # Penalize generic static metadata.
    score = (
        3.0 * recent
        + 1.2 * freshness
        + 1.5 * specific
        + 0.8 * finance_hint
        - 2.5 * int(is_generic(relation))
        - 1.0 * int(is_static(relation))
    )

    r["retrieval_score"] = round(float(score), 6)
    r["financial_relevance_hint"] = bool(finance_hint)
    r["economically_verified"] = False

    return r


def rank_with_diversity(records):
    """
    Stable greedy diversity: limit one relation family
    from dominating early selections, while allowing
    fallback if limited groups cannot fill the budget.
    """
    ranked = sorted(
        records,
        key=lambda r: (
            -r["retrieval_score"],
            r["available_at"],
            r["event_id"],
        ),
    )

    output = []
    deferred = []
    counts = {}

    for r in ranked:
        rel = r["relation_id"]
        if counts.get(rel, 0) < 8:
            output.append(r)
            counts[rel] = counts.get(rel, 0) + 1
        else:
            deferred.append(r)

    return output + deferred


def select_event_aware(records, decision_date,
                       top_k, recent_days):
    recent, long_term = build_unique_memory(
        records, decision_date, recent_days
    )

    for r in recent + long_term:
        evidence_score(r, recent_days)

    recent = rank_with_diversity(recent)
    long_term = rank_with_diversity(long_term)

    # FinMem-inspired layered retrieval:
    # 75% recent updates, 25% historical background,
    # with spillover if one layer is undersupplied.
    recent_budget = int(round(top_k * 0.75))
    background_budget = top_k - recent_budget

    selected = (
        recent[:recent_budget]
        + long_term[:background_budget]
    )

    used = {r["event_id"] for r in selected}

    remainder = rank_with_diversity([
        r for r in recent + long_term
        if r["event_id"] not in used
    ])

    selected.extend(
        remainder[:max(0, top_k - len(selected))]
    )

    return selected[:top_k], {
        "unique_recent_relations": len(recent),
        "unique_historical_relations": len(long_term),
    }


def select_recent_baseline(records, top_k):
    recent_first = sorted(
        records,
        key=lambda r: (
            r["available_at"], r["event_id"]
        ),
        reverse=True,
    )

    seen = set()
    out = []

    for r in recent_first:
        key = (
            r["head_id"],
            r["relation_id"],
            r["tail_id"],
        )

        if key in seen:
            continue

        seen.add(key)
        out.append(r)

        if len(out) >= top_k:
            break

    return out


def summarize_selection(records):
    return {
        "count": len(records),
        "generic_count": sum(
            is_generic(r["relation"]) for r in records
        ),
        "static_count": sum(
            is_static(r["relation"]) for r in records
        ),
        "unique_relation_types": len({
            r["relation_id"] for r in records
        }),
        "sources": {
            s: sum(r["source"] == s for r in records)
            for s in sorted({r["source"] for r in records})
        },
        "first_10_event_ids": [
            r["event_id"] for r in records[:10]
        ],
    }


def call_qwen(payload, model, host, timeout):
    """
    Local/connected Ollama chat endpoint.
    Not the LightRAG library; this is the new
    event-aware evidence-to-analysis channel.
    """
    url = host.rstrip("/") + "/api/chat"

    instructions = """
You are a historical financial evidence analyst.

Produce ONLY valid JSON, no markdown.

Use only the supplied event records.
Do not use modern facts, external knowledge, or future prices.

The KG relations may be generic metadata and are not
proof of financial impact or real-world event timing.

For each supported insight:
- Cite actual event_id values from the supplied evidence.
- Identify what was newly observed versus background.
- Specify a plausible economic mechanism ONLY if evidence
  actually supports that inference.
- State limitations and uncertainty explicitly.

Relate_To / Has / generic category membership must NOT be
treated as evidence of causation, active litigation,
contractual relations, or specific financial damage.

Never invent news, earnings values, supply relationships,
regulatory decisions, changes in prices, or returns.

If no materially supported changes exist, say so.

Output schema:
{
 "ticker": "string",
 "decision_date": "YYYY-MM-DD",
 "events": [
  {
   "description": "string",
   "event_ids": ["existing id"],
   "memory_layer": "SHORT_TERM or LONG_TERM",
   "economic_channel": "unknown or specific supported channel",
   "direction": "positive, negative, mixed, or unknown",
   "evidence_strength": "structural_only or textual_verified",
   "limitations": "string"
  }
 ],
 "summary": "string",
 "impact": null,
 "risk": null,
 "confidence": null
}

For this KG-only pilot, all numerical impact/risk/confidence
must remain null because no textual financial impact
has been independently verified.
"""

    req = urllib.request.Request(
        url,
        data=json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": json.dumps(
                        payload, ensure_ascii=False
                    ),
                },
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
            },
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(
        req, timeout=timeout
    ) as response:
        envelope = json.loads(
            response.read().decode("utf-8")
        )

    content = envelope.get("message", {}).get(
        "content", ""
    )

    return json.loads(content)


def validate_analysis(answer, allowed, ticker, date):
    """
    Fail-closed:
    Every listed claim must cite an actual selected event.
    Numeric candidate signals are NOT certified here.
    """
    if not isinstance(answer, dict):
        raise RuntimeError("LLM output is not an object")

    if answer.get("ticker") != ticker:
        raise RuntimeError("Wrong output ticker")

    if answer.get("decision_date") != date:
        raise RuntimeError("Wrong decision date")

    events = answer.get("events")

    if not isinstance(events, list):
        raise RuntimeError("events must be a list")

    valid_ids = {
        r["event_id"]: r for r in allowed
    }

    checked = []
    rejected = []

    for event in events:
        if not isinstance(event, dict):
            rejected.append({
                "reason": "not_object",
                "raw": str(event)[:200],
            })
            continue

        refs = event.get("event_ids", [])

        if (
            not isinstance(refs, list)
            or len(refs) == 0
            or any(
                not isinstance(i, str)
                or i not in valid_ids
                for i in refs
            )
        ):
            rejected.append({
                "reason": "invalid_event_ids",
                "event": event,
            })
            continue

        layers = {
            valid_ids[i]["memory_layer"]
            for i in refs
        }

        if event.get("memory_layer") not in layers:
            rejected.append({
                "reason": "incorrect_memory_layer",
                "event": event,
            })
            continue

        # Event-ID matching is only citation validation,
        # not automatic factual entailment.
        event["citation_ids_valid"] = True
        event["semantic_grounding_certified"] = False
        event["economic_impact_certified"] = False

        checked.append(event)

    return {
        "ticker": ticker,
        "decision_date": date,
        "events_with_valid_citations": checked,
        "rejected_events": rejected,
        "valid_citation_count": len(checked),
        "semantic_grounding_certified": False,
        "numeric_signals_valid": False,
        "impact": None,
        "risk": None,
        "confidence": None,
        "summary_unverified": answer.get("summary", ""),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--ticker", choices=TICKERS, default="AAPL"
    )
    p.add_argument(
        "--date", default="2018-08-06"
    )
    p.add_argument(
        "--source",
        choices=["FINDKG", "ALL"],
        default="FINDKG",
    )
    p.add_argument("--top-k", type=int, default=80)
    p.add_argument("--recent-days", type=int, default=30)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--llm-evidence-k", type=int, default=24)
    p.add_argument("--model", default="qwen3.5:27b")
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument("--timeout", type=int, default=600)

    args = p.parse_args()

    if args.top_k < 1 or args.recent_days < 1:
        p.error("top-k and recent-days must be positive")

    decision_date = pd.Timestamp(
        args.date
    ).normalize()

    if not (
        pd.Timestamp("2018-01-01")
        <= decision_date
        <= pd.Timestamp("2021-12-31")
    ):
        raise RuntimeError("Only 2018-2021 allowed")

    records = load_evidence(
        args.ticker, decision_date, args.source
    )

    event_aware, layers = select_event_aware(
        records, decision_date,
        args.top_k, args.recent_days,
    )

    baseline = select_recent_baseline(
        records, args.top_k
    )

    tag = (
        f"{args.ticker}_{args.date}"
        f"_{args.source}_K{args.top_k}"
        f"_R{args.recent_days}"
    )

    result = {
        "ticker": args.ticker,
        "decision_date": args.date,
        "source": args.source,
        "version": "event_aware_v1",
        "strict_pit_certified": False,
        "economic_grounding_certified": False,
        "lookahead_filter": "available_at < decision_date",
        "available_records": len(records),
        "memory": layers,
        "recent80_summary": summarize_selection(baseline),
        "event_aware_summary": summarize_selection(event_aware),
        "baseline_event_ids": [
            r["event_id"] for r in baseline
        ],
        "event_aware_records": event_aware,
    }

    result["fingerprint"] = fingerprint(result)

    out = OUT_ROOT / (
        tag + "_" + result["fingerprint"]
    )

    output = out / "retrieval_comparison.json"

    print("=" * 75)
    print("EVENT-AWARE FINANCIAL RAG")
    print("=" * 75)
    print("Ticker:", args.ticker)
    print("Decision:", args.date)
    print("Source:", args.source)
    print("All eligible:", len(records))
    print("Memory:", layers)
    print(
        "Recent-80:",
        json.dumps(
            result["recent80_summary"],
            ensure_ascii=False,
        ),
    )
    print(
        "Event-Aware-80:",
        json.dumps(
            result["event_aware_summary"],
            ensure_ascii=False,
        ),
    )

    write_new(output, result)

    print("\n[SAVED]", output)

    print("\n========== TOP EVENT-AWARE EVIDENCE ==========")

    for r in event_aware[:15]:
        print(
            r["available_at"],
            r["memory_layer"],
            f"score={r['retrieval_score']:.3f}",
            r["relation"],
            r["head"][:36],
            "->",
            r["tail"][:48],
            f"[{r['event_id']}]",
        )

    if not args.generate:
        print("\nRETRIEVAL COMPARISON COMPLETE")
        print("No LLM called. No original files modified.")
        return

    # For the LLM, prefer selected event evidence
    # but use a bounded context to prevent truncation.
    llm_records = event_aware[:args.llm_evidence_k]

    payload = {
        "ticker": args.ticker,
        "decision_date": args.date,
        "historical_timestamp_note": (
            "FINDKG weekly availability is not "
            "independently publication-time certified."
        ),
        "short_term_memory_days": args.recent_days,
        "evidence": llm_records,
    }

    print("\n[GENERATE] Calling Qwen...", flush=True)

    raw = call_qwen(
        payload, args.model,
        args.ollama_host, args.timeout,
    )

    checked = validate_analysis(
        raw, llm_records,
        args.ticker, args.date,
    )

    checked["source_fingerprint"] = result["fingerprint"]
    checked["llm_model"] = args.model
    checked["not_ppo_integrated"] = True

    write_new(out / "qwen_raw.json", raw)
    write_new(out / "qwen_checked.json", checked)

    print("\n========== CHECKED ANALYSIS ==========")
    print(json.dumps(
        checked, ensure_ascii=False, indent=2
    ))

    print("\nFINISHED")
    print("Checked citations only; semantics remain unaudited.")


if __name__ == "__main__":
    main()
