#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V7-A
Financial Memory Reranking + Distraction Filtering

Reuses:
    - V5-Full frozen A2C
    - Existing temporal KG/SEC retriever
    - V6 closed-loop backtest
    - Existing trained V6 source gate

Compares:
    BASELINE: Original retrieval
    V7A: Two-query candidate reranking +
         source-level evidence filtering

No model training.
No future returns used in retrieval scoring.
No retrieval cost charged.

IMPORTANT:
    Query-level reranking, not document-level.
    Heuristic filter, not a trained distractor detector.
    Existing V6 Source Gate is unchanged.
"""

import argparse
import json
import math
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from unified_event_memory.finrl_x.aegis_rl import (
    run_v6_a2c_endtoend as v6,
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_stop_counterfactual_v1 as cf,
)


SOURCES = ("KG", "SEC", "BOTH")

FINANCIAL_KEYWORDS = {
    "revenue",
    "profit",
    "earnings",
    "guidance",
    "risk",
    "loss",
    "demand",
    "supply",
    "supplier",
    "customer",
    "contract",
    "competition",
    "competitive",
    "margin",
    "cash",
    "debt",
    "liquidity",
    "growth",
    "decline",
    "regulation",
    "litigation",
    "manufacturing",
    "inventory",
    "filing",
    "acquisition",
    "bankruptcy",
    "material",
}

COMPANY_TERMS = {
    "AAPL": {"aapl", "apple"},
    "AMZN": {"amzn", "amazon"},
    "GOOGL": {"googl", "google", "alphabet"},
    "MSFT": {"msft", "microsoft"},
    "NVDA": {"nvda", "nvidia"},
}


def log(*args):
    print(*args, flush=True)


def tokenize(value):
    return set(
        re.findall(
            r"[a-z0-9]+",
            str(value).lower(),
        )
    )


def query_variants(ticker):
    """
    Fixed query templates.
    No market outcome information.

    Candidate 0 equals the existing baseline query.
    """

    baseline = cf.candidate_query(ticker)

    alternative = (
        f"{ticker} historical SEC filing "
        "earnings revenue risk factors "
        "material business relationships "
        "supply chain financial exposure"
    )

    return [
        baseline,
        alternative,
    ]


def score_item(
    item,
    ticker,
    decision_date,
):
    """
    FinMem-inspired heuristic scoring.

    Scores:
      recency   : time distance
      relevance : ticker / relation / finance tokens
      importance: financial event vocabulary

    All content must be available before the
    historical decision date.
    """

    decision_date = pd.Timestamp(
        decision_date
    )

    available = pd.Timestamp(
        item["available_at"]
    )

    if available >= decision_date:
        raise RuntimeError(
            f"PIT violation: {available} >= "
            f"{decision_date}"
        )

    age_days = max(
        0.0,
        (
            decision_date - available
        ).total_seconds() / 86400.0,
    )

    # Smooth exponential time decay.
    recency = math.exp(
        -age_days / 365.0
    )

    text = " ".join([
        str(
            item.get("text_excerpt", "")
        ),
        str(
            item.get("relation", "")
        ),
        str(
            item.get("ticker", "")
        ),
    ])

    tokens = tokenize(text)

    company_match = bool(
        tokens & COMPANY_TERMS[ticker]
    )

    importance_hits = (
        tokens & FINANCIAL_KEYWORDS
    )

    importance = min(
        1.0,
        len(importance_hits) / 4.0,
    )

    relation_terms = {
        "supplier",
        "customer",
        "competition",
        "supply",
        "contract",
        "risk",
    }

    relation_match = bool(
        tokens & relation_terms
    )

    relevance = (
        0.7 * float(company_match)
        + 0.3 * float(relation_match)
    )

    # All terms are in [0, 1].
    score = (
        0.35 * recency
        + 0.40 * relevance
        + 0.25 * importance
    )

    return {
        "score": float(score),
        "recency": float(recency),
        "relevance": float(relevance),
        "importance": float(importance),
        "age_days": float(age_days),
    }


def score_source(
    audit,
    ticker,
    date,
    source,
):
    """
    Aggregate evidence score for one source.
    No evidence => score zero.
    """

    relevant = [
        item
        for item in audit
        if item.get("source") == source
    ]

    if not relevant:
        return {
            "score": 0.0,
            "hits": 0,
        }

    scores = [
        score_item(
            item,
            ticker,
            date,
        )["score"]
        for item in relevant
    ]

    # Prefer strong evidence while avoiding
    # unchecked benefit from retrieving many hits.
    scores.sort(reverse=True)

    combined = (
        0.7 * scores[0]
        + 0.3 * float(
            np.mean(scores[:3])
        )
    )

    return {
        "score": float(combined),
        "hits": len(relevant),
    }


class MemoryFilteredRetriever:
    """
    Adapter around existing retrieval function.

    Does not change KG/SEC indices.
    Does not mutate original documents.

    The returned evidence vector always
    corresponds to an actual retrieved query.

    Filtering is applied at source level:
      vector[source] = 0
    """

    def __init__(
        self,
        original,
        threshold,
        output_path,
    ):
        self.original = original
        self.threshold = float(threshold)

        self.output_path = Path(
            output_path
        )

        self.rows = []

    def retrieve(
        self,
        retriever,
        source,
        ticker,
        relation,
        date,
        question,
    ):
        if source == "STOP":
            return self.original(
                retriever,
                source,
                ticker,
                relation,
                date,
                question,
            )

        variants = query_variants(
            ticker
        )

        # Candidate 0 is exactly the
        # previous V6 baseline query.
        if variants[0] != question:
            raise RuntimeError(
                "Baseline query mismatch"
            )

        candidates = []

        for index, candidate_query in enumerate(
            variants
        ):
            vector, provenance = (
                self.original(
                    retriever,
                    source,
                    ticker,
                    relation,
                    date,
                    candidate_query,
                )
            )

            vector = np.asarray(
                vector,
                dtype=np.float32,
            ).copy()

            scores = {
                name: score_source(
                    provenance,
                    ticker,
                    date,
                    name,
                )
                for name in ("KG", "SEC")
            }

            # Evaluate only requested sources.
            if source == "KG":
                rank_score = (
                    scores["KG"]["score"]
                )

            elif source == "SEC":
                rank_score = (
                    scores["SEC"]["score"]
                )

            else:
                # BOTH requires the two sources.
                # Average prevents one source
                # from dominating only through
                # the number of retrieved hits.
                rank_score = (
                    scores["KG"]["score"]
                    + scores["SEC"]["score"]
                ) / 2.0

            candidates.append({
                "index": index,
                "question": candidate_query,
                "vector": vector,
                "provenance": provenance,
                "scores": scores,
                "rank_score": rank_score,
            })

        # Prefer higher heuristic evidence quality.
        # Ties use original baseline candidate.
        best = max(
            candidates,
            key=lambda c: (
                c["rank_score"],
                -c["index"],
            ),
        )

        vector = best["vector"].copy()

        provenance = list(
            best["provenance"]
        )

        allowed_sources = (
            ["KG", "SEC"]
            if source == "BOTH"
            else [source]
        )

        removed = []

        for evidence_source in allowed_sources:
            score = best[
                "scores"
            ][evidence_source]["score"]

            if score < self.threshold:
                idx = (
                    0
                    if evidence_source == "KG"
                    else 1
                )

                vector[idx] = 0.0

                removed.append(
                    evidence_source
                )

        # Do not claim discarded documents
        # contributed to actor evidence.
        provenance = [
            item
            for item in provenance
            if item.get("source")
            not in removed
        ]

        row = {
            "date": cf.date_string(date),
            "requested_source": source,
            "ticker": ticker,
            "chosen_query_index":
                best["index"],
            "chosen_question":
                best["question"],
            "ranking_score":
                float(best["rank_score"]),
            "KG_score": float(
                best["scores"]["KG"]["score"]
            ),
            "SEC_score": float(
                best["scores"]["SEC"]["score"]
            ),
            "removed_KG":
                int("KG" in removed),
            "removed_SEC":
                int("SEC" in removed),
            "retained_hits":
                len(provenance),
            "is_doc_level_rerank": False,
        }

        self.rows.append(row)

        return vector, provenance

    def save(self):
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            self.rows
        ).to_csv(
            self.output_path,
            index=False,
        )


def comparison_table(
    baseline,
    enhanced,
):
    a = baseline.set_index("policy")
    b = enhanced.set_index("policy")

    common = sorted(
        set(a.index) & set(b.index)
    )

    rows = []

    for name in common:
        old = a.loc[name]
        new = b.loc[name]

        rows.append({
            "policy": name,
            "baseline_return":
                float(old["total_return"]),
            "v7a_return":
                float(new["total_return"]),
            "delta_return":
                float(
                    new["total_return"]
                    - old["total_return"]
                ),
            "baseline_sharpe":
                float(old["sharpe"]),
            "v7a_sharpe":
                float(new["sharpe"]),
            "delta_sharpe":
                float(
                    new["sharpe"]
                    - old["sharpe"]
                ),
            "baseline_mdd":
                float(old["max_drawdown"]),
            "v7a_mdd":
                float(new["max_drawdown"]),
            "delta_mdd":
                float(
                    new["max_drawdown"]
                    - old["max_drawdown"]
                ),
        })

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )

    p.add_argument(
        "--max-days",
        type=int,
        default=3,
    )

    p.add_argument(
        "--filter-threshold",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    p.add_argument(
        "--gate",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v6_a2c_full/source_gate.joblib"
        ),
    )

    p.add_argument(
        "--selection",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v6_a2c_full/"
            "validation_selection.json"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v7a_memory_filter"
        ),
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    p.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
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
        "--embed",
        default="bge-m3:latest",
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--cost-scale",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--tier",
        default="conservative",
    )

    p.add_argument(
        "--algo",
        choices=["A2C"],
        default="A2C",
    )

    args = p.parse_args()

    if args.cost_scale != 0.0:
        raise RuntimeError(
            "Use --cost-scale 0 for this experiment"
        )

    if not (
        0.0 <= args.filter_threshold <= 1.0
    ):
        raise RuntimeError(
            "Filter threshold must be in [0, 1]"
        )

    if not args.model.is_file():
        raise RuntimeError(
            f"Missing checkpoint: {args.model}"
        )

    if not args.gate.is_file():
        raise RuntimeError(
            f"Missing trained gate: {args.gate}"
        )

    if not args.selection.is_file():
        raise RuntimeError(
            f"Missing validation selection: "
            f"{args.selection}"
        )

    args.out = args.out.resolve()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection = json.loads(
        args.selection.read_text(
            encoding="utf-8"
        )
    )

    threshold = float(
        selection["selected_threshold"]
    )

    gate = joblib.load(
        args.gate
    )

    log("=" * 76)
    log("FINRL-X V7-A MEMORY + FILTER")
    log("=" * 76)

    log("[SPLIT]", args.split)
    log("[FROZEN A2C]", args.model)
    log("[FILTER THRESHOLD]", args.filter_threshold)
    log("[V6 GATE THRESHOLD]", threshold)

    # Confirm the baseline retrieval reference.
    original = v6.historical_retrieve

    # Phase A: baseline.
    log()
    log("===== ORIGINAL RAG =====")

    args.split = args.split

    baseline_env, baseline_actor, baseline_ret = (
        cf.load_system(args)
    )

    baseline_history, baseline_summary = (
        v6.backtest(
            baseline_env,
            baseline_actor,
            baseline_ret,
            gate,
            [threshold],
            args.split,
            args.max_days,
            args.out / "baseline",
        )
    )

    # Phase B: enhanced retrieval.
    log()
    log("===== V7-A MEMORY FILTER =====")

    adapter = MemoryFilteredRetriever(
        original,
        args.filter_threshold,
        args.out / "retrieval_audit.csv",
    )

    v6.historical_retrieve = adapter.retrieve

    try:
        enhanced_env, enhanced_actor, enhanced_ret = (
            cf.load_system(args)
        )

        enhanced_history, enhanced_summary = (
            v6.backtest(
                enhanced_env,
                enhanced_actor,
                enhanced_ret,
                gate,
                [threshold],
                args.split,
                args.max_days,
                args.out / "v7a",
            )
        )

    finally:
        v6.historical_retrieve = original

        adapter.save()

    table = comparison_table(
        baseline_summary,
        enhanced_summary,
    )

    table.to_csv(
        args.out / "comparison.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("V7-A vs BASELINE")
    print("=" * 76)
    print(
        table.to_string(index=False),
        flush=True,
    )

    if not np.isfinite(
        table.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all():
        raise RuntimeError(
            "Nonfinite experimental result"
        )

    stop = table[
        table["policy"] == "STOP"
    ]

    if not stop.empty:
        stop_delta = float(
            stop.iloc[0]["delta_return"]
        )

        if abs(stop_delta) > 1e-10:
            raise RuntimeError(
                "STOP parity failed: "
                f"{stop_delta}"
            )

    log()
    log("[SAVED]", args.out / "comparison.csv")
    log(
        "[SAVED]",
        args.out / "retrieval_audit.csv",
    )
    log("V7-A EXPERIMENT: PASS")


if __name__ == "__main__":
    main()
