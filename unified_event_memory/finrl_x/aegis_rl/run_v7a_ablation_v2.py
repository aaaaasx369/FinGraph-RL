#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V7-A Ablation v2

Separate:
  1. Original retrieval
  2. Query reranking only
  3. Evidence filtering only
  4. Query reranking + filtering

Frozen V5-Full A2C.
Original V6 source gate.
Zero retrieval cost.
2021 validation only.

Uses in-process retrieval caching.
No test data used in model selection.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from unified_event_memory.finrl_x.aegis_rl import (
    run_v6_a2c_endtoend as v6,
    run_stop_counterfactual_v1 as cf,
    run_v7a_memory_filter as v7,
)


class RetrievalCache:
    def __init__(self, original):
        self.original = original
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get(
        self,
        retriever,
        source,
        ticker,
        relation,
        date,
        question,
    ):
        key = (
            source,
            ticker,
            relation,
            str(pd.Timestamp(date)),
            question,
        )

        if key not in self.cache:
            vector, audit = self.original(
                retriever,
                source,
                ticker,
                relation,
                date,
                question,
            )

            self.cache[key] = (
                np.asarray(
                    vector,
                    dtype=np.float32,
                ).copy(),
                list(audit),
            )

            self.misses += 1

        else:
            self.hits += 1

        vector, audit = self.cache[key]

        return (
            vector.copy(),
            list(audit),
        )


class AblationRetriever:
    def __init__(
        self,
        cache,
        use_rerank,
        use_filter,
        threshold,
    ):
        self.cache = cache
        self.use_rerank = use_rerank
        self.use_filter = use_filter
        self.threshold = float(threshold)
        self.records = []

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
            return self.cache.get(
                retriever,
                source,
                ticker,
                relation,
                date,
                question,
            )

        queries = (
            v7.query_variants(ticker)
            if self.use_rerank
            else [question]
        )

        if queries[0] != question:
            raise RuntimeError(
                "Baseline query mismatch"
            )

        candidates = []

        for idx, query in enumerate(queries):
            vector, audit = self.cache.get(
                retriever,
                source,
                ticker,
                relation,
                date,
                query,
            )

            scores = {
                s: v7.score_source(
                    audit,
                    ticker,
                    date,
                    s,
                )
                for s in ("KG", "SEC")
            }

            if source == "KG":
                rank = scores["KG"]["score"]

            elif source == "SEC":
                rank = scores["SEC"]["score"]

            else:
                rank = (
                    scores["KG"]["score"]
                    + scores["SEC"]["score"]
                ) / 2.0

            candidates.append({
                "index": idx,
                "vector": vector,
                "audit": audit,
                "scores": scores,
                "rank": float(rank),
            })

        best = max(
            candidates,
            key=lambda c: (
                c["rank"],
                -c["index"],
            ),
        )

        vector = best["vector"].copy()
        audit = list(best["audit"])

        masked = []

        requested = (
            ["KG", "SEC"]
            if source == "BOTH"
            else [source]
        )

        if self.use_filter:
            for s in requested:
                score = best[
                    "scores"
                ][s]["score"]

                if score < self.threshold:
                    idx = 0 if s == "KG" else 1
                    vector[idx] = 0.0
                    masked.append(s)

            audit = [
                item
                for item in audit
                if item.get("source")
                not in masked
            ]

        # Every surviving provenance record
        # corresponds to the selected candidate.
        self.records.append({
            "date": cf.date_string(date),
            "requested_source": source,
            "chosen_query_index": best["index"],
            "ranking_score": best["rank"],
            "KG_score":
                best["scores"]["KG"]["score"],
            "SEC_score":
                best["scores"]["SEC"]["score"],
            "removed_KG": int("KG" in masked),
            "removed_SEC": int("SEC" in masked),
            "n_remaining_hits": len(audit),
            "KG_vector_norm": float(
                np.linalg.norm(vector[0])
            ),
            "SEC_vector_norm": float(
                np.linalg.norm(vector[1])
            ),
        })

        return vector, audit


def run_mode(
    args,
    gate,
    threshold_gate,
    cache,
    name,
    rerank,
    filtering,
    filter_threshold,
):
    print()
    print("=" * 75, flush=True)
    print("[ABLATION]", name, flush=True)
    print("=" * 75, flush=True)

    adapter = AblationRetriever(
        cache=cache,
        use_rerank=rerank,
        use_filter=filtering,
        threshold=filter_threshold,
    )

    original = v6.historical_retrieve

    v6.historical_retrieve = adapter.retrieve

    try:
        env, actor, ret = cf.load_system(args)

        _, summary = v6.backtest(
            env,
            actor,
            ret,
            gate,
            [threshold_gate],
            "val",
            args.max_days,
            args.out / name,
        )

    finally:
        v6.historical_retrieve = original

    audit = pd.DataFrame(adapter.records)

    audit.to_csv(
        args.out / name / "retrieval_audit.csv",
        index=False,
    )

    summary = summary.copy()
    summary["experiment"] = name
    summary["rerank"] = rerank
    summary["filtering"] = filtering
    summary["filter_threshold"] = (
        filter_threshold if filtering else 0.0
    )

    return summary, audit


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--max-days",
        type=int,
        default=3,
    )

    p.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.45, 0.55, 0.60],
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v7a_ablation_v2"
        ),
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

    p.add_argument("--device", default="cpu")
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
            "rag/data/processed/sec/sec_chunks.jsonl"
        ),
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--embed", default="bge-m3:latest")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )
    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--algo", default="A2C")
    p.add_argument("--split", default="val")

    args = p.parse_args()

    if args.split != "val":
        raise RuntimeError(
            "This ablation is validation-only."
        )

    if args.cost_scale != 0:
        raise RuntimeError(
            "Zero-cost protocol required"
        )

    if not args.thresholds:
        raise RuntimeError("No thresholds")

    if any(
        not 0 <= x <= 1
        for x in args.thresholds
    ):
        raise RuntimeError(
            "Threshold must be within [0,1]"
        )

    for path in (
        args.model,
        args.gate,
        args.selection,
    ):
        if not path.is_file():
            raise RuntimeError(
                f"Missing file: {path}"
            )

    args.out = args.out.resolve()
    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate = joblib.load(args.gate)

    selection = json.loads(
        args.selection.read_text(
            encoding="utf-8"
        )
    )

    threshold_gate = float(
        selection["selected_threshold"]
    )

    original_retrieval = (
        v6.historical_retrieve
    )

    cache = RetrievalCache(
        original_retrieval
    )

    experiments = [
        ("BASELINE", False, False, 0.0),
        ("RERANK_ONLY", True, False, 0.0),
    ]

    for threshold in args.thresholds:
        tag = str(threshold).replace(".", "p")

        experiments.append((
            f"FILTER_ONLY_{tag}",
            False,
            True,
            threshold,
        ))

        experiments.append((
            f"RERANK_FILTER_{tag}",
            True,
            True,
            threshold,
        ))

    summaries = []
    audits = []

    for name, rerank, filtering, threshold in experiments:
        summary, audit = run_mode(
            args,
            gate,
            threshold_gate,
            cache,
            name,
            rerank,
            filtering,
            threshold,
        )

        summaries.append(summary)

        if len(audit):
            audit["experiment"] = name
            audits.append(audit)

        pd.concat(
            summaries,
            ignore_index=True,
        ).to_csv(
            args.out / "partial_summary.csv",
            index=False,
        )

        print(
            "[CACHE] misses:",
            cache.misses,
            "hits:",
            cache.hits,
            flush=True,
        )

    all_summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    all_audits = pd.concat(
        audits,
        ignore_index=True,
    )

    all_summary.to_csv(
        args.out / "all_portfolio_summary.csv",
        index=False,
    )

    all_audits.to_csv(
        args.out / "all_retrieval_audit.csv",
        index=False,
    )

    baseline = all_summary[
        all_summary.experiment == "BASELINE"
    ].set_index("policy")

    rows = []

    for name in all_summary.experiment.unique():
        one = all_summary[
            all_summary.experiment == name
        ].set_index("policy")

        audit = all_audits[
            all_audits.experiment == name
        ]

        for policy in baseline.index:
            b = baseline.loc[policy]
            x = one.loc[policy]

            rows.append({
                "experiment": name,
                "policy": policy,
                "baseline_return": float(
                    b.total_return
                ),
                "experiment_return": float(
                    x.total_return
                ),
                "delta_return_pp": float(
                    (x.total_return - b.total_return)
                    * 100.0
                ),
                "delta_sharpe": float(
                    x.sharpe - b.sharpe
                ),
                "delta_mdd": float(
                    x.max_drawdown - b.max_drawdown
                ),
                "KG_suppressed": int(
                    audit.removed_KG.sum()
                ),
                "SEC_suppressed": int(
                    audit.removed_SEC.sum()
                ),
                "alternative_queries": int(
                    (audit.chosen_query_index == 1).sum()
                ),
            })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        args.out / "ablation_comparison.csv",
        index=False,
    )

    stop = comparison[
        comparison.policy == "STOP"
    ]

    if not np.allclose(
        stop.delta_return_pp,
        0.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "STOP parity failed"
        )

    print()
    print("=" * 90)
    print("V7-A ABLATION COMPARISON")
    print("=" * 90)

    print(
        comparison[
            comparison.policy.isin([
                "KG",
                "SEC",
                "BOTH",
                "V6_THRESHOLD_0",
            ])
        ][[
            "experiment",
            "policy",
            "delta_return_pp",
            "delta_sharpe",
            "KG_suppressed",
            "SEC_suppressed",
            "alternative_queries",
        ]].to_string(index=False),
        flush=True,
    )

    print()
    print(
        "[SAVED]",
        args.out / "ablation_comparison.csv",
    )

    print("V7-A ABLATION V2: PASS")


if __name__ == "__main__":
    main()
