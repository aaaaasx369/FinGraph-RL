#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_unified_aegis_v2 as base

OUT = (
    base.ROOT
    / "results/aegis_rl_v1/unified_kg_rag_v3"
)

OUT.mkdir(parents=True, exist_ok=True)


def percentile_rank(values, reference):
    """
    Use validation-period reference only.
    No test-period recalibration.
    """
    ref = np.sort(
        np.asarray(reference, dtype=float)
    )

    return (
        np.searchsorted(
            ref,
            np.asarray(values, dtype=float),
            side="right",
        )
        / len(ref)
    )


def add_state_scores(df):

    df = df.copy()

    validation = df[
        df["date"].dt.year == 2021
    ]

    if validation.empty:
        raise RuntimeError(
            "2021 validation rows are required"
        )

    # KG information staleness and structural sparsity
    # Larger score = higher retrieval priority proxy.
    #
    # This is NOT a ground-truth knowledge-gap label.

    kg_age = percentile_rank(
        df["kg_age_days"],
        validation["kg_age_days"],
    )

    kg_sparse = 1.0 - percentile_rank(
        df["kg_count_365d"],
        validation["kg_count_365d"],
    )

    kg_relation_sparse = 1.0 - percentile_rank(
        df["kg_relation_diversity"],
        validation["kg_relation_diversity"],
    )

    # Embedding movement is treated as a change/novelty
    # indicator, not proof of missing information.
    dynamic_change = percentile_rank(
        df["embedding_change"],
        validation["embedding_change"],
    )

    sec_age = percentile_rank(
        df["sec_age_days"],
        validation["sec_age_days"],
    )

    sec_sparse = 1.0 - percentile_rank(
        df["sec_count_365d"],
        validation["sec_count_365d"],
    )

    df["kg_priority"] = (
        0.30 * kg_age
        + 0.25 * kg_sparse
        + 0.20 * kg_relation_sparse
        + 0.25 * dynamic_change
    )

    df["sec_priority"] = (
        0.65 * sec_age
        + 0.35 * sec_sparse
    )

    return df


def calibrate_thresholds(df, query_fraction):

    val = df[
        df["date"].dt.year == 2021
    ]

    # Separate, validation-only calibration.
    # query_fraction is a policy budget hyperparameter.
    kg_threshold = float(
        val["kg_priority"].quantile(
            1.0 - query_fraction
        )
    )

    sec_threshold = float(
        val["sec_priority"].quantile(
            1.0 - query_fraction
        )
    )

    return kg_threshold, sec_threshold


def select_action(
    row,
    kg_threshold,
    sec_threshold,
):

    need_kg = (
        row.kg_priority > kg_threshold
    )

    need_sec = (
        row.sec_priority > sec_threshold
    )

    if need_kg and need_sec:
        return "BOTH"

    if need_kg:
        return "KG"

    if need_sec:
        return "SEC"

    return "SKIP"


def build_raw_states(
    embeddings,
    kg_index,
    sec_index,
    embedding_cols,
):

    records = []

    for row in embeddings.itertuples(index=False):

        state = base.company_state(
            kg_index[row.tic],
            sec_index[row.tic],
            row.date,
            row.embedding_change,
        )

        record = {
            "date": row.date,
            "tic": row.tic,
            **state,
        }

        for col in embedding_cols:
            record[col] = getattr(row, col)

        records.append(record)

    return pd.DataFrame(records)


def run_retrieval(
    states,
    kg_index,
    sec_index,
    manifest,
    top_k,
):

    records = []
    audit = []

    for row in states.itertuples(index=False):

        kg_candidates = []
        sec_candidates = []

        if row.action in ("KG", "BOTH"):
            kg_candidates = base.kg_candidates(
                kg_index[row.tic],
                row.tic,
                row.date,
            )

        if row.action in ("SEC", "BOTH"):
            sec_candidates = base.sec_candidates(
                sec_index[row.tic],
                row.tic,
                row.date,
            )

        # Retain source-specific candidates.
        # For BOTH, reserve at least one candidate
        # from each available source.
        if row.action == "BOTH":

            chosen = []

            if kg_candidates:
                chosen.append(kg_candidates[0])

            if sec_candidates:
                chosen.append(sec_candidates[0])

            used = {
                (x["source"], x["id"])
                for x in chosen
            }

            remaining = sorted(
                kg_candidates + sec_candidates,
                key=lambda x: -x["relevance"],
            )

            for item in remaining:

                key = (item["source"], item["id"])

                if key in used:
                    continue

                chosen.append(item)
                used.add(key)

                if len(chosen) >= top_k:
                    break

        else:
            chosen = sorted(
                kg_candidates + sec_candidates,
                key=lambda x: -x["relevance"],
            )[:top_k]

        chosen = chosen[:top_k]

        for rank, ev in enumerate(chosen, 1):

            assert ev["date"] < row.date, (
                "Future evidence detected"
            )

            audit.append({
                "date": row.date,
                "ticker": row.tic,
                "action": row.action,
                "rank": rank,
                "source": ev["source"],
                "evidence_id": ev["id"],
                "evidence_date": ev["date"],
                "relevance": ev["relevance"],
            })

        verified = sum(
            base.evaluate_defense(
                ev, manifest
            )
            for ev in chosen
        )

        poison1 = base.defense_test(
            chosen,
            manifest,
            attack_count=1,
        )

        poison2 = base.defense_test(
            chosen,
            manifest,
            attack_count=2,
        )

        poison3 = base.defense_test(
            chosen,
            manifest,
            attack_count=3,
        )

        records.append({
            "date": row.date,
            "tic": row.tic,
            "retrieved_count": len(chosen),
            "retrieved_kg": sum(
                x["source"] == "KG"
                for x in chosen
            ),
            "retrieved_sec": sum(
                x["source"] == "SEC"
                for x in chosen
            ),
            "verified_count": verified,
            "fallback_top1": poison1[
                "fallback_success"
            ],
            "fallback_top2": poison2[
                "fallback_success"
            ],
            "fallback_top3": poison3[
                "fallback_success"
            ],
            "detected_top1": poison1[
                "detected"
            ],
        })

    retrieval = pd.DataFrame(records)

    audit_columns = [
        "date", "ticker", "action", "rank",
        "source", "evidence_id",
        "evidence_date", "relevance",
    ]

    audit_df = pd.DataFrame(
        audit,
        columns=audit_columns,
    )

    return retrieval, audit_df


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sec-chunks",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--query-fraction",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if not 0 < args.query_fraction < 1:
        raise ValueError(
            "query-fraction must be between 0 and 1"
        )

    if args.top_k < 2:
        raise ValueError(
            "top-k must be at least 2"
        )

    print("=" * 75)
    print("AEGIS UNIFIED KG-RAG V3")
    print("=" * 75)

    embeddings, embedding_cols = (
        base.read_embedding()
    )

    embeddings = embeddings[
        (embeddings["date"] >= "2021-01-01")
        & (embeddings["date"] <= "2023-12-31")
    ].copy()

    kg = base.read_kg()

    kg_index = base.company_edges(kg)

    sec_path = Path(
        args.sec_chunks
    ).resolve()

    sec, sec_parse = base.read_sec(
        sec_path
    )

    if sec.empty:
        raise RuntimeError(
            "SEC parser returned zero chunks"
        )

    sec_index = base.sec_index(sec)

    manifest = base.build_manifest(
        kg_index,
        sec_index,
    )

    print("\nBuilding historical states...")

    states = build_raw_states(
        embeddings,
        kg_index,
        sec_index,
        embedding_cols,
    )

    states = add_state_scores(
        states
    )

    kg_threshold, sec_threshold = (
        calibrate_thresholds(
            states,
            args.query_fraction,
        )
    )

    print("\n===== 2021 CALIBRATION =====")
    print(
        "KG threshold:",
        round(kg_threshold, 6)
    )
    print(
        "SEC threshold:",
        round(sec_threshold, 6)
    )

    states["action"] = [
        select_action(
            row,
            kg_threshold,
            sec_threshold,
        )
        for row in states.itertuples(index=False)
    ]

    states["split"] = np.where(
        states["date"].dt.year == 2021,
        "val",
        "test",
    )

    print("\n===== ACTION DISTRIBUTION =====")
    print(
        pd.crosstab(
            states["split"],
            states["action"],
        ).to_string()
    )

    print("\nRunning historical retrieval...")

    retrieval, audit = run_retrieval(
        states,
        kg_index,
        sec_index,
        manifest,
        args.top_k,
    )

    result = states.merge(
        retrieval,
        on=["date", "tic"],
        how="left",
        validate="one_to_one",
    )

    assert len(result) == len(states)

    result.to_csv(
        OUT / "aegis_daily_features.csv",
        index=False,
    )

    audit.to_csv(
        OUT / "retrieval_audit.csv",
        index=False,
    )

    summary = result.groupby(
        ["split", "action"]
    ).agg(
        cases=("tic", "size"),
        kg_evidence=("retrieved_kg", "sum"),
        sec_evidence=("retrieved_sec", "sum"),
        verified=("verified_count", "sum"),
        fallback_top1=("fallback_top1", "sum"),
        fallback_top2=("fallback_top2", "sum"),
        fallback_top3=("fallback_top3", "sum"),
    ).reset_index()

    summary.to_csv(
        OUT / "summary.csv",
        index=False,
    )

    metadata = {
        "version": "v3",
        "query_fraction": args.query_fraction,
        "top_k": args.top_k,
        "kg_threshold_2021": kg_threshold,
        "sec_threshold_2021": sec_threshold,
        "sec_chunks": str(sec_path),
        "sec_parsing": sec_parse,
        "daily_rows": len(result),
        "audit_rows": len(audit),
        "policy_type": (
            "Validation-calibrated heuristic; "
            "not learned RL"
        ),
        "warnings": [
            "No hidden-KG replay",
            "No natural blind-spot ground truth",
            "Embedding checkpoint trained 2018-2020",
            "SEC availability uses publication date",
            "Date-level as-of only",
            "No portfolio reward",
            "Hash verification is not factual verification",
            "Query budget is a calibration hyperparameter",
            "No claim of better financial performance",
        ],
    }

    with (
        OUT / "metadata.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n===== ACTUAL RETRIEVAL =====")

    if audit.empty:
        print("No evidence retrieved.")
    else:
        print(
            audit["source"]
            .value_counts()
            .to_string()
        )

    print("\n===== SUMMARY =====")
    print(
        summary.to_string(index=False)
    )

    print("\nSaved:", OUT)
    print("AEGIS UNIFIED V3: PASS")


if __name__ == "__main__":
    main()
