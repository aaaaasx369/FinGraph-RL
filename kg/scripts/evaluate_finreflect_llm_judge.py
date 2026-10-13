#!/usr/bin/env python3

import json
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd

from kg.src.finreflect.common import (
    OUTPUT_ROOT,
    load_sec_chunks,
)

from kg.src.finreflect.llm_judge import (
    JUDGE_MODEL,
    judge_extraction,
)


MAX_CHUNKS = int(
    os.getenv(
        "FINREFLECT_JUDGE_MAX_CHUNKS",
        "20",
    )
)


MODES = {
    "single_pass":
        OUTPUT_ROOT
        / "single_pass.jsonl",

    "multi_pass":
        OUTPUT_ROOT
        / "multi_pass.jsonl",

    "reflection":
        OUTPUT_ROOT
        / "reflection.jsonl",
}


def load_edges_by_chunk(
    path: Path,
):

    grouped = defaultdict(list)

    if not path.exists():
        return grouped

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            grouped[
                x.get("chunk_id")
            ].append(
                x
            )

    return grouped


def edge_for_judge(x):

    return {
        "subject":
            x.get("subject"),

        "subject_type":
            x.get("subject_type"),

        "relation":
            x.get("relation"),

        "object":
            x.get("object"),

        "object_type":
            x.get("object_type"),

        "evidence":
            x.get("evidence"),
    }


def main():

    chunks = load_sec_chunks(
        max_chunks=MAX_CHUNKS
    )

    mode_edges = {
        mode:
            load_edges_by_chunk(path)

        for mode, path
        in MODES.items()
    }


    detail_rows = []


    print("=" * 100)
    print("FINREFLECT LLM-AS-A-JUDGE")
    print("=" * 100)

    print(
        "Judge model:",
        JUDGE_MODEL,
    )

    print(
        "Chunks:",
        len(chunks),
    )


    for i, chunk in enumerate(
        chunks,
        1,
    ):

        chunk_id = chunk.get(
            "chunk_id"
        )

        print(
            f"\n[{i}/{len(chunks)}] "
            f"{chunk.get('ticker')} | "
            f"{chunk.get('form')} | "
            f"{chunk.get('published_at')}"
        )


        for mode in [
            "single_pass",
            "multi_pass",
            "reflection",
        ]:

            edges = mode_edges[
                mode
            ].get(
                chunk_id,
                [],
            )

            triples = [
                edge_for_judge(x)
                for x in edges
            ]

            print(
                f"  {mode}: "
                f"{len(triples)} triples",
                flush=True,
            )


            try:

                result = judge_extraction(
                    source_text=
                        chunk["text"],

                    triples=
                        triples,
                )

            except Exception as e:

                print(
                    "    JUDGE ERROR:",
                    repr(e),
                )

                continue


            detail_rows.append(
                {
                    "mode":
                        mode,

                    "ticker":
                        chunk.get(
                            "ticker"
                        ),

                    "published_at":
                        chunk.get(
                            "published_at"
                        ),

                    "form":
                        chunk.get(
                            "form"
                        ),

                    "chunk_id":
                        chunk_id,

                    "n_edges":
                        len(
                            triples
                        ),

                    **result,
                }
            )


            print(
                "    "
                f"P={result['precision']:.1f} "
                f"F={result['faithfulness']:.1f} "
                f"C={result['comprehensiveness']:.1f} "
                f"R={result['relevance']:.1f}"
            )


    detail_df = pd.DataFrame(
        detail_rows
    )


    detail_path = (
        OUTPUT_ROOT
        / "llm_judge_detail.csv"
    )

    summary_path = (
        OUTPUT_ROOT
        / "llm_judge_summary.csv"
    )


    detail_df.to_csv(
        detail_path,
        index=False,
    )


    metrics = [
        "precision",
        "faithfulness",
        "comprehensiveness",
        "relevance",
    ]


    if len(detail_df):

        summary = (
            detail_df
            .groupby("mode")[metrics]
            .agg(
                ["mean", "std"]
            )
        )

        summary.columns = [
            f"{a}_{b}"
            for a, b
            in summary.columns
        ]

        summary = (
            summary
            .reset_index()
        )

        summary[
            "overall_mean"
        ] = summary[
            [
                "precision_mean",
                "faithfulness_mean",
                "comprehensiveness_mean",
                "relevance_mean",
            ]
        ].mean(
            axis=1
        )

    else:

        summary = pd.DataFrame()


    summary.to_csv(
        summary_path,
        index=False,
    )


    print("\n")
    print("=" * 100)
    print("LLM JUDGE SUMMARY")
    print("=" * 100)

    if len(summary):

        print(
            summary.to_string(
                index=False
            )
        )

    print("\nSaved:")
    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
