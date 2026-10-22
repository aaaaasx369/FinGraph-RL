#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd

from kg.src.finreflect.common import (
    OUTPUT_ROOT,
    load_sec_chunks,
)

from kg.src.finreflect.check_rules import (
    evaluate_edge,
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


def load_edges(
    path: Path,
):

    rows = []

    if not path.exists():
        return rows

    with open(
        path,
        encoding="utf-8",
    ) as f:

        for line in f:

            if line.strip():

                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def main():

    # Load enough source chunks to build lookup.
    chunks = load_sec_chunks(
        max_chunks=None
    )

    source_lookup = {}

    for row in chunks:

        key = (
            row.get(
                "chunk_id"
            )
        )

        source_lookup[
            key
        ] = row.get(
            "text",
            "",
        )


    detail_rows = []

    summary_rows = []


    for mode, path in (
        MODES.items()
    ):

        edges = load_edges(
            path
        )

        mode_results = []


        for edge in edges:

            source = (
                source_lookup.get(
                    edge.get(
                        "chunk_id"
                    ),
                    "",
                )
            )

            checks = evaluate_edge(
                edge,
                source,
            )

            row = {
                "mode":
                    mode,

                "ticker":
                    edge.get(
                        "ticker"
                    ),

                "published_at":
                    edge.get(
                        "published_at"
                    ),

                "chunk_id":
                    edge.get(
                        "chunk_id"
                    ),

                "subject":
                    edge.get(
                        "subject"
                    ),

                "subject_type":
                    edge.get(
                        "subject_type"
                    ),

                "relation":
                    edge.get(
                        "relation"
                    ),

                "object":
                    edge.get(
                        "object"
                    ),

                "object_type":
                    edge.get(
                        "object_type"
                    ),

                **checks,
            }

            detail_rows.append(
                row
            )

            mode_results.append(
                checks
            )


        n = len(
            mode_results
        )


        def rate(
            field,
        ):

            if n == 0:
                return 0.0

            return sum(
                bool(
                    x[field]
                )
                for x
                in mode_results
            ) / n


        summary_rows.append(
            {
                "mode":
                    mode,

                "n_edges":
                    n,

                "subject_reference_rate":
                    rate(
                        "subject_reference"
                    ),

                "entity_length_rate":
                    rate(
                        "entity_length"
                    ),

                "entity_schema_rate":
                    rate(
                        "entity_schema"
                    ),

                "relation_schema_rate":
                    rate(
                        "relation_schema"
                    ),

                "all_rules_pass_rate":
                    rate(
                        "all_rules_pass"
                    ),
            }
        )


    detail_df = pd.DataFrame(
        detail_rows
    )

    summary_df = pd.DataFrame(
        summary_rows
    )


    detail_path = (
        OUTPUT_ROOT
        / "checkrules_detail.csv"
    )

    summary_path = (
        OUTPUT_ROOT
        / "checkrules_summary.csv"
    )


    detail_df.to_csv(
        detail_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )


    print("\n")
    print("=" * 110)
    print("FINREFLECT CHECKRULES COMPARISON")
    print("=" * 110)

    print(
        summary_df.to_string(
            index=False,
        )
    )


    print(
        "\nFAILED EXAMPLES"
    )

    failed = detail_df[
        ~detail_df[
            "all_rules_pass"
        ]
    ]

    if len(failed):

        print(
            failed[
                [
                    "mode",
                    "subject",
                    "relation",
                    "object",
                    "subject_reference",
                    "entity_length",
                    "entity_schema",
                    "relation_schema",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    print(
        "\nSaved:"
    )

    print(
        summary_path
    )

    print(
        detail_path
    )


if __name__ == "__main__":
    main()
