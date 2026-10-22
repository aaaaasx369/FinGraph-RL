#!/usr/bin/env python3

import json
import os
from collections import Counter

from kg.src.finreflect.common import (
    OUTPUT_ROOT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    attach_metadata,
    load_sec_chunks,
)

from kg.src.finreflect.reflection import (
    extract_reflection,
)


MAX_CHUNKS = int(
    os.getenv(
        "KG_MAX_CHUNKS",
        "20",
    )
)

CHUNK_OFFSET = int(
    os.getenv(
        "KG_CHUNK_OFFSET",
        "0",
    )
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "reflection.jsonl"
)

FEEDBACK_PATH = (
    OUTPUT_ROOT
    / "reflection_feedback.jsonl"
)


def main():

    rows = load_sec_chunks(
        max_chunks=MAX_CHUNKS,
        offset=CHUNK_OFFSET,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("FINREFLECTKG REIMPLEMENTATION")
    print("MODE: REFLECTION-AGENT")
    print("=" * 90)

    print("Model:", OLLAMA_MODEL)
    print("Host:", OLLAMA_HOST)
    print("Chunks:", len(rows))

    all_edges = []

    append_mode = os.getenv(
        "KG_APPEND",
        "0",
    ) == "1"

    file_mode = (
        "a"
        if append_mode
        else "w"
    )

    print(
        "Output mode:",
        "append" if append_mode else "overwrite",
    )

    with open(
        OUTPUT_PATH,
        file_mode,
        encoding="utf-8",
    ) as out, open(
        FEEDBACK_PATH,
        file_mode,
        encoding="utf-8",
    ) as feedback_out:

        for i, row in enumerate(
            rows,
            1,
        ):

            print(
                f"\n[{i}/{len(rows)}] "
                f"{row.get('ticker')} | "
                f"{row.get('form')} | "
                f"{row.get('published_at')}"
            )

            try:

                triples, feedback_history = (
                    extract_reflection(
                        row
                    )
                )

            except Exception as e:

                print(
                    "ERROR:",
                    repr(e),
                )

                continue

            feedback_out.write(
                json.dumps(
                    {
                        "ticker":
                            row.get("ticker"),

                        "published_at":
                            row.get(
                                "published_at"
                            ),

                        "form":
                            row.get("form"),

                        "doc_id":
                            row.get("doc_id"),

                        "chunk_id":
                            row.get("chunk_id"),

                        "feedback":
                            feedback_history,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            feedback_out.flush()

            print(
                "Final triples:",
                len(triples),
            )

            for triple in triples:

                edge = attach_metadata(
                    row,
                    triple,
                    "reflection",
                )

                all_edges.append(
                    edge
                )

                out.write(
                    json.dumps(
                        edge,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                out.flush()

                print(
                    f"  {edge['subject']} "
                    f"--{edge['relation']}--> "
                    f"{edge['object']}"
                )

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        "Total edges:",
        len(all_edges),
    )

    counts = Counter(
        x["relation"]
        for x in all_edges
    )

    print("\nRELATION COUNTS")

    for relation, n in (
        counts.most_common()
    ):

        print(
            f"{relation:25s}",
            n,
        )

    print("\nOUTPUT:")
    print(OUTPUT_PATH)

    print("\nFEEDBACK:")
    print(FEEDBACK_PATH)


if __name__ == "__main__":
    main()
