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

from kg.src.finreflect.multi_pass import (
    extract_multi_pass,
)


MAX_CHUNKS = int(
    os.getenv(
        "KG_MAX_CHUNKS",
        "20",
    )
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / "multi_pass.jsonl"
)


def main():

    rows = load_sec_chunks(
        max_chunks=MAX_CHUNKS
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("FINREFLECTKG REIMPLEMENTATION")
    print("MODE: MULTI-PASS")
    print("=" * 90)

    print("Model:", OLLAMA_MODEL)
    print("Host:", OLLAMA_HOST)
    print("Chunks:", len(rows))

    all_edges = []

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as out:

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

                triples = extract_multi_pass(
                    row
                )

            except Exception as e:

                print(
                    "ERROR:",
                    repr(e),
                )

                continue

            print(
                "Triples:",
                len(triples),
            )

            for triple in triples:

                edge = attach_metadata(
                    row,
                    triple,
                    "multi_pass",
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


if __name__ == "__main__":
    main()
