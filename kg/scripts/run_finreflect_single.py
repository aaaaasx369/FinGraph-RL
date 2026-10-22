import json
import os
from collections import Counter
from pathlib import Path

from kg.src.finreflect.common import (
    OUTPUT_ROOT,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    attach_metadata,
    load_sec_chunks,
)

from kg.src.finreflect.single_pass import (
    extract_single_pass,
)


# ============================================================
# Runtime config
# ============================================================

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

APPEND = (
    os.getenv(
        "KG_APPEND",
        "0",
    )
    == "1"
)

RESUME = (
    os.getenv(
        "KG_RESUME",
        "0",
    )
    == "1"
)


OUTPUT_NAME = os.getenv(
    "KG_OUTPUT_NAME",
    "single_pass.jsonl",
)

OUTPUT_PATH = (
    OUTPUT_ROOT
    / OUTPUT_NAME
)

COMPLETED_PATH = (
    OUTPUT_ROOT
    / (
        Path(
            OUTPUT_NAME
        ).stem
        + "_completed.txt"
    )
)


# ============================================================
# Helpers
# ============================================================

def load_completed_ids():

    if not COMPLETED_PATH.exists():
        return set()

    completed = set()

    with open(
        COMPLETED_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            cid = line.strip()

            if cid:
                completed.add(cid)

    return completed


def mark_completed(
    chunk_id: str,
):

    with open(
        COMPLETED_PATH,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            chunk_id
            + "\n"
        )

        f.flush()


def load_existing_relation_counts():

    counts = Counter()

    if not OUTPUT_PATH.exists():
        return counts

    with open(
        OUTPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            try:
                x = json.loads(
                    line
                )

                relation = x.get(
                    "relation"
                )

                if relation:
                    counts[
                        relation
                    ] += 1

            except Exception:
                continue

    return counts


# ============================================================
# Main
# ============================================================

def main():

    rows = load_sec_chunks(
        max_chunks=MAX_CHUNKS,
        offset=CHUNK_OFFSET,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Fresh run
    # --------------------------------------------------------

    if not APPEND:

        if OUTPUT_PATH.exists():
            OUTPUT_PATH.unlink()

        if COMPLETED_PATH.exists():
            COMPLETED_PATH.unlink()


    # --------------------------------------------------------
    # Resume state
    # --------------------------------------------------------

    completed_ids = (
        load_completed_ids()
        if RESUME
        else set()
    )


    print("=" * 100)
    print(
        "FINREFLECTKG PAPER-FAITHFUL REIMPLEMENTATION"
    )
    print("MODE: SINGLE-PASS")
    print("=" * 100)

    print(
        "Model:",
        OLLAMA_MODEL,
    )

    print(
        "Host:",
        OLLAMA_HOST,
    )

    print(
        "Chunk offset:",
        CHUNK_OFFSET,
    )

    print(
        "Max chunks:",
        MAX_CHUNKS,
    )

    print(
        "Loaded chunks:",
        len(rows),
    )

    print(
        "Append:",
        APPEND,
    )

    print(
        "Resume:",
        RESUME,
    )

    print(
        "Already completed:",
        len(completed_ids),
    )

    print(
        "Output:",
        OUTPUT_PATH,
    )

    print(
        "Completion ledger:",
        COMPLETED_PATH,
    )


    # Statistics for THIS run.
    run_edges = []

    run_completed = 0
    run_skipped = 0
    run_errors = 0
    run_zero_edge = 0


    mode = (
        "a"
        if APPEND
        else "w"
    )


    with open(
        OUTPUT_PATH,
        mode,
        encoding="utf-8",
    ) as out:

        for i, row in enumerate(
            rows,
            start=1,
        ):

            chunk_id = row.get(
                "chunk_id"
            )

            print(
                f"\n[{i}/{len(rows)}] "
                f"{row.get('ticker')} | "
                f"{row.get('form')} | "
                f"{row.get('published_at')} | "
                f"{chunk_id}"
            )


            # ------------------------------------------------
            # Resume skip
            # ------------------------------------------------

            if (
                RESUME
                and
                chunk_id in completed_ids
            ):

                print(
                    "SKIP: already completed"
                )

                run_skipped += 1

                continue


            # ------------------------------------------------
            # Extraction
            # ------------------------------------------------

            try:

                triples = (
                    extract_single_pass(
                        row
                    )
                )

            except Exception as e:

                print(
                    "ERROR:",
                    repr(e),
                )

                run_errors += 1

                # IMPORTANT:
                # Do not mark errored chunks complete,
                # so RESUME can retry them later.
                continue


            print(
                "Triples:",
                len(triples),
            )


            if len(triples) == 0:
                run_zero_edge += 1


            # ------------------------------------------------
            # Write edges
            # ------------------------------------------------

            for triple in triples:

                edge = attach_metadata(
                    row=row,
                    triple=triple,
                    mode="single_pass",
                )

                run_edges.append(
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


            # ------------------------------------------------
            # Mark successful chunk complete
            #
            # IMPORTANT:
            # A zero-edge chunk is still completed.
            # ------------------------------------------------

            mark_completed(
                chunk_id
            )

            completed_ids.add(
                chunk_id
            )

            run_completed += 1


    # ========================================================
    # Summary
    # ========================================================

    print("\n")
    print("=" * 100)
    print("RUN SUMMARY")
    print("=" * 100)

    print(
        "Chunks loaded:",
        len(rows),
    )

    print(
        "Chunks completed this run:",
        run_completed,
    )

    print(
        "Chunks skipped:",
        run_skipped,
    )

    print(
        "Chunks with zero edges:",
        run_zero_edge,
    )

    print(
        "Errors:",
        run_errors,
    )

    print(
        "Edges generated this run:",
        len(run_edges),
    )


    run_relation_counts = Counter(
        x["relation"]
        for x in run_edges
    )


    print(
        "\nRELATION COUNTS — THIS RUN"
    )

    for relation, n in (
        run_relation_counts
        .most_common()
    ):

        print(
            f"{relation:30s}",
            n,
        )


    # ========================================================
    # Entire output statistics
    # ========================================================

    total_relation_counts = (
        load_existing_relation_counts()
    )

    total_edges = sum(
        total_relation_counts.values()
    )


    print("\n")
    print("=" * 100)
    print("TOTAL OUTPUT SUMMARY")
    print("=" * 100)

    print(
        "Completed chunk IDs:",
        len(
            load_completed_ids()
        ),
    )

    print(
        "Total edges in output:",
        total_edges,
    )

    print(
        "\nRELATION COUNTS — TOTAL"
    )

    for relation, n in (
        total_relation_counts
        .most_common()
    ):

        print(
            f"{relation:30s}",
            n,
        )


    print(
        "\nOUTPUT:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nCOMPLETED:"
    )

    print(
        COMPLETED_PATH
    )


if __name__ == "__main__":
    main()
