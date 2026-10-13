#!/usr/bin/env python3

import os
import subprocess
import sys

from kg.src.finreflect.common import (
    load_sec_chunks,
)


BATCH_SIZE = int(
    os.getenv(
        "KG_BATCH_SIZE",
        "100",
    )
)

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3.5:9b",
)

OUTPUT_NAME = os.getenv(
    "KG_OUTPUT_NAME",
    "single_pass_full_9b.jsonl",
)


def main():

    all_rows = load_sec_chunks(
        max_chunks=None
    )

    total = len(
        all_rows
    )

    print("=" * 100)
    print("FULL FINREFLECT SINGLE-PASS EXTRACTION")
    print("=" * 100)

    print(
        "Total chunks:",
        total,
    )

    print(
        "Batch size:",
        BATCH_SIZE,
    )

    print(
        "Model:",
        MODEL,
    )

    print(
        "Output:",
        OUTPUT_NAME,
    )

    for offset in range(
        0,
        total,
        BATCH_SIZE,
    ):

        n = min(
            BATCH_SIZE,
            total - offset,
        )

        print("\n")
        print("=" * 100)

        print(
            f"BATCH offset={offset} "
            f"size={n}"
        )

        print("=" * 100)

        env = os.environ.copy()

        env[
            "PYTHONPATH"
        ] = "."

        env[
            "KG_CHUNK_OFFSET"
        ] = str(offset)

        env[
            "KG_MAX_CHUNKS"
        ] = str(n)

        env[
            "KG_APPEND"
        ] = "1"

        env[
            "KG_RESUME"
        ] = "1"

        env[
            "KG_OUTPUT_NAME"
        ] = OUTPUT_NAME

        env[
            "OLLAMA_MODEL"
        ] = MODEL

        env[
            "FINREFLECT_DEBUG"
        ] = "0"

        cmd = [
            "./.venv-drl/bin/python",
            "kg/scripts/run_finreflect_single.py",
        ]

        result = subprocess.run(
            cmd,
            env=env,
        )

        if result.returncode != 0:

            print(
                "\nBATCH FAILED:",
                offset,
            )

            sys.exit(
                result.returncode
            )

    print("\n")
    print("=" * 100)
    print("FULL EXTRACTION COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
