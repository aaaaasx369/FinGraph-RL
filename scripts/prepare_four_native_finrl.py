#!/usr/bin/env python3

from pathlib import Path
import ast
import py_compile
import re

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

EXPORT_SRC = SCRIPTS / "export_original_kgtransformer.py"
RUNNER_SRC = SCRIPTS / "run_original_kgtransformer_finrl.py"

EXPORT_DST = SCRIPTS / "export_four_native_kgtransformer.py"
RUNNER_DST = SCRIPTS / "run_four_native_finrl.py"

BASE = ROOT / "outputs/finrl_dkg_four_native"
RUN = BASE / "original_run"

CHECKPOINT = (
    RUN
    / "result/FinDKG-full/DKG/"
    "FinDKG-full_KGTransformer_"
    "overall_best_checkpoint_opt_edge.pt"
)

DATASET = RUN / "FinDKG_dataset/FinDKG-full"

EXPECTED_RELATIONS = [
    "Invests_In",
    "Operate_In",
    "Positive_Impact_On",
    "Negative_Impact_On",
]


def stop(message):
    raise RuntimeError("FOUR_NATIVE PREP STOP: " + message)


def replace_once(source, old, new, label):
    count = source.count(old)

    if count != 1:
        stop(
            f"{label}: expected one match, "
            f"found {count}"
        )

    return source.replace(old, new, 1)


def safe_write(path, text):
    if path.exists():
        previous = path.read_text(encoding="utf-8")

        if previous != text:
            stop(
                f"Existing generated script differs: {path}\n"
                "Refusing to overwrite it."
            )

        print("UNCHANGED:", path.name)
        return

    ast.parse(text, filename=str(path))
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)

    print("CREATED:", path.name)


def check_files():
    for path in [
        EXPORT_SRC,
        RUNNER_SRC,
        CHECKPOINT,
        DATASET / "train.txt",
        DATASET / "valid.txt",
        DATASET / "test.txt",
        DATASET / "relation2id.txt",
        DATASET / "time2id.txt",
    ]:
        if not path.is_file():
            stop(f"Missing file: {path}")

    relations = {}

    for line in (
        DATASET / "relation2id.txt"
    ).read_text().splitlines():
        if not line.strip():
            continue

        name, rid = line.rsplit(None, 1)
        relations[int(rid)] = name.strip()

    actual = [
        relations.get(i) for i in range(4)
    ]

    if actual != EXPECTED_RELATIONS:
        stop(
            "Wrong four-relation mapping: "
            + repr(actual)
        )

    print("FOUR RELATIONS:", actual)
    print("CHECKPOINT:", CHECKPOINT)


def generate_exporter():
    source = EXPORT_SRC.read_text(encoding="utf-8")

    # Preserve the original replay implementation.
    # Only switch experiment root and relation count.
    source = replace_once(
        source,
        'BASE = ROOT / "outputs/finrl_dkg_original_full"',
        'BASE = ROOT / "outputs/finrl_dkg_four_native"',
        "Exporter experiment base",
    )

    source = replace_once(
        source,
        "if G.num_relations != 15:",
        "if G.num_relations != 4:",
        "KG relation count",
    )

    # Checkpoint, output directory, replay dataset
    # are all derived from BASE -> RUN.
    required = [
        'RUN = BASE / "original_run"',
        'dataset = RUN / "FinDKG_dataset"',
        '"overall_best_checkpoint_opt_edge.pt"',
        'model.load_state_dict(',
        'strict=True',
        '"2018-2021"',
    ]

    for token in required:
        if token not in source:
            stop(
                "Exporter expected logic missing: "
                + token
            )

    # Clear documentation: native four relations.
    source = source.replace(
        "Export original Financial-TKG KGTransformer embeddings.",
        "Export FOUR_NATIVE Financial-TKG KGTransformer embeddings.",
        1,
    )

    source = source.replace(
        '"four_requested_relations_complete": False',
        '"four_native_relations_present": True',
    )

    # Keep the filename as original_daily_embeddings.csv
    # within the NEW isolated experiment base. This
    # preserves compatibility with the old runner.
    safe_write(EXPORT_DST, source)


def generate_runner():
    source = RUNNER_SRC.read_text(encoding="utf-8")

    source = replace_once(
        source,
        'BASE = ROOT / "outputs/finrl_dkg_original_full"',
        'BASE = ROOT / "outputs/finrl_dkg_four_native"',
        "FinRL experiment base",
    )

    replacements = {
        "ORIGINAL_KGTRANSFORMER":
            "FOUR_NATIVE_KGTRANSFORMER",
        "ORIGINAL_KG_PLACEBO":
            "FOUR_NATIVE_KG_PLACEBO",
    }

    for old, new in replacements.items():
        if old not in source:
            stop(f"Runner experiment label missing: {old}")

        source = source.replace(old, new)

    required = [
        'def read_embeddings(',
        'def build(',
        'def execute(',
        '"--input-dir"',
        '"--output-dir"',
    ]

    for token in required:
        if token not in source:
            stop(
                "Runner expected logic missing: "
                + token
            )

    safe_write(RUNNER_DST, source)


def main():
    check_files()
    generate_exporter()
    generate_runner()

    print("\n=== PREPARATION COMPLETE ===")
    print("Exporter:", EXPORT_DST)
    print("FinRL runner:", RUNNER_DST)
    print("15-relation source scripts: UNCHANGED")
    print("New experiment directory:", BASE)
    print("Model replay executed: NO")
    print("FinRL executed: NO")


if __name__ == "__main__":
    main()
