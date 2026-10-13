#!/usr/bin/env python3

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original"
SOURCE = ROOT / "references/Financial-TKG"
RUN = BASE / "original_run"

AUDIT = BASE / "coverage_audit.json"
DATA = BASE / "dataset/SupplyChain"

TRAINER = SOURCE / "supply_chain_DKG.py"
PACKAGE = SOURCE / "DKG"


def fail(message):
    raise RuntimeError("ORIGINAL DKG STOP: " + str(message))


def require_file(path):
    if not path.exists():
        fail(f"Missing: {path}")


def replace_once(text, pattern, replacement):
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if count != 1:
        fail(
            "Original source signature changed: "
            + pattern
        )

    return updated


def main():
    for path in [
        AUDIT, TRAINER, PACKAGE,
        DATA / "train.txt",
        DATA / "valid.txt",
        DATA / "test.txt",
        DATA / "stat.txt",
        DATA / "entity2id.txt",
    ]:
        require_file(path)

    audit = json.loads(
        AUDIT.read_text(encoding="utf-8")
    )

    if not audit.get("all_four_present"):
        fail(
            "Four historical relations are not all present."
        )

    for split in ["train", "valid", "test"]:
        if audit.get(f"{split}_events", 0) <= 0:
            fail(f"Empty {split} split")

    if audit.get("strict_pit_certified") is True:
        print("PIT externally certified.")
    else:
        print(
            "WARNING: strict PIT certification "
            "is still pending."
        )

    import torch
    import dgl

    print("PyTorch:", torch.__version__)
    print("DGL:", dgl.__version__)

    RUN.mkdir(parents=True, exist_ok=True)

    # Copy original model code to a separate run directory.
    target_package = RUN / "DKG"

    if not target_package.exists():
        shutil.copytree(
            PACKAGE, target_package
        )

    target_data_root = RUN / "FinDKG_dataset"
    target_data_root.mkdir(exist_ok=True)

    target_dataset = target_data_root / "SupplyChain"

    if not target_dataset.exists():
        target_dataset.symlink_to(
            DATA.resolve(),
            target_is_directory=True,
        )

    # Keep the architecture from the original source.
    # Only change experiment-specific hyperparameters.
    code = TRAINER.read_text(encoding="utf-8")

    if "EmbeddingUpdater(" not in code:
        fail("Original EmbeddingUpdater not found")

    if "DynamicGraphModel(" not in code:
        fail("Original DynamicGraphModel not found")

    if "KGT+RNN" not in code:
        fail("Original KGTransformer setting not found")

    code = replace_once(
        code,
        r"^epoch_times\s*=\s*\d+\s*$",
        "epoch_times = 5",
    )

    code = replace_once(
        code,
        r"^dim_num\s*=\s*\d+.*$",
        "dim_num = 64",
    )

    code = replace_once(
        code,
        r"^random_seed\s*=\s*\d+\s*$",
        "random_seed = 41",
    )

    code = replace_once(
        code,
        r"^graph_mode\s*=\s*[\"'][^\"']+[\"'].*$",
        'graph_mode = "SupplyChain"',
    )

    # CPU smoke test.
    code = replace_once(
        code,
        r"^args\.seed\s*=\s*random_seed.*$",
        "args.seed = random_seed\nargs.gpu = -1",
    )

    output = RUN / "run_original.py"
    output.write_text(
        code, encoding="utf-8"
    )

    ast.parse(code)

    print("\n=== ORIGINAL CODE CHECK ===")
    print("Source:", TRAINER)
    print("Copied DKG:", target_package)
    print("Dataset:", target_dataset)
    print("Model: KGTransformer / KGT+RNN")
    print("Epochs: 5")
    print("Dimension: 64")
    print("Device: CPU")

    # Validate the original loader before training.
    check = """
import sys
sys.path.insert(0, '.')
from DKG import settings
from DKG import data

print("Supported graphs:", settings.ALL_GRAPHS)

if "SupplyChain" not in settings.ALL_GRAPHS:
    raise RuntimeError(
        "SupplyChain is not registered in "
        "the original DKG settings"
    )

graph = data.load_temporal_knowledge_graph(
    "SupplyChain",
    data_root="./FinDKG_dataset",
)

print("Nodes:", graph.num_nodes())
print("Edges:", graph.num_edges())
print("Relations:", graph.num_relations)
print("Train timestamps:", len(graph.train_times))
print("Valid timestamps:", len(graph.val_times))
print("Test timestamps:", len(graph.test_times))
"""

    subprocess.run(
        [
            sys.executable,
            "-c",
            check,
        ],
        cwd=RUN,
        check=True,
    )

    print("\n=== ORIGINAL KGTRANSFORMER TRAINING ===")

    log = BASE / "original_training.log"

    with log.open("w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, str(output)],
            cwd=RUN,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    print("Training exit code:", result.returncode)
    print("Log:", log)

    print("\n=== FINAL LOG ===")
    print(
        "\n".join(
            log.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()[-60:]
        )
    )

    if result.returncode:
        fail("Original training returned nonzero exit code")

    print("\n=== ORIGINAL TRAINER FINISHED ===")
    print(
        "Inspect training log and checkpoint "
        "before declaring model training successful."
    )


if __name__ == "__main__":
    main()
