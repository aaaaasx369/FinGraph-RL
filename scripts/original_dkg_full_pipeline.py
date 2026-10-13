#!/usr/bin/env python3

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
SOURCE = ROOT / "references/Financial-TKG"
ORIGINAL = SOURCE / "FinDKG_dataset/FinDKG-full"

RUN = BASE / "original_run"
DATA = RUN / "FinDKG_dataset/FinDKG-full"

TARGETS = {
    "AAPL": 9,
    "MSFT": 30,
    "NVDA": 221,
    "AMZN": 20,
    "GOOGL": 11,
}


def fail(message):
    raise RuntimeError("ORIGINAL DKG STOP: " + message)


def prepare():
    DATA.mkdir(parents=True, exist_ok=True)

    times = pd.read_csv(ORIGINAL / "time2id.txt")
    times["DATE_WK"] = pd.to_datetime(times["DATE_WK"])

    frames = []

    for split in ["train", "valid", "test"]:
        part = pd.read_csv(
            ORIGINAL / f"{split}.txt",
            sep="\t",
            header=None,
            names=["head", "rel", "tail", "time", "extra"],
        )

        frames.append(part)

    events = pd.concat(frames, ignore_index=True)

    events = events.merge(
        times,
        left_on="time",
        right_on="TimeID",
        how="left",
        validate="many_to_one",
    )

    if events["DATE_WK"].isna().any():
        fail("Missing historical dates")

    splits = {
        "train": events[
            events["DATE_WK"] < "2022-01-01"
        ],
        "valid": events[
            (events["DATE_WK"] >= "2022-01-01")
            & (events["DATE_WK"] < "2023-01-01")
        ],
        "test": events[
            events["DATE_WK"] >= "2023-01-01"
        ],
    }

    for name, frame in splits.items():
        if frame.empty:
            fail(f"Empty {name} split")

        frame[
            ["head", "rel", "tail", "time", "extra"]
        ].to_csv(
            DATA / f"{name}.txt",
            sep="\t",
            index=False,
            header=False,
        )

        print(
            f"{name}: {len(frame)} events, "
            f"{frame['time'].nunique()} weeks"
        )

    for filename in [
        "entity2id.txt",
        "relation2id.txt",
        "stat.txt",
        "time2id.txt",
    ]:
        shutil.copy2(
            ORIGINAL / filename,
            DATA / filename,
        )

    BASE.mkdir(parents=True, exist_ok=True)

    (BASE / "data_manifest.json").write_text(
        json.dumps(
            {
                "total_events": len(events),
                "train_events": len(splits["train"]),
                "prediction_2022_events": len(splits["valid"]),
                "prediction_2023_events": len(splits["test"]),
                "targets": TARGETS,
                "training_period": "2018-2021",
                "prediction_period": "2022-2023",
                "strict_pit_certified": False,
            },
            indent=2,
        )
    )

    print("\n=== ORIGINAL DATA PREPARED ===")


def replace_once(text, pattern, replacement):
    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )

    if count != 1:
        fail("Original trainer changed: " + pattern)

    return updated


def setup(epochs, dim):
    if not (DATA / "train.txt").exists():
        fail("Run prepare first")

    package_src = SOURCE / "DKG"
    package_dst = RUN / "DKG"

    if not package_src.exists():
        fail("Missing original DKG package")

    if package_dst.exists() or package_dst.is_symlink():
        if package_dst.is_symlink():
            package_dst.unlink()
        else:
            shutil.rmtree(package_dst)

    shutil.copytree(package_src, package_dst)

    settings = package_dst / "settings.py"
    text = settings.read_text()

    text += (
        '\nGRAPH_FIN_FULL = "FinDKG-full"\n'
        'ALL_GRAPHS = list(ALL_GRAPHS) + [GRAPH_FIN_FULL]\n'
    )

    settings.write_text(text)

    loader = package_dst / "data.py"
    text = loader.read_text()

    text = text.replace(
        "['FinDKG', 'FinDKG-live']",
        "['FinDKG', 'FinDKG-live', 'FinDKG-full']",
    )

    loader.write_text(text)

    original_trainer = SOURCE / "supply_chain_DKG.py"

    if not original_trainer.exists():
        fail("Missing original training code")

    code = original_trainer.read_text()

    changes = [
        (
            r"^graph_mode\s*=.*$",
            'graph_mode = "FinDKG-full"',
        ),
        (
            r"^data_root_path\s*=.*$",
            "data_root_path = './FinDKG_dataset'",
        ),
        (
            r"^epoch_times\s*=.*$",
            f"epoch_times = {epochs}",
        ),
        (
            r"^dim_num\s*=.*$",
            f"dim_num = {dim}",
        ),
        (
            r"^flag_eval\s*=.*$",
            "flag_eval = False",
        ),
    ]

    for pattern, replacement in changes:
        code = replace_once(
            code, pattern, replacement
        )

    code = replace_once(
        code,
        r"^args\.seed\s*=\s*random_seed.*$",
        "args.seed = random_seed\nargs.gpu = -1",
    )

    if "EmbeddingUpdater(" not in code:
        fail("Original EmbeddingUpdater not found")

    if "DynamicGraphModel(" not in code:
        fail("Original DynamicGraphModel not found")

    if "KGT+RNN" not in code:
        fail("Original KGTransformer setting not found")

    ast.parse(code)

    trainer = RUN / "run_original.py"
    trainer.write_text(code)

    print("\n=== ORIGINAL KGTRANSFORMER SETUP ===")
    print("Trainer:", trainer)
    print("Epochs:", epochs)
    print("Dimension:", dim)
    print("Training: 2018-2021")
    print("Prediction datasets: 2022-2023")
    print("Device: CPU")
    print("Model: Original Financial-TKG")


def run():
    trainer = RUN / "run_original.py"

    if not trainer.exists():
        fail("Run setup first")

    try:
        import torch
        import dgl
    except ImportError as exc:
        fail("DGL environment not ready: " + str(exc))

    print("Torch:", torch.__version__)
    print("DGL:", dgl.__version__)

    log = BASE / "original_train.log"

    with log.open("w") as f:
        result = subprocess.run(
            [sys.executable, "-u", str(trainer)],
            cwd=RUN,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

    print("Exit code:", result.returncode)
    print("Log:", log)

    print("\n=== LAST 80 LOG LINES ===")
    print(
        "\n".join(
            log.read_text(errors="replace").splitlines()[-80:]
        )
    )

    if result.returncode != 0:
        fail("Original KGTransformer training failed")

    print("\n=== ORIGINAL TRAINING SCRIPT FINISHED ===")
    print(
        "Training completed; checkpoint and "
        "time-safe embedding export still require audit."
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=["prepare", "setup", "run"],
    )

    p.add_argument(
        "--epochs", type=int, default=3
    )

    p.add_argument(
        "--dim", type=int, default=64
    )

    args = p.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "setup":
        setup(args.epochs, args.dim)
    else:
        run()


if __name__ == "__main__":
    main()
