#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
COUNTERFACTUAL = (
    ROOT
    / "unified_event_memory/finrl_x/aegis_rl"
    / "run_stop_counterfactual_v1.py"
)

ALGOS = ("A2C", "PPO", "DDPG")


def discover_models():
    results_root = ROOT / "results/aegis_rl_v1"

    checkpoints = list(
        results_root.rglob("model.zip")
    )

    found = {}

    for algo in ALGOS:
        override = os.environ.get(
            f"{algo}_MODEL", ""
        ).strip()

        if override:
            path = Path(override).expanduser()

            if not path.is_absolute():
                path = ROOT / path

            if not path.is_file():
                raise RuntimeError(
                    f"{algo}_MODEL does not exist: {path}"
                )

            found[algo] = path
            continue

        candidates = []

        for path in checkpoints:
            text = str(path).lower()

            # Require a V5 checkpoint to avoid
            # silently comparing incompatible versions.
            if "v5" not in text:
                continue

            # Exact algorithm token.
            if re.search(
                rf"(?<![a-z0-9]){algo.lower()}"
                rf"(?![a-z0-9])",
                text,
            ):
                candidates.append(path)

        if len(candidates) == 1:
            found[algo] = candidates[0]

        elif len(candidates) == 0:
            raise RuntimeError(
                f"\nNo V5 {algo} checkpoint found.\n"
                f"Set {algo}_MODEL to its exact model.zip path.\n"
                "Available model.zip files:\n"
                + "\n".join(
                    str(p)
                    for p in checkpoints[:40]
                )
            )

        else:
            raise RuntimeError(
                f"\nMultiple V5 {algo} checkpoints.\n"
                f"Set {algo}_MODEL explicitly.\n"
                + "\n".join(
                    str(p) for p in candidates
                )
            )

    return found


def run_one(
    algo,
    model_path,
    split,
    stride,
    max_days,
    output_root,
    device,
):
    output = output_root / algo

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        "-u",
        str(COUNTERFACTUAL),
        "--algo", algo,
        "--split", split,
        "--stride", str(stride),
        "--max-days", str(max_days),
        "--model", str(model_path),
        "--out", str(output),
        "--cost-scale", "0",
        "--device", device,
    ]

    print()
    print("=" * 75, flush=True)
    print(
        f"[RUNNING] {algo} / {split}",
        flush=True,
    )
    print(
        "[CHECKPOINT]", model_path,
        flush=True,
    )
    print("=" * 75, flush=True)

    log_path = output / "run.log"

    with log_path.open(
        "w", encoding="utf-8"
    ) as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        code = process.wait()

    if code != 0:
        raise RuntimeError(
            f"{algo} failed, exit={code}. "
            f"See {log_path}"
        )

    result = (
        output
        / "financial_utility_labels.csv"
    )

    if not result.is_file():
        raise RuntimeError(
            f"Missing labels: {result}"
        )

    frame = pd.read_csv(result)

    expected = {
        "STOP", "KG", "SEC", "BOTH"
    }

    for date, group in frame.groupby("date"):
        if set(group.source) != expected:
            raise RuntimeError(
                f"{algo}: incomplete {date}"
            )

    if not (frame.retrieval_cost == 0).all():
        raise RuntimeError(
            f"{algo}: nonzero retrieval cost!"
        )

    return frame


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="train",
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max-days",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "v6_three_rl_source_ablation"
        ),
    )

    args = parser.parse_args()

    output_root = ROOT / args.out / args.split

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = discover_models()

    print("\n===== THREE RL CHECKPOINTS =====")

    for algo, path in models.items():
        print(f"{algo}: {path}")

    all_frames = []

    for algo in ALGOS:
        frame = run_one(
            algo=algo,
            model_path=models[algo],
            split=args.split,
            stride=args.stride,
            max_days=args.max_days,
            output_root=output_root,
            device=args.device,
        )

        all_frames.append(frame)

    combined = pd.concat(
        all_frames,
        ignore_index=True,
    )

    combined.to_csv(
        output_root / "all_rl_labels.csv",
        index=False,
    )

    # Decision utility with zero retrieval cost.
    utility = combined.pivot_table(
        index=["algorithm", "date"],
        columns="source",
        values="log_reward",
        aggfunc="first",
    )

    utility = utility[
        ["STOP", "KG", "SEC", "BOTH"]
    ]

    utility = utility.subtract(
        utility["STOP"], axis=0
    )

    rows = []

    for algo in ALGOS:
        one = utility.loc[algo]

        kg = one["KG"]
        sec = one["SEC"]
        both = one["BOTH"]

        best_single = pd.concat(
            [kg, sec], axis=1
        ).max(axis=1)

        winner = one[
            ["STOP", "KG", "SEC", "BOTH"]
        ].idxmax(axis=1)

        rows.append({
            "algorithm": algo,
            "days": len(one),

            "mean_KG": kg.mean(),
            "mean_SEC": sec.mean(),
            "mean_BOTH": both.mean(),

            "KG_positive_rate":
                (kg > 0).mean(),

            "SEC_positive_rate":
                (sec > 0).mean(),

            "BOTH_positive_rate":
                (both > 0).mean(),

            "SEC_given_KG":
                (both - kg).mean(),

            "KG_given_SEC":
                (both - sec).mean(),

            "mean_synergy":
                (both - kg - sec).mean(),

            "negative_synergy_rate":
                ((both - kg - sec) < 0).mean(),

            "BOTH_better_than_best_single":
                (both > best_single).mean(),

            "oracle_STOP":
                int((winner == "STOP").sum()),

            "oracle_KG":
                int((winner == "KG").sum()),

            "oracle_SEC":
                int((winner == "SEC").sum()),

            "oracle_BOTH":
                int((winner == "BOTH").sum()),
        })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        output_root
        / "three_rl_source_comparison.csv",
        index=False,
    )

    print()
    print("=" * 90)
    print("THREE RL — ZERO COST SOURCE UTILITY")
    print("=" * 90)

    print(
        summary.to_string(index=False)
    )

    print()
    print("[SAVED]", output_root)
    print(
        "THREE RL SOURCE ABLATION: PASS"
    )


if __name__ == "__main__":
    main()
