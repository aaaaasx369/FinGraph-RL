#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT / "unified_event_memory"
    / "finrl_x"
    / "run_dynamic_kg_drl.py"
)

SEEDS = [42, 123, 2026, 7, 999]
GROUPS = ["clean", "static", "dynamic"]
ALGORITHMS = ["A2C", "PPO", "DDPG"]

METRICS = [
    "final_value",
    "final_multiple",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
]

def summarize(all_df, out):
    all_df.to_csv(
        out / "all_seeds_all_models.csv",
        index=False,
    )

    stats = (
        all_df
        .groupby(["group", "algorithm"])[METRICS]
        .agg(["mean", "std", "count"])
    )

    stats.columns = [
        f"{metric}_{stat}"
        for metric, stat in stats.columns
    ]

    stats = stats.reset_index()

    stats.to_csv(
        out / "multi_seed_summary.csv",
        index=False,
    )

    print("\n" + "=" * 85)
    print("MULTI-SEED PERFORMANCE")
    print("=" * 85)

    cols = [
        "group",
        "algorithm",
        "final_multiple_mean",
        "final_multiple_std",
        "sharpe_mean",
        "sharpe_std",
        "max_drawdown_mean",
        "max_drawdown_std",
    ]

    print(
        stats[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # Paired seed comparison.
    # Each algorithm is compared under identical
    # seed offsets for all feature groups.
    paired_rows = []

    for alg in ALGORITHMS:
        x = all_df[
            all_df["algorithm"] == alg
        ]

        for benchmark in ("clean", "static"):
            dyn = x[x["group"] == "dynamic"].set_index("seed")
            base = x[x["group"] == benchmark].set_index("seed")

            common = sorted(
                set(dyn.index) & set(base.index)
            )

            for seed in common:
                row = {
                    "algorithm": alg,
                    "benchmark": benchmark,
                    "seed": seed,
                }

                for metric in METRICS:
                    row[f"delta_{metric}"] = (
                        float(dyn.loc[seed, metric])
                        - float(base.loc[seed, metric])
                    )

                paired_rows.append(row)

    paired = pd.DataFrame(paired_rows)

    paired.to_csv(
        out / "paired_seed_differences.csv",
        index=False,
    )

    summary = (
        paired
        .groupby(["algorithm", "benchmark"])
        .agg(
            n=("seed", "count"),
            mean_delta_multiple=(
                "delta_final_multiple", "mean"
            ),
            std_delta_multiple=(
                "delta_final_multiple", "std"
            ),
            mean_delta_sharpe=(
                "delta_sharpe", "mean"
            ),
            std_delta_sharpe=(
                "delta_sharpe", "std"
            ),
            positive_multiple_count=(
                "delta_final_multiple",
                lambda x: int((x > 0).sum()),
            ),
            positive_sharpe_count=(
                "delta_sharpe",
                lambda x: int((x > 0).sum()),
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "paired_comparison_summary.csv",
        index=False,
    )

    print("\n" + "=" * 85)
    print("DYNAMIC KG MINUS BASELINE: PAIRED SEED DIFFERENCES")
    print("=" * 85)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:+.6f}",
        )
    )

    return stats, summary


def read_seed(seed, seed_dir):
    rows = []

    for group in GROUPS:
        summary_path = (
            seed_dir / group
            / "all_algorithms_test_summary.csv"
        )

        if not summary_path.exists():
            return None

        df = pd.read_csv(summary_path)

        if (
            len(df) != len(ALGORITHMS)
            or set(df["algorithm"]) != set(ALGORITHMS)
            or set(df["group"]) != {group}
        ):
            raise RuntimeError(
                f"Invalid result file: {summary_path}"
            )

        for _, record in df.iterrows():
            row = record.to_dict()
            row["seed"] = seed

            for metric in METRICS:
                if metric not in row:
                    raise RuntimeError(
                        f"Missing metric {metric}: {summary_path}"
                    )

                if not np.isfinite(float(row[metric])):
                    raise RuntimeError(
                        f"Nonfinite metric {metric}: {summary_path}"
                    )

            rows.append(row)

    if len(rows) != 9:
        raise RuntimeError(
            f"Expected 9 models for seed {seed}"
        )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=80000,
    )

    parser.add_argument(
        "--out",
        default="results/unified_dynamic_kgt_multiseed_v1",
    )

    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if len(args.seeds) != len(set(args.seeds)):
        raise RuntimeError("Duplicate seeds")

    all_rows = []

    for seed in args.seeds:
        seed_dir = out / f"seed_{seed}"

        print("\n" + "#" * 85)
        print(f"SEED {seed}")
        print("#" * 85, flush=True)

        existing = read_seed(seed, seed_dir)

        if existing is not None:
            print(
                "All nine model results already exist.",
                "Reusing completed seed.",
                flush=True,
            )

            all_rows.extend(existing)
            continue

        command = [
            sys.executable,
            str(RUNNER),
            "--groups", *GROUPS,
            "--algorithms", *ALGORITHMS,
            "--timesteps", str(args.timesteps),
            "--seed-offset", str(seed),
            "--out", str(seed_dir),
        ]

        print("Command:", " ".join(command), flush=True)

        log_path = out / f"seed_{seed}.log"

        with open(
            log_path,
            "w",
            encoding="utf-8",
        ) as log:
            proc = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            returncode = proc.wait()

        if returncode != 0:
            raise RuntimeError(
                f"Seed {seed} failed with exit code "
                f"{returncode}. Log: {log_path}"
            )

        current = read_seed(seed, seed_dir)

        if current is None:
            raise RuntimeError(
                f"Seed {seed} completed but the complete "
                "nine-model summary is missing."
            )

        all_rows.extend(current)

        summarize(
            pd.DataFrame(all_rows),
            out,
        )

    all_df = pd.DataFrame(all_rows)

    expected = len(args.seeds) * 9

    if len(all_df) != expected:
        raise RuntimeError(
            f"Expected {expected} model results, "
            f"received {len(all_df)}"
        )

    stats, paired = summarize(all_df, out)

    metadata = {
        "seeds": args.seeds,
        "groups": GROUPS,
        "algorithms": ALGORITHMS,
        "num_models": len(all_df),
        "timesteps_per_model": args.timesteps,
        "train": "2018-2021",
        "validation": "2022",
        "test": "2023",
        "test_used_for_model_selection": False,
        "comparison": "paired_same_seed_same_algorithm",
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 85)
    print("DYNAMIC KG MULTI-SEED EXPERIMENT COMPLETED")
    print("=" * 85)
    print("Models:", len(all_df))
    print("Output:", out)


if __name__ == "__main__":
    main()
