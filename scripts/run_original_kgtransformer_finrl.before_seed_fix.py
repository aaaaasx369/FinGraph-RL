#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
V20 = ROOT / "outputs/v20/fusion"

RUNNER = (
    ROOT / "scripts/run_v21_dynamic_ablation.py"
)

sys.path.insert(0, str(ROOT / "scripts"))
import v36_finrl_kg_ablation as v36

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

FEATURES = [
    f"dkg_emb_{i:02d}" for i in range(11)
]


def fail(message):
    raise RuntimeError(
        "ORIGINAL KG FINRL STOP: " + str(message)
    )


def read_embeddings():
    path = BASE / "original_daily_embeddings.csv"
    meta_path = BASE / "original_embedding_manifest.json"

    if not path.exists() or not meta_path.exists():
        fail(
            "Run original KGTransformer exporter first"
        )

    meta = json.loads(meta_path.read_text())

    if (
        meta.get("model")
        != "Financial-TKG_KGTransformer"
    ):
        fail("Wrong embedding model")

    if not meta.get(
        "original_model_loaded_strictly"
    ):
        fail("Original model weights not verified")

    df = pd.read_csv(path)

    required = {"date", "tic"} | set(FEATURES)

    if not required.issubset(df.columns):
        fail("Missing embedding columns")

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    if df.duplicated(["date", "tic"]).any():
        fail("Duplicate ticker-day")

    if set(df["tic"]) != set(TICKERS):
        fail("Wrong trading universe")

    if not np.isfinite(
        df[FEATURES].to_numpy(dtype=float)
    ).all():
        fail("Nonfinite KG features")

    return df


def make_placebo(source):
    """
    Date-shuffle negative control, scoped by split.
    Shared date permutation across all tickers.
    """

    rng = np.random.default_rng(2026)
    result = source.copy()

    for begin, end in [
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", "2022-12-31"),
        ("2023-01-01", "2023-12-31"),
    ]:
        mask = result["date"].between(
            begin, end
        )

        dates = sorted(
            result.loc[mask, "date"].unique()
        )

        permutation = rng.permutation(
            len(dates)
        )

        for ticker in TICKERS:
            source_rows = (
                source[
                    source["tic"].eq(ticker)
                    & source["date"].isin(dates)
                ]
                .sort_values("date")
            )

            dest_index = (
                result[
                    result["tic"].eq(ticker)
                    & result["date"].isin(dates)
                ]
                .sort_values("date")
                .index
            )

            if len(source_rows) != len(dates):
                fail("Incomplete placebo source")

            result.loc[
                dest_index, FEATURES
            ] = source_rows[
                FEATURES
            ].to_numpy()[permutation]

    return result


def build():
    embeddings = read_embeddings()

    arms = {
        "ORIGINAL_KGTRANSFORMER": embeddings,
        "ORIGINAL_KG_PLACEBO": make_placebo(
            embeddings
        ),
    }

    feature_set_path = (
        V20 / "feature_sets.json"
    )

    config = json.loads(
        feature_set_path.read_text()
    )

    for arm, signal in arms.items():
        out = BASE / "inputs" / arm
        out.mkdir(
            parents=True, exist_ok=True
        )

        sig = signal.rename(
            columns={
                "date": "_date",
                "tic": "_tic",
            }
        )

        for tier in [
            "conservative",
            "inclusive",
        ]:
            original = pd.read_csv(
                V20 / f"dataset_{tier}.csv",
                low_memory=False,
            )

            date_col, ticker_col = (
                v36.identify_keys(original)
            )

            slots = v36.discover_dynamic_slots(
                config,
                set(original.columns),
                "",
            )

            if len(slots) != 11:
                fail("Expected 11 dynamic slots")

            frame = original.copy()

            frame["_date"] = pd.to_datetime(
                frame[date_col]
            ).dt.normalize()

            frame["_tic"] = (
                frame[ticker_col]
                .astype(str)
                .str.upper()
            )

            merged = frame.merge(
                sig[
                    ["_date", "_tic"] + FEATURES
                ],
                on=["_date", "_tic"],
                how="left",
                validate="one_to_one",
            )

            if len(merged) != len(original):
                fail("Market row count changed")

            if merged[
                FEATURES
            ].isna().any().any():
                fail("Missing embeddings")

            for target, feature in zip(
                slots, FEATURES
            ):
                merged[target] = (
                    merged[feature].astype(float)
                )

            output = merged[
                list(original.columns)
            ].copy()

            for column in original.columns:
                if column not in slots:
                    if not original[column].equals(
                        output[column]
                    ):
                        fail(
                            "Original field modified: "
                            + column
                        )

            output.to_csv(
                out / f"dataset_{tier}.csv",
                index=False,
            )

        (out / "feature_sets.json").write_text(
            feature_set_path.read_text()
        )

        print("FINRL INPUT READY:", arm)


def execute(args):
    for arm in [
        "ORIGINAL_KGTRANSFORMER",
        "ORIGINAL_KG_PLACEBO",
    ]:
        inp = BASE / "inputs" / arm

        if not (
            inp / "dataset_conservative.csv"
        ).exists():
            fail("Run build first")

        command = [
            sys.executable,
            str(RUNNER),
            "--input-dir",
            str(inp),
            "--tiers",
            "conservative",
            "inclusive",
            "--agents",
            "A2C",
            "PPO",
            "DDPG",
            "--seeds",
            args.seeds,
            "--timesteps",
            str(args.timesteps),
            "--output-dir",
            str(BASE / "runs" / arm),
        ]

        if args.command == "check":
            command.append("--check-only")

        print(
            "\nRUN:",
            " ".join(command),
            flush=True,
        )

        subprocess.run(
            command,
            cwd=ROOT,
            check=True,
        )


def report():
    for arm in [
        "ORIGINAL_KGTRANSFORMER",
        "ORIGINAL_KG_PLACEBO",
    ]:
        path = (
            BASE / "runs" / arm / "summary.csv"
        )

        if not path.exists():
            print(arm, "NOT RUN")
            continue

        df = pd.read_csv(path)

        selected = df[
            df["tier"].eq("conservative")
            & df["split"].eq("test")
            & df["condition"].isin(
                ["CLEAN", "FULL_DYNAMIC"]
            )
        ]

        print("\n===", arm, "===")

        print(
            selected[
                [
                    "agent",
                    "condition",
                    "total_return_mean",
                    "sharpe_mean",
                    "max_drawdown_mean",
                ]
            ].to_string(index=False)
        )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=[
            "build",
            "check",
            "finrl",
            "report",
        ],
    )

    p.add_argument(
        "--seeds",
        default="0",
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=2000,
    )

    args = p.parse_args()

    if args.command == "build":
        build()
    elif args.command in [
        "check",
        "finrl",
    ]:
        execute(args)
    else:
        report()


if __name__ == "__main__":
    main()
