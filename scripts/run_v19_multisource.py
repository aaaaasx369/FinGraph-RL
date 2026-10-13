#!/usr/bin/env python3
"""V19 multi-source FinRL adapter. Exploratory only."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_v18():
    path = ROOT / "scripts/run_v18_frkg_ablation.py"

    spec = importlib.util.spec_from_file_location(
        "v18_for_v19", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="outputs/v19")
    p.add_argument("--output-dir", default="results/v19_check")
    p.add_argument("--check-only", action="store_true")
    p.add_argument(
        "--tiers", nargs="+",
        default=["conservative", "inclusive"]
    )
    p.add_argument(
        "--conditions", nargs="+",
        default=["CLEAN", "SEC_ONLY", "FINDKG_ONLY",
                 "REFLECTKG_ONLY", "SEC_FINDKG",
                 "SEC_REFLECTKG", "FULL_STATIC",
                 "FULL_DYNAMIC"]
    )
    p.add_argument(
        "--transaction-cost", type=float, default=0.001
    )
    p.add_argument(
        "--reward-scaling", type=float, default=1.0
    )
    p.add_argument(
        "--learning-rate", type=float, default=3e-4
    )
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    v18 = load_v18()
    runner, tickers = v18.load_original_runner()

    input_dir = ROOT / args.input_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    feature_sets = json.loads(
        (input_dir / "feature_sets.json").read_text()
    )

    reports = {}

    for tier in args.tiers:
        path = input_dir / f"dataset_{tier}.csv"
        df = pd.read_csv(path)

        df["date"] = pd.to_datetime(
            df["date"], errors="raise"
        )
        df["ticker"] = df["ticker"].str.upper()

        v18.validate_balanced(df, tickers, tier)

        for condition in args.conditions:
            if condition not in feature_sets:
                raise ValueError(
                    "Unknown condition: " + condition
                )

            cols = feature_sets[condition]

            missing = [c for c in cols if c not in df]
            if missing:
                raise ValueError(
                    f"{tier}/{condition} missing {missing}"
                )

            features = df[cols].to_numpy(dtype=float)

            if not np.isfinite(features).all():
                raise ValueError(
                    f"{tier}/{condition}: nonfinite"
                )

            train = df.loc[
                df["split"] == "train"
            ].copy()

            key = f"{tier}_{condition}"

            reports[key] = v18.env_smoke(
                runner, train, cols, args
            )

            print(
                key,
                "=>",
                reports[key]
            )

    (out / "observation_audit.json").write_text(
        json.dumps(reports, indent=2)
    )

    print("\n=== V19 CHECK COMPLETE ===")
    print("Verified conditions:", len(reports))
    print("No models were trained.")
    print("Output:", out / "observation_audit.json")


if __name__ == "__main__":
    main()
