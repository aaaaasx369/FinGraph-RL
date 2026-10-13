#!/usr/bin/env python3
"""
V21 — Dynamic KG Ablation with Same-Dimension Placebo

Inputs:
    outputs/v20/fusion/dataset_conservative.csv
    outputs/v20/fusion/dataset_inclusive.csv
    outputs/v20/fusion/feature_sets.json

Conditions:
    CLEAN
    FULL_STATIC
    FULL_DYNAMIC_PLACEBO
    FULL_DYNAMIC

Split:
    Train: 2018-2021
    Val:   2022
    Test:  2023

Models:
    A2C, PPO, DDPG

Reuses the original PortfolioEnv, model builder,
and evaluator from V18 / sixway runner.

No modifications to previous experiments.
Research status: EXPLORATORY.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TIERS = ["conservative", "inclusive"]
AGENTS = ["A2C", "PPO", "DDPG"]

CONDITIONS = [
    "CLEAN",
    "FULL_STATIC",
    "FULL_DYNAMIC_PLACEBO",
    "FULL_DYNAMIC",
]

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

SPLITS = {
    "train": ("2018-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2023-12-31"),
}


def load_module(path, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(path.name + ".tmp")

    tmp.write_text(
        json.dumps(
            obj, indent=2,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_helpers():
    v18_path = (
        ROOT / "scripts/run_v18_frkg_ablation.py"
    )
    v18 = load_module(
        v18_path, "v21_v18_helpers"
    )

    original, tickers = v18.load_original_runner()

    if set(tickers) != set(TICKERS):
        raise RuntimeError(
            "Original PortfolioEnv ticker universe changed"
        )

    return v18, original, tickers


def feature_schema(path):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    for name in ["CLEAN", "FULL_STATIC", "FULL_DYNAMIC"]:
        if name not in data:
            raise ValueError(
                f"Missing feature set {name}"
            )

    static = list(data["FULL_STATIC"])
    dynamic_full = list(data["FULL_DYNAMIC"])
    clean = list(data["CLEAN"])

    if dynamic_full[:len(static)] != static:
        raise ValueError(
            "FULL_DYNAMIC must start with FULL_STATIC"
        )

    dynamic = dynamic_full[len(static):]

    if len(clean) != 9:
        raise ValueError(
            f"Expected CLEAN=9, got {len(clean)}"
        )

    if len(static) != 42:
        raise ValueError(
            f"Expected STATIC=42, got {len(static)}"
        )

    if len(dynamic) != 11:
        raise ValueError(
            f"Expected DYNAMIC extra=11, got {len(dynamic)}"
        )

    if len(dynamic_full) != 53:
        raise ValueError(
            "Expected FULL_DYNAMIC=53"
        )

    for name, cols in data.items():
        if len(cols) != len(set(cols)):
            raise ValueError(
                f"Duplicated features: {name}"
            )

    return {
        "CLEAN": clean,
        "FULL_STATIC": static,
        "FULL_DYNAMIC_PLACEBO": (
            static + [
                "placebo_" + c for c in dynamic
            ]
        ),
        "FULL_DYNAMIC": dynamic_full,
    }, dynamic


def load_data(path, tickers):
    df = pd.read_csv(path)

    essential = [
        "date", "ticker", "split",
        "market_close",
        "kg_mask_sec",
        "kg_mask_findkg",
        "frkg_fresh",
        "frkg_stale",
        "frkg_missing",
    ]
    missing = [
        c for c in essential
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{path}: missing {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
    )

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{path}: duplicate date/ticker"
        )

    for split, (start, end) in SPLITS.items():
        rows = df[df["split"].eq(split)]
        if rows.empty:
            raise ValueError(
                f"{path}: empty {split}"
            )
        if not rows["date"].between(
            start, end
        ).all():
            raise ValueError(
                f"{path}: invalid {split} dates"
            )

    if not df["split"].isin(SPLITS).all():
        raise ValueError(
            f"{path}: unexpected split labels"
        )

    for c in [
        "kg_mask_sec",
        "kg_mask_findkg",
        "frkg_fresh",
        "frkg_stale",
        "frkg_missing",
    ]:
        if not df[c].isin([0, 1]).all():
            raise ValueError(
                f"{path}: invalid mask {c}"
            )

    if not (
        df[
            ["frkg_fresh", "frkg_stale", "frkg_missing"]
        ].sum(axis=1).eq(1)
    ).all():
        raise ValueError(
            "ReflectKG masks not exclusive"
        )

    # Source dates must precede observations.
    date_specs = [
        ("kg_mask_sec", "sec_source_available_at"),
        ("frkg_fresh", "available_at"),
    ]
    for mask, date_col in date_specs:
        if date_col not in df.columns:
            raise ValueError(
                f"Missing source date {date_col}"
            )

        available = pd.to_datetime(
            df[date_col], errors="coerce"
        ).dt.normalize()

        observed = df[mask].eq(1)
        bad = observed & (
            available.isna()
            | (available >= df["date"])
        )
        if bad.any():
            raise ValueError(
                f"PIT date gate failed: {date_col}"
            )

    if (
        (pd.to_numeric(
            df["market_close"], errors="raise"
        ) <= 0)
    ).any():
        raise ValueError("Invalid market_close")

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if set(df["ticker"]) != set(tickers):
        raise ValueError(
            "Unexpected ticker universe"
        )

    # Every date must contain all five stocks.
    for d, g in df.groupby("date"):
        if set(g["ticker"]) != set(tickers):
            raise ValueError(
                f"Incomplete panel: {d}"
            )

    return df


def prepare_conditions(raw, schema, dynamic):
    output = {}

    for name in CONDITIONS:
        df = raw.copy()

        if name == "FULL_DYNAMIC_PLACEBO":
            for c in dynamic:
                df["placebo_" + c] = 0.0

        cols = schema[name]

        if any(c not in df.columns for c in cols):
            missing = [
                c for c in cols if c not in df
            ]
            raise ValueError(
                f"{name} missing {missing}"
            )

        matrix = df[cols].to_numpy(
            dtype=np.float32
        )
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"{name}: nonfinite observations"
            )

        output[name] = df

    # Exact same features in static vs placebo prefix.
    a = output["FULL_STATIC"][
        schema["FULL_STATIC"]
    ].to_numpy()

    b = output["FULL_DYNAMIC_PLACEBO"][
        schema["FULL_STATIC"]
    ].to_numpy()

    if not np.array_equal(a, b):
        raise ValueError(
            "Placebo altered static features"
        )

    extras = [
        "placebo_" + c for c in dynamic
    ]
    if not (
        output["FULL_DYNAMIC_PLACEBO"][extras]
        .to_numpy() == 0
    ).all():
        raise ValueError(
            "Placebo features are not zero"
        )

    return output


def new_environment(v18, original, df, cols, args):
    return v18.create_env(
        original, df, cols, args
    )


def verify_conditions(
    v18, original, frames, schema, args
):
    report = {}

    for tier in args.tiers:
        for condition in CONDITIONS:
            df = frames[tier][condition]

            train = df[
                df["split"].eq("train")
            ].copy()

            key = f"{tier}_{condition}"

            report[key] = v18.env_smoke(
                original,
                train,
                schema[condition],
                args,
            )

            print(
                "CHECK",
                key,
                report[key],
                flush=True
            )

    return report


def metric_to_native(metrics):
    out = {}
    for k, v in metrics.items():
        if isinstance(v, np.generic):
            v = v.item()
        elif isinstance(v, np.ndarray):
            if v.size != 1:
                raise ValueError(
                    f"Unexpected metric array: {k}"
                )
            v = v.item()

        if isinstance(v, (float, int)):
            if not np.isfinite(v):
                raise ValueError(
                    f"Nonfinite metric {k}"
                )
        out[k] = v
    return out


def save_dataframe(obj, path):
    if hasattr(obj, "to_csv"):
        obj.to_csv(path, index=False)
    else:
        pd.DataFrame(obj).to_csv(
            path, index=False
        )


def train_one(
    args, v18, original, tier,
    condition, agent, seed,
    df, features, output_dir,
    manifest_hash,
):
    key = (
        f"{tier}__{condition}"
        f"__{agent}__seed{seed}"
    )

    run_dir = output_dir / "runs" / key
    status_path = run_dir / "result.json"

    if status_path.exists():
        stored = json.loads(
            status_path.read_text(
                encoding="utf-8"
            )
        )
        if stored.get("manifest_hash") != manifest_hash:
            raise RuntimeError(
                f"Config changed for {key}"
            )

        model_path = run_dir / "model.zip"
        required = [
            run_dir / "val_equity.csv",
            run_dir / "test_equity.csv",
            run_dir / "val_weights.csv",
            run_dir / "test_weights.csv",
            run_dir / "val_returns.csv",
            run_dir / "test_returns.csv",
        ]
        if model_path.exists() and all(
            p.exists() for p in required
        ):
            print(
                "SKIP completed:", key,
                flush=True
            )
            return stored["metrics"]

        raise RuntimeError(
            f"Incomplete saved run: {key}. "
            "Inspect it before retrying."
        )

    print(
        "\n" + "=" * 65,
        "\nTRAIN:", key,
        "\n" + "=" * 65,
        flush=True
    )

    run_dir.mkdir(
        parents=True, exist_ok=True
    )

    train = df[
        df["split"].eq("train")
    ].copy()
    val = df[
        df["split"].eq("val")
    ].copy()
    test = df[
        df["split"].eq("test")
    ].copy()

    original.set_seed(seed)

    vec_env = original.DummyVecEnv([
        lambda:
            new_environment(
                v18, original,
                train, features, args
            )
    ])

    try:
        model = original.build_model(
            agent=agent,
            env=vec_env,
            seed=seed,
            learning_rate=args.learning_rate,
            verbose=args.verbose,
        )

        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=False,
        )

        model.save(str(run_dir / "model"))

        collected = []

        for split, data in [
            ("val", val),
            ("test", test),
        ]:
            env = new_environment(
                v18, original,
                data, features, args
            )

            metrics, equity, weights, returns = (
                original.evaluate_model(
                    model, env
                )
            )

            metrics = metric_to_native(metrics)

            row = {
                "tier": tier,
                "condition": condition,
                "agent": agent,
                "seed": seed,
                "split": split,
                "timesteps": args.timesteps,
                "feature_dims": len(features),
                **metrics,
            }
            collected.append(row)

            save_dataframe(
                equity,
                run_dir / f"{split}_equity.csv",
            )
            save_dataframe(
                weights,
                run_dir / f"{split}_weights.csv",
            )
            save_dataframe(
                returns,
                run_dir / f"{split}_returns.csv",
            )

            print(
                split.upper(),
                "return=",
                round(
                    float(metrics["total_return"]), 4
                ),
                "sharpe=",
                round(
                    float(metrics["sharpe"]), 4
                ),
                flush=True,
            )

        atomic_json(status_path, {
            "run_id": key,
            "manifest_hash": manifest_hash,
            "metrics": collected,
        })

        return collected

    finally:
        vec_env.close()


def gather_completed(root):
    rows = []

    for path in sorted(
        (root / "runs").glob("*/result.json")
    ):
        info = json.loads(
            path.read_text(encoding="utf-8")
        )
        rows.extend(info["metrics"])

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def make_summary(df, root):
    if df.empty:
        return

    df.to_csv(
        root / "runs.csv",
        index=False
    )

    metrics = [
        "total_return",
        "sharpe",
        "max_drawdown",
    ]

    if "avg_turnover" in df.columns:
        metrics.append("avg_turnover")

    summary = (
        df.groupby(
            ["tier", "condition", "agent", "split"]
        )[metrics]
        .agg(["mean", "std", "count"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary.reset_index().to_csv(
        root / "summary.csv",
        index=False
    )

    # Paired comparisons preserve the same seed.
    comparisons = [
        ("FULL_STATIC", "CLEAN"),
        ("FULL_DYNAMIC_PLACEBO", "FULL_STATIC"),
        ("FULL_DYNAMIC", "FULL_STATIC"),
        ("FULL_DYNAMIC", "FULL_DYNAMIC_PLACEBO"),
    ]

    paired = []

    for tier in df["tier"].unique():
        for agent in df["agent"].unique():
            for split in df["split"].unique():
                block = df[
                    df["tier"].eq(tier)
                    & df["agent"].eq(agent)
                    & df["split"].eq(split)
                ]

                for candidate, baseline in comparisons:
                    a = block[
                        block["condition"].eq(candidate)
                    ].set_index("seed")

                    b = block[
                        block["condition"].eq(baseline)
                    ].set_index("seed")

                    common = a.index.intersection(b.index)

                    for seed in common:
                        record = {
                            "tier": tier,
                            "agent": agent,
                            "split": split,
                            "seed": int(seed),
                            "candidate": candidate,
                            "baseline": baseline,
                        }
                        for metric in metrics:
                            record[
                                metric + "_diff"
                            ] = (
                                float(a.loc[seed, metric])
                                - float(b.loc[seed, metric])
                            )
                        paired.append(record)

    if paired:
        paired_df = pd.DataFrame(paired)
        paired_df.to_csv(
            root / "paired_differences.csv",
            index=False
        )

        diff_cols = [
            c for c in paired_df.columns
            if c.endswith("_diff")
        ]

        ps = paired_df.groupby(
            [
                "tier", "agent", "split",
                "candidate", "baseline",
            ]
        )[diff_cols].agg(
            ["mean", "std", "count"]
        )

        ps.columns = [
            f"{m}_{s}" for m, s in ps.columns
        ]
        ps.reset_index().to_csv(
            root / "paired_summary.csv",
            index=False
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="outputs/v20/fusion",
    )
    parser.add_argument(
        "--output-dir",
        default="results/v21",
    )
    parser.add_argument(
        "--tiers", nargs="+",
        choices=TIERS,
        default=["conservative"],
    )
    parser.add_argument(
        "--agents", nargs="+",
        choices=AGENTS,
        default=AGENTS,
    )
    parser.add_argument(
        "--seeds", nargs="+",
        type=int,
        default=[0],
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        parser.error(
            "Timesteps must be positive"
        )

    if len(args.seeds) != len(set(args.seeds)):
        parser.error(
            "Repeated random seeds"
        )

    input_dir = (
        ROOT / args.input_dir
    ).resolve()
    output_dir = (
        ROOT / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    v18, original, tickers = load_helpers()

    schema_path = input_dir / "feature_sets.json"
    schema, dynamic = feature_schema(
        schema_path
    )

    files = {}
    prepared = {}

    reference_keys = None

    for tier in args.tiers:
        path = (
            input_dir / f"dataset_{tier}.csv"
        )
        files[tier] = path

        raw = load_data(
            path, tickers
        )

        keys = raw[
            ["date", "ticker", "split"]
        ].reset_index(drop=True)

        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            raise ValueError(
                "Tier market panels differ"
            )

        prepared[tier] = prepare_conditions(
            raw, schema, dynamic
        )

    print("\n=== V21 OBSERVATION AUDIT ===")

    checks = verify_conditions(
        v18, original,
        prepared, schema, args,
    )

    atomic_json(
        output_dir / "observation_audit.json",
        checks,
    )

    manifest = {
        "version": "V21",
        "research_status": "EXPLORATORY",
        "input_files": {
            tier: {
                "path": str(path),
                "sha256": file_hash(path),
            }
            for tier, path in files.items()
        },
        "schema_sha256": file_hash(schema_path),
        "original_trainer_sha256": file_hash(
            ROOT / "scripts/run_sixway_train2020.py"
        ),
        "v18_helper_sha256": file_hash(
            ROOT / "scripts/run_v18_frkg_ablation.py"
        ),
        "v21_script_sha256": file_hash(
            Path(__file__).resolve()
        ),
        "tiers": args.tiers,
        "conditions": schema,
        "agents": args.agents,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "transaction_cost": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "splits": SPLITS,
        "known_limitations": [
            "SEC keyword counts are not LightRAG",
            "FinDKG public historical availability unverified",
            "FinReflectKG PDF accession mapping unverified",
            "Trading-execution timing not independently audited",
            "Zero-placebo controls dimension, not full distribution",
            "Dynamic features are observed changes, not KG event truth",
        ],
    }

    # Check-only should not lock full training config.
    if args.check_only:
        atomic_json(
            output_dir / "check_manifest.json",
            manifest,
        )
        print("\n=== V21 CHECK PASSED ===")
        print("No models trained.")
        print("Output:", output_dir)
        return

    canonical = json.dumps(
        manifest,
        sort_keys=True,
        ensure_ascii=False,
    )
    manifest_hash = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    manifest["manifest_hash"] = manifest_hash

    saved_manifest = output_dir / "manifest.json"

    if saved_manifest.exists():
        previous = json.loads(
            saved_manifest.read_text(
                encoding="utf-8"
            )
        )
        if (
            previous.get("manifest_hash")
            != manifest_hash
        ):
            raise RuntimeError(
                "Output directory was created with "
                "different inputs or hyperparameters. "
                "Use another --output-dir."
            )
    else:
        atomic_json(
            saved_manifest, manifest
        )

    for tier in args.tiers:
        for condition in CONDITIONS:
            df = prepared[tier][condition]
            cols = schema[condition]

            for agent in args.agents:
                for seed in args.seeds:
                    train_one(
                        args,
                        v18,
                        original,
                        tier,
                        condition,
                        agent,
                        seed,
                        df,
                        cols,
                        output_dir,
                        manifest_hash,
                    )

                    completed = gather_completed(
                        output_dir
                    )
                    make_summary(
                        completed, output_dir
                    )

    print("\n=== V21 FINISHED ===")
    print("Runs:", output_dir / "runs.csv")
    print(
        "Paired:",
        output_dir / "paired_summary.csv"
    )
    print(
        "Strict PIT approval: NOT COMPLETE"
    )


if __name__ == "__main__":
    main()
