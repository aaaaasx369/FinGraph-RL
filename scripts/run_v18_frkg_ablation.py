#!/usr/bin/env python3
"""
V18 — FinReflectKG extraction-quality FinRL ablation.

Reuses:
    scripts/run_sixway_train2020.py
        PortfolioEnv
        build_model
        evaluate_model
        set_seed

Experiments:
    CLEAN_V18
    FRKG_CONSERVATIVE_V18
    FRKG_INCLUSIVE_V18

Date split:
    Train: 2018-01-01 .. 2021-12-31
    Val:   2022-01-01 .. 2022-12-31
    Test:  2023-01-01 .. 2023-12-31

Design:
- Five identical stocks/dates across all conditions
- Reuses V17 train-fitted KG normalization
- Technical indicators calculated using past prices only
- Technical scaling fitted on training rows only
- Same seed/hyperparameters across conditions
- Validation is diagnostic; test is held out
- Explicit KG availability and age masks
- No formal point-in-time certification
- Never modifies historical sixway results
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

SPLITS = {
    "train": ("2018-01-01", "2021-12-31"),
    "val":   ("2022-01-01", "2022-12-31"),
    "test":  ("2023-01-01", "2023-12-31"),
}

KG_FEATURES = [
    "frkg_triple_count",
    "frkg_unique_relation_count",
    "frkg_unique_target_count",
    "frkg_competition",
    "frkg_dependency",
    "frkg_disclosure",
    "frkg_exposure",
    "frkg_impact",
    "frkg_investment",
    "frkg_negative",
    "frkg_operations",
    "frkg_other",
    "frkg_partnership",
    "frkg_production",
    "frkg_supply",
]

KG_MASKS = [
    "frkg_fresh",
    "frkg_stale",
    "frkg_missing",
    "frkg_age_scaled",
]

TECH_FEATURES = [
    "tech_return_1d",
    "tech_return_5d",
    "tech_return_20d",
    "tech_vol_20d",
    "tech_ma_10_ratio",
    "tech_ma_30_ratio",
    "tech_volume_20_ratio",
    "tech_range_5d",
    "tech_momentum_20d",
]

CONDITIONS = {
    "CLEAN_V18": None,
    "FRKG_CONSERVATIVE_V18": "conservative",
    "FRKG_INCLUSIVE_V18": "inclusive",
}

AGENTS = ["A2C", "PPO", "DDPG"]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()


def load_original_runner():
    """
    Import original environment and training functions.
    The original script is not changed.
    """
    path = ROOT / "scripts/run_sixway_train2020.py"
    if not path.exists():
        raise FileNotFoundError(path)

    sys.path.insert(0, str(path.parent))

    spec = importlib.util.spec_from_file_location(
        "original_sixway_runner_v18", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    required = [
        "PortfolioEnv",
        "build_model",
        "evaluate_model",
        "set_seed",
        "DummyVecEnv",
    ]

    missing = [
        name for name in required
        if not hasattr(module, name)
    ]
    if missing:
        raise RuntimeError(
            "Original runner missing: " + str(missing)
        )

    original_tickers = list(
        getattr(module, "TICKERS", [])
    )
    if set(original_tickers) != set(TICKERS):
        raise RuntimeError(
            "Ticker mismatch with original runner: "
            + str(original_tickers)
        )

    return module, original_tickers


def read_market(path):
    df = pd.read_csv(path)
    required = [
        "date", "ticker", "open", "high",
        "low", "close", "volume",
    ]

    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(
            "Market missing columns: " + str(missing)
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()
    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
    )

    df = df.loc[
        df["ticker"].isin(TICKERS)
    ].copy()

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate market keys")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if not np.isfinite(
        df[["open", "high", "low", "close", "volume"]]
        .to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite market values")

    if (
        df[["open", "high", "low", "close"]] <= 0
    ).any().any():
        raise ValueError("Nonpositive market prices")

    return df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def build_past_only_technical(market):
    """
    All technical features are computed using prices
    strictly earlier than observation date.

    shift(1) avoids using same-day OHLCV.
    """
    blocks = []

    for ticker, g in market.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()

        close = g["close"].astype(float)
        high = g["high"].astype(float)
        low = g["low"].astype(float)
        volume = g["volume"].astype(float)

        prior_close = close.shift(1)
        daily_return = close.pct_change(fill_method=None)

        g["tech_return_1d"] = daily_return.shift(1)

        g["tech_return_5d"] = (
            close.shift(1) / close.shift(6) - 1
        )

        g["tech_return_20d"] = (
            close.shift(1) / close.shift(21) - 1
        )

        g["tech_vol_20d"] = (
            daily_return.shift(1)
            .rolling(20, min_periods=20)
            .std(ddof=0)
        )

        g["tech_ma_10_ratio"] = (
            prior_close
            / close.shift(1).rolling(
                10, min_periods=10
            ).mean()
            - 1
        )

        g["tech_ma_30_ratio"] = (
            prior_close
            / close.shift(1).rolling(
                30, min_periods=30
            ).mean()
            - 1
        )

        g["tech_volume_20_ratio"] = (
            volume.shift(1)
            / volume.shift(1).rolling(
                20, min_periods=20
            ).mean()
            - 1
        )

        historical_range = (
            high / low - 1
        ).shift(1)

        g["tech_range_5d"] = (
            historical_range.rolling(
                5, min_periods=5
            ).mean()
        )

        g["tech_momentum_20d"] = (
            close.shift(1)
            / close.shift(21)
            - 1
        )

        blocks.append(g)

    result = pd.concat(
        blocks, ignore_index=True
    )

    # Warm-up is removed from every experiment.
    # No future backfill is performed.
    result = result.dropna(
        subset=TECH_FEATURES
    ).copy()

    if not np.isfinite(
        result[TECH_FEATURES].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError(
            "Non-finite technical features"
        )

    # Compatibility with original PortfolioEnv:
    # it reads market_close, not close.
    result["market_close"] = result["close"].astype(float)

    return result


def fit_technical_scaler(df):
    train = df.loc[
        df["split"] == "train",
        TECH_FEATURES
    ]

    if train.empty:
        raise ValueError("No technical train samples")

    mean = train.mean()
    std = train.std(ddof=0).mask(
        lambda s: s <= 1e-12, 1.0
    )

    return {
        "fit_split": "train",
        "features": {
            c: {
                "mean": float(mean[c]),
                "std": float(std[c]),
            }
            for c in TECH_FEATURES
        }
    }


def transform_technical(df, scaler):
    df = df.copy()
    for c in TECH_FEATURES:
        settings = scaler["features"][c]
        df[c] = (
            df[c] - settings["mean"]
        ) / settings["std"]

    return df


def assign_split(df):
    df = df.copy()
    df["split"] = ""

    for name, (start, end) in SPLITS.items():
        mask = df["date"].between(
            pd.Timestamp(start),
            pd.Timestamp(end)
        )
        df.loc[mask, "split"] = name

    df = df.loc[
        df["split"] != ""
    ].copy()

    return df


def validate_balanced(df, tickers, label):
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{label}: duplicate date/ticker"
        )

    for d, g in df.groupby("date"):
        if set(g["ticker"]) != set(tickers):
            raise ValueError(
                f"{label}: incomplete ticker panel at {d}"
            )

    if len(df) != df["date"].nunique() * len(tickers):
        raise ValueError(
            f"{label}: unexpected panel size"
        )


def read_v17(path, tier):
    df = pd.read_csv(path)

    required = [
        "date", "ticker", "split",
        *KG_FEATURES, *KG_MASKS,
        "available_at",
        "source_file",
    ]

    missing = [c for c in required if c not in df]
    if missing:
        raise ValueError(
            f"{tier}: missing V17 columns {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["available_at"] = pd.to_datetime(
        df["available_at"], errors="coerce"
    )

    df["ticker"] = df["ticker"].str.upper()

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{tier}: duplicated V17 keys"
        )

    observed = df["frkg_fresh"] == 1

    if (
        df.loc[observed, "available_at"]
        >= df.loc[observed, "date"]
    ).any():
        raise ValueError(
            f"{tier}: FUTURE FILING LEAKAGE"
        )

    for c in KG_FEATURES + KG_MASKS:
        values = pd.to_numeric(
            df[c], errors="raise"
        )
        if not np.isfinite(
            values.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{tier}: non-finite {c}"
            )
        df[c] = values.astype(float)

    if not (
        df[
            ["frkg_fresh", "frkg_stale", "frkg_missing"]
        ].sum(axis=1) == 1
    ).all():
        raise ValueError(
            f"{tier}: invalid freshness masks"
        )

    # Prevent accidental model access to source metadata.
    return df[
        [
            "date", "ticker", "split",
            *KG_FEATURES, *KG_MASKS,
            "available_at", "source_file",
        ]
    ].copy()


def assemble_datasets(args, tickers):
    market = read_market(args.market)

    technical = build_past_only_technical(market)
    technical = assign_split(technical)

    # Keep five-stock dates synchronized.
    validate_balanced(
        technical, tickers, "technical"
    )

    scaler = fit_technical_scaler(technical)
    technical = transform_technical(
        technical, scaler
    )

    # Existing V17 data uses the same train/val/test years.
    tiers = {}
    for tier in ("conservative", "inclusive"):
        path = (
            Path(args.v17_root)
            / f"{tier}_normalized_all.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)

        tiers[tier] = read_v17(path, tier)

    left_keys = technical[
        ["date", "ticker", "split"]
    ].sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    datasets = {
        "CLEAN_V18": technical.copy()
    }

    for tier in ("conservative", "inclusive"):
        aligned = technical.merge(
            tiers[tier],
            on=["date", "ticker", "split"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        if not aligned["_merge"].eq("both").all():
            raise ValueError(
                f"{tier}: V17 missing dates after merge"
            )

        aligned = aligned.drop(
            columns="_merge"
        )

        datasets[
            "FRKG_" + tier.upper() + "_V18"
        ] = aligned

    # Ensure the exact same market-date panel.
    for name, df in datasets.items():
        actual = df[
            ["date", "ticker", "split"]
        ].sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        if not actual.equals(left_keys):
            raise ValueError(
                f"{name}: inconsistent experiment dates"
            )

        validate_balanced(df, tickers, name)

    feature_cols = {
        "CLEAN_V18": list(TECH_FEATURES),
        "FRKG_CONSERVATIVE_V18": (
            TECH_FEATURES + KG_FEATURES + KG_MASKS
        ),
        "FRKG_INCLUSIVE_V18": (
            TECH_FEATURES + KG_FEATURES + KG_MASKS
        ),
    }

    # IMPORTANT: pass only approved observation features.
    # Dates, source filenames, public approval flags,
    # raw filing ages, and test labels are excluded.
    for name, df in datasets.items():
        values = df[
            feature_cols[name]
        ].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError(
                f"{name}: non-finite observation features"
            )

    return datasets, feature_cols, scaler


def create_env(
    runner, frame, features, args
):
    return runner.PortfolioEnv(
        frame,
        feature_cols=features,
        transaction_cost_pct=args.transaction_cost,
        reward_scaling=args.reward_scaling,
        initial_amount=runner.INITIAL_AMOUNT,
    )


def env_smoke(runner, frame, features, args):
    """
    Verify observation dimension and one step before
    spending time on a real training run.
    """
    env = create_env(
        runner, frame, features, args
    )
    obs, _ = env.reset()

    array = np.asarray(obs, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(
            "Invalid PortfolioEnv observation"
        )

    expected_shape = env.observation_space.shape
    if array.shape != expected_shape:
        raise ValueError(
            "Observation dimension mismatch: "
            f"{array.shape} vs {expected_shape}"
        )

    action = env.action_space.sample()
    result = env.step(action)

    if len(result) != 5:
        raise ValueError(
            "PortfolioEnv must use Gymnasium step API"
        )

    next_obs, reward, terminated, truncated, info = result
    if not np.isfinite(
        np.asarray(next_obs, dtype=float)
    ).all():
        raise ValueError(
            "Non-finite next observation"
        )

    if not np.isfinite(float(reward)):
        raise ValueError("Non-finite reward")

    return {
        "observation_shape": list(array.shape),
        "feature_count_per_asset": len(features),
        "one_step_test": "PASS",
    }


def save_config(out, config):
    out.mkdir(parents=True, exist_ok=True)
    with (out / "config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            config, f, ensure_ascii=False, indent=2
        )


def run(args):
    runner, tickers = load_original_runner()

    datasets, feature_cols, tech_scaler = (
        assemble_datasets(args, tickers)
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "technical_train_scaler.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            tech_scaler, f, indent=2
        )

    config = {
        "version": "V18",
        "experiment": "EXPLORATORY_FRKG_ABLATION",
        "tickers": tickers,
        "splits": SPLITS,
        "conditions": feature_cols,
        "agents": args.agents,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "transaction_cost": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "input_market_sha256": file_hash(args.market),
        "input_v17_hashes": {
            tier: file_hash(
                Path(args.v17_root)
                / f"{tier}_normalized_all.csv"
            )
            for tier in ("conservative", "inclusive")
        },
        "external_public_availability_verified": False,
        "formal_finrl_approved": False,
        "notes": [
            "Original sixway runner reused, not modified",
            "V17 source dates not independently EDGAR-verified",
            "Validation year 2022, held-out test year 2023",
            "Technical indicators use lagged market data",
            "Technical scaler uses 2018-2021 train only",
            "KG scaler is reused from V17 train only",
            "Do not compare directly with old train2020 tests",
        ],
    }
    save_config(out, config)

    print("\n=== V18 DATA CHECK ===")

    for name, frame in datasets.items():
        counts = frame["split"].value_counts().to_dict()
        print(
            name,
            "rows=", len(frame),
            "features=", len(feature_cols[name]),
            "split_rows=", counts,
        )

    smoke_reports = {}

    for name, frame in datasets.items():
        train = frame.loc[
            frame["split"] == "train"
        ].copy()

        smoke_reports[name] = env_smoke(
            runner, train, feature_cols[name], args
        )

        print(
            name,
            "OBSERVATION:",
            smoke_reports[name]
        )

    with (out / "observation_audit.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            smoke_reports, f, indent=2
        )

    if args.check_only:
        print("\nCHECK-ONLY PASSED")
        print("No models trained.")
        print("Saved:", out)
        return

    from stable_baselines3 import (
        A2C, PPO, DDPG
    )

    model_class = {
        "A2C": A2C,
        "PPO": PPO,
        "DDPG": DDPG,
    }

    runs = []
    curve_dir = out / "equity_curves"
    model_dir = out / "models"
    curve_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)

    for condition in CONDITIONS:
        frame = datasets[condition]
        cols = feature_cols[condition]

        train_df = frame.loc[
            frame["split"] == "train"
        ].copy()
        val_df = frame.loc[
            frame["split"] == "val"
        ].copy()
        test_df = frame.loc[
            frame["split"] == "test"
        ].copy()

        for agent in args.agents:
            for seed in args.seeds:
                key = (
                    f"{condition}_{agent}_seed{seed}"
                )
                print("\n" + "=" * 72)
                print("RUN:", key)
                print("=" * 72)

                runner.set_seed(seed)

                vec_env = runner.DummyVecEnv([
                    lambda d=train_df, c=cols:
                        create_env(runner, d, c, args)
                ])

                model = runner.build_model(
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

                model.save(str(model_dir / key))

                for split, subset in [
                    ("val", val_df),
                    ("test", test_df),
                ]:
                    eval_env = create_env(
                        runner, subset, cols, args
                    )

                    (
                        metrics,
                        equity,
                        weights,
                        returns_df,
                    ) = runner.evaluate_model(
                        model, eval_env
                    )

                    result = {
                        "condition": condition,
                        "agent": agent,
                        "seed": seed,
                        "split": split,
                        "timesteps": args.timesteps,
                        "feature_dims": len(cols),
                        **metrics,
                    }
                    runs.append(result)

                    stem = f"{key}_{split}"
                    equity.to_csv(
                        curve_dir / f"{stem}_equity.csv",
                        index=False,
                    )
                    weights.to_csv(
                        curve_dir / f"{stem}_weights.csv",
                        index=False,
                    )
                    returns_df.to_csv(
                        curve_dir / f"{stem}_returns.csv",
                        index=False,
                    )

                    print(
                        f"{split}: "
                        f"return={metrics['total_return']:.4f} "
                        f"sharpe={metrics['sharpe']:.4f} "
                        f"mdd={metrics['max_drawdown']:.4f}"
                    )

                pd.DataFrame(runs).to_csv(
                    out / "runs.csv", index=False
                )

                vec_env.close()

    result_df = pd.DataFrame(runs)

    metric_names = [
        "final_value", "total_return",
        "annual_return", "annual_volatility",
        "sharpe", "max_drawdown", "avg_turnover",
    ]

    summary = (
        result_df.groupby(
            ["condition", "agent", "split"]
        )[metric_names]
        .agg(["mean", "std", "count"])
    )

    summary.columns = [
        f"{a}_{b}" for a, b in summary.columns
    ]
    summary = summary.reset_index()

    summary.to_csv(
        out / "summary.csv", index=False
    )

    print("\n=== V18 COMPLETE ===")
    print("Results:", out / "runs.csv")
    print("Summary:", out / "summary.csv")
    print(
        "External availability verification: NOT DONE"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--market",
        default="kg/data/market_5stocks_2017_2026.csv"
    )
    p.add_argument(
        "--v17-root", default="outputs/v17"
    )
    p.add_argument(
        "--output-dir", default="results/v18"
    )
    p.add_argument(
        "--agents", nargs="+",
        choices=AGENTS, default=AGENTS
    )
    p.add_argument(
        "--seeds", nargs="+", type=int,
        default=[0]
    )
    p.add_argument(
        "--timesteps", type=int,
        default=2000
    )
    p.add_argument(
        "--transaction-cost", type=float,
        default=0.001
    )
    p.add_argument(
        "--reward-scaling", type=float,
        default=1.0
    )
    p.add_argument(
        "--learning-rate", type=float,
        default=3e-4
    )
    p.add_argument(
        "--verbose", type=int,
        default=0
    )
    p.add_argument(
        "--check-only", action="store_true"
    )
    args = p.parse_args()

    if args.timesteps <= 0:
        p.error("--timesteps must be positive")
    if not 0 <= args.transaction_cost < 1:
        p.error("Invalid transaction cost")

    args.market = str(
        (ROOT / args.market).resolve()
    )
    args.v17_root = str(
        (ROOT / args.v17_root).resolve()
    )
    args.output_dir = str(
        (ROOT / args.output_dir).resolve()
    )

    run(args)


if __name__ == "__main__":
    main()
