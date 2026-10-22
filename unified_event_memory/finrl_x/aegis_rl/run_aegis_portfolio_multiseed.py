#!/usr/bin/env python3

"""
AEGIS-RL Portfolio Multi-seed Experiment

Train       2018-01-01 .. 2020-12-31
Validation  2021-01-01 .. 2021-12-31
Test        2022-01-01 .. 2023-12-31

Feature groups:
    clean
    kg_only
    sec_only
    kg_sec
    active

Models:
    A2C
    PPO
    DDPG

Uses existing FinRL:
    StockPortfolioEnv
    build_market_data
    make_env_kwargs
    evaluate_model
    calculate_metrics

Saves every completed model and evaluation.

Important:
    Integration/backtest experiment, not yet
    a fully point-in-time-certified evaluation.
"""

import os
import sys
import json
import random
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
)

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(ROOT))

from finrl.meta.env_portfolio_allocation.env_portfolio import (
    StockPortfolioEnv,
)

# Existing experiment contains top-level argparse.
# Hide our runner arguments during import.
_original_argv = sys.argv[:]

try:
    sys.argv = ["multi_drl_rag_kg_compare.py"]
    import experiments.multi_drl_rag_kg_compare as exp
finally:
    sys.argv = _original_argv


# ============================================================
# CONSTANTS
# ============================================================

GROUPS = [
    "clean",
    "kg_only",
    "sec_only",
    "kg_sec",
    "active",
]

ALGORITHMS = [
    "A2C",
    "PPO",
    "DDPG",
]

SEEDS = [
    42,
    123,
    2026,
    7,
    999,
]

DATE_SPLITS = {
    "train": (
        "2018-01-01",
        "2021-01-01",
    ),
    "validation": (
        "2021-01-01",
        "2022-01-01",
    ),
    "test": (
        "2022-01-01",
        "2024-01-01",
    ),
}

FUSION_ROOT = (
    ROOT
    / "results/aegis_rl_v1/portfolio_fusion_v1"
)

DEFAULT_OUT = (
    ROOT
    / "results/aegis_rl_v1/portfolio_multiseed_v1"
)

KG_COLS = [
    f"kg_{i:03d}"
    for i in range(200)
]

STRUCTURAL_COLS = [
    "hist_kg_count",
    "hist_kg_relations",
]

SEC_COLS = [
    "hist_sec_chunks",
    "hist_sec_documents",
    "hist_sec_signals",
    "hist_impact",
    "hist_risk",
    "hist_confidence",
    "hist_text_growth",
    "hist_text_risk",
    "hist_text_supply",
    "hist_text_competition",
]

EXTRA_COLS = (
    STRUCTURAL_COLS
    + SEC_COLS
)

ALL_KNOWLEDGE_COLS = (
    EXTRA_COLS
    + KG_COLS
)

ACTIONS = [
    "SKIP",
    "KG",
    "SEC",
    "BOTH",
]

ACTION_COSTS = {
    "SKIP": 0.0,
    "KG": 0.03,
    "SEC": 0.08,
    "BOTH": 0.11,
}

METRICS = [
    "final_multiple",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "max_drawdown",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configuration_hash(config):

    payload = json.dumps(
        config,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(
        payload
    ).hexdigest()[:16]


# ============================================================
# TRAINABLE KNOWLEDGE PROJECTION
#
# Keep original covariance + technical features.
#
# Input:
#   [batch, observation_rows, 5 stocks]
#
# Knowledge:
#   200 KG embedding rows
#
# Output:
#   flattened base features
#   + projected KG features
#
# Identical architecture across ablations.
# ============================================================

class KnowledgeProjectionExtractor(
    BaseFeaturesExtractor
):

    def __init__(
        self,
        observation_space,
        kg_dim=200,
        projected_dim=16,
        stock_dim=5,
    ):

        shape = observation_space.shape

        if len(shape) != 2:
            raise RuntimeError(
                f"Expected 2D observations: {shape}"
            )

        n_rows, n_stocks = shape

        if n_stocks != stock_dim:
            raise RuntimeError(
                f"Wrong stock dimension: {shape}"
            )

        if n_rows <= kg_dim:
            raise RuntimeError(
                f"Missing KG feature block: {shape}"
            )

        self.kg_dim = kg_dim

        self.base_rows = (
            n_rows - kg_dim
        )

        self.stock_dim = stock_dim

        output_dim = (
            self.base_rows * stock_dim
            + projected_dim * stock_dim
        )

        super().__init__(
            observation_space,
            features_dim=output_dim,
        )

        self.projector = nn.Sequential(
            nn.Linear(kg_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, projected_dim),
        )

    def forward(self, observations):

        if observations.ndim != 3:
            raise RuntimeError(
                f"Unexpected batch: {observations.shape}"
            )

        base = observations[
            :,
            :self.base_rows,
            :,
        ]

        kg = observations[
            :,
            self.base_rows:,
            :,
        ]

        if kg.shape[1] != self.kg_dim:
            raise RuntimeError(
                "KG dimension mismatch"
            )

        kg = kg.permute(
            0, 2, 1
        )

        projected = self.projector(
            kg
        )

        return torch.cat(
            [
                base.flatten(start_dim=1),
                projected.flatten(start_dim=1),
            ],
            dim=1,
        )


# ============================================================
# DATA LOADING
# ============================================================

def load_fusion_file(group):

    path = (
        FUSION_ROOT
        / f"{group}.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing fusion file: {path}"
        )

    df = pd.read_csv(path)

    required = {
        "date",
        "tic",
        *ALL_KNOWLEDGE_COLS,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{group}: missing columns "
            f"{sorted(missing)}"
        )

    df["date"] = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y-%m-%d")
    )

    df["tic"] = (
        df["tic"]
        .astype(str)
        .str.upper()
    )

    if df.duplicated(
        ["date", "tic"]
    ).any():
        raise RuntimeError(
            f"{group}: duplicate date/tic"
        )

    values = df[
        ALL_KNOWLEDGE_COLS
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{group}: nonfinite knowledge values"
        )

    return df


def build_common_market():

    print("\n===== BUILD MARKET =====", flush=True)

    market = exp.build_market_data()

    if market.empty:
        raise RuntimeError(
            "Market data empty"
        )

    market["date"] = (
        pd.to_datetime(
            market["date"]
        ).dt.strftime("%Y-%m-%d")
    )

    market["tic"] = (
        market["tic"]
        .astype(str)
        .str.upper()
    )

    market = market[
        (market.date >= "2018-01-01")
        & (market.date < "2024-01-01")
    ].copy()

    market = market.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    if market.duplicated(
        ["date", "tic"]
    ).any():
        raise RuntimeError(
            "Duplicate market keys"
        )

    exp.validate_panel(
        market,
        "MARKET",
    )

    print(
        "Market rows:",
        len(market),
        flush=True,
    )

    print(
        "Market dates:",
        market.date.nunique(),
        flush=True,
    )

    return market


# ============================================================
# ATTACH EXTERNAL FEATURES
# ============================================================

def merge_features(
    market,
    external,
    group,
):

    # Do not merge duplicate technical columns.
    external = external[
        ["date", "tic"]
        + ALL_KNOWLEDGE_COLS
    ].copy()

    out = market.merge(
        external,
        on=["date", "tic"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing = (
        out["_merge"] != "both"
    )

    if missing.any():

        print(
            out.loc[
                missing,
                ["date", "tic"]
            ].head(10).to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"{group}: "
            f"{int(missing.sum())} missing feature rows"
        )

    out = out.drop(
        columns="_merge"
    )

    return out


def normalize_knowledge(
    panel,
    group,
):

    """
    Train-only scale normalization.

    Divide by training standard deviation.
    Do not subtract the mean, so 0 stays 0.

    This preserves Active SKIP masking.
    """

    df = panel.copy()

    train_mask = (
        df["date"] < "2021-01-01"
    )

    if not train_mask.any():
        raise RuntimeError(
            "No normalization training rows"
        )

    scales = {}

    for col in ALL_KNOWLEDGE_COLS:

        x = df.loc[
            train_mask,
            col,
        ].astype(float)

        scale = float(
            x.std(ddof=0)
        )

        if (
            not np.isfinite(scale)
            or scale < 1e-8
        ):
            scale = 1.0

        df[col] = (
            df[col]
            .astype(float)
            / scale
        ).clip(-5, 5)

        scales[col] = scale

    values = df[
        ALL_KNOWLEDGE_COLS
    ].to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{group}: normalization failed"
        )

    return df, scales


def build_panels(market, selected_groups):

    panels = {}
    normalizers = {}

    expected_keys = market[
        ["date", "tic"]
    ].reset_index(drop=True)

    # All groups must use exactly the same days
    # and exactly the same five stocks.
    for group in selected_groups:

        print(
            f"\n===== PREPARE {group} =====",
            flush=True,
        )

        external = load_fusion_file(
            group
        )

        panel = merge_features(
            market,
            external,
            group,
        )

        panel, scales = normalize_knowledge(
            panel,
            group,
        )

        panel = panel.sort_values(
            ["date", "tic"]
        ).reset_index(drop=True)

        actual_keys = panel[
            ["date", "tic"]
        ].reset_index(drop=True)

        if not actual_keys.equals(
            expected_keys
        ):
            raise RuntimeError(
                f"{group}: market alignment failed"
            )

        exp.validate_panel(
            panel,
            group,
        )

        panels[group] = panel
        normalizers[group] = scales

        print(
            group,
            "rows:",
            len(panel),
            "knowledge cols:",
            len(ALL_KNOWLEDGE_COLS),
            flush=True,
        )

    return panels, normalizers


# ============================================================
# DATE SPLITS
# ============================================================

def slice_period(df, period):

    start, end = DATE_SPLITS[
        period
    ]

    out = df[
        (df.date >= start)
        & (df.date < end)
    ].copy()

    if out.empty:
        raise RuntimeError(
            f"Empty {period}"
        )

    out = out.sort_values(
        ["date", "tic"]
    )

    # FinRL StockPortfolioEnv accesses
    # the panel through integer day indices.
    dates = sorted(
        out.date.unique()
    )

    index_map = {
        date: i
        for i, date in enumerate(dates)
    }

    out.index = out.date.map(
        index_map
    ).astype(int)

    return out


def verify_splits(train, val, test):

    dates = [
        set(x.date.unique())
        for x in (train, val, test)
    ]

    if (
        dates[0] & dates[1]
        or dates[0] & dates[2]
        or dates[1] & dates[2]
    ):
        raise RuntimeError(
            "Overlapping temporal splits"
        )

    print(
        "Train:",
        train.date.min(),
        train.date.max(),
        train.date.nunique(),
    )

    print(
        "Validation:",
        val.date.min(),
        val.date.max(),
        val.date.nunique(),
    )

    print(
        "Test:",
        test.date.min(),
        test.date.max(),
        test.date.nunique(),
    )


# ============================================================
# ENVIRONMENT
# ============================================================

def make_environment(df, features):

    return StockPortfolioEnv(
        df=df,
        **exp.make_env_kwargs(
            features
        ),
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    algorithm,
    train_df,
    features,
    seed,
    timesteps,
    device,
):

    set_seed(seed)

    env = make_environment(
        train_df,
        features,
    )

    vec_env, _ = env.get_sb_env()

    policy_kwargs = {
        "features_extractor_class":
            KnowledgeProjectionExtractor,

        "features_extractor_kwargs": {
            "kg_dim": 200,
            "projected_dim": 16,
            "stock_dim": 5,
        },
    }

    kwargs = {
        "policy_kwargs": policy_kwargs,
        "device": device,
        "seed": seed,
        "verbose": 0,
    }

    if algorithm == "A2C":

        model_class = A2C

        kwargs.update({
            "n_steps": 1024,
            "ent_coef": 0.005,
            "learning_rate": 0.0002,
        })

    elif algorithm == "PPO":

        model_class = PPO

        kwargs.update({
            "n_steps": 2048,
            "batch_size": 1024,
            "ent_coef": 0.005,
            "learning_rate": 0.0001,
        })

    elif algorithm == "DDPG":

        model_class = DDPG

        kwargs.update({
            "batch_size": 1024,
            "buffer_size": 100000,
            "learning_rate": 0.001,
        })

    else:
        raise ValueError(
            algorithm
        )

    model = model_class(
        "MlpPolicy",
        vec_env,
        **kwargs,
    )

    model.learn(
        total_timesteps=timesteps,
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    panel,
    features,
):

    returns, weights = (
        exp.evaluate_model(
            model,
            panel,
            features,
        )
    )

    metrics = exp.calculate_metrics(
        returns
    )

    return (
        returns,
        weights,
        metrics,
    )


def evaluate_period(
    model,
    panel,
    features,
    out,
    label,
):

    returns, weights, metrics = evaluate(
        model,
        panel,
        features,
    )

    returns.to_csv(
        out / f"{label}_returns.csv",
        index=False,
    )

    weights.to_csv(
        out / f"{label}_weights.csv"
    )

    return metrics


# ============================================================
# RESULTS AND RESUME
# ============================================================

def save_json(path, obj):

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def model_result_path(
    out,
    group,
    algorithm,
    seed,
):

    return (
        out
        / f"seed_{seed}"
        / group
        / algorithm.lower()
    )


def get_completed_result(path, config_hash):

    path = path / "result.json"

    if not path.exists():
        return None

    obj = json.loads(
        path.read_text()
    )

    if obj.get("config_hash") != config_hash:
        raise RuntimeError(
            f"Resume configuration mismatch: {path}"
        )

    return obj


def save_progress(out, rows):

    if rows:

        pd.DataFrame(rows).to_csv(
            out / "all_results.csv",
            index=False,
        )


# ============================================================
# SUMMARIES
# ============================================================

def create_summary(out, rows):

    df = pd.DataFrame(rows)

    if df.empty:
        return

    stats = (
        df.groupby(
            ["group", "algorithm"]
        )[METRICS]
        .agg(
            ["mean", "std", "count"]
        )
    )

    stats.columns = [
        f"{metric}_{agg}"
        for metric, agg in stats.columns
    ]

    stats = stats.reset_index()

    stats.to_csv(
        out / "summary_mean_std.csv",
        index=False,
    )

    paired = []

    for algorithm in sorted(
        df.algorithm.unique()
    ):

        sub = df[
            df.algorithm == algorithm
        ]

        clean = sub[
            sub.group == "clean"
        ].set_index("seed")

        for group in GROUPS:

            if group == "clean":
                continue

            target = sub[
                sub.group == group
            ].set_index("seed")

            common_seeds = sorted(
                set(clean.index)
                & set(target.index)
            )

            for seed in common_seeds:

                result = {
                    "algorithm": algorithm,
                    "group": group,
                    "baseline": "clean",
                    "seed": seed,
                }

                for metric in METRICS:

                    result[
                        f"delta_{metric}"
                    ] = (
                        float(
                            target.loc[
                                seed, metric
                            ]
                        )
                        -
                        float(
                            clean.loc[
                                seed, metric
                            ]
                        )
                    )

                paired.append(result)

    if paired:

        pd.DataFrame(paired).to_csv(
            out / "paired_vs_clean.csv",
            index=False,
        )

    print(
        "\n===== MULTI-SEED SUMMARY =====",
        flush=True,
    )

    print(
        stats.to_string(
            index=False
        ),
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--groups",
        nargs="+",
        choices=GROUPS,
        default=GROUPS,
    )

    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
    )

    parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    parser.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.timesteps <= 0:
        raise RuntimeError(
            "Timesteps must be positive"
        )

    out = Path(
        args.out
    ).resolve()

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.device == "auto":

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = args.device

    if (
        device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    print("=" * 80)
    print("AEGIS-RL PORTFOLIO MULTI-SEED")
    print("=" * 80)

    print("Device:", device)
    print("Groups:", args.groups)
    print("Algorithms:", args.algorithms)
    print("Seeds:", args.seeds)
    print("Timesteps:", args.timesteps)
    print("Output:", out)

    # Stable observation ordering:
    #
    # Covariance rows
    # + technical features
    # + structural/SEC features
    # + last 200 KG embedding features

    features = (
        list(exp.BASE_FEATURES)
        + EXTRA_COLS
        + KG_COLS
    )

    market = build_common_market()

    panels, normalizers = build_panels(
        market,
        args.groups,
    )

    save_json(
        out / "normalization.json",
        normalizers,
    )

    experiment_config = {
        "groups": args.groups,
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "splits": DATE_SPLITS,
        "features": features,
        "device": device,
        "fused_data_path": str(FUSION_ROOT),
        "initial_amount": exp.INITIAL_AMOUNT,
        "transaction_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "status": "integration_experiment",
        "pit_certified": False,
        "limitations": [
            "KGT and policy in-sample training-period states.",
            "SEC-derived signal provenance not independently verified.",
            "Feature fusion is historical aggregation.",
            "Not a full retrieved-evidence text encoder.",
            "Acquisition cost not part of StockPortfolioEnv reward.",
            "No attack/defense portfolio replay.",
        ],
    }

    config_hash = configuration_hash(
        experiment_config
    )

    experiment_config[
        "config_hash"
    ] = config_hash

    save_json(
        out / "experiment_config.json",
        experiment_config,
    )

    if args.prepare_only:

        print(
            "\nAEGIS DATA PREPARATION: PASS",
            flush=True,
        )

        return

    rows = []

    for seed in args.seeds:

        for group in args.groups:

            panel = panels[group]

            train = slice_period(
                panel,
                "train",
            )

            val = slice_period(
                panel,
                "validation",
            )

            test = slice_period(
                panel,
                "test",
            )

            verify_splits(
                train,
                val,
                test,
            )

            for algorithm in args.algorithms:

                run_out = model_result_path(
                    out,
                    group,
                    algorithm,
                    seed,
                )

                run_out.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                existing = get_completed_result(
                    run_out,
                    config_hash,
                )

                if existing is not None:

                    print(
                        "[RESUME]",
                        seed,
                        group,
                        algorithm,
                        flush=True,
                    )

                    rows.append(existing)

                    save_progress(
                        out,
                        rows,
                    )

                    continue

                print(
                    "\n" + "=" * 80,
                    flush=True,
                )

                print(
                    "TRAIN",
                    group,
                    algorithm,
                    "seed",
                    seed,
                    flush=True,
                )

                print(
                    "=" * 80,
                    flush=True,
                )

                model = train_model(
                    algorithm,
                    train,
                    features,
                    seed,
                    args.timesteps,
                    device,
                )

                model.save(
                    str(
                        run_out / "model"
                    )
                )

                val_metrics = evaluate_period(
                    model,
                    val,
                    features,
                    run_out,
                    "validation",
                )

                test_metrics = evaluate_period(
                    model,
                    test,
                    features,
                    run_out,
                    "test",
                )

                result = {
                    "seed": seed,
                    "group": group,
                    "algorithm": algorithm,
                    "config_hash": config_hash,
                    "validation_multiple":
                        val_metrics["final_multiple"],
                    "validation_sharpe":
                        val_metrics["sharpe"],
                    "validation_max_drawdown":
                        val_metrics["max_drawdown"],
                }

                for metric in METRICS:

                    result[metric] = (
                        test_metrics[metric]
                    )

                save_json(
                    run_out / "result.json",
                    result,
                )

                rows.append(result)

                save_progress(
                    out,
                    rows,
                )

                print(
                    "TEST:",
                    group,
                    algorithm,
                    "seed",
                    seed,
                    "multiple",
                    f"{result['final_multiple']:.6f}",
                    "sharpe",
                    f"{result['sharpe']:.6f}",
                    "MDD",
                    f"{result['max_drawdown']:.6f}",
                    flush=True,
                )

    create_summary(
        out,
        rows,
    )

    print(
        "\nAEGIS PORTFOLIO MULTISEED: PASS",
        flush=True,
    )


if __name__ == "__main__":
    main()
