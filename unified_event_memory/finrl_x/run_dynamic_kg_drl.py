#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import data_split

# Reuse your existing experiment's market preparation,
# stock ordering, evaluation and performance metrics.
#
# Its argparse parser runs at import, therefore preserve
# and temporarily replace sys.argv.
saved_argv = sys.argv[:]
sys.argv = ["multi_drl_rag_kg_compare.py"]
import experiments.multi_drl_rag_kg_compare as exp
sys.argv = saved_argv

TICKERS = sorted(["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"])
KG_DIM = 200
KG_COLS = [f"kg_{i:03d}" for i in range(KG_DIM)]


class KGProjectionExtractor(BaseFeaturesExtractor):
    """
    Original StockPortfolioEnv observation:
        rows = covariance(5) + technical indicators + KG(200)
        columns = five stocks

    Learnable KG projection:
        [batch, 200, 5] -> [batch, 5, 16] -> [batch, 80]

    Original covariance and technical features are
    preserved without compression.
    """

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
                f"Expected 2D FinRL observation, got {shape}"
            )

        n_rows, n_stocks = shape

        if n_stocks != stock_dim:
            raise RuntimeError(
                f"Expected {stock_dim} stocks, got {shape}"
            )

        if n_rows <= kg_dim:
            raise RuntimeError(
                f"Observation does not contain KG rows: {shape}"
            )

        self.base_rows = n_rows - kg_dim
        self.kg_dim = kg_dim
        self.stock_dim = stock_dim
        self.projected_dim = projected_dim

        output_dim = (
            self.base_rows * stock_dim
            + stock_dim * projected_dim
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
                f"Bad observation batch: {observations.shape}"
            )

        base = observations[:, :self.base_rows, :]
        kg = observations[:, self.base_rows:, :]

        if kg.shape[1] != self.kg_dim:
            raise RuntimeError("KG feature dimension mismatch")

        kg = kg.permute(0, 2, 1)
        projected = self.projector(kg)

        return torch.cat(
            [
                base.flatten(start_dim=1),
                projected.flatten(start_dim=1),
            ],
            dim=1,
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_kg(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p)
    required = {"date", "tic", *KG_COLS}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{p}: missing columns {sorted(missing)}"
        )

    df = df[["date", "tic"] + KG_COLS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime(
        "%Y-%m-%d"
    )
    df["tic"] = df["tic"].astype(str).str.upper()

    if df.duplicated(["date", "tic"]).any():
        raise RuntimeError(f"Duplicate KG keys in {p}")

    arr = df[KG_COLS].to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        raise RuntimeError(f"Nonfinite KG embeddings: {p}")

    return df


def standardize_kg(panel):
    """
    Fit KG normalization on 2018-2021 training rows only.
    The same learned statistics are used for validation/test.
    """
    df = panel.copy()
    train = df["date"] < "2022-01-01"

    if not train.any():
        raise RuntimeError("No KG training rows")

    mu = df.loc[train, KG_COLS].mean()
    sigma = df.loc[train, KG_COLS].std(ddof=0)

    sigma = sigma.mask(
        ~np.isfinite(sigma) | (sigma < 1e-8),
        1.0,
    )

    df[KG_COLS] = (
        (df[KG_COLS] - mu) / sigma
    ).clip(-5.0, 5.0)

    arr = df[KG_COLS].to_numpy(dtype=np.float32)

    if not np.isfinite(arr).all():
        raise RuntimeError("Nonfinite normalized KG values")

    return df


def attach_kg(market, kg):
    # Force identical date/ticker keys.
    panel = market.copy()
    panel["date"] = pd.to_datetime(
        panel["date"]
    ).dt.strftime("%Y-%m-%d")
    panel["tic"] = panel["tic"].str.upper()

    result = panel.merge(
        kg,
        how="left",
        on=["date", "tic"],
        validate="one_to_one",
        indicator=True,
    )

    missing = result["_merge"] != "both"

    if missing.any():
        print(
            result.loc[
                missing, ["date", "tic"]
            ].head(10).to_string(index=False)
        )
        raise RuntimeError(
            f"KG dates unavailable: {int(missing.sum())} rows. "
            "Do not silently replace them with future features."
        )

    result = result.drop(columns="_merge")
    return standardize_kg(result)


def prepare_panel(df):
    out = df.copy()
    out["date"] = pd.to_datetime(
        out["date"]
    ).dt.strftime("%Y-%m-%d")

    out = out[
        (out["date"] >= "2018-01-01")
        & (out["date"] < "2024-01-01")
    ].copy()

    out = out.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    counts = out.groupby("date")["tic"].nunique()
    if not (counts == len(TICKERS)).all():
        raise RuntimeError("Incomplete portfolio stock panel")

    return out


def environment(df, features):
    return StockPortfolioEnv(
        df=df,
        **exp.make_env_kwargs(features),
    )


def train_model(
    model_name,
    train_df,
    features,
    use_kg,
    seed,
    timesteps,
):
    seed_everything(seed)

    e_train = environment(train_df, features)
    vec_env, _ = e_train.get_sb_env()
    agent = DRLAgent(env=vec_env)

    policy_kwargs = {}

    if use_kg:
        policy_kwargs = {
            "features_extractor_class": KGProjectionExtractor,
            "features_extractor_kwargs": {
                "kg_dim": KG_DIM,
                "projected_dim": 16,
                "stock_dim": len(TICKERS),
            },
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "A2C":
        kwargs = {
            "n_steps": 1024,
            "ent_coef": 0.005,
            "learning_rate": 0.0002,
            "device": device,
            
        }

    elif model_name == "PPO":
        kwargs = {
            "n_steps": 2048,
            "batch_size": 1024,
            "ent_coef": 0.005,
            "learning_rate": 0.0001,
            "device": device,
            
        }

    elif model_name == "DDPG":
        kwargs = {
            "batch_size": 1024,
            "buffer_size": 100000,
            "learning_rate": 0.001,
            "device": device,
            
        }

    else:
        raise ValueError(model_name)

    if use_kg:
        kwargs["policy_kwargs"] = policy_kwargs

    # Instantiate Stable-Baselines3 directly.
    # FinRL's DRLAgent.get_model() injects seed and
    # policy_kwargs internally, causing duplicate
    # keyword argument errors in this environment.
    from stable_baselines3 import A2C, PPO, DDPG

    model_classes = {
        "A2C": A2C,
        "PPO": PPO,
        "DDPG": DDPG,
    }

    model = model_classes[model_name](
        policy="MlpPolicy",
        env=vec_env,
        **kwargs,
    )

    model.learn(
        total_timesteps=timesteps,
        tb_log_name=(
            f"dynamic_kg_{model_name.lower()}"
        ),
    )

    return model


def evaluate(model, df, features):
    return exp.evaluate_model(model, df, features)


def run_one_group(
    group,
    panel,
    features,
    out,
    algorithms,
    seed,
    timesteps,
):
    out.mkdir(parents=True, exist_ok=True)

    train_df = data_split(
        panel, "2018-01-01", "2022-01-01"
    )
    val_df = data_split(
        panel, "2022-01-01", "2023-01-01"
    )
    test_df = data_split(
        panel, "2023-01-01", "2024-01-01"
    )

    for label, split in (
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ):
        if split.empty:
            raise RuntimeError(f"{group}: empty {label} split")

        print(
            group, label,
            split["date"].min(),
            "->",
            split["date"].max(),
            "days=", split["date"].nunique(),
        )

    results = []
    models = {}

    for algorithm in algorithms:
        print("\n" + "=" * 72)
        print("TRAIN", group, algorithm)
        print("=" * 72)

        alg_seed = {
            "A2C": 101,
            "PPO": 202,
            "DDPG": 303,
        }[algorithm] + seed

        model = train_model(
            algorithm,
            train_df,
            features,
            use_kg=(group != "clean"),
            seed=alg_seed,
            timesteps=timesteps,
        )

        model.save(
            str(out / f"{algorithm.lower()}_model")
        )

        val_returns, val_weights = evaluate(
            model, val_df, features
        )
        metrics = exp.calculate_metrics(val_returns)
        metrics["algorithm"] = algorithm
        results.append(metrics)
        models[algorithm] = model

        val_returns.to_csv(
            out / f"{algorithm.lower()}_val_returns.csv",
            index=False,
        )
        val_weights.to_csv(
            out / f"{algorithm.lower()}_val_weights.csv"
        )

        print(
            "VAL", group, algorithm,
            "multiple=", metrics["final_multiple"],
            "Sharpe=", metrics["sharpe"],
        )

    # ======================================================
    # Evaluate ALL trained algorithms on the 2023 test set.
    #
    # These are evaluation-only results. Model selection
    # still uses 2022 validation, not 2023.
    # ======================================================
    all_test_rows = []

    for algorithm in algorithms:
        test_model = models[algorithm]

        returns, weights = evaluate(
            test_model,
            test_df,
            features,
        )

        metrics = exp.calculate_metrics(returns)

        metrics["group"] = group
        metrics["algorithm"] = algorithm
        metrics["seed_offset"] = seed

        all_test_rows.append(metrics)

        returns.to_csv(
            out / f"{algorithm.lower()}_test_returns.csv",
            index=False,
        )

        weights.to_csv(
            out / f"{algorithm.lower()}_test_weights.csv"
        )

        print(
            "ALL-MODEL TEST:",
            group,
            algorithm,
            "multiple=",
            f"{metrics['final_multiple']:.6f}",
            "Sharpe=",
            f"{metrics['sharpe']:.6f}",
        )

    pd.DataFrame(all_test_rows).to_csv(
        out / "all_algorithms_test_summary.csv",
        index=False,
    )

    table = pd.DataFrame(results).sort_values(
        "final_multiple",
        ascending=False,
    )
    table.to_csv(out / "validation_comparison.csv", index=False)

    best_name = str(table.iloc[0]["algorithm"])
    best_model = models[best_name]

    print("BEST BY 2022 VALIDATION:", group, best_name)

    test_returns, test_weights = evaluate(
        best_model, test_df, features
    )

    test_metrics = exp.calculate_metrics(test_returns)
    test_metrics["group"] = group
    test_metrics["best_algorithm"] = best_name
    test_metrics["seed_offset"] = seed

    test_returns.to_csv(
        out / "best_model_test_returns.csv",
        index=False,
    )
    test_weights.to_csv(
        out / "best_model_test_weights.csv"
    )
    pd.DataFrame([test_metrics]).to_csv(
        out / "best_model_test_summary.csv",
        index=False,
    )

    print(
        "TEST", group, best_name,
        "multiple=", test_metrics["final_multiple"],
        "Sharpe=", test_metrics["sharpe"],
        "MDD=", test_metrics["max_drawdown"],
    )

    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embeddings",
        default="outputs/unified_kgt_finrl_x_v1"
    )
    parser.add_argument(
        "--out",
        default="results/unified_dynamic_kgt_finrl_x_v1"
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["clean", "static", "dynamic"],
        default=["clean", "static", "dynamic"],
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=["A2C", "PPO", "DDPG"],
        default=["A2C", "PPO", "DDPG"],
    )
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()

    if args.timesteps <= 0:
        raise RuntimeError("timesteps must be positive")

    embeddings = ROOT / args.embeddings
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("Building existing FinRL market data...")
    market = exp.build_market_data()
    market = prepare_panel(market)

    # Load exactly the same market panel for all arms.
    static_kg = load_kg(
        embeddings / "daily_static_kgt.csv"
    )
    dynamic_kg = load_kg(
        embeddings / "daily_dynamic_kgt.csv"
    )

    panels = {
        "clean": market,
        "static": attach_kg(market, static_kg),
        "dynamic": attach_kg(market, dynamic_kg),
    }

    # No KG feature for Clean. For Static/Dynamic,
    # identical feature list, observation size,
    # trainable projection structure.
    feature_lists = {
        "clean": list(exp.BASE_FEATURES),
        "static": list(exp.BASE_FEATURES) + KG_COLS,
        "dynamic": list(exp.BASE_FEATURES) + KG_COLS,
    }

    reference_dates = set(market["date"].unique())

    for name, panel in panels.items():
        assert set(panel["date"].unique()) == reference_dates
        exp.validate_panel(panel, name)

        print(
            name,
            "observation features:",
            len(feature_lists[name])
        )

    # Confirm dynamic embeddings actually vary over time
    # and static embeddings remain constant per ticker.
    for tic in TICKERS:
        st = static_kg[
            static_kg["tic"] == tic
        ][KG_COLS].to_numpy()

        dy = dynamic_kg[
            dynamic_kg["tic"] == tic
        ][KG_COLS].to_numpy()

        if not np.allclose(st, st[0], atol=1e-7):
            raise RuntimeError(
                f"Static KG unexpectedly changes: {tic}"
            )

        changed = np.any(
            np.abs(np.diff(dy, axis=0)) > 1e-8,
            axis=1,
        ).sum()

        print(tic, "dynamic changes:", int(changed))

    summary = []

    for group in args.groups:
        result = run_one_group(
            group,
            panels[group],
            feature_lists[group],
            out / group,
            args.algorithms,
            args.seed_offset,
            args.timesteps,
        )
        summary.append(result)

        # Persist after each completed group.
        pd.DataFrame(summary).to_csv(
            out / "all_groups_test_summary.csv",
            index=False,
        )

    meta = {
        "groups": args.groups,
        "algorithms": args.algorithms,
        "timesteps_per_model": args.timesteps,
        "seed_offset": args.seed_offset,
        "train": "2018-2021",
        "validation": "2022",
        "test": "2023",
        "kg_dim": KG_DIM,
        "projected_kg_dim_per_stock": 16,
        "num_stocks": len(TICKERS),
        "kg_projection_trainable": True,
        "sec_fusion_used": False,
        "strict_pit_certified": False,
    }
    (out / "experiment_metadata.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("CLEAN / STATIC KG / DYNAMIC KG EXPERIMENT COMPLETED")
    print("=" * 72)
    print(pd.DataFrame(summary).to_string(index=False))
    print("Output:", out)


if __name__ == "__main__":
    main()
