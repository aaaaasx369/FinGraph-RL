#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V16: SEC-RAG + Dynamic KG-RAG + RL

Historical data:
  Train: 2018-2020
  Validation: 2021
  Test 2022-2023: not loaded

4 RL experiments:
  MARKET
  MARKET_SEC
  MARKET_KG
  MARKET_SEC_KG

Retrieval:
  SEC: V11-F PIT SEC + V13 SEC retriever
  KG:  V13 temporal direct-company 1-hop retriever

RL:
  SB3 A2C / PPO
  5-stock long-only portfolio
  target allocations = softmax(actions)
  close-to-close market return
  proportional turnover cost

Caveats:
  V11-F filing time is conservatively shifted,
  but source excerpts are not independently verified.
  KG first_observed is not certified publication time.
  No generative LLM reasoning in V16.
"""

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from stable_baselines3 import A2C, PPO

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v13_unified_sec_kg_rag as v13,
    run_v14_0_ablation as base,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
ARMS = [
    "MARKET",
    "MARKET_SEC",
    "MARKET_KG",
    "MARKET_SEC_KG",
]


def say(*a):
    print(*a, flush=True)


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


class DualRAGPortfolioEnv(gym.Env):
    """
    Self-contained 5-stock portfolio environment.

    Position convention:
      Observation at t uses only features available
      by decision date t.

      At t, action chooses post-trade weights.
      Reward is net portfolio return t -> t+1.

      Turnover = sum(abs(target - pretrade weights))
      Trading cost is deducted before market return.

      Market movement drifts portfolio weights
      until the next decision.

    This is a V16 experimental adapter using FinRL-X
    datasets and SB3 RL algorithms, not a claim of
    modifying the original FinRL-X gym environment.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features,
        prices,
        dates,
        transaction_cost=0.001,
        reward_scale=10.0,
    ):
        super().__init__()

        self.features = np.asarray(
            features, dtype=np.float32
        )

        self.prices = np.asarray(
            prices, dtype=np.float64
        )

        self.dates = pd.DatetimeIndex(dates)
        self.transaction_cost = float(
            transaction_cost
        )
        self.reward_scale = float(
            reward_scale
        )

        if (
            self.features.ndim != 3
            or self.features.shape[1] != 5
            or self.prices.shape != (
                len(self.features), 5
            )
            or len(self.dates)
            != len(self.features)
        ):
            raise ValueError(
                "Invalid portfolio data shapes: "
                f"{self.features.shape}, "
                f"{self.prices.shape}"
            )

        if len(self.features) < 20:
            raise ValueError("Insufficient trading dates")

        if (
            not np.isfinite(self.features).all()
            or not np.isfinite(self.prices).all()
            or np.any(self.prices <= 0)
        ):
            raise ValueError(
                "Invalid features or prices"
            )

        if not self.dates.is_unique:
            raise ValueError("Duplicate dates")

        if not self.dates.is_monotonic_increasing:
            raise ValueError("Dates out of order")

        if self.transaction_cost < 0:
            raise ValueError(
                "Negative transaction cost"
            )

        self.feature_dim = (
            self.features.shape[1]
            * self.features.shape[2]
        )

        obs_dim = self.feature_dim + 5

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.reset()

    def _obs(self):
        return np.concatenate(
            [
                self.features[self.t].reshape(-1),
                self.weights,
            ]
        ).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.value = 1.0
        self.weights = np.full(
            5, 0.2, dtype=np.float64
        )

        self.history = []

        return self._obs(), {}

    def step(self, action):
        t = self.t

        if t >= len(self.features) - 1:
            raise RuntimeError(
                "step called after final trading step"
            )

        action = np.asarray(
            action,
            dtype=np.float64,
        )

        if (
            action.shape != (5,)
            or not np.isfinite(action).all()
        ):
            raise RuntimeError(
                "Invalid policy action"
            )

        # SB3 action is bounded to [-1,1].
        # Softmax converts it to long-only weights.
        target = softmax(
            np.clip(action, -1.0, 1.0)
        )

        pretrade = self.weights.copy()

        turnover = float(
            np.abs(target - pretrade).sum()
        )

        cost_fraction = (
            self.transaction_cost * turnover
        )

        if cost_fraction >= 1.0:
            raise RuntimeError(
                "Transaction cost exceeds portfolio"
            )

        relative = (
            self.prices[t+1]
            / self.prices[t]
        )

        gross_multiple = float(
            np.dot(target, relative)
        )

        net_multiple = (
            1.0 - cost_fraction
        ) * gross_multiple

        if net_multiple <= 0:
            raise RuntimeError(
                "Nonpositive portfolio multiple"
            )

        # Drift weights after close-to-close return.
        self.weights = (
            target * relative
        ) / gross_multiple

        self.value *= net_multiple

        log_return = float(
            np.log(net_multiple)
        )

        reward = (
            self.reward_scale
            * log_return
        )

        self.history.append({
            "date": str(
                self.dates[t+1].date()
            ),
            "decision_date": str(
                self.dates[t].date()
            ),
            "portfolio_value": float(
                self.value
            ),
            "net_return": float(
                net_multiple-1.0
            ),
            "gross_return": float(
                gross_multiple-1.0
            ),
            "turnover": turnover,
            "cost_fraction": float(
                cost_fraction
            ),
            **{
                f"w_{ticker}": float(
                    target[j]
                )
                for j, ticker in enumerate(TICKERS)
            },
        })

        self.t += 1

        terminated = (
            self.t >= len(self.features) - 1
        )

        return (
            self._obs(),
            float(reward),
            bool(terminated),
            False,
            {
                "portfolio_value": float(
                    self.value
                ),
                "net_multiple": float(
                    net_multiple
                ),
                "turnover": turnover,
            },
        )


def metrics_from_history(
    history,
    name,
    seed,
    algo,
):
    df = pd.DataFrame(history)

    returns = df[
        "net_return"
    ].to_numpy(dtype=np.float64)

    wealth = df[
        "portfolio_value"
    ].to_numpy(dtype=np.float64)

    # Include initial wealth of 1 in drawdown.
    full_wealth = np.r_[1.0, wealth]
    high = np.maximum.accumulate(full_wealth)
    drawdowns = full_wealth / high - 1.0

    volatility = float(
        np.std(returns, ddof=1)
    ) if len(returns) > 1 else 0.0

    sharpe = (
        float(
            np.sqrt(252.0)
            * np.mean(returns)
            / volatility
        )
        if volatility > 1e-12
        else np.nan
    )

    return {
        "arm": name,
        "seed": seed,
        "algo": algo,
        "days": len(returns),
        "final_multiple": float(
            full_wealth[-1]
        ),
        "annualized_return": float(
            full_wealth[-1]
            ** (252.0 / len(returns))
            - 1.0
        ),
        "annualized_volatility": (
            volatility * np.sqrt(252.0)
        ),
        "sharpe": sharpe,
        "max_drawdown": float(
            drawdowns.min()
        ),
        "mean_turnover": float(
            df["turnover"].mean()
        ),
        "sum_turnover": float(
            df["turnover"].sum()
        ),
        "total_transaction_cost_fraction_sum": float(
            df["cost_fraction"].sum()
        ),
    }


def train_policy(
    algo,
    env,
    seed,
    timesteps,
    learning_rate,
):
    cls = {
        "A2C": A2C,
        "PPO": PPO,
    }[algo]

    kwargs = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=learning_rate,
        device="cpu",
        verbose=0,
        policy_kwargs=dict(
            net_arch=dict(
                pi=[128, 64],
                vf=[128, 64],
            )
        ),
    )

    if algo == "A2C":
        kwargs.update(
            n_steps=20,
            gamma=0.99,
            gae_lambda=0.95,
        )
    else:
        kwargs.update(
            n_steps=128,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

    model = cls(**kwargs)

    model.learn(
        total_timesteps=timesteps,
        progress_bar=False,
    )

    return model


def evaluate(model, env, name, seed, algo):
    obs, _ = env.reset()

    done = False

    while not done:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, _, terminated, truncated, _ = (
            env.step(action)
        )

        done = terminated or truncated

    history = pd.DataFrame(
        env.history
    )

    result = metrics_from_history(
        env.history,
        name,
        seed,
        algo,
    )

    return history, result


def equal_weight_baseline(
    features,
    prices,
    dates,
    transaction_cost,
):
    env = DualRAGPortfolioEnv(
        features,
        prices,
        dates,
        transaction_cost=transaction_cost,
    )

    env.reset()

    done = False

    # softmax(zeros) = equal weights.
    while not done:
        _, _, done, _, _ = env.step(
            np.zeros(5, dtype=np.float32)
        )

    history = pd.DataFrame(env.history)

    metrics = metrics_from_history(
        env.history,
        "EQUAL_WEIGHT",
        -1,
        "NONE",
    )

    return history, metrics


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v16_dual_rag_finrl"
        ),
    )

    p.add_argument(
        "--pit-sec",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/"
            "sec_pit_retrieval.parquet"
        ),
    )

    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--algo",
        choices=["A2C", "PPO"],
        default="PPO",
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=30000,
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--reward-scale",
        type=float,
        default=10.0,
    )

    p.add_argument(
        "--sec-top-k",
        type=int,
        default=5,
    )

    p.add_argument(
        "--kg-top-k",
        type=int,
        default=12,
    )

    # Arguments needed by existing loaders.
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    p.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    p.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
        ),
    )

    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--embed", default="bge-m3:latest")
    p.add_argument("--device", default="cpu")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--reward-scaling", type=float, default=1.0)

    return p.parse_args()


def main():
    args = parse_args()

    if (
        args.timesteps < 100
        or args.transaction_cost < 0
    ):
        raise ValueError("Invalid experiment parameters")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    say("========== LOAD ORIGINAL FINRL-X ==========")

    # Existing loader internally requires A2C model.
    # Our new RL experiment is independently trained.
    requested_algo = args.algo
    args.algo = "A2C"

    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    args.algo = requested_algo

    dates = base.normalize_dates(dates)

    market_flat = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    )

    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    )

    market, kgt = base.validate_market_layout(
        market_flat
    )

    if not (
        len(dates) == len(prices) == len(market)
        and dates.min().year == 2018
        and dates.max().year == 2021
        and dates[n_train].year == 2021
    ):
        raise RuntimeError(
            "Invalid chronological split"
        )

    say("[DATES]", dates.min(), dates.max())
    say("[TRAIN]", n_train, "[VAL]", len(dates)-n_train)

    say("========== SEC-RAG ==========")

    sec, sec_document_rows = v13.load_sec(
        dates,
        contexts,
        args.pit_sec,
        args.sec_top_k,
        args.out / "sec_retrieval_audit.csv",
    )

    say("[SEC RAG]", sec.shape)

    say("========== DYNAMIC KG-RAG ==========")

    kg_events = v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg, relations = v13.build_kg_rag(
        dates,
        kg_events,
        args.kg_top_k,
        args.out,
    )

    say("[KG RAG]", kg.shape)

    if not (
        sec.shape == (len(dates), 5, 64)
        and kg.shape[:2] == (len(dates), 5)
        and market.shape == (len(dates), 5, 106)
    ):
        raise RuntimeError(
            "Unified RAG alignment/dimension error"
        )

    say("========== BUILD UNIFIED OBSERVATIONS ==========")

    datasets = {
        "MARKET": base.compose(
            market,
        ),
        "MARKET_SEC": base.compose(
            market, sec,
        ),
        "MARKET_KG": base.compose(
            market, kg,
        ),
        "MARKET_SEC_KG": base.compose(
            market, sec, kg,
        ),
    }

    for arm, x in datasets.items():
        if not np.isfinite(x).all():
            raise RuntimeError(
                f"Nonfinite {arm} features"
            )

        say(
            "[ARM]",
            arm,
            "company features=", x.shape[2],
            "RL observation=",
            5*x.shape[2]+5,
        )

    say("========== TRAIN AND VALIDATE RL ==========")

    all_metrics = []

    eq_history, eq_metrics = equal_weight_baseline(
        datasets["MARKET"][n_train:],
        prices[n_train:],
        dates[n_train:],
        args.transaction_cost,
    )

    all_metrics.append(eq_metrics)

    eq_history.to_csv(
        args.out / "EQUAL_WEIGHT_2021.csv",
        index=False,
    )

    for arm, x in datasets.items():
        xtrain = x[:n_train]
        xval = x[n_train:]

        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)

            say()
            say(
                "[TRAIN]",
                arm,
                args.algo,
                "seed=", seed,
            )

            train_env = DualRAGPortfolioEnv(
                xtrain,
                prices[:n_train],
                dates[:n_train],
                args.transaction_cost,
                args.reward_scale,
            )

            val_env = DualRAGPortfolioEnv(
                xval,
                prices[n_train:],
                dates[n_train:],
                args.transaction_cost,
                args.reward_scale,
            )

            policy = train_policy(
                args.algo,
                train_env,
                seed,
                args.timesteps,
                args.learning_rate,
            )

            model_path = (
                args.out
                / f"{arm}_{args.algo}_seed{seed}"
            )

            policy.save(str(model_path))

            history, row = evaluate(
                policy,
                val_env,
                arm,
                seed,
                args.algo,
            )

            history.to_csv(
                args.out
                / (
                    f"{arm}_{args.algo}_"
                    f"seed{seed}_2021.csv"
                ),
                index=False,
            )

            all_metrics.append(row)

            say(
                "[VALIDATION]",
                arm,
                seed,
                "multiple=",
                round(row["final_multiple"], 6),
                "sharpe=",
                round(row["sharpe"], 6),
                "mdd=",
                round(row["max_drawdown"], 6),
                "turnover=",
                round(row["mean_turnover"], 6),
            )

    result = pd.DataFrame(all_metrics)

    result.to_csv(
        args.out / "portfolio_metrics.csv",
        index=False,
    )

    summary = (
        result.loc[
            result["arm"] != "EQUAL_WEIGHT"
        ]
        .groupby("arm")
        .agg(
            seeds=("seed", "count"),
            mean_multiple=("final_multiple", "mean"),
            std_multiple=("final_multiple", "std"),
            mean_sharpe=("sharpe", "mean"),
            mean_mdd=("max_drawdown", "mean"),
            mean_turnover=("mean_turnover", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        args.out / "portfolio_summary.csv",
        index=False,
    )

    manifest = {
        "experiment": "FinRL-X V16 Dual-RAG",
        "train": "2018-2020",
        "validation": "2021",
        "test_2022_2023_used": False,
        "algo": args.algo,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "transaction_cost": args.transaction_cost,
        "reward_scale": args.reward_scale,
        "market_company_dim": int(market.shape[2]),
        "sec_company_dim": int(sec.shape[2]),
        "kg_company_dim": int(kg.shape[2]),
        "sec_source_rows": int(sec_document_rows),
        "kg_company_triples": int(len(kg_events)),
        "kg_relations": relations,
        "arms": {
            arm: int(x.shape[2])
            for arm, x in datasets.items()
        },
        "limitations": [
            "This is an experimental SB3 portfolio adapter.",
            "Not a modification of the pretrained original FinRL-X policy.",
            "Uses existing retrieval feature encodings, not LLM generation.",
            "SEC excerpt content not independently certified.",
            "KG first-observed does not certify public release time.",
            "Features are computed before train/val slicing with historical date filters.",
            "2021 previously inspected; not a new blind test.",
            "Long-only fully invested softmax action.",
            "Close-to-close execution is an approximation and requires executable-price timing review.",
            "No cash allocation, slippage, taxes, or market impact.",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    say()
    say("========== PORTFOLIO SUMMARY ==========")
    say(summary.to_string(index=False))
    say()
    say("V16 DUAL-RAG FINRL-X: PASS")


if __name__ == "__main__":
    main()
