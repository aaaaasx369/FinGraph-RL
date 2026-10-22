#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V7-B: Structured SEC Evidence Pilot

Arms:
    MARKET:
        Market + Dynamic KGT + portfolio weights

    STRUCTURED:
        Same inputs + point-in-time SEC financial signals

Frozen retrieval pipeline / no new LLM calls.
A2C actors trained from scratch.
Validation only: 2021.
Test 2022-2023 is intentionally disabled.

Important:
    We impose an extra one-calendar-day delay
    on SEC available_at, but this does not independently
    verify the true SEC public release timestamp.

    Existing market/KGT feature engineering is inherited
    from the project and must be audited separately.
"""

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd

from gymnasium import spaces
from stable_baselines3 import A2C

from unified_event_memory.finrl_x.aegis_rl import (
    run_stop_counterfactual_v1 as cf
)

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]

SIGNAL_COLUMNS = [
    "impact", "risk", "confidence"
]


def say(*x):
    print(*x, flush=True)


def build_sec_features(csv_path, dates):
    """
    Each ticker gets:
        impact
        risk
        confidence
        recency
        exists

    Total = 5 stocks * 5 = 25D.

    Uses only latest conservatively available signal.
    """

    raw = pd.read_csv(csv_path)

    required = {
        "head",
        "available_at",
        "impact",
        "risk",
        "confidence",
    }

    missing = required - set(raw.columns)
    if missing:
        raise RuntimeError(
            f"Missing SEC columns: {missing}"
        )

    raw = raw.copy()

    raw["ticker"] = (
        raw["head"]
        .astype(str)
        .str.replace(
            "COMPANY:", "",
            regex=False
        )
    )

    raw = raw[
        raw["ticker"].isin(TICKERS)
    ].copy()

    raw["available_at"] = pd.to_datetime(
        raw["available_at"],
        errors="raise",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    # Conservative release convention:
    # not usable until at least the next calendar day.
    raw["effective_date"] = (
        raw["available_at"]
        + pd.Timedelta(days=1)
    )

    for col in SIGNAL_COLUMNS:
        raw[col] = pd.to_numeric(
            raw[col],
            errors="raise"
        )

    if not raw[SIGNAL_COLUMNS].notna().all().all():
        raise RuntimeError(
            "Non-finite SEC numeric signals"
        )

    limits = {
        "impact": (-1.0, 1.0),
        "risk": (0.0, 1.0),
        "confidence": (0.0, 1.0),
    }

    for col, (low, high) in limits.items():
        if not raw[col].between(low, high).all():
            raise RuntimeError(
                f"SEC range violation: {col}"
            )

    if "event_id" not in raw.columns:
        raw["event_id"] = (
            raw.index.astype(str)
        )

    # One signal per ticker/effective date:
    # highest confidence, stable ID tie break.
    raw = raw.sort_values(
        [
            "ticker",
            "effective_date",
            "confidence",
            "event_id",
        ],
        ascending=[True, True, False, True],
        kind="stable",
    )

    raw = raw.drop_duplicates(
        subset=["ticker", "effective_date"],
        keep="first",
    )

    raw = raw.sort_values(
        ["ticker", "effective_date"]
    )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            dates,
            utc=True
        ).tz_convert(None).normalize()
    )

    output = np.zeros(
        (len(dates), 5, 5),
        dtype=np.float32,
    )

    coverage = []

    for j, ticker in enumerate(TICKERS):
        one = raw[
            raw["ticker"] == ticker
        ].copy()

        signal_days = (
            one["effective_date"]
            .to_numpy(dtype="datetime64[ns]")
        )

        decision_days = (
            dates.to_numpy(dtype="datetime64[ns]")
        )

        # Latest signal whose effective date is
        # <= decision date. Its recorded available_at
        # is therefore strictly before decision date.
        indices = np.searchsorted(
            signal_days,
            decision_days,
            side="right",
        ) - 1

        valid = indices >= 0

        for i in np.flatnonzero(valid):
            record = one.iloc[int(indices[i])]

            age = (
                dates[i] - record["effective_date"]
            ).days

            if age < 0:
                raise RuntimeError(
                    "Point-in-time violation"
                )

            # No forced expiry. Recency is explicit.
            recency = np.exp(
                -float(age) / 180.0
            )

            output[i, j] = [
                float(record["impact"]),
                float(record["risk"]),
                float(record["confidence"]),
                float(recency),
                1.0,
            ]

        coverage.append({
            "ticker": ticker,
            "available_days": int(valid.sum()),
            "total_days": len(dates),
            "coverage": float(valid.mean()),
            "signal_records": len(one),
        })

    if not np.isfinite(output).all():
        raise RuntimeError(
            "Nonfinite structured features"
        )

    return output.reshape(len(dates), -1), pd.DataFrame(coverage)


def load_split(args, split):
    say("\n[LOAD SPLIT]", split)

    args.split = split
    args.algo = "A2C"

    env, _, _ = cf.load_system(args)

    market = np.asarray(
        env.features,
        dtype=np.float32,
    )

    if market.ndim == 3:
        if market.shape[1] != 5:
            raise RuntimeError(
                f"Unexpected market shape: {market.shape}"
            )
        market = market.reshape(
            len(market), -1
        )

    if market.ndim != 2 or market.shape[1] != 1265:
        raise RuntimeError(
            f"Expected market+KGT 1265D, got {market.shape}"
        )

    prices = np.asarray(
        env.price,
        dtype=np.float64,
    )

    dates = pd.DatetimeIndex(
        pd.to_datetime(
            env.dates,
            utc=True
        )
    ).tz_convert(None).normalize()

    if prices.ndim != 2 or prices.shape[1] != 5:
        raise RuntimeError(
            f"Unexpected prices: {prices.shape}"
        )

    n = min(
        len(market),
        len(prices),
        len(dates),
    )

    market = market[:n]
    prices = prices[:n]
    dates = dates[:n]

    if n < 3:
        raise RuntimeError(
            "Not enough observations"
        )

    if not dates.is_monotonic_increasing:
        raise RuntimeError(
            "Dates are not chronological"
        )

    if not dates.is_unique:
        raise RuntimeError(
            "Duplicate trading dates"
        )

    if not np.isfinite(market).all():
        raise RuntimeError(
            "Market features contain NaN/Inf"
        )

    if (
        not np.isfinite(prices).all()
        or (prices <= 0).any()
    ):
        raise RuntimeError(
            "Invalid stock prices"
        )

    expected = {
        "train": ("2018-01-01", "2020-12-31"),
        "val": ("2021-01-01", "2021-12-31"),
        "test": ("2022-01-01", "2023-12-31"),
    }

    lo, hi = expected[split]

    if not all(
        pd.Timestamp(lo) <= d <= pd.Timestamp(hi)
        for d in dates
    ):
        raise RuntimeError(
            f"Split {split} outside expected dates: "
            f"{dates.min()}..{dates.max()}"
        )

    structured, coverage = build_sec_features(
        args.sec_signals,
        dates,
    )

    say(
        "[DATA]",
        split,
        "days=", n,
        "market_dim=", market.shape[1],
        "structured_dim=", structured.shape[1],
    )

    say("[SEC COVERAGE]")
    say(coverage.to_string(index=False))

    return {
        "market": market,
        "structured": structured,
        "prices": prices,
        "dates": dates,
        "coverage": coverage,
    }


class PortfolioEnv(gym.Env):
    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        data,
        arm,
        transaction_cost=0.001,
        reward_scale=100.0,
        max_days=0,
    ):
        super().__init__()

        self.arm = arm
        self.transaction_cost = transaction_cost
        self.reward_scale = reward_scale

        n = len(data["dates"])

        if max_days > 0:
            n = min(
                n,
                max_days + 1
            )

        self.market = data["market"][:n]
        self.structured = data["structured"][:n]
        self.prices = data["prices"][:n]
        self.dates = data["dates"][:n]

        if arm == "MARKET":
            self.features = self.market

        elif arm == "STRUCTURED":
            self.features = np.concatenate(
                [
                    self.market,
                    self.structured,
                ],
                axis=1,
            )

        else:
            raise ValueError(arm)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.features.shape[1] + 5,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

        self.reset()

    @staticmethod
    def softmax(x):
        z = np.asarray(
            x,
            dtype=np.float64
        )

        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def _obs(self):
        return np.concatenate([
            self.features[self.day],
            self.weights.astype(np.float32),
        ]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.day = 0
        self.weights = np.ones(5) / 5.0
        self.equity = 1.0
        self.history = []

        return self._obs(), {}

    def step(self, action):
        if self.day + 1 >= len(self.prices):
            raise RuntimeError(
                "Episode already complete"
            )

        old_weights = self.weights.copy()

        target = self.softmax(action)

        day_returns = (
            self.prices[self.day + 1]
            / self.prices[self.day]
            - 1.0
        )

        turnover = float(
            np.abs(
                target - old_weights
            ).sum()
        )

        cost = (
            self.transaction_cost
            * turnover
        )

        gross = float(
            target @ day_returns
        )

        net = max(
            gross - cost,
            -0.999999
        )

        self.equity *= 1.0 + net

        after = target * (1.0 + day_returns)

        if after.sum() > 0:
            self.weights = after / after.sum()
        else:
            self.weights = np.ones(5) / 5.0

        current_date = str(
            self.dates[self.day].date()
        )

        self.history.append({
            "date": current_date,
            "net_return": float(net),
            "gross_return": gross,
            "transaction_cost": float(cost),
            "turnover": turnover,
            "equity": float(self.equity),
            **{
                f"weight_{ticker}": float(target[j])
                for j, ticker in enumerate(TICKERS)
            },
        })

        self.day += 1

        terminated = (
            self.day >= len(self.prices) - 1
        )

        reward = (
            float(np.log1p(net))
            * self.reward_scale
        )

        return (
            self._obs(),
            reward,
            terminated,
            False,
            {
                "equity": float(self.equity),
                "net_return": float(net),
            },
        )


def evaluate(model, env):
    obs, _ = env.reset()

    finished = False

    while not finished:
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, _, finished, _, _ = env.step(action)

    history = pd.DataFrame(env.history)

    returns = history["net_return"].to_numpy(
        dtype=np.float64
    )

    equity = history["equity"].to_numpy(
        dtype=np.float64
    )

    curve = np.r_[
        1.0,
        equity,
    ]

    drawdown = (
        curve / np.maximum.accumulate(curve)
        - 1.0
    )

    vol = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0

    sharpe = (
        float(
            np.sqrt(252)
            * returns.mean()
            / vol
        )
        if vol > 0
        else float("nan")
    )

    metrics = {
        "days": len(history),
        "total_return": float(equity[-1] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "annualized_volatility": float(
            vol * np.sqrt(252)
        ),
        "mean_turnover": float(
            history["turnover"].mean()
        ),
    }

    return history, metrics


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v7b_structured_pilot"
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
        "--timesteps",
        type=int,
        default=3000,
    )

    p.add_argument(
        "--max-days",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

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
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )

    p.add_argument(
        "--embed",
        default="bge-m3:latest",
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--cost-scale",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--tier",
        default="conservative",
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = p.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.timesteps <= 0:
        raise RuntimeError(
            "Timesteps must be positive"
        )

    say("===== V7-B DATA PREPARATION =====")

    train = load_split(
        args,
        "train"
    )

    val = load_split(
        args,
        "val"
    )

    pd.concat(
        [
            train["coverage"].assign(split="train"),
            val["coverage"].assign(split="val"),
        ],
        ignore_index=True,
    ).to_csv(
        args.out / "sec_coverage.csv",
        index=False,
    )

    results = []

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)

        for arm in [
            "MARKET",
            "STRUCTURED",
        ]:
            say()
            say("=" * 70)
            say("TRAIN", arm, "SEED", seed)
            say("=" * 70)

            train_env = PortfolioEnv(
                train,
                arm,
                transaction_cost=args.transaction_cost,
                reward_scale=100.0,
                max_days=args.max_days,
            )

            val_env = PortfolioEnv(
                val,
                arm,
                transaction_cost=args.transaction_cost,
                reward_scale=100.0,
                max_days=args.max_days,
            )

            model = A2C(
                "MlpPolicy",
                train_env,
                learning_rate=args.learning_rate,
                n_steps=10,
                gamma=0.99,
                gae_lambda=1.0,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs={
                    "net_arch": [128, 128]
                },
                seed=seed,
                device=args.device,
                verbose=0,
            )

            model.learn(
                total_timesteps=args.timesteps,
            )

            run_dir = (
                args.out
                / f"{arm}_seed{seed}"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            model.save(
                str(run_dir / "model.zip")
            )

            history, metrics = evaluate(
                model,
                val_env,
            )

            history.to_csv(
                run_dir / "validation_history.csv",
                index=False,
            )

            row = {
                "arm": arm,
                "seed": seed,
                "timesteps": args.timesteps,
                "split": "val",
                **metrics,
            }

            results.append(row)

            say("[RESULT]", row)

            pd.DataFrame(
                results
            ).to_csv(
                args.out / "partial_results.csv",
                index=False,
            )

    result = pd.DataFrame(results)

    result.to_csv(
        args.out / "validation_results.csv",
        index=False,
    )

    wide = result.pivot(
        index="seed",
        columns="arm",
        values="total_return",
    )

    wide["structured_minus_market_pp"] = (
        wide["STRUCTURED"] - wide["MARKET"]
    ) * 100.0

    wide.to_csv(
        args.out / "paired_seed_comparison.csv"
    )

    say()
    say("===== VALIDATION RESULTS =====")
    say(result.to_string(index=False))

    say()
    say("===== STRUCTURED vs MARKET =====")
    say(wide.to_string())

    say()
    say("V7-B STRUCTURED PILOT: PASS")
    say("NO TEST DATA EVALUATED")


if __name__ == "__main__":
    main()
