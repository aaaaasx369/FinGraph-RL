from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch

import gymnasium as gym
from gymnasium import spaces

from sklearn.preprocessing import StandardScaler

from stable_baselines3 import (
    A2C,
    PPO,
    DDPG,
)

from stable_baselines3.common.noise import (
    NormalActionNoise,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
)


# ============================================================
# Configuration
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

AGENTS = [
    "A2C",
    "PPO",
    "DDPG",
]

FEATURE_SETS = [
    "F0",
    "F1",
]

INITIAL_AMOUNT = 1_000_000.0

TRADING_DAYS = 252


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Helpers
# ============================================================

def normalize_ticker(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "ticker" not in df.columns:

        if "tic" in df.columns:

            df = df.rename(
                columns={
                    "tic":
                        "ticker"
                }
            )

        else:

            raise ValueError(
                "No ticker/tic column."
            )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


def normalize_date(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    if "date" not in df.columns:

        for candidate in [
            "trade_date",
            "datetime",
        ]:

            if candidate in df.columns:

                df = df.rename(
                    columns={
                        candidate:
                            "date"
                    }
                )

                break

    if "date" not in df.columns:

        raise ValueError(
            "No date column."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    return df


def detect_price_column(
    df: pd.DataFrame,
) -> str:

    candidates = [
        "close",
        "adj_close",
        "adjclose",
        "adjusted_close",
        "adjcp",
        "Close",
    ]

    for c in candidates:

        if c in df.columns:

            return c

    raise ValueError(
        "Could not find close-price column "
        f"in market data. Columns: "
        f"{df.columns.tolist()}"
    )


def split_name(
    dates: pd.Series,
) -> pd.Series:

    out = pd.Series(
        "other",
        index=dates.index,
        dtype="object",
    )

    out[
        (dates >= pd.Timestamp("2018-01-08"))
        &
        (dates <= pd.Timestamp("2021-12-31"))
    ] = "train"

    out[
        (dates >= pd.Timestamp("2022-01-01"))
        &
        (dates <= pd.Timestamp("2022-06-30"))
    ] = "val"

    out[
        (dates >= pd.Timestamp("2022-07-01"))
        &
        (dates <= pd.Timestamp("2022-12-30"))
    ] = "test"

    return out


# ============================================================
# Load market data
#
# This is used ONLY for realized next-day returns.
# Feature CSVs remain M0/M1/M2.
# ============================================================

def load_market(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    df = normalize_ticker(
        df
    )

    df = normalize_date(
        df
    )

    price_col = detect_price_column(
        df
    )

    df = df[
        [
            "date",
            "ticker",
            price_col,
        ]
    ].copy()

    df = df.rename(
        columns={
            price_col:
                "market_close"
        }
    )

    df["market_close"] = (
        pd.to_numeric(
            df["market_close"],
            errors="coerce",
        )
    )

    df = df.dropna(
        subset=[
            "date",
            "ticker",
            "market_close",
        ]
    )

    df = df[
        df["ticker"].isin(
            TICKERS
        )
    ]

    df = (
        df
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .drop_duplicates(
            subset=[
                "date",
                "ticker",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# Feature selection
#
# Important:
#
# M0 base features = numeric columns existing in M0.
#
# M1 = exact M0 base
#      + text_emb_*
#
# M2 = exact M0 base
#      + text_emb_*
#      + relation KG dimensions only
#
# Helper QA columns such as kg norm/nonzero are NOT included.
# ============================================================

def get_base_feature_columns(
    df: pd.DataFrame,
) -> List[str]:

    excluded = {
        "date",
        "ticker",
        "tic",
        "split",
        "market_close",
        "time_id",
        "entity_id",
        "findkg_weekly_split",
        "findkg_age_days",
        "findkg_embedding_norm",
    }

    excluded_prefixes = [
        "signal_available_date",
        "text_embedding_available_date",
        "kg_latest_edge_date",
        "findkg_available_date",
        "findkg_emb_",
    ]

    cols = []

    for col in df.columns:

        if col in excluded:
            continue

        if any(
            col.startswith(prefix)
            for prefix in excluded_prefixes
        ):
            continue

        if pd.api.types.is_numeric_dtype(
            df[col]
        ):
            cols.append(col)

    if not cols:
        raise ValueError(
            "No numeric F0 features found."
        )

    return sorted(cols)


def get_findkg_columns(
    df: pd.DataFrame,
) -> List[str]:

    return sorted(
        [
            c
            for c in df.columns
            if c.startswith("findkg_emb_")
        ]
    )


# ============================================================
# Load feature set
# ============================================================

def load_feature_set(
    path: Path,
    market: pd.DataFrame,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    df = normalize_ticker(
        df
    )

    df = normalize_date(
        df
    )

    df = df[
        df[
            "ticker"
        ].isin(
            TICKERS
        )
    ].copy()

    # Ensure consistent official split.
    df["split"] = split_name(
        df["date"]
    )

    # Price comes from one common market source,
    # not from different M0/M1/M2 files.
    if "market_close" in df.columns:

        df = df.drop(
            columns=[
                "market_close"
            ]
        )

    df = df.merge(
        market,
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_price = int(
        df[
            "market_close"
        ].isna().sum()
    )

    if missing_price:

        raise ValueError(
            f"{path}: "
            f"{missing_price} rows missing "
            "market_close after merge."
        )

    return (
        df
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Scaler
#
# Fit ONLY using 2018-2023.
# ============================================================

def fit_transform_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_path: Path,
) -> Tuple[
    pd.DataFrame,
    StandardScaler,
]:

    out = df.copy()

    train_mask = (
        out[
            "split"
        ]
        ==
        "train"
    )

    X_train = (
        out.loc[
            train_mask,
            feature_cols,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    if len(
        X_train
    ) == 0:

        raise ValueError(
            "No train rows for scaler."
        )

    scaler = StandardScaler()

    scaler.fit(
        X_train
    )

    X_all = (
        out[
            feature_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
        .to_numpy(
            dtype=np.float32
        )
    )

    X_scaled = scaler.transform(
        X_all
    ).astype(
        np.float32
    )

    for i, col in enumerate(
        feature_cols
    ):

        out[
            col
        ] = X_scaled[
            :,
            i
        ]

    scaler_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    return (
        out,
        scaler,
    )


# ============================================================
# Portfolio environment
#
# Observation at date t:
#
# [features for 5 stocks,
#  current portfolio weights]
#
# Action:
# 5 continuous logits -> softmax -> target weights
#
# Reward:
# log portfolio growth from close(t) to close(t+1)
# minus transaction cost.
#
# This avoids using t+1 information in state.
# ============================================================

class PortfolioEnv(
    gym.Env
):

    metadata = {
        "render_modes": []
    }

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        transaction_cost_pct: float,
        reward_scaling: float,
        initial_amount: float,
    ):

        super().__init__()

        self.feature_cols = list(
            feature_cols
        )

        self.transaction_cost_pct = (
            float(
                transaction_cost_pct
            )
        )

        self.reward_scaling = float(
            reward_scaling
        )

        self.initial_amount = float(
            initial_amount
        )

        df = df.copy()

        df = df[
            df[
                "ticker"
            ].isin(
                TICKERS
            )
        ]

        # ----------------------------------------------------
        # Keep only complete dates containing all 5 stocks.
        # ----------------------------------------------------

        count_by_date = (
            df.groupby(
                "date"
            )[
                "ticker"
            ]
            .nunique()
        )

        complete_dates = (
            count_by_date[
                count_by_date
                ==
                len(
                    TICKERS
                )
            ]
            .index
        )

        df = df[
            df[
                "date"
            ].isin(
                complete_dates
            )
        ].copy()

        self.dates = sorted(
            df[
                "date"
            ].unique()
        )

        if len(
            self.dates
        ) < 2:

            raise ValueError(
                "Environment requires "
                "at least 2 market dates."
            )

        # ----------------------------------------------------
        # Price matrix:
        # dates × stocks
        # ----------------------------------------------------

        price = (
            df.pivot(
                index="date",
                columns="ticker",
                values="market_close",
            )
            .reindex(
                columns=TICKERS
            )
            .reindex(
                self.dates
            )
        )

        if price.isna().any().any():

            raise ValueError(
                "Price matrix has missing values."
            )

        self.price_matrix = (
            price.to_numpy(
                dtype=np.float64
            )
        )

        # ----------------------------------------------------
        # Feature tensor:
        # dates × stocks × features
        # ----------------------------------------------------

        feature_arrays = []

        for feature in self.feature_cols:

            matrix = (
                df.pivot(
                    index="date",
                    columns="ticker",
                    values=feature,
                )
                .reindex(
                    columns=TICKERS
                )
                .reindex(
                    self.dates
                )
            )

            if matrix.isna().any().any():

                raise ValueError(
                    f"Missing feature values: "
                    f"{feature}"
                )

            feature_arrays.append(
                matrix.to_numpy(
                    dtype=np.float32
                )
            )

        # list(feature)[feature, date, stock]
        # -> date, stock, feature
        self.feature_tensor = (
            np.stack(
                feature_arrays,
                axis=-1,
            )
            .astype(
                np.float32
            )
        )

        self.n_assets = len(
            TICKERS
        )

        self.n_features = len(
            self.feature_cols
        )

        obs_dim = (
            self.n_assets
            *
            self.n_features
            +
            self.n_assets
        )

        self.observation_space = (
            spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(
                    obs_dim,
                ),
                dtype=np.float32,
            )
        )

        self.action_space = (
            spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(
                    self.n_assets,
                ),
                dtype=np.float32,
            )
        )

        self._reset_state()


    # --------------------------------------------------------
    # Softmax target allocation
    # --------------------------------------------------------

    @staticmethod
    def action_to_weights(
        action: np.ndarray,
    ) -> np.ndarray:

        x = np.asarray(
            action,
            dtype=np.float64,
        )

        x = np.clip(
            x,
            -20,
            20,
        )

        x = x - np.max(
            x
        )

        exp_x = np.exp(
            x
        )

        total = exp_x.sum()

        if (
            not np.isfinite(
                total
            )
            or
            total <= 0
        ):

            return np.ones(
                len(x),
                dtype=np.float64,
            ) / len(x)

        return (
            exp_x
            /
            total
        )


    def _reset_state(
        self,
    ):

        self.step_idx = 0

        self.portfolio_value = (
            self.initial_amount
        )

        self.weights = (
            np.ones(
                self.n_assets,
                dtype=np.float64,
            )
            /
            self.n_assets
        )

        self.portfolio_values = [
            self.portfolio_value
        ]

        self.weight_history = [
            self.weights.copy()
        ]

        self.date_history = [
            pd.Timestamp(
                self.dates[0]
            )
        ]

        self.daily_returns = []

        self.turnover_history = []


    def _get_obs(
        self,
    ) -> np.ndarray:

        features = (
            self.feature_tensor[
                self.step_idx
            ]
            .reshape(
                -1
            )
        )

        obs = np.concatenate(
            [
                features,
                self.weights.astype(
                    np.float32
                ),
            ]
        )

        return obs.astype(
            np.float32
        )


    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        self._reset_state()

        return (
            self._get_obs(),
            {},
        )


    def step(
        self,
        action,
    ):

        current_idx = (
            self.step_idx
        )

        next_idx = (
            current_idx
            +
            1
        )

        target_weights = (
            self.action_to_weights(
                action
            )
        )

        # ----------------------------------------------------
        # Rebalance cost
        #
        # L1 weight movement = traded fraction
        # Turnover statistic = 0.5 L1
        # ----------------------------------------------------

        l1_trade = float(
            np.abs(
                target_weights
                -
                self.weights
            ).sum()
        )

        turnover = (
            0.5
            *
            l1_trade
        )

        transaction_cost = (
            self.transaction_cost_pct
            *
            l1_trade
        )

        # ----------------------------------------------------
        # Realized next-period asset returns
        # ----------------------------------------------------

        p0 = (
            self.price_matrix[
                current_idx
            ]
        )

        p1 = (
            self.price_matrix[
                next_idx
            ]
        )

        asset_returns = (
            p1
            /
            p0
            -
            1.0
        )

        gross_return = float(
            np.dot(
                target_weights,
                asset_returns,
            )
        )

        net_return = (
            gross_return
            -
            transaction_cost
        )

        # Prevent numerical impossible <-100%
        net_return = max(
            net_return,
            -0.999999,
        )

        old_value = (
            self.portfolio_value
        )

        self.portfolio_value *= (
            1.0
            +
            net_return
        )

        # ----------------------------------------------------
        # Drift weights after price move
        # ----------------------------------------------------

        post_values = (
            target_weights
            *
            (
                1.0
                +
                asset_returns
            )
        )

        denom = (
            post_values.sum()
        )

        if (
            np.isfinite(
                denom
            )
            and
            denom > 0
        ):

            self.weights = (
                post_values
                /
                denom
            )

        else:

            self.weights = (
                target_weights
            )

        self.step_idx = (
            next_idx
        )

        realized_return = (
            self.portfolio_value
            /
            old_value
            -
            1.0
        )

        reward = (
            np.log1p(
                realized_return
            )
            *
            self.reward_scaling
        )

        self.portfolio_values.append(
            self.portfolio_value
        )

        self.daily_returns.append(
            realized_return
        )

        self.turnover_history.append(
            turnover
        )

        self.weight_history.append(
            self.weights.copy()
        )

        self.date_history.append(
            pd.Timestamp(
                self.dates[
                    next_idx
                ]
            )
        )

        terminated = (
            next_idx
            >=
            len(
                self.dates
            )
            -
            1
        )

        truncated = False

        info = {
            "date":
                pd.Timestamp(
                    self.dates[
                        next_idx
                    ]
                ),

            "portfolio_value":
                self.portfolio_value,

            "daily_return":
                realized_return,

            "turnover":
                turnover,

            "weights":
                self.weights.copy(),
        }

        return (
            self._get_obs(),
            float(
                reward
            ),
            terminated,
            truncated,
            info,
        )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    values: np.ndarray,
    returns: np.ndarray,
    turnovers: np.ndarray,
) -> Dict[str, float]:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    returns = np.asarray(
        returns,
        dtype=np.float64,
    )

    turnovers = np.asarray(
        turnovers,
        dtype=np.float64,
    )

    if len(values) < 2:

        raise ValueError(
            "Not enough portfolio values."
        )

    total_return = (
        values[-1]
        /
        values[0]
        -
        1.0
    )

    n_days = max(
        len(returns),
        1,
    )

    annual_return = (
        (
            values[-1]
            /
            values[0]
        )
        **
        (
            TRADING_DAYS
            /
            n_days
        )
        -
        1.0
    )

    annual_volatility = (
        np.std(
            returns,
            ddof=1,
        )
        *
        np.sqrt(
            TRADING_DAYS
        )
        if len(
            returns
        ) > 1
        else 0.0
    )

    mean_daily = (
        np.mean(
            returns
        )
        if len(
            returns
        )
        else 0.0
    )

    std_daily = (
        np.std(
            returns,
            ddof=1,
        )
        if len(
            returns
        ) > 1
        else 0.0
    )

    sharpe = (
        mean_daily
        /
        std_daily
        *
        np.sqrt(
            TRADING_DAYS
        )
        if std_daily > 0
        else 0.0
    )

    running_max = (
        np.maximum.accumulate(
            values
        )
    )

    drawdown = (
        values
        /
        running_max
        -
        1.0
    )

    max_drawdown = float(
        np.min(
            drawdown
        )
    )

    avg_turnover = (
        float(
            np.mean(
                turnovers
            )
        )
        if len(
            turnovers
        )
        else 0.0
    )

    return {
        "final_value":
            float(
                values[-1]
            ),

        "total_return":
            float(
                total_return
            ),

        "annual_return":
            float(
                annual_return
            ),

        "annual_volatility":
            float(
                annual_volatility
            ),

        "sharpe":
            float(
                sharpe
            ),

        "max_drawdown":
            float(
                max_drawdown
            ),

        "avg_turnover":
            float(
                avg_turnover
            ),
    }


# ============================================================
# Model
# ============================================================

def build_model(
    agent: str,
    env,
    seed: int,
    learning_rate: float,
    verbose: int,
):

    policy_kwargs = dict(
        net_arch=[
            256,
            256,
        ]
    )

    if agent == "A2C":

        return A2C(
            "MlpPolicy",
            env,
            learning_rate=
                learning_rate,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0,
            vf_coef=0.5,
            seed=seed,
            policy_kwargs=
                policy_kwargs,
            verbose=verbose,
            device="auto",
        )

    if agent == "PPO":

        return PPO(
            "MlpPolicy",
            env,
            learning_rate=
                learning_rate,
            n_steps=256,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            seed=seed,
            policy_kwargs=
                policy_kwargs,
            verbose=verbose,
            device="auto",
        )

    if agent == "DDPG":

        action_dim = len(
            TICKERS
        )

        noise = NormalActionNoise(
            mean=np.zeros(
                action_dim
            ),
            sigma=0.1
            *
            np.ones(
                action_dim
            ),
        )

        return DDPG(
            "MlpPolicy",
            env,
            learning_rate=
                learning_rate,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=128,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=noise,
            seed=seed,
            policy_kwargs=
                policy_kwargs,
            verbose=verbose,
            device="auto",
        )

    raise ValueError(
        agent
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    env: PortfolioEnv,
):

    obs, _ = env.reset()

    terminated = False
    truncated = False

    while (
        not terminated
        and
        not truncated
    ):

        action, _ = (
            model.predict(
                obs,
                deterministic=True,
            )
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )

    values = np.asarray(
        env.portfolio_values
    )

    returns = np.asarray(
        env.daily_returns
    )

    turnovers = np.asarray(
        env.turnover_history
    )

    metrics = calculate_metrics(
        values,
        returns,
        turnovers,
    )

    equity = pd.DataFrame(
        {
            "date":
                env.date_history,

            "portfolio_value":
                env.portfolio_values,
        }
    )

    weights = pd.DataFrame(
        env.weight_history,
        columns=TICKERS,
    )

    weights.insert(
        0,
        "date",
        env.date_history,
    )

    returns_df = pd.DataFrame(
        {
            "date":
                env.date_history[
                    1:
                ],

            "daily_return":
                env.daily_returns,

            "turnover":
                env.turnover_history,
        }
    )

    return (
        metrics,
        equity,
        weights,
        returns_df,
    )


# ============================================================
# Summary
# ============================================================

def build_summary(
    runs: pd.DataFrame,
) -> pd.DataFrame:

    metric_cols = [
        "final_value",
        "total_return",
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "avg_turnover",
    ]

    rows = []

    for (
        feature_set,
        agent,
    ), group in runs.groupby(
        [
            "feature_set",
            "agent",
        ]
    ):

        row = {
            "feature_set":
                feature_set,

            "agent":
                agent,

            "n_seeds":
                len(
                    group
                ),
        }

        for metric in metric_cols:

            row[
                f"{metric}_mean"
            ] = float(
                group[
                    metric
                ].mean()
            )

            row[
                f"{metric}_std"
            ] = float(
                group[
                    metric
                ].std(
                    ddof=1
                )
                if len(
                    group
                ) > 1
                else 0.0
            )

        rows.append(
            row
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "feature_set",
                "agent",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--findkg-data",
        default=(
            "kg/data/findkg/"
            "findkg_finrl_daily_2018_2022.csv"
        ),
    )

    parser.add_argument(
        "--market",
        default=(
            "kg/data/"
            "market_5stocks_2017_2026.csv"
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/"
            "findkg_smoke"
        ),
    )

    parser.add_argument(
        "--seeds",
        default="0",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=100.0,
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

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = (
        output_dir
        /
        "models"
    )

    curve_dir = (
        output_dir
        /
        "curves"
    )

    weight_dir = (
        output_dir
        /
        "weights"
    )

    scaler_dir = (
        output_dir
        /
        "scalers"
    )

    for d in [
        model_dir,
        curve_dir,
        weight_dir,
        scaler_dir,
    ]:

        d.mkdir(
            parents=True,
            exist_ok=True,
        )

    seeds = [
        int(
            x.strip()
        )
        for x
        in args.seeds.split(",")
        if x.strip()
    ]

    # ========================================================
    # Market
    # ========================================================

    print("=" * 76)
    print("LOAD MARKET")
    print("=" * 76)

    market = load_market(
        Path(
            args.market
        )
    )

    print(
        "market rows:",
        len(
            market
        )
    )

    print(
        "market range:",
        market[
            "date"
        ].min(),
        "->",
        market[
            "date"
        ].max(),
    )

    # ========================================================
    # Load FinDKG daily dataset
    #
    # F0 = base / SEC-RAG features
    # F1 = F0 + FinDKG 200d
    # ========================================================

    source_path = Path(
        args.findkg_data
    )

    raw = {}

    for name in FEATURE_SETS:

        print()
        print("=" * 76)
        print("LOAD", name)
        print("=" * 76)

        raw[name] = load_feature_set(
            source_path,
            market,
        )

        print(
            "shape:",
            raw[name].shape,
        )

    base_cols = get_base_feature_columns(
        raw["F0"]
    )

    findkg_cols = get_findkg_columns(
        raw["F1"]
    )

    if len(findkg_cols) != 200:

        raise ValueError(
            "Expected 200 FinDKG dimensions, "
            f"found {len(findkg_cols)}"
        )

    feature_columns = {
        "F0": base_cols,
        "F1": base_cols + findkg_cols,
    }

    print()
    print("=" * 76)
    print("FEATURE DEFINITIONS")
    print("=" * 76)

    print(
        "F0 dims:",
        len(base_cols),
    )

    print(
        "FinDKG dims:",
        len(findkg_cols),
    )

    print(
        "F1 dims:",
        len(
            feature_columns["F1"]
        ),
    )

    # Save feature definitions.
    with open(
        output_dir
        /
        "feature_columns.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2,
        )

    # ========================================================
    # Fit train-only scaler for every feature set
    # ========================================================

    prepared = {}

    for name in FEATURE_SETS:

        print()
        print("=" * 76)
        print(
            "TRAIN-ONLY SCALER",
            name,
        )
        print("=" * 76)

        prepared[
            name
        ], scaler = (
            fit_transform_features(
                raw[
                    name
                ],
                feature_columns[
                    name
                ],
                scaler_dir
                /
                f"{name.lower()}_scaler.joblib",
            )
        )

        print(
            "fit rows:",
            (
                raw[
                    name
                ][
                    "split"
                ]
                ==
                "train"
            ).sum()
        )

    # ========================================================
    # Save experiment configuration
    # ========================================================

    config = {
        "tickers":
            TICKERS,

        "agents":
            AGENTS,

        "feature_sets":
            FEATURE_SETS,

        "train_period":
            "2018-01-08 to 2021-12-31",

        "validation_period":
            "2022-01-01 to 2022-06-30",

        "test_period":
            "2022-07-01 to 2022-12-30",

        "holdout_period":
            "none",

        "seeds":
            seeds,

        "timesteps":
            args.timesteps,

        "transaction_cost":
            args.transaction_cost,

        "reward_scaling":
            args.reward_scaling,

        "learning_rate":
            args.learning_rate,

        "initial_amount":
            INITIAL_AMOUNT,
    }

    with open(
        output_dir
        /
        "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # Experiment loop
    # ========================================================

    run_rows = []

    total_runs = (
        len(
            FEATURE_SETS
        )
        *
        len(
            AGENTS
        )
        *
        len(
            seeds
        )
    )

    run_number = 0

    for feature_set in FEATURE_SETS:

        feature_cols = (
            feature_columns[
                feature_set
            ]
        )

        data = prepared[
            feature_set
        ]

        train_df = data[
            data[
                "split"
            ]
            ==
            "train"
        ].copy()

        val_df = data[
            data[
                "split"
            ]
            ==
            "val"
        ].copy()

        print()
        print("=" * 76)
        print(
            feature_set,
            "DATA"
        )
        print("=" * 76)

        print(
            "train rows:",
            len(
                train_df
            ),
        )

        print(
            "val rows:",
            len(
                val_df
            ),
        )

        print(
            "feature dims:",
            len(
                feature_cols
            ),
        )

        for agent in AGENTS:

            for seed in seeds:

                run_number += 1

                print()
                print(
                    "=" * 76
                )

                print(
                    f"[{run_number}/"
                    f"{total_runs}] "
                    f"{feature_set} | "
                    f"{agent} | "
                    f"seed={seed}"
                )

                print(
                    "=" * 76
                )

                set_seed(
                    seed
                )

                # --------------------------------------------
                # Training environment
                # --------------------------------------------

                def make_train_env():

                    return PortfolioEnv(
                        train_df,
                        feature_cols=
                            feature_cols,
                        transaction_cost_pct=
                            args.transaction_cost,
                        reward_scaling=
                            args.reward_scaling,
                        initial_amount=
                            INITIAL_AMOUNT,
                    )

                vec_env = DummyVecEnv(
                    [
                        make_train_env
                    ]
                )

                model = build_model(
                    agent=agent,
                    env=vec_env,
                    seed=seed,
                    learning_rate=
                        args.learning_rate,
                    verbose=
                        args.verbose,
                )

                print(
                    "Training..."
                )

                model.learn(
                    total_timesteps=
                        args.timesteps,
                    progress_bar=False,
                )

                model_path = (
                    model_dir
                    /
                    (
                        f"{feature_set}_"
                        f"{agent}_"
                        f"seed{seed}"
                    )
                )

                model.save(
                    str(
                        model_path
                    )
                )

                # --------------------------------------------
                # Validation environment
                # --------------------------------------------

                val_env = PortfolioEnv(
                    val_df,
                    feature_cols=
                        feature_cols,
                    transaction_cost_pct=
                        args.transaction_cost,
                    reward_scaling=
                        args.reward_scaling,
                    initial_amount=
                        INITIAL_AMOUNT,
                )

                (
                    metrics,
                    equity,
                    weights,
                    returns_df,
                ) = evaluate_model(
                    model,
                    val_env,
                )

                metrics.update(
                    {
                        "feature_set":
                            feature_set,

                        "agent":
                            agent,

                        "seed":
                            seed,

                        "timesteps":
                            args.timesteps,

                        "feature_dims":
                            len(
                                feature_cols
                            ),
                    }
                )

                run_rows.append(
                    metrics
                )

                stem = (
                    f"{feature_set}_"
                    f"{agent}_"
                    f"seed{seed}_"
                    "val2022H1"
                )

                equity.to_csv(
                    curve_dir
                    /
                    f"{stem}_equity.csv",
                    index=False,
                )

                returns_df.to_csv(
                    curve_dir
                    /
                    f"{stem}_returns.csv",
                    index=False,
                )

                weights.to_csv(
                    weight_dir
                    /
                    f"{stem}_weights.csv",
                    index=False,
                )

                run_df = pd.DataFrame(
                    run_rows
                )

                run_df.to_csv(
                    output_dir
                    /
                    "validation_runs.csv",
                    index=False,
                )

                summary = build_summary(
                    run_df
                )

                summary.to_csv(
                    output_dir
                    /
                    "validation_summary.csv",
                    index=False,
                )

                print(
                    "Validation 2022 H1:"
                )

                print(
                    f"  Final       "
                    f"${metrics['final_value']:,.0f}"
                )

                print(
                    f"  Total Return "
                    f"{metrics['total_return']*100:.2f}%"
                )

                print(
                    f"  Ann Return   "
                    f"{metrics['annual_return']*100:.2f}%"
                )

                print(
                    f"  Vol          "
                    f"{metrics['annual_volatility']*100:.2f}%"
                )

                print(
                    f"  Sharpe       "
                    f"{metrics['sharpe']:.4f}"
                )

                print(
                    f"  Max DD       "
                    f"{metrics['max_drawdown']*100:.2f}%"
                )

                print(
                    f"  Turnover     "
                    f"{metrics['avg_turnover']*100:.2f}%"
                )

                vec_env.close()

                del model
                del vec_env

                if torch.cuda.is_available():

                    torch.cuda.empty_cache()

    # ========================================================
    # Final validation summary
    # ========================================================

    runs = pd.DataFrame(
        run_rows
    )

    summary = build_summary(
        runs
    )

    runs.to_csv(
        output_dir
        /
        "validation_runs.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        /
        "validation_summary.csv",
        index=False,
    )

    print()
    print("=" * 76)
    print("ALL VALIDATION RUNS COMPLETE")
    print("=" * 76)

    display_cols = [
        "feature_set",
        "agent",
        "n_seeds",
        "total_return_mean",
        "total_return_std",
        "sharpe_mean",
        "sharpe_std",
        "max_drawdown_mean",
        "max_drawdown_std",
    ]

    print(
        summary[
            display_cols
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Runs ->",
        output_dir
        /
        "validation_runs.csv",
    )

    print(
        "Summary ->",
        output_dir
        /
        "validation_summary.csv",
    )


if __name__ == "__main__":
    main()
