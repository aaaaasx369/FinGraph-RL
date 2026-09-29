import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO

from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv


# ============================================================
# Settings
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

START_DATE = "2018-01-01"
TRAIN_END = "2023-01-01"
END_DATE = "2024-01-01"

LOOKBACK = 252

MODEL_PATH = ROOT / "results" / "ppo_smoke" / "ppo_model"

RESULT_DIR = ROOT / "results" / "ppo_analysis"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("PPO BEHAVIOR ANALYSIS")
print("=" * 70)


# ============================================================
# 1. Prepare identical dataset
# ============================================================

print("\n[1/6] Downloading data...")

df = YahooDownloader(
    start_date=START_DATE,
    end_date=END_DATE,
    ticker_list=TICKERS,
).fetch_data()


print("\n[2/6] Feature engineering...")

fe = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.INDICATORS,
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False,
)

df = fe.preprocess_data(df)

df = df.sort_values(
    ["date", "tic"]
).reset_index(drop=True)

df.index = df["date"].factorize()[0]


# ============================================================
# 2. Covariance matrices
# ============================================================

print("\n[3/6] Building covariance matrices...")

cov_list = []
date_list = []

unique_days = df.index.unique()

for i in range(LOOKBACK, len(unique_days)):

    window = df.loc[
        i - LOOKBACK:i,
        :
    ]

    price = window.pivot_table(
        index="date",
        columns="tic",
        values="close",
    )

    price = price.reindex(
        columns=TICKERS
    ).dropna()

    returns = price.pct_change().dropna()

    if returns.shape[1] != len(TICKERS):
        continue

    cov_list.append(
        returns.cov().values
    )

    date_list.append(
        df.loc[i, "date"].iloc[0]
    )


df_cov = pd.DataFrame(
    {
        "date": date_list,
        "cov_list": cov_list,
    }
)

df = df.merge(
    df_cov,
    on="date",
)

df = df.sort_values(
    ["date", "tic"]
).reset_index(drop=True)

df.index = df["date"].factorize()[0]


trade = data_split(
    df,
    TRAIN_END,
    END_DATE,
)

stock_dimension = len(
    trade["tic"].unique()
)

print("Trade rows:", len(trade))
print("Stocks:", stock_dimension)


# ============================================================
# Environment config
# ============================================================

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1_000_000,
    "transaction_cost_pct": 0.001,
    "state_space": stock_dimension,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}


# ============================================================
# Load PPO
# ============================================================

print("\n[4/6] Loading PPO model...")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = PPO.load(
    str(MODEL_PATH),
    device=device,
)

print("Model loaded.")


# ============================================================
# Helper
# ============================================================

def softmax(x):

    x = np.asarray(x, dtype=float)

    e = np.exp(
        x - np.max(x)
    )

    return e / e.sum()


def metrics(r):

    r = pd.Series(r).astype(float)

    cumulative = (
        1 + r
    ).cumprod()

    final_multiple = float(
        cumulative.iloc[-1]
    )

    n = max(
        len(r) - 1,
        1,
    )

    ann_return = (
        final_multiple
        ** (252 / n)
        - 1
    )

    daily_std = r.std()

    ann_vol = (
        daily_std
        * np.sqrt(252)
    )

    if (
        pd.notna(daily_std)
        and daily_std > 0
    ):

        sharpe = (
            np.sqrt(252)
            * r.mean()
            / daily_std
        )

    else:
        sharpe = np.nan

    running_max = (
        cumulative.cummax()
    )

    drawdown = (
        cumulative
        / running_max
        - 1
    )

    mdd = float(
        drawdown.min()
    )

    return {
        "Final Multiple": final_multiple,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": mdd,
    }


# ============================================================
# 5A. Evaluate PPO
# ============================================================

print("\n[5/6] Running PPO policy...")

ppo_env = StockPortfolioEnv(
    df=trade,
    **env_kwargs,
)

obs, info = ppo_env.reset()

terminated = False
truncated = False

raw_actions = []
softmax_actions = []
dates = []

while not (terminated or truncated):

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=float,
    ).reshape(-1)

    raw_actions.append(
        action.copy()
    )

    softmax_actions.append(
        softmax(action)
    )

    dates.append(
        trade.loc[
            ppo_env.day,
            "date"
        ].iloc[0]
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = ppo_env.step(action)


ppo_returns = (
    ppo_env
    .save_asset_memory()
)

ppo_weights = (
    ppo_env
    .save_action_memory()
)


raw_action_df = pd.DataFrame(
    raw_actions,
    columns=TICKERS,
)

raw_action_df.insert(
    0,
    "date",
    dates,
)

softmax_action_df = pd.DataFrame(
    softmax_actions,
    columns=TICKERS,
)

softmax_action_df.insert(
    0,
    "date",
    dates,
)


# ============================================================
# 5B. Evaluate TRUE Equal Weight
#
# Zero raw actions -> softmax -> exactly 20% each
# ============================================================

print("\nRunning Equal Weight policy...")

ew_env = StockPortfolioEnv(
    df=trade,
    **env_kwargs,
)

obs, info = ew_env.reset()

terminated = False
truncated = False

zero_action = np.zeros(
    stock_dimension,
    dtype=float,
)

while not (terminated or truncated):

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = ew_env.step(
        zero_action
    )


ew_returns = (
    ew_env
    .save_asset_memory()
)

ew_weights = (
    ew_env
    .save_action_memory()
)


# ============================================================
# 6. Analysis
# ============================================================

print("\n[6/6] Comparing PPO vs Equal Weight...")


ppo_r = (
    ppo_returns["daily_return"]
    .astype(float)
)

ew_r = (
    ew_returns["daily_return"]
    .astype(float)
)


ppo_metrics = metrics(
    ppo_r
)

ew_metrics = metrics(
    ew_r
)


# ============================================================
# Portfolio weight behavior
# ============================================================

ppo_weight_values = (
    ppo_weights[TICKERS]
)

equal_weight = (
    1 / stock_dimension
)

weight_deviation = (
    ppo_weight_values
    - equal_weight
)

mean_abs_weight_dev = (
    weight_deviation
    .abs()
    .mean()
)

max_abs_weight_dev = (
    weight_deviation
    .abs()
    .max()
)

turnover = (
    ppo_weight_values
    .diff()
    .abs()
    .sum(axis=1)
)


# ============================================================
# Raw action behavior
# ============================================================

raw_only = (
    raw_action_df[TICKERS]
)

raw_action_std = (
    raw_only.std()
)

raw_action_mean = (
    raw_only.mean()
)

# Spread between largest and smallest action each day
raw_action_spread = (
    raw_only.max(axis=1)
    - raw_only.min(axis=1)
)


# ============================================================
# PPO vs Equal Weight daily return difference
# ============================================================

min_len = min(
    len(ppo_r),
    len(ew_r),
)

return_difference = (
    ppo_r.iloc[:min_len].reset_index(drop=True)
    -
    ew_r.iloc[:min_len].reset_index(drop=True)
)

tracking_error_daily = (
    return_difference.std()
)

tracking_error_annual = (
    tracking_error_daily
    * np.sqrt(252)
)


# ============================================================
# Save files
# ============================================================

ppo_returns.to_csv(
    RESULT_DIR / "ppo_returns.csv",
    index=False,
)

ew_returns.to_csv(
    RESULT_DIR / "equal_weight_returns.csv",
    index=False,
)

ppo_weights.to_csv(
    RESULT_DIR / "ppo_weights.csv",
)

ew_weights.to_csv(
    RESULT_DIR / "equal_weight_weights.csv",
)

raw_action_df.to_csv(
    RESULT_DIR / "ppo_raw_actions.csv",
    index=False,
)

softmax_action_df.to_csv(
    RESULT_DIR / "ppo_softmax_actions.csv",
    index=False,
)


# ============================================================
# Print everything
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE")
print("=" * 70)

comparison = pd.DataFrame(
    {
        "PPO": ppo_metrics,
        "Equal Weight": ew_metrics,
    }
)

print(comparison)


print("\n" + "=" * 70)
print("PPO WEIGHT BEHAVIOR")
print("=" * 70)

print("\nMean weights:")
print(
    ppo_weight_values.mean()
)

print("\nWeight standard deviation:")
print(
    ppo_weight_values.std()
)

print("\nMean absolute deviation from 20%:")
print(
    mean_abs_weight_dev
)

print("\nMaximum absolute deviation from 20%:")
print(
    max_abs_weight_dev
)

print("\nAverage daily turnover:")
print(
    turnover.mean()
)


print("\n" + "=" * 70)
print("RAW PPO ACTIONS")
print("=" * 70)

print("\nFirst 10 raw actions:")
print(
    raw_action_df.head(10)
)

print("\nMean raw action:")
print(
    raw_action_mean
)

print("\nRaw action standard deviation:")
print(
    raw_action_std
)

print("\nMean daily raw-action spread:")
print(
    raw_action_spread.mean()
)

print("\nMaximum raw-action spread:")
print(
    raw_action_spread.max()
)


print("\n" + "=" * 70)
print("PPO vs EQUAL WEIGHT DIFFERENCE")
print("=" * 70)

print(
    "Mean daily return difference:",
    return_difference.mean()
)

print(
    "Maximum absolute daily return difference:",
    return_difference.abs().max()
)

print(
    "Annualized tracking error:",
    tracking_error_annual
)


print("\nSaved to:")
print(RESULT_DIR)

print("\nDONE")