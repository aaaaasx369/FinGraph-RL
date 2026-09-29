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
from finrl.agents.stablebaselines3.models import DRLAgent

from src.strategies.rl_model import train_ppo, USE_GPU


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

RESULT_DIR = ROOT / "results" / "ppo_smoke"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = RESULT_DIR / "ppo_model"


print("=" * 70)
print("FinRL-X PPO Portfolio Allocation")
print("=" * 70)

print("Tickers:", TICKERS)
print("Training:", START_DATE, "->", TRAIN_END)
print("Testing :", TRAIN_END, "->", END_DATE)
print("CUDA available:", torch.cuda.is_available())
print("FinRL-X USE_GPU:", USE_GPU)


# ============================================================
# 1. Download market data
# ============================================================

print("\n[1/7] Downloading Yahoo Finance data...")

df = YahooDownloader(
    start_date=START_DATE,
    end_date=END_DATE,
    ticker_list=TICKERS,
).fetch_data()

print("Raw shape:", df.shape)
print("Dates:", df["date"].min(), "->", df["date"].max())


# ============================================================
# 2. Technical indicators
# ============================================================

print("\n[2/7] Adding technical indicators...")

feature_engineer = FeatureEngineer(
    use_technical_indicator=True,
    tech_indicator_list=config.INDICATORS,
    use_vix=False,
    use_turbulence=False,
    user_defined_feature=False,
)

df = feature_engineer.preprocess_data(df)

df = df.sort_values(
    ["date", "tic"]
).reset_index(drop=True)

df.index = df["date"].factorize()[0]

print("Feature-engineered shape:", df.shape)


# ============================================================
# 3. Rolling covariance matrices
# ============================================================

print("\n[3/7] Building rolling covariance matrices...")

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

    if len(price) < 2:
        continue

    returns = price.pct_change().dropna()

    if returns.shape[1] != len(TICKERS):
        continue

    cov_list.append(
        returns.cov().values
    )

    current_date = df.loc[
        i,
        "date"
    ].iloc[0]

    date_list.append(current_date)


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

print("Model-ready shape:", df.shape)
print(
    "Model-ready dates:",
    df["date"].min(),
    "->",
    df["date"].max(),
)


# ============================================================
# 4. Train / Test split
# ============================================================

print("\n[4/7] Splitting train/test data...")

train = data_split(
    df,
    df["date"].min(),
    TRAIN_END,
)

trade = data_split(
    df,
    TRAIN_END,
    END_DATE,
)

stock_dimension = len(
    train["tic"].unique()
)

print("Stocks     :", stock_dimension)
print("Train rows :", len(train))
print("Test rows  :", len(trade))

print(
    "Train:",
    train["date"].min(),
    "->",
    train["date"].max(),
)

print(
    "Test :",
    trade["date"].min(),
    "->",
    trade["date"].max(),
)


# ============================================================
# Environment settings
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
# 5. Load existing PPO or train new PPO
# ============================================================

print("\n[5/7] Preparing PPO model...")

model_zip = Path(
    str(MODEL_PATH) + ".zip"
)

if model_zip.exists():

    print("Existing model found:")
    print(model_zip)

    print("Loading existing PPO model...")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    trained_ppo = PPO.load(
        str(MODEL_PATH),
        device=device,
    )

else:

    print("No existing model found.")
    print("Training PPO from scratch...")

    train_env_raw = StockPortfolioEnv(
        df=train,
        **env_kwargs,
    )

    train_env, _ = (
        train_env_raw.get_sb_env()
    )

    agent = DRLAgent(
        env=train_env
    )

    trained_ppo = train_ppo(
        agent,
        USE_GPU,
    )

    trained_ppo.save(
        str(MODEL_PATH)
    )

    print("Model saved:")
    print(model_zip)


# ============================================================
# 6. Out-of-sample evaluation
# ============================================================

print("\n[6/7] Testing PPO...")

test_env = StockPortfolioEnv(
    df=trade,
    **env_kwargs,
)

obs, info = test_env.reset()

terminated = False
truncated = False
step_count = 0

while not (terminated or truncated):

    action, _ = trained_ppo.predict(
        obs,
        deterministic=True,
    )

    (
        obs,
        reward,
        terminated,
        truncated,
        info,
    ) = test_env.step(action)

    step_count += 1


print("Evaluation steps:", step_count)


# ============================================================
# 7. Extract results BEFORE any reset
# ============================================================

print("\n[7/7] Saving results...")

df_daily_return = (
    test_env.save_asset_memory()
)

df_weights = (
    test_env.save_action_memory()
)


daily_return_path = (
    RESULT_DIR / "daily_returns.csv"
)

weights_path = (
    RESULT_DIR / "portfolio_weights.csv"
)


df_daily_return.to_csv(
    daily_return_path,
    index=False,
)

df_weights.to_csv(
    weights_path,
)


# ============================================================
# Portfolio weights
# ============================================================

print("\n" + "=" * 70)
print("PORTFOLIO WEIGHTS")
print("=" * 70)

print("\nFirst 10 days:")
print(df_weights.head(10))

print("\nLast 5 days:")
print(df_weights.tail())

print(
    "\nNumber of weight rows:",
    len(df_weights),
)

print("\nWeight sums:")
print(
    df_weights.sum(axis=1).head(10)
)


# ============================================================
# Daily returns
# ============================================================

print("\n" + "=" * 70)
print("DAILY RETURNS")
print("=" * 70)

print(df_daily_return.head(10))

print(
    "\nNumber of return rows:",
    len(df_daily_return),
)


# ============================================================
# Performance metrics
# ============================================================

r = (
    df_daily_return["daily_return"]
    .astype(float)
)

cumulative = (
    1 + r
).cumprod()

final_multiple = float(
    cumulative.iloc[-1]
)

n_periods = max(
    len(r) - 1,
    1,
)

annualized_return = (
    final_multiple
    ** (252 / n_periods)
    - 1
)

daily_std = r.std()

annualized_volatility = (
    daily_std * np.sqrt(252)
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


running_max = cumulative.cummax()

drawdown = (
    cumulative / running_max
    - 1
)

max_drawdown = float(
    drawdown.min()
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("PPO OUT-OF-SAMPLE RESULT")
print("=" * 70)

print(
    f"Initial capital          : "
    f"${env_kwargs['initial_amount']:,.2f}"
)

print(
    f"Final portfolio value    : "
    f"${test_env.portfolio_value:,.2f}"
)

print(
    f"Final cumulative multiple: "
    f"{final_multiple:.6f}x"
)

print(
    f"Annualized return        : "
    f"{annualized_return:.2%}"
)

print(
    f"Annualized volatility    : "
    f"{annualized_volatility:.2%}"
)

print(
    f"Sharpe ratio             : "
    f"{sharpe:.4f}"
)

print(
    f"Maximum drawdown         : "
    f"{max_drawdown:.2%}"
)


print("\nSaved files:")
print(model_zip)
print(daily_return_path)
print(weights_path)

print("\nDONE")