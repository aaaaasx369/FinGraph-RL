import sys
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from finrl import config
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.agents.stablebaselines3.models import DRLAgent

from src.strategies.rl_model import (
    train_a2c,
    train_ppo,
    train_ddpg,
    USE_GPU,
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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

# Train: model-ready beginning -> end of 2021
VALID_START = "2022-01-01"

# Validation: 2022
TEST_START = "2023-01-01"

# Final untouched test: 2023
END_DATE = "2024-01-01"

LOOKBACK = 252

RESULT_DIR = ROOT / "results" / "multi_drl"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("FinRL-X MULTI-DRL MODEL SELECTION")
print("=" * 70)

print("Tickers:", TICKERS)
print("Validation:", VALID_START, "->", TEST_START)
print("Final Test :", TEST_START, "->", END_DATE)
print("CUDA available:", torch.cuda.is_available())
print("USE_GPU:", USE_GPU)


# ============================================================
# 1. Download data
# ============================================================

print("\n[1/8] Downloading Yahoo Finance data...")

df = YahooDownloader(
    start_date=START_DATE,
    end_date=END_DATE,
    ticker_list=TICKERS,
).fetch_data()

print("Raw shape:", df.shape)
print(
    "Raw dates:",
    df["date"].min(),
    "->",
    df["date"].max(),
)


# ============================================================
# 2. Feature engineering
# ============================================================

print("\n[2/8] Adding technical indicators...")

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

print("Feature shape:", df.shape)


# ============================================================
# 3. Rolling covariance
# ============================================================

print("\n[3/8] Building covariance matrices...")

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

print("Model-ready shape:", df.shape)
print(
    "Model-ready dates:",
    df["date"].min(),
    "->",
    df["date"].max(),
)


# ============================================================
# 4. Train / Validation / Test
# ============================================================

print("\n[4/8] Splitting data...")

train = data_split(
    df,
    df["date"].min(),
    VALID_START,
)

validation = data_split(
    df,
    VALID_START,
    TEST_START,
)

test = data_split(
    df,
    TEST_START,
    END_DATE,
)

stock_dimension = len(
    train["tic"].unique()
)

print("\nStocks:", stock_dimension)

print(
    "Train:",
    train["date"].min(),
    "->",
    train["date"].max(),
    "rows =",
    len(train),
)

print(
    "Validation:",
    validation["date"].min(),
    "->",
    validation["date"].max(),
    "rows =",
    len(validation),
)

print(
    "Final Test:",
    test["date"].min(),
    "->",
    test["date"].max(),
    "rows =",
    len(test),
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
# Helper: run one complete episode without VecEnv reset problem
# ============================================================

def evaluate_model(model, eval_df):

    env = StockPortfolioEnv(
        df=eval_df,
        **env_kwargs,
    )

    obs, info = env.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated):

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

    returns = env.save_asset_memory()
    weights = env.save_action_memory()

    cumulative = (
        1
        + returns["daily_return"].astype(float)
    ).cumprod()

    final_multiple = float(
        cumulative.iloc[-1]
    )

    return (
        final_multiple,
        returns,
        weights,
        env.portfolio_value,
    )


def calculate_metrics(df_returns):

    r = (
        df_returns["daily_return"]
        .astype(float)
    )

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

    annual_return = (
        final_multiple
        ** (252 / n)
        - 1
    )

    daily_std = r.std()

    annual_volatility = (
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

    max_drawdown = float(
        drawdown.min()
    )

    return {
        "Final Multiple": final_multiple,
        "Annualized Return": annual_return,
        "Annualized Volatility": annual_volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
    }


# ============================================================
# 5. Create training environment
# ============================================================

print("\n[5/8] Creating training environment...")

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


# ============================================================
# 6. Train A2C / PPO / DDPG
# ============================================================

print("\n[6/8] Training A2C...")
a2c_model = train_a2c(
    agent,
    USE_GPU,
)

a2c_model.save(
    str(RESULT_DIR / "a2c_model")
)


print("\n[6/8] Training PPO...")
ppo_model = train_ppo(
    agent,
    USE_GPU,
)

ppo_model.save(
    str(RESULT_DIR / "ppo_model")
)


print("\n[6/8] Training DDPG...")
ddpg_model = train_ddpg(
    agent,
    USE_GPU,
)

ddpg_model.save(
    str(RESULT_DIR / "ddpg_model")
)


# ============================================================
# 7. Validation: choose BEST MODEL
# Same basic rule as FinRL-X run_models:
# highest cumulative return wins.
# ============================================================

print("\n[7/8] Validating all models...")

models = {
    "A2C": a2c_model,
    "PPO": ppo_model,
    "DDPG": ddpg_model,
}

validation_results = {}

for name, model in models.items():

    print(f"\nValidating {name}...")

    (
        multiple,
        returns,
        weights,
        portfolio_value,
    ) = evaluate_model(
        model,
        validation,
    )

    validation_results[name] = {
        "multiple": multiple,
        "returns": returns,
        "weights": weights,
        "portfolio_value": portfolio_value,
    }

    print(
        f"{name} validation multiple:"
        f" {multiple:.6f}x"
    )


# Select highest validation cumulative return
best_name = max(
    validation_results,
    key=lambda name:
        validation_results[name]["multiple"],
)

best_model = models[
    best_name
]


print("\n" + "=" * 70)
print("VALIDATION MODEL SELECTION")
print("=" * 70)

for name in [
    "A2C",
    "PPO",
    "DDPG",
]:
    print(
        f"{name}: "
        f"{validation_results[name]['multiple']:.6f}x"
    )

print("\nBEST MODEL:")
print(best_name)


# Save validation comparison
validation_table = pd.DataFrame(
    {
        name: calculate_metrics(
            validation_results[name]["returns"]
        )
        for name in models
    }
).T

validation_table.to_csv(
    RESULT_DIR / "validation_comparison.csv"
)

print("\nValidation metrics:")
print(validation_table)


# ============================================================
# 8. FINAL OUT-OF-SAMPLE TEST
# Only the winner is evaluated here.
# ============================================================

print("\n[8/8] Running FINAL TEST with best model...")

(
    final_multiple,
    final_returns,
    final_weights,
    final_portfolio_value,
) = evaluate_model(
    best_model,
    test,
)


final_metrics = calculate_metrics(
    final_returns
)


# Save winner outputs
final_returns.to_csv(
    RESULT_DIR
    / "best_model_test_returns.csv",
    index=False,
)

final_weights.to_csv(
    RESULT_DIR
    / "best_model_test_weights.csv",
)


pd.DataFrame(
    [final_metrics],
    index=[best_name],
).to_csv(
    RESULT_DIR
    / "best_model_test_summary.csv"
)


with open(
    RESULT_DIR
    / "best_model_name.txt",
    "w",
) as f:
    f.write(best_name + "\n")


# ============================================================
# Final print
# ============================================================

print("\n" + "=" * 70)
print("FINAL OUT-OF-SAMPLE RESULT")
print("=" * 70)

print("Selected model :", best_name)

print(
    f"Initial capital : "
    f"${env_kwargs['initial_amount']:,.2f}"
)

print(
    f"Final value     : "
    f"${final_portfolio_value:,.2f}"
)

for key, value in final_metrics.items():

    if key == "Final Multiple":
        print(
            f"{key:22s}: "
            f"{value:.6f}x"
        )

    elif key in [
        "Annualized Return",
        "Annualized Volatility",
        "Max Drawdown",
    ]:
        print(
            f"{key:22s}: "
            f"{value:.2%}"
        )

    else:
        print(
            f"{key:22s}: "
            f"{value:.4f}"
        )


print("\nFirst 10 final weights:")
print(
    final_weights.head(10)
)

print(
    "\nNumber of final weight rows:",
    len(final_weights),
)


print("\nSaved to:")
print(RESULT_DIR)

print("\nDONE")
