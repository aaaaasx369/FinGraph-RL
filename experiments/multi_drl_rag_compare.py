import os
import sys
import json
import math
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ============================================================
# Project path
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# FinRL imports
# ============================================================

from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import (
    StockPortfolioEnv,
)
from finrl.meta.preprocessor.preprocessors import (
    FeatureEngineer,
    data_split,
)
from finrl.meta.preprocessor.yahoodownloader import (
    YahooDownloader,
)

from src.strategies.rl_model import (
    train_a2c,
    train_ppo,
    train_ddpg,
    USE_GPU,
)


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=[
        "both",
        "baseline",
        "rag",
    ],
    default="both",
)

args = parser.parse_args()


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

TIC_ORDER = sorted(TICKERS)

# Download one year earlier so covariance has warm-up data.
DOWNLOAD_START = "2017-01-01"
DOWNLOAD_END = "2024-01-01"

# Experimental boundaries
TRAIN_END = "2022-01-01"
VALIDATION_END = "2023-01-01"
TEST_END = "2024-01-01"

INITIAL_AMOUNT = 1_000_000
LOOKBACK = 252

BASE_FEATURES = list(
    config.INDICATORS
)

RAG_FEATURES = [
    "impact_score",
    "risk_score",
    "confidence",
]

MODEL_SEEDS = {
    "A2C": 101,
    "PPO": 202,
    "DDPG": 303,
}


# ============================================================
# Paths
# ============================================================

RAG_EVENT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "sec_rag_events.csv"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Covariance feature
#
# StockPortfolioEnv expects:
#
# covariance matrix
# +
# rows of technical features
# ============================================================

def add_covariance_matrix(
    df,
    lookback=252,
):

    print("\nBuilding rolling covariance matrices...")

    df = df.copy()

    df = df.sort_values(
        [
            "date",
            "tic",
        ]
    ).reset_index(
        drop=True
    )

    price = df.pivot(
        index="date",
        columns="tic",
        values="close",
    )

    # Force exactly the same ticker order everywhere.
    price = price.reindex(
        columns=TIC_ORDER
    )

    dates = list(
        price.index
    )

    cov_rows = []

    for i in range(
        lookback,
        len(dates),
    ):

        window = price.iloc[
            i - lookback:
            i
        ]

        returns = (
            window
            .pct_change()
            .dropna(
                how="any"
            )
        )

        if returns.empty:
            continue

        if (
            returns.shape[1]
            != len(TIC_ORDER)
        ):
            continue

        cov = (
            returns
            .cov()
            .reindex(
                index=TIC_ORDER,
                columns=TIC_ORDER,
            )
            .values
        )

        cov_rows.append(
            {
                "date":
                    dates[i],

                "cov_list":
                    cov,
            }
        )

    cov_df = pd.DataFrame(
        cov_rows
    )

    df = df.merge(
        cov_df,
        on="date",
        how="inner",
    )

    df = df.sort_values(
        [
            "date",
            "tic",
        ]
    ).reset_index(
        drop=True
    )

    print(
        "Covariance-ready dates:",
        df["date"].nunique(),
    )

    return df


# ============================================================
# Market data
# ============================================================

def build_market_data():

    print("\n" + "=" * 70)
    print("DOWNLOAD + FEATURE ENGINEERING")
    print("=" * 70)

    print(
        "Tickers:",
        TIC_ORDER,
    )

    raw = YahooDownloader(
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        ticker_list=TIC_ORDER,
    ).fetch_data()

    if raw.empty:
        raise RuntimeError(
            "YahooDownloader returned no data."
        )

    print(
        "Raw rows:",
        len(raw),
    )

    print(
        "Raw dates:",
        raw["date"].nunique(),
    )

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=BASE_FEATURES,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )

    market = fe.preprocess_data(
        raw
    )

    market = market[
        market["tic"].isin(
            TIC_ORDER
        )
    ].copy()

    market = market.sort_values(
        [
            "date",
            "tic",
        ]
    ).reset_index(
        drop=True
    )

    # Need exactly five stocks each market day.
    counts = market.groupby(
        "date"
    )["tic"].nunique()

    good_dates = counts[
        counts == len(TIC_ORDER)
    ].index

    market = market[
        market["date"].isin(
            good_dates
        )
    ].copy()

    market = add_covariance_matrix(
        market,
        lookback=LOOKBACK,
    )

    market["date_dt"] = pd.to_datetime(
        market["date"]
    )

    print(
        "Model-ready date range:",
        market["date"].min(),
        "->",
        market["date"].max(),
    )

    return market


# ============================================================
# Strict point-in-time RAG merge
#
# CRITICAL:
#
# allow_exact_matches=False
#
# means:
#
# filing date = t
# signal becomes usable only AFTER t.
#
# No same-day leakage.
# ============================================================

def merge_rag_features(
    market,
):

    print("\n" + "=" * 70)
    print("STRICT POINT-IN-TIME RAG MERGE")
    print("=" * 70)

    if not RAG_EVENT_PATH.exists():

        raise FileNotFoundError(
            f"Missing RAG event file:\n"
            f"{RAG_EVENT_PATH}"
        )

    events = pd.read_csv(
        RAG_EVENT_PATH
    )

    events = events[
        events["status"] == "ok"
    ].copy()

    events["ticker"] = (
        events["ticker"]
        .astype(str)
        .str.upper()
    )

    events[
        "source_filing_date"
    ] = pd.to_datetime(
        events["event_date"]
    )

    events = events[
        [
            "ticker",
            "source_filing_date",
            "impact_score",
            "risk_score",
            "confidence",
            "direction",
            "reason",
        ]
    ].copy()

    merged_parts = []

    for ticker in TIC_ORDER:

        market_ticker = market[
            market["tic"] == ticker
        ].copy()

        market_ticker = market_ticker.sort_values(
            "date_dt"
        )

        signal_ticker = events[
            events["ticker"] == ticker
        ].copy()

        signal_ticker = signal_ticker.sort_values(
            "source_filing_date"
        )

        # Remove ticker because market already has tic.
        signal_ticker = signal_ticker.drop(
            columns=[
                "ticker",
            ]
        )

        temp = pd.merge_asof(
            market_ticker,
            signal_ticker,

            left_on="date_dt",
            right_on="source_filing_date",

            direction="backward",

            # THIS is the key leakage protection.
            allow_exact_matches=False,
        )

        merged_parts.append(
            temp
        )

    rag = pd.concat(
        merged_parts,
        ignore_index=True,
    )

    rag = rag.sort_values(
        [
            "date_dt",
            "tic",
        ]
    ).reset_index(
        drop=True
    )


    # ========================================================
    # Hard leakage check
    # ========================================================

    nonmissing = rag.dropna(
        subset=[
            "source_filing_date",
        ]
    )

    leakage = nonmissing[
        nonmissing[
            "source_filing_date"
        ]
        >=
        nonmissing[
            "date_dt"
        ]
    ]

    print(
        "Same-day/future leakage rows:",
        len(leakage),
    )

    if len(leakage) != 0:

        raise RuntimeError(
            "RAG leakage detected."
        )


    # ========================================================
    # Find first date where ALL 5 stocks have historical
    # RAG signals.
    # ========================================================

    date_quality = (
        rag.groupby(
            "date_dt"
        )
        .agg(
            ticker_count=(
                "tic",
                "nunique",
            ),

            impact_count=(
                "impact_score",
                "count",
            ),

            risk_count=(
                "risk_score",
                "count",
            ),

            confidence_count=(
                "confidence",
                "count",
            ),
        )
    )

    valid_dates = date_quality[
        (
            date_quality[
                "ticker_count"
            ]
            == len(TIC_ORDER)
        )
        &
        (
            date_quality[
                "impact_count"
            ]
            == len(TIC_ORDER)
        )
        &
        (
            date_quality[
                "risk_count"
            ]
            == len(TIC_ORDER)
        )
        &
        (
            date_quality[
                "confidence_count"
            ]
            == len(TIC_ORDER)
        )
    ]

    if valid_dates.empty:

        raise RuntimeError(
            "No date has complete RAG features "
            "for all five stocks."
        )

    common_start = (
        valid_dates
        .index
        .min()
    )

    print(
        "First complete strict-RAG trading date:",
        common_start.date(),
    )


    # ========================================================
    # Restrict BOTH experiments to the same dates.
    # ========================================================

    rag = rag[
        rag["date_dt"]
        >= common_start
    ].copy()

    rag = rag[
        rag["date_dt"]
        < pd.Timestamp(
            TEST_END
        )
    ].copy()


    # No missing signals allowed after common start.
    missing = rag[
        RAG_FEATURES
    ].isna().sum()

    print(
        "\nMissing RAG values after common start:"
    )

    print(
        missing
    )

    if (
        missing.sum()
        != 0
    ):

        raise RuntimeError(
            "Missing RAG features remain "
            "after common start."
        )


    # Save merge audit
    audit = rag[
        [
            "date",
            "tic",
            "source_filing_date",
            "impact_score",
            "risk_score",
            "confidence",
            "direction",
        ]
    ].copy()

    audit.to_csv(
        RESULT_ROOT
        / "rag_merge_audit.csv",
        index=False,
    )

    return rag, common_start


# ============================================================
# Build identical baseline panel
# ============================================================

def build_baseline_panel(
    market,
    common_start,
):

    baseline = market[
        market["date_dt"]
        >= common_start
    ].copy()

    baseline = baseline[
        baseline["date_dt"]
        < pd.Timestamp(
            TEST_END
        )
    ].copy()

    baseline = baseline.sort_values(
        [
            "date",
            "tic",
        ]
    ).reset_index(
        drop=True
    )

    return baseline


# ============================================================
# Panel validation
# ============================================================

def validate_panel(
    df,
    label,
):

    counts = (
        df.groupby(
            "date"
        )["tic"]
        .nunique()
    )

    bad = counts[
        counts
        != len(TIC_ORDER)
    ]

    if len(bad):

        raise RuntimeError(
            f"{label}: dates with incomplete ticker panel."
        )

    print(
        f"{label}:",
        df["date"].nunique(),
        "dates,",
        len(df),
        "rows",
    )


# ============================================================
# Environment kwargs
# ============================================================

def make_env_kwargs(
    feature_list,
):

    return {
        "hmax":
            100,

        "initial_amount":
            INITIAL_AMOUNT,

        "transaction_cost_pct":
            0.001,

        "reward_scaling":
            1e-4,

        "state_space":
            len(TIC_ORDER),

        "stock_dim":
            len(TIC_ORDER),

        "action_space":
            len(TIC_ORDER),

        "tech_indicator_list":
            feature_list,
    }


# ============================================================
# Raw environment evaluation
#
# We intentionally avoid DRL_prediction() because the old
# StockPortfolioEnv + VecEnv terminal reset can lose the
# episode memory.
# ============================================================

def evaluate_model(
    model,
    df,
    feature_list,
):

    env = StockPortfolioEnv(
        df=df,
        **make_env_kwargs(
            feature_list
        ),
    )

    obs, _ = env.reset()

    while True:

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
        ) = env.step(
            action
        )

        if (
            terminated
            or truncated
        ):
            break


    returns = (
        env
        .save_asset_memory()
        .copy()
    )

    weights = (
        env
        .save_action_memory()
        .copy()
    )

    return (
        returns,
        weights,
    )


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    returns_df,
):

    r = (
        returns_df[
            "daily_return"
        ]
        .astype(float)
        .copy()
    )

    # First row is environment initialization = 0.
    if (
        len(r) > 0
        and abs(r.iloc[0]) < 1e-15
    ):
        r = r.iloc[
            1:
        ].copy()

    n = len(r)

    if n == 0:

        raise RuntimeError(
            "No test returns."
        )


    wealth = (
        1.0 + r
    ).cumprod()

    final_multiple = float(
        wealth.iloc[-1]
    )

    final_value = (
        INITIAL_AMOUNT
        * final_multiple
    )


    annual_return = (
        final_multiple
        ** (
            252.0
            / n
        )
        - 1.0
    )


    std_daily = float(
        r.std(
            ddof=1
        )
    )

    annual_vol = (
        std_daily
        * math.sqrt(252)
    )


    if std_daily > 0:

        sharpe = (
            float(
                r.mean()
            )
            / std_daily
            * math.sqrt(252)
        )

    else:

        sharpe = np.nan


    drawdown = (
        wealth
        / wealth.cummax()
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )


    return {
        "final_value":
            final_value,

        "final_multiple":
            final_multiple,

        "annual_return":
            annual_return,

        "annual_volatility":
            annual_vol,

        "sharpe":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "n_return_days":
            n,
    }


# ============================================================
# Train one algorithm
# ============================================================

def train_algorithm(
    model_name,
    train_df,
    feature_list,
):

    seed = MODEL_SEEDS[
        model_name
    ]

    set_seed(
        seed
    )

    e_train = StockPortfolioEnv(
        df=train_df,
        **make_env_kwargs(
            feature_list
        ),
    )

    env_train, _ = (
        e_train.get_sb_env()
    )

    agent = DRLAgent(
        env=env_train
    )


    if model_name == "A2C":

        model = train_a2c(
            agent,
            USE_GPU,
        )

    elif model_name == "PPO":

        model = train_ppo(
            agent,
            USE_GPU,
        )

    elif model_name == "DDPG":

        model = train_ddpg(
            agent,
            USE_GPU,
        )

    else:

        raise ValueError(
            model_name
        )

    return model


# ============================================================
# One full pipeline:
#
# train A2C/PPO/DDPG
# -> validation
# -> select highest cumulative multiple
# -> final untouched test
# ============================================================

def run_pipeline(
    name,
    panel,
    feature_list,
    common_start,
):

    print("\n" + "#" * 70)
    print(
        "PIPELINE:",
        name,
    )
    print("#" * 70)

    output_dir = (
        RESULT_ROOT
        / name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    start_string = (
        common_start
        .strftime(
            "%Y-%m-%d"
        )
    )


    train_df = data_split(
        panel,
        start_string,
        TRAIN_END,
    )

    validation_df = data_split(
        panel,
        TRAIN_END,
        VALIDATION_END,
    )

    test_df = data_split(
        panel,
        VALIDATION_END,
        TEST_END,
    )


    print(
        "\nTRAIN:",
        train_df[
            "date"
        ].min(),
        "->",
        train_df[
            "date"
        ].max(),
        "| dates:",
        train_df[
            "date"
        ].nunique(),
    )

    print(
        "VALIDATION:",
        validation_df[
            "date"
        ].min(),
        "->",
        validation_df[
            "date"
        ].max(),
        "| dates:",
        validation_df[
            "date"
        ].nunique(),
    )

    print(
        "TEST:",
        test_df[
            "date"
        ].min(),
        "->",
        test_df[
            "date"
        ].max(),
        "| dates:",
        test_df[
            "date"
        ].nunique(),
    )


    models = {}

    validation_rows = []


    for model_name in [
        "A2C",
        "PPO",
        "DDPG",
    ]:

        print(
            "\n"
            + "=" * 70
        )

        print(
            name,
            "- TRAIN",
            model_name,
        )

        print(
            "=" * 70
        )


        model = train_algorithm(
            model_name,
            train_df,
            feature_list,
        )

        models[
            model_name
        ] = model


        model.save(
            str(
                output_dir
                / (
                    model_name
                    .lower()
                    + "_model"
                )
            )
        )


        # -----------------------------------------------
        # Validation only
        # -----------------------------------------------

        val_returns, val_weights = (
            evaluate_model(
                model,
                validation_df,
                feature_list,
            )
        )


        metrics = calculate_metrics(
            val_returns
        )


        metrics[
            "model"
        ] = model_name


        validation_rows.append(
            metrics
        )


        val_returns.to_csv(
            output_dir
            / (
                model_name
                .lower()
                + "_validation_returns.csv"
            ),
            index=False,
        )


        val_weights.to_csv(
            output_dir
            / (
                model_name
                .lower()
                + "_validation_weights.csv"
            ),
        )


        print(
            model_name,
            "validation multiple:",
            f"{metrics['final_multiple']:.6f}",
        )

        print(
            model_name,
            "validation Sharpe:",
            f"{metrics['sharpe']:.4f}",
        )


    # ========================================================
    # Select winner using VALIDATION ONLY
    # ========================================================

    validation_table = pd.DataFrame(
        validation_rows
    )

    validation_table = (
        validation_table
        .sort_values(
            "final_multiple",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    validation_table.to_csv(
        output_dir
        / "validation_comparison.csv",
        index=False,
    )


    best_model_name = (
        validation_table
        .iloc[0][
            "model"
        ]
    )


    print(
        "\nBEST MODEL BY VALIDATION RETURN:",
        best_model_name,
    )


    with open(
        output_dir
        / "best_model_name.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            str(
                best_model_name
            )
        )


    # ========================================================
    # FINAL TEST
    #
    # Test set has not been used for model selection.
    # ========================================================

    best_model = models[
        best_model_name
    ]


    test_returns, test_weights = (
        evaluate_model(
            best_model,
            test_df,
            feature_list,
        )
    )


    test_metrics = calculate_metrics(
        test_returns
    )


    test_metrics[
        "pipeline"
    ] = name

    test_metrics[
        "selected_model"
    ] = best_model_name


    test_returns.to_csv(
        output_dir
        / "best_model_test_returns.csv",
        index=False,
    )


    test_weights.to_csv(
        output_dir
        / "best_model_test_weights.csv",
    )


    pd.DataFrame(
        [
            test_metrics
        ]
    ).to_csv(
        output_dir
        / "best_model_test_summary.csv",
        index=False,
    )


    print(
        "\nFINAL TEST"
    )

    print(
        "Selected model :",
        best_model_name,
    )

    print(
        "Final value    :",
        f"${test_metrics['final_value']:,.2f}",
    )

    print(
        "Multiple       :",
        f"{test_metrics['final_multiple']:.6f}x",
    )

    print(
        "Annual return  :",
        f"{test_metrics['annual_return']:.2%}",
    )

    print(
        "Annual vol     :",
        f"{test_metrics['annual_volatility']:.2%}",
    )

    print(
        "Sharpe         :",
        f"{test_metrics['sharpe']:.4f}",
    )

    print(
        "Max drawdown   :",
        f"{test_metrics['max_drawdown']:.2%}",
    )


    return {
        "pipeline":
            name,

        "selected_model":
            best_model_name,

        **test_metrics,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-DRL BASELINE vs SEC-RAG")
    print("=" * 70)

    print(
        "GPU available:",
        torch.cuda.is_available(),
    )

    print(
        "FinRL USE_GPU:",
        USE_GPU,
    )


    # ========================================================
    # Market data
    # ========================================================

    market = build_market_data()


    # ========================================================
    # RAG strict merge
    # ========================================================

    rag_panel, common_start = (
        merge_rag_features(
            market
        )
    )


    baseline_panel = (
        build_baseline_panel(
            market,
            common_start,
        )
    )


    validate_panel(
        baseline_panel,
        "BASELINE",
    )

    validate_panel(
        rag_panel,
        "RAG",
    )


    # ========================================================
    # Confirm exact same market dates
    # ========================================================

    baseline_dates = list(
        baseline_panel[
            "date"
        ].unique()
    )

    rag_dates = list(
        rag_panel[
            "date"
        ].unique()
    )


    if (
        baseline_dates
        != rag_dates
    ):

        raise RuntimeError(
            "Baseline and RAG dates differ."
        )


    print(
        "\nCommon experiment start:",
        common_start.date(),
    )

    print(
        "Common number of dates:",
        len(
            baseline_dates
        ),
    )


    # ========================================================
    # Save experiment configuration
    # ========================================================

    config_out = {
        "tickers":
            TIC_ORDER,

        "common_start":
            str(
                common_start.date()
            ),

        "train_end_exclusive":
            TRAIN_END,

        "validation_end_exclusive":
            VALIDATION_END,

        "test_end_exclusive":
            TEST_END,

        "base_features":
            BASE_FEATURES,

        "rag_features":
            RAG_FEATURES,

        "strict_same_day_leakage_protection":
            True,

        "allow_exact_matches":
            False,

        "model_seeds":
            MODEL_SEEDS,
    }


    with open(
        RESULT_ROOT
        / "experiment_config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config_out,
            f,
            indent=2,
        )


    summaries = []


    # ========================================================
    # BASELINE
    # ========================================================

    if args.mode in {
        "both",
        "baseline",
    }:

        baseline_result = (
            run_pipeline(
                name="baseline",
                panel=baseline_panel,
                feature_list=BASE_FEATURES,
                common_start=common_start,
            )
        )

        summaries.append(
            baseline_result
        )


    # ========================================================
    # RAG
    # ========================================================

    if args.mode in {
        "both",
        "rag",
    }:

        rag_feature_list = (
            BASE_FEATURES
            + RAG_FEATURES
        )


        rag_result = (
            run_pipeline(
                name="rag",
                panel=rag_panel,
                feature_list=rag_feature_list,
                common_start=common_start,
            )
        )

        summaries.append(
            rag_result
        )


    # ========================================================
    # Final side-by-side table
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )


    summary_df.to_csv(
        RESULT_ROOT
        / "final_comparison.csv",
        index=False,
    )


    if len(summary_df) == 2:

        baseline_row = summary_df[
            summary_df[
                "pipeline"
            ]
            == "baseline"
        ].iloc[0]

        rag_row = summary_df[
            summary_df[
                "pipeline"
            ]
            == "rag"
        ].iloc[0]


        delta = {
            "delta_final_multiple":
                float(
                    rag_row[
                        "final_multiple"
                    ]
                    -
                    baseline_row[
                        "final_multiple"
                    ]
                ),

            "delta_annual_return":
                float(
                    rag_row[
                        "annual_return"
                    ]
                    -
                    baseline_row[
                        "annual_return"
                    ]
                ),

            "delta_sharpe":
                float(
                    rag_row[
                        "sharpe"
                    ]
                    -
                    baseline_row[
                        "sharpe"
                    ]
                ),

            "delta_max_drawdown":
                float(
                    rag_row[
                        "max_drawdown"
                    ]
                    -
                    baseline_row[
                        "max_drawdown"
                    ]
                ),
        }


        pd.DataFrame(
            [
                delta
            ]
        ).to_csv(
            RESULT_ROOT
            / "rag_minus_baseline.csv",
            index=False,
        )


        print(
            "\n"
            + "=" * 70
        )

        print(
            "RAG - BASELINE"
        )

        print(
            "=" * 70
        )

        print(
            "Δ Final multiple:",
            f"{delta['delta_final_multiple']:+.6f}",
        )

        print(
            "Δ Annual return :",
            f"{delta['delta_annual_return']:+.2%}",
        )

        print(
            "Δ Sharpe        :",
            f"{delta['delta_sharpe']:+.4f}",
        )

        print(
            "Δ Max drawdown  :",
            f"{delta['delta_max_drawdown']:+.2%}",
        )


    print(
        "\nSaved results to:"
    )

    print(
        RESULT_ROOT
    )

    print(
        "\nDONE"
    )


if __name__ == "__main__":
    main()