import argparse
import contextlib
import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import A2C, PPO, DDPG


# ============================================================
# Project
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# FinRL
# ============================================================

from finrl import config

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


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--permutations",
    type=int,
    default=10,
)

args = parser.parse_args()

N_PERMUTATIONS = args.permutations


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

DOWNLOAD_START = "2017-01-01"
DOWNLOAD_END = "2024-01-01"

LOOKBACK = 252

INITIAL_AMOUNT = 1_000_000

TECH_FEATURES = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]

RAG_FEATURES = [
    "impact_score",
    "risk_score",
    "confidence",
]

ALL_SCALAR_FEATURES = (
    TECH_FEATURES
    + RAG_FEATURES
)

ENV_FEATURES = (
    TECH_FEATURES
    + RAG_FEATURES
)

SEED = 20260913


# ============================================================
# Paths
# ============================================================

EVENT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "sec_rag_events.csv"
)

DRL_ROOT = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
)

CONFIG_PATH = (
    DRL_ROOT
    / "experiment_config.json"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "all_feature_importance"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Verify indicators
# ============================================================

print("=" * 70)
print("FEATURES")
print("=" * 70)

print("Technical:")
for x in TECH_FEATURES:
    print(" ", x)

print("RAG:")
for x in RAG_FEATURES:
    print(" ", x)

print("Special block:")
print("  covariance")


# ============================================================
# Load RAG events
# ============================================================

events = pd.read_csv(
    EVENT_PATH
)

events = events[
    events["status"] == "ok"
].copy()

events["ticker"] = (
    events["ticker"]
    .astype(str)
    .str.upper()
)

events["event_date"] = (
    events["event_date"]
    .astype(str)
)

for col in RAG_FEATURES:

    events[col] = pd.to_numeric(
        events[col],
        errors="coerce",
    )


if events[
    RAG_FEATURES
].isna().any().any():

    raise RuntimeError(
        "Missing RAG feature values."
    )


# ============================================================
# Covariance
# ============================================================

def add_covariance_matrix(
    df,
    lookback=252,
):

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

    price = price.reindex(
        columns=TIC_ORDER
    )

    dates = list(
        price.index
    )

    rows = []


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

        cov = (
            returns
            .cov()
            .reindex(
                index=TIC_ORDER,
                columns=TIC_ORDER,
            )
            .values
        )

        rows.append(
            {
                "date":
                    dates[i],

                "cov_list":
                    cov,
            }
        )


    cov_df = pd.DataFrame(
        rows
    )

    output = df.merge(
        cov_df,
        on="date",
        how="inner",
    )

    return (
        output
        .sort_values(
            [
                "date",
                "tic",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Market data
# ============================================================

def build_market():

    print("\n" + "=" * 70)
    print("BUILD MARKET DATA")
    print("=" * 70)

    raw = YahooDownloader(
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        ticker_list=TIC_ORDER,
    ).fetch_data()

    if raw.empty:
        raise RuntimeError(
            "No Yahoo data."
        )

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECH_FEATURES,
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
        LOOKBACK,
    )

    market["date_dt"] = pd.to_datetime(
        market["date"]
    )

    return market


# ============================================================
# Merge point-in-time RAG
# ============================================================

def merge_rag(
    market,
    event_table,
    common_start,
    test_end,
):

    event_table = event_table.copy()

    event_table[
        "source_filing_date"
    ] = pd.to_datetime(
        event_table[
            "event_date"
        ]
    )

    parts = []


    for ticker in TIC_ORDER:

        m = market[
            market["tic"] == ticker
        ].copy()

        e = event_table[
            event_table["ticker"] == ticker
        ][
            [
                "source_filing_date",
                "impact_score",
                "risk_score",
                "confidence",
            ]
        ].copy()

        m = m.sort_values(
            "date_dt"
        )

        e = e.sort_values(
            "source_filing_date"
        )

        merged = pd.merge_asof(
            m,
            e,
            left_on="date_dt",
            right_on="source_filing_date",
            direction="backward",
            allow_exact_matches=False,
        )

        parts.append(
            merged
        )


    panel = pd.concat(
        parts,
        ignore_index=True,
    )

    panel = panel[
        panel["date_dt"]
        >= pd.Timestamp(
            common_start
        )
    ].copy()

    panel = panel[
        panel["date_dt"]
        <
        pd.Timestamp(
            test_end
        )
    ].copy()

    panel = (
        panel
        .sort_values(
            [
                "date",
                "tic",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if (
        panel[
            RAG_FEATURES
        ]
        .isna()
        .sum()
        .sum()
        != 0
    ):

        raise RuntimeError(
            "Missing RAG features."
        )


    leakage = panel[
        panel["source_filing_date"]
        >=
        panel["date_dt"]
    ]

    if len(leakage):

        raise RuntimeError(
            "Point-in-time leakage."
        )


    return panel


# ============================================================
# Environment
# ============================================================

def env_kwargs():

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
            ENV_FEATURES,
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    df,
    quiet=True,
):

    def run():

        env = StockPortfolioEnv(
            df=df,
            **env_kwargs(),
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

        return (
            env
            .save_asset_memory()
            .copy(),

            env
            .save_action_memory()
            .copy(),
        )


    if quiet:

        buffer = io.StringIO()

        with contextlib.redirect_stdout(
            buffer
        ):
            return run()

    return run()


# ============================================================
# Metrics
# ============================================================

def performance_metrics(
    returns_df,
):

    r = (
        returns_df[
            "daily_return"
        ]
        .astype(float)
        .copy()
    )

    if (
        len(r)
        and
        abs(
            r.iloc[0]
        )
        < 1e-15
    ):

        r = r.iloc[
            1:
        ]

    n = len(r)

    wealth = (
        1.0 + r
    ).cumprod()

    multiple = float(
        wealth.iloc[-1]
    )

    annual_return = (
        multiple
        ** (
            252.0 / n
        )
        - 1.0
    )

    std = float(
        r.std(
            ddof=1
        )
    )

    annual_vol = (
        std
        * math.sqrt(252)
    )

    sharpe = (
        float(
            r.mean()
        )
        / std
        * math.sqrt(252)
        if std > 0
        else np.nan
    )

    dd = (
        wealth
        / wealth.cummax()
        - 1.0
    )

    return {
        "final_multiple":
            multiple,

        "annual_return":
            annual_return,

        "annual_volatility":
            annual_vol,

        "sharpe":
            sharpe,

        "max_drawdown":
            float(
                dd.min()
            ),
    }


# ============================================================
# Clean weight table
# ============================================================

def get_weights(
    weights,
):

    return (
        weights[
            TIC_ORDER
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# Policy distance
# ============================================================

def policy_distance(
    ref_weights,
    new_weights,
):

    ref = get_weights(
        ref_weights
    )

    new = get_weights(
        new_weights
    )

    n = min(
        len(ref),
        len(new),
    )

    ref = ref.iloc[
        :n
    ]

    new = new.iloc[
        :n
    ]

    diff = (
        new
        - ref
    )

    abs_diff = (
        diff.abs()
    )

    daily_l1 = (
        abs_diff
        .sum(
            axis=1
        )
    )

    return {
        "mean_abs_weight_diff":
            float(
                abs_diff
                .to_numpy()
                .mean()
            ),

        "average_daily_l1":
            float(
                daily_l1.mean()
            ),

        "average_reallocated_weight":
            float(
                daily_l1.mean()
                / 2.0
            ),

        "maximum_daily_l1":
            float(
                daily_l1.max()
            ),

        "maximum_reallocated_weight":
            float(
                daily_l1.max()
                / 2.0
            ),

        "maximum_single_asset_diff":
            float(
                abs_diff
                .to_numpy()
                .max()
            ),
    }


# ============================================================
# Shuffle scalar feature through time
# within EACH STOCK.
#
# Keeps feature distribution but destroys time information.
# ============================================================

def shuffle_scalar_feature(
    df,
    feature,
    seed,
):

    out = df.copy()

    rng = np.random.default_rng(
        seed
    )

    feature_col = (
        out.columns.get_loc(
            feature
        )
    )

    tic_array = (
        out["tic"]
        .to_numpy()
    )


    for ticker in TIC_ORDER:

        positions = np.where(
            tic_array == ticker
        )[0]

        values = (
            out.iloc[
                positions,
                feature_col
            ]
            .to_numpy(
                copy=True
            )
        )

        permutation = rng.permutation(
            len(values)
        )

        out.iloc[
            positions,
            feature_col
        ] = values[
            permutation
        ]


    return out


# ============================================================
# Shuffle entire 5x5 covariance block BY DATE.
#
# The complete matrix is moved together.
# Every stock row on the same date receives the same matrix.
# ============================================================

def shuffle_covariance(
    df,
    seed,
):

    out = df.copy()

    rng = np.random.default_rng(
        seed
    )

    unique_dates = list(
        out["date"].unique()
    )


    matrices = []

    for date in unique_dates:

        matrix = (
            out.loc[
                out["date"] == date,
                "cov_list",
            ]
            .iloc[0]
        )

        matrices.append(
            np.array(
                matrix,
                copy=True,
            )
        )


    permutation = rng.permutation(
        len(unique_dates)
    )


    shuffled_matrices = [
        matrices[i]
        for i in permutation
    ]


    mapping = {
        date:
            matrix
        for date, matrix
        in zip(
            unique_dates,
            shuffled_matrices,
        )
    }


    out["cov_list"] = (
        out["date"]
        .map(
            mapping
        )
    )


    return out


# ============================================================
# Load experiment config
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    experiment_config = json.load(
        f
    )


COMMON_START = (
    experiment_config[
        "common_start"
    ]
)

TEST_START = (
    experiment_config[
        "validation_end_exclusive"
    ]
)

TEST_END = (
    experiment_config[
        "test_end_exclusive"
    ]
)


# ============================================================
# Build clean test dataset
# ============================================================

market = build_market()

panel = merge_rag(
    market,
    events,
    COMMON_START,
    TEST_END,
)

test_df = data_split(
    panel,
    TEST_START,
    TEST_END,
)


print("\n" + "=" * 70)
print("TEST DATA")
print("=" * 70)

print(
    "Dates:",
    test_df["date"].nunique(),
)

print(
    "Rows:",
    len(test_df),
)


# ============================================================
# Load fixed Clean-RAG model
# ============================================================

best_model_name = (
    (
        DRL_ROOT
        / "rag"
        / "best_model_name.txt"
    )
    .read_text()
    .strip()
    .upper()
)


MODEL_CLASSES = {
    "A2C":
        A2C,

    "PPO":
        PPO,

    "DDPG":
        DDPG,
}


model_path = (
    DRL_ROOT
    / "rag"
    / (
        best_model_name.lower()
        + "_model.zip"
    )
)


print("\n" + "=" * 70)
print("LOAD FIXED MODEL")
print("=" * 70)

print(
    "Model:",
    best_model_name,
)


model = MODEL_CLASSES[
    best_model_name
].load(
    str(
        model_path
    )
)


# ============================================================
# Normal reference
# ============================================================

print("\n" + "=" * 70)
print("NORMAL REFERENCE")
print("=" * 70)

normal_returns, normal_weights = (
    evaluate(
        model,
        test_df,
        quiet=False,
    )
)

normal_perf = performance_metrics(
    normal_returns
)


print(
    "Multiple:",
    f"{normal_perf['final_multiple']:.6f}",
)

print(
    "Sharpe:",
    f"{normal_perf['sharpe']:.4f}",
)


# ============================================================
# Feature groups
# ============================================================

FEATURE_GROUPS = [
    "covariance",
    *TECH_FEATURES,
    *RAG_FEATURES,
]


# ============================================================
# Repeated permutation importance
# ============================================================

print("\n" + "=" * 70)
print("ALL-FEATURE PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    "Features:",
    len(FEATURE_GROUPS),
)

print(
    "Permutations each:",
    N_PERMUTATIONS,
)


raw_rows = []


for feature_index, feature in enumerate(
    FEATURE_GROUPS
):

    print("\n" + "-" * 70)

    print(
        f"{feature_index + 1}/"
        f"{len(FEATURE_GROUPS)}",
        feature,
    )

    print("-" * 70)


    for repetition in range(
        N_PERMUTATIONS
    ):

        seed = (
            SEED
            + feature_index * 1000
            + repetition
            + 1
        )


        if feature == "covariance":

            modified_df = (
                shuffle_covariance(
                    test_df,
                    seed,
                )
            )

        else:

            modified_df = (
                shuffle_scalar_feature(
                    test_df,
                    feature,
                    seed,
                )
            )


        (
            modified_returns,
            modified_weights,
        ) = evaluate(
            model,
            modified_df,
            quiet=True,
        )


        perf = performance_metrics(
            modified_returns
        )


        shift = policy_distance(
            normal_weights,
            modified_weights,
        )


        raw_rows.append(
            {
                "feature":
                    feature,

                "repetition":
                    repetition + 1,

                "seed":
                    seed,

                **shift,

                "final_multiple":
                    perf[
                        "final_multiple"
                    ],

                "delta_final_multiple":
                    (
                        perf[
                            "final_multiple"
                        ]
                        -
                        normal_perf[
                            "final_multiple"
                        ]
                    ),

                "sharpe":
                    perf[
                        "sharpe"
                    ],

                "delta_sharpe":
                    (
                        perf[
                            "sharpe"
                        ]
                        -
                        normal_perf[
                            "sharpe"
                        ]
                    ),

                "max_drawdown":
                    perf[
                        "max_drawdown"
                    ],

                "delta_max_drawdown":
                    (
                        perf[
                            "max_drawdown"
                        ]
                        -
                        normal_perf[
                            "max_drawdown"
                        ]
                    ),
            }
        )


        print(
            f"  {repetition + 1:02d}/"
            f"{N_PERMUTATIONS}",
            "reallocated=",
            f"{shift['average_reallocated_weight']:.4%}",
            "ΔSharpe=",
            f"{perf['sharpe'] - normal_perf['sharpe']:+.4f}",
        )


# ============================================================
# Save raw runs
# ============================================================

raw_results = pd.DataFrame(
    raw_rows
)


raw_results.to_csv(
    RESULT_DIR
    / "all_feature_permutation_runs.csv",
    index=False,
)


# ============================================================
# Aggregate
# ============================================================

summary_rows = []


for feature in FEATURE_GROUPS:

    temp = raw_results[
        raw_results[
            "feature"
        ]
        == feature
    ]


    # State input size:
    # covariance = 5x5 = 25 numbers
    # every other feature = 5 stocks = 5 numbers
    if feature == "covariance":

        input_elements = (
            len(TIC_ORDER)
            * len(TIC_ORDER)
        )

        feature_type = (
            "5x5 block"
        )

    else:

        input_elements = (
            len(TIC_ORDER)
        )

        feature_type = (
            "feature row"
        )


    mean_reallocated = float(
        temp[
            "average_reallocated_weight"
        ].mean()
    )


    summary_rows.append(
        {
            "feature":
                feature,

            "feature_type":
                feature_type,

            "input_elements":
                input_elements,

            "mean_daily_l1":
                float(
                    temp[
                        "average_daily_l1"
                    ].mean()
                ),

            "std_daily_l1":
                float(
                    temp[
                        "average_daily_l1"
                    ].std(
                        ddof=1
                    )
                ),

            "mean_reallocated_weight":
                mean_reallocated,

            "std_reallocated_weight":
                float(
                    temp[
                        "average_reallocated_weight"
                    ].std(
                        ddof=1
                    )
                ),

            "max_reallocated_weight_mean":
                float(
                    temp[
                        "maximum_reallocated_weight"
                    ].mean()
                ),

            "mean_max_single_asset_diff":
                float(
                    temp[
                        "maximum_single_asset_diff"
                    ].mean()
                ),

            "mean_delta_multiple":
                float(
                    temp[
                        "delta_final_multiple"
                    ].mean()
                ),

            "std_delta_multiple":
                float(
                    temp[
                        "delta_final_multiple"
                    ].std(
                        ddof=1
                    )
                ),

            "mean_delta_sharpe":
                float(
                    temp[
                        "delta_sharpe"
                    ].mean()
                ),

            "std_delta_sharpe":
                float(
                    temp[
                        "delta_sharpe"
                    ].std(
                        ddof=1
                    )
                ),

            # Diagnostic only:
            # covariance has more elements, so raw block
            # importance is not perfectly apples-to-apples.
            "reallocated_per_input_element":
                (
                    mean_reallocated
                    / input_elements
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Rank by POLICY reliance
# ============================================================

policy_ranking = (
    summary
    .sort_values(
        "mean_reallocated_weight",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


policy_ranking[
    "policy_rank"
] = np.arange(
    1,
    len(policy_ranking) + 1,
)


policy_ranking.to_csv(
    RESULT_DIR
    / "policy_importance_ranking.csv",
    index=False,
)


# ============================================================
# Performance ranking
#
# Here we rank by magnitude of Sharpe change.
# This means "the policy's performance is sensitive to it",
# not necessarily "removing it makes performance worse".
# ============================================================

performance_ranking = (
    summary
    .assign(
        abs_mean_delta_sharpe=
            summary[
                "mean_delta_sharpe"
            ].abs()
    )
    .sort_values(
        "abs_mean_delta_sharpe",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


performance_ranking[
    "performance_rank"
] = np.arange(
    1,
    len(performance_ranking) + 1,
)


performance_ranking.to_csv(
    RESULT_DIR
    / "performance_importance_ranking.csv",
    index=False,
)


# ============================================================
# Scalar-row ranking only
#
# This excludes covariance because covariance has 25 values,
# while each indicator/RAG row has 5 values.
# This is the fairest direct feature-vs-feature comparison.
# ============================================================

scalar_ranking = (
    summary[
        summary[
            "feature"
        ]
        != "covariance"
    ]
    .sort_values(
        "mean_reallocated_weight",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


scalar_ranking[
    "rank"
] = np.arange(
    1,
    len(scalar_ranking) + 1,
)


scalar_ranking.to_csv(
    RESULT_DIR
    / "scalar_feature_ranking.csv",
    index=False,
)


# ============================================================
# Final output
# ============================================================

print("\n" + "=" * 70)
print("ALL FEATURE IMPORTANCE RESULT")
print("=" * 70)

print(
    "Fixed model:",
    best_model_name,
)

print(
    "Normal multiple:",
    f"{normal_perf['final_multiple']:.6f}",
)

print(
    "Normal Sharpe:",
    f"{normal_perf['sharpe']:.4f}",
)


print("\n" + "=" * 70)
print("POLICY IMPORTANCE RANKING")
print("=" * 70)

print(
    policy_ranking[
        [
            "policy_rank",
            "feature",
            "feature_type",
            "mean_reallocated_weight",
            "std_reallocated_weight",
            "mean_delta_sharpe",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("SCALAR FEATURE RANKING")
print("(fairest comparison: each feature = 5 stock values)")
print("=" * 70)

print(
    scalar_ranking[
        [
            "rank",
            "feature",
            "mean_reallocated_weight",
            "std_reallocated_weight",
            "mean_delta_sharpe",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("PERFORMANCE SENSITIVITY RANKING")
print("=" * 70)

print(
    performance_ranking[
        [
            "performance_rank",
            "feature",
            "mean_delta_multiple",
            "mean_delta_sharpe",
            "std_delta_sharpe",
        ]
    ].to_string(
        index=False
    )
)


print("\nImportant:")
print(
    "covariance is a 5x5 block (25 inputs), "
    "while each other feature is one 5-stock row."
)

print(
    "Use SCALAR FEATURE RANKING when directly comparing "
    "MACD vs RSI vs RAG features."
)


print("\nResults saved to:")

print(
    RESULT_DIR
)

print("\nDONE")