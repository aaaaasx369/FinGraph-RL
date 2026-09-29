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
    "--year",
    type=int,
    default=2023,
)

parser.add_argument(
    "--permutations",
    type=int,
    default=10,
)

args = parser.parse_args()


TARGET_YEAR = args.year
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

BASE_FEATURES = list(
    config.INDICATORS
)

RAG_FEATURES = [
    "impact_score",
    "risk_score",
    "confidence",
]

FEATURE_LIST = (
    BASE_FEATURES
    + RAG_FEATURES
)

SEED = 20260913


# ============================================================
# Paths
# ============================================================

CLEAN_EVENT_PATH = (
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
    / "rag_feature_reliance"
    / str(TARGET_YEAR)
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
# Load original Clean RAG event signals
# ============================================================

print("=" * 70)
print("LOAD CLEAN RAG EVENTS")
print("=" * 70)

events = pd.read_csv(
    CLEAN_EVENT_PATH
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

events["event_date_dt"] = pd.to_datetime(
    events["event_date"]
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
        "Missing numeric RAG values in event table."
    )


print(
    "Clean events:",
    len(events),
)


# ============================================================
# Covariance matrix
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


    return output.sort_values(
        [
            "date",
            "tic",
        ]
    ).reset_index(
        drop=True
    )


# ============================================================
# Market data
# ============================================================

def build_market_data():

    print("\n" + "=" * 70)
    print("REBUILD MARKET PANEL")
    print("=" * 70)


    raw = YahooDownloader(
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        ticker_list=TIC_ORDER,
    ).fetch_data()


    if raw.empty:

        raise RuntimeError(
            "YahooDownloader returned no data."
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


    counts = market.groupby(
        "date"
    )["tic"].nunique()


    good_dates = counts[
        counts
        == len(TIC_ORDER)
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


    print(
        "Complete market dates:",
        market["date"].nunique(),
    )


    return market


# ============================================================
# Strict point-in-time RAG merge
#
# Filing on t cannot be used on trading day t itself.
# ============================================================

def merge_clean_rag(
    market,
    event_table,
    common_start,
    test_end,
):

    event_table = event_table.copy()

    event_table[
        "source_filing_date"
    ] = pd.to_datetime(
        event_table["event_date"]
    )


    parts = []


    for ticker in TIC_ORDER:

        m = market[
            market["tic"]
            == ticker
        ].copy()


        e = event_table[
            event_table["ticker"]
            == ticker
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
        < pd.Timestamp(
            test_end
        )
    ].copy()


    panel = panel.sort_values(
        [
            "date",
            "tic",
        ]
    ).reset_index(
        drop=True
    )


    missing = panel[
        RAG_FEATURES
    ].isna().sum()


    if missing.sum() != 0:

        raise RuntimeError(
            f"Missing RAG values:\n{missing}"
        )


    leakage = panel[
        panel["source_filing_date"]
        >=
        panel["date_dt"]
    ]


    if len(leakage) != 0:

        raise RuntimeError(
            "Same-day/future information leakage found."
        )


    return panel


# ============================================================
# Environment
# ============================================================

def make_env_kwargs():

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
            FEATURE_LIST,
    }


# ============================================================
# Rollout
# ============================================================

def evaluate_model(
    model,
    df,
    quiet=False,
):

    def run():

        env = StockPortfolioEnv(
            df=df,
            **make_env_kwargs(),
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


    if quiet:

        buffer = io.StringIO()

        with contextlib.redirect_stdout(
            buffer
        ):

            return run()


    return run()


# ============================================================
# Financial metrics
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


    if (
        len(r) > 0
        and
        abs(
            r.iloc[0]
        )
        < 1e-15
    ):

        r = r.iloc[
            1:
        ].copy()


    n = len(r)


    if n == 0:

        raise RuntimeError(
            "No daily returns."
        )


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


    daily_std = float(
        r.std(
            ddof=1
        )
    )


    annual_vol = (
        daily_std
        * math.sqrt(252)
    )


    sharpe = (
        float(
            r.mean()
        )
        / daily_std
        * math.sqrt(252)
        if daily_std > 0
        else np.nan
    )


    drawdown = (
        wealth
        / wealth.cummax()
        - 1.0
    )


    return {
        "final_value":
            INITIAL_AMOUNT
            * multiple,

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
                drawdown.min()
            ),

        "n_return_days":
            n,
    }


# ============================================================
# Extract only portfolio weight columns
# ============================================================

def clean_weight_table(
    weights,
):

    missing = [
        ticker
        for ticker in TIC_ORDER
        if ticker not in weights.columns
    ]


    if missing:

        raise RuntimeError(
            f"Missing weight columns: {missing}"
        )


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
# Policy-shift metrics
# ============================================================

def calculate_policy_shift(
    reference_weights,
    candidate_weights,
):

    ref = clean_weight_table(
        reference_weights
    )


    cand = clean_weight_table(
        candidate_weights
    )


    n = min(
        len(ref),
        len(cand),
    )


    ref = ref.iloc[
        :n
    ].copy()

    cand = cand.iloc[
        :n
    ].copy()


    diff = (
        cand
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


    mean_abs = float(
        abs_diff
        .to_numpy()
        .mean()
    )


    max_single = float(
        abs_diff
        .to_numpy()
        .max()
    )


    rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    diff.to_numpy()
                )
            )
        )
    )


    flattened_ref = (
        ref.to_numpy()
        .ravel()
    )

    flattened_cand = (
        cand.to_numpy()
        .ravel()
    )


    if (
        np.std(flattened_ref) > 0
        and
        np.std(flattened_cand) > 0
    ):

        correlation = float(
            np.corrcoef(
                flattened_ref,
                flattened_cand,
            )[0, 1]
        )

    else:

        correlation = np.nan


    per_asset = []


    for ticker in TIC_ORDER:

        per_asset.append(
            {
                "ticker":
                    ticker,

                "mean_abs_weight_diff":
                    float(
                        abs_diff[
                            ticker
                        ].mean()
                    ),

                "max_abs_weight_diff":
                    float(
                        abs_diff[
                            ticker
                        ].max()
                    ),

                "mean_signed_weight_delta":
                    float(
                        diff[
                            ticker
                        ].mean()
                    ),
            }
        )


    metrics = {
        "mean_abs_weight_diff":
            mean_abs,

        "average_daily_l1":
            float(
                daily_l1.mean()
            ),

        # L1/2 = actual probability mass reallocated
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
            max_single,

        "weight_rmse":
            rmse,

        "weight_correlation":
            correlation,
    }


    return (
        metrics,
        pd.DataFrame(
            per_asset
        ),
        diff,
    )


# ============================================================
# Scenario mutation helpers
# ============================================================

def zero_features(
    df,
    cols,
):

    out = df.copy()

    for col in cols:

        out[col] = 0.0


    return out


# ============================================================
# Shuffle features through time WITHIN EACH STOCK.
#
# For joint shuffle:
# impact/risk/confidence move together as one RAG triplet.
#
# This preserves each ticker's feature distribution but destroys
# correct temporal alignment with market conditions.
# ============================================================

def shuffle_within_ticker(
    df,
    cols,
    seed,
):

    out = df.copy()

    rng = np.random.default_rng(
        seed
    )


    col_positions = [
        out.columns.get_loc(
            col
        )
        for col in cols
    ]


    tic_array = (
        out["tic"]
        .to_numpy()
    )


    for ticker in TIC_ORDER:

        positions = np.where(
            tic_array == ticker
        )[0]


        if len(positions) <= 1:
            continue


        permutation = rng.permutation(
            len(positions)
        )


        original = (
            out.iloc[
                positions,
                col_positions
            ]
            .to_numpy(
                copy=True
            )
        )


        shuffled = original[
            permutation
        ]


        out.iloc[
            positions,
            col_positions
        ] = shuffled


    return out


# ============================================================
# Load original train/test configuration
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


print("\n" + "=" * 70)
print("EXPERIMENT RANGE")
print("=" * 70)

print(
    "Common start:",
    COMMON_START,
)

print(
    "Test:",
    TEST_START,
    "to",
    TEST_END,
)


# ============================================================
# Build clean 2023 panel
# ============================================================

market = build_market_data()


clean_panel = merge_clean_rag(
    market,
    events,
    COMMON_START,
    TEST_END,
)


clean_test = data_split(
    clean_panel,
    TEST_START,
    TEST_END,
)


print(
    "Test trading dates:",
    clean_test[
        "date"
    ].nunique(),
)

print(
    "Test rows:",
    len(clean_test),
)


# ============================================================
# Save clean RAG feature statistics
# ============================================================

feature_stats = (
    clean_test.groupby(
        "tic"
    )[
        RAG_FEATURES
    ]
    .agg(
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    )
)


feature_stats.to_csv(
    RESULT_DIR
    / "clean_rag_feature_statistics.csv"
)


# ============================================================
# Load EXACT fixed trained RAG policy
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


if (
    best_model_name
    not in MODEL_CLASSES
):

    raise RuntimeError(
        f"Unknown model: {best_model_name}"
    )


model_path = (
    DRL_ROOT
    / "rag"
    / (
        best_model_name.lower()
        + "_model.zip"
    )
)


print("\n" + "=" * 70)
print("LOAD FIXED CLEAN-RAG POLICY")
print("=" * 70)

print(
    "Model:",
    best_model_name,
)

print(
    "Path:",
    model_path,
)


model = MODEL_CLASSES[
    best_model_name
].load(
    str(
        model_path
    )
)


# ============================================================
# NORMAL reference
# ============================================================

print("\n" + "=" * 70)
print("NORMAL CLEAN-RAG REFERENCE")
print("=" * 70)


normal_returns, normal_weights = (
    evaluate_model(
        model,
        clean_test,
        quiet=False,
    )
)


normal_metrics = calculate_metrics(
    normal_returns
)


normal_returns.to_csv(
    RESULT_DIR
    / "normal_returns.csv",
    index=False,
)


normal_weights.to_csv(
    RESULT_DIR
    / "normal_weights.csv",
)


print(
    "Multiple:",
    f"{normal_metrics['final_multiple']:.6f}",
)

print(
    "Sharpe:",
    f"{normal_metrics['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{normal_metrics['max_drawdown']:.2%}",
)


# ============================================================
# Deterministic ablation scenarios
# ============================================================

SCENARIOS = {
    "zero_all_rag":
        zero_features(
            clean_test,
            RAG_FEATURES,
        ),

    "zero_impact":
        zero_features(
            clean_test,
            [
                "impact_score",
            ],
        ),

    "zero_risk":
        zero_features(
            clean_test,
            [
                "risk_score",
            ],
        ),

    "zero_confidence":
        zero_features(
            clean_test,
            [
                "confidence",
            ],
        ),

    "shuffle_all_joint":
        shuffle_within_ticker(
            clean_test,
            RAG_FEATURES,
            seed=SEED + 1000,
        ),
}


# ============================================================
# Evaluate deterministic scenarios
# ============================================================

scenario_rows = []


scenario_rows.append(
    {
        "scenario":
            "normal",

        **normal_metrics,

        "delta_final_multiple":
            0.0,

        "delta_annual_return":
            0.0,

        "delta_sharpe":
            0.0,

        "delta_max_drawdown":
            0.0,

        "mean_abs_weight_diff":
            0.0,

        "average_daily_l1":
            0.0,

        "average_reallocated_weight":
            0.0,

        "maximum_daily_l1":
            0.0,

        "maximum_reallocated_weight":
            0.0,

        "maximum_single_asset_diff":
            0.0,

        "weight_rmse":
            0.0,

        "weight_correlation":
            1.0,
    }
)


per_asset_tables = []


print("\n" + "=" * 70)
print("DETERMINISTIC RAG ABLATION")
print("=" * 70)


for scenario_name, scenario_df in SCENARIOS.items():

    print(
        "\nRunning:",
        scenario_name,
    )


    scenario_returns, scenario_weights = (
        evaluate_model(
            model,
            scenario_df,
            quiet=True,
        )
    )


    scenario_metrics = calculate_metrics(
        scenario_returns
    )


    (
        shift,
        per_asset,
        weight_diff,
    ) = calculate_policy_shift(
        normal_weights,
        scenario_weights,
    )


    per_asset[
        "scenario"
    ] = scenario_name


    per_asset_tables.append(
        per_asset
    )


    scenario_returns.to_csv(
        RESULT_DIR
        / f"{scenario_name}_returns.csv",
        index=False,
    )


    scenario_weights.to_csv(
        RESULT_DIR
        / f"{scenario_name}_weights.csv",
    )


    weight_diff.to_csv(
        RESULT_DIR
        / f"{scenario_name}_minus_normal_weights.csv",
    )


    row = {
        "scenario":
            scenario_name,

        **scenario_metrics,

        "delta_final_multiple":
            (
                scenario_metrics[
                    "final_multiple"
                ]
                -
                normal_metrics[
                    "final_multiple"
                ]
            ),

        "delta_annual_return":
            (
                scenario_metrics[
                    "annual_return"
                ]
                -
                normal_metrics[
                    "annual_return"
                ]
            ),

        "delta_sharpe":
            (
                scenario_metrics[
                    "sharpe"
                ]
                -
                normal_metrics[
                    "sharpe"
                ]
            ),

        "delta_max_drawdown":
            (
                scenario_metrics[
                    "max_drawdown"
                ]
                -
                normal_metrics[
                    "max_drawdown"
                ]
            ),

        **shift,
    }


    scenario_rows.append(
        row
    )


    print(
        "  Average daily L1:",
        f"{shift['average_daily_l1']:.6f}",
    )

    print(
        "  Average reallocated:",
        f"{shift['average_reallocated_weight']:.4%}",
    )

    print(
        "  Max single asset:",
        f"{shift['maximum_single_asset_diff']:.4%}",
    )

    print(
        "  Δ Multiple:",
        f"{row['delta_final_multiple']:+.6f}",
    )

    print(
        "  Δ Sharpe:",
        f"{row['delta_sharpe']:+.4f}",
    )


# ============================================================
# Save deterministic results
# ============================================================

scenario_results = pd.DataFrame(
    scenario_rows
)


scenario_results.to_csv(
    RESULT_DIR
    / "ablation_summary.csv",
    index=False,
)


if per_asset_tables:

    pd.concat(
        per_asset_tables,
        ignore_index=True,
    ).to_csv(
        RESULT_DIR
        / "ablation_policy_effect_by_asset.csv",
        index=False,
    )


# ============================================================
# Repeated permutation importance
#
# We repeat shuffling because one random shuffle can be lucky.
#
# impact_score      shuffled alone
# risk_score        shuffled alone
# confidence        shuffled alone
# all_rag_joint     all 3 shuffled together
# ============================================================

print("\n" + "=" * 70)
print("REPEATED PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    "Permutations per feature:",
    N_PERMUTATIONS,
)


PERMUTATION_GROUPS = {
    "impact_score":
        [
            "impact_score",
        ],

    "risk_score":
        [
            "risk_score",
        ],

    "confidence":
        [
            "confidence",
        ],

    "all_rag_joint":
        RAG_FEATURES,
}


permutation_rows = []


for group_name, cols in PERMUTATION_GROUPS.items():

    print(
        "\nPermuting:",
        group_name,
    )


    for repetition in range(
        N_PERMUTATIONS
    ):

        permutation_seed = (
            SEED
            + 10_000
            + (
                list(
                    PERMUTATION_GROUPS.keys()
                ).index(
                    group_name
                )
                * 1_000
            )
            + repetition
        )


        shuffled_test = shuffle_within_ticker(
            clean_test,
            cols,
            seed=permutation_seed,
        )


        (
            perm_returns,
            perm_weights,
        ) = evaluate_model(
            model,
            shuffled_test,
            quiet=True,
        )


        perm_metrics = calculate_metrics(
            perm_returns
        )


        (
            perm_shift,
            _,
            _,
        ) = calculate_policy_shift(
            normal_weights,
            perm_weights,
        )


        permutation_rows.append(
            {
                "feature_group":
                    group_name,

                "repetition":
                    repetition + 1,

                "seed":
                    permutation_seed,

                "final_multiple":
                    perm_metrics[
                        "final_multiple"
                    ],

                "delta_final_multiple":
                    (
                        perm_metrics[
                            "final_multiple"
                        ]
                        -
                        normal_metrics[
                            "final_multiple"
                        ]
                    ),

                "sharpe":
                    perm_metrics[
                        "sharpe"
                    ],

                "delta_sharpe":
                    (
                        perm_metrics[
                            "sharpe"
                        ]
                        -
                        normal_metrics[
                            "sharpe"
                        ]
                    ),

                "max_drawdown":
                    perm_metrics[
                        "max_drawdown"
                    ],

                "delta_max_drawdown":
                    (
                        perm_metrics[
                            "max_drawdown"
                        ]
                        -
                        normal_metrics[
                            "max_drawdown"
                        ]
                    ),

                **perm_shift,
            }
        )


        print(
            f"  {repetition + 1:02d}/"
            f"{N_PERMUTATIONS}",
            "L1=",
            f"{perm_shift['average_daily_l1']:.6f}",
        )


# ============================================================
# Save raw permutation results
# ============================================================

permutation_results = pd.DataFrame(
    permutation_rows
)


permutation_results.to_csv(
    RESULT_DIR
    / "permutation_runs.csv",
    index=False,
)


# ============================================================
# Permutation summary
# ============================================================

summary_rows = []


for group_name in PERMUTATION_GROUPS:

    temp = permutation_results[
        permutation_results[
            "feature_group"
        ]
        == group_name
    ]


    summary_rows.append(
        {
            "feature_group":
                group_name,

            "mean_average_daily_l1":
                float(
                    temp[
                        "average_daily_l1"
                    ].mean()
                ),

            "std_average_daily_l1":
                float(
                    temp[
                        "average_daily_l1"
                    ].std(
                        ddof=1
                    )
                )
                if len(temp) > 1
                else 0.0,

            "mean_reallocated_weight":
                float(
                    temp[
                        "average_reallocated_weight"
                    ].mean()
                ),

            "std_reallocated_weight":
                float(
                    temp[
                        "average_reallocated_weight"
                    ].std(
                        ddof=1
                    )
                )
                if len(temp) > 1
                else 0.0,

            "mean_maximum_daily_l1":
                float(
                    temp[
                        "maximum_daily_l1"
                    ].mean()
                ),

            "mean_max_single_asset_diff":
                float(
                    temp[
                        "maximum_single_asset_diff"
                    ].mean()
                ),

            "mean_delta_final_multiple":
                float(
                    temp[
                        "delta_final_multiple"
                    ].mean()
                ),

            "std_delta_final_multiple":
                float(
                    temp[
                        "delta_final_multiple"
                    ].std(
                        ddof=1
                    )
                )
                if len(temp) > 1
                else 0.0,

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
                )
                if len(temp) > 1
                else 0.0,
        }
    )


permutation_summary = pd.DataFrame(
    summary_rows
)


permutation_summary = (
    permutation_summary
    .sort_values(
        "mean_average_daily_l1",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


permutation_summary.to_csv(
    RESULT_DIR
    / "permutation_importance_summary.csv",
    index=False,
)


# ============================================================
# Final display
# ============================================================

print("\n" + "=" * 70)
print("RAG FEATURE RELIANCE RESULT")
print("=" * 70)


print(
    "Fixed model:",
    best_model_name,
)

print(
    "Year:",
    TARGET_YEAR,
)


print("\nNORMAL CLEAN RAG")

print(
    "Multiple:",
    f"{normal_metrics['final_multiple']:.6f}",
)

print(
    "Sharpe:",
    f"{normal_metrics['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{normal_metrics['max_drawdown']:.2%}",
)


# ============================================================
# Main ablation display
# ============================================================

display_cols = [
    "scenario",
    "average_daily_l1",
    "average_reallocated_weight",
    "maximum_daily_l1",
    "maximum_single_asset_diff",
    "delta_final_multiple",
    "delta_sharpe",
]


print("\nDETERMINISTIC ABLATION")

print(
    scenario_results[
        display_cols
    ].to_string(
        index=False,
    )
)


# ============================================================
# Permutation importance display
# ============================================================

print("\nPERMUTATION IMPORTANCE")

print(
    permutation_summary[
        [
            "feature_group",
            "mean_average_daily_l1",
            "std_average_daily_l1",
            "mean_reallocated_weight",
            "mean_delta_final_multiple",
            "mean_delta_sharpe",
        ]
    ].to_string(
        index=False,
    )
)


# ============================================================
# Rank deterministic feature reliance
# ============================================================

ranking = (
    scenario_results[
        scenario_results[
            "scenario"
        ]
        != "normal"
    ][
        [
            "scenario",
            "average_daily_l1",
            "average_reallocated_weight",
        ]
    ]
    .sort_values(
        "average_daily_l1",
        ascending=False,
    )
)


print("\nFEATURE-RELIANCE RANKING")

print(
    ranking.to_string(
        index=False
    )
)


# ============================================================
# Interpret zero-all test numerically
# ============================================================

zero_all_row = (
    scenario_results[
        scenario_results[
            "scenario"
        ]
        == "zero_all_rag"
    ]
    .iloc[0]
)


print("\n" + "=" * 70)
print("CORE QUESTION")
print("=" * 70)


print(
    "If ALL RAG features are forced to zero:"
)


print(
    "Average daily portfolio L1 change:",
    f"{zero_all_row['average_daily_l1']:.6f}",
)


print(
    "Average portfolio capital reallocated:",
    f"{zero_all_row['average_reallocated_weight']:.4%}",
)


print(
    "Maximum one-day L1 change:",
    f"{zero_all_row['maximum_daily_l1']:.6f}",
)


print(
    "Maximum single-asset weight change:",
    f"{zero_all_row['maximum_single_asset_diff']:.4%}",
)


print(
    "Change in final multiple:",
    f"{zero_all_row['delta_final_multiple']:+.6f}",
)


print(
    "Change in Sharpe:",
    f"{zero_all_row['delta_sharpe']:+.4f}",
)


print("\nResults saved to:")

print(
    RESULT_DIR
)

print("\nDONE")