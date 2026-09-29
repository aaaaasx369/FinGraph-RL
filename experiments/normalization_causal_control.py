import argparse
import contextlib
import io
import json
import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import A2C

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


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--seeds",
    nargs="+",
    type=int,
    default=[
        101,
        202,
        303,
        404,
        505,
    ],
)

parser.add_argument(
    "--permutations",
    type=int,
    default=10,
)

parser.add_argument(
    "--timesteps",
    type=int,
    default=50_000,
)

parser.add_argument(
    "--force-retrain-raw",
    action="store_true",
)

parser.add_argument(
    "--force-retrain-normalized",
    action="store_true",
)

args = parser.parse_args()

SEEDS = args.seeds
N_PERMUTATIONS = args.permutations
TOTAL_TIMESTEPS = args.timesteps

FORCE_RETRAIN_RAW = (
    args.force_retrain_raw
)

FORCE_RETRAIN_NORMALIZED = (
    args.force_retrain_normalized
)


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(ROOT),
    )


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

TIC_ORDER = sorted(
    TICKERS
)

TECH_FEATURES = list(
    config.INDICATORS
)

RAG_FEATURES = [
    "impact_score",
    "risk_score",
    "confidence",
]

OBS_FEATURES = (
    TECH_FEATURES
    + RAG_FEATURES
)

DOWNLOAD_START = "2017-01-01"
DOWNLOAD_END = "2024-01-01"

LOOKBACK = 252

INITIAL_AMOUNT = 1_000_000

TARGET_YEAR = 2023


# ============================================================
# PATHS
# ============================================================

EVENT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "sec_rag_events.csv"
)

COMPARE_ROOT = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
)

CONFIG_PATH = (
    COMPARE_ROOT
    / "experiment_config.json"
)

SCALER_PATH = (
    ROOT
    / "results"
    / "normalized_rag_feature_importance"
    / "train_only_scaler.json"
)

ATTACK_PAIR_PATH = (
    ROOT
    / "results"
    / "normalized_broad_persistent_poison"
    / "ALL5_2023_opposite_p3"
    / "replayed_attack_event_pairs.csv"
)

NORMALIZED_MULTI_ROOT = (
    ROOT
    / "results"
    / "normalized_multiseed_security"
)

NORMALIZED_SINGLE_MODEL = (
    ROOT
    / "results"
    / "normalized_rag_feature_importance"
    / "normalized_a2c_model.zip"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "normalization_causal_control"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# REQUIRED FILE CHECK
# ============================================================

for required in [
    EVENT_PATH,
    CONFIG_PATH,
    SCALER_PATH,
    ATTACK_PAIR_PATH,
]:

    if not required.exists():

        raise FileNotFoundError(
            f"Missing required file:\n"
            f"{required}"
        )


# ============================================================
# DATES
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    exp_cfg = json.load(
        f
    )


COMMON_START = (
    exp_cfg[
        "common_start"
    ]
)

TRAIN_END = (
    exp_cfg[
        "train_end_exclusive"
    ]
)

VALIDATION_END = (
    exp_cfg[
        "validation_end_exclusive"
    ]
)

TEST_END = (
    exp_cfg[
        "test_end_exclusive"
    ]
)


print("=" * 80)
print("NORMALIZATION CAUSAL CONTROL EXPERIMENT")
print("=" * 80)

print(
    "Seeds:",
    SEEDS,
)

print(
    "Timesteps:",
    TOTAL_TIMESTEPS,
)

print(
    "RAG permutations:",
    N_PERMUTATIONS,
)

print(
    "Train:",
    COMMON_START,
    "->",
    TRAIN_END,
)

print(
    "Validation:",
    TRAIN_END,
    "->",
    VALIDATION_END,
)

print(
    "Test:",
    VALIDATION_END,
    "->",
    TEST_END,
)


# ============================================================
# SEED
# ============================================================

def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# LOAD CLEAN EVENTS
# ============================================================

events = pd.read_csv(
    EVENT_PATH
)

events = events[
    events["status"]
    == "ok"
].copy()


events[
    "ticker"
] = (
    events[
        "ticker"
    ]
    .astype(str)
    .str.upper()
)


events[
    "event_date"
] = pd.to_datetime(
    events[
        "event_date"
    ]
)


for feature in RAG_FEATURES:

    events[
        feature
    ] = pd.to_numeric(
        events[
            feature
        ],
        errors="coerce",
    )


if events[
    RAG_FEATURES
].isna().any().any():

    raise RuntimeError(
        "Missing RAG values."
    )


# ============================================================
# LOAD EXACT SAME 62-EVENT ATTACK
# ============================================================

pairs = pd.read_csv(
    ATTACK_PAIR_PATH
)

pairs[
    "ticker"
] = (
    pairs[
        "ticker"
    ]
    .astype(str)
    .str.upper()
)

pairs[
    "event_date"
] = pd.to_datetime(
    pairs[
        "event_date"
    ]
)


PAIR_COLUMNS = [
    "clean_impact_score",
    "clean_risk_score",
    "clean_confidence",
    "attack_impact_score",
    "attack_risk_score",
    "attack_confidence",
]


for col in PAIR_COLUMNS:

    pairs[
        col
    ] = pd.to_numeric(
        pairs[
            col
        ],
        errors="coerce",
    )


if len(pairs) != 62:

    raise RuntimeError(
        f"Expected 62 attack events, "
        f"found {len(pairs)}."
    )


if pairs[
    PAIR_COLUMNS
].isna().any().any():

    raise RuntimeError(
        "Attack pair table contains NaN."
    )


pairs[
    "impact_delta"
] = (
    pairs[
        "attack_impact_score"
    ]
    -
    pairs[
        "clean_impact_score"
    ]
)


pairs[
    "impact_sign_changed"
] = (
    np.sign(
        pairs[
            "attack_impact_score"
        ]
    )
    !=
    np.sign(
        pairs[
            "clean_impact_score"
        ]
    )
)


print("\n" + "=" * 80)
print("ATTACK QA")
print("=" * 80)

print(
    "Target events:",
    len(pairs),
)

print(
    "Impact sign changes:",
    int(
        pairs[
            "impact_sign_changed"
        ].sum()
    ),
    "/",
    len(pairs),
)

print(
    "Mean impact delta:",
    f"{pairs['impact_delta'].mean():+.4f}",
)

print(
    "Mean |impact delta|:",
    f"{pairs['impact_delta'].abs().mean():.4f}",
)


# ============================================================
# BUILD PAIRED CLEAN + ATTACK EVENT TABLES
# ============================================================

paired_events = (
    events.copy()
)

attack_events = (
    events.copy()
)


for _, row in pairs.iterrows():

    ticker = (
        row[
            "ticker"
        ]
    )

    event_date = (
        row[
            "event_date"
        ]
    )


    mask = (
        (
            events[
                "ticker"
            ]
            == ticker
        )
        &
        (
            events[
                "event_date"
            ]
            == event_date
        )
    )


    if int(
        mask.sum()
    ) != 1:

        raise RuntimeError(
            f"Expected one event for "
            f"{ticker} "
            f"{event_date.date()}"
        )


    # --------------------------------------------------------
    # Paired clean
    # --------------------------------------------------------

    paired_events.loc[
        mask,
        "impact_score",
    ] = row[
        "clean_impact_score"
    ]

    paired_events.loc[
        mask,
        "risk_score",
    ] = row[
        "clean_risk_score"
    ]

    paired_events.loc[
        mask,
        "confidence",
    ] = row[
        "clean_confidence"
    ]


    # --------------------------------------------------------
    # Attack
    # --------------------------------------------------------

    attack_events.loc[
        mask,
        "impact_score",
    ] = row[
        "attack_impact_score"
    ]

    attack_events.loc[
        mask,
        "risk_score",
    ] = row[
        "attack_risk_score"
    ]

    attack_events.loc[
        mask,
        "confidence",
    ] = row[
        "attack_confidence"
    ]


# ============================================================
# COVARIANCE
# ============================================================

def add_covariance_matrix(
    df,
    lookback=252,
):

    df = (
        df.sort_values(
            [
                "date",
                "tic",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )


    prices = df.pivot(
        index="date",
        columns="tic",
        values="close",
    )


    prices = prices.reindex(
        columns=TIC_ORDER
    )


    dates = list(
        prices.index
    )


    rows = []


    for i in range(
        lookback,
        len(dates),
    ):

        window = prices.iloc[
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


        covariance = (
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
                    covariance,
            }
        )


    covariance_df = pd.DataFrame(
        rows
    )


    output = df.merge(
        covariance_df,
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
# MARKET
# ============================================================

def build_market():

    print("\n" + "=" * 80)
    print("BUILD MARKET DATA")
    print("=" * 80)


    raw = YahooDownloader(
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        ticker_list=TIC_ORDER,
    ).fetch_data()


    if raw.empty:

        raise RuntimeError(
            "YahooDownloader returned nothing."
        )


    feature_engineer = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=TECH_FEATURES,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )


    market = (
        feature_engineer
        .preprocess_data(
            raw
        )
    )


    market = market[
        market[
            "tic"
        ].isin(
            TIC_ORDER
        )
    ].copy()


    counts = market.groupby(
        "date"
    )[
        "tic"
    ].nunique()


    complete_dates = counts[
        counts
        == len(
            TIC_ORDER
        )
    ].index


    market = market[
        market[
            "date"
        ].isin(
            complete_dates
        )
    ].copy()


    market = add_covariance_matrix(
        market,
        LOOKBACK,
    )


    market[
        "date_dt"
    ] = pd.to_datetime(
        market[
            "date"
        ]
    )


    print(
        "Complete dates:",
        market[
            "date"
        ].nunique(),
    )


    return market


# ============================================================
# STRICT POINT-IN-TIME RAG MERGE
# ============================================================

def merge_rag(
    market,
    event_table,
):

    event_table = (
        event_table.copy()
    )


    event_table[
        "source_filing_date"
    ] = pd.to_datetime(
        event_table[
            "event_date"
        ]
    )


    pieces = []


    for ticker in TIC_ORDER:

        market_ticker = market[
            market[
                "tic"
            ]
            == ticker
        ].copy()


        event_ticker = event_table[
            event_table[
                "ticker"
            ]
            == ticker
        ][
            [
                "source_filing_date",
                "impact_score",
                "risk_score",
                "confidence",
            ]
        ].copy()


        market_ticker = (
            market_ticker
            .sort_values(
                "date_dt"
            )
        )


        event_ticker = (
            event_ticker
            .sort_values(
                "source_filing_date"
            )
        )


        merged = pd.merge_asof(
            market_ticker,
            event_ticker,
            left_on="date_dt",
            right_on="source_filing_date",
            direction="backward",
            allow_exact_matches=False,
        )


        pieces.append(
            merged
        )


    panel = pd.concat(
        pieces,
        ignore_index=True,
    )


    panel = panel[
        (
            panel[
                "date_dt"
            ]
            >=
            pd.Timestamp(
                COMMON_START
            )
        )
        &
        (
            panel[
                "date_dt"
            ]
            <
            pd.Timestamp(
                TEST_END
            )
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


    if panel[
        RAG_FEATURES
    ].isna().any().any():

        raise RuntimeError(
            "Missing RAG feature after merge."
        )


    leakage = panel[
        panel[
            "source_filing_date"
        ]
        >=
        panel[
            "date_dt"
        ]
    ]


    if len(
        leakage
    ) != 0:

        raise RuntimeError(
            "Same-day/future information leakage detected."
        )


    return panel


# ============================================================
# LOAD EXACT TRAIN-ONLY NORMALIZATION
# ============================================================

with open(
    SCALER_PATH,
    "r",
    encoding="utf-8",
) as f:

    scaler = json.load(
        f
    )


scalar_scaler = (
    scaler[
        "scalar_scaler"
    ]
)


covariance_mean = np.asarray(
    scaler[
        "covariance_mean"
    ],
    dtype=float,
)


covariance_std = np.asarray(
    scaler[
        "covariance_std"
    ],
    dtype=float,
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_panel(
    df,
):

    output = df.copy()


    # --------------------------------------------------------
    # Scalar observation features
    # --------------------------------------------------------

    for feature in OBS_FEATURES:

        for ticker in TIC_ORDER:

            mask = (
                output[
                    "tic"
                ]
                == ticker
            )


            mean = float(
                scalar_scaler[
                    feature
                ][
                    ticker
                ][
                    "mean"
                ]
            )


            std = float(
                scalar_scaler[
                    feature
                ][
                    ticker
                ][
                    "std"
                ]
            )


            output.loc[
                mask,
                feature,
            ] = (
                (
                    output.loc[
                        mask,
                        feature,
                    ]
                    .astype(float)
                    -
                    mean
                )
                /
                std
            )


    # --------------------------------------------------------
    # Covariance block
    # --------------------------------------------------------

    output[
        "cov_list"
    ] = (
        output[
            "cov_list"
        ]
        .apply(
            lambda matrix:
                (
                    np.asarray(
                        matrix,
                        dtype=float,
                    )
                    -
                    covariance_mean
                )
                /
                covariance_std
        )
    )


    return output


# ============================================================
# ENV
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
            len(
                TIC_ORDER
            ),

        "stock_dim":
            len(
                TIC_ORDER
            ),

        "action_space":
            len(
                TIC_ORDER
            ),

        "tech_indicator_list":
            OBS_FEATURES,
    }


# ============================================================
# ROLLOUT
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


        observation, _ = (
            env.reset()
        )


        while True:

            action, _ = model.predict(
                observation,
                deterministic=True,
            )


            (
                observation,
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
# PERFORMANCE METRICS
# ============================================================

def performance_metrics(
    returns_df,
):

    returns = (
        returns_df[
            "daily_return"
        ]
        .astype(float)
        .copy()
    )


    if (
        len(returns) > 0
        and
        abs(
            returns.iloc[0]
        )
        < 1e-15
    ):

        returns = returns.iloc[
            1:
        ].copy()


    n = len(
        returns
    )


    if n == 0:

        raise RuntimeError(
            "No return observations."
        )


    wealth = (
        1.0
        +
        returns
    ).cumprod()


    multiple = float(
        wealth.iloc[
            -1
        ]
    )


    annual_return = (
        multiple
        **
        (
            252.0
            /
            n
        )
        -
        1.0
    )


    daily_std = float(
        returns.std(
            ddof=1
        )
    )


    annual_volatility = (
        daily_std
        *
        math.sqrt(
            252
        )
    )


    if daily_std > 0:

        sharpe = (
            float(
                returns.mean()
            )
            /
            daily_std
            *
            math.sqrt(
                252
            )
        )

    else:

        sharpe = np.nan


    drawdown = (
        wealth
        /
        wealth.cummax()
        -
        1.0
    )


    return {
        "final_value":
            INITIAL_AMOUNT
            *
            multiple,

        "final_multiple":
            multiple,

        "annual_return":
            annual_return,

        "annual_volatility":
            annual_volatility,

        "sharpe":
            sharpe,

        "max_drawdown":
            float(
                drawdown.min()
            ),
    }


# ============================================================
# PORTFOLIO DISTANCE
# ============================================================

def weight_table(
    weights,
):

    missing = [
        ticker
        for ticker in TIC_ORDER
        if ticker
        not in weights.columns
    ]


    if missing:

        raise RuntimeError(
            f"Missing weight columns: "
            f"{missing}"
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


def policy_distance(
    reference,
    candidate,
):

    ref = weight_table(
        reference
    )

    cand = weight_table(
        candidate
    )


    n = min(
        len(ref),
        len(cand),
    )


    ref = ref.iloc[
        :n
    ]

    cand = cand.iloc[
        :n
    ]


    diff = (
        cand
        -
        ref
    )


    absolute = (
        diff.abs()
    )


    daily_l1 = (
        absolute.sum(
            axis=1
        )
    )


    return {
        "mean_abs_weight_diff":
            float(
                absolute
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
                /
                2.0
            ),

        "maximum_daily_l1":
            float(
                daily_l1.max()
            ),

        "maximum_reallocated_weight":
            float(
                daily_l1.max()
                /
                2.0
            ),

        "maximum_single_asset_diff":
            float(
                absolute
                .to_numpy()
                .max()
            ),
    }


# ============================================================
# JOINT RAG PERMUTATION
#
# impact/risk/confidence move together.
#
# IMPORTANT:
# The permutation indices are determined only by seed.
#
# Therefore raw and normalized conditions receive the exact
# same temporal permutation.
# ============================================================

def shuffle_rag_joint(
    df,
    permutation_seed,
):

    output = df.copy()


    rng = np.random.default_rng(
        permutation_seed
    )


    column_positions = [
        output.columns.get_loc(
            feature
        )
        for feature in RAG_FEATURES
    ]


    ticker_array = (
        output[
            "tic"
        ]
        .to_numpy()
    )


    for ticker in TIC_ORDER:

        positions = np.where(
            ticker_array
            == ticker
        )[0]


        original = (
            output.iloc[
                positions,
                column_positions
            ]
            .to_numpy(
                copy=True
            )
        )


        permutation = (
            rng.permutation(
                len(
                    positions
                )
            )
        )


        output.iloc[
            positions,
            column_positions
        ] = (
            original[
                permutation
            ]
        )


    return output


# ============================================================
# BUILD RAW PANELS ONCE
# ============================================================

market = build_market()


print("\n" + "=" * 80)
print("BUILD ORIGINAL / PAIRED / ATTACK PANELS")
print("=" * 80)


original_raw_panel = merge_rag(
    market,
    events,
)


paired_raw_panel = merge_rag(
    market,
    paired_events,
)


attack_raw_panel = merge_rag(
    market,
    attack_events,
)


# ============================================================
# NORMALIZED VERSIONS
# ============================================================

original_normalized_panel = (
    normalize_panel(
        original_raw_panel
    )
)


paired_normalized_panel = (
    normalize_panel(
        paired_raw_panel
    )
)


attack_normalized_panel = (
    normalize_panel(
        attack_raw_panel
    )
)


# ============================================================
# SPLITS — RAW
# ============================================================

raw_train = data_split(
    original_raw_panel,
    COMMON_START,
    TRAIN_END,
)


raw_validation = data_split(
    original_raw_panel,
    TRAIN_END,
    VALIDATION_END,
)


raw_original_test = data_split(
    original_raw_panel,
    VALIDATION_END,
    TEST_END,
)


raw_paired_test = data_split(
    paired_raw_panel,
    VALIDATION_END,
    TEST_END,
)


raw_attack_test = data_split(
    attack_raw_panel,
    VALIDATION_END,
    TEST_END,
)


# ============================================================
# SPLITS — NORMALIZED
# ============================================================

normalized_train = data_split(
    original_normalized_panel,
    COMMON_START,
    TRAIN_END,
)


normalized_validation = data_split(
    original_normalized_panel,
    TRAIN_END,
    VALIDATION_END,
)


normalized_original_test = data_split(
    original_normalized_panel,
    VALIDATION_END,
    TEST_END,
)


normalized_paired_test = data_split(
    paired_normalized_panel,
    VALIDATION_END,
    TEST_END,
)


normalized_attack_test = data_split(
    attack_normalized_panel,
    VALIDATION_END,
    TEST_END,
)


print(
    "Train dates:",
    raw_train[
        "date"
    ].nunique(),
)

print(
    "Validation dates:",
    raw_validation[
        "date"
    ].nunique(),
)

print(
    "Test dates:",
    raw_original_test[
        "date"
    ].nunique(),
)


# ============================================================
# PANEL IDENTITY CHECK
# ============================================================

if not np.array_equal(
    raw_original_test[
        [
            "date",
            "tic",
        ]
    ].to_numpy(),
    normalized_original_test[
        [
            "date",
            "tic",
        ]
    ].to_numpy(),
):

    raise RuntimeError(
        "Raw and normalized test row identities differ."
    )


print(
    "Raw vs normalized row alignment: PASS"
)


# ============================================================
# ATTACK ROW CHECK
# ============================================================

raw_paired_rag = (
    raw_paired_test[
        RAG_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


raw_attack_rag = (
    raw_attack_test[
        RAG_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


raw_changed_rows = int(
    np.any(
        np.abs(
            raw_paired_rag
            -
            raw_attack_rag
        )
        >
        1e-12,
        axis=1,
    ).sum()
)


normalized_paired_rag = (
    normalized_paired_test[
        RAG_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


normalized_attack_rag = (
    normalized_attack_test[
        RAG_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


normalized_changed_rows = int(
    np.any(
        np.abs(
            normalized_paired_rag
            -
            normalized_attack_rag
        )
        >
        1e-12,
        axis=1,
    ).sum()
)


print(
    "Raw attack-changed rows:",
    raw_changed_rows,
)

print(
    "Normalized attack-changed rows:",
    normalized_changed_rows,
)


if (
    raw_changed_rows
    !=
    normalized_changed_rows
):

    raise RuntimeError(
        "Attack row counts differ between conditions."
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_a2c(
    train_df,
    seed,
    save_path,
    label,
):

    print(
        f"Training {label} "
        f"A2C seed={seed}"
    )


    set_seed(
        seed
    )


    env = StockPortfolioEnv(
        df=train_df,
        **env_kwargs(),
    )


    sb_env, _ = (
        env.get_sb_env()
    )


    agent = DRLAgent(
        env=sb_env
    )


    # EXACT SAME HYPERPARAMETERS FOR BOTH CONDITIONS.
    model_kwargs = {
        "n_steps":
            5,

        "ent_coef":
            0.005,

        "learning_rate":
            0.0002,

        "device":
            "cpu",
    }


    model = agent.get_model(
        model_name="a2c",
        model_kwargs=model_kwargs,
        seed=seed,
    )


    model = agent.train_model(
        model=model,
        tb_log_name=(
            f"{label}_seed_{seed}"
        ),
        total_timesteps=TOTAL_TIMESTEPS,
    )


    model.save(
        str(
            save_path
        )
    )


    return model


# ============================================================
# LOAD / TRAIN NORMALIZED
# ============================================================

def get_normalized_model(
    seed,
    seed_dir,
):

    destination_base = (
        seed_dir
        / "normalized_a2c_model"
    )

    destination_zip = Path(
        str(
            destination_base
        )
        +
        ".zip"
    )


    if (
        destination_zip.exists()
        and
        not FORCE_RETRAIN_NORMALIZED
    ):

        return (
            A2C.load(
                str(
                    destination_zip
                )
            ),
            "causal_cache",
        )


    # --------------------------------------------------------
    # Reuse exact previous 5-seed normalized model
    # --------------------------------------------------------

    source = (
        NORMALIZED_MULTI_ROOT
        / f"seed_{seed}"
        / "normalized_a2c_model.zip"
    )


    if (
        source.exists()
        and
        not FORCE_RETRAIN_NORMALIZED
    ):

        shutil.copy2(
            source,
            destination_zip,
        )


        return (
            A2C.load(
                str(
                    destination_zip
                )
            ),
            "previous_multiseed_exact",
        )


    # --------------------------------------------------------
    # Seed 101 fallback
    # --------------------------------------------------------

    if (
        seed == 101
        and
        NORMALIZED_SINGLE_MODEL.exists()
        and
        not FORCE_RETRAIN_NORMALIZED
    ):

        shutil.copy2(
            NORMALIZED_SINGLE_MODEL,
            destination_zip,
        )


        return (
            A2C.load(
                str(
                    destination_zip
                )
            ),
            "previous_seed101_exact",
        )


    model = train_a2c(
        normalized_train,
        seed,
        destination_base,
        "normalized",
    )


    return (
        model,
        "new_training",
    )


# ============================================================
# LOAD / TRAIN RAW
# ============================================================

def get_raw_model(
    seed,
    seed_dir,
):

    model_base = (
        seed_dir
        / "raw_a2c_model"
    )


    model_zip = Path(
        str(
            model_base
        )
        +
        ".zip"
    )


    if (
        model_zip.exists()
        and
        not FORCE_RETRAIN_RAW
    ):

        return (
            A2C.load(
                str(
                    model_zip
                )
            ),
            "causal_cache",
        )


    model = train_a2c(
        raw_train,
        seed,
        model_base,
        "raw",
    )


    return (
        model,
        "new_training",
    )


# ============================================================
# JOINT RAG RELIANCE
# ============================================================

def measure_joint_rag_reliance(
    model,
    test_df,
    clean_weights,
    training_seed,
    condition,
):

    rows = []


    for repetition in range(
        N_PERMUTATIONS
    ):

        # Same exact permutation seeds for raw & normalized
        permutation_seed = (
            training_seed
            *
            100_000
            +
            repetition
        )


        modified_test = (
            shuffle_rag_joint(
                test_df,
                permutation_seed,
            )
        )


        (
            modified_returns,
            modified_weights,
        ) = evaluate(
            model,
            modified_test,
            quiet=True,
        )


        shift = policy_distance(
            clean_weights,
            modified_weights,
        )


        perf = performance_metrics(
            modified_returns
        )


        rows.append(
            {
                "training_seed":
                    training_seed,

                "condition":
                    condition,

                "repetition":
                    repetition + 1,

                "permutation_seed":
                    permutation_seed,

                **shift,

                "final_multiple":
                    perf[
                        "final_multiple"
                    ],

                "sharpe":
                    perf[
                        "sharpe"
                    ],
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# ONE CONDITION
# ============================================================

def evaluate_condition(
    model,
    condition,
    original_test,
    paired_test,
    attack_test,
    validation_df,
    seed,
):

    print(
        f"\n[{condition}] validation"
    )


    (
        validation_returns,
        validation_weights,
    ) = evaluate(
        model,
        validation_df,
        quiet=True,
    )


    validation_perf = (
        performance_metrics(
            validation_returns
        )
    )


    print(
        f"[{condition}] original clean"
    )


    (
        original_returns,
        original_weights,
    ) = evaluate(
        model,
        original_test,
        quiet=True,
    )


    original_perf = (
        performance_metrics(
            original_returns
        )
    )


    print(
        f"[{condition}] paired clean"
    )


    (
        paired_returns,
        paired_weights,
    ) = evaluate(
        model,
        paired_test,
        quiet=True,
    )


    paired_perf = (
        performance_metrics(
            paired_returns
        )
    )


    print(
        f"[{condition}] attack"
    )


    (
        attack_returns,
        attack_weights,
    ) = evaluate(
        model,
        attack_test,
        quiet=True,
    )


    attack_perf = (
        performance_metrics(
            attack_returns
        )
    )


    regen_shift = policy_distance(
        original_weights,
        paired_weights,
    )


    attack_shift = policy_distance(
        paired_weights,
        attack_weights,
    )


    print(
        f"[{condition}] joint RAG permutation"
    )


    permutation_df = (
        measure_joint_rag_reliance(
            model=model,
            test_df=original_test,
            clean_weights=original_weights,
            training_seed=seed,
            condition=condition,
        )
    )


    rag_reliance = float(
        permutation_df[
            "average_reallocated_weight"
        ].mean()
    )


    rag_reliance_std = float(
        permutation_df[
            "average_reallocated_weight"
        ].std(
            ddof=1
        )
    )


    result = {
        "seed":
            seed,

        "condition":
            condition,

        "validation_multiple":
            validation_perf[
                "final_multiple"
            ],

        "validation_sharpe":
            validation_perf[
                "sharpe"
            ],

        "clean_multiple":
            original_perf[
                "final_multiple"
            ],

        "clean_sharpe":
            original_perf[
                "sharpe"
            ],

        "clean_mdd":
            original_perf[
                "max_drawdown"
            ],

        "paired_clean_multiple":
            paired_perf[
                "final_multiple"
            ],

        "paired_clean_sharpe":
            paired_perf[
                "sharpe"
            ],

        # ----------------------------------------------------
        # Clean RAG reliance
        # ----------------------------------------------------

        "rag_joint_reallocated":
            rag_reliance,

        "rag_joint_reallocated_std":
            rag_reliance_std,

        # ----------------------------------------------------
        # Regeneration drift
        # ----------------------------------------------------

        "regen_daily_l1":
            regen_shift[
                "average_daily_l1"
            ],

        "regen_reallocated":
            regen_shift[
                "average_reallocated_weight"
            ],

        # ----------------------------------------------------
        # Attack action propagation
        # ----------------------------------------------------

        "attack_daily_l1":
            attack_shift[
                "average_daily_l1"
            ],

        "attack_reallocated":
            attack_shift[
                "average_reallocated_weight"
            ],

        "attack_max_daily_l1":
            attack_shift[
                "maximum_daily_l1"
            ],

        "attack_max_reallocated":
            attack_shift[
                "maximum_reallocated_weight"
            ],

        "attack_max_single_asset":
            attack_shift[
                "maximum_single_asset_diff"
            ],

        # ----------------------------------------------------
        # Attack financial consequences
        # ----------------------------------------------------

        "attack_delta_multiple":
            (
                attack_perf[
                    "final_multiple"
                ]
                -
                paired_perf[
                    "final_multiple"
                ]
            ),

        "attack_delta_dollars":
            (
                (
                    attack_perf[
                        "final_multiple"
                    ]
                    -
                    paired_perf[
                        "final_multiple"
                    ]
                )
                *
                INITIAL_AMOUNT
            ),

        "attack_delta_annual_return":
            (
                attack_perf[
                    "annual_return"
                ]
                -
                paired_perf[
                    "annual_return"
                ]
            ),

        "attack_delta_volatility":
            (
                attack_perf[
                    "annual_volatility"
                ]
                -
                paired_perf[
                    "annual_volatility"
                ]
            ),

        "attack_delta_sharpe":
            (
                attack_perf[
                    "sharpe"
                ]
                -
                paired_perf[
                    "sharpe"
                ]
            ),

        "attack_delta_mdd":
            (
                attack_perf[
                    "max_drawdown"
                ]
                -
                paired_perf[
                    "max_drawdown"
                ]
            ),
    }


    if (
        result[
            "regen_reallocated"
        ]
        > 0
    ):

        result[
            "attack_to_regen_ratio"
        ] = (
            result[
                "attack_reallocated"
            ]
            /
            result[
                "regen_reallocated"
            ]
        )

    else:

        result[
            "attack_to_regen_ratio"
        ] = np.nan


    return {
        "summary":
            result,

        "permutations":
            permutation_df,

        "original_returns":
            original_returns,

        "original_weights":
            original_weights,

        "paired_returns":
            paired_returns,

        "paired_weights":
            paired_weights,

        "attack_returns":
            attack_returns,

        "attack_weights":
            attack_weights,
    }


# ============================================================
# RUN ALL SEEDS
# ============================================================

condition_rows = []

causal_rows = []

all_permutation_tables = []


for index, seed in enumerate(
    SEEDS,
    start=1,
):

    print("\n")
    print("#" * 80)

    print(
        f"SEED {seed} "
        f"({index}/{len(SEEDS)})"
    )

    print("#" * 80)


    seed_dir = (
        RESULT_ROOT
        /
        f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # GET NORMALIZED MODEL
    # ========================================================

    (
        normalized_model,
        normalized_source,
    ) = get_normalized_model(
        seed,
        seed_dir,
    )


    print(
        "Normalized model source:",
        normalized_source,
    )


    # ========================================================
    # GET RAW MODEL
    # ========================================================

    (
        raw_model,
        raw_source,
    ) = get_raw_model(
        seed,
        seed_dir,
    )


    print(
        "Raw model source:",
        raw_source,
    )


    # ========================================================
    # RAW CONDITION
    # ========================================================

    raw_result = evaluate_condition(
        model=raw_model,
        condition="raw",
        original_test=raw_original_test,
        paired_test=raw_paired_test,
        attack_test=raw_attack_test,
        validation_df=raw_validation,
        seed=seed,
    )


    raw_summary = (
        raw_result[
            "summary"
        ]
    )


    raw_summary[
        "model_source"
    ] = raw_source


    # ========================================================
    # NORMALIZED CONDITION
    # ========================================================

    normalized_result = (
        evaluate_condition(
            model=normalized_model,
            condition="normalized",
            original_test=normalized_original_test,
            paired_test=normalized_paired_test,
            attack_test=normalized_attack_test,
            validation_df=normalized_validation,
            seed=seed,
        )
    )


    normalized_summary = (
        normalized_result[
            "summary"
        ]
    )


    normalized_summary[
        "model_source"
    ] = normalized_source


    condition_rows.append(
        raw_summary
    )

    condition_rows.append(
        normalized_summary
    )


    all_permutation_tables.append(
        raw_result[
            "permutations"
        ]
    )

    all_permutation_tables.append(
        normalized_result[
            "permutations"
        ]
    )


    # ========================================================
    # WITHIN-SEED CAUSAL COMPARISON
    # ========================================================

    raw_reliance = (
        raw_summary[
            "rag_joint_reallocated"
        ]
    )

    normalized_reliance = (
        normalized_summary[
            "rag_joint_reallocated"
        ]
    )


    raw_attack = (
        raw_summary[
            "attack_reallocated"
        ]
    )

    normalized_attack = (
        normalized_summary[
            "attack_reallocated"
        ]
    )


    causal_row = {
        "seed":
            seed,

        # ----------------------------------------------------
        # Clean RAG reliance
        # ----------------------------------------------------

        "raw_rag_reliance":
            raw_reliance,

        "normalized_rag_reliance":
            normalized_reliance,

        "rag_reliance_absolute_increase":
            (
                normalized_reliance
                -
                raw_reliance
            ),

        "rag_reliance_ratio":
            (
                normalized_reliance
                /
                raw_reliance
                if raw_reliance > 0
                else np.nan
            ),

        # ----------------------------------------------------
        # Attack propagation
        # ----------------------------------------------------

        "raw_attack_reallocated":
            raw_attack,

        "normalized_attack_reallocated":
            normalized_attack,

        "attack_absolute_increase":
            (
                normalized_attack
                -
                raw_attack
            ),

        "attack_propagation_ratio":
            (
                normalized_attack
                /
                raw_attack
                if raw_attack > 0
                else np.nan
            ),

        # ----------------------------------------------------
        # Regeneration drift
        # ----------------------------------------------------

        "raw_regen_reallocated":
            raw_summary[
                "regen_reallocated"
            ],

        "normalized_regen_reallocated":
            normalized_summary[
                "regen_reallocated"
            ],

        # ----------------------------------------------------
        # Financial effect
        # ----------------------------------------------------

        "raw_attack_delta_multiple":
            raw_summary[
                "attack_delta_multiple"
            ],

        "normalized_attack_delta_multiple":
            normalized_summary[
                "attack_delta_multiple"
            ],

        "raw_attack_delta_sharpe":
            raw_summary[
                "attack_delta_sharpe"
            ],

        "normalized_attack_delta_sharpe":
            normalized_summary[
                "attack_delta_sharpe"
            ],

        "raw_clean_multiple":
            raw_summary[
                "clean_multiple"
            ],

        "normalized_clean_multiple":
            normalized_summary[
                "clean_multiple"
            ],

        "raw_clean_sharpe":
            raw_summary[
                "clean_sharpe"
            ],

        "normalized_clean_sharpe":
            normalized_summary[
                "clean_sharpe"
            ],
    }


    causal_rows.append(
        causal_row
    )


    # ========================================================
    # SAVE SEED DATA
    # ========================================================

    pd.DataFrame(
        [
            raw_summary,
            normalized_summary,
        ]
    ).to_csv(
        seed_dir
        /
        "condition_summary.csv",
        index=False,
    )


    pd.DataFrame(
        [
            causal_row
        ]
    ).to_csv(
        seed_dir
        /
        "causal_comparison.csv",
        index=False,
    )


    raw_result[
        "permutations"
    ].to_csv(
        seed_dir
        /
        "raw_rag_permutations.csv",
        index=False,
    )


    normalized_result[
        "permutations"
    ].to_csv(
        seed_dir
        /
        "normalized_rag_permutations.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Save portfolio data
    # --------------------------------------------------------

    for condition_name, result in [
        (
            "raw",
            raw_result,
        ),
        (
            "normalized",
            normalized_result,
        ),
    ]:

        result[
            "original_returns"
        ].to_csv(
            seed_dir
            /
            f"{condition_name}_original_returns.csv",
            index=False,
        )


        result[
            "original_weights"
        ].to_csv(
            seed_dir
            /
            f"{condition_name}_original_weights.csv",
        )


        result[
            "paired_returns"
        ].to_csv(
            seed_dir
            /
            f"{condition_name}_paired_returns.csv",
            index=False,
        )


        result[
            "paired_weights"
        ].to_csv(
            seed_dir
            /
            f"{condition_name}_paired_weights.csv",
        )


        result[
            "attack_returns"
        ].to_csv(
            seed_dir
            /
            f"{condition_name}_attack_returns.csv",
            index=False,
        )


        result[
            "attack_weights"
        ].to_csv(
            seed_dir
            /
            f"{condition_name}_attack_weights.csv",
        )


    # ========================================================
    # SEED REPORT
    # ========================================================

    print("\nSeed causal result:")

    print(
        "  RAW RAG reliance:",
        f"{raw_reliance:.4%}",
    )

    print(
        "  NORMALIZED RAG reliance:",
        f"{normalized_reliance:.4%}",
    )

    print(
        "  RAG reliance ratio:",
        f"{causal_row['rag_reliance_ratio']:.2f}x",
    )


    print(
        "  RAW attack reallocated:",
        f"{raw_attack:.4%}",
    )

    print(
        "  NORMALIZED attack reallocated:",
        f"{normalized_attack:.4%}",
    )

    print(
        "  Attack propagation ratio:",
        f"{causal_row['attack_propagation_ratio']:.2f}x",
    )


# ============================================================
# SAVE ALL
# ============================================================

condition_summary = pd.DataFrame(
    condition_rows
)


causal_summary = pd.DataFrame(
    causal_rows
)


all_permutations = pd.concat(
    all_permutation_tables,
    ignore_index=True,
)


condition_summary.to_csv(
    RESULT_ROOT
    /
    "all_condition_summary.csv",
    index=False,
)


causal_summary.to_csv(
    RESULT_ROOT
    /
    "normalization_causal_summary.csv",
    index=False,
)


all_permutations.to_csv(
    RESULT_ROOT
    /
    "all_rag_permutation_runs.csv",
    index=False,
)


# ============================================================
# AGGREGATE RAW VS NORMALIZED
# ============================================================

aggregate_rows = []


for condition in [
    "raw",
    "normalized",
]:

    temp = condition_summary[
        condition_summary[
            "condition"
        ]
        == condition
    ]


    aggregate_rows.append(
        {
            "condition":
                condition,

            "n_seeds":
                len(
                    temp
                ),

            "rag_reliance_mean":
                float(
                    temp[
                        "rag_joint_reallocated"
                    ].mean()
                ),

            "rag_reliance_std":
                float(
                    temp[
                        "rag_joint_reallocated"
                    ].std(
                        ddof=1
                    )
                ),

            "attack_reallocated_mean":
                float(
                    temp[
                        "attack_reallocated"
                    ].mean()
                ),

            "attack_reallocated_std":
                float(
                    temp[
                        "attack_reallocated"
                    ].std(
                        ddof=1
                    )
                ),

            "regen_reallocated_mean":
                float(
                    temp[
                        "regen_reallocated"
                    ].mean()
                ),

            "attack_delta_multiple_mean":
                float(
                    temp[
                        "attack_delta_multiple"
                    ].mean()
                ),

            "attack_delta_multiple_std":
                float(
                    temp[
                        "attack_delta_multiple"
                    ].std(
                        ddof=1
                    )
                ),

            "attack_delta_sharpe_mean":
                float(
                    temp[
                        "attack_delta_sharpe"
                    ].mean()
                ),

            "attack_delta_sharpe_std":
                float(
                    temp[
                        "attack_delta_sharpe"
                    ].std(
                        ddof=1
                    )
                ),

            "clean_multiple_mean":
                float(
                    temp[
                        "clean_multiple"
                    ].mean()
                ),

            "clean_sharpe_mean":
                float(
                    temp[
                        "clean_sharpe"
                    ].mean()
                ),
        }
    )


aggregate = pd.DataFrame(
    aggregate_rows
)


aggregate.to_csv(
    RESULT_ROOT
    /
    "condition_aggregate.csv",
    index=False,
)


# ============================================================
# PAIRED NORMALIZATION EFFECT
# ============================================================

mean_raw_reliance = float(
    causal_summary[
        "raw_rag_reliance"
    ].mean()
)


mean_normalized_reliance = float(
    causal_summary[
        "normalized_rag_reliance"
    ].mean()
)


mean_raw_attack = float(
    causal_summary[
        "raw_attack_reallocated"
    ].mean()
)


mean_normalized_attack = float(
    causal_summary[
        "normalized_attack_reallocated"
    ].mean()
)


mean_reliance_ratio = float(
    causal_summary[
        "rag_reliance_ratio"
    ].mean()
)


mean_attack_ratio = float(
    causal_summary[
        "attack_propagation_ratio"
    ].mean()
)


median_reliance_ratio = float(
    causal_summary[
        "rag_reliance_ratio"
    ].median()
)


median_attack_ratio = float(
    causal_summary[
        "attack_propagation_ratio"
    ].median()
)


n_rag_increased = int(
    (
        causal_summary[
            "normalized_rag_reliance"
        ]
        >
        causal_summary[
            "raw_rag_reliance"
        ]
    ).sum()
)


n_attack_increased = int(
    (
        causal_summary[
            "normalized_attack_reallocated"
        ]
        >
        causal_summary[
            "raw_attack_reallocated"
        ]
    ).sum()
)


# ============================================================
# RELIANCE -> ATTACK CORRELATION BY CONDITION
# ============================================================

correlation_rows = []


for condition in [
    "raw",
    "normalized",
]:

    temp = (
        condition_summary[
            condition_summary[
                "condition"
            ]
            == condition
        ]
        .sort_values(
            "seed"
        )
    )


    x = (
        temp[
            "rag_joint_reallocated"
        ]
        .astype(float)
        .to_numpy()
    )


    y = (
        temp[
            "attack_reallocated"
        ]
        .astype(float)
        .to_numpy()
    )


    if (
        len(x) >= 2
        and
        np.std(x) > 0
        and
        np.std(y) > 0
    ):

        pearson = float(
            np.corrcoef(
                x,
                y,
            )[0, 1]
        )


        x_rank = (
            pd.Series(
                x
            )
            .rank()
            .to_numpy()
        )


        y_rank = (
            pd.Series(
                y
            )
            .rank()
            .to_numpy()
        )


        spearman = float(
            np.corrcoef(
                x_rank,
                y_rank,
            )[0, 1]
        )


    else:

        pearson = np.nan
        spearman = np.nan


    correlation_rows.append(
        {
            "condition":
                condition,

            "n":
                len(
                    temp
                ),

            "pearson":
                pearson,

            "spearman":
                spearman,
        }
    )


correlation_df = pd.DataFrame(
    correlation_rows
)


correlation_df.to_csv(
    RESULT_ROOT
    /
    "reliance_attack_correlations.csv",
    index=False,
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 80)
print("NORMALIZATION CAUSAL CONTROL RESULT")
print("=" * 80)


display = causal_summary[
    [
        "seed",
        "raw_rag_reliance",
        "normalized_rag_reliance",
        "rag_reliance_ratio",
        "raw_attack_reallocated",
        "normalized_attack_reallocated",
        "attack_propagation_ratio",
        "raw_attack_delta_multiple",
        "normalized_attack_delta_multiple",
    ]
].copy()


print(
    display.to_string(
        index=False
    )
)


print("\n" + "=" * 80)
print("RAG RELIANCE: RAW VS NORMALIZED")
print("=" * 80)


print(
    "Raw:",
    f"{mean_raw_reliance:.4%}",
    "±",
    f"{causal_summary['raw_rag_reliance'].std(ddof=1):.4%}",
)


print(
    "Normalized:",
    f"{mean_normalized_reliance:.4%}",
    "±",
    f"{causal_summary['normalized_rag_reliance'].std(ddof=1):.4%}",
)


print(
    "Mean normalized/raw reliance ratio:",
    f"{mean_reliance_ratio:.2f}x",
)


print(
    "Median normalized/raw reliance ratio:",
    f"{median_reliance_ratio:.2f}x",
)


print(
    "Seeds where normalization increased RAG reliance:",
    f"{n_rag_increased}/{len(causal_summary)}",
)


print("\n" + "=" * 80)
print("ATTACK PROPAGATION: RAW VS NORMALIZED")
print("=" * 80)


print(
    "Raw attack capital reallocated:",
    f"{mean_raw_attack:.4%}",
    "±",
    f"{causal_summary['raw_attack_reallocated'].std(ddof=1):.4%}",
)


print(
    "Normalized attack capital reallocated:",
    f"{mean_normalized_attack:.4%}",
    "±",
    f"{causal_summary['normalized_attack_reallocated'].std(ddof=1):.4%}",
)


print(
    "Mean normalized/raw attack ratio:",
    f"{mean_attack_ratio:.2f}x",
)


print(
    "Median normalized/raw attack ratio:",
    f"{median_attack_ratio:.2f}x",
)


print(
    "Seeds where normalization increased attack propagation:",
    f"{n_attack_increased}/{len(causal_summary)}",
)


# ============================================================
# REGENERATION NOISE FLOOR
# ============================================================

raw_regen_mean = float(
    causal_summary[
        "raw_regen_reallocated"
    ].mean()
)


normalized_regen_mean = float(
    causal_summary[
        "normalized_regen_reallocated"
    ].mean()
)


print("\n" + "=" * 80)
print("PAIRED-CLEAN REGENERATION NOISE FLOOR")
print("=" * 80)


print(
    "Raw regeneration reallocated:",
    f"{raw_regen_mean:.4%}",
)


print(
    "Normalized regeneration reallocated:",
    f"{normalized_regen_mean:.4%}",
)


if raw_regen_mean > 0:

    print(
        "Raw attack / regeneration:",
        f"{mean_raw_attack / raw_regen_mean:.2f}x",
    )


if normalized_regen_mean > 0:

    print(
        "Normalized attack / regeneration:",
        f"{mean_normalized_attack / normalized_regen_mean:.2f}x",
    )


# ============================================================
# FINANCIAL EFFECT
# ============================================================

print("\n" + "=" * 80)
print("FINANCIAL EFFECT")
print("=" * 80)


raw_condition = condition_summary[
    condition_summary[
        "condition"
    ]
    == "raw"
]


normalized_condition = condition_summary[
    condition_summary[
        "condition"
    ]
    == "normalized"
]


print(
    "Raw attack mean Δ multiple:",
    f"{raw_condition['attack_delta_multiple'].mean():+.6f}",
    "±",
    f"{raw_condition['attack_delta_multiple'].std(ddof=1):.6f}",
)


print(
    "Normalized attack mean Δ multiple:",
    f"{normalized_condition['attack_delta_multiple'].mean():+.6f}",
    "±",
    f"{normalized_condition['attack_delta_multiple'].std(ddof=1):.6f}",
)


print(
    "Raw attack mean Δ Sharpe:",
    f"{raw_condition['attack_delta_sharpe'].mean():+.6f}",
    "±",
    f"{raw_condition['attack_delta_sharpe'].std(ddof=1):.6f}",
)


print(
    "Normalized attack mean Δ Sharpe:",
    f"{normalized_condition['attack_delta_sharpe'].mean():+.6f}",
    "±",
    f"{normalized_condition['attack_delta_sharpe'].std(ddof=1):.6f}",
)


# ============================================================
# CORRELATION
# ============================================================

print("\n" + "=" * 80)
print("RAG RELIANCE -> ATTACK PROPAGATION")
print("=" * 80)


print(
    correlation_df.to_string(
        index=False
    )
)


print(
    "\nCorrelation is exploratory only "
    "because n=5 per condition."
)


# ============================================================
# MAIN CAUSAL-CONTROL INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("CAUSAL CONTROL CHECK")
print("=" * 80)


print(
    "Same seeds:",
    SEEDS,
)

print(
    "Same architecture: A2C"
)

print(
    "Same timesteps:",
    TOTAL_TIMESTEPS,
)

print(
    "Same observation dimensions:",
    len(
        OBS_FEATURES
    ),
    "feature rows + 5x5 covariance",
)

print(
    "Same attack events:",
    len(
        pairs
    ),
)

print(
    "Same changed RAG test rows:",
    raw_changed_rows,
)

print(
    "Only intended treatment: "
    "train-fitted feature normalization."
)


print("\nResults saved to:")

print(
    RESULT_ROOT
)

print("\nDONE")