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

from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--permutations",
    type=int,
    default=10,
)

parser.add_argument(
    "--reuse-model",
    action="store_true",
    help="Reuse previously trained normalized A2C model.",
)

args = parser.parse_args()

N_PERMUTATIONS = args.permutations
REUSE_MODEL = args.reuse_model


# ============================================================
# Project
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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

SEED = 101

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

FEATURE_GROUPS = [
    "covariance",
    *TECH_FEATURES,
    *RAG_FEATURES,
]


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

OLD_EXPERIMENT_ROOT = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
)

CONFIG_PATH = (
    OLD_EXPERIMENT_ROOT
    / "experiment_config.json"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "normalized_rag_feature_importance"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_BASENAME = (
    RESULT_DIR
    / "normalized_a2c_model"
)

MODEL_ZIP = Path(
    str(MODEL_BASENAME)
    + ".zip"
)

SCALER_PATH = (
    RESULT_DIR
    / "train_only_scaler.json"
)


# ============================================================
# Seed
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# Load original experiment dates
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    experiment_config = json.load(f)


COMMON_START = experiment_config.get(
    "common_start",
    "2018-02-09",
)

TRAIN_END = experiment_config.get(
    "train_end_exclusive",
    "2022-01-01",
)

VALIDATION_END = experiment_config.get(
    "validation_end_exclusive",
    "2023-01-01",
)

TEST_END = experiment_config.get(
    "test_end_exclusive",
    "2024-01-01",
)


print("=" * 70)
print("EXPERIMENT DATES")
print("=" * 70)

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
# Show features
# ============================================================

print("\n" + "=" * 70)
print("OBSERVATION FEATURES")
print("=" * 70)

print("\nTechnical indicators:")

for i, feature in enumerate(
    TECH_FEATURES,
    start=1,
):
    print(
        f"{i:2d}.",
        feature,
    )


print("\nRAG features:")

for feature in RAG_FEATURES:
    print(
        "   ",
        feature,
    )


print(
    "\nSpecial block: covariance 5x5"
)


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

for feature in RAG_FEATURES:

    events[feature] = pd.to_numeric(
        events[feature],
        errors="coerce",
    )


if (
    events[
        RAG_FEATURES
    ]
    .isna()
    .any()
    .any()
):

    raise RuntimeError(
        "RAG event table contains missing feature values."
    )


# ============================================================
# Covariance
# ============================================================

def add_covariance_matrix(
    df,
    lookback=252,
):

    df = df.copy()

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

        covariance = (
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
                    covariance,
            }
        )


    cov_df = pd.DataFrame(
        cov_rows
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
# Build market data
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
            "YahooDownloader returned no data."
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


    complete_counts = market.groupby(
        "date"
    )["tic"].nunique()

    complete_dates = complete_counts[
        complete_counts
        == len(TIC_ORDER)
    ].index

    market = market[
        market["date"].isin(
            complete_dates
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
        "Complete dates:",
        market["date"].nunique(),
    )

    return market


# ============================================================
# Strict point-in-time RAG merge
# ============================================================

def merge_rag(
    market,
    event_table,
):

    event_table = event_table.copy()

    event_table[
        "source_filing_date"
    ] = pd.to_datetime(
        event_table["event_date"]
    )

    parts = []


    for ticker in TIC_ORDER:

        market_ticker = market[
            market["tic"]
            == ticker
        ].copy()

        event_ticker = event_table[
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


        market_ticker = market_ticker.sort_values(
            "date_dt"
        )

        event_ticker = event_ticker.sort_values(
            "source_filing_date"
        )


        merged = pd.merge_asof(
            market_ticker,
            event_ticker,
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
            COMMON_START
        )
    ].copy()

    panel = panel[
        panel["date_dt"]
        <
        pd.Timestamp(
            TEST_END
        )
    ].copy()


    panel = (
        panel.sort_values(
            [
                "date",
                "tic",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    missing = (
        panel[
            RAG_FEATURES
        ]
        .isna()
        .sum()
    )


    if missing.sum() != 0:

        raise RuntimeError(
            f"Missing RAG features:\n{missing}"
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


    if len(leakage) != 0:

        raise RuntimeError(
            "Same-day/future RAG leakage detected."
        )


    return panel


# ============================================================
# Fit scalar normalization using TRAIN ONLY
#
# Every:
#   feature x ticker
#
# gets its own mean and std.
#
# This is important because close_30_sma for NVDA and AAPL
# have very different raw price levels.
# ============================================================

def fit_scalar_scaler(
    train_df,
):

    scaler = {}


    for feature in OBS_FEATURES:

        scaler[feature] = {}


        for ticker in TIC_ORDER:

            values = (
                train_df.loc[
                    train_df["tic"]
                    == ticker,
                    feature,
                ]
                .astype(float)
                .to_numpy()
            )


            if len(values) == 0:

                raise RuntimeError(
                    f"No training values for "
                    f"{ticker} {feature}"
                )


            mean = float(
                np.mean(values)
            )

            std = float(
                np.std(
                    values,
                    ddof=0,
                )
            )


            if (
                not np.isfinite(std)
                or std < 1e-12
            ):

                print(
                    "WARNING:",
                    ticker,
                    feature,
                    "has near-zero std; using std=1.",
                )

                std = 1.0


            scaler[
                feature
            ][
                ticker
            ] = {
                "mean":
                    mean,

                "std":
                    std,
            }


    return scaler


# ============================================================
# Fit covariance scaler using TRAIN ONLY
#
# One mean/std for each cell [i,j] over TRAIN dates.
#
# Shape:
#   mean = 5x5
#   std  = 5x5
# ============================================================

def fit_covariance_scaler(
    train_df,
):

    matrices = []


    for date in train_df[
        "date"
    ].unique():

        matrix = (
            train_df.loc[
                train_df["date"]
                == date,
                "cov_list",
            ]
            .iloc[0]
        )

        matrix = np.asarray(
            matrix,
            dtype=np.float64,
        )


        if matrix.shape != (
            len(TIC_ORDER),
            len(TIC_ORDER),
        ):

            raise RuntimeError(
                f"Unexpected covariance shape: "
                f"{matrix.shape}"
            )


        matrices.append(
            matrix
        )


    stack = np.stack(
        matrices,
        axis=0,
    )


    mean = np.mean(
        stack,
        axis=0,
    )

    std = np.std(
        stack,
        axis=0,
        ddof=0,
    )


    std = np.where(
        np.isfinite(std)
        &
        (
            std >= 1e-12
        ),
        std,
        1.0,
    )


    return {
        "mean":
            mean,

        "std":
            std,
    }


# ============================================================
# Apply TRAIN-FITTED normalization
# ============================================================

def transform_panel(
    df,
    scalar_scaler,
    covariance_scaler,
):

    output = df.copy()


    # --------------------------------------------------------
    # Scalar / row features
    # --------------------------------------------------------

    for feature in OBS_FEATURES:

        for ticker in TIC_ORDER:

            mask = (
                output["tic"]
                == ticker
            )

            mean = (
                scalar_scaler[
                    feature
                ][
                    ticker
                ][
                    "mean"
                ]
            )

            std = (
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
                    ].astype(float)
                    - mean
                )
                / std
            )


    # --------------------------------------------------------
    # Covariance matrices
    # --------------------------------------------------------

    cov_mean = (
        covariance_scaler[
            "mean"
        ]
    )

    cov_std = (
        covariance_scaler[
            "std"
        ]
    )


    output[
        "cov_list"
    ] = output[
        "cov_list"
    ].apply(
        lambda matrix:
            (
                np.asarray(
                    matrix,
                    dtype=np.float64,
                )
                - cov_mean
            )
            / cov_std
    )


    return output


# ============================================================
# Save scaler
# ============================================================

def save_scaler(
    scalar_scaler,
    covariance_scaler,
):

    payload = {
        "fitted_on":
            "training_period_only",

        "train_start":
            COMMON_START,

        "train_end_exclusive":
            TRAIN_END,

        "scalar_scaler":
            scalar_scaler,

        "covariance_mean":
            covariance_scaler[
                "mean"
            ].tolist(),

        "covariance_std":
            covariance_scaler[
                "std"
            ].tolist(),

        "ticker_order":
            TIC_ORDER,

        "features":
            OBS_FEATURES,
    }


    with open(
        SCALER_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
            indent=2,
        )


# ============================================================
# Normalization QA
# ============================================================

def normalization_qa(
    normalized_train,
):

    rows = []


    for feature in OBS_FEATURES:

        for ticker in TIC_ORDER:

            values = (
                normalized_train.loc[
                    normalized_train["tic"]
                    == ticker,
                    feature,
                ]
                .astype(float)
            )


            rows.append(
                {
                    "feature":
                        feature,

                    "ticker":
                        ticker,

                    "mean":
                        float(
                            values.mean()
                        ),

                    "std":
                        float(
                            values.std(
                                ddof=0
                            )
                        ),

                    "min":
                        float(
                            values.min()
                        ),

                    "max":
                        float(
                            values.max()
                        ),
                }
            )


    qa = pd.DataFrame(
        rows
    )


    qa.to_csv(
        RESULT_DIR
        / "normalization_qa_scalar.csv",
        index=False,
    )


    # --------------------------------------------------------
    # covariance QA
    # --------------------------------------------------------

    matrices = []


    for date in normalized_train[
        "date"
    ].unique():

        matrix = (
            normalized_train.loc[
                normalized_train["date"]
                == date,
                "cov_list",
            ]
            .iloc[0]
        )

        matrices.append(
            np.asarray(
                matrix,
                dtype=float,
            )
        )


    stack = np.stack(
        matrices,
        axis=0,
    )


    cov_mean = np.mean(
        stack,
        axis=0,
    )

    cov_std = np.std(
        stack,
        axis=0,
        ddof=0,
    )


    pd.DataFrame(
        cov_mean,
        index=TIC_ORDER,
        columns=TIC_ORDER,
    ).to_csv(
        RESULT_DIR
        / "normalization_qa_cov_mean.csv"
    )


    pd.DataFrame(
        cov_std,
        index=TIC_ORDER,
        columns=TIC_ORDER,
    ).to_csv(
        RESULT_DIR
        / "normalization_qa_cov_std.csv"
    )


    print("\n" + "=" * 70)
    print("NORMALIZATION QA")
    print("=" * 70)


    print(
        "Maximum absolute scalar train mean:",
        f"{qa['mean'].abs().max():.8f}",
    )


    print(
        "Scalar train std range:",
        f"{qa['std'].min():.6f}",
        "to",
        f"{qa['std'].max():.6f}",
    )


    print(
        "Maximum absolute covariance train mean:",
        f"{np.max(np.abs(cov_mean)):.8f}",
    )


    print(
        "Covariance train std range:",
        f"{np.min(cov_std):.6f}",
        "to",
        f"{np.max(cov_std):.6f}",
    )


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
            OBS_FEATURES,
    }


# ============================================================
# Evaluation
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
# Metrics
# ============================================================

def calculate_metrics(
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


    n = len(returns)


    if n == 0:

        raise RuntimeError(
            "No returns."
        )


    wealth = (
        1.0 + returns
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
        returns.std(
            ddof=1
        )
    )


    annual_volatility = (
        daily_std
        * math.sqrt(252)
    )


    sharpe = (
        float(
            returns.mean()
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
            annual_volatility,

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
# Weight helpers
# ============================================================

def get_weight_table(
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


def policy_distance(
    reference_weights,
    candidate_weights,
):

    reference = get_weight_table(
        reference_weights
    )

    candidate = get_weight_table(
        candidate_weights
    )


    n = min(
        len(reference),
        len(candidate),
    )


    reference = reference.iloc[
        :n
    ].copy()

    candidate = candidate.iloc[
        :n
    ].copy()


    diff = (
        candidate
        - reference
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
# Permute one normalized scalar feature
#
# Shuffle through time separately within each ticker.
# Distribution is unchanged.
# ============================================================

def shuffle_scalar_feature(
    df,
    feature,
    seed,
):

    output = df.copy()

    rng = np.random.default_rng(
        seed
    )


    column_position = (
        output.columns.get_loc(
            feature
        )
    )


    ticker_array = (
        output["tic"]
        .to_numpy()
    )


    for ticker in TIC_ORDER:

        positions = np.where(
            ticker_array
            == ticker
        )[0]


        values = (
            output.iloc[
                positions,
                column_position
            ]
            .to_numpy(
                copy=True
            )
        )


        permutation = rng.permutation(
            len(values)
        )


        output.iloc[
            positions,
            column_position
        ] = values[
            permutation
        ]


    return output


# ============================================================
# Permute normalized covariance block
#
# Whole 5x5 matrix moves together across dates.
# ============================================================

def shuffle_covariance(
    df,
    seed,
):

    output = df.copy()

    rng = np.random.default_rng(
        seed
    )


    unique_dates = list(
        output[
            "date"
        ].unique()
    )


    matrices = []


    for date in unique_dates:

        matrix = (
            output.loc[
                output["date"]
                == date,
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


    shuffled = [
        matrices[i]
        for i in permutation
    ]


    matrix_map = {
        date:
            matrix
        for date, matrix
        in zip(
            unique_dates,
            shuffled,
        )
    }


    output[
        "cov_list"
    ] = output[
        "date"
    ].map(
        matrix_map
    )


    return output


# ============================================================
# Build raw point-in-time panel
# ============================================================

market = build_market()

raw_panel = merge_rag(
    market,
    events,
)


# ============================================================
# TRAIN-ONLY raw subset used to fit scaler
# ============================================================

raw_train_for_scaler = raw_panel[
    (
        raw_panel["date_dt"]
        >= pd.Timestamp(
            COMMON_START
        )
    )
    &
    (
        raw_panel["date_dt"]
        < pd.Timestamp(
            TRAIN_END
        )
    )
].copy()


print("\n" + "=" * 70)
print("FIT NORMALIZATION USING TRAIN ONLY")
print("=" * 70)

print(
    "Training rows for scaler:",
    len(
        raw_train_for_scaler
    ),
)

print(
    "Training dates for scaler:",
    raw_train_for_scaler[
        "date"
    ].nunique(),
)


scalar_scaler = fit_scalar_scaler(
    raw_train_for_scaler
)

covariance_scaler = fit_covariance_scaler(
    raw_train_for_scaler
)


save_scaler(
    scalar_scaler,
    covariance_scaler,
)


# ============================================================
# Transform COMPLETE panel using TRAIN stats
# ============================================================

normalized_panel = transform_panel(
    raw_panel,
    scalar_scaler,
    covariance_scaler,
)


# ============================================================
# Splits
# ============================================================

train_df = data_split(
    normalized_panel,
    COMMON_START,
    TRAIN_END,
)

validation_df = data_split(
    normalized_panel,
    TRAIN_END,
    VALIDATION_END,
)

test_df = data_split(
    normalized_panel,
    VALIDATION_END,
    TEST_END,
)


print("\n" + "=" * 70)
print("NORMALIZED SPLITS")
print("=" * 70)

print(
    "Train dates:",
    train_df["date"].nunique(),
)

print(
    "Validation dates:",
    validation_df["date"].nunique(),
)

print(
    "Test dates:",
    test_df["date"].nunique(),
)


# ============================================================
# Normalization QA
# ============================================================

normalization_qa(
    train_df
)


# ============================================================
# Train NORMALIZED A2C
#
# Same public FinRL-X A2C parameters + explicit seed.
# ============================================================

if (
    REUSE_MODEL
    and MODEL_ZIP.exists()
):

    print("\n" + "=" * 70)
    print("REUSE NORMALIZED A2C")
    print("=" * 70)

    print(
        MODEL_ZIP
    )

    from stable_baselines3 import A2C

    model = A2C.load(
        str(
            MODEL_ZIP
        )
    )


else:

    print("\n" + "=" * 70)
    print("TRAIN NORMALIZED A2C")
    print("=" * 70)

    set_seed(SEED)


    train_env = StockPortfolioEnv(
        df=train_df,
        **make_env_kwargs(),
    )


    sb_env, _ = (
        train_env.get_sb_env()
    )


    agent = DRLAgent(
        env=sb_env
    )


    if torch.cuda.is_available():

        a2c_params = {
            "n_steps":
                1024,

            "ent_coef":
                0.005,

            "learning_rate":
                0.0002,

            "device":
                "cuda",

        }

    else:

        a2c_params = {
            "n_steps":
                5,

            "ent_coef":
                0.005,

            "learning_rate":
                0.0002,

        }


    model = agent.get_model(
        model_name="a2c",
        model_kwargs=a2c_params,
        seed=SEED,
    )


    model = agent.train_model(
        model=model,
        tb_log_name="normalized_a2c",
        total_timesteps=50_000,
    )


    model.save(
        str(
            MODEL_BASENAME
        )
    )


    print(
        "Saved model:",
        MODEL_ZIP,
    )


# ============================================================
# Validation
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED VALIDATION")
print("=" * 70)


validation_returns, validation_weights = (
    evaluate_model(
        model,
        validation_df,
        quiet=False,
    )
)


validation_metrics = (
    calculate_metrics(
        validation_returns
    )
)


print(
    "Multiple:",
    f"{validation_metrics['final_multiple']:.6f}",
)

print(
    "Annual return:",
    f"{validation_metrics['annual_return']:.2%}",
)

print(
    "Sharpe:",
    f"{validation_metrics['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{validation_metrics['max_drawdown']:.2%}",
)


validation_returns.to_csv(
    RESULT_DIR
    / "validation_returns.csv",
    index=False,
)

validation_weights.to_csv(
    RESULT_DIR
    / "validation_weights.csv",
)


# ============================================================
# Test reference
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED TEST REFERENCE")
print("=" * 70)


normal_returns, normal_weights = (
    evaluate_model(
        model,
        test_df,
        quiet=False,
    )
)


normal_metrics = (
    calculate_metrics(
        normal_returns
    )
)


print(
    "Multiple:",
    f"{normal_metrics['final_multiple']:.6f}",
)

print(
    "Annual return:",
    f"{normal_metrics['annual_return']:.2%}",
)

print(
    "Annual vol:",
    f"{normal_metrics['annual_volatility']:.2%}",
)

print(
    "Sharpe:",
    f"{normal_metrics['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{normal_metrics['max_drawdown']:.2%}",
)


normal_returns.to_csv(
    RESULT_DIR
    / "normalized_test_returns.csv",
    index=False,
)

normal_weights.to_csv(
    RESULT_DIR
    / "normalized_test_weights.csv",
)


with open(
    RESULT_DIR
    / "normalized_model_metrics.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {

            "train_timesteps":
                50_000,

            "validation":
                validation_metrics,

            "test":
                normal_metrics,
        },
        f,
        indent=2,
    )


# ============================================================
# Repeated normalized feature importance
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED ALL-FEATURE PERMUTATION IMPORTANCE")
print("=" * 70)

print(
    "Feature groups:",
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

        permutation_seed = (
            100_000
            + feature_index * 1000
            + repetition
        )


        if feature == "covariance":

            modified_test = shuffle_covariance(
                test_df,
                permutation_seed,
            )

        else:

            modified_test = shuffle_scalar_feature(
                test_df,
                feature,
                permutation_seed,
            )


        (
            modified_returns,
            modified_weights,
        ) = evaluate_model(
            model,
            modified_test,
            quiet=True,
        )


        modified_metrics = (
            calculate_metrics(
                modified_returns
            )
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
                    permutation_seed,

                **shift,

                "final_multiple":
                    modified_metrics[
                        "final_multiple"
                    ],

                "delta_final_multiple":
                    (
                        modified_metrics[
                            "final_multiple"
                        ]
                        -
                        normal_metrics[
                            "final_multiple"
                        ]
                    ),

                "sharpe":
                    modified_metrics[
                        "sharpe"
                    ],

                "delta_sharpe":
                    (
                        modified_metrics[
                            "sharpe"
                        ]
                        -
                        normal_metrics[
                            "sharpe"
                        ]
                    ),

                "max_drawdown":
                    modified_metrics[
                        "max_drawdown"
                    ],

                "delta_max_drawdown":
                    (
                        modified_metrics[
                            "max_drawdown"
                        ]
                        -
                        normal_metrics[
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
            f"{modified_metrics['sharpe'] - normal_metrics['sharpe']:+.4f}",
        )


# ============================================================
# Raw results
# ============================================================

raw_results = pd.DataFrame(
    raw_rows
)


raw_results.to_csv(
    RESULT_DIR
    / "normalized_feature_permutation_runs.csv",
    index=False,
)


# ============================================================
# Aggregate feature importance
# ============================================================

summary_rows = []


for feature in FEATURE_GROUPS:

    temp = raw_results[
        raw_results["feature"]
        == feature
    ].copy()


    if feature == "covariance":

        input_elements = (
            len(TIC_ORDER)
            * len(TIC_ORDER)
        )

        feature_type = (
            "normalized 5x5 block"
        )

    else:

        input_elements = (
            len(TIC_ORDER)
        )

        feature_type = (
            "normalized feature row"
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
                ),

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
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# Policy ranking
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
    / "normalized_policy_importance_ranking.csv",
    index=False,
)


# ============================================================
# Scalar-only ranking
# ============================================================

scalar_ranking = (
    summary[
        summary["feature"]
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
    / "normalized_scalar_feature_ranking.csv",
    index=False,
)


# ============================================================
# Performance sensitivity
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
    / "normalized_performance_importance_ranking.csv",
    index=False,
)


# ============================================================
# RAG aggregate vs technical aggregate
# ============================================================

rag_importance = policy_ranking[
    policy_ranking[
        "feature"
    ].isin(
        RAG_FEATURES
    )
][
    "mean_reallocated_weight"
].mean()


technical_importance = policy_ranking[
    policy_ranking[
        "feature"
    ].isin(
        TECH_FEATURES
    )
][
    "mean_reallocated_weight"
].mean()


# ============================================================
# Final result
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED FEATURE IMPORTANCE RESULT")
print("=" * 70)

print(
    "Fixed training seed:",
    SEED,
)

print(
    "Training timesteps:",
    50_000,
)


print("\nNORMALIZED TEST")

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


print("\n" + "=" * 70)
print("NORMALIZED POLICY IMPORTANCE RANKING")
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
print("NORMALIZED SCALAR FEATURE RANKING")
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
print("NORMALIZED PERFORMANCE SENSITIVITY")
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


print("\n" + "=" * 70)
print("RAG VS TECHNICAL RELIANCE")
print("=" * 70)

print(
    "Mean technical-feature reallocated weight:",
    f"{technical_importance:.6%}",
)

print(
    "Mean RAG-feature reallocated weight:",
    f"{rag_importance:.6%}",
)


if technical_importance > 0:

    print(
        "RAG / technical reliance ratio:",
        f"{rag_importance / technical_importance:.4f}",
    )


print("\nResults saved to:")

print(
    RESULT_DIR
)

print("\nDONE")