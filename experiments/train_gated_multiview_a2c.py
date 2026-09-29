import argparse
import contextlib
import io
import json
import math
import random
import sys
from collections import Counter
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
    "--timesteps",
    type=int,
    default=50_000,
)

parser.add_argument(
    "--permutations",
    type=int,
    default=10,
)

args = parser.parse_args()

SEEDS = args.seeds
TOTAL_TIMESTEPS = args.timesteps
N_PERMUTATIONS = args.permutations


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

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

RAG_MEDIAN_COLUMNS = [
    "median_impact_score",
    "median_risk_score",
    "median_confidence",
]

RAG_MAPPING = {
    "impact_score":
        "median_impact_score",

    "risk_score":
        "median_risk_score",

    "confidence":
        "median_confidence",
}

OBS_FEATURES = (
    TECH_FEATURES
    +
    RAG_FEATURES
)

DOWNLOAD_START = "2017-01-01"
DOWNLOAD_END = "2024-01-01"

LOOKBACK = 252

INITIAL_AMOUNT = 1_000_000


# ============================================================
# PATHS
# ============================================================

MULTIVIEW_EVENT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "features"
    / "multiview5"
    / "sec_rag_multiview5_events.csv"
)

CONFIG_PATH = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
    / "experiment_config.json"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "gated_multiview5_a2c"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SCALER_PATH = (
    RESULT_DIR
    / "gated_train_scaler.json"
)

OLD_CONTROL_PATH = (
    ROOT
    / "results"
    / "normalization_causal_control"
    / "all_condition_summary.csv"
)


for path in [
    MULTIVIEW_EVENT_PATH,
    CONFIG_PATH,
]:

    if not path.exists():
        raise FileNotFoundError(
            path
        )


# ============================================================
# DATES
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    cfg = json.load(
        f
    )


COMMON_START = cfg[
    "common_start"
]

TRAIN_END = cfg[
    "train_end_exclusive"
]

VALIDATION_END = cfg[
    "validation_end_exclusive"
]

TEST_END = cfg[
    "test_end_exclusive"
]


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# MULTI-VIEW EVENTS
# ============================================================

mv_events = pd.read_csv(
    MULTIVIEW_EVENT_PATH
)

mv_events["ticker"] = (
    mv_events["ticker"]
    .astype(str)
    .str.upper()
)

mv_events["event_date"] = pd.to_datetime(
    mv_events["event_date"]
).dt.normalize()


NUMERIC_EVENT_COLS = [
    "median_impact_score",
    "median_risk_score",
    "median_confidence",
    "reliability_gate",
]


for col in NUMERIC_EVENT_COLS:

    mv_events[col] = pd.to_numeric(
        mv_events[col],
        errors="coerce",
    )


if mv_events[
    NUMERIC_EVENT_COLS
].isna().any().any():

    raise RuntimeError(
        "Missing multi-view event values."
    )


print("=" * 75)
print("PUBLICATION-GRADE GATED MULTI-VIEW A2C")
print("=" * 75)

print(
    "Events:",
    len(mv_events),
)

print(
    "Mean gate:",
    f"{mv_events['reliability_gate'].mean():.4f}",
)

print(
    "Seeds:",
    SEEDS,
)


# ============================================================
# COVARIANCE
# ============================================================

def add_covariance(
    df,
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
        LOOKBACK,
        len(dates),
    ):

        window = price.iloc[
            i - LOOKBACK:
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


    return (
        df.merge(
            pd.DataFrame(
                rows
            ),
            on="date",
            how="inner",
        )
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

print("\nBuilding market data...")


raw = YahooDownloader(
    start_date=DOWNLOAD_START,
    end_date=DOWNLOAD_END,
    ticker_list=TIC_ORDER,
).fetch_data()


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
    counts
    ==
    len(
        TIC_ORDER
    )
].index


market = market[
    market["date"].isin(
        good_dates
    )
].copy()


market = add_covariance(
    market
)


market["date_dt"] = pd.to_datetime(
    market["date"]
)


# ============================================================
# STRICT POINT-IN-TIME MERGE
# ============================================================

pieces = []


for ticker in TIC_ORDER:

    m = market[
        market["tic"]
        ==
        ticker
    ].copy()


    e = mv_events[
        mv_events["ticker"]
        ==
        ticker
    ][
        [
            "event_date",
            "median_impact_score",
            "median_risk_score",
            "median_confidence",
            "reliability_gate",
        ]
    ].copy()


    e = e.rename(
        columns={
            "event_date":
                "source_filing_date"
        }
    )


    merged = pd.merge_asof(
        m.sort_values(
            "date_dt"
        ),
        e.sort_values(
            "source_filing_date"
        ),
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
        panel["date_dt"]
        >=
        pd.Timestamp(
            COMMON_START
        )
    )
    &
    (
        panel["date_dt"]
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
    NUMERIC_EVENT_COLS
].isna().any().any():

    raise RuntimeError(
        "Missing gated RAG values after merge."
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


if len(leakage):

    raise RuntimeError(
        "Information leakage detected."
    )


# ============================================================
# TRAIN-ONLY RAW SLICE
# ============================================================

train_raw = panel[
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
        <
        pd.Timestamp(
            TRAIN_END
        )
    )
].copy()


# ============================================================
# FIT SCALERS
# ============================================================

scalar_scaler = {}


SCALER_INPUTS = (
    TECH_FEATURES
    +
    RAG_MEDIAN_COLUMNS
)


for feature in SCALER_INPUTS:

    scalar_scaler[
        feature
    ] = {}


    for ticker in TIC_ORDER:

        values = (
            train_raw.loc[
                train_raw["tic"]
                ==
                ticker,
                feature,
            ]
            .astype(float)
            .to_numpy()
        )


        mean = float(
            values.mean()
        )

        std = float(
            values.std(
                ddof=0
            )
        )


        if (
            not np.isfinite(std)
            or
            std < 1e-12
        ):

            std = 1.0


        scalar_scaler[
            feature
        ][
            ticker
        ] = {
            "mean":
                mean,

            "std":
                std,
        }


# ============================================================
# COV SCALER
# ============================================================

cov_matrices = []


for date in train_raw[
    "date"
].unique():

    cov_matrices.append(
        np.asarray(
            train_raw.loc[
                train_raw["date"]
                ==
                date,
                "cov_list",
            ].iloc[0],
            dtype=float,
        )
    )


cov_stack = np.stack(
    cov_matrices
)

cov_mean = cov_stack.mean(
    axis=0
)

cov_std = cov_stack.std(
    axis=0,
    ddof=0,
)


cov_std = np.where(
    (
        np.isfinite(
            cov_std
        )
    )
    &
    (
        cov_std
        >=
        1e-12
    ),
    cov_std,
    1.0,
)


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    SCALER_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            "training_only":
                True,

            "scalar_scaler":
                scalar_scaler,

            "covariance_mean":
                cov_mean.tolist(),

            "covariance_std":
                cov_std.tolist(),

            "ticker_order":
                TIC_ORDER,

            "gate_formula":
                (
                    "g = normalized_direction_consensus "
                    "* (1 - normalized_MAD_dispersion)"
                ),
        },
        f,
        indent=2,
    )


# ============================================================
# TRANSFORM
#
# Technical:
#
# z = (x - mu) / sigma
#
# RAG:
#
# z_gated = reliability_gate * (median - mu) / sigma
#
# Therefore g=0 maps the RAG signal exactly to the
# training mean in standardized space.
# ============================================================

transformed = panel.copy()


for feature in TECH_FEATURES:

    for ticker in TIC_ORDER:

        mask = (
            transformed["tic"]
            ==
            ticker
        )


        stats = scalar_scaler[
            feature
        ][
            ticker
        ]


        transformed.loc[
            mask,
            feature,
        ] = (
            (
                transformed.loc[
                    mask,
                    feature,
                ].astype(float)
                -
                stats["mean"]
            )
            /
            stats["std"]
        )


for output_feature, median_feature in (
    RAG_MAPPING.items()
):

    transformed[
        output_feature
    ] = 0.0


    for ticker in TIC_ORDER:

        mask = (
            transformed["tic"]
            ==
            ticker
        )


        stats = scalar_scaler[
            median_feature
        ][
            ticker
        ]


        z = (
            (
                transformed.loc[
                    mask,
                    median_feature,
                ].astype(float)
                -
                stats["mean"]
            )
            /
            stats["std"]
        )


        gate = (
            transformed.loc[
                mask,
                "reliability_gate",
            ].astype(float)
        )


        transformed.loc[
            mask,
            output_feature,
        ] = (
            gate
            *
            z
        )


transformed[
    "cov_list"
] = transformed[
    "cov_list"
].apply(
    lambda matrix:
        (
            np.asarray(
                matrix,
                dtype=float,
            )
            -
            cov_mean
        )
        /
        cov_std
)


# ============================================================
# SPLITS
# ============================================================

train_df = data_split(
    transformed,
    COMMON_START,
    TRAIN_END,
)

validation_df = data_split(
    transformed,
    TRAIN_END,
    VALIDATION_END,
)

test_df = data_split(
    transformed,
    VALIDATION_END,
    TEST_END,
)


print(
    "\nTrain dates:",
    train_df[
        "date"
    ].nunique(),
)

print(
    "Validation dates:",
    validation_df[
        "date"
    ].nunique(),
)

print(
    "Test dates:",
    test_df[
        "date"
    ].nunique(),
)


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
# EVAL
# ============================================================

def evaluate(
    model,
    df,
):

    buffer = io.StringIO()


    with contextlib.redirect_stdout(
        buffer
    ):

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
                or
                truncated
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
# METRICS
# ============================================================

def metrics(
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
        <
        1e-15
    ):
        r = r.iloc[
            1:
        ]


    n = len(r)


    wealth = (
        1.0
        +
        r
    ).cumprod()


    multiple = float(
        wealth.iloc[-1]
    )


    std = float(
        r.std(
            ddof=1
        )
    )


    sharpe = (
        float(
            r.mean()
        )
        /
        std
        *
        math.sqrt(
            252
        )
        if std > 0
        else np.nan
    )


    drawdown = (
        wealth
        /
        wealth.cummax()
        -
        1
    )


    return {
        "multiple":
            multiple,

        "sharpe":
            sharpe,

        "mdd":
            float(
                drawdown.min()
            ),
    }


# ============================================================
# POLICY DISTANCE
# ============================================================

def policy_distance(
    ref,
    candidate,
):

    a = (
        ref[
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


    b = (
        candidate[
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


    n = min(
        len(a),
        len(b),
    )


    diff = (
        b.iloc[
            :n
        ]
        -
        a.iloc[
            :n
        ]
    ).abs()


    l1 = diff.sum(
        axis=1
    )


    return float(
        l1.mean()
        /
        2.0
    )


# ============================================================
# JOINT RAG PERMUTATION
# ============================================================

def shuffle_rag(
    df,
    seed,
):

    out = df.copy()

    rng = np.random.default_rng(
        seed
    )


    positions = [
        out.columns.get_loc(
            x
        )
        for x in RAG_FEATURES
    ]


    tic_array = out[
        "tic"
    ].to_numpy()


    for ticker in TIC_ORDER:

        rows = np.where(
            tic_array
            ==
            ticker
        )[0]


        original = (
            out.iloc[
                rows,
                positions
            ]
            .to_numpy(
                copy=True
            )
        )


        permutation = rng.permutation(
            len(rows)
        )


        out.iloc[
            rows,
            positions
        ] = original[
            permutation
        ]


    return out


# ============================================================
# TRAIN / EVALUATE SEEDS
# ============================================================

summary_rows = []


for seed in SEEDS:

    print("\n" + "=" * 75)

    print(
        "SEED",
        seed,
    )

    print("=" * 75)


    seed_dir = (
        RESULT_DIR
        /
        f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    model_base = (
        seed_dir
        /
        "gated_multiview5_a2c"
    )

    model_zip = Path(
        str(
            model_base
        )
        +
        ".zip"
    )


    if model_zip.exists():

        print(
            "Reusing:",
            model_zip,
        )


        model = A2C.load(
            str(
                model_zip
            )
        )


    else:

        set_seed(
            seed
        )


        train_env = StockPortfolioEnv(
            df=train_df,
            **env_kwargs(),
        )


        sb_env, _ = (
            train_env.get_sb_env()
        )


        agent = DRLAgent(
            env=sb_env
        )


        model = agent.get_model(
            model_name="a2c",
            model_kwargs={
                "n_steps":
                    5,

                "ent_coef":
                    0.005,

                "learning_rate":
                    0.0002,

                "device":
                    "cpu",
            },
            seed=seed,
        )


        model = agent.train_model(
            model=model,
            tb_log_name=(
                f"gated_mv5_seed_{seed}"
            ),
            total_timesteps=TOTAL_TIMESTEPS,
        )


        model.save(
            str(
                model_base
            )
        )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_returns, _ = evaluate(
        model,
        validation_df,
    )


    validation_metrics = metrics(
        validation_returns
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_returns, test_weights = evaluate(
        model,
        test_df,
    )


    test_metrics = metrics(
        test_returns
    )


    # --------------------------------------------------------
    # RAG reliance
    # --------------------------------------------------------

    reliance = []


    for rep in range(
        N_PERMUTATIONS
    ):

        shuffled = shuffle_rag(
            test_df,
            seed
            *
            100_000
            +
            rep,
        )


        _, shuffled_weights = evaluate(
            model,
            shuffled,
        )


        reliance.append(
            policy_distance(
                test_weights,
                shuffled_weights,
            )
        )


    rag_reliance = float(
        np.mean(
            reliance
        )
    )


    summary_rows.append(
        {
            "seed":
                seed,

            "validation_multiple":
                validation_metrics[
                    "multiple"
                ],

            "validation_sharpe":
                validation_metrics[
                    "sharpe"
                ],

            "test_multiple":
                test_metrics[
                    "multiple"
                ],

            "test_sharpe":
                test_metrics[
                    "sharpe"
                ],

            "test_mdd":
                test_metrics[
                    "mdd"
                ],

            "gated_rag_reliance":
                rag_reliance,
        }
    )


    print(
        "Test multiple:",
        f"{test_metrics['multiple']:.6f}",
    )

    print(
        "Test Sharpe:",
        f"{test_metrics['sharpe']:.4f}",
    )

    print(
        "MDD:",
        f"{test_metrics['mdd']:.2%}",
    )

    print(
        "Gated RAG reliance:",
        f"{rag_reliance:.4%}",
    )


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    summary_rows
)


summary.to_csv(
    RESULT_DIR
    / "gated_multiseed_summary.csv",
    index=False,
)


print("\n" + "=" * 75)
print("GATED MULTI-VIEW A2C RESULT")
print("=" * 75)


print(
    summary.to_string(
        index=False
    )
)


print("\nCROSS-SEED")

print(
    "Test multiple:",
    f"{summary['test_multiple'].mean():.6f}",
    "±",
    f"{summary['test_multiple'].std(ddof=1):.6f}",
)

print(
    "Test Sharpe:",
    f"{summary['test_sharpe'].mean():.4f}",
    "±",
    f"{summary['test_sharpe'].std(ddof=1):.4f}",
)

print(
    "RAG reliance:",
    f"{summary['gated_rag_reliance'].mean():.4%}",
    "±",
    f"{summary['gated_rag_reliance'].std(ddof=1):.4%}",
)


# ============================================================
# OPTIONAL PRIOR NORMALIZED SINGLE-SHOT COMPARISON
# ============================================================

if OLD_CONTROL_PATH.exists():

    old = pd.read_csv(
        OLD_CONTROL_PATH
    )


    old = old[
        old["condition"]
        ==
        "normalized"
    ].copy()


    print("\n" + "=" * 75)
    print("SINGLE-SHOT NORMALIZED REFERENCE")
    print("=" * 75)


    print(
        "Single-shot clean multiple:",
        f"{old['clean_multiple'].mean():.6f}",
    )

    print(
        "Single-shot clean Sharpe:",
        f"{old['clean_sharpe'].mean():.4f}",
    )

    print(
        "Single-shot RAG reliance:",
        f"{old['rag_joint_reallocated'].mean():.4%}",
    )


    print("\nGATED - SINGLE-SHOT")

    print(
        "Δ mean multiple:",
        f"{summary['test_multiple'].mean() - old['clean_multiple'].mean():+.6f}",
    )

    print(
        "Δ mean Sharpe:",
        f"{summary['test_sharpe'].mean() - old['clean_sharpe'].mean():+.4f}",
    )

    print(
        "Δ RAG reliance:",
        f"{summary['gated_rag_reliance'].mean() - old['rag_joint_reallocated'].mean():+.4%}",
    )


print("\nResults saved to:")

print(
    RESULT_DIR
)

print("\nDONE")