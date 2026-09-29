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
    "--force-retrain",
    action="store_true",
)

args = parser.parse_args()

SEEDS = args.seeds
N_PERMUTATIONS = args.permutations
TOTAL_TIMESTEPS = args.timesteps
FORCE_RETRAIN = args.force_retrain


# ============================================================
# PROJECT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# BASIC CONFIG
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

TIC_ORDER = sorted(TICKERS)

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


# Previous unnormalized broad attack:
#
# Average daily L1 = 0.000528
# Capital reallocated = 0.000264 = 0.0264%
#
OLD_UNNORMALIZED_DAILY_L1 = 0.000528


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

NORMALIZED_SINGLE_ROOT = (
    ROOT
    / "results"
    / "normalized_rag_feature_importance"
)

SEED101_SOURCE_MODEL = (
    NORMALIZED_SINGLE_ROOT
    / "normalized_a2c_model.zip"
)

SCALER_PATH = (
    NORMALIZED_SINGLE_ROOT
    / "train_only_scaler.json"
)

ATTACK_PAIR_PATH = (
    ROOT
    / "results"
    / "normalized_broad_persistent_poison"
    / "ALL5_2023_opposite_p3"
    / "replayed_attack_event_pairs.csv"
)

RESULT_ROOT = (
    ROOT
    / "results"
    / "normalized_multiseed_security"
)

RESULT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


for path in [
    EVENT_PATH,
    CONFIG_PATH,
    SCALER_PATH,
    ATTACK_PAIR_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
        )


# ============================================================
# EXPERIMENT DATES
# ============================================================

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8",
) as f:

    exp_cfg = json.load(f)


COMMON_START = exp_cfg[
    "common_start"
]

TRAIN_END = exp_cfg[
    "train_end_exclusive"
]

VALIDATION_END = exp_cfg[
    "validation_end_exclusive"
]

TEST_END = exp_cfg[
    "test_end_exclusive"
]


print("=" * 75)
print("NORMALIZED MULTI-SEED SECURITY REPLICATION")
print("=" * 75)

print(
    "Seeds:",
    SEEDS,
)

print(
    "Timesteps per newly trained model:",
    TOTAL_TIMESTEPS,
)

print(
    "RAG permutations per group:",
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

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# LOAD ORIGINAL RAG EVENTS
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

events["event_date"] = pd.to_datetime(
    events["event_date"]
)


for feature in RAG_FEATURES:

    events[feature] = pd.to_numeric(
        events[feature],
        errors="coerce",
    )


if events[
    RAG_FEATURES
].isna().any().any():

    raise RuntimeError(
        "Missing RAG values in original event table."
    )


# ============================================================
# LOAD SAME 62-EVENT PAIRED ATTACK
# ============================================================

pairs = pd.read_csv(
    ATTACK_PAIR_PATH
)

pairs["ticker"] = (
    pairs["ticker"]
    .astype(str)
    .str.upper()
)

pairs["event_date"] = pd.to_datetime(
    pairs["event_date"]
)


PAIR_NUMERIC_COLUMNS = [
    "clean_impact_score",
    "clean_risk_score",
    "clean_confidence",
    "attack_impact_score",
    "attack_risk_score",
    "attack_confidence",
]


for col in PAIR_NUMERIC_COLUMNS:

    pairs[col] = pd.to_numeric(
        pairs[col],
        errors="coerce",
    )


if pairs[
    PAIR_NUMERIC_COLUMNS
].isna().any().any():

    raise RuntimeError(
        "Missing values in attack pair table."
    )


if len(pairs) != 62:

    raise RuntimeError(
        f"Expected 62 attack events, found {len(pairs)}."
    )


pairs["impact_delta"] = (
    pairs["attack_impact_score"]
    -
    pairs["clean_impact_score"]
)


pairs["impact_sign_changed"] = (
    np.sign(
        pairs["attack_impact_score"]
    )
    !=
    np.sign(
        pairs["clean_impact_score"]
    )
)


print("\n" + "=" * 75)
print("ATTACK REPLAY QA")
print("=" * 75)

print(
    "Events:",
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

paired_events = events.copy()
attack_events = events.copy()


for _, row in pairs.iterrows():

    ticker = row[
        "ticker"
    ]

    event_date = row[
        "event_date"
    ]


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


    if int(mask.sum()) != 1:

        raise RuntimeError(
            f"Expected exactly one original event: "
            f"{ticker} {event_date}"
        )


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


    cov_df = pd.DataFrame(
        rows
    )


    out = df.merge(
        cov_df,
        on="date",
        how="inner",
    )


    return (
        out.sort_values(
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
# BUILD MARKET ONCE
# ============================================================

def build_market():

    print("\n" + "=" * 75)
    print("BUILD MARKET DATA ONCE")
    print("=" * 75)


    raw = YahooDownloader(
        start_date=DOWNLOAD_START,
        end_date=DOWNLOAD_END,
        ticker_list=TIC_ORDER,
    ).fetch_data()


    if raw.empty:

        raise RuntimeError(
            "Yahoo returned no data."
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
        == len(TIC_ORDER)
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
# STRICT RAG MERGE
# ============================================================

def merge_rag(
    market,
    event_table,
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
            market[
                "tic"
            ]
            == ticker
        ].copy()


        e = event_table[
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


    if panel[
        RAG_FEATURES
    ].isna().any().any():

        raise RuntimeError(
            "Missing RAG features after merge."
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
            "Point-in-time leakage detected."
        )


    return panel


# ============================================================
# LOAD EXACT TRAIN-ONLY SCALER
# ============================================================

with open(
    SCALER_PATH,
    "r",
    encoding="utf-8",
) as f:

    scaler = json.load(f)


if scaler.get(
    "ticker_order"
) != TIC_ORDER:

    raise RuntimeError(
        "Scaler ticker order does not match."
    )


scalar_scaler = scaler[
    "scalar_scaler"
]

cov_mean = np.asarray(
    scaler[
        "covariance_mean"
    ],
    dtype=float,
)

cov_std = np.asarray(
    scaler[
        "covariance_std"
    ],
    dtype=float,
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_panel(
    df,
):

    out = df.copy()


    for feature in OBS_FEATURES:

        for ticker in TIC_ORDER:

            mask = (
                out[
                    "tic"
                ]
                == ticker
            )


            stats = (
                scalar_scaler[
                    feature
                ][
                    ticker
                ]
            )


            mean = float(
                stats[
                    "mean"
                ]
            )

            std = float(
                stats[
                    "std"
                ]
            )


            out.loc[
                mask,
                feature,
            ] = (
                (
                    out.loc[
                        mask,
                        feature,
                    ].astype(float)
                    -
                    mean
                )
                /
                std
            )


    out[
        "cov_list"
    ] = out[
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


    return out


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
            len(TIC_ORDER),

        "stock_dim":
            len(TIC_ORDER),

        "action_space":
            len(TIC_ORDER),

        "tech_indicator_list":
            OBS_FEATURES,
    }


# ============================================================
# EVALUATE
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
                or
                truncated
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
# PERFORMANCE METRICS
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
        ].copy()


    n = len(r)


    if n == 0:

        raise RuntimeError(
            "No return observations."
        )


    wealth = (
        1.0 + r
    ).cumprod()


    multiple = float(
        wealth.iloc[
            -1
        ]
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


    annual_volatility = (
        daily_std
        * math.sqrt(
            252
        )
    )


    sharpe = (
        float(
            r.mean()
        )
        /
        daily_std
        *
        math.sqrt(
            252
        )
        if daily_std > 0
        else np.nan
    )


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
# POLICY DISTANCE
# ============================================================

def weight_table(
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
                abs_diff
                .to_numpy()
                .max()
            ),
    }


# ============================================================
# RAG PERMUTATION
#
# Joint shuffle:
# impact/risk/confidence remain together as one triplet,
# but are moved to the wrong dates within the same ticker.
# ============================================================

def shuffle_features_within_ticker(
    df,
    features,
    seed,
):

    out = df.copy()

    rng = np.random.default_rng(
        seed
    )


    col_positions = [
        out.columns.get_loc(
            feature
        )
        for feature in features
    ]


    ticker_array = (
        out[
            "tic"
        ]
        .to_numpy()
    )


    for ticker in TIC_ORDER:

        positions = np.where(
            ticker_array
            ==
            ticker
        )[0]


        original = (
            out.iloc[
                positions,
                col_positions
            ]
            .to_numpy(
                copy=True
            )
        )


        permutation = rng.permutation(
            len(positions)
        )


        out.iloc[
            positions,
            col_positions
        ] = original[
            permutation
        ]


    return out


# ============================================================
# BUILD ALL PANELS ONCE
# ============================================================

market = build_market()


print("\n" + "=" * 75)
print("BUILD ORIGINAL / PAIRED / ATTACK NORMALIZED PANELS")
print("=" * 75)


original_raw = merge_rag(
    market,
    events,
)

paired_raw = merge_rag(
    market,
    paired_events,
)

attack_raw = merge_rag(
    market,
    attack_events,
)


original_norm = normalize_panel(
    original_raw
)

paired_norm = normalize_panel(
    paired_raw
)

attack_norm = normalize_panel(
    attack_raw
)


train_df = data_split(
    original_norm,
    COMMON_START,
    TRAIN_END,
)

validation_df = data_split(
    original_norm,
    TRAIN_END,
    VALIDATION_END,
)

original_test = data_split(
    original_norm,
    VALIDATION_END,
    TEST_END,
)

paired_test = data_split(
    paired_norm,
    VALIDATION_END,
    TEST_END,
)

attack_test = data_split(
    attack_norm,
    VALIDATION_END,
    TEST_END,
)


print(
    "Train dates:",
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
    original_test[
        "date"
    ].nunique(),
)


# ============================================================
# ATTACK ROW QA
# ============================================================

paired_rag = (
    paired_test[
        RAG_FEATURES
    ]
    .astype(float)
    .to_numpy()
)

attack_rag = (
    attack_test[
        RAG_FEATURES
    ]
    .astype(float)
    .to_numpy()
)


changed_rows = int(
    np.any(
        np.abs(
            paired_rag
            -
            attack_rag
        )
        >
        1e-12,
        axis=1,
    ).sum()
)


print(
    "Changed normalized RAG rows:",
    changed_rows,
)


if changed_rows != 1168:

    print(
        "WARNING: previous run had 1168 changed rows."
    )


# ============================================================
# MODEL TRAIN / REUSE
# ============================================================

def get_or_train_model(
    seed,
):

    seed_dir = (
        RESULT_ROOT
        /
        f"seed_{seed}"
    )

    seed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    model_basename = (
        seed_dir
        /
        "normalized_a2c_model"
    )

    model_zip = Path(
        str(
            model_basename
        )
        +
        ".zip"
    )


    # --------------------------------------------------------
    # Reuse a model already produced by this multi-seed script
    # --------------------------------------------------------

    if (
        model_zip.exists()
        and
        not FORCE_RETRAIN
    ):

        print(
            "Reusing existing model:",
            model_zip,
        )

        return (
            A2C.load(
                str(
                    model_zip
                )
            ),
            "multiseed_cache",
            seed_dir,
        )


    # --------------------------------------------------------
    # Exact previous normalized seed-101 model
    # --------------------------------------------------------

    if (
        seed == 101
        and
        SEED101_SOURCE_MODEL.exists()
        and
        not FORCE_RETRAIN
    ):

        shutil.copy2(
            SEED101_SOURCE_MODEL,
            model_zip,
        )


        print(
            "Reusing exact previous seed-101 model:",
            SEED101_SOURCE_MODEL,
        )


        return (
            A2C.load(
                str(
                    model_zip
                )
            ),
            "previous_exact_seed101",
            seed_dir,
        )


    # --------------------------------------------------------
    # Train new model
    # --------------------------------------------------------

    print(
        "Training new normalized A2C seed",
        seed,
    )


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


    # Match the CPU parameters used in the successful
    # normalized seed-101 run.
    a2c_params = {
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
        model_kwargs=a2c_params,
        seed=seed,
    )


    model = agent.train_model(
        model=model,
        tb_log_name=(
            f"normalized_a2c_seed_{seed}"
        ),
        total_timesteps=TOTAL_TIMESTEPS,
    )


    model.save(
        str(
            model_basename
        )
    )


    return (
        model,
        "new_training",
        seed_dir,
    )


# ============================================================
# CLEAN RAG RELIANCE
# ============================================================

RAG_PERMUTATION_GROUPS = {
    "all_rag_joint":
        [
            "impact_score",
            "risk_score",
            "confidence",
        ],

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
}


def measure_rag_reliance(
    seed,
    model,
    normal_weights,
):

    rows = []


    for group_index, (
        group_name,
        features,
    ) in enumerate(
        RAG_PERMUTATION_GROUPS.items()
    ):

        for repetition in range(
            N_PERMUTATIONS
        ):

            permutation_seed = (
                seed
                * 100_000
                +
                group_index
                * 1000
                +
                repetition
            )


            modified_test = (
                shuffle_features_within_ticker(
                    original_test,
                    features,
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
                normal_weights,
                modified_weights,
            )


            perf = performance_metrics(
                modified_returns
            )


            rows.append(
                {
                    "training_seed":
                        seed,

                    "feature_group":
                        group_name,

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
# RUN SEEDS
# ============================================================

seed_summary_rows = []

all_permutation_rows = []


for seed_index, seed in enumerate(
    SEEDS,
    start=1,
):

    print("\n")
    print("#" * 75)
    print(
        f"SEED {seed} "
        f"({seed_index}/{len(SEEDS)})"
    )
    print("#" * 75)


    model, model_source, seed_dir = (
        get_or_train_model(
            seed
        )
    )


    # ========================================================
    # Validation
    # ========================================================

    print(
        "Evaluating validation..."
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


    # ========================================================
    # Original clean
    # ========================================================

    print(
        "Evaluating original clean..."
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


    # ========================================================
    # Paired clean
    # ========================================================

    print(
        "Evaluating paired clean..."
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


    # ========================================================
    # Attack
    # ========================================================

    print(
        "Evaluating attack..."
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


    # ========================================================
    # Distances
    # ========================================================

    regen_shift = policy_distance(
        original_weights,
        paired_weights,
    )


    attack_shift = policy_distance(
        paired_weights,
        attack_weights,
    )


    # ========================================================
    # RAG reliance
    # ========================================================

    print(
        "Running RAG permutation reliance..."
    )


    permutation_df = (
        measure_rag_reliance(
            seed,
            model,
            original_weights,
        )
    )


    all_permutation_rows.append(
        permutation_df
    )


    permutation_df.to_csv(
        seed_dir
        /
        "rag_permutation_runs.csv",
        index=False,
    )


    reliance_means = (
        permutation_df.groupby(
            "feature_group"
        )[
            "average_reallocated_weight"
        ]
        .mean()
        .to_dict()
    )


    joint_rag_reliance = float(
        reliance_means[
            "all_rag_joint"
        ]
    )


    impact_reliance = float(
        reliance_means[
            "impact_score"
        ]
    )


    risk_reliance = float(
        reliance_means[
            "risk_score"
        ]
    )


    confidence_reliance = float(
        reliance_means[
            "confidence"
        ]
    )


    mean_individual_rag_reliance = float(
        np.mean(
            [
                impact_reliance,
                risk_reliance,
                confidence_reliance,
            ]
        )
    )


    attack_delta_multiple = (
        attack_perf[
            "final_multiple"
        ]
        -
        paired_perf[
            "final_multiple"
        ]
    )


    attack_delta_sharpe = (
        attack_perf[
            "sharpe"
        ]
        -
        paired_perf[
            "sharpe"
        ]
    )


    attack_delta_mdd = (
        attack_perf[
            "max_drawdown"
        ]
        -
        paired_perf[
            "max_drawdown"
        ]
    )


    attack_delta_vol = (
        attack_perf[
            "annual_volatility"
        ]
        -
        paired_perf[
            "annual_volatility"
        ]
    )


    attack_delta_return = (
        attack_perf[
            "annual_return"
        ]
        -
        paired_perf[
            "annual_return"
        ]
    )


    attack_vs_old_ratio = (
        attack_shift[
            "average_daily_l1"
        ]
        /
        OLD_UNNORMALIZED_DAILY_L1
    )


    attack_to_regen_ratio = (
        attack_shift[
            "average_reallocated_weight"
        ]
        /
        regen_shift[
            "average_reallocated_weight"
        ]
        if
        regen_shift[
            "average_reallocated_weight"
        ]
        > 0
        else np.nan
    )


    attack_to_clean_rag_reliance = (
        attack_shift[
            "average_reallocated_weight"
        ]
        /
        joint_rag_reliance
        if
        joint_rag_reliance > 0
        else np.nan
    )


    seed_row = {
        "seed":
            seed,

        "model_source":
            model_source,

        # validation
        "validation_multiple":
            validation_perf[
                "final_multiple"
            ],

        "validation_sharpe":
            validation_perf[
                "sharpe"
            ],

        # original clean
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

        # paired clean
        "paired_clean_multiple":
            paired_perf[
                "final_multiple"
            ],

        "paired_clean_sharpe":
            paired_perf[
                "sharpe"
            ],

        # clean RAG reliance
        "rag_joint_reallocated":
            joint_rag_reliance,

        "rag_individual_mean_reallocated":
            mean_individual_rag_reliance,

        "impact_reallocated":
            impact_reliance,

        "risk_reallocated":
            risk_reliance,

        "confidence_reallocated":
            confidence_reliance,

        # regeneration noise floor
        "regen_daily_l1":
            regen_shift[
                "average_daily_l1"
            ],

        "regen_reallocated":
            regen_shift[
                "average_reallocated_weight"
            ],

        # attack shift
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

        "attack_max_single_asset_diff":
            attack_shift[
                "maximum_single_asset_diff"
            ],

        # attack performance
        "attack_delta_multiple":
            attack_delta_multiple,

        "attack_dollar_difference":
            INITIAL_AMOUNT
            *
            attack_delta_multiple,

        "attack_delta_annual_return":
            attack_delta_return,

        "attack_delta_volatility":
            attack_delta_vol,

        "attack_delta_sharpe":
            attack_delta_sharpe,

        "attack_delta_mdd":
            attack_delta_mdd,

        # useful ratios
        "attack_to_regen_ratio":
            attack_to_regen_ratio,

        "attack_to_clean_rag_reliance":
            attack_to_clean_rag_reliance,

        "attack_l1_amplification_vs_old_unnormalized":
            attack_vs_old_ratio,
    }


    seed_summary_rows.append(
        seed_row
    )


    # ========================================================
    # Save seed-level artifacts
    # ========================================================

    original_returns.to_csv(
        seed_dir
        /
        "original_clean_returns.csv",
        index=False,
    )

    original_weights.to_csv(
        seed_dir
        /
        "original_clean_weights.csv",
    )

    paired_returns.to_csv(
        seed_dir
        /
        "paired_clean_returns.csv",
        index=False,
    )

    paired_weights.to_csv(
        seed_dir
        /
        "paired_clean_weights.csv",
    )

    attack_returns.to_csv(
        seed_dir
        /
        "attack_returns.csv",
        index=False,
    )

    attack_weights.to_csv(
        seed_dir
        /
        "attack_weights.csv",
    )


    with open(
        seed_dir
        /
        "seed_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            seed_row,
            f,
            indent=2,
        )


    print("\nSeed result:")

    print(
        "  Clean multiple:",
        f"{original_perf['final_multiple']:.6f}",
    )

    print(
        "  Clean Sharpe:",
        f"{original_perf['sharpe']:.4f}",
    )

    print(
        "  Joint RAG reliance:",
        f"{joint_rag_reliance:.4%}",
    )

    print(
        "  Attack reallocated:",
        f"{attack_shift['average_reallocated_weight']:.4%}",
    )

    print(
        "  Attack Δ multiple:",
        f"{attack_delta_multiple:+.6f}",
    )

    print(
        "  Attack Δ Sharpe:",
        f"{attack_delta_sharpe:+.6f}",
    )

    print(
        "  Attack/regen ratio:",
        f"{attack_to_regen_ratio:.2f}x",
    )

    print(
        "  Amplification vs old unnormalized:",
        f"{attack_vs_old_ratio:.2f}x",
    )


# ============================================================
# SAVE ALL RAW PERMUTATIONS
# ============================================================

all_permutations = pd.concat(
    all_permutation_rows,
    ignore_index=True,
)


all_permutations.to_csv(
    RESULT_ROOT
    /
    "all_seed_rag_permutation_runs.csv",
    index=False,
)


# ============================================================
# MULTI-SEED SUMMARY
# ============================================================

summary = pd.DataFrame(
    seed_summary_rows
)


summary.to_csv(
    RESULT_ROOT
    /
    "multiseed_summary.csv",
    index=False,
)


# ============================================================
# AGGREGATE MEAN / STD
# ============================================================

AGG_COLUMNS = [
    "clean_multiple",
    "clean_sharpe",
    "rag_joint_reallocated",
    "rag_individual_mean_reallocated",
    "impact_reallocated",
    "risk_reallocated",
    "confidence_reallocated",
    "regen_reallocated",
    "attack_daily_l1",
    "attack_reallocated",
    "attack_max_reallocated",
    "attack_delta_multiple",
    "attack_dollar_difference",
    "attack_delta_annual_return",
    "attack_delta_volatility",
    "attack_delta_sharpe",
    "attack_delta_mdd",
    "attack_to_regen_ratio",
    "attack_l1_amplification_vs_old_unnormalized",
]


aggregate_rows = []


for col in AGG_COLUMNS:

    values = (
        summary[
            col
        ]
        .astype(float)
    )


    aggregate_rows.append(
        {
            "metric":
                col,

            "mean":
                float(
                    values.mean()
                ),

            "std":
                float(
                    values.std(
                        ddof=1
                    )
                )
                if len(values) > 1
                else 0.0,

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


aggregate = pd.DataFrame(
    aggregate_rows
)


aggregate.to_csv(
    RESULT_ROOT
    /
    "multiseed_aggregate.csv",
    index=False,
)


# ============================================================
# CORRELATION:
#
# Does learned clean RAG reliance predict how strongly
# poisoning propagates into the portfolio?
#
# n=5 -> exploratory, not inferential proof.
# ============================================================

x = (
    summary[
        "rag_joint_reallocated"
    ]
    .astype(float)
    .to_numpy()
)

y = (
    summary[
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


correlation_result = {
    "n_seeds":
        len(summary),

    "x":
        "clean_joint_RAG_reallocated_weight",

    "y":
        "poisoning_attack_reallocated_weight",

    "pearson":
        pearson,

    "spearman":
        spearman,

    "note":
        (
            "Exploratory correlation only; "
            "small number of training seeds."
        ),
}


with open(
    RESULT_ROOT
    /
    "rag_reliance_attack_correlation.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        correlation_result,
        f,
        indent=2,
    )


# ============================================================
# COUNT DIRECTION OF FINANCIAL EFFECT
# ============================================================

n_negative_multiple = int(
    (
        summary[
            "attack_delta_multiple"
        ]
        <
        0
    ).sum()
)

n_negative_sharpe = int(
    (
        summary[
            "attack_delta_sharpe"
        ]
        <
        0
    ).sum()
)

n_worse_mdd = int(
    (
        summary[
            "attack_delta_mdd"
        ]
        <
        0
    ).sum()
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 75)
print("MULTI-SEED NORMALIZED SECURITY RESULT")
print("=" * 75)


DISPLAY = summary[
    [
        "seed",
        "clean_multiple",
        "clean_sharpe",
        "rag_joint_reallocated",
        "attack_reallocated",
        "attack_delta_multiple",
        "attack_delta_sharpe",
        "attack_to_regen_ratio",
        "attack_l1_amplification_vs_old_unnormalized",
    ]
].copy()


print(
    DISPLAY.to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("CLEAN RAG RELIANCE ACROSS SEEDS")
print("=" * 75)


print(
    "Joint RAG reliance:",
    f"{summary['rag_joint_reallocated'].mean():.4%}",
    "±",
    f"{summary['rag_joint_reallocated'].std(ddof=1):.4%}",
)


print(
    "Impact reliance:",
    f"{summary['impact_reallocated'].mean():.4%}",
    "±",
    f"{summary['impact_reallocated'].std(ddof=1):.4%}",
)


print(
    "Risk reliance:",
    f"{summary['risk_reallocated'].mean():.4%}",
    "±",
    f"{summary['risk_reallocated'].std(ddof=1):.4%}",
)


print(
    "Confidence reliance:",
    f"{summary['confidence_reallocated'].mean():.4%}",
    "±",
    f"{summary['confidence_reallocated'].std(ddof=1):.4%}",
)


print("\n" + "=" * 75)
print("POISONING PROPAGATION ACROSS SEEDS")
print("=" * 75)


print(
    "Attack average capital reallocated:",
    f"{summary['attack_reallocated'].mean():.4%}",
    "±",
    f"{summary['attack_reallocated'].std(ddof=1):.4%}",
)


print(
    "Attack average daily L1:",
    f"{summary['attack_daily_l1'].mean():.6f}",
    "±",
    f"{summary['attack_daily_l1'].std(ddof=1):.6f}",
)


print(
    "Mean amplification vs old unnormalized:",
    f"{summary['attack_l1_amplification_vs_old_unnormalized'].mean():.2f}x",
    "±",
    f"{summary['attack_l1_amplification_vs_old_unnormalized'].std(ddof=1):.2f}x",
)


print(
    "Mean attack / paired-regeneration ratio:",
    f"{summary['attack_to_regen_ratio'].mean():.2f}x",
)


print("\n" + "=" * 75)
print("FINANCIAL EFFECT ACROSS SEEDS")
print("=" * 75)


print(
    "Mean Δ multiple:",
    f"{summary['attack_delta_multiple'].mean():+.6f}",
    "±",
    f"{summary['attack_delta_multiple'].std(ddof=1):.6f}",
)


print(
    "Mean $ difference per $1M:",
    f"${summary['attack_dollar_difference'].mean():+,.2f}",
    "±",
    f"${summary['attack_dollar_difference'].std(ddof=1):,.2f}",
)


print(
    "Mean Δ Sharpe:",
    f"{summary['attack_delta_sharpe'].mean():+.6f}",
    "±",
    f"{summary['attack_delta_sharpe'].std(ddof=1):.6f}",
)


print(
    "Seeds with lower final multiple:",
    f"{n_negative_multiple}/{len(summary)}",
)


print(
    "Seeds with lower Sharpe:",
    f"{n_negative_sharpe}/{len(summary)}",
)


print(
    "Seeds with worse MDD:",
    f"{n_worse_mdd}/{len(summary)}",
)


print("\n" + "=" * 75)
print("RAG RELIANCE -> ATTACK PROPAGATION")
print("=" * 75)


print(
    "Pearson correlation:",
    f"{pearson:+.4f}",
)


print(
    "Spearman correlation:",
    f"{spearman:+.4f}",
)


print(
    "NOTE: n =",
    len(summary),
    "seeds, so treat correlation as exploratory."
)


print("\nResults saved to:")

print(
    RESULT_ROOT
)

print("\nDONE")