import argparse
import contextlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import A2C

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
    "--rounds",
    type=int,
    default=20,
)

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

args = parser.parse_args()

N_ROUNDS = args.rounds
SEEDS = args.seeds


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

OBS_FEATURES = (
    TECH_FEATURES
    + RAG_FEATURES
)

DOWNLOAD_START = "2017-01-01"
DOWNLOAD_END = "2024-01-01"

LOOKBACK = 252

INITIAL_AMOUNT = 1_000_000


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

BENIGN_PATH = (
    ROOT
    / "results"
    / "benign_rag_controls"
    / "benign_event_signals.csv"
)

PAIR_PATH = (
    ROOT
    / "results"
    / "normalized_broad_persistent_poison"
    / "ALL5_2023_opposite_p3"
    / "replayed_attack_event_pairs.csv"
)

CONFIG_PATH = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
    / "experiment_config.json"
)

SCALER_PATH = (
    ROOT
    / "results"
    / "normalized_rag_feature_importance"
    / "train_only_scaler.json"
)

MODEL_ROOT = (
    ROOT
    / "results"
    / "normalization_causal_control"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "benign_vs_poison"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for path in [
    EVENT_PATH,
    BENIGN_PATH,
    PAIR_PATH,
    CONFIG_PATH,
    SCALER_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Missing required file:\n{path}"
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

TEST_START = cfg[
    "validation_end_exclusive"
]

TEST_END = cfg[
    "test_end_exclusive"
]


# ============================================================
# CLEAN EVENT TABLE
# ============================================================

events = pd.read_csv(
    EVENT_PATH
)

events = events[
    events[
        "status"
    ]
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
).dt.normalize()


for feature in RAG_FEATURES:

    events[
        feature
    ] = pd.to_numeric(
        events[
            feature
        ],
        errors="coerce",
    )


# ============================================================
# ATTACK / PAIRED CLEAN
# ============================================================

pairs = pd.read_csv(
    PAIR_PATH
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
).dt.normalize()


if len(
    pairs
) != 62:

    raise RuntimeError(
        f"Expected 62 attack events, "
        f"found {len(pairs)}."
    )


# ============================================================
# BENIGN SIGNALS
# ============================================================

benign = pd.read_csv(
    BENIGN_PATH
)

benign[
    "ticker"
] = (
    benign[
        "ticker"
    ]
    .astype(str)
    .str.upper()
)

benign[
    "event_date"
] = pd.to_datetime(
    benign[
        "event_date"
    ]
).dt.normalize()


benign = benign[
    benign[
        "round"
    ]
    <
    N_ROUNDS
].copy()


expected_benign_rows = (
    N_ROUNDS
    *
    62
)


if len(
    benign
) != expected_benign_rows:

    raise RuntimeError(
        f"Expected {expected_benign_rows} benign rows, "
        f"found {len(benign)}."
    )


# ============================================================
# BUILD EVENT TABLE HELPER
# ============================================================

def replace_target_signals(
    base_events,
    replacements,
    impact_col,
    risk_col,
    confidence_col,
):

    output = base_events.copy()


    for _, row in replacements.iterrows():

        mask = (
            (
                output[
                    "ticker"
                ]
                ==
                row[
                    "ticker"
                ]
            )
            &
            (
                output[
                    "event_date"
                ]
                ==
                row[
                    "event_date"
                ]
            )
        )


        if int(
            mask.sum()
        ) != 1:

            raise RuntimeError(
                f"Expected one target event: "
                f"{row['ticker']} "
                f"{row['event_date']}"
            )


        output.loc[
            mask,
            "impact_score",
        ] = float(
            row[
                impact_col
            ]
        )


        output.loc[
            mask,
            "risk_score",
        ] = float(
            row[
                risk_col
            ]
        )


        output.loc[
            mask,
            "confidence",
        ] = float(
            row[
                confidence_col
            ]
        )


    return output


# ============================================================
# CANONICAL PAIRED CLEAN
# ============================================================

paired_events = replace_target_signals(
    events,
    pairs,
    "clean_impact_score",
    "clean_risk_score",
    "clean_confidence",
)


# ============================================================
# ATTACK
# ============================================================

attack_events = replace_target_signals(
    events,
    pairs,
    "attack_impact_score",
    "attack_risk_score",
    "attack_confidence",
)


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
        len(
            dates
        ),
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


        rows.append(
            {
                "date":
                    dates[
                        i
                    ],

                "cov_list":
                    covariance,
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
# MARKET
# ============================================================

def build_market():

    print("=" * 75)
    print("BUILD MARKET")
    print("=" * 75)


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


    good_dates = counts[
        counts
        ==
        len(
            TIC_ORDER
        )
    ].index


    market = market[
        market[
            "date"
        ].isin(
            good_dates
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


    return market


# ============================================================
# STRICT MERGE
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

        m = market[
            market[
                "tic"
            ]
            ==
            ticker
        ].copy()


        e = event_table[
            event_table[
                "ticker"
            ]
            ==
            ticker
        ][
            [
                "source_filing_date",
                "impact_score",
                "risk_score",
                "confidence",
            ]
        ].copy()


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
            "Missing RAG after merge."
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
    ):

        raise RuntimeError(
            "Point-in-time leakage."
        )


    return panel


# ============================================================
# NORMALIZATION
# ============================================================

with open(
    SCALER_PATH,
    "r",
    encoding="utf-8",
) as f:

    scaler = json.load(
        f
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


def normalize_panel(
    df,
):

    output = df.copy()


    for feature in OBS_FEATURES:

        for ticker in TIC_ORDER:

            mask = (
                output[
                    "tic"
                ]
                ==
                ticker
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
                    ].astype(
                        float
                    )
                    -
                    mean
                )
                /
                std
            )


    output[
        "cov_list"
    ] = output[
        "cov_list"
    ].apply(
        lambda x:
            (
                np.asarray(
                    x,
                    dtype=float,
                )
                -
                cov_mean
            )
            /
            cov_std
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
# EVALUATE
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
# PERFORMANCE
# ============================================================

def performance(
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


    n = len(
        r
    )


    wealth = (
        1.0
        +
        r
    ).cumprod()


    multiple = float(
        wealth.iloc[
            -1
        ]
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


    return {
        "multiple":
            multiple,

        "sharpe":
            sharpe,
    }


# ============================================================
# POLICY DISTANCE
# ============================================================

def policy_distance(
    reference,
    candidate,
):

    ref = (
        reference[
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


    cand = (
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
        len(ref),
        len(cand),
    )


    diff = (
        cand.iloc[
            :n
        ]
        -
        ref.iloc[
            :n
        ]
    ).abs()


    daily_l1 = (
        diff.sum(
            axis=1
        )
    )


    return {
        "daily_l1":
            float(
                daily_l1.mean()
            ),

        "reallocated":
            float(
                daily_l1.mean()
                /
                2.0
            ),

        "max_reallocated":
            float(
                daily_l1.max()
                /
                2.0
            ),
    }


# ============================================================
# BUILD MARKET + BASE TESTS
# ============================================================

market = build_market()


paired_raw_panel = merge_rag(
    market,
    paired_events,
)


attack_raw_panel = merge_rag(
    market,
    attack_events,
)


paired_norm_panel = normalize_panel(
    paired_raw_panel
)


attack_norm_panel = normalize_panel(
    attack_raw_panel
)


paired_raw_test = data_split(
    paired_raw_panel,
    TEST_START,
    TEST_END,
)


attack_raw_test = data_split(
    attack_raw_panel,
    TEST_START,
    TEST_END,
)


paired_norm_test = data_split(
    paired_norm_panel,
    TEST_START,
    TEST_END,
)


attack_norm_test = data_split(
    attack_norm_panel,
    TEST_START,
    TEST_END,
)


# ============================================================
# PREBUILD BENIGN TEST PANELS
# ============================================================

print("\n" + "=" * 75)
print("BUILD BENIGN TEST PANELS")
print("=" * 75)


benign_raw_tests = {}

benign_norm_tests = {}


for round_id in range(
    N_ROUNDS
):

    replacements = benign[
        benign[
            "round"
        ]
        ==
        round_id
    ].copy()


    if len(
        replacements
    ) != 62:

        raise RuntimeError(
            f"Round {round_id} has "
            f"{len(replacements)} events."
        )


    benign_events = replace_target_signals(
        events,
        replacements,
        "benign_impact_score",
        "benign_risk_score",
        "benign_confidence",
    )


    benign_raw_panel = merge_rag(
        market,
        benign_events,
    )


    benign_norm_panel = normalize_panel(
        benign_raw_panel
    )


    benign_raw_tests[
        round_id
    ] = data_split(
        benign_raw_panel,
        TEST_START,
        TEST_END,
    )


    benign_norm_tests[
        round_id
    ] = data_split(
        benign_norm_panel,
        TEST_START,
        TEST_END,
    )


    print(
        f"Round {round_id + 1:02d}/"
        f"{N_ROUNDS} ready"
    )


# ============================================================
# MODELS + RUNS
# ============================================================

run_rows = []

attack_rows = []


for seed in SEEDS:

    print("\n" + "#" * 75)
    print(
        "SEED",
        seed
    )
    print("#" * 75)


    model_paths = {
        "raw":
            (
                MODEL_ROOT
                /
                f"seed_{seed}"
                /
                "raw_a2c_model.zip"
            ),

        "normalized":
            (
                MODEL_ROOT
                /
                f"seed_{seed}"
                /
                "normalized_a2c_model.zip"
            ),
    }


    for condition in [
        "raw",
        "normalized",
    ]:

        model_path = (
            model_paths[
                condition
            ]
        )


        if not model_path.exists():

            raise FileNotFoundError(
                model_path
            )


        model = A2C.load(
            str(
                model_path
            )
        )


        if condition == "raw":

            baseline_test = (
                paired_raw_test
            )

            attack_test = (
                attack_raw_test
            )

            benign_tests = (
                benign_raw_tests
            )

        else:

            baseline_test = (
                paired_norm_test
            )

            attack_test = (
                attack_norm_test
            )

            benign_tests = (
                benign_norm_tests
            )


        # ----------------------------------------------------
        # Canonical paired-clean reference
        # ----------------------------------------------------

        (
            baseline_returns,
            baseline_weights,
        ) = evaluate(
            model,
            baseline_test,
        )


        baseline_perf = performance(
            baseline_returns
        )


        # ----------------------------------------------------
        # Attack
        # ----------------------------------------------------

        (
            attack_returns,
            attack_weights,
        ) = evaluate(
            model,
            attack_test,
        )


        attack_shift = policy_distance(
            baseline_weights,
            attack_weights,
        )


        attack_perf = performance(
            attack_returns
        )


        attack_rows.append(
            {
                "seed":
                    seed,

                "condition":
                    condition,

                "attack_daily_l1":
                    attack_shift[
                        "daily_l1"
                    ],

                "attack_reallocated":
                    attack_shift[
                        "reallocated"
                    ],

                "attack_max_reallocated":
                    attack_shift[
                        "max_reallocated"
                    ],

                "attack_delta_multiple":
                    (
                        attack_perf[
                            "multiple"
                        ]
                        -
                        baseline_perf[
                            "multiple"
                        ]
                    ),

                "attack_delta_sharpe":
                    (
                        attack_perf[
                            "sharpe"
                        ]
                        -
                        baseline_perf[
                            "sharpe"
                        ]
                    ),
            }
        )


        # ----------------------------------------------------
        # Benign controls
        # ----------------------------------------------------

        for round_id in range(
            N_ROUNDS
        ):

            (
                benign_returns,
                benign_weights,
            ) = evaluate(
                model,
                benign_tests[
                    round_id
                ],
            )


            benign_shift = policy_distance(
                baseline_weights,
                benign_weights,
            )


            benign_perf = performance(
                benign_returns
            )


            upstream = benign[
                benign[
                    "round"
                ]
                ==
                round_id
            ]


            run_rows.append(
                {
                    "seed":
                        seed,

                    "condition":
                        condition,

                    "round":
                        round_id,

                    "benign_daily_l1":
                        benign_shift[
                            "daily_l1"
                        ],

                    "benign_reallocated":
                        benign_shift[
                            "reallocated"
                        ],

                    "benign_max_reallocated":
                        benign_shift[
                            "max_reallocated"
                        ],

                    "benign_delta_multiple":
                        (
                            benign_perf[
                                "multiple"
                            ]
                            -
                            baseline_perf[
                                "multiple"
                            ]
                        ),

                    "benign_delta_sharpe":
                        (
                            benign_perf[
                                "sharpe"
                            ]
                            -
                            baseline_perf[
                                "sharpe"
                            ]
                        ),

                    "benign_mean_abs_impact_delta":
                        float(
                            upstream[
                                "abs_impact_delta_vs_canonical"
                            ].mean()
                        ),

                    "benign_sign_changes":
                        int(
                            upstream[
                                "impact_sign_changed"
                            ].sum()
                        ),
                }
            )


            print(
                f"{condition:10s} "
                f"round {round_id + 1:02d}/"
                f"{N_ROUNDS} "
                f"reallocated="
                f"{benign_shift['reallocated']:.4%}"
            )


# ============================================================
# TABLES
# ============================================================

runs = pd.DataFrame(
    run_rows
)


attacks = pd.DataFrame(
    attack_rows
)


runs.to_csv(
    RESULT_DIR
    / "benign_control_runs.csv",
    index=False,
)


attacks.to_csv(
    RESULT_DIR
    / "attack_reference.csv",
    index=False,
)


# ============================================================
# PER-SEED COMPARISON
# ============================================================

seed_summary_rows = []


for _, attack_row in attacks.iterrows():

    seed = int(
        attack_row[
            "seed"
        ]
    )

    condition = (
        attack_row[
            "condition"
        ]
    )


    temp = runs[
        (
            runs[
                "seed"
            ]
            ==
            seed
        )
        &
        (
            runs[
                "condition"
            ]
            ==
            condition
        )
    ]


    attack_value = float(
        attack_row[
            "attack_reallocated"
        ]
    )


    benign_values = (
        temp[
            "benign_reallocated"
        ]
        .astype(float)
        .to_numpy()
    )


    benign_mean = float(
        np.mean(
            benign_values
        )
    )


    benign_std = float(
        np.std(
            benign_values,
            ddof=1,
        )
    )


    exceed_count = int(
        np.sum(
            benign_values
            >=
            attack_value
        )
    )


    empirical_p = (
        1
        +
        exceed_count
    ) / (
        len(
            benign_values
        )
        +
        1
    )


    if benign_std > 0:

        attack_z = (
            attack_value
            -
            benign_mean
        ) / benign_std

    else:

        attack_z = np.nan


    seed_summary_rows.append(
        {
            "seed":
                seed,

            "condition":
                condition,

            "attack_reallocated":
                attack_value,

            "benign_mean":
                benign_mean,

            "benign_std":
                benign_std,

            "benign_median":
                float(
                    np.median(
                        benign_values
                    )
                ),

            "benign_q95":
                float(
                    np.quantile(
                        benign_values,
                        0.95,
                    )
                ),

            "attack_over_benign_mean":
                (
                    attack_value
                    /
                    benign_mean
                    if benign_mean > 0
                    else np.nan
                ),

            "attack_z_vs_benign":
                attack_z,

            "benign_rounds_ge_attack":
                exceed_count,

            "empirical_tail_p":
                empirical_p,
        }
    )


seed_summary = pd.DataFrame(
    seed_summary_rows
)


seed_summary.to_csv(
    RESULT_DIR
    / "seed_level_benign_vs_attack.csv",
    index=False,
)


# ============================================================
# ROUND-LEVEL ACROSS-SEED NULL DISTRIBUTION
#
# Each benign round produces one cross-seed mean.
# Compare attack cross-seed mean against this distribution.
# ============================================================

condition_summary_rows = []


for condition in [
    "raw",
    "normalized",
]:

    attack_condition = attacks[
        attacks[
            "condition"
        ]
        ==
        condition
    ]


    attack_mean = float(
        attack_condition[
            "attack_reallocated"
        ].mean()
    )


    round_means = (
        runs[
            runs[
                "condition"
            ]
            ==
            condition
        ]
        .groupby(
            "round"
        )[
            "benign_reallocated"
        ]
        .mean()
        .sort_index()
    )


    benign_mean = float(
        round_means.mean()
    )


    benign_std = float(
        round_means.std(
            ddof=1
        )
    )


    benign_q95 = float(
        round_means.quantile(
            0.95
        )
    )


    exceed_count = int(
        (
            round_means
            >=
            attack_mean
        ).sum()
    )


    empirical_p = (
        1
        +
        exceed_count
    ) / (
        len(
            round_means
        )
        +
        1
    )


    attack_z = (
        (
            attack_mean
            -
            benign_mean
        )
        /
        benign_std
        if benign_std > 0
        else np.nan
    )


    condition_summary_rows.append(
        {
            "condition":
                condition,

            "n_seeds":
                len(
                    attack_condition
                ),

            "n_benign_rounds":
                len(
                    round_means
                ),

            "attack_mean_reallocated":
                attack_mean,

            "benign_round_mean":
                benign_mean,

            "benign_round_std":
                benign_std,

            "benign_round_q95":
                benign_q95,

            "attack_over_benign":
                (
                    attack_mean
                    /
                    benign_mean
                    if benign_mean > 0
                    else np.nan
                ),

            "attack_z_vs_benign":
                attack_z,

            "benign_rounds_ge_attack":
                exceed_count,

            "empirical_tail_p":
                empirical_p,
        }
    )


condition_summary = pd.DataFrame(
    condition_summary_rows
)


condition_summary.to_csv(
    RESULT_DIR
    / "condition_benign_vs_attack_summary.csv",
    index=False,
)


# ============================================================
# UPSTREAM BENIGN VS ATTACK
# ============================================================

attack_mean_abs_impact = float(
    (
        pairs[
            "attack_impact_score"
        ]
        -
        pairs[
            "clean_impact_score"
        ]
    )
    .abs()
    .mean()
)


attack_sign_changes = int(
    (
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
    ).sum()
)


benign_round_upstream = (
    benign.groupby(
        "round"
    )
    .agg(
        mean_abs_impact_delta=(
            "abs_impact_delta_vs_canonical",
            "mean",
        ),

        sign_changes=(
            "impact_sign_changed",
            "sum",
        ),
    )
    .reset_index()
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 75)
print("UPSTREAM BENIGN VS ATTACK")
print("=" * 75)

print(
    "Attack mean |impact delta|:",
    f"{attack_mean_abs_impact:.4f}",
)

print(
    "Benign mean |impact delta|:",
    f"{benign_round_upstream['mean_abs_impact_delta'].mean():.4f}",
    "±",
    f"{benign_round_upstream['mean_abs_impact_delta'].std(ddof=1):.4f}",
)

print(
    "Attack sign changes:",
    f"{attack_sign_changes}/62",
)

print(
    "Benign sign changes per round:",
    f"{benign_round_upstream['sign_changes'].mean():.2f}",
    "±",
    f"{benign_round_upstream['sign_changes'].std(ddof=1):.2f}",
)


print("\n" + "=" * 75)
print("SEED-LEVEL BENIGN VS ATTACK")
print("=" * 75)

print(
    seed_summary[
        [
            "seed",
            "condition",
            "attack_reallocated",
            "benign_mean",
            "benign_q95",
            "attack_over_benign_mean",
            "benign_rounds_ge_attack",
            "empirical_tail_p",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("CROSS-SEED BENIGN NULL VS ATTACK")
print("=" * 75)

print(
    condition_summary.to_string(
        index=False
    )
)


print("\nInterpretation rule:")

print(
    "If attack_mean_reallocated > benign_round_q95 "
    "and benign_rounds_ge_attack == 0, "
    "the malicious attack exceeds the observed benign-control distribution."
)


print(
    "\nWith 20 benign rounds, the smallest corrected empirical tail "
    "probability is 1/(20+1) = 0.0476."
)


print(
    "\nResults saved to:"
)

print(
    RESULT_DIR
)

print("\nDONE")