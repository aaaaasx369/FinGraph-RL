import contextlib
import io
import itertools
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

SEEDS = [
    101,
    202,
    303,
    404,
    505,
]

DOWNLOAD_START = "2017-01-01"
DOWNLOAD_END = "2024-01-01"

LOOKBACK = 252
INITIAL_AMOUNT = 1_000_000

# 20 benign rounds -> 4 disjoint ensembles of 5
ENSEMBLE_GROUPS = [
    [0, 1, 2, 3, 4],
    [5, 6, 7, 8, 9],
    [10, 11, 12, 13, 14],
    [15, 16, 17, 18, 19],
]


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

PREVIOUS_BENIGN_RUNS = (
    ROOT
    / "results"
    / "benign_vs_poison"
    / "benign_control_runs.csv"
)

PREVIOUS_ATTACK_REFERENCE = (
    ROOT
    / "results"
    / "benign_vs_poison"
    / "attack_reference.csv"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "median5_rag_defense"
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
# EXPERIMENT DATES
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
# LOAD ORIGINAL EVENTS
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
).dt.normalize()


for feature in RAG_FEATURES:

    events[feature] = pd.to_numeric(
        events[feature],
        errors="coerce",
    )


if events[
    RAG_FEATURES
].isna().any().any():

    raise RuntimeError(
        "Missing original RAG signals."
    )


# ============================================================
# LOAD PAIRED CLEAN / ATTACK SIGNALS
# ============================================================

pairs = pd.read_csv(
    PAIR_PATH
)

pairs["ticker"] = (
    pairs["ticker"]
    .astype(str)
    .str.upper()
)

pairs["event_date"] = pd.to_datetime(
    pairs["event_date"]
).dt.normalize()


PAIR_COLUMNS = [
    "clean_impact_score",
    "clean_risk_score",
    "clean_confidence",
    "attack_impact_score",
    "attack_risk_score",
    "attack_confidence",
]


for col in PAIR_COLUMNS:

    pairs[col] = pd.to_numeric(
        pairs[col],
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
        "Attack-pair table contains missing values."
    )


# ============================================================
# LOAD 20 BENIGN ROUNDS
# ============================================================

benign = pd.read_csv(
    BENIGN_PATH
)

benign["ticker"] = (
    benign["ticker"]
    .astype(str)
    .str.upper()
)

benign["event_date"] = pd.to_datetime(
    benign["event_date"]
).dt.normalize()

benign["round"] = pd.to_numeric(
    benign["round"],
    errors="raise",
).astype(int)


BENIGN_NUMERIC = [
    "benign_impact_score",
    "benign_risk_score",
    "benign_confidence",
]


for col in BENIGN_NUMERIC:

    benign[col] = pd.to_numeric(
        benign[col],
        errors="coerce",
    )


benign = benign[
    benign["round"].between(
        0,
        19,
    )
].copy()


if len(benign) != 20 * 62:

    raise RuntimeError(
        f"Expected {20*62} benign rows, "
        f"found {len(benign)}."
    )


# ============================================================
# UPSTREAM SINGLE-SHOT BENIGN STATISTICS
# ============================================================

single_abs_impact_delta = (
    benign[
        "abs_impact_delta_vs_canonical"
    ]
    .astype(float)
)


single_sign_summary = (
    benign.groupby(
        "round"
    )[
        "impact_sign_changed"
    ]
    .sum()
)


single_benign_mean_abs_delta = float(
    single_abs_impact_delta.mean()
)

single_benign_mean_sign_changes = float(
    single_sign_summary.mean()
)


# ============================================================
# EVENT REPLACEMENT HELPER
# ============================================================

def replace_target_signals(
    base_events,
    replacements,
):

    output = base_events.copy()


    for _, row in replacements.iterrows():

        mask = (
            (
                output["ticker"]
                ==
                row["ticker"]
            )
            &
            (
                output["event_date"]
                ==
                row["event_date"]
            )
        )


        if int(
            mask.sum()
        ) != 1:

            raise RuntimeError(
                f"Expected exactly one event: "
                f"{row['ticker']} "
                f"{row['event_date']}"
            )


        output.loc[
            mask,
            "impact_score",
        ] = float(
            row[
                "impact_score"
            ]
        )


        output.loc[
            mask,
            "risk_score",
        ] = float(
            row[
                "risk_score"
            ]
        )


        output.loc[
            mask,
            "confidence",
        ] = float(
            row[
                "confidence"
            ]
        )


    return output


# ============================================================
# PAIRED CLEAN REPLACEMENTS
# ============================================================

paired_replacements = pd.DataFrame(
    {
        "ticker":
            pairs["ticker"],

        "event_date":
            pairs["event_date"],

        "impact_score":
            pairs[
                "clean_impact_score"
            ],

        "risk_score":
            pairs[
                "clean_risk_score"
            ],

        "confidence":
            pairs[
                "clean_confidence"
            ],
    }
)


attack_replacements = pd.DataFrame(
    {
        "ticker":
            pairs["ticker"],

        "event_date":
            pairs["event_date"],

        "impact_score":
            pairs[
                "attack_impact_score"
            ],

        "risk_score":
            pairs[
                "attack_risk_score"
            ],

        "confidence":
            pairs[
                "attack_confidence"
            ],
    }
)


paired_events = replace_target_signals(
    events,
    paired_replacements,
)

attack_events = replace_target_signals(
    events,
    attack_replacements,
)


# ============================================================
# BUILD MEDIAN-5 SIGNALS
#
# attacked_positions:
# which of the five views are replaced by the malicious signal.
#
# k=0 -> clean Median-of-5
# k=5 -> all five views attacked
# ============================================================

def build_median5_replacements(
    round_ids,
    attacked_positions,
):

    rows = []


    for _, pair_row in pairs.iterrows():

        ticker = pair_row[
            "ticker"
        ]

        event_date = pair_row[
            "event_date"
        ]


        temp = benign[
            (
                benign["ticker"]
                ==
                ticker
            )
            &
            (
                benign["event_date"]
                ==
                event_date
            )
            &
            (
                benign["round"].isin(
                    round_ids
                )
            )
        ].copy()


        temp = temp.sort_values(
            "round"
        )


        if len(temp) != 5:

            raise RuntimeError(
                f"Expected 5 benign views for "
                f"{ticker} {event_date}, "
                f"found {len(temp)}."
            )


        impacts = (
            temp[
                "benign_impact_score"
            ]
            .astype(float)
            .to_numpy(
                copy=True
            )
        )

        risks = (
            temp[
                "benign_risk_score"
            ]
            .astype(float)
            .to_numpy(
                copy=True
            )
        )

        confidences = (
            temp[
                "benign_confidence"
            ]
            .astype(float)
            .to_numpy(
                copy=True
            )
        )


        for position in attacked_positions:

            impacts[
                position
            ] = float(
                pair_row[
                    "attack_impact_score"
                ]
            )

            risks[
                position
            ] = float(
                pair_row[
                    "attack_risk_score"
                ]
            )

            confidences[
                position
            ] = float(
                pair_row[
                    "attack_confidence"
                ]
            )


        rows.append(
            {
                "ticker":
                    ticker,

                "event_date":
                    event_date,

                "impact_score":
                    float(
                        np.median(
                            impacts
                        )
                    ),

                "risk_score":
                    float(
                        np.median(
                            risks
                        )
                    ),

                "confidence":
                    float(
                        np.median(
                            confidences
                        )
                    ),
            }
        )


    return pd.DataFrame(
        rows
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
    print("BUILD MARKET DATA")
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
        ==
        len(
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
# STRICT POINT-IN-TIME MERGE
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


    pieces = []


    for ticker in TIC_ORDER:

        m = market[
            market["tic"]
            ==
            ticker
        ].copy()


        e = event_table[
            event_table["ticker"]
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
    ):

        raise RuntimeError(
            "Point-in-time leakage detected."
        )


    return panel


# ============================================================
# TRAIN-FITTED NORMALIZATION
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
                output["tic"]
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
                    ]
                    .astype(float)
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


    return output


# ============================================================
# TEST PANEL BUILDER
# ============================================================

def build_test_panel(
    market,
    event_table,
):

    panel = merge_rag(
        market,
        event_table,
    )


    normalized = normalize_panel(
        panel
    )


    return data_split(
        normalized,
        TEST_START,
        TEST_END,
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
# EVALUATION
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
        "average_daily_l1":
            float(
                daily_l1.mean()
            ),

        "average_reallocated":
            float(
                daily_l1.mean()
                /
                2.0
            ),

        "maximum_reallocated":
            float(
                daily_l1.max()
                /
                2.0
            ),

        "maximum_single_asset":
            float(
                diff
                .to_numpy()
                .max()
            ),
    }


# ============================================================
# BUILD MARKET
# ============================================================

market = build_market()


# ============================================================
# PAIRED CLEAN / FULL ATTACK TEST PANELS
# ============================================================

paired_test = build_test_panel(
    market,
    paired_events,
)


full_attack_test = build_test_panel(
    market,
    attack_events,
)


# ============================================================
# LOAD MODELS
# ============================================================

models = {}


for seed in SEEDS:

    path = (
        MODEL_ROOT
        /
        f"seed_{seed}"
        /
        "normalized_a2c_model.zip"
    )


    if not path.exists():

        raise FileNotFoundError(
            path
        )


    models[
        seed
    ] = A2C.load(
        str(
            path
        )
    )


# ============================================================
# PAIRED CLEAN REFERENCES
# ============================================================

paired_reference = {}


for seed in SEEDS:

    (
        returns,
        weights,
    ) = evaluate(
        models[
            seed
        ],
        paired_test,
    )


    paired_reference[
        seed
    ] = {
        "returns":
            returns,

        "weights":
            weights,

        "performance":
            performance_metrics(
                returns
            ),
    }


# ============================================================
# NO-DEFENSE FULL ATTACK REFERENCE
# ============================================================

no_defense_rows = []


for seed in SEEDS:

    (
        attack_returns,
        attack_weights,
    ) = evaluate(
        models[
            seed
        ],
        full_attack_test,
    )


    shift = policy_distance(
        paired_reference[
            seed
        ][
            "weights"
        ],
        attack_weights,
    )


    attack_perf = performance_metrics(
        attack_returns
    )


    clean_perf = (
        paired_reference[
            seed
        ][
            "performance"
        ]
    )


    no_defense_rows.append(
        {
            "seed":
                seed,

            "attack_reallocated":
                shift[
                    "average_reallocated"
                ],

            "attack_daily_l1":
                shift[
                    "average_daily_l1"
                ],

            "delta_multiple":
                (
                    attack_perf[
                        "multiple"
                    ]
                    -
                    clean_perf[
                        "multiple"
                    ]
                ),

            "delta_sharpe":
                (
                    attack_perf[
                        "sharpe"
                    ]
                    -
                    clean_perf[
                        "sharpe"
                    ]
                ),
        }
    )


no_defense = pd.DataFrame(
    no_defense_rows
)


no_defense.to_csv(
    RESULT_DIR
    / "no_defense_full_attack_reference.csv",
    index=False,
)


# ============================================================
# UPSTREAM MEDIAN-5 CLEAN STABILITY
# ============================================================

median5_clean_rows = []

clean_median_replacements = {}


for ensemble_id, round_ids in enumerate(
    ENSEMBLE_GROUPS
):

    replacements = (
        build_median5_replacements(
            round_ids,
            attacked_positions=(),
        )
    )


    clean_median_replacements[
        ensemble_id
    ] = replacements


    merged = replacements.merge(
        paired_replacements[
            [
                "ticker",
                "event_date",
                "impact_score",
            ]
        ],
        on=[
            "ticker",
            "event_date",
        ],
        suffixes=(
            "_median5",
            "_canonical",
        ),
    )


    merged[
        "abs_impact_delta"
    ] = (
        merged[
            "impact_score_median5"
        ]
        -
        merged[
            "impact_score_canonical"
        ]
    ).abs()


    merged[
        "sign_changed"
    ] = (
        np.sign(
            merged[
                "impact_score_median5"
            ]
        )
        !=
        np.sign(
            merged[
                "impact_score_canonical"
            ]
        )
    )


    median5_clean_rows.append(
        {
            "ensemble":
                ensemble_id + 1,

            "rounds":
                ",".join(
                    str(
                        x
                    )
                    for x in round_ids
                ),

            "mean_abs_impact_delta":
                float(
                    merged[
                        "abs_impact_delta"
                    ].mean()
                ),

            "sign_changes":
                int(
                    merged[
                        "sign_changed"
                    ].sum()
                ),
        }
    )


median5_clean_summary = pd.DataFrame(
    median5_clean_rows
)


median5_clean_summary.to_csv(
    RESULT_DIR
    / "median5_clean_upstream_stability.csv",
    index=False,
)


# ============================================================
# BUILD CLEAN MEDIAN-5 BASELINES
# ============================================================

median_baselines = {}


clean_drift_rows = []


for ensemble_id, round_ids in enumerate(
    ENSEMBLE_GROUPS
):

    replacements = (
        clean_median_replacements[
            ensemble_id
        ]
    )


    event_table = replace_target_signals(
        events,
        replacements,
    )


    test_df = build_test_panel(
        market,
        event_table,
    )


    median_baselines[
        ensemble_id
    ] = {
        "test_df":
            test_df,

        "by_seed":
            {},
    }


    for seed in SEEDS:

        (
            returns,
            weights,
        ) = evaluate(
            models[
                seed
            ],
            test_df,
        )


        perf = performance_metrics(
            returns
        )


        median_baselines[
            ensemble_id
        ][
            "by_seed"
        ][
            seed
        ] = {
            "returns":
                returns,

            "weights":
                weights,

            "performance":
                perf,
        }


        shift = policy_distance(
            paired_reference[
                seed
            ][
                "weights"
            ],
            weights,
        )


        clean_drift_rows.append(
            {
                "ensemble":
                    ensemble_id + 1,

                "seed":
                    seed,

                "median5_clean_reallocated":
                    shift[
                        "average_reallocated"
                    ],

                "median5_clean_daily_l1":
                    shift[
                        "average_daily_l1"
                    ],
            }
        )


clean_drift = pd.DataFrame(
    clean_drift_rows
)


clean_drift.to_csv(
    RESULT_DIR
    / "median5_clean_portfolio_drift.csv",
    index=False,
)


# ============================================================
# CONTAMINATION SWEEP
#
# For each group of 5:
#
# k = 0,1,2,3,4,5 attacked views.
#
# For k=1:
# all 5 possible attacked slots.
#
# For k=2:
# all 10 combinations.
#
# ...
#
# This removes arbitrary slot-selection effects.
# ============================================================

downstream_rows = []

upstream_rows = []


for ensemble_id, round_ids in enumerate(
    ENSEMBLE_GROUPS
):

    print("\n" + "=" * 75)

    print(
        f"ENSEMBLE {ensemble_id + 1}/"
        f"{len(ENSEMBLE_GROUPS)} "
        f"ROUNDS={round_ids}"
    )

    print(
        "=" * 75
    )


    clean_replacements = (
        clean_median_replacements[
            ensemble_id
        ]
    )


    clean_by_seed = (
        median_baselines[
            ensemble_id
        ][
            "by_seed"
        ]
    )


    # --------------------------------------------------------
    # k = 0 baseline row
    # --------------------------------------------------------

    for seed in SEEDS:

        downstream_rows.append(
            {
                "ensemble":
                    ensemble_id + 1,

                "attacked_views":
                    0,

                "attack_fraction":
                    0.0,

                "combination":
                    "none",

                "seed":
                    seed,

                "average_daily_l1":
                    0.0,

                "average_reallocated":
                    0.0,

                "maximum_reallocated":
                    0.0,

                "maximum_single_asset":
                    0.0,

                "delta_multiple":
                    0.0,

                "delta_sharpe":
                    0.0,
            }
        )


    upstream_rows.append(
        {
            "ensemble":
                ensemble_id + 1,

            "attacked_views":
                0,

            "attack_fraction":
                0.0,

            "combination":
                "none",

            "mean_abs_impact_shift_vs_clean_median5":
                0.0,

            "impact_sign_changes_vs_clean_median5":
                0,
        }
    )


    # --------------------------------------------------------
    # k = 1 ... 5
    # --------------------------------------------------------

    for k in range(
        1,
        6,
    ):

        combinations = list(
            itertools.combinations(
                range(
                    5
                ),
                k,
            )
        )


        print(
            f"\n{k}/5 attacked views "
            f"({len(combinations)} patterns)"
        )


        for pattern_index, attacked_positions in enumerate(
            combinations,
            start=1,
        ):

            replacements = (
                build_median5_replacements(
                    round_ids,
                    attacked_positions,
                )
            )


            # =================================================
            # UPSTREAM DIFFERENCE FROM CLEAN MEDIAN-5
            # =================================================

            upstream_compare = replacements.merge(
                clean_replacements[
                    [
                        "ticker",
                        "event_date",
                        "impact_score",
                    ]
                ],
                on=[
                    "ticker",
                    "event_date",
                ],
                suffixes=(
                    "_contaminated",
                    "_clean",
                ),
            )


            upstream_compare[
                "abs_impact_shift"
            ] = (
                upstream_compare[
                    "impact_score_contaminated"
                ]
                -
                upstream_compare[
                    "impact_score_clean"
                ]
            ).abs()


            upstream_compare[
                "sign_changed"
            ] = (
                np.sign(
                    upstream_compare[
                        "impact_score_contaminated"
                    ]
                )
                !=
                np.sign(
                    upstream_compare[
                        "impact_score_clean"
                    ]
                )
            )


            combination_label = ",".join(
                str(
                    x
                )
                for x in attacked_positions
            )


            upstream_rows.append(
                {
                    "ensemble":
                        ensemble_id + 1,

                    "attacked_views":
                        k,

                    "attack_fraction":
                        k / 5.0,

                    "combination":
                        combination_label,

                    "mean_abs_impact_shift_vs_clean_median5":
                        float(
                            upstream_compare[
                                "abs_impact_shift"
                            ].mean()
                        ),

                    "impact_sign_changes_vs_clean_median5":
                        int(
                            upstream_compare[
                                "sign_changed"
                            ].sum()
                        ),
                }
            )


            # =================================================
            # DOWNSTREAM
            # =================================================

            if k == 5:

                # All 5 views replaced by the same attack:
                # median = original attack signal exactly.
                test_df = (
                    full_attack_test
                )

            else:

                event_table = replace_target_signals(
                    events,
                    replacements,
                )


                test_df = build_test_panel(
                    market,
                    event_table,
                )


            for seed in SEEDS:

                (
                    returns,
                    weights,
                ) = evaluate(
                    models[
                        seed
                    ],
                    test_df,
                )


                baseline_weights = (
                    clean_by_seed[
                        seed
                    ][
                        "weights"
                    ]
                )


                baseline_perf = (
                    clean_by_seed[
                        seed
                    ][
                        "performance"
                    ]
                )


                shift = policy_distance(
                    baseline_weights,
                    weights,
                )


                perf = performance_metrics(
                    returns
                )


                downstream_rows.append(
                    {
                        "ensemble":
                            ensemble_id + 1,

                        "attacked_views":
                            k,

                        "attack_fraction":
                            k / 5.0,

                        "combination":
                            combination_label,

                        "seed":
                            seed,

                        "average_daily_l1":
                            shift[
                                "average_daily_l1"
                            ],

                        "average_reallocated":
                            shift[
                                "average_reallocated"
                            ],

                        "maximum_reallocated":
                            shift[
                                "maximum_reallocated"
                            ],

                        "maximum_single_asset":
                            shift[
                                "maximum_single_asset"
                            ],

                        "delta_multiple":
                            (
                                perf[
                                    "multiple"
                                ]
                                -
                                baseline_perf[
                                    "multiple"
                                ]
                            ),

                        "delta_sharpe":
                            (
                                perf[
                                    "sharpe"
                                ]
                                -
                                baseline_perf[
                                    "sharpe"
                                ]
                            ),
                    }
                )


            print(
                f"  pattern "
                f"{pattern_index:02d}/"
                f"{len(combinations)}",
                end="\r",
                flush=True,
            )


        print()


# ============================================================
# SAVE RAW RESULTS
# ============================================================

downstream = pd.DataFrame(
    downstream_rows
)


upstream = pd.DataFrame(
    upstream_rows
)


downstream.to_csv(
    RESULT_DIR
    / "median5_contamination_downstream_runs.csv",
    index=False,
)


upstream.to_csv(
    RESULT_DIR
    / "median5_contamination_upstream_runs.csv",
    index=False,
)


# ============================================================
# SUMMARIZE UPSTREAM BY NUMBER OF ATTACKED VIEWS
# ============================================================

upstream_summary = (
    upstream.groupby(
        "attacked_views"
    )
    .agg(
        mean_abs_impact_shift=(
            "mean_abs_impact_shift_vs_clean_median5",
            "mean",
        ),

        std_abs_impact_shift=(
            "mean_abs_impact_shift_vs_clean_median5",
            "std",
        ),

        mean_sign_changes=(
            "impact_sign_changes_vs_clean_median5",
            "mean",
        ),

        max_sign_changes=(
            "impact_sign_changes_vs_clean_median5",
            "max",
        ),
    )
    .reset_index()
)


upstream_summary[
    "attack_fraction"
] = (
    upstream_summary[
        "attacked_views"
    ]
    /
    5.0
)


upstream_summary.to_csv(
    RESULT_DIR
    / "median5_upstream_summary.csv",
    index=False,
)


# ============================================================
# DOWNSTREAM:
# FIRST AVERAGE ACROSS ENSEMBLE/PATTERN WITHIN EACH SEED
#
# Then average across the 5 independently trained policies.
# ============================================================

seed_k_summary = (
    downstream.groupby(
        [
            "seed",
            "attacked_views",
        ]
    )
    .agg(
        mean_reallocated=(
            "average_reallocated",
            "mean",
        ),

        mean_daily_l1=(
            "average_daily_l1",
            "mean",
        ),

        mean_max_reallocated=(
            "maximum_reallocated",
            "mean",
        ),

        mean_delta_multiple=(
            "delta_multiple",
            "mean",
        ),

        mean_delta_sharpe=(
            "delta_sharpe",
            "mean",
        ),
    )
    .reset_index()
)


seed_k_summary[
    "attack_fraction"
] = (
    seed_k_summary[
        "attacked_views"
    ]
    /
    5.0
)


seed_k_summary.to_csv(
    RESULT_DIR
    / "median5_seed_level_summary.csv",
    index=False,
)


cross_seed_summary = (
    seed_k_summary.groupby(
        "attacked_views"
    )
    .agg(
        mean_reallocated=(
            "mean_reallocated",
            "mean",
        ),

        std_reallocated=(
            "mean_reallocated",
            "std",
        ),

        mean_daily_l1=(
            "mean_daily_l1",
            "mean",
        ),

        std_daily_l1=(
            "mean_daily_l1",
            "std",
        ),

        mean_delta_multiple=(
            "mean_delta_multiple",
            "mean",
        ),

        std_delta_multiple=(
            "mean_delta_multiple",
            "std",
        ),

        mean_delta_sharpe=(
            "mean_delta_sharpe",
            "mean",
        ),

        std_delta_sharpe=(
            "mean_delta_sharpe",
            "std",
        ),
    )
    .reset_index()
)


cross_seed_summary[
    "attack_fraction"
] = (
    cross_seed_summary[
        "attacked_views"
    ]
    /
    5.0
)


cross_seed_summary.to_csv(
    RESULT_DIR
    / "median5_cross_seed_summary.csv",
    index=False,
)


# ============================================================
# CLEAN-STABILITY COMPARISON
# ============================================================

median5_mean_abs_delta = float(
    median5_clean_summary[
        "mean_abs_impact_delta"
    ].mean()
)


median5_mean_sign_changes = float(
    median5_clean_summary[
        "sign_changes"
    ].mean()
)


median5_portfolio_clean_drift = float(
    clean_drift[
        "median5_clean_reallocated"
    ].mean()
)


single_shot_portfolio_drift = np.nan


if PREVIOUS_BENIGN_RUNS.exists():

    previous = pd.read_csv(
        PREVIOUS_BENIGN_RUNS
    )


    previous = previous[
        previous[
            "condition"
        ]
        ==
        "normalized"
    ]


    if len(
        previous
    ):

        single_shot_portfolio_drift = float(
            previous[
                "benign_reallocated"
            ].mean()
        )


# ============================================================
# FULL ATTACK REFERENCE
# ============================================================

no_defense_attack_mean = float(
    no_defense[
        "attack_reallocated"
    ].mean()
)


# ============================================================
# MEDIAN DEFENSE FULL-COMPROMISE EFFECT
#
# k=5 relative to Median-of-5 clean baseline.
# ============================================================

k5 = cross_seed_summary[
    cross_seed_summary[
        "attacked_views"
    ]
    ==
    5
].iloc[
    0
]


k5_attack_mean = float(
    k5[
        "mean_reallocated"
    ]
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print("MEDIAN-OF-5 CLEAN STABILITY")
print("=" * 75)

print(
    "Single-shot benign mean |impact delta|:",
    f"{single_benign_mean_abs_delta:.4f}",
)

print(
    "Median-of-5 mean |impact delta|:",
    f"{median5_mean_abs_delta:.4f}",
)


if single_benign_mean_abs_delta > 0:

    print(
        "Upstream magnitude reduction:",
        f"{1 - median5_mean_abs_delta / single_benign_mean_abs_delta:.2%}",
    )


print(
    "\nSingle-shot benign sign changes / 62:",
    f"{single_benign_mean_sign_changes:.2f}",
)

print(
    "Median-of-5 sign changes / 62:",
    f"{median5_mean_sign_changes:.2f}",
)


if single_benign_mean_sign_changes > 0:

    print(
        "Sign-instability reduction:",
        f"{1 - median5_mean_sign_changes / single_benign_mean_sign_changes:.2%}",
    )


print(
    "\nMedian-of-5 clean portfolio drift "
    "vs paired-clean reference:",
    f"{median5_portfolio_clean_drift:.4%}",
)


if np.isfinite(
    single_shot_portfolio_drift
):

    print(
        "Single-shot benign portfolio drift:",
        f"{single_shot_portfolio_drift:.4%}",
    )


    if single_shot_portfolio_drift > 0:

        print(
            "Portfolio benign-variation reduction:",
            f"{1 - median5_portfolio_clean_drift / single_shot_portfolio_drift:.2%}",
        )


print("\n" + "=" * 75)
print("MEDIAN-OF-5 ATTACK CONTAMINATION SWEEP — UPSTREAM")
print("=" * 75)

print(
    upstream_summary[
        [
            "attacked_views",
            "attack_fraction",
            "mean_abs_impact_shift",
            "mean_sign_changes",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("MEDIAN-OF-5 ATTACK CONTAMINATION SWEEP — PORTFOLIO")
print("=" * 75)

print(
    cross_seed_summary[
        [
            "attacked_views",
            "attack_fraction",
            "mean_reallocated",
            "std_reallocated",
            "mean_delta_multiple",
            "mean_delta_sharpe",
        ]
    ].to_string(
        index=False
    )
)


print("\n" + "=" * 75)
print("FULL-COMPROMISE CHECK")
print("=" * 75)

print(
    "No-defense full attack "
    "(paired clean -> attack):",
    f"{no_defense_attack_mean:.4%}",
)

print(
    "Median-5 with 5/5 views attacked "
    "(median clean -> attack):",
    f"{k5_attack_mean:.4%}",
)


print(
    "\nIMPORTANT:"
)

print(
    "Median-of-5 can reject minority view contamination, "
    "but it is NOT expected to stop a corpus-level poisoning attack "
    "that compromises the evidence seen by all 5 views."
)


# ============================================================
# SIMPLE BREAKDOWN INDICATOR
# ============================================================

max_effect = float(
    cross_seed_summary[
        "mean_reallocated"
    ].max()
)


if max_effect > 0:

    temp = cross_seed_summary.copy()

    temp[
        "fraction_of_max_effect"
    ] = (
        temp[
            "mean_reallocated"
        ]
        /
        max_effect
    )


    reached = temp[
        temp[
            "fraction_of_max_effect"
        ]
        >=
        0.5
    ]


    if len(
        reached
    ):

        first_k = int(
            reached.iloc[
                0
            ][
                "attacked_views"
            ]
        )


        print(
            "\nFirst contamination level reaching "
            ">=50% of maximum portfolio attack effect:",
            f"{first_k}/5 views"
        )


print("\nResults saved to:")

print(
    RESULT_DIR
)

print("\nDONE")