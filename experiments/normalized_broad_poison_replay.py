import contextlib
import io
import json
import math
import re
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
    sys.path.insert(0, str(ROOT))


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

OLD_COMPARE_ROOT = (
    ROOT
    / "results"
    / "multi_drl_rag_compare"
)

CONFIG_PATH = (
    OLD_COMPARE_ROOT
    / "experiment_config.json"
)

NORMALIZED_ROOT = (
    ROOT
    / "results"
    / "normalized_rag_feature_importance"
)

MODEL_PATH = (
    NORMALIZED_ROOT
    / "normalized_a2c_model.zip"
)

SCALER_PATH = (
    NORMALIZED_ROOT
    / "train_only_scaler.json"
)

MODEL_METRICS_PATH = (
    NORMALIZED_ROOT
    / "normalized_model_metrics.json"
)

OLD_ATTACK_ROOT = (
    ROOT
    / "results"
    / "broad_persistent_poison"
    / "ALL5_2023_opposite_p3"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "normalized_broad_persistent_poison"
    / "ALL5_2023_opposite_p3"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CHECK FILES
# ============================================================

for required in [
    EVENT_PATH,
    CONFIG_PATH,
    MODEL_PATH,
    SCALER_PATH,
    OLD_ATTACK_ROOT,
]:

    if not required.exists():

        raise FileNotFoundError(
            f"Missing required path:\n{required}"
        )


# ============================================================
# LOAD EXPERIMENT DATES
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

TEST_START = exp_cfg[
    "validation_end_exclusive"
]

TEST_END = exp_cfg[
    "test_end_exclusive"
]


# ============================================================
# HELPERS FOR OLD ATTACK ARTIFACT DISCOVERY
#
# We do not assume the exact filename from the old experiment.
# Search its CSV / JSONL outputs for paired-clean + attack scores.
# ============================================================

def norm_name(x):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(x).lower(),
    ).strip("_")


def read_table(path):

    try:

        if path.suffix.lower() == ".csv":

            return pd.read_csv(
                path
            )

        if path.suffix.lower() in {
            ".jsonl",
            ".ndjson",
        }:

            return pd.read_json(
                path,
                lines=True,
            )

    except Exception:

        return None

    return None


def find_exact_alias(
    columns,
    aliases,
):

    normalized = {
        norm_name(c): c
        for c in columns
    }

    for alias in aliases:

        if alias in normalized:

            return normalized[
                alias
            ]

    return None


def choose_role_feature_column(
    columns,
    feature_token,
    role,
):

    candidates = []

    for col in columns:

        n = norm_name(
            col
        )

        if feature_token not in n:
            continue

        if role == "clean":

            if "clean" not in n:
                continue

            # Avoid delta / comparison columns.
            if "delta" in n:
                continue

            score = 0

            if "paired" in n:
                score += 100

            if "paired_clean" in n:
                score += 100

            if "score" in n:
                score += 10

            # Prefer a pure clean column.
            if (
                "attack" not in n
                and
                "poison" not in n
            ):
                score += 20

            candidates.append(
                (
                    score,
                    col,
                )
            )

        elif role == "attack":

            if not (
                "attack" in n
                or
                "poison" in n
            ):
                continue

            if "delta" in n:
                continue

            score = 0

            if "attack" in n:
                score += 50

            if "poison" in n:
                score += 40

            if "score" in n:
                score += 10

            candidates.append(
                (
                    score,
                    col,
                )
            )


    if not candidates:

        return None


    candidates.sort(
        reverse=True
    )

    return candidates[
        0
    ][
        1
    ]


def standardize_pair_table(
    df,
    source_name,
):

    ticker_col = find_exact_alias(
        df.columns,
        [
            "ticker",
            "tic",
            "symbol",
        ],
    )

    date_col = find_exact_alias(
        df.columns,
        [
            "event_date",
            "filing_date",
            "source_filing_date",
            "date",
        ],
    )


    if (
        ticker_col is None
        or
        date_col is None
    ):

        return None


    clean_impact = choose_role_feature_column(
        df.columns,
        "impact",
        "clean",
    )

    clean_risk = choose_role_feature_column(
        df.columns,
        "risk",
        "clean",
    )

    clean_conf = choose_role_feature_column(
        df.columns,
        "confidence",
        "clean",
    )

    attack_impact = choose_role_feature_column(
        df.columns,
        "impact",
        "attack",
    )

    attack_risk = choose_role_feature_column(
        df.columns,
        "risk",
        "attack",
    )

    attack_conf = choose_role_feature_column(
        df.columns,
        "confidence",
        "attack",
    )


    required = [
        clean_impact,
        clean_risk,
        clean_conf,
        attack_impact,
        attack_risk,
        attack_conf,
    ]


    if any(
        x is None
        for x in required
    ):

        return None


    out = pd.DataFrame(
        {
            "ticker":
                df[
                    ticker_col
                ]
                .astype(str)
                .str.upper(),

            "event_date":
                pd.to_datetime(
                    df[
                        date_col
                    ],
                    errors="coerce",
                ),

            "clean_impact_score":
                pd.to_numeric(
                    df[
                        clean_impact
                    ],
                    errors="coerce",
                ),

            "clean_risk_score":
                pd.to_numeric(
                    df[
                        clean_risk
                    ],
                    errors="coerce",
                ),

            "clean_confidence":
                pd.to_numeric(
                    df[
                        clean_conf
                    ],
                    errors="coerce",
                ),

            "attack_impact_score":
                pd.to_numeric(
                    df[
                        attack_impact
                    ],
                    errors="coerce",
                ),

            "attack_risk_score":
                pd.to_numeric(
                    df[
                        attack_risk
                    ],
                    errors="coerce",
                ),

            "attack_confidence":
                pd.to_numeric(
                    df[
                        attack_conf
                    ],
                    errors="coerce",
                ),
        }
    )


    out = out.dropna().copy()

    out = out[
        out["ticker"].isin(
            TIC_ORDER
        )
    ].copy()

    out = out[
        out["event_date"].dt.year
        == TARGET_YEAR
    ].copy()


    out = out.drop_duplicates(
        subset=[
            "ticker",
            "event_date",
        ],
        keep="last",
    )


    out[
        "source_artifact"
    ] = source_name


    return out


# ============================================================
# FALLBACK:
# Search for separate paired-clean and attack tables.
# ============================================================

def find_raw_signal_table(
    path,
):

    df = read_table(
        path
    )

    if df is None:
        return None


    ticker_col = find_exact_alias(
        df.columns,
        [
            "ticker",
            "tic",
            "symbol",
        ],
    )

    date_col = find_exact_alias(
        df.columns,
        [
            "event_date",
            "filing_date",
            "source_filing_date",
            "date",
        ],
    )


    normalized_cols = {
        norm_name(c): c
        for c in df.columns
    }


    impact_col = normalized_cols.get(
        "impact_score"
    )

    risk_col = normalized_cols.get(
        "risk_score"
    )

    conf_col = normalized_cols.get(
        "confidence"
    )


    if any(
        x is None
        for x in [
            ticker_col,
            date_col,
            impact_col,
            risk_col,
            conf_col,
        ]
    ):

        return None


    out = pd.DataFrame(
        {
            "ticker":
                df[
                    ticker_col
                ]
                .astype(str)
                .str.upper(),

            "event_date":
                pd.to_datetime(
                    df[
                        date_col
                    ],
                    errors="coerce",
                ),

            "impact_score":
                pd.to_numeric(
                    df[
                        impact_col
                    ],
                    errors="coerce",
                ),

            "risk_score":
                pd.to_numeric(
                    df[
                        risk_col
                    ],
                    errors="coerce",
                ),

            "confidence":
                pd.to_numeric(
                    df[
                        conf_col
                    ],
                    errors="coerce",
                ),
        }
    )


    out = out.dropna()

    out = out[
        out["ticker"].isin(
            TIC_ORDER
        )
    ]

    out = out[
        out["event_date"].dt.year
        == TARGET_YEAR
    ]

    out = out.drop_duplicates(
        [
            "ticker",
            "event_date",
        ],
        keep="last",
    )


    return out


def discover_attack_pairs():

    files = sorted(
        list(
            OLD_ATTACK_ROOT.rglob(
                "*.csv"
            )
        )
        +
        list(
            OLD_ATTACK_ROOT.rglob(
                "*.jsonl"
            )
        )
    )


    if not files:

        raise RuntimeError(
            f"No CSV/JSONL files found under:\n"
            f"{OLD_ATTACK_ROOT}"
        )


    # --------------------------------------------------------
    # First try a combined event-audit table
    # --------------------------------------------------------

    combined_candidates = []


    for path in files:

        df = read_table(
            path
        )

        if df is None:
            continue


        standardized = standardize_pair_table(
            df,
            str(path),
        )


        if standardized is None:
            continue


        score = len(
            standardized
        )


        if "event" in path.name.lower():
            score += 1000

        if "audit" in path.name.lower():
            score += 1000

        if "signal" in path.name.lower():
            score += 500


        combined_candidates.append(
            (
                score,
                path,
                standardized,
            )
        )


    if combined_candidates:

        combined_candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        score, path, table = (
            combined_candidates[0]
        )

        print(
            "Found combined paired signal artifact:"
        )

        print(
            path
        )

        print(
            "Detected rows:",
            len(table),
        )


        return table


    # --------------------------------------------------------
    # Separate files fallback
    # --------------------------------------------------------

    clean_candidates = []
    attack_candidates = []


    for path in files:

        table = find_raw_signal_table(
            path
        )

        if table is None:
            continue


        name = path.name.lower()


        if (
            "paired" in name
            and
            "clean" in name
        ) or (
            "clean" in name
            and
            "attack" not in name
            and
            "poison" not in name
        ):

            clean_candidates.append(
                (
                    abs(
                        len(table)
                        - 62
                    ),
                    path,
                    table,
                )
            )


        if (
            "attack" in name
            or
            "poison" in name
        ):

            attack_candidates.append(
                (
                    abs(
                        len(table)
                        - 62
                    ),
                    path,
                    table,
                )
            )


    if (
        not clean_candidates
        or
        not attack_candidates
    ):

        print(
            "\nCould not automatically identify "
            "the old paired-clean / attack signal table."
        )

        print(
            "\nFiles found:"
        )

        for path in files:

            df = read_table(
                path
            )

            if df is None:
                continue

            print(
                "\n",
                path,
            )

            print(
                list(
                    df.columns
                )
            )


        raise RuntimeError(
            "Attack artifacts not recognized. "
            "The file list/columns above show what is available."
        )


    clean_candidates.sort(
        key=lambda x: x[0]
    )

    attack_candidates.sort(
        key=lambda x: x[0]
    )


    _, clean_path, clean_df = (
        clean_candidates[0]
    )

    _, attack_path, attack_df = (
        attack_candidates[0]
    )


    print(
        "Found paired clean artifact:"
    )

    print(
        clean_path
    )

    print(
        "Found attack artifact:"
    )

    print(
        attack_path
    )


    merged = clean_df.merge(
        attack_df,
        on=[
            "ticker",
            "event_date",
        ],
        how="inner",
        suffixes=(
            "_clean",
            "_attack",
        ),
    )


    out = pd.DataFrame(
        {
            "ticker":
                merged[
                    "ticker"
                ],

            "event_date":
                merged[
                    "event_date"
                ],

            "clean_impact_score":
                merged[
                    "impact_score_clean"
                ],

            "clean_risk_score":
                merged[
                    "risk_score_clean"
                ],

            "clean_confidence":
                merged[
                    "confidence_clean"
                ],

            "attack_impact_score":
                merged[
                    "impact_score_attack"
                ],

            "attack_risk_score":
                merged[
                    "risk_score_attack"
                ],

            "attack_confidence":
                merged[
                    "confidence_attack"
                ],
        }
    )


    out[
        "source_artifact"
    ] = (
        f"{clean_path} | {attack_path}"
    )


    return out


# ============================================================
# DISCOVER OLD 62-EVENT ATTACK
# ============================================================

print("=" * 70)
print("DISCOVER PREVIOUS BROAD POISONING SIGNALS")
print("=" * 70)


pairs = discover_attack_pairs()


pairs = pairs.sort_values(
    [
        "ticker",
        "event_date",
    ]
).reset_index(
    drop=True
)


print(
    "\nTarget events discovered:",
    len(pairs),
)


print(
    "\nEvents by ticker:"
)

print(
    pairs.groupby(
        "ticker"
    ).size()
)


if len(pairs) != 62:

    raise RuntimeError(
        f"Expected exactly 62 attacked events, "
        f"but discovered {len(pairs)}.\n"
        f"Do not continue because this would not reproduce "
        f"the previous ALL5 2023 experiment."
    )


# ============================================================
# ATTACK SIGNAL QA
# ============================================================

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
    "risk_delta"
] = (
    pairs[
        "attack_risk_score"
    ]
    -
    pairs[
        "clean_risk_score"
    ]
)


pairs[
    "confidence_delta"
] = (
    pairs[
        "attack_confidence"
    ]
    -
    pairs[
        "clean_confidence"
    ]
)


pairs[
    "impact_sign_changed"
] = (
    np.sign(
        pairs[
            "clean_impact_score"
        ]
    )
    !=
    np.sign(
        pairs[
            "attack_impact_score"
        ]
    )
)


print(
    "\nMean impact delta:",
    f"{pairs['impact_delta'].mean():+.4f}",
)

print(
    "Mean |impact delta|:",
    f"{pairs['impact_delta'].abs().mean():.4f}",
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


pairs.to_csv(
    RESULT_DIR
    / "replayed_attack_event_pairs.csv",
    index=False,
)


# ============================================================
# ORIGINAL CLEAN EVENTS
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
        "Missing numeric values in SEC RAG event table."
    )


# ============================================================
# BUILD PAIRED-CLEAN / ATTACK EVENT TABLES
#
# Everything outside the attacked 62 events stays identical.
# ============================================================

paired_events = events.copy()
attack_events = events.copy()


replacement_count = 0


for _, row in pairs.iterrows():

    ticker = row[
        "ticker"
    ]

    event_date = pd.Timestamp(
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


    n_match = int(
        mask.sum()
    )


    if n_match != 1:

        raise RuntimeError(
            f"Expected exactly one original event for "
            f"{ticker} {event_date.date()}, "
            f"found {n_match}."
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


    replacement_count += 1


print(
    "\nReplaced target events:",
    replacement_count,
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
# MARKET DATA
# ============================================================

def build_market():

    print("\n" + "=" * 70)
    print("BUILD MARKET PANEL")
    print("=" * 70)


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
            "Missing RAG values after merge."
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
# LOAD EXACT TRAIN-ONLY NORMALIZATION
# ============================================================

with open(
    SCALER_PATH,
    "r",
    encoding="utf-8",
) as f:

    scaler = json.load(f)


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
# APPLY SAME NORMALIZATION USED TO TRAIN THE MODEL
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
                    - mean
                )
                / std
            )


    out[
        "cov_list"
    ] = out[
        "cov_list"
    ].apply(
        lambda x:
            (
                np.asarray(
                    x,
                    dtype=float,
                )
                - cov_mean
            )
            / cov_std
    )


    return out


# ============================================================
# BUILD THREE PANELS
# ============================================================

market = build_market()


print("\n" + "=" * 70)
print("BUILD ORIGINAL / PAIRED CLEAN / ATTACK PANELS")
print("=" * 70)


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


original_test = data_split(
    original_norm,
    TEST_START,
    TEST_END,
)

paired_test = data_split(
    paired_norm,
    TEST_START,
    TEST_END,
)

attack_test = data_split(
    attack_norm,
    TEST_START,
    TEST_END,
)


print(
    "Test dates:",
    original_test[
        "date"
    ].nunique(),
)

print(
    "Test rows:",
    len(
        original_test
    ),
)


# ============================================================
# RAG FEATURE ROW DIFFERENCE QA
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


rag_row_changed = np.any(
    np.abs(
        paired_rag
        - attack_rag
    )
    > 1e-12,
    axis=1,
)


changed_rag_rows = int(
    rag_row_changed.sum()
)


print(
    "Changed normalized RAG feature rows:",
    changed_rag_rows,
)


# ============================================================
# ENVIRONMENT
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
# EVALUATION
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


    std = float(
        r.std(
            ddof=1
        )
    )


    annual_volatility = (
        std
        * math.sqrt(
            252
        )
    )


    sharpe = (
        float(
            r.mean()
        )
        / std
        * math.sqrt(
            252
        )
        if std > 0
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


    per_asset = pd.DataFrame(
        {
            "ticker":
                TIC_ORDER,

            "mean_abs_weight_diff":
                [
                    float(
                        abs_diff[
                            ticker
                        ].mean()
                    )
                    for ticker in TIC_ORDER
                ],

            "max_abs_weight_diff":
                [
                    float(
                        abs_diff[
                            ticker
                        ].max()
                    )
                    for ticker in TIC_ORDER
                ],
        }
    )


    metrics = {
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


    return (
        metrics,
        per_asset,
    )


# ============================================================
# LOAD NORMALIZED MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOAD NORMALIZED CLEAN-RAG A2C")
print("=" * 70)

print(
    MODEL_PATH
)


model = A2C.load(
    str(
        MODEL_PATH
    )
)


# ============================================================
# RUN ALL THREE
# ============================================================

print("\nEvaluating original clean...")

original_returns, original_weights = (
    evaluate(
        model,
        original_test,
        quiet=True,
    )
)


print("Evaluating paired clean...")

paired_returns, paired_weights = (
    evaluate(
        model,
        paired_test,
        quiet=True,
    )
)


print("Evaluating broad persistent attack...")

attack_returns, attack_weights = (
    evaluate(
        model,
        attack_test,
        quiet=True,
    )
)


original_perf = performance_metrics(
    original_returns
)

paired_perf = performance_metrics(
    paired_returns
)

attack_perf = performance_metrics(
    attack_returns
)


# ============================================================
# POLICY SHIFTS
# ============================================================

attack_shift, attack_by_asset = (
    policy_distance(
        paired_weights,
        attack_weights,
    )
)


regen_shift, regen_by_asset = (
    policy_distance(
        original_weights,
        paired_weights,
    )
)


# ============================================================
# SAVE
# ============================================================

original_returns.to_csv(
    RESULT_DIR
    / "original_clean_returns.csv",
    index=False,
)

paired_returns.to_csv(
    RESULT_DIR
    / "paired_clean_returns.csv",
    index=False,
)

attack_returns.to_csv(
    RESULT_DIR
    / "attack_returns.csv",
    index=False,
)


original_weights.to_csv(
    RESULT_DIR
    / "original_clean_weights.csv",
)

paired_weights.to_csv(
    RESULT_DIR
    / "paired_clean_weights.csv",
)

attack_weights.to_csv(
    RESULT_DIR
    / "attack_weights.csv",
)


attack_by_asset.to_csv(
    RESULT_DIR
    / "attack_policy_shift_by_asset.csv",
    index=False,
)


regen_by_asset.to_csv(
    RESULT_DIR
    / "paired_regeneration_shift_by_asset.csv",
    index=False,
)


summary = {
    "target_year":
        TARGET_YEAR,

    "target_events":
        len(pairs),

    "changed_rag_test_rows":
        changed_rag_rows,

    "original_clean":
        original_perf,

    "paired_clean":
        paired_perf,

    "attack":
        attack_perf,

    "attack_minus_paired_clean":
        {
            "delta_final_multiple":
                attack_perf[
                    "final_multiple"
                ]
                -
                paired_perf[
                    "final_multiple"
                ],

            "delta_annual_return":
                attack_perf[
                    "annual_return"
                ]
                -
                paired_perf[
                    "annual_return"
                ],

            "delta_annual_volatility":
                attack_perf[
                    "annual_volatility"
                ]
                -
                paired_perf[
                    "annual_volatility"
                ],

            "delta_sharpe":
                attack_perf[
                    "sharpe"
                ]
                -
                paired_perf[
                    "sharpe"
                ],

            "delta_max_drawdown":
                attack_perf[
                    "max_drawdown"
                ]
                -
                paired_perf[
                    "max_drawdown"
                ],

            **attack_shift,
        },

    "original_minus_paired_regeneration_check":
        {
            "delta_final_multiple":
                original_perf[
                    "final_multiple"
                ]
                -
                paired_perf[
                    "final_multiple"
                ],

            **regen_shift,
        },
}


with open(
    RESULT_DIR
    / "normalized_broad_poison_summary.json",
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# REFERENCE CHECK AGAINST ORIGINAL NORMALIZED RUN
# ============================================================

expected_multiple = None


if MODEL_METRICS_PATH.exists():

    with open(
        MODEL_METRICS_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        old_metrics = json.load(f)


    expected_multiple = float(
        old_metrics[
            "test"
        ][
            "final_multiple"
        ]
    )


# ============================================================
# OLD UNNORMALIZED ATTACK REFERENCE
#
# Previous broad experiment:
# average daily L1 = 0.000528
# actual capital reallocated = L1 / 2 = 0.000264
# ============================================================

OLD_UNNORMALIZED_DAILY_L1 = (
    0.000528
)

OLD_UNNORMALIZED_REALLOCATED = (
    OLD_UNNORMALIZED_DAILY_L1
    / 2.0
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("NORMALIZED PORTFOLIO-WIDE PERSISTENT POISONING RESULT")
print("=" * 70)

print(
    "Target year:",
    TARGET_YEAR,
)

print(
    "Tickers:",
    ", ".join(
        TIC_ORDER
    ),
)

print(
    "Target events:",
    len(pairs),
)

print(
    "Changed normalized RAG test rows:",
    changed_rag_rows,
)


print("\nSIGNAL REPLAY")

print(
    "Mean impact delta:",
    f"{pairs['impact_delta'].mean():+.4f}",
)

print(
    "Mean |impact delta|:",
    f"{pairs['impact_delta'].abs().mean():.4f}",
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


print("\nORIGINAL NORMALIZED CLEAN")

print(
    "Multiple:",
    f"{original_perf['final_multiple']:.6f}",
)

print(
    "Sharpe:",
    f"{original_perf['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{original_perf['max_drawdown']:.2%}",
)


print("\nPAIRED NORMALIZED CLEAN")

print(
    "Multiple:",
    f"{paired_perf['final_multiple']:.6f}",
)

print(
    "Sharpe:",
    f"{paired_perf['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{paired_perf['max_drawdown']:.2%}",
)


print("\nNORMALIZED BROAD ATTACK")

print(
    "Multiple:",
    f"{attack_perf['final_multiple']:.6f}",
)

print(
    "Sharpe:",
    f"{attack_perf['sharpe']:.4f}",
)

print(
    "MDD:",
    f"{attack_perf['max_drawdown']:.2%}",
)


print("\nATTACK - PAIRED CLEAN")

print(
    "Δ Multiple:",
    f"{attack_perf['final_multiple'] - paired_perf['final_multiple']:+.6f}",
)

print(
    "Approx portfolio value difference:",
    f"${INITIAL_AMOUNT * (attack_perf['final_multiple'] - paired_perf['final_multiple']):+,.2f}",
)

print(
    "Δ Annual return:",
    f"{attack_perf['annual_return'] - paired_perf['annual_return']:+.4%}",
)

print(
    "Δ Volatility:",
    f"{attack_perf['annual_volatility'] - paired_perf['annual_volatility']:+.4%}",
)

print(
    "Δ Sharpe:",
    f"{attack_perf['sharpe'] - paired_perf['sharpe']:+.6f}",
)

print(
    "Δ MDD:",
    f"{attack_perf['max_drawdown'] - paired_perf['max_drawdown']:+.4%}",
)


print("\nPOLICY SHIFT: ATTACK VS PAIRED CLEAN")

print(
    "Mean abs weight diff:",
    f"{attack_shift['mean_abs_weight_diff']:.6f}",
)

print(
    "Average daily L1:",
    f"{attack_shift['average_daily_l1']:.6f}",
)

print(
    "Average capital reallocated:",
    f"{attack_shift['average_reallocated_weight']:.4%}",
)

print(
    "Maximum daily L1:",
    f"{attack_shift['maximum_daily_l1']:.6f}",
)

print(
    "Maximum capital reallocated:",
    f"{attack_shift['maximum_reallocated_weight']:.4%}",
)

print(
    "Maximum single-asset weight difference:",
    f"{attack_shift['maximum_single_asset_diff']:.4%}",
)


print("\nMEAN ABS WEIGHT DIFFERENCE BY ASSET")

print(
    attack_by_asset.to_string(
        index=False
    )
)


print("\nPAIRED REGENERATION CHECK")

print(
    "Original clean multiple:",
    f"{original_perf['final_multiple']:.6f}",
)

print(
    "Paired clean multiple:",
    f"{paired_perf['final_multiple']:.6f}",
)

print(
    "Original vs paired average daily L1:",
    f"{regen_shift['average_daily_l1']:.6f}",
)

print(
    "Original vs paired capital reallocated:",
    f"{regen_shift['average_reallocated_weight']:.4%}",
)


if expected_multiple is not None:

    drift = (
        original_perf[
            "final_multiple"
        ]
        -
        expected_multiple
    )

    print("\nNORMALIZED MODEL REPLAY CHECK")

    print(
        "Original normalized experiment multiple:",
        f"{expected_multiple:.6f}",
    )

    print(
        "Current replay multiple:",
        f"{original_perf['final_multiple']:.6f}",
    )

    print(
        "Replay drift:",
        f"{drift:+.8f}",
    )


print("\nOLD VS NORMALIZED ATTACK PROPAGATION")

print(
    "Old unnormalized average daily L1:",
    f"{OLD_UNNORMALIZED_DAILY_L1:.6f}",
)

print(
    "Old unnormalized capital reallocated:",
    f"{OLD_UNNORMALIZED_REALLOCATED:.4%}",
)

print(
    "New normalized average daily L1:",
    f"{attack_shift['average_daily_l1']:.6f}",
)

print(
    "New normalized capital reallocated:",
    f"{attack_shift['average_reallocated_weight']:.4%}",
)


if OLD_UNNORMALIZED_DAILY_L1 > 0:

    print(
        "Action-propagation amplification ratio:",
        f"{attack_shift['average_daily_l1'] / OLD_UNNORMALIZED_DAILY_L1:.2f}x",
    )


print("\nResults saved to:")

print(
    RESULT_DIR
)

print("\nDONE")