from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import copy

import numpy as np
import pandas as pd

from .features import add_technical_features


RAG_COLS = ["impact", "risk", "confidence"]


# ============================================================
# KG feature selection
# ============================================================

def get_kg_cols(df: pd.DataFrame) -> List[str]:
    """
    Select FinReflectKG features used by the policy.

    Included:
      - kg_rel_* normalized relation-frequency features
      - log1p graph statistics
      - kg_available availability indicator

    Excluded:
      - raw count features such as kg_total_triples
        because their scale is much larger.
    """

    cols = []

    structural_cols = {
        "kg_total_triples_log1p",
        "kg_unique_entities_log1p",
        "kg_unique_targets_log1p",
        "kg_unique_relations_log1p",
        "kg_available",
    }

    for c in df.columns:
        if c.startswith("kg_rel_"):
            cols.append(c)
        elif c in structural_cols:
            cols.append(c)

    return sorted(cols)


# ============================================================
# Data containers
# ============================================================

@dataclass
class FeatureScaler:
    tech_mean: np.ndarray
    tech_std: np.ndarray

    rag_mean: np.ndarray
    rag_std: np.ndarray

    kg_mean: np.ndarray
    kg_std: np.ndarray


@dataclass
class PortfolioData:
    dates: pd.DatetimeIndex
    tickers: List[str]

    # [T, N]
    close: np.ndarray

    # normalized technical observation
    # [T, N, F_tech]
    tech: np.ndarray

    # normalized SEC-RAG observation
    # [T, N, 3]
    rag_state: np.ndarray

    # raw SEC-RAG used by gate / veto
    # [T, N, 3]
    rag_raw: np.ndarray

    # normalized FinReflectKG observation
    # [T, N, F_kg]
    kg_state: np.ndarray

    tech_names: List[str]
    rag_names: List[str]
    kg_names: List[str]


# ============================================================
# Market data
# ============================================================

def load_market_csv(
    path: str,
    tickers: Optional[List[str]] = None,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"market.csv missing columns: {sorted(missing)}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    if tickers:
        tickers = [
            str(t).upper().strip()
            for t in tickers
        ]

        df = df[
            df["ticker"].isin(tickers)
        ].copy()

    if df.empty:
        raise ValueError(
            "No market rows after ticker filtering."
        )

    return add_technical_features(df)


# ============================================================
# SEC-RAG
# ============================================================

def load_rag_csv(
    path: str,
    available_date_col: str = "available_date",
) -> pd.DataFrame:

    rag = pd.read_csv(path)

    required = {
        "ticker",
        available_date_col,
        *RAG_COLS,
    }

    missing = required - set(rag.columns)

    if missing:
        raise ValueError(
            f"rag_signals.csv missing columns: {sorted(missing)}"
        )

    rag = rag.rename(
        columns={
            available_date_col: "available_date"
        }
    ).copy()

    rag["available_date"] = pd.to_datetime(
        rag["available_date"]
    )

    rag["ticker"] = (
        rag["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    rag["impact"] = (
        pd.to_numeric(
            rag["impact"],
            errors="coerce",
        )
        .clip(-1.0, 1.0)
    )

    rag["risk"] = (
        pd.to_numeric(
            rag["risk"],
            errors="coerce",
        )
        .clip(0.0, 1.0)
    )

    rag["confidence"] = (
        pd.to_numeric(
            rag["confidence"],
            errors="coerce",
        )
        .clip(0.0, 1.0)
    )

    rag = (
        rag
        .sort_values(
            ["ticker", "available_date"]
        )
        .drop_duplicates(
            ["ticker", "available_date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return rag


def attach_rag_asof(
    market: pd.DataFrame,
    rag: pd.DataFrame,
) -> pd.DataFrame:
    """
    Anti-leakage rule:

    each trading row receives only the latest
    RAG signal with

        available_date <= trading date

    for the same ticker.
    """

    pieces = []

    for ticker, m in market.groupby(
        "ticker",
        sort=False,
    ):

        m = (
            m
            .sort_values("date")
            .copy()
        )

        r = (
            rag[
                rag["ticker"] == ticker
            ]
            .sort_values("available_date")
            .copy()
        )

        if r.empty:

            for c in RAG_COLS:
                m[c] = 0.0

            m["source_available_date"] = pd.NaT

        else:

            r = r[
                [
                    "available_date",
                    *RAG_COLS,
                ]
            ].copy()

            merged = pd.merge_asof(
                m.sort_values("date"),
                r.sort_values(
                    "available_date"
                ),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )

            merged = merged.rename(
                columns={
                    "available_date":
                    "source_available_date"
                }
            )

            for c in RAG_COLS:
                merged[c] = (
                    merged[c]
                    .fillna(0.0)
                )

            m = merged

        pieces.append(m)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


# ============================================================
# FinReflectKG
# ============================================================

def load_kg_csv(
    path: str,
) -> tuple[pd.DataFrame, List[str]]:

    kg = pd.read_csv(path)

    required = {
        "date",
        "ticker",
    }

    missing = required - set(kg.columns)

    if missing:
        raise ValueError(
            f"KG csv missing columns: {sorted(missing)}"
        )

    kg["date"] = pd.to_datetime(
        kg["date"]
    )

    kg["ticker"] = (
        kg["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    kg_cols = get_kg_cols(kg)

    if not kg_cols:
        raise ValueError(
            "No usable KG features found.\n"
            "Expected kg_rel_* and/or "
            "*_log1p / kg_available columns."
        )

    for c in kg_cols:
        kg[c] = (
            pd.to_numeric(
                kg[c],
                errors="coerce",
            )
            .fillna(0.0)
        )

    kg = (
        kg[
            [
                "date",
                "ticker",
                *kg_cols,
            ]
        ]
        .drop_duplicates(
            ["date", "ticker"],
            keep="last",
        )
        .sort_values(
            ["date", "ticker"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(kg_cols)} "
        "FinReflectKG state features."
    )

    return kg, kg_cols


def attach_kg(
    panel: pd.DataFrame,
    kg: pd.DataFrame,
    kg_cols: List[str],
) -> pd.DataFrame:

    out = panel.merge(
        kg,
        how="left",
        on=["date", "ticker"],
        validate="one_to_one",
    )

    for c in kg_cols:
        out[c] = (
            out[c]
            .fillna(0.0)
        )

    return out


# ============================================================
# Complete portfolio panel
# ============================================================

def _complete_panel(
    df: pd.DataFrame,
    tickers: List[str],
    required_cols: List[str],
) -> pd.DataFrame:

    df = df[
        df["ticker"].isin(tickers)
    ].copy()

    good = df.dropna(
        subset=[
            "close",
            *required_cols,
        ]
    ).copy()

    counts = (
        good
        .groupby("date")["ticker"]
        .nunique()
    )

    valid_dates = counts[
        counts == len(tickers)
    ].index

    good = good[
        good["date"].isin(valid_dates)
    ].copy()

    good = good.drop_duplicates(
        ["date", "ticker"],
        keep="last",
    )

    return (
        good
        .sort_values(
            ["date", "ticker"]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Scaler
# ============================================================

def fit_scaler(
    panel_df: pd.DataFrame,
    tech_names: List[str],
    kg_names: List[str],
    train_start: str,
    train_end: str,
    normalize_technical: bool = True,
    normalize_rag: bool = True,
    normalize_kg: bool = True,
) -> FeatureScaler:

    mask = (
        (
            panel_df["date"]
            >= pd.Timestamp(train_start)
        )
        &
        (
            panel_df["date"]
            <= pd.Timestamp(train_end)
        )
    )

    tr = panel_df.loc[mask]

    if tr.empty:
        raise ValueError(
            "Training date range contains no rows."
        )

    tech = tr[
        tech_names
    ].to_numpy(dtype=np.float64)

    rag = tr[
        RAG_COLS
    ].to_numpy(dtype=np.float64)

    kg = tr[
        kg_names
    ].to_numpy(dtype=np.float64)


    # --------------------------------------------------------
    # Technical
    # --------------------------------------------------------

    if normalize_technical:

        tmean = np.nanmean(
            tech,
            axis=0,
        )

        tstd = np.nanstd(
            tech,
            axis=0,
        )

        tstd[
            tstd < 1e-8
        ] = 1.0

    else:

        tmean = np.zeros(
            len(tech_names)
        )

        tstd = np.ones(
            len(tech_names)
        )


    # --------------------------------------------------------
    # SEC-RAG
    # --------------------------------------------------------

    if normalize_rag:

        rmean = np.nanmean(
            rag,
            axis=0,
        )

        rstd = np.nanstd(
            rag,
            axis=0,
        )

        rstd[
            rstd < 1e-8
        ] = 1.0

    else:

        rmean = np.zeros(
            len(RAG_COLS)
        )

        rstd = np.ones(
            len(RAG_COLS)
        )


    # --------------------------------------------------------
    # FinReflectKG
    # --------------------------------------------------------

    if normalize_kg:

        kmean = np.nanmean(
            kg,
            axis=0,
        )

        kstd = np.nanstd(
            kg,
            axis=0,
        )

        kstd[
            kstd < 1e-8
        ] = 1.0

    else:

        kmean = np.zeros(
            len(kg_names)
        )

        kstd = np.ones(
            len(kg_names)
        )


    # kg_available should stay 0 / 1,
    # not become a z-score.
    if "kg_available" in kg_names:

        idx = kg_names.index(
            "kg_available"
        )

        kmean[idx] = 0.0
        kstd[idx] = 1.0


    return FeatureScaler(
        tech_mean=tmean,
        tech_std=tstd,
        rag_mean=rmean,
        rag_std=rstd,
        kg_mean=kmean,
        kg_std=kstd,
    )


# ============================================================
# Convert dataframe -> RL tensor data
# ============================================================

def dataframe_to_portfolio_data(
    panel_df: pd.DataFrame,
    tickers: List[str],
    tech_names: List[str],
    kg_names: List[str],
    scaler: FeatureScaler,
    start: str,
    end: str,
) -> PortfolioData:

    df = panel_df[
        (
            panel_df["date"]
            >= pd.Timestamp(start)
        )
        &
        (
            panel_df["date"]
            <= pd.Timestamp(end)
        )
    ].copy()

    if df.empty:
        raise ValueError(
            f"No rows in requested date range "
            f"{start} .. {end}"
        )

    dates = pd.DatetimeIndex(
        sorted(
            df["date"].unique()
        )
    )

    N = len(tickers)
    T = len(dates)

    F = len(tech_names)
    K = len(kg_names)

    close = np.zeros(
        (T, N),
        dtype=np.float32,
    )

    tech = np.zeros(
        (T, N, F),
        dtype=np.float32,
    )

    rag_raw = np.zeros(
        (T, N, 3),
        dtype=np.float32,
    )

    kg_raw = np.zeros(
        (T, N, K),
        dtype=np.float32,
    )


    ticker_to_j = {
        t: j
        for j, t in enumerate(tickers)
    }

    date_to_i = {
        pd.Timestamp(d): i
        for i, d in enumerate(dates)
    }


    for row in df.itertuples(
        index=False
    ):

        i = date_to_i[
            pd.Timestamp(row.date)
        ]

        j = ticker_to_j[
            row.ticker
        ]

        close[i, j] = float(
            row.close
        )

        tech[i, j] = np.array(
            [
                getattr(row, c)
                for c in tech_names
            ],
            dtype=np.float32,
        )

        rag_raw[i, j] = np.array(
            [
                row.impact,
                row.risk,
                row.confidence,
            ],
            dtype=np.float32,
        )

        kg_raw[i, j] = np.array(
            [
                getattr(row, c)
                for c in kg_names
            ],
            dtype=np.float32,
        )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    tech = (
        tech
        - scaler.tech_mean.reshape(
            1,
            1,
            -1,
        )
    ) / scaler.tech_std.reshape(
        1,
        1,
        -1,
    )


    rag_state = (
        rag_raw
        - scaler.rag_mean.reshape(
            1,
            1,
            -1,
        )
    ) / scaler.rag_std.reshape(
        1,
        1,
        -1,
    )


    kg_state = (
        kg_raw
        - scaler.kg_mean.reshape(
            1,
            1,
            -1,
        )
    ) / scaler.kg_std.reshape(
        1,
        1,
        -1,
    )


    # --------------------------------------------------------
    # Important:
    # missing historical KG should be a neutral zero vector.
    #
    # Example:
    # 2018 currently has kg_available = 0
    # because our downloaded KG starts in 2018 and
    # trading year Y uses Y-1 KG.
    # --------------------------------------------------------

    if "kg_available" in kg_names:

        available_idx = kg_names.index(
            "kg_available"
        )

        available = (
            kg_raw[
                :,
                :,
                available_idx,
            ]
            > 0.5
        )

        # unavailable KG => neutral observation
        kg_state[
            ~available
        ] = 0.0

        # restore explicit 0/1 flag
        kg_state[
            :,
            :,
            available_idx,
        ] = kg_raw[
            :,
            :,
            available_idx,
        ]


    # --------------------------------------------------------
    # Safety cleanup
    # --------------------------------------------------------

    tech = np.nan_to_num(
        tech,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    rag_state = np.nan_to_num(
        rag_state,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    rag_raw = np.nan_to_num(
        rag_raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    kg_state = np.nan_to_num(
        kg_state,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


    return PortfolioData(
        dates=dates,
        tickers=tickers,
        close=close,
        tech=tech,
        rag_state=rag_state,
        rag_raw=rag_raw,
        kg_state=kg_state,
        tech_names=tech_names,
        rag_names=RAG_COLS.copy(),
        kg_names=kg_names.copy(),
    )


# ============================================================
# Main data preparation
# ============================================================

def prepare_data(cfg: dict):

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]


    # --------------------------------------------------------
    # Market
    # --------------------------------------------------------

    market = load_market_csv(
        data_cfg["market_csv"],
        data_cfg.get("tickers"),
    )

    tickers = (
        data_cfg.get("tickers")
        or sorted(
            market[
                "ticker"
            ]
            .unique()
            .tolist()
        )
    )

    tickers = [
        str(t).upper().strip()
        for t in tickers
    ]


    # --------------------------------------------------------
    # SEC-RAG
    # --------------------------------------------------------

    rag = load_rag_csv(
        data_cfg["rag_csv"],
        data_cfg.get(
            "rag_available_date_col",
            "available_date",
        ),
    )

    merged = attach_rag_asof(
        market,
        rag,
    )


    # --------------------------------------------------------
    # FinReflectKG
    # --------------------------------------------------------

    kg, kg_names = load_kg_csv(
        data_cfg["kg_csv"]
    )

    merged = attach_kg(
        merged,
        kg,
        kg_names,
    )


    # --------------------------------------------------------
    # Technical features
    # --------------------------------------------------------

    tech_names = list(
        feat_cfg["technical"]
    )

    panel = _complete_panel(
        merged,
        tickers,
        tech_names,
    )


    # --------------------------------------------------------
    # Fit scaler ONLY on training period
    # --------------------------------------------------------

    scaler = fit_scaler(
        panel,
        tech_names,
        kg_names,
        data_cfg["train_start"],
        data_cfg["train_end"],
        normalize_technical=feat_cfg.get(
            "normalize_technical",
            True,
        ),
        normalize_rag=feat_cfg.get(
            "normalize_rag_for_state",
            True,
        ),
        normalize_kg=feat_cfg.get(
            "normalize_kg_for_state",
            True,
        ),
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train = dataframe_to_portfolio_data(
        panel,
        tickers,
        tech_names,
        kg_names,
        scaler,
        data_cfg["train_start"],
        data_cfg["train_end"],
    )


    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test = dataframe_to_portfolio_data(
        panel,
        tickers,
        tech_names,
        kg_names,
        scaler,
        data_cfg["test_start"],
        data_cfg["test_end"],
    )


    print("\n===== PREPARED DATA =====")

    print(
        "Train technical:",
        train.tech.shape,
    )

    print(
        "Train RAG:",
        train.rag_state.shape,
    )

    print(
        "Train KG:",
        train.kg_state.shape,
    )

    print(
        "Test technical:",
        test.tech.shape,
    )

    print(
        "Test RAG:",
        test.rag_state.shape,
    )

    print(
        "Test KG:",
        test.kg_state.shape,
    )

    print(
        "KG feature count:",
        len(kg_names),
    )


    return (
        train,
        test,
        scaler,
        panel,
    )


# ============================================================
# Permutation attribution
# ============================================================

def permute_observation_feature(
    data: PortfolioData,
    feature_group: str,
    feature_index: int,
    seed: int,
    permute_gate_signal: bool = False,
) -> PortfolioData:
    """
    Permute one feature across TIME while preserving asset identity.

    Supported groups:
      technical
      rag
      kg

    For RAG:
      default changes policy observation only.
      If permute_gate_signal=True, raw gate signal is also
      permuted.
    """

    out = copy.deepcopy(data)

    rng = np.random.default_rng(
        seed
    )

    perm = rng.permutation(
        len(out.dates)
    )


    if feature_group == "technical":

        out.tech[
            :,
            :,
            feature_index,
        ] = out.tech[
            perm,
            :,
            feature_index,
        ]


    elif feature_group == "rag":

        out.rag_state[
            :,
            :,
            feature_index,
        ] = out.rag_state[
            perm,
            :,
            feature_index,
        ]

        if permute_gate_signal:

            out.rag_raw[
                :,
                :,
                feature_index,
            ] = out.rag_raw[
                perm,
                :,
                feature_index,
            ]


    elif feature_group == "kg":

        out.kg_state[
            :,
            :,
            feature_index,
        ] = out.kg_state[
            perm,
            :,
            feature_index,
        ]


    else:

        raise ValueError(
            "feature_group must be "
            "'technical', 'rag', or 'kg'"
        )


    return out
