from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import copy
import numpy as np
import pandas as pd

from .features import add_technical_features


RAG_COLS = ["impact", "risk", "confidence"]


@dataclass
class FeatureScaler:
    tech_mean: np.ndarray
    tech_std: np.ndarray
    rag_mean: np.ndarray
    rag_std: np.ndarray


@dataclass
class PortfolioData:
    dates: pd.DatetimeIndex
    tickers: List[str]
    close: np.ndarray                  # [T, N]
    tech: np.ndarray                   # normalized observation technical [T, N, F]
    rag_state: np.ndarray              # normalized/raw observation RAG [T, N, 3]
    rag_raw: np.ndarray                # raw gate RAG [T, N, 3]
    tech_names: List[str]
    rag_names: List[str]


def load_market_csv(path: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"date", "ticker", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"market.csv missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if tickers:
        df = df[df["ticker"].isin(tickers)].copy()
    if df.empty:
        raise ValueError("No market rows after ticker filtering.")
    return add_technical_features(df)


def load_rag_csv(path: str, available_date_col: str = "available_date") -> pd.DataFrame:
    rag = pd.read_csv(path)
    required = {"ticker", available_date_col, *RAG_COLS}
    missing = required - set(rag.columns)
    if missing:
        raise ValueError(f"rag_signals.csv missing columns: {sorted(missing)}")

    rag = rag.rename(columns={available_date_col: "available_date"}).copy()
    rag["available_date"] = pd.to_datetime(rag["available_date"])
    rag["ticker"] = rag["ticker"].astype(str)

    rag["impact"] = pd.to_numeric(rag["impact"], errors="coerce").clip(-1.0, 1.0)
    rag["risk"] = pd.to_numeric(rag["risk"], errors="coerce").clip(0.0, 1.0)
    rag["confidence"] = pd.to_numeric(rag["confidence"], errors="coerce").clip(0.0, 1.0)

    rag = (
        rag.sort_values(["ticker", "available_date"])
        .drop_duplicates(["ticker", "available_date"], keep="last")
        .reset_index(drop=True)
    )
    return rag


def attach_rag_asof(market: pd.DataFrame, rag: pd.DataFrame) -> pd.DataFrame:
    """
    Critical anti-leakage rule:
    each trading row receives only the most recent RAG signal whose
    available_date <= trading date for the same ticker.
    """
    pieces = []
    for ticker, m in market.groupby("ticker", sort=False):
        m = m.sort_values("date").copy()
        r = rag[rag["ticker"] == ticker].sort_values("available_date").copy()

        if r.empty:
            for c in RAG_COLS:
                m[c] = 0.0
            m["source_available_date"] = pd.NaT
        else:
            r = r[["available_date", *RAG_COLS]].copy()
            merged = pd.merge_asof(
                m.sort_values("date"),
                r.sort_values("available_date"),
                left_on="date",
                right_on="available_date",
                direction="backward",
                allow_exact_matches=True,
            )
            merged = merged.rename(columns={"available_date": "source_available_date"})
            for c in RAG_COLS:
                merged[c] = merged[c].fillna(0.0)
            m = merged
        pieces.append(m)

    return pd.concat(pieces, ignore_index=True)


def _complete_panel(df: pd.DataFrame, tickers: List[str], required_cols: List[str]) -> pd.DataFrame:
    # Keep only dates for which every asset has a close and all required technical features.
    df = df[df["ticker"].isin(tickers)].copy()
    good = df.dropna(subset=["close", *required_cols]).copy()

    counts = good.groupby("date")["ticker"].nunique()
    valid_dates = counts[counts == len(tickers)].index
    good = good[good["date"].isin(valid_dates)].copy()

    # One row per date/ticker.
    good = good.drop_duplicates(["date", "ticker"], keep="last")
    return good.sort_values(["date", "ticker"]).reset_index(drop=True)


def fit_scaler(
    panel_df: pd.DataFrame,
    tech_names: List[str],
    train_start: str,
    train_end: str,
    normalize_technical: bool = True,
    normalize_rag: bool = True,
) -> FeatureScaler:
    mask = (
        (panel_df["date"] >= pd.Timestamp(train_start))
        & (panel_df["date"] <= pd.Timestamp(train_end))
    )
    tr = panel_df.loc[mask]
    if tr.empty:
        raise ValueError("Training date range contains no rows.")

    tech = tr[tech_names].to_numpy(dtype=np.float64)
    rag = tr[RAG_COLS].to_numpy(dtype=np.float64)

    if normalize_technical:
        tmean = np.nanmean(tech, axis=0)
        tstd = np.nanstd(tech, axis=0)
        tstd[tstd < 1e-8] = 1.0
    else:
        tmean = np.zeros(len(tech_names))
        tstd = np.ones(len(tech_names))

    if normalize_rag:
        rmean = np.nanmean(rag, axis=0)
        rstd = np.nanstd(rag, axis=0)
        rstd[rstd < 1e-8] = 1.0
    else:
        rmean = np.zeros(len(RAG_COLS))
        rstd = np.ones(len(RAG_COLS))

    return FeatureScaler(tmean, tstd, rmean, rstd)


def dataframe_to_portfolio_data(
    panel_df: pd.DataFrame,
    tickers: List[str],
    tech_names: List[str],
    scaler: FeatureScaler,
    start: str,
    end: str,
) -> PortfolioData:
    df = panel_df[
        (panel_df["date"] >= pd.Timestamp(start))
        & (panel_df["date"] <= pd.Timestamp(end))
    ].copy()
    if df.empty:
        raise ValueError(f"No rows in requested date range {start} .. {end}")

    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    N = len(tickers)
    T = len(dates)
    F = len(tech_names)

    close = np.zeros((T, N), dtype=np.float32)
    tech = np.zeros((T, N, F), dtype=np.float32)
    rag_raw = np.zeros((T, N, 3), dtype=np.float32)

    ticker_to_j = {t: j for j, t in enumerate(tickers)}
    date_to_i = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    for row in df.itertuples(index=False):
        i = date_to_i[pd.Timestamp(row.date)]
        j = ticker_to_j[row.ticker]
        close[i, j] = float(row.close)
        tech[i, j] = np.array([getattr(row, c) for c in tech_names], dtype=np.float32)
        rag_raw[i, j] = np.array([row.impact, row.risk, row.confidence], dtype=np.float32)

    tech = (tech - scaler.tech_mean.reshape(1, 1, -1)) / scaler.tech_std.reshape(1, 1, -1)
    rag_state = (rag_raw - scaler.rag_mean.reshape(1, 1, -1)) / scaler.rag_std.reshape(1, 1, -1)

    tech = np.nan_to_num(tech, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rag_state = np.nan_to_num(rag_state, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    rag_raw = np.nan_to_num(rag_raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    return PortfolioData(
        dates=dates,
        tickers=tickers,
        close=close,
        tech=tech,
        rag_state=rag_state,
        rag_raw=rag_raw,
        tech_names=tech_names,
        rag_names=RAG_COLS.copy(),
    )


def prepare_data(cfg: dict):
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]

    market = load_market_csv(data_cfg["market_csv"], data_cfg.get("tickers"))
    tickers = data_cfg.get("tickers") or sorted(market["ticker"].unique().tolist())

    rag = load_rag_csv(
        data_cfg["rag_csv"],
        data_cfg.get("rag_available_date_col", "available_date"),
    )
    merged = attach_rag_asof(market, rag)
    tech_names = list(feat_cfg["technical"])
    panel = _complete_panel(merged, tickers, tech_names)

    scaler = fit_scaler(
        panel,
        tech_names,
        data_cfg["train_start"],
        data_cfg["train_end"],
        normalize_technical=feat_cfg.get("normalize_technical", True),
        normalize_rag=feat_cfg.get("normalize_rag_for_state", True),
    )

    train = dataframe_to_portfolio_data(
        panel, tickers, tech_names, scaler,
        data_cfg["train_start"], data_cfg["train_end"]
    )
    test = dataframe_to_portfolio_data(
        panel, tickers, tech_names, scaler,
        data_cfg["test_start"], data_cfg["test_end"]
    )
    return train, test, scaler, panel


def permute_observation_feature(
    data: PortfolioData,
    feature_group: str,
    feature_index: int,
    seed: int,
    permute_gate_signal: bool = False,
) -> PortfolioData:
    """
    Permutes one feature across TIME, keeping asset identity fixed.

    By default RAG permutation changes only the policy observation (rag_state),
    not rag_raw used by the gate. That measures policy-level feature influence.
    Set permute_gate_signal=True to perturb the full hybrid system.
    """
    out = copy.deepcopy(data)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out.dates))

    if feature_group == "technical":
        out.tech[:, :, feature_index] = out.tech[perm, :, feature_index]
    elif feature_group == "rag":
        out.rag_state[:, :, feature_index] = out.rag_state[perm, :, feature_index]
        if permute_gate_signal:
            out.rag_raw[:, :, feature_index] = out.rag_raw[perm, :, feature_index]
    else:
        raise ValueError("feature_group must be 'technical' or 'rag'")
    return out
