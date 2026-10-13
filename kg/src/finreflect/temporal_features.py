from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_temporal_kg_features(
    path: str | Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    df["published_at"] = (
        pd.to_datetime(
            df["published_at"]
        )
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
    )

    return df


def merge_temporal_kg_features(
    market_df: pd.DataFrame,
    kg_df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "tic",
) -> pd.DataFrame:

    market = (
        market_df
        .copy()
    )

    kg = (
        kg_df
        .copy()
    )

    market[date_col] = (
        pd.to_datetime(
            market[date_col]
        )
    )

    market[ticker_col] = (
        market[ticker_col]
        .astype(str)
        .str.upper()
    )

    kg["published_at"] = (
        pd.to_datetime(
            kg["published_at"]
        )
    )

    kg["ticker"] = (
        kg["ticker"]
        .astype(str)
        .str.upper()
    )

    pieces = []

    # merge_asof per ticker is less error-prone
    # than mixing all tickers in one call.
    for ticker, market_part in (
        market.groupby(
            ticker_col,
            sort=False,
        )
    ):

        m = (
            market_part
            .sort_values(
                date_col
            )
            .copy()
        )

        k = (
            kg[
                kg["ticker"]
                == ticker
            ]
            .sort_values(
                "published_at"
            )
            .copy()
        )

        if k.empty:

            pieces.append(
                m
            )

            continue

        merged = pd.merge_asof(
            m,
            k,
            left_on=date_col,
            right_on="published_at",
            direction="backward",
            allow_exact_matches=False,
        )

        pieces.append(
            merged
        )

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    kg_cols = [
        c
        for c in kg.columns
        if c.startswith("kg_")
    ]

    for c in kg_cols:

        if c not in out.columns:
            out[c] = 0.0

        out[c] = (
            pd.to_numeric(
                out[c],
                errors="coerce",
            )
            .fillna(0.0)
        )

    return (
        out
        .sort_values(
            [
                date_col,
                ticker_col,
            ]
        )
        .reset_index(
            drop=True
        )
    )
