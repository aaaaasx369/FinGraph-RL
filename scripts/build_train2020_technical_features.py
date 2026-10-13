from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

INPUT = Path(
    "kg/data/market_5stocks_2017_2026.csv"
)

OUTPUT = Path(
    "kg/data/findkg/train2020/"
    "technical_9_daily_2018_2022.csv"
)


def find_close_col(df):

    for c in [
        "close",
        "adj_close",
        "adjclose",
        "adjusted_close",
        "adjcp",
        "Close",
    ]:

        if c in df.columns:
            return c

    raise ValueError(
        "No close-price column found."
    )


def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = (
        -delta.clip(
            upper=0
        )
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = (
        avg_gain
        /
        avg_loss.replace(
            0,
            np.nan,
        )
    )

    out = (
        100
        -
        (
            100
            /
            (
                1 + rs
            )
        )
    )

    return out


df = pd.read_csv(
    INPUT
)

if "ticker" not in df.columns:

    if "tic" in df.columns:

        df = df.rename(
            columns={
                "tic":
                    "ticker"
            }
        )

    else:

        raise ValueError(
            "No ticker/tic column."
        )


df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df[
    df["ticker"].isin(
        TICKERS
    )
].copy()


close_col = find_close_col(
    df
)


df[close_col] = pd.to_numeric(
    df[close_col],
    errors="coerce",
)


parts = []


for ticker, g in df.groupby(
    "ticker"
):

    g = (
        g
        .sort_values(
            "date"
        )
        .copy()
    )

    close = g[
        close_col
    ]


    g[
        "ret_1"
    ] = close.pct_change(
        1
    )

    g[
        "ret_5"
    ] = close.pct_change(
        5
    )

    g[
        "ret_20"
    ] = close.pct_change(
        20
    )


    daily_ret = (
        close.pct_change()
    )

    g[
        "vol_20"
    ] = (
        daily_ret
        .rolling(
            20
        )
        .std()
    )


    sma10 = close.rolling(
        10
    ).mean()

    sma20 = close.rolling(
        20
    ).mean()


    g[
        "sma_10_ratio"
    ] = (
        close
        /
        sma10
        -
        1
    )

    g[
        "sma_20_ratio"
    ] = (
        close
        /
        sma20
        -
        1
    )


    g[
        "rsi_14"
    ] = rsi(
        close,
        14,
    ) / 100.0


    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    macd = (
        ema12
        -
        ema26
    )

    signal = macd.ewm(
        span=9,
        adjust=False,
    ).mean()


    # Normalize MACD by price so scales are comparable.
    g[
        "macd"
    ] = (
        macd
        /
        close
    )

    g[
        "macd_signal"
    ] = (
        signal
        /
        close
    )


    parts.append(
        g[
            [
                "date",
                "ticker",
                "ret_1",
                "ret_5",
                "ret_20",
                "vol_20",
                "sma_10_ratio",
                "sma_20_ratio",
                "rsi_14",
                "macd",
                "macd_signal",
            ]
        ]
    )


out = pd.concat(
    parts,
    ignore_index=True,
)

out = out[
    (
        out["date"]
        >=
        pd.Timestamp(
            "2018-01-01"
        )
    )
    &
    (
        out["date"]
        <=
        pd.Timestamp(
            "2022-12-31"
        )
    )
].copy()


out = (
    out
    .sort_values(
        [
            "date",
            "ticker",
        ]
    )
    .reset_index(
        drop=True
    )
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTPUT,
    index=False,
)


print(
    "rows:",
    len(out),
)

print(
    "date range:",
    out["date"].min(),
    "->",
    out["date"].max(),
)

print(
    "tickers:",
    sorted(
        out["ticker"].unique()
    ),
)

print(
    "missing values:"
)

print(
    out.isna().sum()
)

print(
    "\nSaved:",
    OUTPUT,
)
