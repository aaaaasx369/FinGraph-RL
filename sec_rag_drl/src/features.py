from __future__ import annotations
import numpy as np
import pandas as pd


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    avg_down = down.ewm(alpha=1/window, adjust=False, min_periods=window).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns required: date, ticker, close.
    Features use current/past data only; no centered windows.
    """
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"])
    x = x.sort_values(["ticker", "date"]).reset_index(drop=True)

    out = []
    for ticker, g in x.groupby("ticker", sort=False):
        g = g.copy()
        c = g["close"].astype(float)

        g["ret_1"] = c.pct_change(1)
        g["ret_5"] = c.pct_change(5)
        g["ret_20"] = c.pct_change(20)
        g["vol_20"] = g["ret_1"].rolling(20, min_periods=10).std()

        sma10 = c.rolling(10, min_periods=5).mean()
        sma20 = c.rolling(20, min_periods=10).mean()
        g["sma_10_ratio"] = c / sma10 - 1.0
        g["sma_20_ratio"] = c / sma20 - 1.0

        g["rsi_14"] = (_rsi(c, 14) - 50.0) / 50.0

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd_raw = ema12 - ema26
        macd_signal_raw = macd_raw.ewm(span=9, adjust=False).mean()

        # Scale MACD by price to make it more comparable across assets.
        g["macd"] = macd_raw / c.replace(0, np.nan)
        g["macd_signal"] = macd_signal_raw / c.replace(0, np.nan)
        out.append(g)

    return pd.concat(out, ignore_index=True)
