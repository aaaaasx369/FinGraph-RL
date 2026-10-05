from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FINDKG_PATH = (
    ROOT
    / "kg/data/findkg/train2020/"
      "findkg_finrl_daily_2018_2022.csv"
)

OUT_PATH = (
    ROOT
    / "kg/data/findkg/train2020/"
      "fourway_finrl_daily_2018_2022.csv"
)


BASE_COLS = [
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


SEC_COLS = [
    "sec_chunks_30d",
    "sec_activity_30d",
    "latest_sec_embedding_norm",
    "impact",
    "risk",
    "confidence",
    "sec_signal_age_days",
]


def normalize_ticker(df):
    if "ticker" not in df.columns and "tic" in df.columns:
        df = df.rename(columns={"tic": "ticker"})

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


def find_market_file():
    candidates = [
        ROOT / "data/portfolio_data.csv",
        ROOT / "data/processed_data.csv",
        ROOT / "data/processed/portfolio_data.csv",
        ROOT / "data/processed/processed_data.csv",
    ]

    for p in candidates:
        if p.exists():
            df = pd.read_csv(p, nrows=5)

            cols = {c.lower() for c in df.columns}

            has_date = "date" in cols
            has_ticker = (
                "ticker" in cols
                or "tic" in cols
            )
            has_close = any(
                x in cols
                for x in [
                    "close",
                    "adj_close",
                    "adjclose",
                    "adjusted_close",
                ]
            )

            if has_date and has_ticker and has_close:
                return p

    # fallback: find a CSV with date/ticker/close
    for p in ROOT.rglob("*.csv"):
        try:
            df = pd.read_csv(p, nrows=3)
        except Exception:
            continue

        cols = {c.lower() for c in df.columns}

        if (
            "date" in cols
            and (
                "ticker" in cols
                or "tic" in cols
            )
            and any(
                x in cols
                for x in [
                    "close",
                    "adj_close",
                    "adjclose",
                    "adjusted_close",
                ]
            )
        ):
            # Do not accidentally use our feature tables
            if (
                "findkg_emb_000" not in df.columns
                and "impact" not in df.columns
            ):
                return p

    raise FileNotFoundError(
        "Could not automatically locate market CSV."
    )


def get_close_col(df):
    for c in [
        "close",
        "adj_close",
        "adjclose",
        "adjusted_close",
    ]:
        if c in df.columns:
            return c

    raise ValueError("No close-price column found.")


def add_technical_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])
    df = normalize_ticker(df)

    close_col = get_close_col(df)

    df = df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)

    pieces = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):
        g = g.copy()

        close = pd.to_numeric(
            g[close_col],
            errors="coerce",
        )

        # --------------------------
        # Returns
        # --------------------------
        g["ret_1"] = close.pct_change(1)
        g["ret_5"] = close.pct_change(5)
        g["ret_20"] = close.pct_change(20)

        # --------------------------
        # 20-day realized vol
        # strictly historical/current information only
        # --------------------------
        g["vol_20"] = (
            g["ret_1"]
            .rolling(
                20,
                min_periods=20,
            )
            .std()
        )

        # --------------------------
        # SMA ratios
        # --------------------------
        sma10 = close.rolling(
            10,
            min_periods=10,
        ).mean()

        sma20 = close.rolling(
            20,
            min_periods=20,
        ).mean()

        g["sma_10_ratio"] = (
            close / sma10 - 1.0
        )

        g["sma_20_ratio"] = (
            close / sma20 - 1.0
        )

        # --------------------------
        # RSI(14)
        # --------------------------
        delta = close.diff()

        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        ).mean()

        avg_loss = loss.ewm(
            alpha=1.0 / 14.0,
            adjust=False,
            min_periods=14,
        ).mean()

        rs = avg_gain / avg_loss.replace(
            0.0,
            np.nan,
        )

        g["rsi_14"] = (
            100.0
            - 100.0 / (1.0 + rs)
        )

        # --------------------------
        # MACD 12/26, signal 9
        # --------------------------
        ema12 = close.ewm(
            span=12,
            adjust=False,
        ).mean()

        ema26 = close.ewm(
            span=26,
            adjust=False,
        ).mean()

        macd = ema12 - ema26

        g["macd"] = macd

        g["macd_signal"] = macd.ewm(
            span=9,
            adjust=False,
        ).mean()

        pieces.append(g)

    return pd.concat(
        pieces,
        ignore_index=True,
    )


def main():
    print("=" * 72)
    print("BUILD 4-WAY LEAKAGE-CONTROLLED DATASET")
    print("=" * 72)

    print("\nFinDKG/SEC source:")
    print(FINDKG_PATH)

    feature_df = pd.read_csv(
        FINDKG_PATH
    )

    feature_df = normalize_ticker(
        feature_df
    )

    feature_df["date"] = pd.to_datetime(
        feature_df["date"]
    )

    print("\nFinding clean market data...")

    market_path = find_market_file()

    print("Market file:")
    print(market_path)

    market = pd.read_csv(
        market_path
    )

    market = add_technical_features(
        market
    )

    tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
        "GOOGL",
    ]

    market = market[
        market["ticker"].isin(tickers)
    ].copy()

    tech = market[
        ["date", "ticker"] + BASE_COLS
    ].copy()

    out = feature_df.merge(
        tech,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    # Dataset period used in this experiment
    out = out[
        (
            out["date"]
            >= pd.Timestamp("2018-01-08")
        )
        &
        (
            out["date"]
            <= pd.Timestamp("2022-12-30")
        )
    ].copy()

    # Reassign experiment split explicitly
    out["split"] = np.where(
        out["date"]
        <= pd.Timestamp("2020-12-31"),
        "train",
        "test",
    )

    findkg_cols = sorted(
        [
            c
            for c in out.columns
            if c.startswith("findkg_emb_")
        ]
    )

    print("\n" + "=" * 72)
    print("FEATURE GROUPS")
    print("=" * 72)

    print("BASE:", len(BASE_COLS))
    print(BASE_COLS)

    print("\nSEC:", len(SEC_COLS))
    print(SEC_COLS)

    print(
        "\nFinDKG:",
        len(findkg_cols),
    )

    if len(findkg_cols) != 200:
        raise ValueError(
            f"Expected 200 FinDKG dimensions, "
            f"found {len(findkg_cols)}"
        )

    missing_base = [
        c
        for c in BASE_COLS
        if c not in out.columns
    ]

    missing_sec = [
        c
        for c in SEC_COLS
        if c not in out.columns
    ]

    if missing_base:
        raise ValueError(
            f"Missing BASE columns: "
            f"{missing_base}"
        )

    if missing_sec:
        raise ValueError(
            f"Missing SEC columns: "
            f"{missing_sec}"
        )

    print("\n" + "=" * 72)
    print("FOUR-WAY DEFINITIONS")
    print("=" * 72)

    groups = {
        "BASE":
            BASE_COLS,

        "BASE_SEC":
            BASE_COLS
            + SEC_COLS,

        "BASE_FINDKG":
            BASE_COLS
            + findkg_cols,

        "BASE_SEC_FINDKG":
            BASE_COLS
            + SEC_COLS
            + findkg_cols,
    }

    for name, cols in groups.items():
        print(
            f"{name:20s}: "
            f"{len(cols)} dims"
        )

    print("\n" + "=" * 72)
    print("QA")
    print("=" * 72)

    print(
        "rows:",
        len(out),
    )

    print(
        "date:",
        out["date"].min().date(),
        "->",
        out["date"].max().date(),
    )

    print("\nsplit counts:")
    print(
        out["split"]
        .value_counts()
        .sort_index()
    )

    print("\nrows per ticker:")
    print(
        out.groupby("ticker")
        .size()
    )

    print(
        "\nduplicates:",
        out.duplicated(
            ["date", "ticker"]
        ).sum(),
    )

    for name, cols in groups.items():
        n_missing = (
            out[cols]
            .isna()
            .sum()
            .sum()
        )

        print(
            f"{name:20s} "
            f"missing values: {n_missing}"
        )

    # Remove initial rows where rolling indicators
    # cannot yet be computed.
    before = len(out)

    out = out.dropna(
        subset=BASE_COLS
    ).copy()

    print(
        "\nRows removed for initial "
        "technical warm-up:",
        before - len(out),
    )

    print(
        "final rows:",
        len(out),
    )

    print(
        "final date:",
        out["date"].min().date(),
        "->",
        out["date"].max().date(),
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out["date"] = (
        out["date"]
        .dt.strftime("%Y-%m-%d")
    )

    out.to_csv(
        OUT_PATH,
        index=False,
    )

    print("\nSAVED:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
