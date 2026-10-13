#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(
    "outputs/finrl_dkg_pit_2022_2023/finrl_runs"
)

OUT = BASE / "merged_5seeds"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

CONDITIONS = [
    "CLEAN",
    "FULL_STATIC",
    "FULL_DYNAMIC_PLACEBO",
    "FULL_DYNAMIC",
]


def read_weights(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print(
        "Reading:",
        path.parent.name,
        "| columns:",
        list(df.columns),
    )

    date_col = next(
        (
            c for c in df.columns
            if c.lower() in [
                "date", "datetime", "timestamp"
            ]
        ),
        None,
    )

    if date_col is None:
        raise RuntimeError(
            f"No date column: {path}\n"
            f"Columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])

    # Case 1: Wide format
    # date,AAPL,MSFT,NVDA,AMZN,GOOGL
    ticker_cols = {
        c.upper(): c
        for c in df.columns
        if c.upper() in TICKERS
    }

    if set(ticker_cols) == set(TICKERS):
        result = df[
            [date_col] +
            [ticker_cols[t] for t in TICKERS]
        ].copy()

        result.columns = ["date"] + TICKERS

    else:
        # Case 2: Long format
        # date,ticker,weight
        ticker_col = next(
            (
                c for c in df.columns
                if c.lower() in [
                    "ticker", "tic", "asset"
                ]
            ),
            None,
        )

        weight_col = next(
            (
                c for c in df.columns
                if c.lower() in [
                    "weight",
                    "portfolio_weight",
                    "allocation",
                ]
            ),
            None,
        )

        if ticker_col is None or weight_col is None:
            raise RuntimeError(
                f"Unsupported weights format: {path}\n"
                f"Columns: {list(df.columns)}\n"
                f"Sample:\n{df.head(3)}"
            )

        df[ticker_col] = (
            df[ticker_col]
            .astype(str)
            .str.upper()
        )

        result = df.pivot(
            index=date_col,
            columns=ticker_col,
            values=weight_col,
        ).reset_index()

        result = result.rename(
            columns={date_col: "date"}
        )

        if not set(TICKERS).issubset(result.columns):
            raise RuntimeError(
                f"Missing ticker weights: {path}"
            )

        result = result[
            ["date"] + TICKERS
        ]

    result = (
        result.sort_values("date")
        .reset_index(drop=True)
    )

    if result["date"].duplicated().any():
        raise RuntimeError(
            f"Duplicate weight dates: {path}"
        )

    values = result[TICKERS].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"Nonfinite weights: {path}"
        )

    if (values < -1e-6).any():
        raise RuntimeError(
            f"Negative weights: {path}"
        )

    totals = values.sum(axis=1)

    if not np.allclose(
        totals, 1.0, atol=0.02
    ):
        raise RuntimeError(
            f"Weights do not sum to 1: {path}\n"
            f"Min={totals.min()}, Max={totals.max()}"
        )

    return result


rows = []

for seed in range(5):

    folder = (
        "FULL15"
        if seed == 0
        else "FULL15_seeds1to4"
    )

    for condition in CONDITIONS:

        run_dir = (
            BASE
            / folder
            / "runs"
            / (
                f"conservative__{condition}"
                f"__A2C__seed{seed}"
            )
        )

        path = run_dir / "test_weights.csv"

        weights = read_weights(path)

        years = set(
            weights["date"].dt.year.unique()
        )

        if years != {2022, 2023}:
            raise RuntimeError(
                f"Unexpected years {years}: {path}"
            )

        for year, group in weights.groupby(
            weights["date"].dt.year
        ):

            for ticker in TICKERS:
                rows.append({
                    "seed": seed,
                    "year": int(year),
                    "condition": condition,
                    "ticker": ticker,
                    "mean_weight": (
                        group[ticker].mean()
                    ),
                    "min_weight": (
                        group[ticker].min()
                    ),
                    "max_weight": (
                        group[ticker].max()
                    ),
                })


result = pd.DataFrame(rows)

keys = [
    "seed", "year", "condition", "ticker"
]

if result.duplicated(keys).any():
    raise RuntimeError(
        "Duplicate attribution keys"
    )

expected = (
    5 * 2 * len(CONDITIONS) * len(TICKERS)
)

if len(result) != expected:
    raise RuntimeError(
        f"Expected {expected} rows, "
        f"got {len(result)}"
    )

result.to_csv(
    OUT / "a2c_full15_ticker_exposure_by_seed.csv",
    index=False,
)

summary = (
    result
    .groupby(
        ["year", "condition", "ticker"]
    )["mean_weight"]
    .agg(["mean", "std"])
    .reset_index()
)

summary.to_csv(
    OUT / "a2c_full15_ticker_exposure_summary.csv",
    index=False,
)

dynamic = result[
    result["condition"].eq(
        "FULL_DYNAMIC"
    )
][
    ["seed", "year", "ticker", "mean_weight"]
]

placebo = result[
    result["condition"].eq(
        "FULL_DYNAMIC_PLACEBO"
    )
][
    ["seed", "year", "ticker", "mean_weight"]
]

delta = dynamic.merge(
    placebo,
    on=["seed", "year", "ticker"],
    suffixes=("_dynamic", "_placebo"),
    validate="one_to_one",
)

delta["weight_difference"] = (
    delta["mean_weight_dynamic"]
    - delta["mean_weight_placebo"]
)

delta.to_csv(
    OUT / "a2c_full15_exposure_vs_placebo.csv",
    index=False,
)

print("\n" + "=" * 65)
print("FULL15 A2C EXPOSURE")
print("CONSERVATIVE / FIVE SEEDS")
print("=" * 65)

selected = summary[
    summary["condition"].isin([
        "FULL_DYNAMIC",
        "FULL_DYNAMIC_PLACEBO",
    ])
]

print(
    selected[
        ["year", "condition", "ticker", "mean"]
    ]
    .sort_values(
        ["year", "condition", "ticker"]
    )
    .to_string(index=False)
)

print("\n" + "=" * 65)
print("DYNAMIC - ZERO PLACEBO")
print("AVERAGE WEIGHT DIFFERENCES")
print("=" * 65)

comparison = (
    delta
    .groupby(["year", "ticker"])
    ["weight_difference"]
    .mean()
    .reset_index()
)

print(
    comparison.to_string(index=False)
)

print("\n=== OUTPUT FILES ===")
print(
    OUT / "a2c_full15_ticker_exposure_by_seed.csv"
)
print(
    OUT / "a2c_full15_ticker_exposure_summary.csv"
)
print(
    OUT / "a2c_full15_exposure_vs_placebo.csv"
)

print("\nPortfolio exposure audit: COMPLETE")
print("Causal return attribution: NOT COMPLETE")
