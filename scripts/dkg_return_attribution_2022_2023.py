#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/doralin/projects/FinRL-Trading")
BASE = ROOT / "outputs/finrl_dkg_pit_2022_2023"
RUNS = BASE / "finrl_runs"
OUT = RUNS / "merged_5seeds"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
CONDITIONS = ["FULL_DYNAMIC", "FULL_DYNAMIC_PLACEBO"]
TOL = 1e-6


def load_prices():
    path = (
        BASE / "finrl_inputs/FULL15/"
        "dataset_conservative.csv"
    )

    df = pd.read_csv(
        path,
        usecols=["date", "ticker", "market_close"]
    )

    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].str.upper()

    if df.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Duplicate market prices")

    prices = df.pivot(
        index="date",
        columns="ticker",
        values="market_close",
    ).sort_index()

    prices = prices[TICKERS]

    if prices.isna().any().any():
        raise RuntimeError("Missing market prices")

    if (prices <= 0).any().any():
        raise RuntimeError("Nonpositive market prices")

    return prices


def read_run(seed, condition, prices):
    folder = (
        "FULL15" if seed == 0
        else "FULL15_seeds1to4"
    )

    run = (
        RUNS / folder / "runs"
        / f"conservative__{condition}__A2C__seed{seed}"
    )

    weights = pd.read_csv(run / "test_weights.csv")
    returns = pd.read_csv(run / "test_returns.csv")
    equity = pd.read_csv(run / "test_equity.csv")

    for frame in (weights, returns, equity):
        frame["date"] = pd.to_datetime(frame["date"])

        if frame["date"].duplicated().any():
            raise RuntimeError("Duplicate evaluation dates")

        frame.sort_values("date", inplace=True)
        frame.reset_index(drop=True, inplace=True)

    if len(weights) != len(equity):
        raise RuntimeError("Weights/equity length mismatch")

    if len(returns) != len(equity) - 1:
        raise RuntimeError("Returns/equity length mismatch")

    if not weights["date"].equals(equity["date"]):
        raise RuntimeError("Weight/equity date mismatch")

    if not returns["date"].equals(
        equity["date"].iloc[1:].reset_index(drop=True)
    ):
        raise RuntimeError("Return/equity date mismatch")

    dates = pd.DatetimeIndex(equity["date"])

    if not dates.isin(prices.index).all():
        raise RuntimeError("Missing evaluation price dates")

    p = prices.loc[dates, TICKERS].to_numpy(
        dtype=float
    )

    asset_returns = p[1:] / p[:-1] - 1.0

    post_weights = weights[TICKERS].to_numpy(
        dtype=float
    )[1:]

    if not np.isfinite(post_weights).all():
        raise RuntimeError("Invalid post-return weights")

    # Reverse price drift to recover target weights.
    growth = 1.0 + asset_returns

    if (growth <= 0).any():
        raise RuntimeError("Invalid asset gross return")

    unnormalized = post_weights / growth

    target = (
        unnormalized /
        unnormalized.sum(axis=1, keepdims=True)
    )

    gross = np.sum(
        target * asset_returns,
        axis=1
    )

    net = returns["daily_return"].to_numpy(
        dtype=float
    )

    values = equity["portfolio_value"].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite portfolio values")

    observed_net = values[1:] / values[:-1] - 1.0

    error = np.max(
        np.abs(observed_net - net)
    )

    print(
        f"Seed {seed} {condition}: "
        f"net/equity max error={error:.3e}"
    )

    if error > TOL:
        raise RuntimeError(
            "Net returns do not match equity curve"
        )

    # The residual is the transaction cost implied by
    # the environment's gross/net return equations.
    implied_cost = gross - net

    if np.min(implied_cost) < -TOL:
        raise RuntimeError(
            "Negative implied transaction cost; "
            "check price and weight alignment"
        )

    implied_cost = np.maximum(implied_cost, 0.0)

    reconstructed = gross - implied_cost

    if np.max(np.abs(reconstructed - net)) > TOL:
        raise RuntimeError("Return reconstruction failed")

    # Contributions are converted into dollar P&L
    # using beginning-of-interval portfolio equity.
    stock_dollars = (
        values[:-1, None]
        * target
        * asset_returns
    )

    cost_dollars = (
        -values[:-1] * implied_cost
    )

    delta_equity = values[1:] - values[:-1]

    reconstructed_pnl = (
        stock_dollars.sum(axis=1) + cost_dollars
    )

    relative_error = np.max(
        np.abs(reconstructed_pnl - delta_equity)
        / np.maximum(np.abs(values[:-1]), 1)
    )

    print(
        f"  P&L relative max error="
        f"{relative_error:.3e}"
    )

    if relative_error > TOL:
        raise RuntimeError("P&L reconstruction mismatch")

    result = pd.DataFrame({
        "date": dates[1:],
        "seed": seed,
        "condition": condition,
        "gross_return": gross,
        "net_return": net,
        "implied_cost_return": implied_cost,
        "cost_dollars": cost_dollars,
        "equity_start": values[:-1],
        "equity_end": values[1:],
    })

    for i, ticker in enumerate(TICKERS):
        result[f"target_{ticker}"] = target[:, i]
        result[f"contrib_dollars_{ticker}"] = (
            stock_dollars[:, i]
        )

    return result


prices = load_prices()

daily = pd.concat(
    [
        read_run(seed, condition, prices)
        for seed in range(5)
        for condition in CONDITIONS
    ],
    ignore_index=True,
)

daily.to_csv(
    OUT / "a2c_full15_daily_pnl_attribution.csv",
    index=False,
)

rows = []

for (seed, condition), group in daily.groupby(
    ["seed", "condition"]
):
    group = group.sort_values("date")

    for period, sub in [
        ("2022", group[group["date"].dt.year == 2022]),
        ("2023", group[group["date"].dt.year == 2023]),
        ("2022-2023", group),
    ]:
        if sub.empty:
            raise RuntimeError("Empty attribution period")

        initial = float(sub["equity_start"].iloc[0])
        final = float(sub["equity_end"].iloc[-1])

        contributions = {
            ticker: (
                sub[f"contrib_dollars_{ticker}"].sum()
                / initial
            )
            for ticker in TICKERS
        }

        cost_contribution = (
            sub["cost_dollars"].sum() / initial
        )

        actual_return = final / initial - 1

        reconstructed_return = (
            sum(contributions.values())
            + cost_contribution
        )

        if not np.isclose(
            actual_return,
            reconstructed_return,
            atol=TOL,
            rtol=0,
        ):
            raise RuntimeError(
                f"Attribution sum mismatch: {seed}, "
                f"{condition}, {period}"
            )

        rows.append({
            "seed": seed,
            "condition": condition,
            "period": period,
            "actual_total_return": actual_return,
            "cost_contribution": cost_contribution,
            **{
                f"{t}_contribution": contributions[t]
                for t in TICKERS
            },
        })

attribution = pd.DataFrame(rows)

attribution.to_csv(
    OUT / "a2c_full15_return_contribution_by_seed.csv",
    index=False,
)

dynamic = attribution[
    attribution["condition"].eq("FULL_DYNAMIC")
]

placebo = attribution[
    attribution["condition"].eq(
        "FULL_DYNAMIC_PLACEBO"
    )
]

paired = dynamic.merge(
    placebo,
    on=["seed", "period"],
    suffixes=("_dynamic", "_placebo"),
    validate="one_to_one",
)

components = [
    "actual_total_return",
    "cost_contribution",
    *[f"{t}_contribution" for t in TICKERS],
]

for col in components:
    paired[f"delta_{col}"] = (
        paired[f"{col}_dynamic"]
        - paired[f"{col}_placebo"]
    )

paired.to_csv(
    OUT / "a2c_full15_return_attribution_vs_placebo.csv",
    index=False,
)

print("\n" + "=" * 75)
print("FULL15 A2C: DYNAMIC MINUS ZERO-PLACEBO")
print("Dollar P&L attribution, divided by period-start equity")
print("Five Seeds, Conservative")
print("=" * 75)

display = (
    paired.groupby("period")[
        [f"delta_{c}" for c in components]
    ].mean()
    .reset_index()
)

for col in display.columns:
    if col.startswith("delta_"):
        display[col] *= 100

print(display.to_string(index=False, float_format="%.3f"))

print("\nAll contribution values above are percentage points.")
print("2022-2023 contributions use initial 2022 equity.")
print("2023 contributions use beginning-2023 equity.")
print("No extra model training was performed.")
print("Strict PIT: NOT CERTIFIED.")
