#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V7-C: SEC Predictive Value Test

Purpose
-------
Test incremental predictive value of structured SEC signals,
without training RL or re-querying RAG.

Data
----
2018-2020 training (labels fully mature by 2020-12-31)
2021 validation only.
2022-2023 is NEVER evaluated.

Arms
----
MARKET: historical price features, 25D per stock-market state
SEC:    impact, risk, confidence, recency, availability
COMBINED: MARKET + SEC

Targets:
    5- and 20-trading-day forward simple stock returns

Models:
    Ridge regression, StandardScaler fit on TRAIN only.

Metrics:
    OOS R2 vs train-mean forecast
    MAE
    Directional Accuracy
    Daily cross-sectional Spearman Rank IC
    Daily cross-sectional IC coverage

No RL training. No future SEC information.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from unified_event_memory.finrl_x.aegis_rl import (
    run_v7b_structured_evidence as v7b,
)

TICKERS = v7b.TICKERS
HORIZONS = [5, 20]


def log(*args):
    print(*args, flush=True)


def feature_matrix(prices, lookback=60):
    """
    Date-t features computed only from prices <= t.

    For each of 5 stocks:
      logret_1d, logret_5d, logret_20d,
      mean_return_20d, volatility_20d

    Returns [N, 5, 5].
    """

    p = np.asarray(prices, dtype=np.float64)

    if p.ndim != 2 or p.shape[1] != 5:
        raise RuntimeError(f"Invalid prices shape: {p.shape}")

    if not np.isfinite(p).all() or (p <= 0).any():
        raise RuntimeError("Invalid price values")

    n = len(p)

    out = np.zeros((n, 5, 5), dtype=np.float64)
    logr = np.diff(np.log(p), axis=0, prepend=np.log(p[:1]))

    for t in range(n):
        for j in range(5):
            r1 = logr[t, j] if t >= 1 else 0.0
            r5 = np.log(p[t, j] / p[t - 5, j]) if t >= 5 else 0.0
            r20 = np.log(p[t, j] / p[t - 20, j]) if t >= 20 else 0.0

            if t >= 20:
                recent = logr[t - 19:t + 1, j]
                mean20 = recent.mean()
                vol20 = recent.std(ddof=1)
            else:
                mean20 = 0.0
                vol20 = 0.0

            out[t, j] = [r1, r5, r20, mean20, vol20]

    return out.astype(np.float32)


def assemble_rows(data, horizon, split):
    """
    Only include t where t+h exists inside loaded split.
    This also removes train labels crossing 2020/2021.

    Each row = one company × decision day.
    MARKET features are all-stock historical signals (25D),
    same 25D market state repeated for each target company
    plus 5D ticker one-hot vector.
    SEC has 5D own-company structured signals.
    """

    dates = pd.DatetimeIndex(data["dates"])
    prices = np.asarray(data["prices"], dtype=np.float64)
    sec = np.asarray(data["structured"], dtype=np.float32).reshape(-1, 5, 5)
    market = feature_matrix(prices).reshape(-1, 25)

    n = min(len(dates), len(prices), len(sec), len(market))

    records = []
    x_market = []
    x_sec = []
    y = []

    for t in range(n - horizon):
        end_t = t + horizon

        if split == "train":
            if dates[end_t] > pd.Timestamp("2020-12-31"):
                continue
        elif split == "val":
            if dates[t] < pd.Timestamp("2021-01-01"):
                continue
            if dates[end_t] > pd.Timestamp("2021-12-31"):
                continue
        else:
            raise RuntimeError("Test split forbidden in pilot")

        state = market[t]

        for j, ticker in enumerate(TICKERS):
            future_return = (
                prices[end_t, j] / prices[t, j] - 1.0
            )

            ticker_id = np.zeros(5, dtype=np.float32)
            ticker_id[j] = 1.0

            # Both arms receive ticker identity.
            market_row = np.concatenate([
                state,
                ticker_id,
            ])

            sec_row = np.concatenate([
                sec[t, j],
                ticker_id,
            ])

            x_market.append(market_row)
            x_sec.append(sec_row)
            y.append(float(future_return))

            records.append({
                "split": split,
                "date": str(dates[t].date()),
                "target_date": str(dates[end_t].date()),
                "ticker": ticker,
                "ticker_idx": j,
            })

    if not records:
        raise RuntimeError(f"No rows for {split}, horizon={horizon}")

    meta = pd.DataFrame(records)

    return {
        "meta": meta,
        "market": np.asarray(x_market, dtype=np.float32),
        "sec": np.asarray(x_sec, dtype=np.float32),
        "combined": np.concatenate([
            np.asarray(x_market, dtype=np.float32),
            np.asarray(x_sec, dtype=np.float32)[:, :5],
        ], axis=1),
        "y": np.asarray(y, dtype=np.float64),
    }


def daily_ic(meta, actual, predicted):
    """
    One Spearman rank correlation per date, across five stocks.
    Dates with constant y or predictions are undefined, not zero.
    """

    df = meta[["date", "ticker"]].copy()
    df["actual"] = actual
    df["predicted"] = predicted

    rows = []

    for date, group in df.groupby("date", sort=True):
        a = group["actual"].to_numpy()
        p = group["predicted"].to_numpy()

        if (
            len(a) < 3
            or np.ptp(a) < 1e-12
            or np.ptp(p) < 1e-12
        ):
            ic = np.nan
        else:
            ic = spearmanr(a, p).statistic

        rows.append({
            "date": date,
            "rank_ic": float(ic) if np.isfinite(ic) else np.nan,
        })

    return pd.DataFrame(rows)


def directional_accuracy(actual, predicted):
    return float(
        np.mean(
            (actual > 0.0) == (predicted > 0.0)
        )
    )


def fit_evaluate(train, val, arm, alpha, horizon, out):
    X_train = train[arm]
    X_val = val[arm]

    y_train = train["y"]
    y_val = val["y"]

    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    # Train-only mean forecast reference.
    base = np.full_like(y_val, y_train.mean())

    ic = daily_ic(
        val["meta"],
        y_val,
        pred_val,
    )

    mean_ic = float(ic["rank_ic"].mean())

    valid_ic = int(ic["rank_ic"].notna().sum())

    oos_r2_train_mean = (
        1.0
        - np.sum((y_val - pred_val) ** 2)
        / np.sum((y_val - base) ** 2)
    )

    metrics = {
        "horizon": horizon,
        "arm": arm.upper(),
        "train_samples": len(y_train),
        "val_samples": len(y_val),
        "features": X_train.shape[1],
        "alpha": alpha,
        "r2_val": float(r2_score(y_val, pred_val)),
        "oos_r2_vs_train_mean": float(oos_r2_train_mean),
        "mae": float(mean_absolute_error(y_val, pred_val)),
        "directional_accuracy": directional_accuracy(y_val, pred_val),
        "rank_ic_mean": mean_ic,
        "rank_ic_valid_days": valid_ic,
        "rank_ic_total_days": len(ic),
        "train_r2": float(r2_score(y_train, pred_train)),
    }

    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        **{
            col: val["meta"][col].to_numpy()
            for col in val["meta"].columns
        },
        "actual": y_val,
        "prediction": pred_val,
    }).to_csv(
        out / f"{arm}_h{horizon}_predictions.csv",
        index=False,
    )

    ic.to_csv(
        out / f"{arm}_h{horizon}_daily_ic.csv",
        index=False,
    )

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/v7c_sec_predictive_value"
        ),
    )

    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--max-days", type=int, default=0)

    parser.add_argument(
        "--sec-signals",
        default="outputs/unified_event_memory_v2/sec_numeric_signals.csv",
    )

    parser.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/active_rag_v5_smoke/"
            "runs/ACTIVE__A2C__seed42/model.zip"
        ),
    )

    parser.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    parser.add_argument(
        "--kg",
        default="outputs/unified_temporal_kg_v3/unified_temporal_edges.csv",
    )

    parser.add_argument(
        "--sec",
        default="rag/data/processed/sec/sec_chunks.jsonl",
    )

    parser.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    parser.add_argument("--llm", default="qwen2.5:3b")
    parser.add_argument("--embed", default="bge-m3:latest")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--cost-scale", type=float, default=0.0)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--tier", default="conservative")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train")
    parser.add_argument("--algo", default="A2C")

    args = parser.parse_args()

    if args.max_days != 0:
        raise RuntimeError(
            "V7-C requires complete historical splits: --max-days 0"
        )

    if args.alpha <= 0:
        raise RuntimeError("alpha must be positive")

    args.out.mkdir(parents=True, exist_ok=True)

    log("===== V7-C LOAD TRAIN AND VAL =====")

    train_data = v7b.load_split(args, "train")
    val_data = v7b.load_split(args, "val")

    results = []

    for horizon in HORIZONS:
        log()
        log("=" * 80)
        log("HORIZON:", horizon, "TRADING DAYS")
        log("=" * 80)

        train = assemble_rows(
            train_data, horizon, "train"
        )

        val = assemble_rows(
            val_data, horizon, "val"
        )

        log(
            "[SAMPLES]",
            "train=", len(train["y"]),
            "val=", len(val["y"]),
        )

        for arm in ["market", "sec", "combined"]:
            row = fit_evaluate(
                train,
                val,
                arm,
                args.alpha,
                horizon,
                args.out,
            )

            results.append(row)

            log(
                f"[{arm.upper():8s}] "
                f"OOS_R2={row['oos_r2_vs_train_mean']:+.6f} "
                f"IC={row['rank_ic_mean']:+.5f} "
                f"DA={row['directional_accuracy']:.4f}"
            )

    df = pd.DataFrame(results)
    df.to_csv(
        args.out / "metrics.csv",
        index=False,
    )

    rows = []

    for horizon in HORIZONS:
        subset = df.set_index(
            ["horizon", "arm"]
        )

        market = subset.loc[
            (horizon, "MARKET")
        ]

        combined = subset.loc[
            (horizon, "COMBINED")
        ]

        rows.append({
            "horizon": horizon,
            "delta_oos_r2": (
                combined["oos_r2_vs_train_mean"]
                - market["oos_r2_vs_train_mean"]
            ),
            "delta_rank_ic": (
                combined["rank_ic_mean"]
                - market["rank_ic_mean"]
            ),
            "delta_directional_accuracy": (
                combined["directional_accuracy"]
                - market["directional_accuracy"]
            ),
            "delta_mae": (
                combined["mae"] - market["mae"]
            ),
        })

    improvements = pd.DataFrame(rows)

    improvements.to_csv(
        args.out / "incremental_value.csv",
        index=False,
    )

    log()
    log("===== FULL METRICS =====")
    log(df.to_string(index=False))

    log()
    log("===== SEC INCREMENTAL VALUE =====")
    log(improvements.to_string(index=False))

    log()
    log("V7-C PREDICTIVE VALUE: PASS")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
