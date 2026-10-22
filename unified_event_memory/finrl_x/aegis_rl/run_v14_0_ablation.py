#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V14-0: Company-aware Feature Ablation

A0 = Market
A1 = Market + KGT
A2 = Market + KG RAG
A3 = Market + KGT + KG RAG
A4 = Market + KGT + PIT SEC RAG
A5 = Market + KGT + PIT SEC RAG + KG RAG

Each ticker has its own Ridge model.

2020: monthly expanding-window, label-maturity-safe OOS
2021: fixed hyperparameters, fit historical matured labels
2022-2023: not evaluated

Uses existing V13:
  - load_sec
  - kg_events_from_csv
  - build_kg_rag

Known provenance limitations:
  - KG observed_date is not independently verified release time.
  - SEC excerpts are not content-verified against official filing.
  - Historical KGT provenance is not fully verified.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v13_unified_sec_kg_rag as v13,
)

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]

HORIZON = 5
N = len(TICKERS)

ARMS = [
    "A0", "A1", "A2",
    "A3", "A4", "A5"
]

COMPARISONS = [
    ("A0", "A1"),  # KGT
    ("A0", "A2"),  # KG RAG without KGT
    ("A1", "A3"),  # KG incremental to KGT
    ("A1", "A4"),  # SEC incremental to KGT
    ("A3", "A5"),  # SEC incremental to KGT+KG
    ("A4", "A5"),  # KG incremental to KGT+SEC
    ("A2", "A3"),  # KGT incremental to KG RAG
    ("A0", "A5"),  # Full vs Market-only
]


def log(*args):
    print(*args, flush=True)


def normalize_dates(values):
    return pd.DatetimeIndex(
        pd.to_datetime(values, utc=True)
    ).tz_convert(None).normalize()


def validate_market_layout(market):
    """
    Verified original MultiStepEnv:
       [days, 5, 53 Market + 200 KGT]

    The original 1265D observation is company-major.
    """
    arr = np.asarray(
        market,
        dtype=np.float32,
    )

    if arr.ndim != 2 or arr.shape[1] != 1265:
        raise RuntimeError(
            f"Expected flat 1265D features, got {arr.shape}"
        )

    blocks = arr.reshape(
        len(arr), 5, 253
    )

    market53 = blocks[:, :, :53].copy()
    kgt200 = blocks[:, :, 53:].copy()

    if not np.array_equal(
        blocks.reshape(len(arr), 1265),
        arr,
    ):
        raise RuntimeError(
            "Market reconstruction failed"
        )

    if not (
        np.isfinite(market53).all()
        and np.isfinite(kgt200).all()
    ):
        raise RuntimeError(
            "Nonfinite Market/KGT features"
        )

    cross_company = float(
        np.mean(
            np.abs(
                kgt200[:, 1:, :]
                - kgt200[:, :1, :]
            )
        )
    )

    temporal = float(
        np.mean(
            np.abs(
                kgt200[1:] - kgt200[:-1]
            )
        )
    )

    log("[MARKET]", market53.shape)
    log("[KGT]", kgt200.shape)
    log("[KGT COMPANY DIFFERENCE]", cross_company)
    log("[KGT TEMPORAL DIFFERENCE]", temporal)

    if cross_company <= 1e-12:
        raise RuntimeError(
            "KGT identical across tickers"
        )

    if temporal <= 1e-12:
        raise RuntimeError(
            "KGT temporally constant"
        )

    global_mean = market53.mean(
        axis=1,
        keepdims=True,
    )

    relative = market53 - global_mean

    # Market-only baseline retains absolute and
    # contemporaneous relative 53D features.
    market_base = np.concatenate(
        [market53, relative],
        axis=2,
    ).astype(np.float32)

    return market_base, kgt200


def compose(*arrays):
    return np.concatenate(
        arrays,
        axis=2,
    ).astype(np.float32)


def build_targets(prices, dates):
    """
    y[i,t] = forward 5-day stock return
             - equal-weight 5-stock return

    Also records the trading date on which the
    five-day outcome becomes observable.

    No backward filling.
    """
    n = len(prices)

    targets = np.full(
        (n, N),
        np.nan,
        dtype=np.float64,
    )

    maturity = np.full(
        n,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    for t in range(n - HORIZON):
        future = t + HORIZON

        p0 = prices[t]
        p1 = prices[future]

        if (
            not np.isfinite(p0).all()
            or not np.isfinite(p1).all()
            or np.any(p0 <= 0)
        ):
            continue

        returns = p1 / p0 - 1.0

        targets[t] = (
            returns - returns.mean()
        )

        maturity[t] = np.datetime64(
            dates[future].to_datetime64()
        )

    return targets, maturity


def fit_company_models(x, y, ids, alpha):
    models = []

    if len(ids) < 200:
        raise RuntimeError(
            f"Too few training days: {len(ids)}"
        )

    for j, ticker in enumerate(TICKERS):
        xx = x[ids, j]
        yy = y[ids, j]

        if (
            not np.isfinite(xx).all()
            or not np.isfinite(yy).all()
        ):
            raise RuntimeError(
                f"Nonfinite training values: {ticker}"
            )

        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha),
        )

        model.fit(xx, yy)
        models.append(model)

    return models


def predict_company_models(models, x, ids):
    output = np.empty(
        (len(ids), N),
        dtype=np.float64,
    )

    for j, model in enumerate(models):
        output[:, j] = model.predict(
            x[ids, j]
        )

    return output


def evaluate_predictions(
    actual,
    predicted,
    dates,
    arm,
    period,
):
    if len(actual) != len(predicted):
        raise RuntimeError(
            "Prediction length mismatch"
        )

    rows = []

    for i, date in enumerate(dates):
        y = actual[i]
        p = predicted[i]

        if not (
            np.isfinite(y).all()
            and np.isfinite(p).all()
        ):
            raise RuntimeError(
                "Nonfinite evaluation values"
            )

        ic = (
            spearmanr(y, p).statistic
            if np.ptp(y) > 1e-12
            and np.ptp(p) > 1e-12
            else np.nan
        )

        spread = (
            y[int(np.argmax(p))]
            - y[int(np.argmin(p))]
        )

        rows.append({
            "period": period,
            "date": str(date.date()),
            "arm": arm,
            "mse": float(
                np.mean((y-p)**2)
            ),
            "mae": float(
                np.mean(np.abs(y-p))
            ),
            "rank_ic": float(ic),
            "gross_rank_spread": float(spread),
            "sign_accuracy": float(
                np.mean((y > 0) == (p > 0))
            ),
        })

    return pd.DataFrame(rows)


def summarize(daily):
    group_cols = ["period", "arm"]

    result = (
        daily.groupby(group_cols)
        .agg(
            days=("date", "nunique"),
            mse=("mse", "mean"),
            mae=("mae", "mean"),
            mean_rank_ic=("rank_ic", "mean"),
            gross_rank_spread=(
                "gross_rank_spread", "mean"
            ),
            sign_accuracy=(
                "sign_accuracy", "mean"
            ),
        )
        .reset_index()
    )

    return result


def moving_block_comparison(
    daily,
    period,
    baseline,
    challenger,
    metric,
    seed,
    reps=1500,
    block=10,
):
    a = daily[
        (daily["period"] == period)
        & (daily["arm"] == baseline)
    ][["date", metric]].rename(
        columns={metric: "a"}
    )

    b = daily[
        (daily["period"] == period)
        & (daily["arm"] == challenger)
    ][["date", metric]].rename(
        columns={metric: "b"}
    )

    paired = a.merge(
        b,
        on="date",
        how="inner",
        validate="one_to_one",
    ).dropna()

    if metric in (
        "rank_ic",
        "gross_rank_spread",
    ):
        delta = (
            paired["b"].to_numpy()
            - paired["a"].to_numpy()
        )
    else:
        delta = (
            paired["a"].to_numpy()
            - paired["b"].to_numpy()
        )

    n = len(delta)

    if n < 20:
        return {
            "period": period,
            "baseline": baseline,
            "challenger": challenger,
            "metric": metric,
            "n_days": n,
            "mean_improvement": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    rng = np.random.default_rng(seed)
    length = min(block, n)

    draws = np.empty(
        reps,
        dtype=np.float64,
    )

    for k in range(reps):
        sample = []

        while len(sample) < n:
            start = int(
                rng.integers(
                    0,
                    n-length+1,
                )
            )

            sample.extend(
                range(start, start+length)
            )

        sample = np.asarray(
            sample[:n],
            dtype=int,
        )

        draws[k] = delta[sample].mean()

    lo, hi = np.quantile(
        draws,
        [0.025, 0.975],
    )

    return {
        "period": period,
        "baseline": baseline,
        "challenger": challenger,
        "metric": metric,
        "n_days": n,
        "mean_improvement": float(
            np.mean(delta)
        ),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def run_2020_expanding(
    datasets,
    targets,
    maturity,
    dates,
    alpha,
):
    """
    Each calendar month of 2020 is a separate
    out-of-sample fold.

    Training labels must have matured STRICTLY
    before the first decision date of the fold.

    Evaluation labels must mature by 2020-12-31
    so the internal development assessment never
    uses 2021 prices.
    """
    results = []
    predictions = []
    folds = []

    months = pd.period_range(
        "2020-01",
        "2020-12",
        freq="M",
    )

    date_array = dates.to_numpy(
        dtype="datetime64[ns]"
    )

    finite = np.isfinite(
        targets
    ).all(axis=1)

    for month in months:
        first = pd.Timestamp(
            month.start_time
        )
        last = pd.Timestamp(
            month.end_time
        )

        test_ids = np.flatnonzero(
            (dates >= first)
            & (dates <= last)
            & finite
            & (
                maturity
                <= np.datetime64("2020-12-31")
            )
        )

        if len(test_ids) == 0:
            continue

        cutoff = date_array[test_ids[0]]

        train_ids = np.flatnonzero(
            finite
            & (maturity < cutoff)
            & (date_array < cutoff)
        )

        if len(train_ids) < 200:
            raise RuntimeError(
                "Insufficient 2020 fold training history"
            )

        folds.append({
            "month": str(month),
            "training_days": len(train_ids),
            "test_days": len(test_ids),
            "latest_training_maturity": str(
                pd.Timestamp(
                    maturity[train_ids].max()
                ).date()
            ),
            "first_evaluation_date": str(
                dates[test_ids[0]].date()
            ),
            "last_evaluation_date": str(
                dates[test_ids[-1]].date()
            ),
        })

        log(
            "[2020 FOLD]",
            str(month),
            "train=", len(train_ids),
            "eval=", len(test_ids),
        )

        for arm in ARMS:
            models = fit_company_models(
                datasets[arm],
                targets,
                train_ids,
                alpha,
            )

            pred = predict_company_models(
                models,
                datasets[arm],
                test_ids,
            )

            daily = evaluate_predictions(
                targets[test_ids],
                pred,
                dates[test_ids],
                arm,
                "2020_WALK_FORWARD",
            )

            results.append(daily)

            for k, day in enumerate(test_ids):
                for j, ticker in enumerate(TICKERS):
                    predictions.append({
                        "period": "2020_WALK_FORWARD",
                        "date": str(
                            dates[day].date()
                        ),
                        "ticker": ticker,
                        "arm": arm,
                        "actual": float(
                            targets[day, j]
                        ),
                        "prediction": float(
                            pred[k, j]
                        ),
                    })

    if not results:
        raise RuntimeError(
            "No 2020 walk-forward evaluation rows"
        )

    return (
        pd.concat(
            results,
            ignore_index=True,
        ),
        pd.DataFrame(predictions),
        pd.DataFrame(folds),
    )


def run_2021_fixed(
    datasets,
    targets,
    maturity,
    dates,
    alpha,
):
    """
    Fit once using labels matured before 2021.

    2021 outcomes must mature within 2021.
    No repeated 2021 tuning/refitting.
    """
    date_array = dates.to_numpy(
        dtype="datetime64[ns]"
    )

    finite = np.isfinite(
        targets
    ).all(axis=1)

    train_ids = np.flatnonzero(
        finite
        & (
            maturity < np.datetime64(
                "2021-01-01"
            )
        )
        & (dates.year <= 2020)
    )

    val_ids = np.flatnonzero(
        finite
        & (dates.year == 2021)
        & (
            maturity <= np.datetime64(
                "2021-12-31"
            )
        )
    )

    if (
        len(train_ids) < 600
        or len(val_ids) < 200
    ):
        raise RuntimeError(
            "2021 split unexpectedly small: "
            f"{len(train_ids)} / {len(val_ids)}"
        )

    log(
        "[2021 FIXED]",
        "train=", len(train_ids),
        "validation=", len(val_ids),
    )

    results = []
    predictions = []

    for arm in ARMS:
        models = fit_company_models(
            datasets[arm],
            targets,
            train_ids,
            alpha,
        )

        pred = predict_company_models(
            models,
            datasets[arm],
            val_ids,
        )

        results.append(
            evaluate_predictions(
                targets[val_ids],
                pred,
                dates[val_ids],
                arm,
                "2021_VALIDATION",
            )
        )

        for k, day in enumerate(val_ids):
            for j, ticker in enumerate(TICKERS):
                predictions.append({
                    "period": "2021_VALIDATION",
                    "date": str(
                        dates[day].date()
                    ),
                    "ticker": ticker,
                    "arm": arm,
                    "actual": float(
                        targets[day, j]
                    ),
                    "prediction": float(
                        pred[k, j]
                    ),
                })

    return (
        pd.concat(
            results,
            ignore_index=True,
        ),
        pd.DataFrame(predictions),
        {
            "train_days": len(train_ids),
            "validation_days": len(val_ids),
            "latest_train_label_maturity": str(
                pd.Timestamp(
                    maturity[train_ids].max()
                ).date()
            ),
        },
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pit-sec",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/"
            "sec_pit_retrieval.parquet"
        ),
    )

    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v14_0_ablation"
        ),
    )

    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1000.0,
    )

    p.add_argument(
        "--sec-top-k",
        type=int,
        default=5,
    )

    p.add_argument(
        "--kg-top-k",
        type=int,
        default=12,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Existing FinRL-X loader compatibility.
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )

    p.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    p.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
        ),
    )

    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )

    p.add_argument(
        "--embed",
        default="bge-m3:latest",
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")

    args = p.parse_args()

    if args.ridge_alpha <= 0:
        raise ValueError(
            "Ridge alpha must be positive"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("===== LOAD FINRL-X MARKET / KGT =====")

    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    dates = normalize_dates(dates)

    market = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    ).astype(np.float32)

    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    ).astype(np.float64)

    if not (
        len(market) == len(prices) == len(dates)
    ):
        raise RuntimeError(
            "Market / price / date misalignment"
        )

    if not (
        dates.min().year == 2018
        and dates.max().year == 2021
        and dates[n_train].year == 2021
    ):
        raise RuntimeError(
            "Unexpected historical train/val dates"
        )

    market_base, kgt200 = (
        validate_market_layout(market)
    )

    log(
        "[MARKET BASE]",
        market_base.shape,
    )

    log("===== SEC PIT RETRIEVAL =====")

    sec, sec_documents = v13.load_sec(
        dates,
        contexts,
        args.pit_sec,
        args.sec_top_k,
        args.out / "sec_retrieval_audit.csv",
    )

    log("===== TEMPORAL KG RAG =====")

    kg_events = v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg_rag, relations = v13.build_kg_rag(
        dates,
        kg_events,
        args.kg_top_k,
        args.out,
    )

    if not (
        sec.shape[:2] == (len(dates), N)
        and kg_rag.shape[:2] == (len(dates), N)
    ):
        raise RuntimeError(
            "SEC/KG shape mismatch"
        )

    if not (
        np.isfinite(sec).all()
        and np.isfinite(kg_rag).all()
    ):
        raise RuntimeError(
            "SEC/KG features are nonfinite"
        )

    log(
        "[SEC]",
        sec.shape,
        "[KG]",
        kg_rag.shape,
    )

    log("===== BUILD SIX ABLATION ARMS =====")

    datasets = {
        "A0": compose(market_base),
        "A1": compose(
            market_base, kgt200
        ),
        "A2": compose(
            market_base, kg_rag
        ),
        "A3": compose(
            market_base, kgt200, kg_rag
        ),
        "A4": compose(
            market_base, kgt200, sec
        ),
        "A5": compose(
            market_base, kgt200, sec, kg_rag
        ),
    }

    for arm, x in datasets.items():
        if not np.isfinite(x).all():
            raise RuntimeError(
                f"{arm} contains nonfinite data"
            )
        log("[ARM]", arm, x.shape)

    log("===== MAKE MATURITY-AWARE LABELS =====")

    targets, maturity = build_targets(
        prices,
        dates,
    )

    log("===== 2020 MONTHLY WALK-FORWARD =====")

    walk_daily, walk_pred, folds = (
        run_2020_expanding(
            datasets,
            targets,
            maturity,
            dates,
            args.ridge_alpha,
        )
    )

    folds.to_csv(
        args.out / "walk_forward_folds.csv",
        index=False,
    )

    log("===== 2021 FIXED VALIDATION =====")

    val_daily, val_pred, val_info = (
        run_2021_fixed(
            datasets,
            targets,
            maturity,
            dates,
            args.ridge_alpha,
        )
    )

    daily = pd.concat(
        [walk_daily, val_daily],
        ignore_index=True,
    )

    daily.to_csv(
        args.out / "daily_metrics.csv",
        index=False,
    )

    predictions = pd.concat(
        [walk_pred, val_pred],
        ignore_index=True,
    )

    predictions.to_csv(
        args.out / "predictions.csv",
        index=False,
    )

    metrics = summarize(daily)

    metrics.to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    log("===== PAIRED BLOCK BOOTSTRAP =====")

    boot_rows = []

    for period in [
        "2020_WALK_FORWARD",
        "2021_VALIDATION",
    ]:
        for i, (baseline, challenger) in enumerate(
            COMPARISONS
        ):
            for j, metric in enumerate(
                ["mse", "mae", "rank_ic"]
            ):
                boot_rows.append(
                    moving_block_comparison(
                        daily,
                        period,
                        baseline,
                        challenger,
                        metric,
                        seed=(
                            args.seed
                            + i*10
                            + j
                            + (
                                1000
                                if period == "2021_VALIDATION"
                                else 0
                            )
                        ),
                    )
                )

    bootstrap = pd.DataFrame(
        boot_rows
    )

    bootstrap.to_csv(
        args.out / "paired_bootstrap.csv",
        index=False,
    )

    coverage = {
        "sec_nonzero_train": float(
            np.mean(
                np.linalg.norm(
                    sec[:n_train],
                    axis=2,
                ) > 1e-8
            )
        ),
        "sec_nonzero_val": float(
            np.mean(
                np.linalg.norm(
                    sec[n_train:],
                    axis=2,
                ) > 1e-8
            )
        ),
        "kg_nonzero_train": float(
            np.mean(
                np.linalg.norm(
                    kg_rag[:n_train],
                    axis=2,
                ) > 1e-8
            )
        ),
        "kg_nonzero_val": float(
            np.mean(
                np.linalg.norm(
                    kg_rag[n_train:],
                    axis=2,
                ) > 1e-8
            )
        ),
    }

    manifest = {
        "experiment": "FinRL-X V14-0",
        "train_source": "2018-2020",
        "internal_evaluation": (
            "2020 expanding monthly "
            "walk-forward"
        ),
        "validation": "2021 fixed setup",
        "test_2022_2023_used": False,
        "forecast_horizon_days": HORIZON,
        "ridge_alpha": args.ridge_alpha,
        "market_dim": 53,
        "market_relative_dim": 53,
        "kgt_dim": 200,
        "sec_dim": int(sec.shape[2]),
        "kg_rag_dim": int(
            kg_rag.shape[2]
        ),
        "sec_document_rows": int(
            sec_documents
        ),
        "kg_unique_company_triples": int(
            len(kg_events)
        ),
        "kg_relation_count": len(relations),
        "kg_relation_names": relations,
        "2020_folds": len(folds),
        "2021": val_info,
        "coverage": coverage,
        "arms": {
            "A0": "Market only",
            "A1": "Market + KGT",
            "A2": "Market + KG RAG",
            "A3": "Market + KGT + KG RAG",
            "A4": "Market + KGT + PIT SEC",
            "A5": "Market + KGT + PIT SEC + KG RAG",
        },
        "limitations": [
            "2021 has been examined in previous experiments.",
            "KG timestamps are source-observation dates, not externally verified public availability.",
            "KG is company-linked direct-triple memory, not multihop GraphRAG.",
            "PIT SEC source metadata is checked; excerpt text is not independently verified.",
            "KGT snapshot source timing still requires provenance verification.",
            "All models use fixed ridge alpha; no 2021 tuning.",
            "No A2C trading or transaction-cost evaluation.",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 80)
    log("V14-0 METHOD METRICS")
    log("=" * 80)
    log(
        metrics.to_string(index=False)
    )

    log()
    log("=" * 80)
    log("V14-0 2020 WALK-FORWARD COMPARISONS")
    log("=" * 80)

    selected = bootstrap[
        (bootstrap["period"] == "2020_WALK_FORWARD")
        & (bootstrap["metric"] == "rank_ic")
    ]

    log(
        selected.to_string(index=False)
    )

    log()
    log("[COVERAGE]")
    log(
        json.dumps(
            coverage, indent=2
        )
    )

    log()
    log("V14-0: PASS")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
