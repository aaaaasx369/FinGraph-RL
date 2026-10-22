#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V9
Reflection-Augmented KG-RAG Portfolio

B0: A2C + Dynamic KGT
B1: B0 + Current SEC/KG RAG Prior
B2: B1 + Historical Reflection
B3: B1 + Shuffled Reflection

Train: 2018-2020
Validation: 2021
Test: NOT USED

Existing modules:
  run_v7b_structured_evidence.py
  run_v8_replug_finagent.py
  run_v8_phase2b.py

No new A2C training.
No future labels at decision time.
No LLM API requirement.

The reflection memory is built from historical
cases with completed 5-day stock returns.

This is a controlled research prototype:
Current evidence scoring uses a historical Ridge
model; reflection retrieval uses nearest historical
cases. It is not an LLM reflection system.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from stable_baselines3 import A2C

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v8_replug_finagent as v8,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
HORIZON = 5
MARKET_DIM = 1265
DOC_DIM = 64
KG_DIM = 8
N_STOCKS = 5

ARMS = ["B0", "B1", "B2", "B3"]


def log(*x):
    print(*x, flush=True)


def normalized_dates(values):
    return pd.DatetimeIndex(
        pd.to_datetime(values, utc=True)
    ).tz_convert(None).normalize()


def find_column(frame, candidates):
    names = {
        str(c).lower(): c
        for c in frame.columns
    }

    for candidate in candidates:
        if candidate.lower() in names:
            return names[candidate.lower()]

    return None


def ticker_match(text, ticker):
    """
    Match explicit ticker tokens such as:
      NVDA
      COMPANY:NVDA
      TICKER_NVDA

    Avoid accidental substring matches.
    """
    pattern = rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])"
    return bool(
        re.search(
            pattern,
            str(text).upper()
        )
    )


def load_temporal_kg(path, dates, out):
    """
    Read actual Temporal KG events.

    Fail closed if schema/date/company links
    cannot be identified.

    No assertion that all graph edges are
    historically available until the source
    provenance has been independently checked.
    """

    log("===== LOAD TEMPORAL KG =====")

    raw = pd.read_csv(path, low_memory=False)

    head_col = find_column(
        raw,
        [
            "head", "subject", "src", "source",
            "source_entity", "head_entity",
            "subject_id", "src_entity"
        ],
    )

    tail_col = find_column(
        raw,
        [
            "tail", "object", "dst", "target",
            "target_entity", "tail_entity",
            "object_id", "dst_entity"
        ],
    )

    rel_col = find_column(
        raw,
        [
            "relation", "predicate", "rel",
            "relation_type", "edge_type"
        ],
    )

    time_col = find_column(
        raw,
        [
            "available_at", "available_from",
            "effective_date", "timestamp",
            "event_date", "date", "time"
        ],
    )

    print(
        "[KG SCHEMA]",
        {
            "head": head_col,
            "tail": tail_col,
            "relation": rel_col,
            "time": time_col,
        },
        flush=True,
    )

    if any(
        x is None
        for x in [head_col, tail_col, rel_col, time_col]
    ):
        raise RuntimeError(
            "Cannot verify KG event columns. "
            f"Available columns: {raw.columns.tolist()}. "
            "Set the correct column aliases in load_temporal_kg()."
        )

    frame = raw[
        [head_col, tail_col, rel_col, time_col]
    ].copy()

    frame.columns = [
        "head", "tail", "relation", "event_time"
    ]

    frame["event_time"] = pd.to_datetime(
        frame["event_time"],
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None).dt.normalize()

    frame = frame.dropna(
        subset=["event_time"]
    ).copy()

    # With date-only publication metadata, do not
    # allow the same-day event in the decision.
    frame["usable_from"] = (
        frame["event_time"]
        + pd.Timedelta(days=1)
    )

    frame["head"] = frame["head"].fillna("").astype(str)
    frame["tail"] = frame["tail"].fillna("").astype(str)
    frame["relation"] = (
        frame["relation"].fillna("").astype(str)
    )

    all_events = {}

    for ticker in TICKERS:
        mask = (
            frame["head"].map(
                lambda z: ticker_match(z, ticker)
            )
            |
            frame["tail"].map(
                lambda z: ticker_match(z, ticker)
            )
        )

        subset = frame.loc[mask].copy()

        subset = subset.sort_values(
            "usable_from",
            kind="stable"
        ).reset_index(drop=True)

        all_events[ticker] = subset

        log(
            f"[KG] {ticker}: "
            f"{len(subset)} directly linked events"
        )

    counts = {
        t: len(all_events[t])
        for t in TICKERS
    }

    if sum(counts.values()) == 0:
        raise RuntimeError(
            "No direct company-linked temporal KG events. "
            "Check entity IDs / node-to-ticker mapping. "
            "Stopping rather than silently using zero KG."
        )

    n = len(dates)

    # Features: log cumulative count, recent counts,
    # recent relation types and latest event age.
    kg = np.zeros(
        (n, N_STOCKS, KG_DIM),
        dtype=np.float32,
    )

    audit = []

    for j, ticker in enumerate(TICKERS):
        events = all_events[ticker]

        usable = events[
            "usable_from"
        ].to_numpy(dtype="datetime64[ns]")

        relations = events[
            "relation"
        ].str.upper().to_numpy()

        for t, date in enumerate(dates):
            day = np.datetime64(date)

            end = int(
                np.searchsorted(
                    usable, day, side="right"
                )
            )

            start_7 = int(
                np.searchsorted(
                    usable,
                    day - np.timedelta64(7, "D"),
                    side="left",
                )
            )

            start_30 = int(
                np.searchsorted(
                    usable,
                    day - np.timedelta64(30, "D"),
                    side="left",
                )
            )

            total = end
            recent7 = max(0, end - start_7)
            recent30 = max(0, end - start_30)

            recent_rels = relations[
                start_30:end
            ]

            supply = sum(
                "SUPPL" in r
                for r in recent_rels
            )

            compete = sum(
                "COMPET" in r
                for r in recent_rels
            )

            sec = sum(
                "SEC" in r
                for r in recent_rels
            )

            if total:
                age = float(
                    (day - usable[end - 1])
                    / np.timedelta64(1, "D")
                )
                recency = np.exp(
                    -max(age, 0.0) / 90.0
                )
                exists = 1.0
            else:
                recency = 0.0
                exists = 0.0

            kg[t, j] = [
                np.log1p(total),
                np.log1p(recent7),
                np.log1p(recent30),
                np.log1p(supply),
                np.log1p(compete),
                np.log1p(sec),
                recency,
                exists,
            ]

            latest = (
                events.iloc[end - 1]
                if end else None
            )

            audit.append({
                "date": str(date.date()),
                "ticker": ticker,
                "available_events": total,
                "recent30_events": recent30,
                "latest_relation": (
                    str(latest["relation"])
                    if latest is not None else ""
                ),
                "latest_head": (
                    str(latest["head"])
                    if latest is not None else ""
                ),
                "latest_tail": (
                    str(latest["tail"])
                    if latest is not None else ""
                ),
                "latest_usable_from": (
                    str(latest["usable_from"].date())
                    if latest is not None else ""
                ),
            })

    audit = pd.DataFrame(audit)
    audit.to_csv(
        out / "temporal_kg_retrieval_audit.csv",
        index=False,
    )

    coverage = float(
        np.mean(kg[:, :, -1] > 0)
    )

    log("[KG COVERAGE]", coverage)

    report = {
        "raw_edges": len(raw),
        "linked_event_counts": counts,
        "coverage": coverage,
        "kg_feature_shape": list(kg.shape),
        "kg_used_in_current_rag": True,
    }

    (out / "kg_info.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return kg, report


def market_context(prices):
    """
    5D historical market context per company.
    Compute across Train + Validation so January
    2021 has its 2020 price warm-up.
    """
    return v8.market_features(
        np.asarray(prices, dtype=np.float64)
    ).astype(np.float32)


def build_memory(
    dates,
    prices,
    context,
    sec,
    kg,
    n_train,
):
    """
    Historical cases only.

    Each case stores:
      company, decision time, maturity time,
      market/SEC/KG information at decision,
      realized 5-day excess return.

    A case becomes eligible for retrieval only
    after its target_date has matured.

    All cases used for 2021 come from Train.
    """

    records = []
    features = []
    targets = []

    for t in range(n_train - HORIZON):
        maturity = t + HORIZON

        if dates[maturity].year > 2020:
            continue

        stock_returns = (
            prices[maturity]
            / prices[t]
            - 1.0
        )

        benchmark = float(
            stock_returns.mean()
        )

        for j, ticker in enumerate(TICKERS):
            x = np.concatenate([
                context[t, j],
                sec[t, j],
                kg[t, j],
                np.eye(N_STOCKS)[j],
            ])

            excess = float(
                stock_returns[j] - benchmark
            )

            features.append(x)
            targets.append(excess)

            records.append({
                "ticker": ticker,
                "decision_date": str(
                    dates[t].date()
                ),
                "maturity_date": str(
                    dates[maturity].date()
                ),
                "decision_index": t,
                "maturity_index": maturity,
                "realized_excess_return": excess,
            })

    memory = pd.DataFrame(records)

    return (
        memory,
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.float64),
    )


def reflection_scores(
    dates,
    contexts,
    sec,
    kg,
    memory,
    memory_features,
    n_train,
    neighbors=12,
):
    """
    Retrieval-Augmented Reflection.

    Query by company and current evidence state.
    Retrieve only completed historical outcomes.

    Reflection is a weighted historical excess
    return estimate, not a guaranteed forecast.
    """

    n = len(dates)

    output = np.zeros(
        (n, N_STOCKS),
        dtype=np.float32,
    )

    audit = []

    # Standardize the query and case vectors
    # using TRAIN memory cases only.
    scaler = StandardScaler()
    Z = scaler.fit_transform(
        memory_features
    )

    maturity = memory[
        "maturity_index"
    ].to_numpy()

    company = memory[
        "ticker"
    ].to_numpy()

    for t in range(n_train, n):
        for j, ticker in enumerate(TICKERS):
            # A fixed 2018-2020 memory bank for
            # 2021; never uses 2021 realized returns.
            eligible = np.flatnonzero(
                (company == ticker)
                & (maturity < n_train)
                & (maturity <= t)
            )

            if len(eligible) == 0:
                continue

            x = np.concatenate([
                contexts[t, j],
                sec[t, j],
                kg[t, j],
                np.eye(N_STOCKS)[j],
            ]).reshape(1, -1)

            query = scaler.transform(x)[0]

            distances = np.mean(
                (Z[eligible] - query) ** 2,
                axis=1,
            )

            best = np.argsort(
                distances,
                kind="stable",
            )[:neighbors]

            selected = eligible[best]

            d = distances[best]

            weights = np.exp(
                -(d - d.min())
            )

            weights /= weights.sum()

            outcomes = memory[
                "realized_excess_return"
            ].to_numpy()[selected]

            estimate = float(
                np.dot(weights, outcomes)
            )

            output[t, j] = estimate

            audit.append({
                "date": str(dates[t].date()),
                "ticker": ticker,
                "reflection_score": estimate,
                "memory_count": len(selected),
                "case_ids": json.dumps(
                    selected.tolist()
                ),
                "latest_case_maturity": str(
                    memory.iloc[selected][
                        "maturity_date"
                    ].max()
                ),
            })

    return output, pd.DataFrame(audit)


def current_prior_model(
    memory_features,
    memory_targets,
):
    """
    Train a historical current-evidence
    prediction model.

    SEC and KG evidence are part of input.
    Stock identity is included.

    This is a simple controlled current RAG prior,
    not an LLM-generated investment decision.
    """

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0),
    )

    model.fit(
        memory_features,
        memory_targets,
    )

    return model


def predict_current(
    model,
    contexts,
    sec,
    kg,
):
    n = len(contexts)

    x = []

    for t in range(n):
        for j in range(N_STOCKS):
            x.append(
                np.concatenate([
                    contexts[t, j],
                    sec[t, j],
                    kg[t, j],
                    np.eye(N_STOCKS)[j],
                ])
            )

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    return model.predict(x).reshape(
        n, N_STOCKS
    )


def portfolio_softmax(values, scale=0.03):
    """
    Convert bounded expected excess returns
    to long-only reference weights.
    """
    x = np.clip(
        np.asarray(values, dtype=np.float64),
        -0.15,
        0.15,
    )

    x = x / scale
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def baseline_action(model, market, holdings):
    """
    Load the existing frozen V8 MARKET actor.

    Observation:
       1265 market/KGT
       320 zero SEC
       160 zero auxiliary
       5 holdings

    Total: 1750D
    """

    obs = np.concatenate([
        market,
        np.zeros(320, dtype=np.float32),
        np.zeros(160, dtype=np.float32),
        holdings,
    ]).astype(np.float32)

    if tuple(model.observation_space.shape) != (1750,):
        raise RuntimeError(
            f"Frozen Actor observation mismatch: "
            f"{model.observation_space.shape}"
        )

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    action = np.asarray(
        action,
        dtype=np.float64,
    ).reshape(-1)

    action = action - action.max()
    z = np.exp(action)

    return z / z.sum()


def backtest_arm(
    model,
    market,
    prices,
    dates,
    current,
    reflection,
    arm,
    alpha,
    transaction_cost,
):
    """
    Closed-loop strategy backtest.

    Each arm has its own evolving portfolio
    holdings and transaction costs.

    Frozen A2C is used identically across arms.
    """

    holdings = np.full(
        N_STOCKS,
        0.2,
        dtype=np.float64,
    )

    nav = 1.0
    rows = []

    for t in range(len(dates) - 1):
        rl = baseline_action(
            model,
            market[t],
            holdings,
        )

        if arm == "B0":
            rag = rl.copy()
            applied_alpha = 0.0

        else:
            score = current[t].copy()

            if arm == "B2":
                score += reflection[t]

            elif arm == "B3":
                score += np.roll(
                    reflection[t],
                    1,
                )

            rag = portfolio_softmax(
                score
            )

            applied_alpha = alpha

        target = (
            (1.0 - applied_alpha) * rl
            + applied_alpha * rag
        )

        turnover = float(
            np.abs(target - holdings).sum()
        )

        costs = transaction_cost * turnover

        stock_ret = (
            prices[t + 1]
            / prices[t]
            - 1.0
        )

        gross = float(
            np.dot(target, stock_ret)
        )

        net = gross - costs

        nav *= 1.0 + net

        drift = target * (
            1.0 + stock_ret
        )

        holdings = drift / drift.sum()

        row = {
            "date": str(dates[t].date()),
            "nav": nav,
            "daily_return": net,
            "turnover": turnover,
            "rag_rl_l1": float(
                np.abs(rag - rl).sum()
            ),
        }

        for j, ticker in enumerate(TICKERS):
            row[f"weight_{ticker}"] = (
                target[j]
            )
            row[f"rl_{ticker}"] = rl[j]
            row[f"rag_{ticker}"] = rag[j]

        rows.append(row)

    df = pd.DataFrame(rows)

    returns = df[
        "daily_return"
    ].to_numpy()

    navs = np.concatenate([
        [1.0],
        df["nav"].to_numpy(),
    ])

    peaks = np.maximum.accumulate(
        navs
    )

    metrics = {
        "arm": arm,
        "days": len(df),
        "total_return": float(
            navs[-1] - 1.0
        ),
        "sharpe": float(
            np.sqrt(252)
            * returns.mean()
            / max(
                returns.std(ddof=1),
                1e-12,
            )
        ),
        "max_drawdown": float(
            np.min(
                navs / peaks - 1.0
            )
        ),
        "volatility": float(
            returns.std(ddof=1)
            * np.sqrt(252)
        ),
        "mean_turnover": float(
            df["turnover"].mean()
        ),
        "mean_rag_rl_l1": float(
            df["rag_rl_l1"].mean()
        ),
    }

    return df, metrics


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/v9_reflection_kg_rag"
        ),
    )

    p.add_argument(
        "--documents",
        default="outputs/v31/evidence_alignment.parquet",
    )

    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--actor-dir",
        default=(
            "results/aegis_rl_v1/v8_phase2b_full"
        ),
    )

    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026],
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.2,
    )

    p.add_argument(
        "--neighbors",
        type=int,
        default=12,
    )

    p.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--max-val-days",
        type=int,
        default=0,
    )

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
        default="rag/data/processed/sec/sec_chunks.jsonl",
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

    p.add_argument(
        "--cost-scale",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )

    p.add_argument(
        "--tier",
        default="conservative",
    )

    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    if not 0 <= args.alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("===== LOAD FINRL-X + DYNAMIC KGT =====")

    train, val, dates, context, n_train = (
        v8b.load_data(args)
    )

    # Existing FinRL-X 1265D features.
    market = np.concatenate([
        train["market"],
        val["market"],
    ]).astype(np.float32)

    prices = np.concatenate([
        train["prices"],
        val["prices"],
    ]).astype(np.float64)

    dates = normalized_dates(dates)

    if market.shape[1] != MARKET_DIM:
        raise RuntimeError(
            "Expected verified 1265D Market/KGT features"
        )

    log("[MARKET/KGT]", market.shape)

    log("===== LOAD CURRENT SEC EVIDENCE =====")

    documents = v8.load_documents(
        args.documents
    )

    vectorizer, doc_vectors = (
        v8.make_document_vectors(
            documents
        )
    )

    sec, sec_audit = (
        v8b.build_asof_sec_retrieval(
            dates,
            context,
            documents,
            doc_vectors,
            vectorizer,
            args.top_k,
        )
    )

    sec_audit.to_csv(
        args.out / "sec_retrieval_audit.csv",
        index=False,
    )

    log("[SEC]", sec.shape)

    kg, kg_info = load_temporal_kg(
        args.temporal_kg,
        dates,
        args.out,
    )

    log("[TEMPORAL KG]", kg.shape)

    log("===== BUILD HISTORICAL REFLECTION MEMORY =====")

    memory, memory_x, memory_y = build_memory(
        dates,
        prices,
        context,
        sec,
        kg,
        n_train,
    )

    memory.to_csv(
        args.out / "historical_memory.csv",
        index=False,
    )

    log("[HISTORICAL CASES]", len(memory))

    log("===== TRAIN CURRENT EVIDENCE PRIOR =====")

    current_model = current_prior_model(
        memory_x,
        memory_y,
    )

    current = predict_current(
        current_model,
        context,
        sec,
        kg,
    )

    # Source-specific ablations use the SAME
    # frozen current-evidence prediction model.
    # They do not refit it on ablated inputs.
    current_without_sec = predict_current(
        current_model,
        context,
        np.zeros_like(sec),
        kg,
    )

    current_without_kg = predict_current(
        current_model,
        context,
        sec,
        np.zeros_like(kg),
    )

    log("===== RETRIEVE REFLECTION MEMORY =====")

    reflection, reflection_audit = (
        reflection_scores(
            dates,
            context,
            sec,
            kg,
            memory,
            memory_x,
            n_train,
            neighbors=args.neighbors,
        )
    )

    reflection_audit.to_csv(
        args.out / "reflection_retrieval_audit.csv",
        index=False,
    )

    # Train/validation separated after constructing
    # historically warmed-up market features.
    val_market = market[n_train:]
    val_prices = prices[n_train:]
    val_dates = dates[n_train:]
    val_current = current[n_train:]
    val_reflection = reflection[n_train:]
    val_no_sec = current_without_sec[n_train:]
    val_no_kg = current_without_kg[n_train:]

    if args.max_val_days > 0:
        limit = min(
            len(val_dates),
            args.max_val_days + 1,
        )

        val_market = val_market[:limit]
        val_prices = val_prices[:limit]
        val_dates = val_dates[:limit]
        val_current = val_current[:limit]
        val_reflection = val_reflection[:limit]
        val_no_sec = val_no_sec[:limit]
        val_no_kg = val_no_kg[:limit]

    # FinRL-X original 2021 validation
    # should contain 252 dates and 251 transitions.
    log(
        "[VALIDATION]",
        len(val_dates),
        "dates",
    )

    results = []

    for seed in args.seeds:
        log()
        log("===== FROZEN A2C SEED", seed, "=====")

        actor_path = (
            Path(args.actor_dir)
            / f"MARKET_seed{seed}"
            / "model.zip"
        )

        if not actor_path.exists():
            raise FileNotFoundError(
                f"Missing MARKET actor: {actor_path}"
            )

        actor = A2C.load(
            str(actor_path),
            device="cpu",
        )

        for arm in ARMS:
            history, metrics = backtest_arm(
                actor,
                val_market,
                val_prices,
                val_dates,
                val_current,
                val_reflection,
                arm,
                args.alpha,
                args.transaction_cost,
            )

            metrics["seed"] = seed

            results.append(metrics)

            run_dir = (
                args.out
                / f"seed{seed}"
            )

            run_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            history.to_csv(
                run_dir / f"{arm}_history.csv",
                index=False,
            )

            log(
                f"[{arm}] "
                f"Return={metrics['total_return']:+.4%} "
                f"Sharpe={metrics['sharpe']:.4f} "
                f"MDD={metrics['max_drawdown']:.4%}"
            )

        # Fixed-policy, source-specific ablations.
        # Change one current-evidence source only.
        for mode, ablated_current in [
            ("B1_NO_SEC", val_no_sec),
            ("B1_NO_KG", val_no_kg),
        ]:
            history, metrics = backtest_arm(
                actor,
                val_market,
                val_prices,
                val_dates,
                ablated_current,
                val_reflection,
                "B1",
                args.alpha,
                args.transaction_cost,
            )

            metrics["arm"] = mode
            metrics["seed"] = seed
            results.append(metrics)

            history.to_csv(
                args.out / f"seed{seed}" / f"{mode}_history.csv",
                index=False,
            )

            log(
                f"[{mode}] "
                f"Return={metrics['total_return']:+.4%}"
            )

    df = pd.DataFrame(results)

    df.to_csv(
        args.out / "validation_results.csv",
        index=False,
    )

    comparison = df.pivot(
        index="seed",
        columns="arm",
        values="total_return",
    )

    for arm in ["B1", "B2", "B3"]:
        comparison[f"{arm}_minus_B0_pp"] = (
            comparison[arm] - comparison["B0"]
        ) * 100.0

    comparison["B2_minus_B1_pp"] = (
        comparison["B2"] - comparison["B1"]
    ) * 100.0

    comparison["B2_minus_B3_pp"] = (
        comparison["B2"] - comparison["B3"]
    ) * 100.0

    comparison.to_csv(
        args.out / "paired_comparison.csv"
    )

    report = {
        "train": "2018-2020",
        "validation": "2021",
        "test_used": False,
        "seeds": args.seeds,
        "arms": ARMS,
        "alpha": args.alpha,
        "transaction_cost": args.transaction_cost,
        "neighbors": args.neighbors,
        "current_sec_documents": len(documents),
        "historical_cases": len(memory),
        "dynamic_kgt_in_market": True,
        "temporal_kg_retrieval": kg_info,
        "note": (
            "Frozen A2C baseline, historical SEC/Temporal KG "
            "retrieval, matured reflection memory, "
            "portfolio prior fusion."
        ),
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("===== V9 VALIDATION RESULTS =====")
    log(df.to_string(index=False))

    log()
    log("===== PAIRED COMPARISON =====")
    log(comparison.to_string())

    log()
    log("V9 REFLECTION KG-RAG: PASS")
    log("TEST 2022-2023 NOT USED")


if __name__ == "__main__":
    main()
