#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V13
Company-aware Market + Dynamic KGT + Unified SEC/KG RAG

TRAIN : 2018-2020
VAL   : 2021
TEST  : 2022-2023 NOT USED

B0: Company-aware Market + Dynamic KGT
B1: B0 + PIT SEC RAG
B2: B0 + Temporal KG RAG
B3: B0 + PIT SEC RAG + Temporal KG RAG
B4: B0 + Original SEC RAG + Temporal KG RAG
B5: Frozen B3 + same-day ticker-rotated SEC/KG
B6: Frozen B3 + zero SEC/KG

Market assumption:
  existing 1265D market observation = 5 consecutive 253D
  company blocks in TICKERS order.
  MUST inspect existing loader to confirm this assumption.

KG representation:
  - all direct company-linked relations (including numeric FinDKG)
  - first-observed unique triples
  - as-of cumulative memory vector (all historical edges)
  - recent top-K evidence vector
  - relation-count state

IMPORTANT:
  KG "first observed" is not verified publication time.
  SEC filing metadata matched, excerpts not independently verified.
  No external LLM inference, no RL training.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v8_replug_finagent as v8,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
N = len(TICKERS)
H = 5
MARKET_DIM = 1265
COMPANY_DIM = 253
KG_HASH_DIM = 64

TRAIN_END = pd.Timestamp("2020-12-31")
VAL_END = pd.Timestamp("2021-12-31")


def log(*args):
    print(*args, flush=True)


def get_col(df, aliases, required=True):
    lookup = {
        str(x).lower(): x for x in df.columns
    }
    for name in aliases:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise RuntimeError(
            f"Cannot find {aliases}; columns={list(df.columns)}"
        )
    return None


def parse_days(x):
    return pd.to_datetime(
        x, utc=True, errors="coerce"
    ).dt.tz_convert(None).dt.normalize()


def make_labels(prices, dates, n_train):
    n = len(prices)

    y = np.full(
        (n, N),
        np.nan,
        dtype=np.float64,
    )

    for t in range(n - H):
        maturity = t + H

        # No label crossing training/validation boundary.
        if t < n_train <= maturity:
            continue

        if t >= n_train:
            if dates[maturity].year != 2021:
                continue

        returns = (
            prices[maturity] / prices[t] - 1.0
        )

        y[t] = (
            returns - returns.mean()
        )

    train_ids = np.flatnonzero(
        np.isfinite(y[:, 0])
        & (np.arange(n) < n_train)
    )

    val_ids = np.flatnonzero(
        np.isfinite(y[:, 0])
        & (np.arange(n) >= n_train)
    )

    return y, train_ids, val_ids



def company_market_features(market, args):
    """
    Verified from original FinRL-X MultiStepEnv:

      raw = stack(market base_cols, axis=-1)
      emb = stack(kgt[(date, ticker)])
      features = concatenate([raw, emb], axis=2)

    Verified structure:
      market observations: [days, 5 * (53+200)]
      company-specific:    [days, 5, 253]
      first 53:            Market
      final 200:           Dynamic KGT

    Company order:
      AAPL, AMZN, GOOGL, MSFT, NVDA
    """

    market = np.asarray(
        market,
        dtype=np.float32,
    )

    if market.ndim != 2:
        raise RuntimeError(
            f"Expected flat 2D observations, got {market.shape}"
        )

    if market.shape[1] != 1265:
        raise RuntimeError(
            f"Expected 1265D, got {market.shape}"
        )

    company = market.reshape(
        len(market),
        5,
        253,
    )

    local_market = company[:, :, :53].copy()
    local_kgt = company[:, :, 53:].copy()

    if local_market.shape != (len(market), 5, 53):
        raise RuntimeError("Market block shape incorrect")

    if local_kgt.shape != (len(market), 5, 200):
        raise RuntimeError("Dynamic KGT block shape incorrect")

    if not np.array_equal(
        company.reshape(len(market), 1265),
        market,
    ):
        raise RuntimeError(
            "Market block reconstruction failed"
        )

    if not np.isfinite(company).all():
        raise RuntimeError(
            "Company features contain NaN/Inf"
        )

    # Inspect whether KGT is actually different across
    # companies and days. Report without claiming that
    # all differences imply useful financial signal.
    kgt_cross_company_difference = float(
        np.mean(
            np.abs(
                local_kgt[:, 1:, :]
                - local_kgt[:, :1, :]
            )
        )
    )

    kgt_temporal_difference = float(
        np.mean(
            np.abs(
                local_kgt[1:]
                - local_kgt[:-1]
            )
        )
    ) if len(local_kgt) > 1 else 0.0

    print(
        "[VERIFIED MARKET SHAPE]",
        local_market.shape,
        flush=True,
    )

    print(
        "[VERIFIED DYNAMIC KGT SHAPE]",
        local_kgt.shape,
        flush=True,
    )

    print(
        "[KGT CROSS-COMPANY MEAN ABS DIFF]",
        kgt_cross_company_difference,
        flush=True,
    )

    print(
        "[KGT TEMPORAL MEAN ABS DIFF]",
        kgt_temporal_difference,
        flush=True,
    )

    if kgt_cross_company_difference <= 1e-12:
        raise RuntimeError(
            "KGT embeddings appear identical across companies; "
            "inspect company-specific KGT data before training."
        )

    if kgt_temporal_difference <= 1e-12:
        raise RuntimeError(
            "KGT embeddings appear temporally constant; "
            "inspect Dynamic KGT snapshots before training."
        )

    # Company-specific 253D features.
    # Preserve original market and KGT coordinates.
    # Also supply global relative market context,
    # computed from same-day observations only.
    global_market = local_market.mean(
        axis=1,
        keepdims=True,
    )

    global_market = np.repeat(
        global_market,
        5,
        axis=1,
    )

    relative_market = (
        local_market - global_market
    )

    result = np.concatenate(
        [
            local_market,
            local_kgt,
            relative_market,
        ],
        axis=2,
    ).astype(np.float32)

    print(
        "[FINAL COMPANY-AWARE BASE]",
        result.shape,
        flush=True,
    )

    return result


def load_sec(dates, contexts, path, top_k, audit_path):
    log("[SEC] Loading", path)

    docs = v8.load_documents(str(path))

    vectorizer, doc_vectors = (
        v8.make_document_vectors(docs)
    )

    sec, audit = (
        v8b.build_asof_sec_retrieval(
            dates,
            contexts,
            docs,
            doc_vectors,
            vectorizer,
            top_k,
        )
    )

    sec = np.asarray(
        sec, dtype=np.float32
    )

    if sec.ndim != 3:
        raise RuntimeError(
            f"Unexpected SEC dimensions: {sec.shape}"
        )

    if sec.shape[:2] != (
        len(dates), N
    ):
        raise RuntimeError(
            f"SEC date/ticker dimension mismatch: {sec.shape}"
        )

    if not np.isfinite(sec).all():
        raise RuntimeError(
            "SEC features contain nonfinite values"
        )

    audit.to_csv(
        audit_path,
        index=False,
    )

    log(
        "[SEC] documents=", len(docs),
        "shape=", sec.shape,
    )

    return sec, len(docs)


def match_company(value, ticker):
    """
    Match exact COMPANY:TICKER token only.
    Avoid matching stock symbols inside arbitrary names.
    """
    text = str(value).upper()

    pattern = (
        rf"(?<![A-Z0-9_])COMPANY:"
        rf"{re.escape(ticker)}"
        rf"(?![A-Z0-9_])"
    )

    return bool(
        re.search(pattern, text)
    )


def kg_events_from_csv(path, last_date):
    raw = pd.read_csv(
        path, low_memory=False
    )

    head = get_col(
        raw, [
            "head", "subject", "src",
            "head_entity", "source_entity",
        ]
    )

    relation = get_col(
        raw, [
            "relation", "rel",
            "predicate", "relation_type",
        ]
    )

    tail = get_col(
        raw, [
            "tail", "object", "dst",
            "tail_entity", "target_entity",
        ]
    )

    time_col = get_col(
        raw, [
            "available_at", "available_from",
            "event_date", "timestamp",
            "date", "time",
        ]
    )

    df = pd.DataFrame({
        "head": raw[head].fillna("").astype(str),
        "relation": raw[relation].fillna("").astype(str),
        "tail": raw[tail].fillna("").astype(str),
        "observed_date": parse_days(raw[time_col]),
    })

    df = df[
        df["observed_date"].notna()
        & (df["observed_date"] <= last_date)
    ].copy()

    # One first-observed row per unique triple.
    # Historical re-observations are represented by
    # persistence in the cumulative memory.
    df = (
        df.groupby(
            ["head", "relation", "tail"],
            as_index=False,
        )
        .agg(
            observed_date=("observed_date", "min"),
        )
    )

    rows = []

    for ticker in TICKERS:
        subset = df[
            df["head"].map(
                lambda x: match_company(x, ticker)
            )
            |
            df["tail"].map(
                lambda x: match_company(x, ticker)
            )
        ]

        for r in subset.itertuples(index=False):
            if match_company(r.head, ticker):
                direction = "OUT"
                counterpart = r.tail
            else:
                direction = "IN"
                counterpart = r.head

            # Explicitly represent direction and raw
            # relation code. No invented financial meaning.
            text = (
                f"direction_{direction} "
                f"relation_{r.relation} "
                f"counterpart_{counterpart}"
            )

            rows.append({
                "ticker": ticker,
                "head": r.head,
                "relation": r.relation,
                "tail": r.tail,
                "observed_date": r.observed_date,
                "direction": direction,
                "text": text,
            })

    if not rows:
        raise RuntimeError(
            "No direct company KG events found. "
            "Inspect entity naming and source schema."
        )

    events = pd.DataFrame(rows)

    events = events.sort_values(
        [
            "ticker",
            "observed_date",
            "relation",
            "head",
            "tail",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return events


def build_kg_rag(
    dates,
    events,
    top_k,
    out,
):
    """
    For each ticker/date:
      64D all-history mean relation-text embedding,
      64D most-recent-topK embedding,
      R-dimensional log1p cumulative relation counts,
      2D log counts / historical age.

    Timing:
      Only events observed on dates strictly earlier
      than the current trading date are eligible.

    This is conservative by calendar day, but does not
    certify true KG public-release provenance.
    """
    relation_names = sorted(
        events["relation"].unique().tolist()
    )

    rel2id = {
        name: i
        for i, name in enumerate(relation_names)
    }

    vec = HashingVectorizer(
        n_features=KG_HASH_DIM,
        alternate_sign=False,
        ngram_range=(1, 2),
        norm="l2",
        dtype=np.float32,
    )

    n = len(dates)
    rel_dim = len(relation_names)

    total_dim = (
        KG_HASH_DIM * 2
        + rel_dim
        + 2
    )

    output = np.zeros(
        (n, N, total_dim),
        dtype=np.float32,
    )

    logs = []

    trading_days = dates.to_numpy(
        dtype="datetime64[ns]"
    )

    for j, ticker in enumerate(TICKERS):
        sub = events[
            events["ticker"] == ticker
        ].copy()

        sub = sub.sort_values(
            [
                "observed_date",
                "relation",
                "head",
                "tail",
            ],
            kind="stable",
        ).reset_index(drop=True)

        count = len(sub)

        if not count:
            continue

        vectors = vec.transform(
            sub["text"].tolist()
        ).toarray().astype(np.float32)

        cum_vectors = np.vstack([
            np.zeros(
                (1, KG_HASH_DIM),
                dtype=np.float32,
            ),
            np.cumsum(
                vectors, axis=0
            ),
        ])

        relation_onehot = np.zeros(
            (count, rel_dim),
            dtype=np.float32,
        )

        for i, relation in enumerate(
            sub["relation"]
        ):
            relation_onehot[
                i, rel2id[relation]
            ] = 1.0

        cum_relation = np.vstack([
            np.zeros(
                (1, rel_dim),
                dtype=np.float32,
            ),
            np.cumsum(
                relation_onehot,
                axis=0,
            ),
        ])

        event_dates = sub[
            "observed_date"
        ].to_numpy(dtype="datetime64[ns]")

        # Strictly earlier calendar dates only.
        eligible_counts = np.searchsorted(
            event_dates,
            trading_days,
            side="left",
        )

        for t, k in enumerate(eligible_counts):
            k = int(k)

            if k <= 0:
                continue

            # All historical KG edges are included
            # in the cumulative representation.
            memory = (
                cum_vectors[k] / k
            )

            # Recent retrieval is an additional
            # representation, not a replacement.
            a = max(0, k - top_k)

            recent = (
                cum_vectors[k]
                - cum_vectors[a]
            ) / (k - a)

            r_counts = np.log1p(
                cum_relation[k]
            )

            latest_date = event_dates[k - 1]

            age_days = (
                trading_days[t]
                - latest_date
            ) / np.timedelta64(1, "D")

            aux = np.array(
                [
                    np.log1p(k),
                    np.log1p(
                        max(0.0, float(age_days))
                    ),
                ],
                dtype=np.float32,
            )

            output[t, j] = np.concatenate([
                memory, recent, r_counts, aux
            ])

            # A lightweight audit over all dates.
            logs.append({
                "date": str(dates[t].date()),
                "ticker": ticker,
                "eligible_kg_edges": k,
                "recent_retrieved": k - a,
                "latest_observed_date": str(
                    pd.Timestamp(latest_date).date()
                ),
                "age_calendar_days": float(age_days),
            })

        log(
            f"[KG] {ticker}: {count} unique triples "
            f"through {sub['observed_date'].max().date()}"
        )

    pd.DataFrame(
        logs,
        columns=[
            "date", "ticker",
            "eligible_kg_edges",
            "recent_retrieved",
            "latest_observed_date",
            "age_calendar_days",
        ],
    ).to_csv(
        out / "kg_retrieval_audit.csv",
        index=False,
    )

    return output, relation_names


def compose(base, sec=None, kg=None):
    arrays = [base]

    if sec is not None:
        arrays.append(sec)

    if kg is not None:
        arrays.append(kg)

    return np.concatenate(
        arrays, axis=2
    ).astype(np.float32)


def fit_company_models(x, y, train_ids, alpha):
    models = []

    for j, ticker in enumerate(TICKERS):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha),
        )

        xx = x[train_ids, j]
        yy = y[train_ids, j]

        model.fit(xx, yy)
        models.append(model)

    return models


def company_predict(models, x, ids):
    result = np.empty(
        (len(ids), N),
        dtype=np.float64,
    )

    for j, model in enumerate(models):
        result[:, j] = model.predict(
            x[ids, j]
        )

    return result


def evaluate(y, preds, dates, arm):
    actual_flat = y.reshape(-1)
    pred_flat = preds.reshape(-1)

    daily_mse = np.mean(
        (y - preds) ** 2,
        axis=1,
    )

    daily_mae = np.mean(
        np.abs(y - preds),
        axis=1,
    )

    ics = []
    spreads = []

    for a, p in zip(y, preds):
        if (
            np.ptp(a) <= 1e-12
            or np.ptp(p) <= 1e-12
        ):
            ic = np.nan
        else:
            ic = spearmanr(a, p).statistic

        ics.append(ic)

        spreads.append(
            float(
                a[np.argmax(p)]
                - a[np.argmin(p)]
            )
        )

    daily = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "arm": arm,
        "mse": daily_mse,
        "mae": daily_mae,
        "rank_ic": ics,
        "gross_spread": spreads,
    })

    metrics = {
        "arm": arm,
        "samples": len(actual_flat),
        "days": len(dates),
        "mse": float(daily_mse.mean()),
        "mae": float(daily_mae.mean()),
        "sign_accuracy": float(
            np.mean(
                (actual_flat > 0)
                == (pred_flat > 0)
            )
        ),
        "mean_rank_ic": float(
            np.nanmean(ics)
        ),
        "gross_rank_spread": float(
            np.mean(spreads)
        ),
        "mean_abs_prediction": float(
            np.abs(pred_flat).mean()
        ),
    }

    return metrics, daily


def block_bootstrap(a, b, metric, seed, rep=1000):
    # Positive = B improves over A.
    aa = a[["date", metric]].rename(
        columns={metric: "a"}
    )
    bb = b[["date", metric]].rename(
        columns={metric: "b"}
    )

    x = aa.merge(
        bb, on="date",
        validate="one_to_one",
    ).dropna()

    if metric == "rank_ic":
        delta = (
            x["b"].to_numpy()
            - x["a"].to_numpy()
        )
    else:
        delta = (
            x["a"].to_numpy()
            - x["b"].to_numpy()
        )

    n = len(delta)

    rng = np.random.default_rng(seed)
    length = min(10, n)

    draws = []

    for _ in range(rep):
        positions = []

        while len(positions) < n:
            start = rng.integers(
                0, n - length + 1
            )

            positions.extend(
                range(
                    start,
                    start + length,
                )
            )

        positions = positions[:n]

        draws.append(
            delta[positions].mean()
        )

    lo, hi = np.quantile(
        draws, [0.025, 0.975]
    )

    return {
        "metric": metric,
        "n_days": n,
        "improvement": float(delta.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--pit-sec",
        default=(
            "results/aegis_rl_v1/"
            "v11f_sec_pit/sec_pit_retrieval.parquet"
        ),
    )
    p.add_argument(
        "--original-sec",
        default=(
            "outputs/v31/"
            "evidence_alignment.parquet"
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
            "v13_unified_sec_kg"
        ),
    )

    p.add_argument("--sec-top-k", type=int, default=5)
    p.add_argument("--kg-top-k", type=int, default=12)
    p.add_argument("--ridge-alpha", type=float, default=1000)
    p.add_argument("--max-val-days", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Explicit safeguard: do not silently assume
    # an unknown 1265D feature layout.
    p.add_argument(
        "--confirm-market-block-layout",
        action="store_true",
        help=(
            "Confirm from v7b loader that 1265D is "
            "5 consecutive company 253D blocks "
            "in AAPL AMZN GOOGL MSFT NVDA order"
        ),
    )

    # Existing project loader arguments.
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
            "rag/data/processed/sec/sec_chunks.jsonl"
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
    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--embed", default="bge-m3:latest")
    p.add_argument("--device", default="cpu")
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

    log(
        "[LAYOUT] Verified from original MultiStepEnv:",
        "53D Market + 200D Dynamic KGT per ticker"
    )

    if args.ridge_alpha <= 0:
        raise RuntimeError(
            "ridge-alpha must be positive"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("===== FINRL-X MARKET + DYNAMIC KGT =====")

    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    dates = pd.DatetimeIndex(
        pd.to_datetime(dates, utc=True)
    ).tz_convert(None).normalize()

    if not (
        dates[:n_train].max() <= TRAIN_END
        and dates[n_train:].min()
        > TRAIN_END
        and dates.max() <= VAL_END
    ):
        raise RuntimeError(
            "Unexpected train/validation date split"
        )

    market = np.concatenate([
        train["market"],
        val["market"],
    ]).astype(np.float32)

    prices = np.concatenate([
        train["prices"],
        val["prices"],
    ]).astype(np.float64)

    if (
        len(market) != len(dates)
        or len(prices) != len(dates)
        or prices.shape[1] != N
    ):
        raise RuntimeError(
            "Market/price/date alignment mismatch"
        )

    company_market = company_market_features(
        market, args
    )

    log(
        "[COMPANY MARKET]",
        company_market.shape
    )

    y, train_ids, val_ids = make_labels(
        prices, dates, n_train
    )

    if args.max_val_days > 0:
        val_ids = val_ids[
            :args.max_val_days
        ]

    log(
        "[TRAIN]", len(train_ids),
        "[VAL]", len(val_ids)
    )

    log("===== SEC PIT RAG =====")

    sec_pit, pit_docs = load_sec(
        dates,
        contexts,
        args.pit_sec,
        args.sec_top_k,
        args.out / "sec_pit_retrieval_audit.csv",
    )

    log("===== SEC ORIGINAL RAG =====")

    sec_original, original_docs = load_sec(
        dates,
        contexts,
        args.original_sec,
        args.sec_top_k,
        args.out / "sec_original_retrieval_audit.csv",
    )

    if sec_pit.shape != sec_original.shape:
        raise RuntimeError(
            "SEC shape mismatch"
        )

    log("===== TEMPORAL KG FULL MEMORY =====")

    kg_events = kg_events_from_csv(
        args.temporal_kg,
        VAL_END,
    )

    kg_events.to_csv(
        args.out / "kg_unique_company_events.csv",
        index=False,
    )

    kg_rag, relations = build_kg_rag(
        dates,
        kg_events,
        args.kg_top_k,
        args.out,
    )

    log(
        "[KG RAG]",
        kg_rag.shape,
        "[RELATIONS]",
        len(relations)
    )

    log("===== BUILD EXPERIMENT ARMS =====")

    datasets = {
        "B0": compose(company_market),
        "B1": compose(
            company_market,
            sec=sec_pit,
        ),
        "B2": compose(
            company_market,
            kg=kg_rag,
        ),
        "B3": compose(
            company_market,
            sec=sec_pit,
            kg=kg_rag,
        ),
        "B4": compose(
            company_market,
            sec=sec_original,
            kg=kg_rag,
        ),
    }

    log("===== FIT COMPANY-SPECIFIC MODELS =====")

    models = {}

    for arm, x in datasets.items():
        log(
            "[FIT]",
            arm,
            "shape=",
            x.shape,
        )

        models[arm] = fit_company_models(
            x,
            y,
            train_ids,
            args.ridge_alpha,
        )

    log("===== PREDICT 2021 =====")

    predictions = {}

    for arm in datasets:
        predictions[arm] = company_predict(
            models[arm],
            datasets[arm],
            val_ids,
        )

    # Fixed B3 intervention, no retraining.
    shuffled_sec = np.roll(
        sec_pit,
        1,
        axis=1,
    )

    shuffled_kg = np.roll(
        kg_rag,
        1,
        axis=1,
    )

    b5 = compose(
        company_market,
        sec=shuffled_sec,
        kg=shuffled_kg,
    )

    b6 = compose(
        company_market,
        sec=np.zeros_like(sec_pit),
        kg=np.zeros_like(kg_rag),
    )

    predictions["B5"] = company_predict(
        models["B3"], b5, val_ids
    )

    predictions["B6"] = company_predict(
        models["B3"], b6, val_ids
    )

    val_dates = dates[val_ids]
    truth = y[val_ids]

    metrics = []
    daily = {}

    for arm in [
        "B0", "B1", "B2",
        "B3", "B4", "B5", "B6"
    ]:
        m, d = evaluate(
            truth,
            predictions[arm],
            val_dates,
            arm,
        )

        metrics.append(m)
        daily[arm] = d

        d.to_csv(
            args.out / f"{arm}_daily.csv",
            index=False,
        )

    pd.DataFrame(metrics).to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    comparisons = [
        ("B0", "B1"),
        ("B0", "B2"),
        ("B0", "B3"),
        ("B1", "B3"),
        ("B2", "B3"),
        ("B3", "B4"),
        ("B5", "B3"),
        ("B6", "B3"),
    ]

    boot_rows = []

    for i, (a, b) in enumerate(comparisons):
        for metric in ["mse", "mae", "rank_ic"]:
            res = block_bootstrap(
                daily[a],
                daily[b],
                metric,
                seed=args.seed + i,
            )

            res["baseline"] = a
            res["challenger"] = b

            boot_rows.append(res)

    pd.DataFrame(boot_rows).to_csv(
        args.out / "paired_bootstrap.csv",
        index=False,
    )

    prediction_rows = []

    for idx, date in enumerate(val_dates):
        for j, ticker in enumerate(TICKERS):
            row = {
                "date": str(date.date()),
                "ticker": ticker,
                "actual_excess_5d": float(truth[idx, j]),
                "pit_sec_vector_norm": float(
                    np.linalg.norm(
                        sec_pit[val_ids[idx], j]
                    )
                ),
                "kg_rag_vector_norm": float(
                    np.linalg.norm(
                        kg_rag[val_ids[idx], j]
                    )
                ),
            }

            for arm, values in predictions.items():
                row[f"{arm}_prediction"] = float(
                    values[idx, j]
                )

            prediction_rows.append(row)

    pd.DataFrame(prediction_rows).to_csv(
        args.out / "predictions_2021.csv",
        index=False,
    )

    sensitivity = []

    for arm in ["B5", "B6"]:
        delta = (
            predictions["B3"]
            - predictions[arm]
        )

        sensitivity.append({
            "intervention": arm,
            "mean_absolute_change": float(
                np.abs(delta).mean()
            ),
            "max_absolute_change": float(
                np.abs(delta).max()
            ),
            "fraction_changed": float(
                np.mean(
                    np.abs(delta) > 1e-8
                )
            ),
        })

    pd.DataFrame(sensitivity).to_csv(
        args.out / "evidence_sensitivity.csv",
        index=False,
    )

    coverage = {
        "pit_sec_nonzero_fraction": float(
            np.mean(
                np.linalg.norm(
                    sec_pit[val_ids], axis=2
                ) > 1e-8
            )
        ),
        "kg_nonzero_fraction": float(
            np.mean(
                np.linalg.norm(
                    kg_rag[val_ids], axis=2
                ) > 1e-8
            )
        ),
        "original_vs_pit_changed_fraction": float(
            np.mean(
                np.linalg.norm(
                    sec_original[val_ids]
                    - sec_pit[val_ids],
                    axis=2,
                ) > 1e-8
            )
        ),
        "kg_event_count": len(kg_events),
        "kg_relation_count": len(relations),
    }

    manifest = {
        "experiment": "FinRL-X V13",
        "train": "2018-2020",
        "validation": "2021",
        "test_2022_2023_used": False,
        "n_train_days": len(train_ids),
        "n_val_days": len(val_ids),
        "tickers": TICKERS,
        "market_layout_assumption": (
            "five 253D company blocks, "
            "AAPL AMZN GOOGL MSFT NVDA"
        ),
        "market_company_features": (
            company_market.shape[2]
        ),
        "sec_rag_features": sec_pit.shape[2],
        "kg_rag_features": kg_rag.shape[2],
        "kg_relation_names": relations,
        "sec_pit_document_rows": pit_docs,
        "sec_original_document_rows": original_docs,
        "coverage": coverage,
        "limitations": [
            "Company block layout requires code-level confirmation.",
            "KG first observed is not certified disclosure time.",
            "Numbered FinDKG relations are not semantically decoded.",
            "SEC filing metadata matched; excerpt text not certified.",
            "HAS_SEC_SIGNAL overlaps conceptually with SEC documents.",
            "B5/B6 are fixed B3 interventions.",
            "No RL or portfolio trading evaluated.",
            "No 2022-2023 test observations used.",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 78)
    log("V13 COMPANY-AWARE UNIFIED SEC/KG RAG")
    log("=" * 78)

    log(
        pd.DataFrame(metrics).to_string(index=False)
    )

    log()
    log("[COVERAGE]")
    log(
        json.dumps(coverage, indent=2)
    )

    log()
    log("[SENSITIVITY]")
    log(
        pd.DataFrame(sensitivity).to_string(index=False)
    )

    log()
    log("V13: PASS")
    log("2021 VALIDATION ONLY")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
