#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V11-A: Event-centric KG-RAG Reflection Audit

Sources:
    SEC Evidence Alignment parquet
    Unified Temporal KG CSV
    Original FinRL-X price loader

Event types:
    SEC_<TOPIC>
    KG_SUPPLIES_TO
    KG_COMPETES_WITH
    KG_CUSTOMER_OF
    KG_<OTHER_RELATION>

Outputs:
    event_memory.csv
    event_retrieval_audit.csv
    event_predictions_2020.csv
    method_metrics.csv
    paired_comparison.csv
    source_coverage.csv
    experiment_manifest.json

Temporal rules:
    1. Events become usable one calendar day AFTER
       their source available_from/event date.
    2. Event decision is the first available trading date.
    3. Forward outcomes start at that decision date.
    4. Historical cases are retrieved only when their
       outcomes matured strictly BEFORE query date.
    5. Evaluation: 2020 only.
    6. 2021 and 2022-2023 labels are never evaluated.

This is event-level retrieval/reflection, not
a portfolio-trading performance claim.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]


def log(*args):
    print(*args, flush=True)


def first_column(frame, names, required=False):
    lookup = {
        str(c).lower(): c
        for c in frame.columns
    }

    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]

    if required:
        raise RuntimeError(
            f"Missing column among {names}. "
            f"Available: {frame.columns.tolist()}"
        )

    return None


def parse_dates(series):
    return pd.to_datetime(
        series,
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None).dt.normalize()


def clean_type(s):
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]+", "_", s)
    return s.strip("_")[:70] or "GENERAL"


def match_company(s, ticker):
    # Handles NVDA, COMPANY:NVDA, etc.
    # Does not assume numeric KG entity IDs
    # magically map to ticker identities.
    return bool(
        re.search(
            rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])",
            str(s).upper()
        )
    )


def stable_id(*parts):
    payload = "|".join(map(str, parts))
    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]


def sec_events(path):
    log("[SEC] Reading", path)

    df = pd.read_parquet(path)

    ticker_col = first_column(
        df, ["ticker", "company"], True
    )
    time_col = first_column(
        df, ["available_from"], True
    )
    text_col = first_column(
        df, ["current_excerpt"], True
    )

    topic_col = first_column(
        df, ["topic", "section", "item"]
    )

    filing_col = first_column(
        df,
        ["current_filing_date", "filing_date"]
    )

    source_col = first_column(
        df,
        ["evidence_id", "pair_id", "source_id"]
    )

    novelty_col = first_column(
        df,
        ["is_new_text_candidate"]
    )

    similarity_col = first_column(
        df,
        ["similarity"]
    )

    work = pd.DataFrame({
        "ticker": df[ticker_col].astype(str).str.upper(),
        "source_time": parse_dates(df[time_col]),
        "evidence_text": (
            df[text_col].fillna("").astype(str)
        ),
        "topic": (
            df[topic_col].fillna("GENERAL").astype(str)
            if topic_col else "GENERAL"
        ),
        "filing_date": (
            parse_dates(df[filing_col])
            if filing_col else parse_dates(df[time_col])
        ),
        "original_id": (
            df[source_col].astype(str)
            if source_col else df.index.astype(str)
        ),
        "novelty_candidate": (
            df[novelty_col].fillna(False).astype(bool)
            if novelty_col else False
        ),
        "alignment_similarity": (
            pd.to_numeric(
                df[similarity_col],
                errors="coerce",
            )
            if similarity_col else np.nan
        ),
    })

    work = work[
        work["ticker"].isin(TICKERS)
        & work["source_time"].notna()
        & (work["evidence_text"].str.len() >= 25)
    ].copy()

    work["event_type"] = (
        "SEC_" + work["topic"].map(clean_type)
    )

    # Avoid turning dozens of excerpts from the same
    # filing/topic into dozens of independent events.
    #
    # Event unit:
    # company + filing + availability day + SEC topic.
    work["length"] = (
        work["evidence_text"].str.len()
    )

    work = work.sort_values(
        ["length", "original_id"],
        ascending=[False, True],
        kind="stable",
    )

    group_cols = [
        "ticker",
        "filing_date",
        "source_time",
        "event_type",
    ]

    multiplicity = (
        work.groupby(
            group_cols,
            dropna=False
        )
        .size()
        .rename("source_records")
        .reset_index()
    )

    work = work.drop_duplicates(
        group_cols,
        keep="first",
    )

    work = work.merge(
        multiplicity,
        on=group_cols,
        how="left",
        validate="one_to_one",
    )

    work["source"] = "SEC"

    work["source_id"] = work[
        "original_id"
    ].astype(str)

    work["event_id"] = [
        stable_id(
            "SEC",
            r.ticker,
            r.source_time,
            r.event_type,
            r.source_id,
        )
        for r in work.itertuples()
    ]

    log(
        "[SEC] filing-topic events:",
        len(work)
    )

    return work[
        [
            "event_id",
            "ticker",
            "source",
            "source_time",
            "event_type",
            "evidence_text",
            "source_id",
            "source_records",
            "novelty_candidate",
            "alignment_similarity",
        ]
    ]


def kg_events(path):
    log("[KG] Reading", path)

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    head = first_column(
        df,
        [
            "head", "subject", "src",
            "source_entity", "head_entity",
            "subject_id", "src_entity",
        ],
        True,
    )

    tail = first_column(
        df,
        [
            "tail", "object", "dst",
            "target_entity", "tail_entity",
            "object_id", "dst_entity",
        ],
        True,
    )

    rel = first_column(
        df,
        [
            "relation", "predicate", "rel",
            "relation_type", "edge_type",
        ],
        True,
    )

    time = first_column(
        df,
        [
            "available_at", "available_from",
            "timestamp", "event_date",
            "date", "time",
        ],
        True,
    )

    source_id_col = first_column(
        df,
        ["event_id", "source_id", "edge_id"]
    )

    log(
        "[KG] schema:",
        head, rel, tail, time
    )

    data = pd.DataFrame({
        "head": df[head].fillna("").astype(str),
        "tail": df[tail].fillna("").astype(str),
        "relation": df[rel].fillna("").astype(str),
        "source_time": parse_dates(df[time]),
        "original_id": (
            df[source_id_col].astype(str)
            if source_id_col
            else df.index.astype(str)
        ),
    })

    data = data[
        data["source_time"].notna()
    ].copy()

    outputs = []

    for ticker in TICKERS:
        mask = (
            data["head"].map(
                lambda s: match_company(s, ticker)
            )
            |
            data["tail"].map(
                lambda s: match_company(s, ticker)
            )
        )

        sub = data[mask].copy()

        if sub.empty:
            log(
                "[KG] WARNING: no events",
                ticker
            )
            continue

        sub["ticker"] = ticker

        sub["event_type"] = (
            "KG_" + sub["relation"].map(clean_type)
        )

        # Keep one event per relation + ticker + day,
        # but preserve the number of contributing edges.
        keys = [
            "ticker",
            "source_time",
            "event_type",
        ]

        counts = (
            sub.groupby(keys)
            .size()
            .rename("source_records")
            .reset_index()
        )

        sub = (
            sub.sort_values(
                "original_id",
                kind="stable"
            )
            .drop_duplicates(keys)
            .merge(
                counts,
                on=keys,
                validate="one_to_one",
            )
        )

        sub["evidence_text"] = (
            sub["head"]
            + " --"
            + sub["relation"]
            + "--> "
            + sub["tail"]
        )

        sub["source"] = "KG"

        sub["source_id"] = (
            sub["original_id"]
        )

        sub["event_id"] = [
            stable_id(
                "KG",
                r.ticker,
                r.source_time,
                r.event_type,
                r.source_id,
            )
            for r in sub.itertuples()
        ]

        sub["novelty_candidate"] = False
        sub["alignment_similarity"] = np.nan

        outputs.append(
            sub[
                [
                    "event_id",
                    "ticker",
                    "source",
                    "source_time",
                    "event_type",
                    "evidence_text",
                    "source_id",
                    "source_records",
                    "novelty_candidate",
                    "alignment_similarity",
                ]
            ]
        )

        log(
            "[KG]",
            ticker,
            "event groups:",
            len(sub)
        )

    if not outputs:
        raise RuntimeError(
            "Temporal KG has no directly linked "
            "ticker events. Check KG entity mapping."
        )

    return pd.concat(
        outputs,
        ignore_index=True
    )


def attach_outcomes(
    events,
    dates,
    prices,
    horizons,
):
    """
    Event source_time is not assumed to be a
    verified SEC acceptance timestamp.

    Conservative convention:
       available source day + 1 calendar day
       => first trading day on/after usable_from

    Prices at decision close form the base price.
    """

    dates = pd.DatetimeIndex(dates)
    day_values = dates.to_numpy(
        dtype="datetime64[ns]"
    )

    events = events.copy()

    events["usable_from"] = (
        events["source_time"]
        + pd.Timedelta(days=1)
    )

    usable = events[
        "usable_from"
    ].to_numpy(dtype="datetime64[ns]")

    index = np.searchsorted(
        day_values,
        usable,
        side="left",
    )

    events["decision_idx"] = index

    events = events[
        events["decision_idx"] < len(dates)
    ].copy()

    events["decision_idx"] = (
        events["decision_idx"].astype(int)
    )

    events["decision_date"] = [
        dates[i]
        for i in events["decision_idx"]
    ]

    ticker_to_index = {
        ticker: j
        for j, ticker in enumerate(TICKERS)
    }

    ticker_idx = (
        events["ticker"]
        .map(ticker_to_index)
        .to_numpy(dtype=int)
    )

    t = events[
        "decision_idx"
    ].to_numpy(dtype=int)

    for h in horizons:
        matured = (
            t + h < len(dates)
        )

        outcomes = np.full(
            len(events),
            np.nan,
            dtype=float,
        )

        mature_date = np.full(
            len(events),
            np.datetime64("NaT"),
            dtype="datetime64[ns]",
        )

        good = np.flatnonzero(matured)

        if len(good):
            start = t[good]
            end = start + h
            company = ticker_idx[good]

            stock_return = (
                prices[end, company]
                / prices[start, company]
                - 1
            )

            benchmark = (
                prices[end] / prices[start] - 1
            ).mean(axis=1)

            outcomes[good] = (
                stock_return - benchmark
            )

            mature_date[good] = (
                day_values[end]
            )

        events[f"outcome_{h}d"] = outcomes
        events[
            f"outcome_{h}d_matured_at"
        ] = pd.to_datetime(mature_date)

    return events


def normalize_vectors(texts):
    vectorizer = HashingVectorizer(
        n_features=128,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        stop_words="english",
        dtype=np.float32,
    )

    X = vectorizer.transform(
        texts.tolist()
    )

    return normalize(X), vectorizer


def predict_event(
    q,
    pool,
    embeddings,
    memory,
    outcome_column,
    top_k,
    rng,
):
    """
    Compare:
      EVENT_RAG: same type preferred, then
                 semantic similarity ranking.
      RECENT: most recently matured events.
      HIST_MEAN: full company mature-event mean.
      SHUFFLED_TYPE: alternative event types.
      ZERO: no excess return prediction.

    Candidate outcomes are known before query.
    """

    if not len(pool):
        return {}, [], []

    labels = memory[
        outcome_column
    ].to_numpy(dtype=float)

    target_type = (
        memory.iloc[q]["event_type"]
    )

    exact = np.asarray(
        [
            i for i in pool
            if memory.iloc[i]["event_type"]
            == target_type
        ],
        dtype=int,
    )

    # Prefer same economic event type.
    # If too few available, extend to other
    # event types, without using any future labels.
    if len(exact) >= 3:
        candidate = exact
    else:
        candidate = pool

    similarities = (
        embeddings[candidate]
        @ embeddings[q].T
    ).toarray().reshape(-1)

    order = np.lexsort(
        (candidate, -similarities)
    )[:top_k]

    chosen = candidate[order]

    sim = similarities[order]

    weight = np.exp(
        np.clip(
            5.0 * (sim - sim.max()),
            -40.0,
            0.0,
        )
    )

    weight /= weight.sum()

    rag = float(
        np.dot(weight, labels[chosen])
    )

    maturity = memory[
        "outcome_matured_at"
    ].to_numpy(dtype="datetime64[ns]")

    # RECENT: newest mature historical cases.
    recent_order = np.lexsort(
        (pool, -maturity[pool].astype("int64"))
    )[:top_k]

    recent = pool[recent_order]

    recent_pred = float(
        labels[recent].mean()
    )

    mean_pred = float(
        labels[pool].mean()
    )

    alternate = np.asarray(
        [
            i for i in pool
            if memory.iloc[i]["event_type"]
            != target_type
        ],
        dtype=int,
    )

    if len(alternate):
        shuffled = rng.choice(
            alternate,
            size=min(top_k, len(alternate)),
            replace=False,
        )

        shuffled_pred = float(
            labels[shuffled].mean()
        )
    else:
        shuffled = np.asarray([], dtype=int)
        shuffled_pred = mean_pred

    predictions = {
        "EVENT_RAG": rag,
        "RECENT": recent_pred,
        "HIST_MEAN": mean_pred,
        "SHUFFLED_TYPE": shuffled_pred,
        "ZERO": 0.0,
    }

    return predictions, chosen.tolist(), shuffled.tolist()


def evaluate_table(predictions):
    rows = []

    for method, sub in predictions.groupby("method"):
        y = sub["actual"].to_numpy(dtype=float)
        p = sub["prediction"].to_numpy(dtype=float)

        error = y - p

        rows.append({
            "method": method,
            "samples": len(sub),
            "dates": sub["date"].nunique(),
            "mse": float(np.mean(error ** 2)),
            "mae": float(np.mean(np.abs(error))),
            "sign_accuracy": float(
                np.mean(
                    (y > 0) == (p > 0)
                )
            ),
            "prediction_std": float(
                p.std()
            ),
        })

    return pd.DataFrame(rows)


def daily_error(predictions, method):
    sub = predictions[
        predictions["method"] == method
    ].copy()

    sub["squared_error"] = (
        sub["actual"] - sub["prediction"]
    ) ** 2

    sub["absolute_error"] = np.abs(
        sub["actual"] - sub["prediction"]
    )

    return (
        sub.groupby("date", as_index=False)
        [["squared_error", "absolute_error"]]
        .mean()
    )


def bootstrap_delta(
    predictions,
    baseline,
    seed,
    block=10,
    repeats=1000,
):
    """
    Paired bootstrap by EVENT DATE.

    Improvement > 0 means Event RAG
    has lower prediction error.

    This does not solve all event clustering
    or cross-company dependence.
    """

    a = daily_error(
        predictions,
        "EVENT_RAG"
    )

    b = daily_error(
        predictions,
        baseline,
    )

    both = a.merge(
        b,
        on="date",
        suffixes=("_rag", "_baseline"),
        validate="one_to_one",
    )

    rng = np.random.default_rng(seed)

    results = []

    for metric in [
        "squared_error",
        "absolute_error",
    ]:
        diff = (
            both[f"{metric}_baseline"]
            - both[f"{metric}_rag"]
        ).to_numpy()

        n = len(diff)

        if n < 20:
            results.append({
                "baseline": baseline,
                "metric": metric,
                "delta": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "days": n,
            })
            continue

        bsize = min(block, n)
        samples = []

        for _ in range(repeats):
            selection = []

            while len(selection) < n:
                start = int(
                    rng.integers(
                        0, n - bsize + 1
                    )
                )

                selection.extend(
                    range(start, start + bsize)
                )

            selection = selection[:n]

            samples.append(
                diff[selection].mean()
            )

        lo, hi = np.quantile(
            samples,
            [0.025, 0.975],
        )

        results.append({
            "baseline": baseline,
            "metric": metric,
            "delta": float(diff.mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "days": n,
        })

    return results


def main():
    p = argparse.ArgumentParser()

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
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/v11_event_reflection"
        ),
    )

    p.add_argument(
        "--neighbors",
        type=int,
        default=8,
    )

    p.add_argument("--seed", type=int, default=42)

    # Existing FinRL-X loader compatibility.
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/active_rag_v5_smoke/"
            "runs/ACTIVE__A2C__seed42/model.zip"
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
        default="rag/data/processed/sec/sec_chunks.jsonl"
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

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )

    p.add_argument("--cost-scale", type=float, default=0)
    p.add_argument("--reward-scaling", type=float, default=1)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")

    args = p.parse_args()

    if args.neighbors < 1:
        raise RuntimeError("--neighbors must be >= 1")

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    log("===== LOAD FINRL-X MARKET DATA =====")

    train, _, dates, _, n_train = (
        v8b.load_data(args)
    )

    dates = pd.DatetimeIndex(
        dates[:n_train]
    )

    prices = np.asarray(
        train["prices"],
        dtype=np.float64
    )

    if (
        len(dates) != len(prices)
        or dates.min().year != 2018
        or dates.max().year != 2020
    ):
        raise RuntimeError(
            "Unexpected train dates"
        )

    log(
        "[MARKET]",
        len(dates),
        "trading dates"
    )

    # Existing market feature loader receives
    # the Dynamic KGT source. V11-A uses the
    # raw KG event graph independently below.

    sec = sec_events(
        args.documents
    )

    kg = kg_events(
        args.temporal_kg
    )

    events = pd.concat(
        [sec, kg],
        ignore_index=True
    )

    # Audit only 2018-2020 source events.
    events = events[
        events["source_time"]
        <= pd.Timestamp("2020-12-31")
    ].copy()

    events = attach_outcomes(
        events,
        dates,
        prices,
        horizons=[5, 20],
    )

    # The historical event memory records events
    # even if forward outcomes are not matured.
    # Retrieval only uses mature outcomes.
    events = events.sort_values(
        [
            "decision_date",
            "ticker",
            "source",
            "event_type",
        ],
        kind="stable",
    ).reset_index(drop=True)

    events.to_csv(
        args.out / "event_memory.csv",
        index=False,
    )

    log("[EVENT MEMORY]", len(events))

    coverage = (
        events.groupby(
            ["source", "event_type"]
        )
        .agg(
            events=("event_id", "size"),
            first_date=("decision_date", "min"),
            last_date=("decision_date", "max"),
            companies=("ticker", "nunique"),
        )
        .reset_index()
        .sort_values(
            "events",
            ascending=False
        )
    )

    coverage.to_csv(
        args.out / "source_coverage.csv",
        index=False,
    )

    log(
        "[SOURCE COVERAGE]\n",
        coverage.head(15).to_string(index=False)
    )

    X, vectorizer = normalize_vectors(
        events["evidence_text"]
    )

    all_predictions = []
    all_retrievals = []

    for horizon in [5, 20]:
        log()
        log(
            "===== EVENT REFLECTION HORIZON",
            horizon,
            "====="
        )

        target_col = f"outcome_{horizon}d"
        mature_col = (
            f"outcome_{horizon}d_matured_at"
        )

        # Only evaluate 2020 queries whose
        # forward outcome fully matures in 2020.
        query_ids = np.flatnonzero(
            (events["decision_date"].dt.year == 2020)
            & events[target_col].notna()
            & (
                events[mature_col]
                <= pd.Timestamp("2020-12-31")
            )
        )

        log(
            "[2020 QUERY EVENTS]",
            len(query_ids)
        )

        maturities = events[
            mature_col
        ].to_numpy(dtype="datetime64[ns]")

        companies = events[
            "ticker"
        ].to_numpy()

        outcome = events[
            target_col
        ].to_numpy(dtype=float)

        for counter, q in enumerate(query_ids):
            row = events.iloc[q]
            date = np.datetime64(
                row["decision_date"]
            )

            # Historical events require both:
            # 1. information was already available
            # 2. outcome had already matured
            pool = np.flatnonzero(
                (companies == row["ticker"])
                & (maturities < date)
                & np.isfinite(outcome)
            )

            if len(pool) < 3:
                continue

            # The current event MUST NOT appear
            # in its own historical retrieval.
            if q in pool:
                raise RuntimeError(
                    "Event retrieval leakage"
                )

            predictions, selected, shuffled = (
                predict_event(
                    q,
                    pool,
                    X,
                    events.rename(
                        columns={
                            target_col: "outcome",
                            mature_col: "outcome_matured_at",
                        }
                    ),
                    "outcome",
                    args.neighbors,
                    rng,
                )
            )

            if not predictions:
                continue

            for method, value in predictions.items():
                all_predictions.append({
                    "horizon": horizon,
                    "date": str(
                        row["decision_date"].date()
                    ),
                    "ticker": row["ticker"],
                    "query_event_id": row["event_id"],
                    "source": row["source"],
                    "event_type": row["event_type"],
                    "method": method,
                    "actual": float(outcome[q]),
                    "prediction": float(value),
                })

            all_retrievals.append({
                "horizon": horizon,
                "query_event_id": row["event_id"],
                "decision_date": str(
                    row["decision_date"].date()
                ),
                "ticker": row["ticker"],
                "source": row["source"],
                "event_type": row["event_type"],
                "evidence_text": (
                    row["evidence_text"][:600]
                ),
                "historical_candidates": len(pool),
                "retrieved_ids": json.dumps(
                    events.iloc[selected][
                        "event_id"
                    ].tolist()
                ),
                "shuffled_ids": json.dumps(
                    events.iloc[shuffled][
                        "event_id"
                    ].tolist()
                ),
                "max_retrieved_maturity": (
                    str(
                        events.iloc[selected][
                            mature_col
                        ].max().date()
                    )
                    if selected else ""
                ),
            })

            if (
                counter == 0
                or (counter + 1) % 500 == 0
                or counter + 1 == len(query_ids)
            ):
                log(
                    f"[H{horizon}] "
                    f"{counter+1}/{len(query_ids)}"
                )

    predictions = pd.DataFrame(
        all_predictions
    )

    retrievals = pd.DataFrame(
        all_retrievals
    )

    if predictions.empty:
        raise RuntimeError(
            "No evaluable 2020 event predictions. "
            "Check event dates and history coverage."
        )

    predictions.to_csv(
        args.out / "event_predictions_2020.csv",
        index=False,
    )

    retrievals.to_csv(
        args.out / "event_retrieval_audit.csv",
        index=False,
    )

    metrics = []

    comparisons = []

    for horizon in [5, 20]:
        sub = predictions[
            predictions["horizon"] == horizon
        ]

        stats = evaluate_table(sub)
        stats["horizon"] = horizon
        metrics.append(stats)

        for i, baseline in enumerate([
            "RECENT",
            "HIST_MEAN",
            "SHUFFLED_TYPE",
            "ZERO",
        ]):
            result = bootstrap_delta(
                sub,
                baseline,
                seed=args.seed + i + horizon,
            )

            for r in result:
                comparisons.append({
                    "horizon": horizon,
                    **r,
                })

    metrics = pd.concat(
        metrics,
        ignore_index=True
    )

    comparisons = pd.DataFrame(
        comparisons
    )

    metrics.to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    comparisons.to_csv(
        args.out / "paired_comparison.csv",
        index=False,
    )

    report = {
        "project": "FinRL-X V11-A",
        "evaluation": "2020 event-level",
        "train_market_dates": len(dates),
        "sec_events": int(
            (events["source"] == "SEC").sum()
        ),
        "kg_events": int(
            (events["source"] == "KG").sum()
        ),
        "total_events": len(events),
        "horizons": [5, 20],
        "neighbors": args.neighbors,
        "test_used": False,
        "validation_2021_evaluated": False,
        "dynamic_kgt": args.kgt,
        "temporal_kg": args.temporal_kg,
        "limitations": [
            "Source available_from/event_time needs provenance audit",
            "Event type comes from recorded SEC topic/KG relation",
            "No assertion that a numeric mention is an actual financial change",
            "One representative SEC excerpt per filing-topic event",
            "Raw KG entity matches require entity mapping verification",
            "Historical realized outcomes are labels only",
            "Event-date bootstrap does not remove every dependency",
            "This is not an A2C portfolio backtest",
        ],
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
    log("===== V11 EVENT REFLECTION RESULTS =====")
    log(metrics.to_string(index=False))

    log()
    log("===== PAIRED COMPARISON =====")
    log(comparisons.to_string(index=False))

    log()
    log("V11 EVENT REFLECTION: PASS")
    log("2021 VALIDATION NOT EVALUATED")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
