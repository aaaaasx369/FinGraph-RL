#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V11-B
Temporal KG Semantics + SEC Novelty + Event Bundles

Train source period : 2018-2020
Evaluation          : 2020 walk-forward
2021 validation     : NOT EVALUATED
2022-2023 test      : NOT EVALUATED

Input:
  outputs/v31/evidence_alignment.parquet
  outputs/unified_temporal_kg_v3/unified_temporal_edges.csv
  FinRL-X market data through existing loader

Method:
  1. Audit KG relation identifiers.
  2. Detect first-observed KG triples.
  3. Deduplicate SEC filing/topic events.
  4. Estimate historical-only SEC text novelty.
  5. Build one bundle per ticker per event decision date.
  6. Attach matured 5D and 20D excess return outcomes.
  7. Walk-forward event retrieval in 2020.
  8. Compare EVENT_RAG, RECENT, HIST_MEAN, ZERO.

Strict memory:
  Historical outcome maturity_date < query decision_date

Important:
  FinDKG numeric relation IDs are NOT assigned invented meanings.
  SEC novelty is descriptive and may not be historically verified.
  Source timestamps must be audited separately for strict PIT.
  No A2C training in this stage.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from scipy.stats import spearmanr

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
HORIZONS = [5, 20]


def log(*x):
    print(*x, flush=True)


def find_col(df, aliases, required=False):
    lookup = {str(x).lower(): x for x in df.columns}
    for a in aliases:
        if a.lower() in lookup:
            return lookup[a.lower()]
    if required:
        raise RuntimeError(
            f"Missing columns {aliases}; got {list(df.columns)}"
        )
    return None


def dates_from(s):
    return pd.to_datetime(
        s, utc=True, errors="coerce"
    ).dt.tz_convert(None).dt.normalize()


def ticker_in(value, ticker):
    return bool(re.search(
        rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])",
        str(value).upper()
    ))


def stable_id(*parts):
    content = "|".join(map(str, parts))
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()[:20]


def clean_relation(value):
    return str(value).strip().upper()


def read_relation_metadata(path, out):
    """
    Typical relation2id.json maps relation names to integers.
    This mapping does not explain financial semantics of
    FINDKG_REL:0..14.

    Inspect and preserve exactly what is recorded.
    """
    result = {
        "path": str(path),
        "available": False,
        "relation_count": 0,
        "relation_names": [],
        "numeric_findkg_relations": [],
        "warning": (
            "A relation-to-ID mapping does not decode "
            "the economic meanings of numbered FinDKG relations."
        ),
    }

    if path and Path(path).is_file():
        obj = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        if isinstance(obj, dict):
            names = [str(k) for k in obj.keys()]
        elif isinstance(obj, list):
            names = [str(k) for k in obj]
        else:
            raise RuntimeError(
                "Unexpected relation2id JSON format"
            )

        result["available"] = True
        result["relation_count"] = len(names)
        result["relation_names"] = names
        result["numeric_findkg_relations"] = [
            x for x in names
            if re.search(r"FINDKG_REL[:_][0-9]+$", x)
        ]

    (out / "relation_semantics_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result


def load_kg_events(path):
    raw = pd.read_csv(path, low_memory=False)

    h = find_col(raw, [
        "head", "subject", "src", "source_entity",
        "head_entity", "subject_id"
    ], True)

    t = find_col(raw, [
        "tail", "object", "dst", "target_entity",
        "tail_entity", "object_id"
    ], True)

    r = find_col(raw, [
        "relation", "predicate", "rel",
        "relation_type", "edge_type"
    ], True)

    d = find_col(raw, [
        "available_at", "available_from",
        "event_date", "timestamp", "date", "time"
    ], True)

    data = pd.DataFrame({
        "head": raw[h].fillna("").astype(str),
        "tail": raw[t].fillna("").astype(str),
        "relation": raw[r].map(clean_relation),
        "source_time": dates_from(raw[d]),
    })

    data = data.dropna(subset=["source_time"])
    data = data[
        data["source_time"] <= pd.Timestamp("2020-12-31")
    ].copy()

    # First-observed relationship is computed from all
    # recorded earlier rows. We do NOT use whether an edge
    # disappears in later data to label earlier decisions.
    data = data.sort_values(
        ["source_time", "head", "relation", "tail"],
        kind="stable",
    ).reset_index(drop=True)

    grouped = (
        data.groupby(
            ["head", "relation", "tail"],
            sort=False
        )
        .agg(
            first_seen=("source_time", "min"),
            recorded_rows=("source_time", "size"),
        )
        .reset_index()
    )

    # First occurrence is one event candidate.
    # Repeated snapshots are NOT independent new events.
    events = []

    for ticker in TICKERS:
        mask = (
            grouped["head"].map(
                lambda x: ticker_in(x, ticker)
            )
            |
            grouped["tail"].map(
                lambda x: ticker_in(x, ticker)
            )
        )

        subset = grouped.loc[mask]

        for row in subset.itertuples(index=False):
            text = (
                f"{row.head} --{row.relation}--> {row.tail}"
            )

            events.append({
                "event_id": stable_id(
                    "KG", ticker, row.head,
                    row.relation, row.tail, row.first_seen
                ),
                "ticker": ticker,
                "source": "KG",
                "source_time": row.first_seen,
                "event_type": "KG_" + row.relation,
                "evidence_text": text,
                "source_id": stable_id(
                    row.head, row.relation, row.tail
                ),
                "is_first_observed": True,
                "recorded_rows": int(row.recorded_rows),
                "novelty_score": 1.0,
                "is_repeat_like": False,
            })

    result = pd.DataFrame(events)

    if result.empty:
        raise RuntimeError(
            "No KG triples matched these company identifiers. "
            "Check the KG entity-to-ticker mapping."
        )

    return result, {
        "raw_rows_through_2020": int(len(data)),
        "distinct_triples": int(len(grouped)),
        "first_observed_company_triples": int(len(result)),
        "relation_counts": dict(
            Counter(result["event_type"])
        ),
    }


def load_sec_events(path):
    raw = pd.read_parquet(path)

    ticker_col = find_col(raw, ["ticker", "company"], True)
    time_col = find_col(raw, ["available_from"], True)
    text_col = find_col(raw, ["current_excerpt"], True)

    topic_col = find_col(raw, [
        "topic", "section", "item"
    ])

    filing_col = find_col(raw, [
        "current_filing_date", "filing_date"
    ])

    id_col = find_col(raw, [
        "evidence_id", "pair_id", "source_id"
    ])

    novelty_col = find_col(raw, [
        "is_new_text_candidate"
    ])

    sim_col = find_col(raw, ["similarity"])

    df = pd.DataFrame({
        "ticker": raw[ticker_col].astype(str).str.upper(),
        "source_time": dates_from(raw[time_col]),
        "text": raw[text_col].fillna("").astype(str),
        "topic": (
            raw[topic_col].fillna("GENERAL").astype(str)
            if topic_col else "GENERAL"
        ),
        "filing": (
            dates_from(raw[filing_col])
            if filing_col else dates_from(raw[time_col])
        ),
        "source_id": (
            raw[id_col].astype(str)
            if id_col else pd.Series(
                raw.index.astype(str), index=raw.index
            )
        ),
        "existing_new_text_flag": (
            raw[novelty_col].fillna(False).astype(bool)
            if novelty_col else False
        ),
        "existing_alignment_similarity": (
            pd.to_numeric(raw[sim_col], errors="coerce")
            if sim_col else np.nan
        ),
    })

    df = df[
        df["ticker"].isin(TICKERS)
        & df["source_time"].notna()
        & (df["source_time"] <= pd.Timestamp("2020-12-31"))
        & (df["text"].str.len() >= 25)
    ].copy()

    df["topic"] = (
        df["topic"].str.upper()
        .str.replace(r"[^A-Z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    df = df.sort_values(
        ["source_time", "ticker", "topic", "filing",
         "source_id"],
        kind="stable",
    )

    # One representative text per filing/topic,
    # retaining the count of original excerpts.
    keys = [
        "ticker", "filing", "source_time", "topic"
    ]

    counts = (
        df.groupby(keys, dropna=False)
        .size().rename("source_records").reset_index()
    )

    df["text_length"] = df["text"].str.len()
    df = df.sort_values(
        ["text_length", "source_id"],
        ascending=[False, True],
        kind="stable",
    ).drop_duplicates(keys)

    df = df.merge(
        counts, on=keys, validate="one_to_one"
    )

    df = df.sort_values(
        ["source_time", "ticker", "topic", "source_id"],
        kind="stable",
    ).reset_index(drop=True)

    # Compare only with documents from strictly
    # earlier availability dates for this ticker/topic.
    # Same-day filing excerpts are NOT used as prior.
    vectorizer = HashingVectorizer(
        n_features=512,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        stop_words="english",
        dtype=np.float32,
    )

    X = vectorizer.transform(df["text"].tolist())

    novelty = np.ones(len(df), dtype=np.float32)
    max_similarity = np.zeros(len(df), dtype=np.float32)

    # Group sorted by source availability.
    for (_, _), index in df.groupby(
        ["ticker", "topic"], sort=False
    ).indices.items():
        ids = np.asarray(index, dtype=int)

        ids = ids[np.argsort(
            df.iloc[ids]["source_time"].to_numpy(),
            kind="stable",
        )]

        times = df.iloc[ids]["source_time"].to_numpy(
            dtype="datetime64[ns]"
        )

        for k, pos in enumerate(ids):
            earlier = ids[times < times[k]]

            if not len(earlier):
                continue

            # Sparse cosine similarity because rows are L2-normalized.
            scores = (X[earlier] @ X[pos].T).toarray().ravel()

            sim = float(scores.max()) if len(scores) else 0.0
            sim = float(np.clip(sim, 0.0, 1.0))

            max_similarity[pos] = sim
            novelty[pos] = 1.0 - sim

    df["novelty_score"] = novelty
    df["max_historical_similarity"] = max_similarity

    # Threshold used only for descriptive grouping.
    # Not tuned for future returns.
    df["is_repeat_like"] = (
        df["max_historical_similarity"] >= 0.85
    )

    records = []

    for row in df.itertuples(index=False):
        records.append({
            "event_id": stable_id(
                "SEC", row.ticker, row.source_time,
                row.filing, row.topic, row.source_id
            ),
            "ticker": row.ticker,
            "source": "SEC",
            "source_time": row.source_time,
            "event_type": "SEC_" + row.topic,
            "evidence_text": row.text,
            "source_id": row.source_id,
            "is_first_observed": False,
            "recorded_rows": int(row.source_records),
            "novelty_score": float(row.novelty_score),
            "is_repeat_like": bool(row.is_repeat_like),
            "max_historical_similarity": float(
                row.max_historical_similarity
            ),
            "existing_new_text_flag": bool(
                row.existing_new_text_flag
            ),
        })

    events = pd.DataFrame(records)

    if events.empty:
        raise RuntimeError("No SEC events found")

    return events, {
        "source_excerpt_rows": int(len(raw)),
        "filing_topic_events": int(len(events)),
        "repeat_like_events": int(
            events["is_repeat_like"].sum()
        ),
        "mean_novelty_score": float(
            events["novelty_score"].mean()
        ),
    }


def attach_trading_dates(events, dates, prices):
    """
    Conservative date-only convention:
      source date + one calendar day,
      then first available trading date.

    The source itself still requires publication
    time/provenance verification.
    """
    result = events.copy()

    result["usable_from"] = (
        result["source_time"] + pd.Timedelta(days=1)
    )

    index = np.searchsorted(
        dates.to_numpy(dtype="datetime64[ns]"),
        result["usable_from"].to_numpy(
            dtype="datetime64[ns]"
        ),
        side="left",
    )

    result["decision_idx"] = index

    result = result[
        result["decision_idx"] < len(dates)
    ].copy()

    result["decision_idx"] = result[
        "decision_idx"
    ].astype(int)

    result["decision_date"] = dates[
        result["decision_idx"].to_numpy()
    ].to_numpy()

    return result


def build_bundles(events, dates, prices):
    """
    The evaluation unit is ticker x decision_date.
    Multiple SEC/KG events on the same day share
    exactly one future-return target.
    """

    grouped = []

    for (ticker, date), g in events.groupby(
        ["ticker", "decision_date"], sort=True
    ):
        # Preserve evidence excerpts with a bounded length.
        # Favor novel SEC evidence and first-observed KG edges.
        selected = g.sort_values(
            ["novelty_score", "event_id"],
            ascending=[False, True],
            kind="stable",
        ).head(15)

        kinds = Counter(
            g["event_type"].astype(str)
        )

        grouped.append({
            "ticker": ticker,
            "date": pd.Timestamp(date),
            "decision_idx": int(g["decision_idx"].iloc[0]),
            "n_events": int(len(g)),
            "n_sec": int((g["source"] == "SEC").sum()),
            "n_kg": int((g["source"] == "KG").sum()),
            "n_repeat_like": int(g["is_repeat_like"].sum()),
            "mean_novelty": float(g["novelty_score"].mean()),
            "event_types": "|".join(sorted(kinds)),
            "event_ids": json.dumps(g["event_id"].tolist()),
            "evidence_text": " ".join(
                (
                    f"[{r.source}/{r.event_type}] "
                    f"{r.evidence_text[:450]}"
                )
                for r in selected.itertuples()
            )[:5000],
        })

    bundles = pd.DataFrame(grouped).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    ticker_index = {t: i for i, t in enumerate(TICKERS)}

    for horizon in HORIZONS:
        returns = []
        maturity = []

        for row in bundles.itertuples():
            t = row.decision_idx
            j = ticker_index[row.ticker]

            if t + horizon >= len(dates):
                returns.append(np.nan)
                maturity.append(pd.NaT)
                continue

            end = t + horizon

            company_ret = (
                prices[end, j] / prices[t, j] - 1
            )

            benchmark = np.mean(
                prices[end] / prices[t] - 1
            )

            returns.append(float(company_ret - benchmark))
            maturity.append(pd.Timestamp(dates[end]))

        bundles[f"excess_{horizon}d"] = returns
        bundles[f"matured_{horizon}d"] = maturity

    return bundles


def score_methods(bundles, horizon, top_k):
    """
    Event-centric walk-forward retrieval.

    Only historical company-date bundles whose
    outcome matured strictly before the query date.

    Semantic query uses actual event description,
    not market-state vector.

    Equal-weight case averaging avoids aggressive
    hindsight-return weighting.
    """

    vec = HashingVectorizer(
        n_features=512,
        ngram_range=(1, 2),
        alternate_sign=False,
        norm="l2",
        stop_words="english",
        dtype=np.float32,
    )

    embeddings = vec.transform(
        bundles["evidence_text"].tolist()
    )

    dates = bundles["date"].to_numpy(
        dtype="datetime64[ns]"
    )

    maturities = bundles[
        f"matured_{horizon}d"
    ].to_numpy(dtype="datetime64[ns]")

    actual = bundles[
        f"excess_{horizon}d"
    ].to_numpy(dtype=float)

    tickers = bundles["ticker"].to_numpy()

    eval_ids = np.flatnonzero(
        (pd.DatetimeIndex(dates).year == 2020)
        & np.isfinite(actual)
        & (maturities <= np.datetime64("2020-12-31"))
    )

    predictions = []
    retrievals = []

    for q in eval_ids:
        query_date = dates[q]

        eligible = np.flatnonzero(
            (tickers == tickers[q])
            & (maturities < query_date)
            & (dates < query_date)
            & np.isfinite(actual)
        )

        if len(eligible) < 3:
            continue

        similarity = (
            embeddings[eligible]
            @ embeddings[q].T
        ).toarray().ravel()

        order = np.lexsort(
            (eligible, -similarity)
        )[:top_k]

        selected = eligible[order]

        recent = eligible[np.lexsort((
            eligible,
            -maturities[eligible].astype("int64")
        ))[:top_k]]

        scores = {
            "EVENT_RAG": float(actual[selected].mean()),
            "RECENT": float(actual[recent].mean()),
            "HIST_MEAN": float(actual[eligible].mean()),
            "ZERO": 0.0,
        }

        for method, estimate in scores.items():
            predictions.append({
                "horizon": horizon,
                "date": str(pd.Timestamp(query_date).date()),
                "ticker": tickers[q],
                "method": method,
                "actual": float(actual[q]),
                "prediction": float(estimate),
                "n_events": int(bundles.iloc[q]["n_events"]),
                "n_sec": int(bundles.iloc[q]["n_sec"]),
                "n_kg": int(bundles.iloc[q]["n_kg"]),
            })

        retrievals.append({
            "horizon": horizon,
            "date": str(pd.Timestamp(query_date).date()),
            "ticker": tickers[q],
            "query_event_ids": bundles.iloc[q]["event_ids"],
            "retrieved_bundle_indices": json.dumps(
                selected.tolist()
            ),
            "latest_maturity": str(
                pd.Timestamp(maturities[selected].max()).date()
            ),
            "semantic_similarity_mean": float(
                similarity[order].mean()
            ),
            "historical_pool_size": int(len(eligible)),
        })

    return pd.DataFrame(predictions), pd.DataFrame(retrievals)


def evaluate(predictions):
    rows = []

    for (horizon, method), sub in predictions.groupby(
        ["horizon", "method"]
    ):
        y = sub["actual"].to_numpy()
        pred = sub["prediction"].to_numpy()

        rows.append({
            "horizon": horizon,
            "method": method,
            "samples": len(sub),
            "days": sub["date"].nunique(),
            "mse": float(np.mean((y - pred) ** 2)),
            "mae": float(np.mean(np.abs(y - pred))),
            "sign_accuracy": float(
                np.mean((y > 0) == (pred > 0))
            ),
        })

    return pd.DataFrame(rows)


def paired_bootstrap(predictions, baseline, horizon,
                     seed=42, repeats=1000, block=10):
    """
    Paired moving-block bootstrap by decision date.
    Improvement positive means lower MSE for EVENT_RAG.
    """

    sub = predictions[predictions["horizon"] == horizon]

    x = sub[sub["method"] == "EVENT_RAG"].copy()
    b = sub[sub["method"] == baseline].copy()

    keys = ["date", "ticker"]
    joined = x[keys + ["actual", "prediction"]].merge(
        b[keys + ["prediction"]],
        on=keys,
        suffixes=("_rag", "_base"),
        validate="one_to_one",
    )

    joined["delta"] = (
        (joined["actual"] - joined["prediction_base"]) ** 2
        - (joined["actual"] - joined["prediction_rag"]) ** 2
    )

    daily = (
        joined.groupby("date")["delta"]
        .mean().sort_index().to_numpy()
    )

    n = len(daily)

    if n < 20:
        return {
            "horizon": horizon,
            "baseline": baseline,
            "mean_improvement": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "days": n,
        }

    rng = np.random.default_rng(seed)
    block = min(block, n)

    draws = []

    for _ in range(repeats):
        positions = []

        while len(positions) < n:
            start = rng.integers(0, n-block+1)
            positions.extend(range(start, start+block))

        draws.append(
            daily[np.asarray(positions[:n])].mean()
        )

    lo, hi = np.quantile(draws, [0.025, 0.975])

    return {
        "horizon": horizon,
        "baseline": baseline,
        "mean_improvement": float(daily.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "days": n,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--documents",
        default="outputs/v31/evidence_alignment.parquet"
    )
    parser.add_argument(
        "--temporal-kg",
        default="outputs/unified_temporal_kg_v3/unified_temporal_edges.csv"
    )
    parser.add_argument(
        "--relation-map",
        default="",
        help="Optional verified relation2id.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/aegis_rl_v1/v11b_event_semantics")
    )
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)

    # Existing loader compatibility.
    parser.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/active_rag_v5_smoke/"
            "runs/ACTIVE__A2C__seed42/model.zip"
        )
    )
    parser.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        )
    )
    parser.add_argument(
        "--kg",
        default="outputs/unified_temporal_kg_v3/unified_temporal_edges.csv"
    )
    parser.add_argument(
        "--sec",
        default="rag/data/processed/sec/sec_chunks.jsonl"
    )
    parser.add_argument(
        "--sec-signals",
        default="outputs/unified_event_memory_v2/sec_numeric_signals.csv"
    )
    parser.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434"
    )
    parser.add_argument("--llm", default="qwen2.5:3b")
    parser.add_argument("--embed", default="bge-m3:latest")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--algo", default="A2C")
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--cost-scale", type=float, default=0.0)
    parser.add_argument("--tier", default="conservative")
    parser.add_argument("--max-days", type=int, default=0)

    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.neighbors < 1:
        raise ValueError("neighbors must be positive")

    log("===== LOAD HISTORICAL FINRL-X DATA =====")

    train, _, dates, _, n_train = v8b.load_data(args)

    dates = pd.DatetimeIndex(dates[:n_train]).normalize()
    prices = np.asarray(train["prices"], dtype=float)

    if (
        len(dates) != len(prices)
        or dates.min().year != 2018
        or dates.max().year != 2020
    ):
        raise RuntimeError("Unexpected historical data dates")

    log("[TRAIN DAYS]", len(dates))

    log("===== RELATION SEMANTICS AUDIT =====")
    relation_info = read_relation_metadata(
        args.relation_map, args.out
    )

    log("===== TEMPORAL KG FIRST-OBSERVED EVENTS =====")
    kg, kg_info = load_kg_events(args.temporal_kg)
    log("[KG EVENTS]", len(kg))

    log("===== SEC NOVELTY AUDIT =====")
    sec, sec_info = load_sec_events(args.documents)
    log("[SEC EVENTS]", len(sec))

    events = pd.concat([sec, kg], ignore_index=True)

    events = attach_trading_dates(
        events, dates, prices
    )

    events.to_csv(
        args.out / "clean_event_memory.csv",
        index=False
    )

    # Metadata-only audit: count first-seen KG relations
    # and SEC novelty, never guessing relation meanings.
    event_audit = (
        events.groupby(["source", "event_type"])
        .agg(
            events=("event_id", "size"),
            companies=("ticker", "nunique"),
            first_date=("decision_date", "min"),
            last_date=("decision_date", "max"),
            mean_novelty=("novelty_score", "mean"),
            repeat_like=("is_repeat_like", "sum"),
        )
        .reset_index()
        .sort_values("events", ascending=False)
    )

    event_audit.to_csv(
        args.out / "event_semantics_audit.csv",
        index=False
    )

    log("===== BUILD COMPANY-DATE EVENT BUNDLES =====")

    bundles = build_bundles(events, dates, prices)

    bundles.to_csv(
        args.out / "company_date_event_bundles.csv",
        index=False,
    )

    log("[EVENTS]", len(events))
    log("[BUNDLES]", len(bundles))

    all_predictions = []
    all_retrievals = []

    for h in HORIZONS:
        log(f"===== WALK-FORWARD HORIZON {h} =====")

        pred, audit = score_methods(
            bundles, h, args.neighbors
        )

        if not pred.empty:
            all_predictions.append(pred)
            all_retrievals.append(audit)

        log(
            f"[H{h}] evaluated company-date rows:",
            len(pred) // 4
        )

    if not all_predictions:
        raise RuntimeError(
            "No valid walk-forward event predictions"
        )

    predictions = pd.concat(
        all_predictions, ignore_index=True
    )

    retrievals = pd.concat(
        all_retrievals, ignore_index=True
    )

    predictions.to_csv(
        args.out / "predictions_2020.csv",
        index=False
    )

    retrievals.to_csv(
        args.out / "retrieval_audit_2020.csv",
        index=False
    )

    metrics = evaluate(predictions)
    metrics.to_csv(
        args.out / "method_metrics.csv",
        index=False
    )

    comparisons = []

    for h in HORIZONS:
        for i, baseline in enumerate([
            "RECENT", "HIST_MEAN", "ZERO"
        ]):
            comparisons.append(
                paired_bootstrap(
                    predictions, baseline, h,
                    seed=args.seed+h+i
                )
            )

    pd.DataFrame(comparisons).to_csv(
        args.out / "paired_bootstrap.csv",
        index=False
    )

    manifest = {
        "experiment": "FinRL-X V11-B",
        "evaluation": "2020 walk-forward company-date",
        "train_date_count": len(dates),
        "raw_kg": kg_info,
        "sec": sec_info,
        "relation_map": relation_info,
        "total_events": len(events),
        "company_date_bundles": len(bundles),
        "validation_2021_evaluated": False,
        "test_2022_2023_evaluated": False,
        "limitations": [
            "KG first-observed means first in available dataset, not necessarily real-world novelty",
            "KG event date is not independently verified public-release date",
            "SEC topic novelty is lexical, not certified financial change",
            "SEC alignment records may contain hindsight-generated metadata",
            "One company-date outcome per event bundle",
            "No semantic interpretation assigned to numbered FinDKG relations",
            "2020 evaluation only, not A2C portfolio return",
        ],
    }

    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log()
    log("===== V11-B RESULTS =====")
    log(metrics.to_string(index=False))

    log()
    log("===== PAIRED BOOTSTRAP =====")
    log(pd.DataFrame(comparisons).to_string(index=False))

    log()
    log("V11-B: PASS")
    log("2021 VALIDATION NOT EVALUATED")
    log("2022-2023 TEST NOT EVALUATED")


if __name__ == "__main__":
    main()
