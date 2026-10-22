#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17-A/B: SEC-RAG + Dynamic KG-RAG + Qwen

Inputs:
  V11-F SEC PIT evidence
  Unified Temporal KG V3
  Historical trading decision dates

Outputs:
  signals.csv
  reasoning_audit.jsonl
  retrieval_audit.jsonl
  manifest.json
  cache.sqlite

Each company-date:
  1. Retrieve historically eligible SEC evidence
  2. Retrieve historically observed 1-hop KG edges
  3. Ask Qwen to reason over BOTH sources
  4. Produce impact / risk / confidence
  5. Validate numeric output and evidence citations
  6. Cache every result

This is the RAG reasoning stage. It does NOT
train a PPO policy or access future stock returns.

SEC availability: available_from < decision date.
KG observation: first_observed < decision date.

KG observation is NOT proof of actual disclosure time.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ollama


TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

SYSTEM_PROMPT = """
You are a financial evidence analysis module.

You receive only historically eligible evidence from:
1. SEC filings
2. A temporal financial knowledge graph

Your task is NOT to predict stock returns.
Your task is to assess evidence-supported financial
impact, risk, and confidence for a named company
at a specified historical decision date.

Return only a valid JSON object with:
{
  "impact": number between -1 and 1,
  "risk": number between 0 and 1,
  "confidence": number between 0 and 1,
  "sec_evidence_ids": array of strings,
  "kg_evidence_ids": array of strings,
  "reasoning": short string
}

Requirements:
- Use ONLY the supplied evidence.
- Never use future financial information.
- Never invent facts, numeric performance, or relations.
- Numbered FINDKG_REL:n relation codes have UNKNOWN
  economic meanings. Do not infer customer, supplier,
  competitor, sentiment, or causality from those codes.
- Preserve original KG directions.
- KG edges are observed graph relationships, not
  verified economic announcements.
- An old, repeated graph edge is NOT a new event.
- Cite only IDs actually supplied to you.
- Empty evidence should produce neutral impact,
  risk 0, confidence 0, and empty citation arrays.
- If financial implications are ambiguous, keep
  impact near zero and reduce confidence.
- Never claim a stock-price movement is certain.
"""


def log(*items):
    print(*items, flush=True)


def normalize_dates(series):
    return (
        pd.to_datetime(
            series,
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )


def choose_col(df, candidates, label):
    lookup = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise RuntimeError(
        f"Missing {label}; columns={list(df.columns)}"
    )


def safe_text(value, limit=800):
    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(r"\s+", " ", text).strip()

    return text[:limit]


def stable_id(prefix, fields):
    payload = json.dumps(
        fields,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:20]

    return prefix + "_" + digest


def load_sec(path):
    log("[SEC LOAD]", path)

    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".parquet":
        raw = pd.read_parquet(p)
    elif p.suffix.lower() == ".csv":
        raw = pd.read_csv(p, low_memory=False)
    else:
        raise RuntimeError(
            "SEC input must be parquet or CSV"
        )

    ticker_col = choose_col(
        raw, ["ticker", "tic", "symbol"], "ticker"
    )

    date_col = choose_col(
        raw,
        ["available_from", "pit_available_from"],
        "PIT available_from",
    )

    excerpt_col = choose_col(
        raw,
        ["current_excerpt", "excerpt", "text"],
        "SEC excerpt",
    )

    out = pd.DataFrame({
        "ticker": raw[ticker_col].astype(str).str.upper(),
        "available_from": normalize_dates(raw[date_col]),
        "excerpt": raw[excerpt_col].fillna("").astype(str),
    })

    optional = {
        "accession": [
            "current_accession", "accession",
            "accession_number",
        ],
        "form": ["form", "filing_type"],
        "topic": ["topic", "section", "item"],
        "filing_date": [
            "current_filing_date", "filing_date",
        ],
        "evidence_source_id": [
            "evidence_id", "chunk_id",
        ],
    }

    for output_name, candidates in optional.items():
        col = next(
            (
                c for c in candidates
                if c in raw.columns
            ),
            None,
        )

        if col is None:
            out[output_name] = ""
        else:
            out[output_name] = (
                raw[col].fillna("").astype(str)
            )

    out = out.loc[
        out["ticker"].isin(TICKERS)
        & out["available_from"].notna()
        & out["excerpt"].str.strip().ne("")
    ].copy()

    if out.empty:
        raise RuntimeError(
            "No usable PIT SEC excerpts"
        )

    out["id"] = [
        stable_id(
            "SEC",
            [
                r.ticker,
                str(r.available_from),
                r.accession,
                r.evidence_source_id,
                r.excerpt,
            ],
        )
        for r in out.itertuples(index=False)
    ]

    out = (
        out.drop_duplicates("id")
        .sort_values(
            ["ticker", "available_from", "id"]
        )
        .reset_index(drop=True)
    )

    log("[SEC EVIDENCE ROWS]", len(out))

    return out


def load_kg(path):
    log("[KG LOAD]", path)

    raw = pd.read_csv(
        path,
        low_memory=False,
    )

    h = choose_col(
        raw,
        ["head", "head_id", "source", "src",
         "subject", "entity1"],
        "head",
    )

    r = choose_col(
        raw,
        ["relation", "rel", "predicate",
         "relation_type"],
        "relation",
    )

    t = choose_col(
        raw,
        ["tail", "tail_id", "target", "dst",
         "object", "entity2"],
        "tail",
    )

    d = choose_col(
        raw,
        ["date", "event_date", "timestamp",
         "observed_date", "snapshot_date",
         "as_of_date", "time", "effective_date"],
        "date",
    )

    out = pd.DataFrame({
        "head": raw[h].astype("string").str.strip(),
        "relation": raw[r].astype("string").str.strip(),
        "tail": raw[t].astype("string").str.strip(),
        "date": normalize_dates(raw[d]),
    })

    valid = (
        out["head"].notna()
        & out["relation"].notna()
        & out["tail"].notna()
        & out["date"].notna()
        & out["head"].str.len().gt(0)
        & out["relation"].str.len().gt(0)
        & out["tail"].str.len().gt(0)
    )

    if not bool(valid.all()):
        raise RuntimeError(
            f"Invalid KG rows: {int((~valid).sum())}"
        )

    # Repeated snapshots of a directed triple are
    # collapsed into their earliest observation.
    out = (
        out.groupby(
            ["head", "relation", "tail"],
            as_index=False,
        )["date"]
        .min()
        .rename(
            columns={"date": "first_observed"}
        )
    )

    company_nodes = {
        f"COMPANY:{ticker}"
        for ticker in TICKERS
    }

    # 1-hop company-linked graph.
    out = out.loc[
        out["head"].isin(company_nodes)
        | out["tail"].isin(company_nodes)
    ].copy()

    if out.empty:
        raise RuntimeError(
            "No company-linked KG edges"
        )

    out["id"] = [
        stable_id(
            "KG",
            [r.head, r.relation, r.tail],
        )
        for r in out.itertuples(index=False)
    ]

    out = (
        out.sort_values(
            ["first_observed", "id"]
        )
        .reset_index(drop=True)
    )

    log("[KG COMPANY EDGES]", len(out))
    log("[KG RELATIONS]", out["relation"].nunique())

    return out


def load_decision_dates(args):
    if args.dates:
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                args.dates,
                errors="raise",
            )
        ).normalize()

    else:
        source = Path(args.dates_csv)

        if not source.exists():
            raise FileNotFoundError(
                "Dates CSV not found. Supply --dates "
                "for smoke testing or --dates-csv "
                "for a historical batch."
            )

        raw = pd.read_csv(source)

        if "date" not in raw.columns:
            raise RuntimeError(
                "Dates CSV must have a date column"
            )

        dates = pd.DatetimeIndex(
            normalize_dates(raw["date"]).dropna().unique()
        )

    dates = dates.sort_values().unique()

    if args.start_date:
        dates = dates[
            dates >= pd.Timestamp(args.start_date)
        ]

    if args.end_date:
        dates = dates[
            dates <= pd.Timestamp(args.end_date)
        ]

    if args.max_dates:
        dates = dates[:args.max_dates]

    if len(dates) == 0:
        raise RuntimeError("No decision dates")

    if dates.min().year < 2018 or dates.max().year > 2021:
        raise RuntimeError(
            "This V17 stage is restricted to 2018-2021"
        )

    return dates


def sec_retrieve(sec, ticker, day, top_k):
    eligible = sec.loc[
        sec["ticker"].eq(ticker)
        & (sec["available_from"] < day)
    ]

    if eligible.empty:
        return []

    # Limit repeated excerpts of the same filing.
    eligible = (
        eligible.sort_values(
            ["available_from", "id"],
            ascending=[False, True],
        )
        .drop_duplicates(
            ["accession", "excerpt"],
        )
        .head(top_k)
    )

    results = []

    for r in eligible.itertuples(index=False):
        results.append({
            "id": r.id,
            "source": "SEC",
            "ticker": ticker,
            "available_from": str(
                r.available_from.date()
            ),
            "filing_date": r.filing_date,
            "accession": r.accession,
            "form": r.form,
            "topic": r.topic,
            "text": safe_text(
                r.excerpt,
                limit=900,
            ),
        })

    return results


def kg_retrieve(kg, ticker, day, top_k):
    node = f"COMPANY:{ticker}"

    eligible = kg.loc[
        (
            kg["head"].eq(node)
            | kg["tail"].eq(node)
        )
        & (kg["first_observed"] < day)
    ]

    if eligible.empty:
        return []

    eligible = (
        eligible.sort_values(
            ["first_observed", "id"],
            ascending=[False, True],
        )
        .head(top_k)
    )

    results = []

    for r in eligible.itertuples(index=False):
        results.append({
            "id": r.id,
            "source": "KG",
            "head": str(r.head),
            "relation": str(r.relation),
            "tail": str(r.tail),
            "direction_from_company": (
                "out"
                if str(r.head) == node
                else "in"
            ),
            "first_observed": str(
                r.first_observed.date()
            ),
            "age_days": int(
                (day - r.first_observed).days
            ),
            "semantic_label_verified": (
                not str(r.relation).startswith(
                    "FINDKG_REL:"
                )
            ),
        })

    return results


class Cache:
    def __init__(self, path):
        self.connection = sqlite3.connect(
            str(path)
        )

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS outputs (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
        """)

        self.connection.commit()

    def get(self, key):
        row = self.connection.execute(
            "SELECT payload FROM outputs "
            "WHERE cache_key = ?",
            (key,),
        ).fetchone()

        return json.loads(row[0]) if row else None

    def put(self, key, result):
        self.connection.execute(
            "INSERT OR REPLACE INTO outputs "
            "(cache_key, payload) VALUES (?, ?)",
            (
                key,
                json.dumps(
                    result,
                    ensure_ascii=False,
                ),
            ),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()


def clamp(value, lo, hi, label):
    number = float(value)

    if not np.isfinite(number):
        raise RuntimeError(
            f"Nonfinite {label}"
        )

    if not lo <= number <= hi:
        raise RuntimeError(
            f"Out-of-range {label}: {number}"
        )

    return number


def validate_response(result, sec_hits, kg_hits):
    if not isinstance(result, dict):
        raise RuntimeError(
            "LLM output is not a JSON object"
        )

    required = [
        "impact", "risk", "confidence",
        "sec_evidence_ids",
        "kg_evidence_ids",
        "reasoning",
    ]

    for key in required:
        if key not in result:
            raise RuntimeError(
                f"Missing LLM output field: {key}"
            )

    impact = clamp(
        result["impact"], -1.0, 1.0, "impact"
    )
    risk = clamp(
        result["risk"], 0.0, 1.0, "risk"
    )
    confidence = clamp(
        result["confidence"], 0.0, 1.0, "confidence"
    )

    if not isinstance(
        result["sec_evidence_ids"], list
    ):
        raise RuntimeError(
            "sec_evidence_ids must be a list"
        )

    if not isinstance(
        result["kg_evidence_ids"], list
    ):
        raise RuntimeError(
            "kg_evidence_ids must be a list"
        )

    allowed_sec = {
        x["id"] for x in sec_hits
    }

    allowed_kg = {
        x["id"] for x in kg_hits
    }

    cited_sec = result["sec_evidence_ids"]
    cited_kg = result["kg_evidence_ids"]

    if (
        not all(
            isinstance(x, str) and x in allowed_sec
            for x in cited_sec
        )
        or not all(
            isinstance(x, str) and x in allowed_kg
            for x in cited_kg
        )
    ):
        raise RuntimeError(
            "LLM cited nonexistent evidence IDs"
        )

    reasoning = str(
        result["reasoning"]
    ).strip()

    if len(reasoning) > 2500:
        raise RuntimeError(
            "LLM reasoning unexpectedly long"
        )

    if not sec_hits and not kg_hits:
        if (
            impact != 0
            or risk != 0
            or confidence != 0
            or cited_sec
            or cited_kg
        ):
            raise RuntimeError(
                "Empty evidence produced nonneutral signal"
            )

    if (sec_hits or kg_hits) and (
        not cited_sec and not cited_kg
    ):
        # Allow explicit abstention only with
        # neutral impact and zero confidence.
        if impact != 0 or confidence != 0:
            raise RuntimeError(
                "Nonneutral uncited LLM inference"
            )

    return {
        "impact": impact,
        "risk": risk,
        "confidence": confidence,
        "sec_evidence_ids": cited_sec,
        "kg_evidence_ids": cited_kg,
        "reasoning": reasoning,
    }


def call_llm(
    client,
    model,
    ticker,
    day,
    sec_hits,
    kg_hits,
):
    if not sec_hits and not kg_hits:
        return {
            "impact": 0.0,
            "risk": 0.0,
            "confidence": 0.0,
            "sec_evidence_ids": [],
            "kg_evidence_ids": [],
            "reasoning": "No eligible evidence.",
        }

    user_payload = {
        "company": ticker,
        "decision_date": str(day.date()),
        "sec_evidence": sec_hits,
        "kg_evidence": kg_hits,
        "instruction": (
            "Provide evidence-grounded qualitative "
            "financial analysis using JSON only."
        ),
    }

    response = client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                ),
            },
        ],
        format="json",
        options={
            "temperature": 0,
            "num_predict": 500,
        },
    )

    content = response.message.content

    parsed = json.loads(content)

    return validate_response(
        parsed,
        sec_hits,
        kg_hits,
    )


def append_jsonl(path, record):
    with open(
        path,
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            ) + "\n"
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
        "--dates-csv",
        default=(
            "results/aegis_rl_v1/"
            "v16_dual_rag_ppo/"
            "kg_retrieval_audit.csv"
        ),
    )

    p.add_argument(
        "--dates",
        nargs="+",
        default=None,
    )

    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--max-dates", type=int, default=0)

    p.add_argument(
        "--tickers",
        nargs="+",
        default=TICKERS,
    )

    p.add_argument("--sec-top-k", type=int, default=5)
    p.add_argument("--kg-top-k", type=int, default=12)

    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )

    p.add_argument("--retries", type=int, default=3)

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v17_unified_llm_rag"
        ),
    )

    args = p.parse_args()

    if not set(args.tickers).issubset(TICKERS):
        raise RuntimeError(
            "Invalid ticker list"
        )

    if (
        args.sec_top_k < 1
        or args.kg_top_k < 1
        or args.retries < 1
    ):
        raise RuntimeError(
            "Invalid retrieval settings"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    sec = load_sec(args.pit_sec)
    kg = load_kg(args.temporal_kg)

    dates = load_decision_dates(args)

    log("[DECISION DATES]", len(dates))
    log("[TICKERS]", args.tickers)

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    model_names = {
        getattr(m, "model", "")
        for m in client.list().models
    }

    if args.llm not in model_names:
        raise RuntimeError(
            f"LLM model {args.llm!r} not available "
            f"on {args.ollama_host}. "
            f"Available: {sorted(model_names)}"
        )

    log("[OLLAMA]", args.ollama_host)
    log("[MODEL]", args.llm)

    cache = Cache(
        args.out / "cache.sqlite"
    )

    signals_path = (
        args.out / "signals.csv"
    )

    reasoning_path = (
        args.out / "reasoning_audit.jsonl"
    )

    retrieval_path = (
        args.out / "retrieval_audit.jsonl"
    )

    # We reconstruct the CSV from all successful
    # cached/current results at the end of the run.
    # JSONL logs remain append-only audit artifacts.
    rows = []

    cache_hits = 0
    llm_calls = 0
    failures = 0

    try:
        for day in dates:
            day = pd.Timestamp(day)

            for ticker in args.tickers:
                sec_hits = sec_retrieve(
                    sec,
                    ticker,
                    day,
                    args.sec_top_k,
                )

                kg_hits = kg_retrieve(
                    kg,
                    ticker,
                    day,
                    args.kg_top_k,
                )

                for item in sec_hits:
                    if not (
                        pd.Timestamp(
                            item["available_from"]
                        ) < day
                    ):
                        raise RuntimeError(
                            "SEC PIT violation"
                        )

                for item in kg_hits:
                    if not (
                        pd.Timestamp(
                            item["first_observed"]
                        ) < day
                    ):
                        raise RuntimeError(
                            "KG temporal violation"
                        )

                payload = {
                    "ticker": ticker,
                    "date": str(day.date()),
                    "model": args.llm,
                    "host": args.ollama_host,
                    "prompt_version": "v17b_1",
                    "sec_hits": sec_hits,
                    "kg_hits": kg_hits,
                }

                key = stable_id(
                    "V17",
                    payload,
                )

                append_jsonl(
                    retrieval_path,
                    {
                        "cache_key": key,
                        **payload,
                    },
                )

                result = cache.get(key)

                if result is not None:
                    # Never trust cached JSON without
                    # applying current validation.
                    result = validate_response(
                        result,
                        sec_hits,
                        kg_hits,
                    )
                    cache_hits += 1

                else:
                    last_error = None

                    for attempt in range(
                        args.retries
                    ):
                        try:
                            result = call_llm(
                                client,
                                args.llm,
                                ticker,
                                day,
                                sec_hits,
                                kg_hits,
                            )

                            cache.put(key, result)
                            llm_calls += 1
                            last_error = None
                            break

                        except Exception as exc:
                            last_error = exc
                            log(
                                "[RETRY]",
                                str(day.date()),
                                ticker,
                                attempt + 1,
                                repr(exc),
                            )
                            time.sleep(
                                min(2 ** attempt, 8)
                            )

                    if last_error is not None:
                        failures += 1

                        append_jsonl(
                            reasoning_path,
                            {
                                "cache_key": key,
                                "date": str(day.date()),
                                "ticker": ticker,
                                "status": "FAILED",
                                "error": repr(last_error),
                            },
                        )

                        # Do not invent zero signals
                        # to hide model/API failure.
                        raise RuntimeError(
                            "V17 LLM reasoning failed "
                            f"for {ticker} {day.date()}"
                        ) from last_error

                row = {
                    "date": str(day.date()),
                    "ticker": ticker,
                    "impact": result["impact"],
                    "risk": result["risk"],
                    "confidence": result["confidence"],
                    "sec_retrieved": len(sec_hits),
                    "kg_retrieved": len(kg_hits),
                    "sec_cited": len(
                        result["sec_evidence_ids"]
                    ),
                    "kg_cited": len(
                        result["kg_evidence_ids"]
                    ),
                    "cache_key": key,
                }

                rows.append(row)

                append_jsonl(
                    reasoning_path,
                    {
                        **row,
                        "status": "OK",
                        "sec_evidence_ids": result[
                            "sec_evidence_ids"
                        ],
                        "kg_evidence_ids": result[
                            "kg_evidence_ids"
                        ],
                        "reasoning": result["reasoning"],
                    },
                )

                log(
                    "[V17]",
                    row["date"],
                    ticker,
                    "SEC=", len(sec_hits),
                    "KG=", len(kg_hits),
                    "impact=", result["impact"],
                    "risk=", result["risk"],
                    "confidence=", result["confidence"],
                )

    finally:
        cache.close()

        # Save partial valid progress even if
        # a later LLM call fails.
        if rows:
            pd.DataFrame(rows).to_csv(
                signals_path,
                index=False,
            )

    signals = pd.DataFrame(rows)

    if len(signals) != (
        len(dates) * len(args.tickers)
    ):
        raise RuntimeError(
            "Incomplete company-date signals"
        )

    if signals.duplicated(
        ["date", "ticker"]
    ).any():
        raise RuntimeError(
            "Duplicate company-date signals"
        )

    values = signals[
        ["impact", "risk", "confidence"]
    ].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "Nonfinite financial signals"
        )

    manifest = {
        "experiment": "FinRL-X V17-A/B",
        "architecture": (
            "PIT SEC Retrieval + Temporal KG "
            "1-hop Retrieval + Qwen Reasoning"
        ),
        "model": args.llm,
        "ollama_host": args.ollama_host,
        "decision_dates": len(dates),
        "tickers": args.tickers,
        "signal_rows": len(signals),
        "sec_source_rows": len(sec),
        "kg_company_edges": len(kg),
        "kg_relations": sorted(
            kg["relation"].unique().tolist()
        ),
        "sec_top_k": args.sec_top_k,
        "kg_top_k": args.kg_top_k,
        "cache_hits": cache_hits,
        "llm_calls": llm_calls,
        "failures": failures,
        "test_2022_2023_used": False,
        "limitations": [
            "SEC metadata PIT aligned, excerpt content not independently verified.",
            "KG first-observed date is not certified public release time.",
            "Numbered FinDKG relations have unknown semantic labels.",
            "LLM numeric scores are judgments, not calibrated probabilities.",
            "LLM citations are checked for membership, not factual entailment.",
            "SEC retrieval is recency-oriented rather than semantic reranking.",
            "KG retrieval is direct 1-hop and recency-oriented.",
            "No downstream PPO training in this stage.",
        ],
    }

    (
        args.out / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 70)
    log("V17-A/B SUMMARY")
    log("=" * 70)
    log(signals.to_string(index=False))
    log()
    log("[CACHE HITS]", cache_hits)
    log("[LLM CALLS]", llm_calls)
    log("[OUTPUT]", signals_path)
    log()
    log("V17-A/B: PASS")


if __name__ == "__main__":
    main()
