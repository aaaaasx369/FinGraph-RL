#!/usr/bin/env python3
"""
Event-Driven LightRAG Planner V2

Sources:
- Unified Temporal KG V3
- Audited SEC PIT Retrieval Parquet
- Native FinRL-X trading calendar

Outputs:
- event_jobs.csv
- source_audit.json

This script NEVER launches RAG or RL training.
No inferred FINDKG entity-to-ticker mapping.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

KG_PATH = ROOT / (
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)

SEC_PATH = ROOT / (
    "results/aegis_rl_v1/v11f_sec_pit/"
    "sec_pit_retrieval.parquet"
)

DUAL_PATH = ROOT / (
    "results/light_rag_dynamic_kg/kg_dual_channel_v1/"
    "kg_dual_channel_finrl_aligned.npz"
)

RAG_PATH = ROOT / (
    "results/light_rag_dynamic_kg/ablation_v1"
)

OUT = ROOT / (
    "results/light_rag_dynamic_kg/event_driven_v2"
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
FORMS = {"10-K", "10-Q", "8-K"}


def normalize_date(x):
    try:
        value = pd.to_datetime(x, errors="coerce")
        if pd.isna(value):
            return None

        if value.tzinfo is not None:
            value = value.tz_convert("UTC").tz_localize(None)

        return value.normalize()
    except Exception:
        return None


def text_value(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def load_calendar():
    with np.load(DUAL_PATH, allow_pickle=True) as z:
        dates = pd.DatetimeIndex(
            pd.to_datetime(z["dates"].astype(str))
        )
        symbols = [str(s) for s in z["tickers"]]

    assert symbols == TICKERS, symbols
    assert not dates.has_duplicates
    assert dates.is_monotonic_increasing

    return dates


def next_session(day, calendar):
    """Conservative: strictly after the source's usable date."""
    if day is None:
        return None

    index = calendar.searchsorted(day, side="right")
    if index >= len(calendar):
        return None

    return calendar[index]


def direct_ticker(value):
    """
    Only unambiguous, explicit ticker identifiers.

    FINDKG:n is intentionally NOT mapped automatically.
    """
    value = text_value(value).upper()

    if value in TICKERS:
        return value

    for prefix in ("TICKER:", "STOCK:", "SYMBOL:"):
        if value.startswith(prefix):
            candidate = value[len(prefix):]
            if candidate in TICKERS:
                return candidate

    return None


def load_sec(calendar, stats):
    print("\n[SEC] Loading PIT filing records")

    columns = [
        "ticker",
        "form",
        "current_accession",
        "matched_accession",
        "pit_usable_from",
        "pit_metadata_status",
        "pit_failure_reason",
        "v11d_status",
    ]

    df = pd.read_parquet(SEC_PATH, columns=columns)

    stats["sec_evidence_rows"] = int(len(df))

    for c in columns:
        df[c] = df[c].fillna("").astype(str)

    df["ticker"] = df["ticker"].str.upper().str.strip()
    df["form"] = df["form"].str.upper().str.strip()

    df = df[
        df["ticker"].isin(TICKERS)
        & df["form"].isin(FORMS)
        & (df["pit_metadata_status"] == "METADATA_MATCHED")
        & (df["v11d_status"] == "VERIFIED_ACCESSION")
        & (df["pit_failure_reason"].str.strip() == "")
    ].copy()

    # A PIT filing may have many retrieved evidence excerpts.
    # Filing trigger should be counted only once per accession.
    df["accession"] = df["matched_accession"].where(
        df["matched_accession"].str.strip() != "",
        df["current_accession"],
    )

    df = df[df["accession"].str.strip() != ""]

    df["usable_day"] = pd.to_datetime(
        df["pit_usable_from"], errors="coerce"
    ).dt.normalize()

    df = df.dropna(subset=["usable_day"])

    # Detect inconsistent PIT dates for the same accession.
    conflict = (
        df.groupby(["ticker", "accession"])["usable_day"]
        .nunique()
    )

    bad_keys = set(
        conflict[conflict > 1].index.tolist()
    )

    stats["sec_conflicting_filing_dates"] = len(bad_keys)

    if bad_keys:
        bad = pd.MultiIndex.from_frame(
            df[["ticker", "accession"]]
        ).isin(bad_keys)

        df = df[~bad].copy()

    df = df.drop_duplicates(
        ["ticker", "accession"]
    ).copy()

    stats["sec_distinct_valid_filings"] = int(len(df))

    triggers = defaultdict(lambda: {
        "sec_accessions": set(),
        "kg_event_ids": set(),
    })

    for row in df.itertuples(index=False):
        trade_day = next_session(
            row.usable_day, calendar
        )

        if trade_day is None:
            continue

        key = (
            row.ticker,
            trade_day.strftime("%Y-%m-%d"),
        )

        triggers[key]["sec_accessions"].add(
            row.accession
        )

    stats["sec_trigger_stock_days"] = sum(
        bool(v["sec_accessions"])
        for v in triggers.values()
    )

    return triggers


def load_kg(calendar, stats):
    print("\n[KG] Loading Unified Temporal KG")

    cols = [
        "event_id",
        "head",
        "relation",
        "tail",
        "available_at",
        "source",
        "time_precision",
    ]

    df = pd.read_csv(
        KG_PATH,
        usecols=cols,
        dtype=str,
        low_memory=False,
    )

    stats["kg_total_events"] = int(len(df))
    stats["kg_by_source"] = {
        str(k): int(v)
        for k, v in df["source"].value_counts().items()
    }

    triggers = defaultdict(lambda: {
        "sec_accessions": set(),
        "kg_event_ids": set(),
    })

    unlinked = Counter()
    linked = 0

    for row in df.itertuples(index=False):
        head_ticker = direct_ticker(row.head)
        tail_ticker = direct_ticker(row.tail)

        matched = {
            ticker
            for ticker in (head_ticker, tail_ticker)
            if ticker is not None
        }

        if not matched:
            unlinked[text_value(row.source)] += 1
            continue

        available = normalize_date(row.available_at)

        trade_day = next_session(
            available, calendar
        )

        if trade_day is None:
            continue

        for ticker in matched:
            key = (
                ticker,
                trade_day.strftime("%Y-%m-%d"),
            )

            triggers[key]["kg_event_ids"].add(
                str(row.event_id)
            )

            linked += 1

    stats["kg_direct_ticker_links"] = linked
    stats["kg_unresolved_by_source"] = dict(unlinked)

    # Explicitly do not claim that KG timestamp
    # is certified historical publication time.
    stats["kg_available_at_pit_certified"] = False

    return triggers


def known_snapshots():
    result = set()

    for path in RAG_PATH.glob("*_A_KG_RAG.json"):
        m = re.fullmatch(
            r"([A-Z]+)_(\d{4}-\d{2}-\d{2})"
            r"_A_KG_RAG\.json",
            path.name,
        )

        if not m:
            continue

        try:
            record = json.loads(
                path.read_text(encoding="utf-8")
            )
        except Exception:
            continue

        # Only classify parseable matching records as existing.
        if (
            str(record.get("ticker", "")).upper()
            == m.group(1)
            and str(record.get("date", ""))
            == m.group(2)
        ):
            result.add((m.group(1), m.group(2)))

    return result


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--min-gap-days",
        type=int,
        default=7,
    )

    p.add_argument(
        "--max-per-month",
        type=int,
        default=3,
    )

    p.add_argument(
        "--min-new-kg",
        type=int,
        default=3,
    )

    args = p.parse_args()

    if args.min_gap_days < 0:
        p.error("--min-gap-days must be >= 0")

    if args.max_per_month < 1:
        p.error("--max-per-month must be >= 1")

    OUT.mkdir(parents=True, exist_ok=True)

    calendar = load_calendar()

    stats = {
        "calendar_start": str(calendar[0].date()),
        "calendar_end": str(calendar[-1].date()),
        "trigger_policy": (
            "first trading day strictly after "
            "source PIT usable date"
        ),
        "light_rag_executed": False,
        "pca_fitted": False,
        "kg_entity_mapping_verified": False,
        "pit_fully_certified": False,
        "sec_excerpt_content_certified": False,
    }

    sec = load_sec(calendar, stats)
    kg = load_kg(calendar, stats)

    combined = defaultdict(lambda: {
        "sec_accessions": set(),
        "kg_event_ids": set(),
    })

    for collection in (sec, kg):
        for key, values in collection.items():
            combined[key]["sec_accessions"].update(
                values["sec_accessions"]
            )

            combined[key]["kg_event_ids"].update(
                values["kg_event_ids"]
            )

    existing = known_snapshots()

    rows = []
    last_selected = {}
    month_counts = Counter()

    for (ticker, date), signals in sorted(
        combined.items(),
        key=lambda item: (
            item[0][1],
            item[0][0],
        ),
    ):
        sec_n = len(signals["sec_accessions"])
        kg_n = len(signals["kg_event_ids"])

        if not (sec_n or kg_n >= args.min_new_kg):
            continue

        day = pd.Timestamp(date)
        month = (ticker, date[:7])

        if sec_n and kg_n:
            trigger = "KG_AND_SEC"
        elif sec_n:
            trigger = "SEC"
        else:
            trigger = "KG"

        if (ticker, date) in existing:
            status = "EXISTING_SNAPSHOT"
        elif (
            ticker in last_selected
            and (day - last_selected[ticker]).days
            < args.min_gap_days
        ):
            status = "SKIP_MIN_GAP"
        elif month_counts[month] >= args.max_per_month:
            status = "SKIP_MONTH_LIMIT"
        else:
            status = "CANDIDATE"
            last_selected[ticker] = day
            month_counts[month] += 1

        command = (
            ".venv-drl/bin/python -u "
            "experiments/light_rag_dynamic_kg/"
            "run_official_lightrag.py "
            f"--ticker {ticker} --date {date} "
            "--kg-k 80 --sec-k 5"
        )

        rows.append({
            "ticker": ticker,
            "date": date,
            "trigger": trigger,
            "new_sec_filings": sec_n,
            "new_kg_events": kg_n,
            "status": status,
            "command_NOT_EXECUTED": command,
        })

    jobs = pd.DataFrame(rows, columns=[
        "ticker", "date", "trigger",
        "new_sec_filings", "new_kg_events",
        "status", "command_NOT_EXECUTED",
    ])

    jobs_path = OUT / "event_jobs.csv"
    jobs.to_csv(jobs_path, index=False)

    stats["job_status_counts"] = {
        str(k): int(v)
        for k, v in (
            jobs["status"].value_counts().items()
            if len(jobs) else []
        )
    }

    stats["total_trigger_dates"] = len(jobs)
    stats["existing_matching_snapshots"] = len(existing)

    audit_path = OUT / "source_audit.json"
    audit_path.write_text(
        json.dumps(
            stats, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("EVENT-DRIVEN PLANNER V2 COMPLETE")
    print("=" * 75)

    print(json.dumps(
        stats, ensure_ascii=False, indent=2
    ))

    print("\nFIRST 15 CANDIDATES")

    if len(jobs):
        print(
            jobs[jobs["status"] == "CANDIDATE"][
                [
                    "ticker",
                    "date",
                    "trigger",
                    "new_sec_filings",
                    "new_kg_events",
                ]
            ].head(15).to_string(index=False)
        )

    print("\nSAVED:", jobs_path)
    print("SAVED:", audit_path)
    print("NO RAG OR PPO EXECUTION")


if __name__ == "__main__":
    main()
