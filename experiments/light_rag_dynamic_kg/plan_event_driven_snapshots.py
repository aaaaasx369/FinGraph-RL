#!/usr/bin/env python3
"""
Event-driven LightRAG Snapshot Planner.

Read-only inputs:
  1. Historical KG event CSV
  2. Historical SEC filing CSV (optional)
  3. Existing FinRL-X trading calendar
  4. Existing LightRAG output filenames

Outputs:
  event_driven_jobs.csv
  event_driven_audit.json

No LLM calls. No training. No historical source modification.

Fail-closed PIT policy:
- Must explicitly identify the historical availability timestamp.
- Schedule on the first trading date STRICTLY AFTER availability date.
- Never use event occurrence date as a substitute for availability.
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TICKERS = ("AAPL", "AMZN", "GOOGL", "MSFT", "NVDA")

DUAL = ROOT / (
    "results/light_rag_dynamic_kg/kg_dual_channel_v1/"
    "kg_dual_channel_finrl_aligned.npz"
)

RAG_DIR = ROOT / "results/light_rag_dynamic_kg/ablation_v1"

OUT_DEFAULT = (
    "results/light_rag_dynamic_kg/event_driven_v1"
)

SYMBOL_COLUMNS = (
    "ticker", "symbol", "stock", "company_ticker",
    "focal_ticker",
)

AVAILABLE_COLUMNS = (
    "available_at", "known_at", "public_at",
    "published_at", "accepted_at", "acceptance_datetime",
    "acceptanceDateTime", "first_public_at",
)

REL_COLUMNS = (
    "relation", "relation_type", "predicate",
    "rel", "relation_code",
)

HEAD_COLUMNS = (
    "head", "subject", "source_entity", "entity1",
)

TAIL_COLUMNS = (
    "tail", "object", "target_entity", "entity2",
)

FORM_COLUMNS = (
    "form", "form_type", "filing_type",
)

IDENTITY_COLUMNS = (
    "accession_number", "accession", "filing_id",
    "document_id",
)


def choose(columns, supplied, defaults, label, required=True):
    lookup = {str(c).lower(): c for c in columns}

    if supplied:
        if supplied.lower() not in lookup:
            raise ValueError(
                f"{label}: column {supplied!r} not found. "
                f"Available: {columns}"
            )
        return lookup[supplied.lower()]

    for name in defaults:
        if name.lower() in lookup:
            return lookup[name.lower()]

    if required:
        raise ValueError(
            f"{label}: cannot safely infer column. "
            f"Choose explicitly from: {columns}"
        )

    return None


def parse_time(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    if not s or s.lower() in (
        "nan", "none", "nat", "null"
    ):
        return None

    try:
        ts = pd.Timestamp(s)

        if pd.isna(ts):
            return None

        if ts.tzinfo is not None:
            # Normalize timezone-aware records to UTC.
            ts = ts.tz_convert("UTC").tz_localize(None)

        return ts

    except Exception:
        return None


def date_to_next_trading_day(ts, calendar):
    """
    Conservative date-level policy.
    Information on a date cannot be used until
    a subsequent trading date.

    Timezone-aware records are normalized to UTC.
    Detailed market-close timezone certification
    must still be performed separately.
    """
    i = calendar.searchsorted(
        ts.normalize(), side="right"
    )

    if i >= len(calendar):
        return None

    return calendar[i]


def normalize_ticker(value):
    s = str(value).strip().upper()

    if s in TICKERS:
        return s

    return None


def read_csv(path, chunk_size=50000):
    return pd.read_csv(
        path,
        chunksize=chunk_size,
        dtype=str,
        encoding="utf-8-sig",
        on_bad_lines="skip",
        low_memory=False,
    )


def load_calendar():
    if not DUAL.exists():
        raise FileNotFoundError(DUAL)

    # Existing locally generated NPZ has object-typed dates.
    with np.load(DUAL, allow_pickle=True) as z:
        calendar = pd.DatetimeIndex(
            pd.to_datetime(z["dates"].astype(str))
        )

        tickers = [
            str(x) for x in z["tickers"]
        ]

    if tickers != list(TICKERS):
        raise RuntimeError(
            f"Unexpected ticker order: {tickers}"
        )

    if calendar.has_duplicates:
        raise RuntimeError("Duplicate trading dates")

    if not calendar.is_monotonic_increasing:
        raise RuntimeError("Trading calendar not sorted")

    return calendar


def collect_kg(args, calendar, audit):
    """
    Event trigger based on newly observed, distinct KG edges.

    This first version requires a focal ticker column
    and head/relation/tail columns.

    No inferred mapping from an arbitrary entity to a
    traded ticker: that would need a separate entity map.
    """
    if not args.kg_csv:
        return {}

    path = Path(args.kg_csv).expanduser()

    if not path.is_file():
        raise FileNotFoundError(path)

    first = pd.read_csv(path, nrows=0)
    columns = list(first.columns)

    tc = choose(
        columns, args.kg_ticker,
        SYMBOL_COLUMNS, "KG ticker"
    )
    ac = choose(
        columns, args.kg_available,
        AVAILABLE_COLUMNS, "KG historical availability"
    )
    hc = choose(
        columns, args.kg_head,
        HEAD_COLUMNS, "KG head"
    )
    rc = choose(
        columns, args.kg_relation,
        REL_COLUMNS, "KG relation"
    )
    oc = choose(
        columns, args.kg_tail,
        TAIL_COLUMNS, "KG tail"
    )

    print("[KG SCHEMA]", {
        "ticker": tc,
        "available_at": ac,
        "head": hc,
        "relation": rc,
        "tail": oc,
    })

    per_day = defaultdict(set)

    audit["kg_rows_read"] = 0
    audit["kg_rows_usable"] = 0

    for chunk in read_csv(path):
        audit["kg_rows_read"] += len(chunk)

        for ticker, available, head, relation, tail in (
            zip(
                chunk[tc],
                chunk[ac],
                chunk[hc],
                chunk[rc],
                chunk[oc],
            )
        ):
            ticker = normalize_ticker(ticker)
            at = parse_time(available)

            if ticker is None or at is None:
                continue

            if any(pd.isna(x) for x in (
                head, relation, tail
            )):
                continue

            trade_date = date_to_next_trading_day(
                at, calendar
            )

            if trade_date is None:
                continue

            if trade_date.year > 2021:
                continue

            edge = (
                str(head).strip(),
                str(relation).strip(),
                str(tail).strip(),
            )

            per_day[(ticker, trade_date)].add(edge)

            audit["kg_rows_usable"] += 1

    # Process dates chronologically per ticker.
    # An edge already observed in an earlier snapshot
    # is not counted as a new relationship.
    by_ticker = defaultdict(list)

    for (ticker, day), edges in per_day.items():
        by_ticker[ticker].append((day, edges))

    result = {}

    for ticker, records in by_ticker.items():
        seen = set()

        for day, edges in sorted(
            records, key=lambda x: x[0]
        ):
            new_edges = edges - seen
            seen.update(edges)

            if len(new_edges) < args.min_new_kg_edges:
                continue

            result[(ticker, day)] = {
                "kg_new_edges": len(new_edges),
                "kg_total_edges": len(edges),
            }

    return result


def collect_sec(args, calendar, audit):
    if not args.sec_csv:
        return {}

    path = Path(args.sec_csv).expanduser()

    if not path.is_file():
        raise FileNotFoundError(path)

    first = pd.read_csv(path, nrows=0)
    columns = list(first.columns)

    tc = choose(
        columns, args.sec_ticker,
        SYMBOL_COLUMNS, "SEC ticker"
    )
    ac = choose(
        columns, args.sec_available,
        AVAILABLE_COLUMNS, "SEC historical availability"
    )
    fc = choose(
        columns, args.sec_form,
        FORM_COLUMNS, "SEC filing form"
    )
    ic = choose(
        columns, args.sec_identity,
        IDENTITY_COLUMNS, "SEC accession",
        required=False,
    )

    print("[SEC SCHEMA]", {
        "ticker": tc,
        "available_at": ac,
        "form": fc,
        "identity": ic,
    })

    per_day = defaultdict(set)

    audit["sec_rows_read"] = 0
    audit["sec_rows_usable"] = 0

    accepted_forms = {
        "10-K", "10-Q", "8-K",
        "10-K/A", "10-Q/A", "8-K/A",
    }

    for chunk in read_csv(path):
        audit["sec_rows_read"] += len(chunk)

        for _, row in chunk.iterrows():
            ticker = normalize_ticker(row[tc])
            at = parse_time(row[ac])
            form = str(row[fc]).strip().upper()

            if (
                ticker is None or at is None
                or form not in accepted_forms
            ):
                continue

            trade_date = date_to_next_trading_day(
                at, calendar
            )

            if trade_date is None:
                continue

            if trade_date.year > 2021:
                continue

            if ic is not None and pd.notna(row[ic]):
                identity = str(row[ic]).strip()
            else:
                identity = (
                    ticker + "|" + form + "|" + str(at)
                )

            per_day[(ticker, trade_date)].add(identity)
            audit["sec_rows_usable"] += 1

    return {
        key: {"sec_new_filings": len(values)}
        for key, values in per_day.items()
    }


def existing_jobs():
    found = set()

    for p in RAG_DIR.glob("*_A_KG_RAG.json"):
        match = re.fullmatch(
            r"([A-Z]+)_(\d{4}-\d{2}-\d{2})_A_KG_RAG\.json",
            p.name,
        )

        if match:
            found.add((match.group(1), match.group(2)))

    return found


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--kg-csv")
    p.add_argument("--sec-csv")

    p.add_argument("--kg-ticker")
    p.add_argument("--kg-available")
    p.add_argument("--kg-head")
    p.add_argument("--kg-relation")
    p.add_argument("--kg-tail")

    p.add_argument("--sec-ticker")
    p.add_argument("--sec-available")
    p.add_argument("--sec-form")
    p.add_argument("--sec-identity")

    p.add_argument(
        "--min-new-kg-edges", type=int, default=3
    )
    p.add_argument(
        "--min-gap-days", type=int, default=7
    )
    p.add_argument(
        "--max-per-ticker-month",
        type=int, default=3,
    )

    p.add_argument(
        "--out", default=OUT_DEFAULT
    )

    args = p.parse_args()

    if not args.kg_csv and not args.sec_csv:
        p.error("Supply at least --kg-csv or --sec-csv")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    calendar = load_calendar()

    audit = {
        "calendar_first": str(calendar[0].date()),
        "calendar_last": str(calendar[-1].date()),
        "policy": "strictly_next_trading_day",
        "historical_pit_certified": False,
        "run_timestamp": datetime.now().isoformat(),
        "min_new_kg_edges": args.min_new_kg_edges,
        "min_gap_days": args.min_gap_days,
        "max_per_ticker_month": args.max_per_ticker_month,
    }

    kg = collect_kg(args, calendar, audit)
    sec = collect_sec(args, calendar, audit)

    merged = defaultdict(dict)

    for key, values in kg.items():
        merged[key].update(values)

    for key, values in sec.items():
        merged[key].update(values)

    done = existing_jobs()

    rows = []
    last_selected = {}
    month_counts = defaultdict(int)

    for (ticker, day), info in sorted(
        merged.items(), key=lambda x: (x[0][1], x[0][0])
    ):
        date = day.strftime("%Y-%m-%d")
        key = (ticker, date)

        if key in done:
            status = "ALREADY_EXISTS"
        else:
            prev = last_selected.get(ticker)
            month_key = (ticker, date[:7])

            too_close = (
                prev is not None
                and (day - prev).days < args.min_gap_days
            )

            if too_close:
                status = "SKIPPED_MIN_GAP"
            elif (
                month_counts[month_key]
                >= args.max_per_ticker_month
            ):
                status = "SKIPPED_MONTH_LIMIT"
            else:
                status = "CANDIDATE"
                last_selected[ticker] = day
                month_counts[month_key] += 1

        kg_count = info.get("kg_new_edges", 0)
        sec_count = info.get("sec_new_filings", 0)

        trigger = (
            "KG+SEC" if kg_count and sec_count
            else "KG" if kg_count
            else "SEC"
        )

        rows.append({
            "ticker": ticker,
            "date": date,
            "trigger": trigger,
            "kg_new_edges": kg_count,
            "sec_new_filings": sec_count,
            "status": status,
            "command": (
                ".venv-drl/bin/python -u "
                "experiments/light_rag_dynamic_kg/"
                "run_official_lightrag.py "
                f"--ticker {ticker} --date {date} "
                "--kg-k 80 --sec-k 5"
            ),
        })

    columns = [
        "ticker", "date", "trigger",
        "kg_new_edges", "sec_new_filings",
        "status", "command",
    ]

    job_csv = out / "event_driven_jobs.csv"

    with job_csv.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {
        status: sum(r["status"] == status for r in rows)
        for status in sorted(set(
            r["status"] for r in rows
        ))
    }

    audit["candidate_counts"] = counts
    audit["total_trigger_dates"] = len(rows)
    audit["kg_trigger_dates"] = len(kg)
    audit["sec_trigger_dates"] = len(sec)
    audit["existing_kg_rag_snapshots"] = len(done)
    audit["light_rag_execution_started"] = False

    audit_path = out / "event_driven_audit.json"
    audit_path.write_text(
        json.dumps(
            audit, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("EVENT-DRIVEN LIGHTRAG PLANNER")
    print("=" * 78)

    print(json.dumps(
        audit, ensure_ascii=False, indent=2
    ))

    print("\nFIRST 20 CANDIDATES")

    candidates = [
        x for x in rows
        if x["status"] == "CANDIDATE"
    ]

    for row in candidates[:20]:
        print(
            row["ticker"],
            row["date"],
            row["trigger"],
            "KG=", row["kg_new_edges"],
            "SEC=", row["sec_new_filings"],
        )

    print("\n[SAVED]", job_csv)
    print("[SAVED]", audit_path)
    print("[INFO] NO LIGHTRAG EXECUTION")
    print("[INFO] NO MODEL TRAINING")


if __name__ == "__main__":
    main()
