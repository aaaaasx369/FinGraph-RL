#!/usr/bin/env python3
"""
Event-driven LightRAG Planner V3.

- Preserve V2 SEC PIT trigger logic.
- Fix verified FinDKG -> COMPANY:TICKER mapping.
- Count distinct KG events for each ticker/day.
- Generate candidates only; DO NOT execute LightRAG.
- Preserve original historical batch and model files.
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg import (
    plan_event_driven_v2 as v2
)

MAPPING_REPORT = (
    ROOT
    / "outputs/unified_temporal_kg_v3/entity_mapping_report.csv"
)

V3_OUT = (
    ROOT
    / "results/light_rag_dynamic_kg/event_driven_v3"
)

PORTFOLIO_MAP = (
    ROOT / "kg/data/findkg/portfolio_entity_map.csv"
)


def verified_company_mapping():
    report = pd.read_csv(MAPPING_REPORT)
    portfolio = pd.read_csv(PORTFOLIO_MAP)

    required = {
        "ticker", "old_id", "canonical_id", "verified"
    }

    if not required.issubset(report.columns):
        raise RuntimeError(
            "Mapping report columns changed"
        )

    result = {}

    for row in portfolio.itertuples(index=False):
        ticker = str(row.ticker).upper()
        original = int(row.entity_id)

        expected_old = f"FINDKG:{original}"
        expected_canonical = f"COMPANY:{ticker}"

        matches = report[
            (report["ticker"] == ticker)
            & (report["old_id"] == expected_old)
            & (report["canonical_id"] == expected_canonical)
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Mapping missing or ambiguous: {ticker}"
            )

        verified = str(
            matches.iloc[0]["verified"]
        ).strip().lower()

        if verified not in ("true", "1"):
            raise RuntimeError(
                f"Mapping unverified: {ticker}"
            )

        result[expected_canonical] = ticker

    if len(result) != 5:
        raise RuntimeError("Expected exactly 5 verified mappings")

    return result


def load_kg_v3(calendar, stats):
    print("\n[KG V3] Verified canonical company mapping")

    company_map = verified_company_mapping()

    print("MAPPING:", company_map)

    columns = [
        "event_id",
        "head",
        "relation",
        "tail",
        "available_at",
        "source",
        "time_precision",
    ]

    df = pd.read_csv(
        v2.KG_PATH,
        usecols=columns,
        dtype=str,
        low_memory=False,
    )

    stats["kg_total_events"] = int(len(df))
    stats["kg_by_source"] = {
        str(k): int(v)
        for k, v in df["source"].value_counts().items()
    }

    triggers = defaultdict(
        lambda: {
            "sec_accessions": set(),
            "kg_event_ids": set(),
        }
    )

    by_source = Counter()
    by_ticker = Counter()
    missing_date = 0
    outside_calendar = 0
    rows_directly_linked = 0

    for row in df.itertuples(index=False):
        tickers = set()

        h = company_map.get(str(row.head))
        t = company_map.get(str(row.tail))

        if h:
            tickers.add(h)

        if t:
            tickers.add(t)

        if not tickers:
            continue

        rows_directly_linked += 1

        available = v2.normalize_date(row.available_at)

        if available is None:
            missing_date += 1
            continue

        day = v2.next_session(available, calendar)

        if day is None:
            outside_calendar += 1
            continue

        day_text = day.strftime("%Y-%m-%d")

        for ticker in tickers:
            key = (ticker, day_text)

            triggers[key]["kg_event_ids"].add(
                str(row.event_id)
            )

            by_source[str(row.source)] += 1
            by_ticker[ticker] += 1

    stats["kg_verified_canonical_mapping"] = True
    stats["kg_mapping"] = company_map
    stats["kg_direct_event_rows"] = rows_directly_linked
    stats["kg_event_links_in_calendar"] = int(
        sum(by_ticker.values())
    )
    stats["kg_event_links_by_ticker"] = dict(by_ticker)
    stats["kg_event_links_by_source"] = dict(by_source)
    stats["kg_missing_availability_dates"] = missing_date
    stats["kg_outside_calendar"] = outside_calendar
    stats["kg_trigger_stock_days"] = len(triggers)

    # Mapping verification is NOT a PIT time certification.
    stats["kg_available_at_pit_certified"] = False
    stats["kg_time_precision_note"] = (
        "Weekly FinDKG timestamps are dataset timestamps; "
        "historical publication-time equivalence is unverified."
    )

    print("Direct KG event rows:", rows_directly_linked)
    print("KG event links in calendar:", sum(by_ticker.values()))
    print("KG trigger stock-days:", len(triggers))
    print("KG links by ticker:", dict(by_ticker))

    return triggers


def main():
    v2.load_kg = load_kg_v3
    v2.OUT = V3_OUT
    v2.main()


if __name__ == "__main__":
    main()
