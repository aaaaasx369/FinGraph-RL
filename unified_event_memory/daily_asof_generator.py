#!/usr/bin/env python3

"""
Unified Temporal KG V3
Daily As-of Graph Generator

Commands:
    build
    asof --date YYYY-MM-DD
    replay --start YYYY-MM-DD --end YYYY-MM-DD

Design:
    - One event is stored once.
    - SEC relations are historical observations.
    - Persistent state means last observed, not verified active.
    - SEC signal numerical attributes remain separate.
    - No future source-reported dates in as-of outputs.
    - Strict point-in-time correctness is not yet certified.
"""

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "outputs/unified_temporal_kg_v3"
OUT = ROOT / "outputs/unified_daily_asof_v1"

DB = OUT / "temporal_kg.sqlite"

EVENT_FILE = SRC / "unified_temporal_edges.csv"
SIGNAL_FILE = SRC / "sec_signal_nodes.csv"

EVENT_FIELDS = [
    "event_id",
    "head",
    "relation",
    "tail",
    "head_id",
    "relation_id",
    "tail_id",
    "available_at",
    "event_date",
    "source",
    "time_precision",
    "persistence",
    "occurrence_count",
    "confidence",
    "source_id",
]

STATE_FIELDS = [
    "head",
    "relation",
    "tail",
    "head_id",
    "relation_id",
    "tail_id",
    "first_observed_at",
    "last_observed_at",
    "age_days",
    "observation_count",
    "latest_event_id",
    "latest_source_id",
    "validity_status",
]


def read_csv(path):
    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:
        yield from csv.DictReader(f)


def write_csv(path, rows, fields):
    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_date(value):
    return date.fromisoformat(value[:10])


def connect():
    if not DB.exists():
        raise RuntimeError(
            "Database not found. Run build first."
        )

    return sqlite3.connect(str(DB))


# =====================================================
# BUILD INDEXED TEMPORAL KG
# =====================================================

def build():

    if not EVENT_FILE.exists():
        raise RuntimeError(
            f"Missing input: {EVENT_FILE}"
        )

    if not SIGNAL_FILE.exists():
        raise RuntimeError(
            f"Missing input: {SIGNAL_FILE}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    if DB.exists():
        DB.unlink()

    con = sqlite3.connect(str(DB))
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            available_at TEXT NOT NULL,
            source TEXT NOT NULL,
            head TEXT NOT NULL,
            relation TEXT NOT NULL,
            tail TEXT NOT NULL,
            payload TEXT NOT NULL
        );

        CREATE INDEX idx_events_time
        ON events(available_at);

        CREATE INDEX idx_events_source_time
        ON events(source, available_at);

        CREATE INDEX idx_events_state
        ON events(
            source, head, relation, tail,
            available_at
        );

        CREATE TABLE signal_attributes (
            node TEXT PRIMARY KEY,
            payload TEXT NOT NULL
        );
    """)

    source_counts = Counter()
    daily_counts = Counter()

    first_date = None
    last_date = None
    inserted = 0

    for r in read_csv(EVENT_FILE):

        d = parse_date(
            r["available_at"]
        ).isoformat()

        cur.execute(
            """
            INSERT INTO events VALUES (
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                r["event_id"],
                d,
                r["source"],
                r["head"],
                r["relation"],
                r["tail"],
                json.dumps(
                    r,
                    ensure_ascii=False
                ),
            )
        )

        source_counts[r["source"]] += 1
        daily_counts[
            (d, r["source"])
        ] += 1

        first_date = (
            d if first_date is None
            else min(first_date, d)
        )

        last_date = (
            d if last_date is None
            else max(last_date, d)
        )

        inserted += 1

    signal_count = 0

    for r in read_csv(SIGNAL_FILE):

        cur.execute(
            """
            INSERT INTO signal_attributes
            VALUES (?, ?)
            """,
            (
                r["node"],
                json.dumps(
                    r,
                    ensure_ascii=False
                )
            )
        )

        signal_count += 1

    con.commit()
    con.close()

    # Daily manifest includes zero-event dates.
    start = parse_date(first_date)
    end = parse_date(last_date)

    manifest = []

    current = start

    while current <= end:

        d = current.isoformat()

        news = daily_counts[
            (d, "FINDKG")
        ]

        relations = daily_counts[
            (d, "SEC_RELATION")
        ]

        signals = daily_counts[
            (d, "SEC_RAG")
        ]

        manifest.append({
            "date": d,
            "new_findkg_events": news,
            "new_sec_relation_observations": relations,
            "new_sec_signal_events": signals,
            "total_new_events": (
                news + relations + signals
            ),
            "has_event_update": int(
                news + relations + signals > 0
            ),
            "has_state_update": int(
                relations > 0 or signals > 0
            ),
        })

        current += timedelta(days=1)

    write_csv(
        OUT / "daily_manifest.csv",
        manifest,
        list(manifest[0])
    )

    metadata = {
        "version": "unified_daily_asof_v1",
        "events": inserted,
        "source_counts": dict(
            source_counts
        ),
        "signal_attributes": signal_count,
        "date_range": [
            first_date,
            last_date
        ],
        "calendar_days": len(manifest),
        "days_with_events": sum(
            x["has_event_update"]
            for x in manifest
        ),
        "days_with_state_updates": sum(
            x["has_state_update"]
            for x in manifest
        ),
        "strict_pit_certified": False,
        "relation_validity_verified": False,
    }

    (
        OUT / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("=" * 65)
    print("DAILY AS-OF INDEX BUILD")
    print("=" * 65)

    print(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        )
    )

    print("BUILD PASS")
    print("Database:", DB)


# =====================================================
# QUERY AS-OF GRAPH
# =====================================================

def asof(target_date, window_days=30, output=None):

    if window_days < 1:
        raise ValueError(
            "window_days must be >= 1"
        )

    target = parse_date(target_date)

    lower = target - timedelta(
        days=window_days - 1
    )

    con = connect()
    con.row_factory = sqlite3.Row

    # Recent events: source-reported dates only.
    recent_raw = con.execute(
        """
        SELECT payload
        FROM events
        WHERE available_at >= ?
          AND available_at <= ?
        ORDER BY available_at, event_id
        """,
        (
            lower.isoformat(),
            target.isoformat()
        )
    ).fetchall()

    recent = [
        json.loads(r["payload"])
        for r in recent_raw
    ]

    # Latest historical observation for each
    # directed company relationship.
    relation_rows = con.execute(
        """
        SELECT
            head,
            relation,
            tail,
            MIN(available_at) AS first_date,
            MAX(available_at) AS last_date,
            COUNT(*) AS observation_count

        FROM events

        WHERE source = 'SEC_RELATION'
          AND available_at <= ?

        GROUP BY head, relation, tail

        ORDER BY head, relation, tail
        """,
        (target.isoformat(),)
    ).fetchall()

    state = []

    for s in relation_rows:

        latest = con.execute(
            """
            SELECT payload
            FROM events

            WHERE source = 'SEC_RELATION'
              AND head = ?
              AND relation = ?
              AND tail = ?
              AND available_at <= ?

            ORDER BY available_at DESC, event_id DESC
            LIMIT 1
            """,
            (
                s["head"],
                s["relation"],
                s["tail"],
                target.isoformat()
            )
        ).fetchone()

        r = json.loads(
            latest["payload"]
        )

        last = parse_date(
            s["last_date"]
        )

        state.append({
            "head": s["head"],
            "relation": s["relation"],
            "tail": s["tail"],
            "head_id": r["head_id"],
            "relation_id": r["relation_id"],
            "tail_id": r["tail_id"],
            "first_observed_at": s["first_date"],
            "last_observed_at": s["last_date"],
            "age_days": (
                target - last
            ).days,
            "observation_count": (
                s["observation_count"]
            ),
            "latest_event_id": r["event_id"],
            "latest_source_id": r["source_id"],
            "validity_status": (
                "LAST_OBSERVED_UNVERIFIED"
            )
        })

    # Latest SEC signal per company.
    signal_raw = con.execute(
        """
        SELECT
            e.head AS company,
            e.available_at,
            e.event_id,
            s.payload

        FROM events e

        JOIN signal_attributes s
          ON e.tail = s.node

        WHERE e.source = 'SEC_RAG'
          AND e.available_at <= ?

        ORDER BY
            e.available_at DESC,
            e.event_id DESC
        """,
        (target.isoformat(),)
    ).fetchall()

    latest_signals = {}

    for r in signal_raw:

        company = r["company"]

        if company in latest_signals:
            continue

        attr = json.loads(
            r["payload"]
        )

        attr["last_observed_at"] = (
            r["available_at"]
        )

        attr["age_days"] = (
            target -
            parse_date(r["available_at"])
        ).days

        latest_signals[company] = attr

    con.close()

    # Verify no reported future dates.
    assert all(
        parse_date(
            r["available_at"]
        ) <= target
        for r in recent
    )

    assert all(
        parse_date(
            r["last_observed_at"]
        ) <= target
        for r in state
    )

    assert all(
        parse_date(
            r["last_observed_at"]
        ) <= target
        for r in latest_signals.values()
    )

    result = {
        "asof": target.isoformat(),
        "window_days": window_days,
        "recent_events": len(recent),
        "recent_by_source": dict(
            Counter(
                r["source"]
                for r in recent
            )
        ),
        "persistent_relation_candidates": (
            len(state)
        ),
        "companies_with_latest_sec_signal": (
            len(latest_signals)
        ),
        "strict_pit_certified": False,
    }

    if output:

        folder = Path(output)
        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        write_csv(
            folder / "recent_event_edges.csv",
            recent,
            EVENT_FIELDS
        )

        write_csv(
            folder / "persistent_state.csv",
            state,
            STATE_FIELDS
        )

        signal_rows = list(
            latest_signals.values()
        )

        signal_fields = [
            "node",
            "company",
            "available_at",
            "impact",
            "risk",
            "confidence",
            "reason",
            "last_observed_at",
            "age_days",
        ]

        write_csv(
            folder / "latest_sec_signals.csv",
            signal_rows,
            signal_fields
        )

        (
            folder / "query_metadata.json"
        ).write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

    print("=" * 65)
    print("AS-OF QUERY")
    print("=" * 65)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print("AS-OF PASS")

    return result


# =====================================================
# DAILY REPLAY MANIFEST
# =====================================================

def replay(start_date, end_date):

    start = parse_date(start_date)
    end = parse_date(end_date)

    if end < start:
        raise ValueError(
            "end must be >= start"
        )

    con = connect()

    rows = con.execute(
        """
        SELECT
            available_at,
            source,
            COUNT(*)

        FROM events

        WHERE available_at >= ?
          AND available_at <= ?

        GROUP BY available_at, source
        """,
        (
            start.isoformat(),
            end.isoformat()
        )
    ).fetchall()

    con.close()

    counts = Counter({
        (d, source): n
        for d, source, n in rows
    })

    output = []

    current = start

    while current <= end:

        d = current.isoformat()

        news = counts[
            d, "FINDKG"
        ]

        rel = counts[
            d, "SEC_RELATION"
        ]

        sec = counts[
            d, "SEC_RAG"
        ]

        output.append({
            "date": d,
            "new_news_events": news,
            "new_relation_observations": rel,
            "new_sec_signal_events": sec,
            "total_events": news + rel + sec,
            "memory_update_required": int(
                news + rel + sec > 0
            ),
        })

        current += timedelta(days=1)

    dest = OUT / (
        f"replay_{start:%Y%m%d}_"
        f"{end:%Y%m%d}.csv"
    )

    write_csv(
        dest,
        output,
        list(output[0])
    )

    print("=" * 65)
    print("DAILY REPLAY MANIFEST")
    print("=" * 65)
    print("Start:", start)
    print("End:", end)
    print("Days:", len(output))
    print(
        "Days with events:",
        sum(
            x["memory_update_required"]
            for x in output
        )
    )
    print(
        "Total events:",
        sum(
            x["total_events"]
            for x in output
        )
    )
    print("Output:", dest)
    print("REPLAY MANIFEST PASS")
    print(
        "NOTE: This is an event schedule, "
        "not neural-network memory replay."
    )


# =====================================================
# CLI
# =====================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "build",
            "asof",
            "replay"
        ]
    )

    parser.add_argument(
        "--date",
        default="2022-06-15"
    )

    parser.add_argument(
        "--window-days",
        type=int,
        default=30
    )

    parser.add_argument(
        "--out",
        default=None
    )

    parser.add_argument(
        "--start",
        default="2018-01-01"
    )

    parser.add_argument(
        "--end",
        default="2023-12-31"
    )

    args = parser.parse_args()

    if args.command == "build":
        build()

    elif args.command == "asof":
        asof(
            args.date,
            args.window_days,
            args.out
        )

    elif args.command == "replay":
        replay(
            args.start,
            args.end
        )


if __name__ == "__main__":
    main()
