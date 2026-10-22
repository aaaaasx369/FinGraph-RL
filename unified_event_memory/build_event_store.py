#!/usr/bin/env python3

import csv
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]

NEWS = ROOT / "kg/data/findkg/findkg_full_edges_with_dates.csv"
SEC = ROOT / "kg/data/edges_v4_full.csv"
SIGNALS = ROOT / "rag/data/features/sec_rag_events_2018_2026.csv"

OUT = ROOT / "outputs/unified_event_memory_v1"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "event_id",
    "source",
    "event_type",
    "head",
    "relation",
    "tail",
    "event_date",
    "available_at",
    "time_precision",
    "persistence",
    "source_id",
    "confidence",
    "impact",
    "risk",
    "evidence",
]

rows = []
counts = Counter()


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            yield row


def next_day(date_str):
    return (
        datetime.strptime(date_str[:10], "%Y-%m-%d")
        + timedelta(days=1)
    ).strftime("%Y-%m-%d")


def make_id(values):
    raw = "|".join(str(v) for v in values)
    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()[:24]


def add(
    source,
    event_type,
    head,
    relation,
    tail,
    event_date,
    available_at,
    time_precision,
    persistence,
    source_id="",
    confidence="",
    impact="",
    risk="",
    evidence="",
):
    if not available_at:
        return

    event_id = make_id([
        source,
        event_type,
        head,
        relation,
        tail,
        available_at,
        source_id,
    ])

    rows.append({
        "event_id": event_id,
        "source": source,
        "event_type": event_type,
        "head": head,
        "relation": relation,
        "tail": tail,
        "event_date": event_date,
        "available_at": available_at,
        "time_precision": time_precision,
        "persistence": persistence,
        "source_id": source_id,
        "confidence": confidence,
        "impact": impact,
        "risk": risk,
        "evidence": evidence[:1000],
    })

    counts[source] += 1


# =================================================
# 1. FinDKG weekly triples
# =================================================

for r in read_csv(NEWS):
    date = r["date"]

    # Important:
    # Original FinDKG identifiers remain namespaced.
    # No assumption that these IDs equal SEC tickers.
    add(
        source="FINDKG",
        event_type="GRAPH_TRIPLE",
        head="FINDKG:" + r["subject_id"],
        relation="FINDKG_REL:" + r["relation_id"],
        tail="FINDKG:" + r["object_id"],
        event_date=date,
        available_at=date,
        time_precision="WEEK",
        persistence="EVENT",
        source_id=(
            "FINDKG_TIME:"
            + r["time_id"]
            + ":"
            + r["original_split"]
        ),
    )


# =================================================
# 2. SEC company relationships
# =================================================

for r in read_csv(SEC):
    if r.get("validation_status") != "ACCEPT":
        continue

    filing_date = r.get("filing_date", "")

    if not filing_date:
        continue

    # Date-only filing timestamp:
    # conservatively allow usage from next calendar day.
    # This is not an intraday PIT guarantee.
    available = next_day(filing_date)

    head = "COMPANY:" + r["source"].upper()
    tail = "COMPANY:" + r["target"].upper()

    add(
        source="SEC_RELATION",
        event_type="RELATION_OBSERVATION",
        head=head,
        relation=r["relation"],
        tail=tail,
        event_date=filing_date,
        available_at=available,
        time_precision="DAY_CONSERVATIVE",
        persistence="PERSISTENT_CANDIDATE",
        source_id=r.get("chunk_id", ""),
        confidence=r.get("confidence", ""),
        evidence=r.get("evidence", ""),
    )


# =================================================
# 3. SEC-RAG numerical signals
# =================================================

for r in read_csv(SIGNALS):
    date = r.get("available_date", "")
    ticker = r.get("ticker", "").upper()

    if not date or not ticker:
        continue

    # SEC-RAG is a numeric signal event,
    # not an invented company-to-company KG edge.
    add(
        source="SEC_RAG",
        event_type="NUMERIC_SIGNAL",
        head="COMPANY:" + ticker,
        relation="HAS_SEC_RAG_SIGNAL",
        tail="",
        event_date=date,
        available_at=date,
        time_precision="DAY_SOURCE_REPORTED",
        persistence="SIGNAL",
        source_id=(
            ticker
            + ":"
            + date
            + ":"
            + r.get("pipeline_version", "")
        ),
        confidence=r.get("confidence", ""),
        impact=r.get("impact", ""),
        risk=r.get("risk", ""),
        evidence=r.get("reason", ""),
    )


# =================================================
# Sort and deduplicate
# =================================================

unique = {}

for r in rows:
    unique[r["event_id"]] = r

rows = sorted(
    unique.values(),
    key=lambda x: (
        x["available_at"],
        x["source"],
        x["event_id"],
    ),
)

path = OUT / "unified_events.csv"

with path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=FIELDS
    )
    writer.writeheader()
    writer.writerows(rows)


print("=" * 65)
print("UNIFIED EVENT STORE V1")
print("=" * 65)

print("OUTPUT:", path)
print("TOTAL UNIQUE EVENTS:", len(rows))

print("\nSOURCE COUNTS")
for source, n in Counter(
    r["source"] for r in rows
).most_common():
    print(f"{source:20s} {n:,}")

print("\nTIME PRECISION")
for precision, n in Counter(
    r["time_precision"] for r in rows
).most_common():
    print(f"{precision:25s} {n:,}")

print("\nDATE RANGE")

if rows:
    print(
        rows[0]["available_at"],
        "→",
        rows[-1]["available_at"]
    )

print("\nSAMPLE EVENTS")

for r in rows[:8]:
    print(
        r["available_at"],
        r["source"],
        r["head"],
        r["relation"],
        r["tail"],
    )

print("\nDONE")
