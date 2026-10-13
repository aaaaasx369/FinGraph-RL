#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]

EVENTS = ROOT / "outputs/unified_event_memory_v1/unified_events.csv"
NEWS = ROOT / "kg/data/findkg/findkg_full_edges_with_dates.csv"
SEC = ROOT / "kg/data/edges_v4_full.csv"
RAG = ROOT / "rag/data/features/sec_rag_events_2018_2026.csv"

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

events = read_csv(EVENTS)
news = read_csv(NEWS)
sec = read_csv(SEC)
rag = read_csv(RAG)

print("=" * 70)
print("1. SOURCE COUNT VERIFICATION")
print("=" * 70)

counts = Counter(r["source"] for r in events)

expected = {
    "FINDKG": len(news),
    "SEC_RELATION": sum(
        r.get("validation_status") == "ACCEPT"
        and bool(r.get("filing_date"))
        for r in sec
    ),
    "SEC_RAG": sum(
        bool(r.get("available_date"))
        and bool(r.get("ticker"))
        for r in rag
    ),
}

for source, n in expected.items():
    actual = counts[source]
    print(
        f"{source:15} expected={n:8,} "
        f"actual={actual:8,} "
        f"match={n == actual}"
    )

print("\nTOTAL:", len(events))
print("UNIQUE EVENT IDS:", len(set(r["event_id"] for r in events)))

print("\n" + "=" * 70)
print("2. TEMPORAL DISTRIBUTION")
print("=" * 70)

for source in expected:
    subset = [
        r for r in events
        if r["source"] == source
    ]

    by_year = Counter(
        r["available_at"][:4]
        for r in subset
    )

    print(f"\n{source}")
    for year, count in sorted(by_year.items()):
        print(f"  {year}: {count:,}")

print("\n" + "=" * 70)
print("3. FINDKG TIME GRANULARITY")
print("=" * 70)

dates = sorted(set(
    r["date"] for r in news if r["date"]
))

time_ids = set(r["time_id"] for r in news)

print("UNIQUE DATES:", len(dates))
print("UNIQUE TIME IDS:", len(time_ids))

weekdays = Counter(
    datetime.strptime(d, "%Y-%m-%d").strftime("%A")
    for d in dates
)

print("WEEKDAY DISTRIBUTION:", dict(weekdays))

gaps = Counter(
    (datetime.strptime(b, "%Y-%m-%d") -
     datetime.strptime(a, "%Y-%m-%d")).days
    for a, b in zip(dates, dates[1:])
)

print("DATE GAPS:", dict(gaps.most_common(10)))

time_to_dates = defaultdict(set)
for r in news:
    time_to_dates[r["time_id"]].add(r["date"])

conflicts = [
    k for k, v in time_to_dates.items()
    if len(v) > 1
]

print("TIME IDS WITH MULTIPLE DATES:", len(conflicts))

print("\n" + "=" * 70)
print("4. SEC RELATION QUALITY")
print("=" * 70)

relations = [
    r for r in events
    if r["source"] == "SEC_RELATION"
]

print("RELATION TYPES:")
for k, v in Counter(
    r["relation"] for r in relations
).most_common():
    print(f"  {k}: {v}")

print("\nCOMPANIES:")
for k, v in Counter(
    r["head"] for r in relations
).most_common(15):
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("5. SEC RAG QUALITY")
print("=" * 70)

rag_rows = [
    r for r in events
    if r["source"] == "SEC_RAG"
]

for k, v in Counter(
    r["head"] for r in rag_rows
).most_common():
    print(f"{k}: {v}")

print("\nORIGINAL RAG STATUS:")
for k, v in Counter(
    r.get("status", "") for r in rag
).most_common():
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("6. ENTITY ALIGNMENT")
print("=" * 70)

findkg_nodes = set()
company_nodes = set()

for r in events:
    if r["source"] == "FINDKG":
        findkg_nodes.add(r["head"])
        findkg_nodes.add(r["tail"])

    elif r["source"] == "SEC_RELATION":
        company_nodes.add(r["head"])
        company_nodes.add(r["tail"])

    elif r["source"] == "SEC_RAG":
        company_nodes.add(r["head"])

print("FINDKG NODES:", len(findkg_nodes))
print("COMPANY NODES:", len(company_nodes))
print("DIRECT ID OVERLAP:", len(findkg_nodes & company_nodes))

print("\n" + "=" * 70)
print("7. BASIC TEMPORAL VALIDATION")
print("=" * 70)

invalid_dates = []
future_order = []

for r in events:
    try:
        available = datetime.fromisoformat(r["available_at"][:10])
        event = datetime.fromisoformat(r["event_date"][:10])

        if available < event:
            future_order.append(r["event_id"])

    except (ValueError, TypeError):
        invalid_dates.append(r["event_id"])

print("INVALID DATES:", len(invalid_dates))
print("AVAILABLE BEFORE EVENT DATE:", len(future_order))
print("NOTE: This does not prove strict PIT correctness.")

print("\n" + "=" * 70)
print("8. EMPTY FOUR-RELATION DATASET")
print("=" * 70)

four = ROOT / (
    "outputs/finrl_dkg_original_full/"
    "four_relation_kg/four_relation_events.csv"
)

if four.exists():
    print("FILE SIZE:", four.stat().st_size)
    print("DATA ROWS:", len(read_csv(four)))

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
