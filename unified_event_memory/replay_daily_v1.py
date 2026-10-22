#!/usr/bin/env python3

import csv
from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/unified_event_memory_v2"
OUT = ROOT / "outputs/unified_daily_replay_v1"
OUT.mkdir(parents=True, exist_ok=True)

START = date(2018, 1, 1)
END = date(2023, 12, 31)

def load(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

edges = load(SRC / "unified_edges.csv")
signals = load(SRC / "sec_numeric_signals.csv")

daily_events = defaultdict(list)
daily_signals = defaultdict(list)

for r in edges:
    d = date.fromisoformat(r["available_at"][:10])
    daily_events[d].append(r)

for r in signals:
    d = date.fromisoformat(r["available_at"][:10])
    daily_signals[d].append(r)

# These are observations, not verified validity intervals.
persistent_candidates = {}
latest_sec_signals = {}

summary = []
current = START

while current <= END:
    new_edges = daily_events.get(current, [])
    new_signals = daily_signals.get(current, [])

    news_count = 0
    relation_count = 0

    for e in new_edges:
        if e["source"] == "FINDKG":
            news_count += 1

        elif e["source"] == "SEC_RELATION":
            relation_count += 1

            key = (
                e["head"],
                e["relation"],
                e["tail"],
            )

            # Latest supporting observation.
            # Does not prove the relationship is still active.
            persistent_candidates[key] = e

    for s in new_signals:
        latest_sec_signals[s["head"]] = s

    summary.append({
        "date": current.isoformat(),
        "new_news_triples": news_count,
        "new_sec_relations": relation_count,
        "new_sec_signals": len(new_signals),
        "persistent_relation_candidates": len(
            persistent_candidates
        ),
        "companies_with_sec_signal": len(
            latest_sec_signals
        ),
        "has_new_events": int(bool(
            new_edges or new_signals
        )),
    })

    current += timedelta(days=1)

path = OUT / "daily_replay_summary.csv"

with path.open(
    "w", encoding="utf-8", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0].keys())
    )
    writer.writeheader()
    writer.writerows(summary)

print("=" * 65)
print("DAILY EVENT REPLAY V1")
print("=" * 65)

print("Trading-calendar-independent daily replay")
print("Period:", START, "to", END)
print("Total days:", len(summary))
print("Days with new events:", sum(
    int(r["has_new_events"]) for r in summary
))
print("Final persistent relation candidates:",
      len(persistent_candidates))
print("Companies with latest SEC signal:",
      len(latest_sec_signals))

print("\nYearly event statistics:")

for year in range(2018, 2024):
    subset = [
        r for r in summary
        if r["date"].startswith(str(year))
    ]

    print(
        year,
        "News:", sum(
            r["new_news_triples"] for r in subset
        ),
        "Relations:", sum(
            r["new_sec_relations"] for r in subset
        ),
        "SEC signals:", sum(
            r["new_sec_signals"] for r in subset
        ),
    )

print("\nOutput:", path)
print("NOTE: No KGTransformer inference performed.")
print("NOTE: News weekly timestamp PIT still unverified.")
print("NOTE: Persistent relations are candidates only.")
