#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/unified_event_memory_v2"
OUT = ROOT / "outputs/unified_daily_graph_v1"
OUT.mkdir(parents=True, exist_ok=True)

START = date(2018, 1, 1)
END = date(2023, 12, 31)

def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

edges = load_csv(SRC / "unified_edges.csv")
signals = load_csv(SRC / "sec_numeric_signals.csv")

# 1. Shared entity and relation vocabularies
entities = sorted({
    node
    for e in edges
    for node in (e["head"], e["tail"])
    if node
})

relations = sorted({
    e["relation"] for e in edges
})

entity2id = {name: i for i, name in enumerate(entities)}
relation2id = {name: i for i, name in enumerate(relations)}

# 2. Group by source-reported availability date
daily_edges = defaultdict(list)
daily_signals = defaultdict(list)

for e in edges:
    d = date.fromisoformat(e["available_at"][:10])
    daily_edges[d].append(e)

for s in signals:
    d = date.fromisoformat(s["available_at"][:10])
    daily_signals[d].append(s)

# 3. Write vocabularies
with (OUT / "entity2id.json").open("w") as f:
    json.dump(entity2id, f, indent=2)

with (OUT / "relation2id.json").open("w") as f:
    json.dump(relation2id, f, indent=2)

# 4. Chronological graph replay
persistent_candidates = {}
latest_signals = {}
summary = []

with (
    (OUT / "event_batches.jsonl").open("w") as event_file,
    (OUT / "state_updates.jsonl").open("w") as state_file,
    (OUT / "signal_updates.jsonl").open("w") as signal_file
):
    current = START

    while current <= END:
        new_edges = daily_edges.get(current, [])
        new_signals = daily_signals.get(current, [])

        event_batch = []

        for e in new_edges:
            record = {
                "event_id": e["event_id"],
                "head_id": entity2id[e["head"]],
                "relation_id": relation2id[e["relation"]],
                "tail_id": entity2id[e["tail"]],
                "source": e["source"],
                "occurrence_count": int(
                    e["occurrence_count"]
                ),
                "time_precision": e["time_precision"],
                "available_at": e["available_at"],
            }

            event_batch.append(record)

            if e["source"] == "SEC_RELATION":
                key = (
                    e["head"],
                    e["relation"],
                    e["tail"]
                )

                persistent_candidates[key] = {
                    "event_id": e["event_id"],
                    "last_observed": e["available_at"],
                }

                state_file.write(json.dumps({
                    "date": current.isoformat(),
                    "action": "OBSERVE",
                    "head": e["head"],
                    "relation": e["relation"],
                    "tail": e["tail"],
                    "event_id": e["event_id"],
                    "validity_verified": False,
                }) + "\n")

        # Store only non-empty event batches.
        # Historical events are NOT replayed every day.
        if event_batch:
            event_file.write(json.dumps({
                "date": current.isoformat(),
                "events": event_batch,
            }) + "\n")

        for s in new_signals:
            latest_signals[s["head"]] = s

            signal_file.write(json.dumps({
                "date": current.isoformat(),
                "company": s["head"],
                "event_id": s["event_id"],
                "impact": s["impact"],
                "risk": s["risk"],
                "confidence": s["confidence"],
                "time_precision": s["time_precision"],
            }) + "\n")

        summary.append({
            "date": current.isoformat(),
            "new_graph_edges": len(new_edges),
            "new_sec_signals": len(new_signals),
            "observed_relation_keys": len(
                persistent_candidates
            ),
            "companies_with_latest_signal": len(
                latest_signals
            ),
        })

        current += timedelta(days=1)

with (OUT / "daily_manifest.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=summary[0].keys()
    )
    writer.writeheader()
    writer.writerows(summary)

print("=" * 65)
print("UNIFIED DAILY GRAPH BUILDER V1")
print("=" * 65)

print("Entities:", len(entity2id))
print("Relations:", len(relation2id))
print("Graph edges:", len(edges))
print("Numeric signals:", len(signals))
print("Days:", len(summary))

print("\nGraph source counts:")
for k, v in Counter(
    e["source"] for e in edges
).most_common():
    print(f"  {k}: {v}")

print("\nPersistent relation candidates:",
      len(persistent_candidates))

print("\nOutput:", OUT)
print("Files:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name, p.stat().st_size, "bytes")

print("\nIMPORTANT:")
print("This is a graph replay staging format.")
print("No KGTransformer inference has been performed.")
print("Event availability has not been PIT-certified.")
print("Persistent relationship validity is unverified.")
