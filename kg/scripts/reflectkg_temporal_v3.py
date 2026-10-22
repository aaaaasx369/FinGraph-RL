#!/usr/bin/env python3
"""
Dynamic ReflectKG V3
- Build a historical event store from Validator V3.
- Reconstruct point-in-time relation snapshots.
- Retrieve newly observed / reasserted relations.
- Fuse temporal ReflectKG evidence with existing V5 JSON.
- Never infer relation termination from missing filings.
- Python standard library only.
"""

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


DEFAULT_EVENTS = (
    "outputs/reflectkg_validation_v3/dynamic_events_v3.csv"
)
DEFAULT_STORE = (
    "outputs/reflectkg_temporal_v3/events.json"
)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_date(value):
    return date.fromisoformat(str(value)[:10])


def normalize(value):
    return str(value or "").strip().upper()


def relation_key(row):
    return (
        normalize(row["subject"]),
        normalize(row["relation"]),
        normalize(row["object"]),
    )


def company_matches(row, ticker):
    ticker = normalize(ticker)
    return (
        normalize(row["subject"]) == ticker
        or normalize(row["object"]) == ticker
    )


def load_store(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["events"]


def build(args):
    rows = read_csv(args.events)
    cleaned = []

    for row in rows:
        if row.get("event_type") not in (
            "FIRST_OBSERVED",
            "REASSERTED",
        ):
            continue

        try:
            observed = parse_date(row["published_at"])
        except (ValueError, KeyError):
            continue

        if not row.get("doc_id") or not row.get("evidence"):
            continue

        cleaned.append({
            "event_type": row["event_type"],
            "subject": normalize(row["subject"]),
            "relation": normalize(row["relation"]),
            "object": normalize(row["object"]),
            "published_at": observed.isoformat(),
            "doc_id": row["doc_id"],
            "chunk_id": row.get("chunk_id", ""),
            "evidence": row["evidence"],
            "confidence": row.get("confidence", ""),
            "validation": row.get("validation", "RULE_ACCEPT_V3"),
        })

    # Re-sort and independently rebuild event types.
    cleaned.sort(key=lambda r: (
        r["published_at"],
        r["doc_id"],
        r["chunk_id"],
    ))

    seen_relation = set()
    seen_documents = set()
    events = []

    for row in cleaned:
        key = relation_key(row)
        doc_key = key + (row["doc_id"],)

        if doc_key in seen_documents:
            continue

        seen_documents.add(doc_key)

        row["event_type"] = (
            "REASSERTED"
            if key in seen_relation
            else "FIRST_OBSERVED"
        )
        seen_relation.add(key)

        events.append(row)

    write_json(args.store, {
        "schema_version": "reflectkg_temporal_v3",
        "date_policy": "strictly_before_as_of_date",
        "event_semantics": (
            "First observed or reasserted in available SEC data; "
            "not confirmed real-world creation or termination."
        ),
        "events": events,
    })

    print(json.dumps({
        "store": args.store,
        "events": len(events),
        "unique_relations": len(seen_relation),
        "event_counts": dict(
            Counter(e["event_type"] for e in events)
        ),
        "relations": dict(
            Counter(e["relation"] for e in events)
        ),
    }, indent=2, ensure_ascii=False))


def retrieve(events, ticker, as_of, days=90, hops=1):
    cutoff = parse_date(as_of)
    start = cutoff - timedelta(days=days)

    # Strict cutoff: data published on as_of is excluded.
    eligible = [
        e for e in events
        if parse_date(e["published_at"]) < cutoff
    ]

    eligible.sort(key=lambda e: (
        e["published_at"],
        e["doc_id"],
        e["chunk_id"],
    ))

    # Seed with focal company.
    reachable = {normalize(ticker)}
    selected_keys = set()

    # Undirected traversal only for retrieval.
    # The stored relation direction is not changed.
    for _ in range(hops):
        new_nodes = set(reachable)

        for e in eligible:
            s = normalize(e["subject"])
            o = normalize(e["object"])

            if s in reachable or o in reachable:
                selected_keys.add(relation_key(e))
                new_nodes.update((s, o))

        reachable = new_nodes

    selected_events = [
        e for e in eligible
        if relation_key(e) in selected_keys
    ]

    # Latest supporting evidence per unique relation.
    latest = {}
    history = defaultdict(list)

    for e in selected_events:
        key = relation_key(e)
        latest[key] = e
        history[key].append(e)

    snapshot = []

    for key, e in sorted(latest.items()):
        hist = history[key]

        snapshot.append({
            "subject": e["subject"],
            "relation": e["relation"],
            "object": e["object"],
            "first_observed_at": hist[0]["published_at"],
            "last_observed_at": e["published_at"],
            "observation_count": len(hist),
            "latest_evidence": e["evidence"],
            "latest_doc_id": e["doc_id"],
            "latest_chunk_id": e["chunk_id"],
            "knowledge_status": "previously_observed",
            "real_world_active_status": "unverified",
        })

    changes = [
        e for e in selected_events
        if start <= parse_date(e["published_at"]) < cutoff
    ]

    return {
        "source": "reflectkg_v3",
        "ticker": ticker,
        "as_of": as_of,
        "lookback_days": days,
        "hops": hops,
        "temporal_policy": "published_at < as_of",
        "snapshot_count": len(snapshot),
        "change_count": len(changes),
        "snapshot": snapshot,
        "changes": changes,
        "change_counts": dict(
            Counter(e["event_type"] for e in changes)
        ),
        "relation_counts": dict(
            Counter(e["relation"] for e in snapshot)
        ),
        "limitations": [
            "Snapshot is known relation evidence, not guaranteed currently active relationships.",
            "FIRST_OBSERVED means first in available data.",
            "REASSERTED means another supporting SEC filing.",
            "No termination events inferred from absence.",
            "Date-only cutoff excludes same-day filings.",
            "Rule-based acceptance is not human ground truth.",
        ],
    }


def query(args):
    events = load_store(args.store)

    result = retrieve(
        events,
        ticker=args.ticker,
        as_of=args.as_of,
        days=args.days,
        hops=args.hops,
    )

    if args.out:
        write_json(args.out, result)

    print(json.dumps({
        "ticker": args.ticker,
        "as_of": args.as_of,
        "snapshot_count": result["snapshot_count"],
        "change_count": result["change_count"],
        "relation_counts": result["relation_counts"],
        "change_counts": result["change_counts"],
        "out": args.out,
    }, indent=2, ensure_ascii=False))


def fuse(args):
    # Reuse the existing V5 for SEC and FinDKG.
    # Do not invoke Qwen or rebuild LightRAG.
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    base_path = output.with_suffix(".base_v5.json")

    cmd = [
        sys.executable,
        "late_fusion_v5.py",
        "query",
        "--ticker", args.ticker,
        "--as-of", args.as_of,
        "--days", str(args.days),
        "--out", str(base_path),
    ]

    print("Running existing V5 retrieval...", flush=True)
    subprocess.run(cmd, check=True)

    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)

    events = load_store(args.store)

    temporal = retrieve(
        events,
        ticker=args.ticker,
        as_of=args.as_of,
        days=args.days,
        hops=args.hops,
    )

    # Keep the original ReflectKG output as a baseline.
    base["reflectkg_baseline"] = base.get("reflectkg")

    # New validated temporal branch.
    base["reflectkg_v3"] = temporal

    base["fusion_version"] = "v5_temporal_reflectkg_v3"

    base["evidence_availability"] = {
        "sec": len(
            base.get("sec", {}).get("snapshot", [])
        ) if isinstance(base.get("sec"), dict) else None,
        "findkg": len(
            base.get("findkg", {}).get("snapshot", [])
        ),
        "reflectkg_v3": temporal["snapshot_count"],
        "reflectkg_changes": temporal["change_count"],
    }

    base["temporal_fusion_status"] = (
        "evidence_assembled_not_qwen_inferred"
    )

    base["qwen_output"] = None

    write_json(output, base)

    print(json.dumps({
        "out": str(output),
        "fusion_version": base["fusion_version"],
        "reflectkg_v3_snapshot": temporal["snapshot_count"],
        "reflectkg_v3_changes": temporal["change_count"],
        "reflectkg_v3_relations": temporal["relation_counts"],
        "qwen_called": False,
    }, indent=2, ensure_ascii=False))


def self_test():
    sample = [
        {
            "subject": "TSM",
            "relation": "SUPPLIES_TO",
            "object": "NVDA",
            "published_at": "2020-01-01",
            "event_type": "FIRST_OBSERVED",
            "doc_id": "D1",
            "chunk_id": "C1",
            "evidence": "Supply evidence",
        },
        {
            "subject": "TSM",
            "relation": "SUPPLIES_TO",
            "object": "NVDA",
            "published_at": "2021-01-01",
            "event_type": "REASSERTED",
            "doc_id": "D2",
            "chunk_id": "C2",
            "evidence": "Repeated evidence",
        },
        {
            "subject": "DELL",
            "relation": "CUSTOMER_OF",
            "object": "NVDA",
            "published_at": "2024-01-01",
            "event_type": "FIRST_OBSERVED",
            "doc_id": "D3",
            "chunk_id": "C3",
            "evidence": "Future evidence",
        },
    ]

    r = retrieve(
        sample, "NVDA", "2021-06-01", 365, 1
    )

    assert r["snapshot_count"] == 1
    assert r["change_count"] == 1
    assert r["changes"][0]["event_type"] == "REASSERTED"

    # Exclude same-day and future events.
    r2 = retrieve(
        sample, "NVDA", "2021-01-01", 365, 1
    )
    assert r2["snapshot_count"] == 1
    assert r2["change_count"] == 0

    # A disconnected company must have no edges.
    r3 = retrieve(
        sample, "AAPL", "2021-06-01", 365, 1
    )
    assert r3["snapshot_count"] == 0

    print("SELF TEST PASSED")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)

    b = sub.add_parser("build")
    b.add_argument("--events", default=DEFAULT_EVENTS)
    b.add_argument("--store", default=DEFAULT_STORE)

    for action in ("query", "fuse"):
        q = sub.add_parser(action)
        q.add_argument("--store", default=DEFAULT_STORE)
        q.add_argument("--ticker", default="NVDA")
        q.add_argument("--as-of", default="2021-06-01")
        q.add_argument("--days", type=int, default=90)
        q.add_argument("--hops", type=int, choices=(1, 2), default=1)
        q.add_argument("--out", default="")

    sub.add_parser("self-test")

    args = p.parse_args()

    if args.action == "build":
        build(args)
    elif args.action == "query":
        query(args)
    elif args.action == "fuse":
        if not args.out:
            p.error("fuse requires --out")
        fuse(args)
    elif args.action == "self-test":
        self_test()


if __name__ == "__main__":
    main()
