#!/usr/bin/env python3

"""
Build/query event + observed persistent state
from Unified KG V3.

No training.
No claims of verified PIT or relation validity.
"""

import argparse
import csv
import json

from collections import Counter
from datetime import date, timedelta
from pathlib import Path


FIELDS = [
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


def read_rows(path):
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        yield from csv.DictReader(f)


def write_rows(path, rows, fieldnames):
    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def stamp(s):
    return date.fromisoformat(s[:10])


def load_config(root):

    src = root / "outputs/unified_temporal_kg_v3"

    if not (src / "unified_temporal_edges.csv").exists():
        raise SystemExit(
            f"Missing V3: {src}"
        )

    dest = root / "outputs/unified_temporal_state_v1"
    dest.mkdir(
        parents=True,
        exist_ok=True
    )

    return src, dest


# ==================================================
# 1. BUILD EVENT STORE + PERSISTENT OBSERVATIONS
# ==================================================

def build(root):

    src, dest = load_config(root)

    edges = list(
        read_rows(
            src / "unified_temporal_edges.csv"
        )
    )

    assert edges, "Empty V3"

    ids = [
        r["event_id"]
        for r in edges
    ]

    assert len(ids) == len(set(ids)), (
        "Duplicate event IDs"
    )

    assert all(
        r["available_at"] and r["source"]
        for r in edges
    )

    edges.sort(
        key=lambda r: (
            r["available_at"],
            r["event_id"],
        )
    )

    observations = [
        r for r in edges
        if r["source"] == "SEC_RELATION"
    ]

    write_rows(
        dest / "event_edges.csv",
        edges,
        FIELDS,
    )

    write_rows(
        dest / "relation_observations.csv",
        observations,
        FIELDS,
    )

    # SEC-RAG node attributes
    sig_src = src / "sec_signal_nodes.csv"

    if not sig_src.exists():
        raise SystemExit(
            "Missing SEC signal node attributes"
        )

    sig = list(
        read_rows(sig_src)
    )

    write_rows(
        dest / "sec_signal_nodes.csv",
        sig,
        list(sig[0]) if sig else [
            "node",
            "company",
            "available_at",
            "impact",
            "risk",
            "confidence",
            "reason",
        ],
    )

    # Build daily source counts
    counts = Counter(
        (
            r["available_at"][:10],
            r["source"]
        )
        for r in edges
    )

    begin = min(
        stamp(r["available_at"])
        for r in edges
    )

    end = max(
        stamp(r["available_at"])
        for r in edges
    )

    daily = []
    current = begin

    while current <= end:

        d = current.isoformat()

        daily.append({
            "date": d,
            "news": counts[d, "FINDKG"],
            "sec_relations": counts[d, "SEC_RELATION"],
            "sec_signals": counts[d, "SEC_RAG"],
        })

        current += timedelta(days=1)

    write_rows(
        dest / "daily_event_manifest.csv",
        daily,
        [
            "date",
            "news",
            "sec_relations",
            "sec_signals",
        ],
    )

    metadata = {
        "input": "unified_temporal_kg_v3",
        "edges": len(edges),
        "source_counts": dict(
            Counter(
                r["source"]
                for r in edges
            )
        ),
        "relation_observations": len(
            observations
        ),
        "signal_node_attributes": len(sig),
        "date_range": [
            begin.isoformat(),
            end.isoformat(),
        ],
        "pit_certified": False,
        "relation_validity_verified": False,
        "weekly_news_label_semantics": "unverified",
    }

    (
        dest / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8",
    )

    print(
        "BUILD PASS",
        json.dumps(
            metadata,
            indent=2
        )
    )

    print("Output:", dest)


# ==================================================
# 2. QUERY AS-OF GRAPH
# ==================================================

def query(root, asof, window, out=None):

    src, dest = load_config(root)

    if not (
        dest / "event_edges.csv"
    ).exists():
        raise SystemExit(
            "Run build first"
        )

    target = stamp(asof)

    lower = target - timedelta(
        days=window - 1
    )

    latest = {}
    recent = []
    latest_signals = {}

    signal_attributes = {
        r["node"]: r
        for r in read_rows(
            dest / "sec_signal_nodes.csv"
        )
    }

    seen_ids = set()

    for r in read_rows(
        dest / "event_edges.csv"
    ):

        when = stamp(
            r["available_at"]
        )

        if when > target:
            continue

        eid = r["event_id"]

        if eid in seen_ids:
            raise AssertionError(
                "Repeated event during query"
            )

        seen_ids.add(eid)

        # Recent event graph
        if lower <= when:
            recent.append(r)

        # Persistent SEC relationship observations
        if r["source"] == "SEC_RELATION":

            key = (
                r["head"],
                r["relation"],
                r["tail"],
            )

            if key not in latest:
                latest[key] = {
                    "first": when,
                    "last": when,
                    "n": 0,
                    "row": r,
                }

            s = latest[key]
            s["n"] += 1

            if when >= s["last"]:
                s["last"] = when
                s["row"] = r

        # SEC-RAG latest signal
        elif r["source"] == "SEC_RAG":

            attr = signal_attributes.get(
                r["tail"]
            )

            if attr is None:
                raise AssertionError(
                    "Missing signal node attributes: "
                    + r["tail"]
                )

            key = r["head"]

            if (
                key not in latest_signals
                or (when, eid) >
                latest_signals[key][0]
            ):
                latest_signals[key] = (
                    (when, eid),
                    attr,
                )

    # Build persistent state
    state = []

    for (h, rel, t), s in sorted(
        latest.items()
    ):

        r = s["row"]

        state.append({
            "head": h,
            "relation": rel,
            "tail": t,
            "head_id": r["head_id"],
            "relation_id": r["relation_id"],
            "tail_id": r["tail_id"],
            "first_observed_at": (
                s["first"].isoformat()
            ),
            "last_observed_at": (
                s["last"].isoformat()
            ),
            "age_days": (
                target - s["last"]
            ).days,
            "observation_count": s["n"],
            "latest_event_id": r["event_id"],
            "latest_source_id": r["source_id"],
            "validity_status": (
                "LAST_OBSERVED_UNVERIFIED"
            ),
        })

    report = {
        "asof": target.isoformat(),
        "recent_event_window_days": window,
        "recent_events": len(recent),
        "recent_by_source": dict(
            Counter(
                r["source"]
                for r in recent
            )
        ),
        "persistent_relation_candidates": len(
            state
        ),
        "companies_with_latest_sec_signal": len(
            latest_signals
        ),
        "pit_certified": False,
    }

    if out:

        folder = Path(out)

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        write_rows(
            folder / "recent_event_edges.csv",
            recent,
            FIELDS,
        )

        write_rows(
            folder / "persistent_state.csv",
            state,
            STATE_FIELDS,
        )

        attrs = [
            dict(
                v[1],
                last_observed_at=(
                    v[0][0].isoformat()
                ),
                age_days=(
                    target - v[0][0]
                ).days,
            )
            for _, v in sorted(
                latest_signals.items()
            )
        ]

        write_rows(
            folder / "latest_sec_signals.csv",
            attrs,
            list(attrs[0]) if attrs else [
                "node",
                "company",
                "available_at",
                "impact",
                "risk",
                "confidence",
                "reason",
                "last_observed_at",
                "age_days",
            ],
        )

        (
            folder / "query_metadata.json"
        ).write_text(
            json.dumps(
                report,
                indent=2
            ),
            encoding="utf-8",
        )

    print(
        "AS-OF RESULT",
        json.dumps(
            report,
            indent=2
        )
    )

    print(
        "No future source-reported dates included. "
        "Strict PIT remains unverified."
    )


# ==================================================
# 3. COMMAND LINE
# ==================================================

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "action",
        choices=["build", "asof"]
    )

    ap.add_argument(
        "--root",
        default="."
    )

    ap.add_argument(
        "--date",
        default="2022-06-15"
    )

    ap.add_argument(
        "--window-days",
        type=int,
        default=30
    )

    ap.add_argument(
        "--out",
        default=None
    )

    a = ap.parse_args()

    if a.window_days < 1:
        ap.error(
            "--window-days must be >= 1"
        )

    root = Path(
        a.root
    ).resolve()

    if a.action == "build":
        build(root)
    else:
        query(
            root,
            a.date,
            a.window_days,
            a.out,
        )
