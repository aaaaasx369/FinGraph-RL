#!/usr/bin/env python3

import argparse
import csv
import json
from datetime import date
from pathlib import Path

import torch
import dgl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/unified_temporal_kg_v3"

entity2id = json.loads(
    (BASE / "entity2id.json").read_text()
)
relation2id = json.loads(
    (BASE / "relation2id.json").read_text()
)


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_graph(folder, target_date):
    folder = Path(folder)

    recent = load_csv(
        folder / "recent_event_edges.csv"
    )

    state = load_csv(
        folder / "persistent_state.csv"
    )

    target = date.fromisoformat(target_date)

    records = []
    event_ids = set()
    state_keys = set()

    # 1. Recent events
    for r in recent:
        assert r["event_id"] not in event_ids
        event_ids.add(r["event_id"])

        assert date.fromisoformat(
            r["available_at"][:10]
        ) <= target

        records.append({
            "head_id": int(r["head_id"]),
            "relation_id": int(r["relation_id"]),
            "tail_id": int(r["tail_id"]),
            "role": 0,
            "age_days": (
                target -
                date.fromisoformat(
                    r["available_at"][:10]
                )
            ).days,
        })

    # 2. Last-observed persistent relations
    for r in state:
        key = (
            r["head_id"],
            r["relation_id"],
            r["tail_id"],
        )

        assert key not in state_keys
        state_keys.add(key)

        assert date.fromisoformat(
            r["last_observed_at"]
        ) <= target

        # Keep the state edge separate from an event edge,
        # even if the latest observation is inside the window.
        records.append({
            "head_id": int(r["head_id"]),
            "relation_id": int(r["relation_id"]),
            "tail_id": int(r["tail_id"]),
            "role": 1,
            "age_days": int(r["age_days"]),
        })

    n_total = len(entity2id)
    r_total = len(relation2id)

    # Compact batch graph
    global_nodes = sorted({
        node
        for r in records
        for node in (r["head_id"], r["tail_id"])
    })

    local = {
        node: i
        for i, node in enumerate(global_nodes)
    }

    src = torch.tensor(
        [local[r["head_id"]] for r in records],
        dtype=torch.int64
    )

    dst = torch.tensor(
        [local[r["tail_id"]] for r in records],
        dtype=torch.int64
    )

    g = dgl.graph(
        (src, dst),
        num_nodes=len(global_nodes)
    )

    g.ndata[dgl.NID] = torch.tensor(
        global_nodes,
        dtype=torch.int64
    )

    g.ndata["node_type"] = torch.zeros(
        len(global_nodes),
        dtype=torch.int64
    )

    g.edata["rel_type"] = torch.tensor(
        [r["relation_id"] for r in records],
        dtype=torch.int64
    )

    g.edata["edge_role"] = torch.tensor(
        [r["role"] for r in records],
        dtype=torch.int64
    )

    g.edata["age_days"] = torch.tensor(
        [r["age_days"] for r in records],
        dtype=torch.float32
    )

    g.num_relations = r_total
    g.num_all_nodes = n_total

    assert g.num_edges() == len(records)
    assert all(
        0 <= r["head_id"] < n_total
        and 0 <= r["tail_id"] < n_total
        and 0 <= r["relation_id"] < r_total
        and r["age_days"] >= 0
        for r in records
    )

    return g, len(recent), len(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--folder", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    graph, n_event, n_state = build_graph(
        args.folder,
        args.date
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dgl.save_graphs(
        str(out / "asof_graph.bin"),
        [graph]
    )

    metadata = {
        "asof": args.date,
        "num_nodes": graph.num_nodes(),
        "num_edges": graph.num_edges(),
        "event_edges": n_event,
        "state_edges": n_state,
        "global_entities": len(entity2id),
        "global_relations": len(relation2id),
        "edge_role": {
            "0": "RECENT_EVENT",
            "1": "LAST_OBSERVED_STATE"
        },
        "pit_certified": False,
        "state_validity_verified": False,
        "model_ready": False,
    }

    (out / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    print("=" * 60)
    print("AS-OF DGL GRAPH")
    print("=" * 60)
    print(json.dumps(metadata, indent=2))
    print("OUTPUT:", out)
    print("GRAPH BUILD PASS")


if __name__ == "__main__":
    main()
