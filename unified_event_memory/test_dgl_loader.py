#!/usr/bin/env python3

import json
from pathlib import Path
from collections import Counter

import torch

try:
    import dgl
except ImportError as e:
    raise SystemExit(
        "DGL unavailable in current Python environment: " + str(e)
    )

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/unified_daily_graph_v1"

entity2id = json.loads(
    (BASE / "entity2id.json").read_text()
)
relation2id = json.loads(
    (BASE / "relation2id.json").read_text()
)

NUM_NODES = len(entity2id)
NUM_RELS = len(relation2id)

print("=" * 65)
print("1. VOCABULARY VALIDATION")
print("=" * 65)

assert sorted(entity2id.values()) == list(range(NUM_NODES))
assert sorted(relation2id.values()) == list(range(NUM_RELS))

print("Entities:", NUM_NODES)
print("Relations:", NUM_RELS)
print("Contiguous IDs: PASS")


def build_dgl_batch(events):
    """
    Create compact DGL event graph with global node mapping.
    Does not implement persistent state or model inference.
    """
    heads = [int(e["head_id"]) for e in events]
    tails = [int(e["tail_id"]) for e in events]
    rels = [int(e["relation_id"]) for e in events]

    assert all(0 <= x < NUM_NODES for x in heads + tails)
    assert all(0 <= x < NUM_RELS for x in rels)

    global_nodes = sorted(set(heads + tails))
    local_id = {
        node: i for i, node in enumerate(global_nodes)
    }

    src = torch.tensor(
        [local_id[h] for h in heads],
        dtype=torch.int64
    )
    dst = torch.tensor(
        [local_id[t] for t in tails],
        dtype=torch.int64
    )

    g = dgl.graph(
        (src, dst),
        num_nodes=len(global_nodes)
    )

    # Global entity IDs needed by the original updater.
    g.ndata[dgl.NID] = torch.tensor(
        global_nodes, dtype=torch.int64
    )

    # Temporary node type.
    # Actual node types must be matched to the model config.
    g.ndata["node_type"] = torch.zeros(
        len(global_nodes), dtype=torch.int64
    )

    g.edata["rel_type"] = torch.tensor(
        rels, dtype=torch.int64
    )

    g.edata["occurrence_count"] = torch.tensor(
        [int(e["occurrence_count"]) for e in events],
        dtype=torch.int64
    )

    # These custom properties are used in the original pipeline.
    g.num_relations = NUM_RELS
    g.num_all_nodes = NUM_NODES

    return g


print("\n" + "=" * 65)
print("2. SELECT SMALL EVENT BATCH")
print("=" * 65)

selected = None
total_batches = 0
total_events = 0
sources = Counter()

with (BASE / "event_batches.jsonl").open() as f:
    for line in f:
        record = json.loads(line)
        events = record["events"]

        total_batches += 1
        total_events += len(events)

        sources.update(e["source"] for e in events)

        # Prefer a small day containing both news and SEC edges.
        source_set = {e["source"] for e in events}

        if (
            selected is None
            and len(events) <= 1000
            and len(events) > 0
        ):
            selected = record

        if (
            len(events) <= 1000
            and {"FINDKG", "SEC_RELATION"} <= source_set
        ):
            selected = record
            break

assert selected is not None, "No eligible event batch"

print("Selected date:", selected["date"])
print("Selected events:", len(selected["events"]))

print("\n" + "=" * 65)
print("3. BUILD DGL GRAPH")
print("=" * 65)

g = build_dgl_batch(selected["events"])

print("DGL nodes:", g.num_nodes())
print("DGL edges:", g.num_edges())
print("Global node count:", g.num_all_nodes)
print("Relation count:", g.num_relations)
print("Node data:", list(g.ndata.keys()))
print("Edge data:", list(g.edata.keys()))

assert g.num_edges() == len(selected["events"])
assert g.num_all_nodes == NUM_NODES
assert len(g.ndata[dgl.NID]) == g.num_nodes()

print("\nDGL GRAPH SMOKE TEST: PASS")
print("No KGTransformer inference was performed.")
print("Timestamp conversion still needs model compatibility checks.")
