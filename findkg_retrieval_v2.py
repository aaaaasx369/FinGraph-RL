import csv
import json
import argparse
from collections import Counter

EDGE_PATH = "kg/data/findkg/findkg_full_edges_with_dates.csv"
BASE = "external/FinDKG-FinRL-Train2020/data/FinDKG"


def load_mapping(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mapping[parts[1]] = parts[0]
    return mapping


def retrieve(entity_id, as_of, top_k=10, max_degree=50):
    entities = load_mapping(f"{BASE}/entity2id.txt")
    relations = load_mapping(f"{BASE}/relation2id.txt")

    # Find latest available graph snapshot <= query date.
    dates = set()
    with open(EDGE_PATH, encoding="utf-8-sig") as f:
        for e in csv.DictReader(f):
            if e["date"] <= as_of:
                dates.add(e["date"])

    if not dates:
        raise ValueError("No graph snapshot before query date")

    snapshot = max(dates)

    # Read only the selected weekly snapshot.
    edges = []
    seen = set()
    with open(EDGE_PATH, encoding="utf-8-sig") as f:
        for e in csv.DictReader(f):
            if e["date"] != snapshot:
                continue

            key = (
                e["subject_id"],
                e["relation_id"],
                e["object_id"]
            )
            if key not in seen:
                seen.add(key)
                edges.append(e)

    # Degree within this snapshot, not entire history.
    degree = Counter()
    for e in edges:
        degree[e["subject_id"]] += 1
        degree[e["object_id"]] += 1

    # Find direct Apple relations.
    direct = [
        e for e in edges
        if entity_id in (e["subject_id"], e["object_id"])
    ]

    def neighbor(e):
        if e["subject_id"] == entity_id:
            return e["object_id"]
        return e["subject_id"]

    # Prefer less-connected first-hop neighbors.
    direct.sort(
        key=lambda e: (
            degree[neighbor(e)],
            e["relation_id"],
            e["subject_id"],
            e["object_id"]
        )
    )

    # Keep direct edges, including a few hub connections.
    selected_direct = direct[:top_k]

    # Expand second hop through selected first-hop nodes.
    valid_neighbors = {
        neighbor(e) for e in selected_direct
        if degree[neighbor(e)] <= max_degree
    }

    second = []
    for e in edges:
        s, o = e["subject_id"], e["object_id"]

        if entity_id in (s, o):
            continue

        via = None
        if s in valid_neighbors:
            via = s
            target = o
        elif o in valid_neighbors:
            via = o
            target = s
        else:
            continue

        second.append((
            degree[via] + degree[target],
            via,
            e
        ))

    second.sort(key=lambda x: x[0])
    selected_second = second[:top_k]

    def readable(e, hop, via=None):
        s, r, o = (
            e["subject_id"],
            e["relation_id"],
            e["object_id"]
        )

        return {
            "hop": hop,
            "subject_id": s,
            "subject": entities.get(s, s),
            "relation_id": r,
            "relation": relations.get(r, f"REL_{r}"),
            "object_id": o,
            "object": entities.get(o, o),
            "via": entities.get(via, via) if via else None,
            "date": e["date"]
        }

    result = {
        "entity_id": entity_id,
        "entity": entities.get(entity_id, entity_id),
        "as_of": as_of,
        "snapshot": snapshot,
        "snapshot_edges": len(edges),
        "direct_edges_available": len(direct),
        "one_hop": [
            readable(e, 1) for e in selected_direct
        ],
        "two_hop": [
            readable(e, 2, via)
            for _, via, e in selected_second
        ],
        "warning": (
            "Graph dates are not verified public availability "
            "dates. Snapshot retrieval is not proof of "
            "leakage-free historical evaluation."
        )
    }

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity-id", default="9")
    parser.add_argument("--as-of", default="2021-06-01")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-degree", type=int, default=50)
    args = parser.parse_args()

    result = retrieve(
        args.entity_id,
        args.as_of,
        args.top_k,
        args.max_degree
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))

    with open("findkg_retrieval_v2.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
