#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path

DEFAULT_MAPPING = (
    "external/FinDKG-FinRL-Train2020/data/FinDKG"
)
FOCAL_IDS = {
    "AAPL": 9,
    "GOOGL": 11,
    "AMZN": 20,
    "MSFT": 30,
    "NVDA": 221,
}

def load_mapping(path, kind):
    result = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            result[idx] = parts[0]
    if not result:
        raise RuntimeError(f"Empty {kind} mapping: {path}")
    return result

def numeric_id(value):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None

def resolve_edge(item, entity_map, relation_map):
    raw = item.get("record")
    if not isinstance(raw, dict):
        return None, "missing_record"

    s = numeric_id(raw.get("subject_id", raw.get("subject")))
    r = numeric_id(raw.get("relation_id", raw.get("relation")))
    o = numeric_id(raw.get("object_id", raw.get("object")))

    if None in (s, r, o):
        return None, "invalid_numeric_id"

    if s not in entity_map or o not in entity_map:
        return None, "unknown_entity_id"
    if r not in relation_map:
        return None, "unknown_relation_id"

    edge = {
        "evidence_id": item.get("evidence_id"),
        "source": "FinDKG",
        "subject_id": s,
        "subject_name": entity_map[s],
        "relation_id": r,
        "relation_name": relation_map[r],
        "object_id": o,
        "object_name": entity_map[o],
        "observed_at": item.get("observed_at"),
        "date_semantics": (
            "unverified_dataset_observation_not_relationship_start"
        ),
        "mapping_resolved": True,
        "original_record": raw,
    }
    return edge, None

def hop_distances(edges, focal, max_hops=2):
    """
    Compute undirected relevance distances within the supplied
    graph subset. This does not infer relationship direction.
    """
    adj = defaultdict(set)
    for e in edges:
        s = e["subject_id"]
        o = e["object_id"]
        adj[s].add(o)
        adj[o].add(s)

    distance = {focal: 0}
    q = deque([focal])

    while q:
        node = q.popleft()
        if distance[node] >= max_hops:
            continue
        for neighbor in adj[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                q.append(neighbor)
    return distance

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mapping-dir", default=DEFAULT_MAPPING)
    p.add_argument("--max-hops", type=int, choices=[1, 2], default=2)
    args = p.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    ticker = data["ticker"]
    if ticker not in FOCAL_IDS:
        raise ValueError(f"Unknown ticker mapping: {ticker}")

    root = Path(args.mapping_dir)
    entities = load_mapping(root / "entity2id.txt", "entity")
    relations = load_mapping(root / "relation2id.txt", "relation")

    focal = FOCAL_IDS[ticker]
    if focal not in entities:
        raise RuntimeError(f"Focal entity {focal} missing")

    resolved = []
    errors = []

    for item in data.get("findkg_evidence", []):
        edge, error = resolve_edge(item, entities, relations)
        if error:
            errors.append({
                "evidence_id": item.get("evidence_id"),
                "reason": error,
            })
        else:
            resolved.append(edge)

    distances = hop_distances(
        resolved, focal, max_hops=args.max_hops
    )

    focal_edges = []
    two_hop_edges = []
    background_edges = []

    for edge in resolved:
        s = edge["subject_id"]
        o = edge["object_id"]

        if s == focal or o == focal:
            edge["relevance"] = "FOCAL_1HOP"
            edge["hop_distance"] = 1
            focal_edges.append(edge)
        elif min(
            distances.get(s, 999),
            distances.get(o, 999)
        ) <= 1 and args.max_hops == 2:
            edge["relevance"] = "FOCAL_2HOP"
            edge["hop_distance"] = 2
            two_hop_edges.append(edge)
        else:
            edge["relevance"] = "UNVERIFIED_BACKGROUND"
            edge["hop_distance"] = None
            background_edges.append(edge)

    # Keep the original V6 intact; construct a new copy.
    output = dict(data)
    output["findkg_evidence_original"] = data.get(
        "findkg_evidence", []
    )

    # Only locally relevant graph edges are eligible for
    # downstream Qwen input. Do not add background edges.
    selected = focal_edges + two_hop_edges
    output["findkg_evidence"] = [
        {
            "evidence_id": e["evidence_id"],
            "source": "FinDKG",
            "observed_at": e["observed_at"],
            "record": {
                "subject_id": e["subject_id"],
                "subject": e["subject_name"],
                "relation_id": e["relation_id"],
                "relation": e["relation_name"],
                "object_id": e["object_id"],
                "object": e["object_name"],
                "relevance": e["relevance"],
                "hop_distance": e["hop_distance"],
                "mapping_resolved": True,
            },
        }
        for e in selected
    ]

    output["findkg_resolution_audit"] = {
        "focal_ticker": ticker,
        "focal_entity_id": focal,
        "focal_entity_name": entities[focal],
        "input_edges": len(data.get("findkg_evidence", [])),
        "resolved_edges": len(resolved),
        "focal_1hop": len(focal_edges),
        "focal_2hop": len(two_hop_edges),
        "background_excluded": len(background_edges),
        "errors": errors,
        "warning": (
            "Hops are computed only within supplied V6 edges. "
            "A zero count does not prove that the full FinDKG "
            "has no relevant relations."
        ),
    }
    output["counts"] = dict(data.get("counts", {}))
    output["counts"]["findkg"] = len(selected)

    # Original financial features and citations were generated
    # from the unmodified V6 evidence and are no longer valid
    # evaluations of the resolved evidence universe.
    output["previous_financial_features"] = data.get(
        "financial_features"
    )
    output["financial_features"] = None
    output["schema_version"] = "late_fusion_v6_resolved_findkg"
    output["warnings"] = list(data.get("warnings", [])) + [
        "Original Qwen features archived, not reused as validated outputs.",
        "FinDKG graph distances calculated from V6 subset only.",
        "Dataset observation timestamps need availability verification.",
    ]

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit = output["findkg_resolution_audit"]
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print("\nSelected relations:")
    for e in selected[:20]:
        print(
            e["subject_name"],
            "--", e["relation_name"], "-->",
            e["object_name"],
            "[" + e["relevance"] + "]",
        )
    print("\nSaved:", dest)

if __name__ == "__main__":
    main()
