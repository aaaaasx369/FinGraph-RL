#!/usr/bin/env python3
"""Point-in-time, focal-company FinDKG retrieval.

Dates in FinDKG are dataset observation labels, NOT independently
verified public information availability timestamps.
No lookahead across the as-of date. No model calls.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

PORTFOLIO = {
    "AAPL": 9,
    "GOOGL": 11,
    "AMZN": 20,
    "MSFT": 30,
    "NVDA": 221,
}

def mapping(path):
    output = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            try:
                idx = int(parts[1])
            except ValueError:
                continue
            if idx in output and output[idx] != parts[0]:
                raise ValueError(f"Conflicting ID {idx}")
            output[idx] = parts[0]
    if not output:
        raise ValueError(f"Empty mapping: {path}")
    return output

def parse_day(value):
    return date.fromisoformat(str(value).strip()[:10])

def read_edges(path, as_of):
    eligible = []
    counters = Counter()
    cutoff = parse_day(as_of)

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "subject_id", "relation_id",
            "object_id", "date"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing columns {required - set(reader.fieldnames or [])}"
            )
        for row in reader:
            counters["input_rows"] += 1
            try:
                observed = parse_day(row["date"])
                s = int(row["subject_id"])
                r = int(row["relation_id"])
                o = int(row["object_id"])
            except (ValueError, TypeError):
                counters["invalid_rows"] += 1
                continue

            # Daily conservative policy: exclude same-day observations.
            if observed >= cutoff:
                counters["future_or_same_day"] += 1
                continue

            eligible.append({
                "subject_id": s,
                "relation_id": r,
                "object_id": o,
                "observed_at": observed.isoformat(),
                "time_id": row.get("time_id", ""),
                "original_split": row.get("original_split", ""),
                "finrl_split": row.get("finrl_split", ""),
            })
            counters["historical_rows"] += 1
    return eligible, counters

def retrieve(edges, focal, hops, max_edges):
    # Edge is directional; neighborhood search is undirected.
    # Focal hop-1 edges are always prioritized over hop-2.
    direct = [
        e for e in edges
        if focal in (e["subject_id"], e["object_id"])
    ]
    neighbors = {focal}
    for e in direct:
        neighbors.update([e["subject_id"], e["object_id"]])

    second = []
    if hops == 2:
        second = [
            e for e in edges
            if focal not in (e["subject_id"], e["object_id"])
            and (
                e["subject_id"] in neighbors
                or e["object_id"] in neighbors
            )
        ]

    # Prefer recent observations and dedupe identical
    # triples within the selected historical window.
    def dedupe(rows):
        newest = {}
        for e in rows:
            key = (
                e["subject_id"],
                e["relation_id"],
                e["object_id"],
            )
            if (
                key not in newest
                or e["observed_at"] > newest[key]["observed_at"]
            ):
                newest[key] = e
        return sorted(
            newest.values(),
            key=lambda x: (
                x["observed_at"],
                x["subject_id"],
                x["relation_id"],
                x["object_id"],
            ),
            reverse=True,
        )

    d1 = dedupe(direct)
    d2 = dedupe(second)

    selected = (
        [(1, e) for e in d1]
        + [(2, e) for e in d2]
    )[:max_edges]

    return selected, {
        "historical_direct_rows": len(direct),
        "historical_second_hop_rows": len(second),
        "unique_direct_triples": len(d1),
        "unique_second_hop_triples": len(d2),
        "selected_direct": sum(h == 1 for h, _ in selected),
        "selected_second_hop": sum(h == 2 for h, _ in selected),
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="NVDA",
                   choices=sorted(PORTFOLIO))
    p.add_argument("--as-of", default="2021-06-01")
    p.add_argument(
        "--graph",
        default="kg/data/findkg/findkg_full_edges_with_dates.csv"
    )
    p.add_argument(
        "--mapping-dir",
        default="external/FinDKG-FinRL-Train2020/data/FinDKG"
    )
    p.add_argument("--hops", type=int, choices=[1, 2], default=2)
    p.add_argument("--max-edges", type=int, default=20)
    p.add_argument("--out",
                   default="outputs/findkg_focal_v2/nvda_20210601.json")
    args = p.parse_args()

    if args.max_edges < 1:
        p.error("--max-edges must be positive")

    entities = mapping(
        Path(args.mapping_dir) / "entity2id.txt"
    )
    relations = mapping(
        Path(args.mapping_dir) / "relation2id.txt"
    )
    focal = PORTFOLIO[args.ticker]
    if focal not in entities:
        raise RuntimeError(f"Missing focal ID {focal}")

    edges, counts = read_edges(args.graph, args.as_of)
    selected, hop_counts = retrieve(
        edges, focal, args.hops, args.max_edges
    )

    evidence = []
    for hop, edge in selected:
        s, r, o = (
            edge["subject_id"],
            edge["relation_id"],
            edge["object_id"],
        )
        # Do not make up names for unresolved nodes/relations.
        if s not in entities or o not in entities or r not in relations:
            raise RuntimeError(
                f"Unmapped triple: {(s, r, o)}"
            )

        evidence.append({
            "evidence_id": f"FIN_{len(evidence)+1:03d}",
            "source": "FinDKG",
            "observed_at": edge["observed_at"],
            "record": {
                "subject": entities[s],
                "relation": relations[r],
                "object": entities[o],
                "subject_id": s,
                "relation_id": r,
                "object_id": o,
                "hop_distance": hop,
                "focal_entity_id": focal,
                "mapping_resolved": True,
                "original_split": edge["original_split"],
                "finrl_split": edge["finrl_split"],
                "time_id": edge["time_id"],
            },
        })

    result = {
        "ticker": args.ticker,
        "focal_entity_id": focal,
        "focal_entity_name": entities[focal],
        "as_of": args.as_of,
        "graph_source": args.graph,
        "date_policy": "observed_at_strictly_before_as_of",
        "evidence": evidence,
        "counts": {**counts, **hop_counts},
        "limitations": [
            "FinDKG dates are dataset observation labels.",
            "Public availability dates have not been verified.",
            "Graph hops are relevance distances, not causal paths.",
            "Triple deduplication keeps latest observation only.",
            "Dataset split provenance must be audited for backtests.",
        ],
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps({
        "ticker": args.ticker,
        "focal": entities[focal],
        "as_of": args.as_of,
        "counts": result["counts"],
        "selected": len(evidence),
    }, ensure_ascii=False, indent=2))

    print("\n=== FOCAL EDGES ===")
    for item in evidence:
        r = item["record"]
        print(
            item["observed_at"],
            f"{r['subject']} --{r['relation']}--> {r['object']}",
            f"[hop={r['hop_distance']}]"
        )
    print("\nSaved:", path)

if __name__ == "__main__":
    main()
