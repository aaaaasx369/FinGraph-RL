#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

DEFAULT_DATES = [
    "2018-01-08",
    "2018-07-02",
    "2019-06-03",
    "2020-03-02",
    "2020-06-01",
    "2020-12-01",
    "2021-06-01",
    "2021-12-30",
]


def log(*args):
    print(*args, flush=True)


def choose_column(df, candidates, field):
    lookup = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]

    raise RuntimeError(
        f"Missing {field}. Available columns: {list(df.columns)}"
    )


def load_graph(path):
    log("[LOAD]", path)

    df = pd.read_csv(path, low_memory=False)

    head = choose_column(
        df,
        [
            "head", "head_id", "source", "src",
            "subject", "entity1", "source_entity",
        ],
        "head",
    )

    relation = choose_column(
        df,
        [
            "relation", "rel", "predicate",
            "relation_type", "edge_type",
        ],
        "relation",
    )

    tail = choose_column(
        df,
        [
            "tail", "tail_id", "target", "dst",
            "object", "entity2", "target_entity",
        ],
        "tail",
    )

    date = choose_column(
        df,
        [
            "date", "event_date", "timestamp",
            "observed_date", "snapshot_date",
            "as_of_date", "time", "effective_date",
        ],
        "date",
    )

    log("[ORIGINAL ROWS]", len(df))

    df = df[[head, relation, tail, date]].copy()
    df.columns = [
        "head", "relation", "tail", "date"
    ]

    for col in ["head", "relation", "tail"]:
        df[col] = df[col].astype("string").str.strip()

    df["date"] = (
        pd.to_datetime(
            df["date"], utc=True, errors="coerce"
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    valid = (
        df["date"].notna()
        & df["head"].notna()
        & df["tail"].notna()
        & df["relation"].notna()
        & (df["head"].str.len() > 0)
        & (df["tail"].str.len() > 0)
        & (df["relation"].str.len() > 0)
    )

    invalid_rows = int((~valid).sum())

    if invalid_rows:
        raise RuntimeError(
            f"{invalid_rows} invalid KG rows. "
            "Inspect source before continuing."
        )

    # Remove repeated snapshots of the same
    # directed triple. Preserve its FIRST observed
    # date; do not treat later snapshots as new edges.
    unique = (
        df.groupby(
            ["head", "relation", "tail"],
            as_index=False,
        )["date"]
        .min()
        .rename(columns={"date": "first_observed"})
    )

    unique = unique.sort_values(
        ["first_observed", "head", "relation", "tail"]
    ).reset_index(drop=True)

    log("[UNIQUE DIRECTED TRIPLES]", len(unique))
    log("[RELATION TYPES]", unique["relation"].nunique())
    log("[FIRST OBSERVED MIN]", unique["first_observed"].min())
    log("[FIRST OBSERVED MAX]", unique["first_observed"].max())

    return unique, len(df)


def build_adjacency(active):
    """
    Preserve original direction:
       outgoing traversal: head -> tail
       incoming traversal: tail -> head

    Traversing incoming edges is allowed for
    neighborhood discovery, but never silently
    reverses the original semantic relation.
    """
    adjacency = defaultdict(list)

    for r in active.itertuples(index=False):
        h = str(r.head)
        t = str(r.tail)
        rel = str(r.relation)
        observed = pd.Timestamp(r.first_observed)

        adjacency[h].append(
            (t, rel, "out", observed, h, t)
        )

        adjacency[t].append(
            (h, rel, "in", observed, h, t)
        )

    # Deterministic priority: recently first-observed
    # edges before older edges.
    for node in adjacency:
        adjacency[node].sort(
            key=lambda e: (
                -e[3].value,
                e[1],
                e[0],
                e[2],
            )
        )

    return adjacency


def audit_ticker(
    ticker,
    day,
    adjacency,
    max_degree,
    max_paths,
):
    root = f"COMPANY:{ticker}"

    if root not in adjacency:
        return {
            "ticker": ticker,
            "date": str(day.date()),
            "root_found": False,
            "hop1_paths": 0,
            "hop2_paths": 0,
            "hop3_paths": 0,
            "unique_reached_nodes": 0,
            "relations_reached": 0,
            "max_original_degree": 0,
            "edges_cut_by_degree_limit": 0,
            "path_budget_exhausted": False,
        }, [], set()

    counts = Counter()
    reached = set()
    relation_set = set()
    examples = []

    cut_edges = 0
    max_original_degree = 0
    exhausted = False
    total_paths = 0

    # DFS simple paths. No repeated nodes along
    # one candidate path, preventing trivial cycles.
    stack = [(root, [root], [])]

    while stack:
        node, nodes, edges = stack.pop()

        if len(edges) >= 3:
            continue

        neighbors = adjacency.get(node, [])

        max_original_degree = max(
            max_original_degree,
            len(neighbors),
        )

        cut_edges += max(
            0,
            len(neighbors) - max_degree,
        )

        selected = neighbors[:max_degree]

        for edge in reversed(selected):
            nxt, relation, direction, observed, h, t = edge

            if nxt in nodes:
                continue

            if not observed < day:
                raise RuntimeError(
                    "PIT boundary violation: "
                    f"{observed} >= {day}"
                )

            next_nodes = nodes + [nxt]

            edge_info = {
                "original_head": h,
                "relation": relation,
                "original_tail": t,
                "traversal_direction": direction,
                "first_observed": str(
                    observed.date()
                ),
            }

            next_edges = edges + [edge_info]
            hop = len(next_edges)

            counts[hop] += 1
            total_paths += 1

            reached.add(nxt)
            relation_set.add(relation)

            if len(examples) < 30:
                examples.append({
                    "date": str(day.date()),
                    "ticker": ticker,
                    "hop": hop,
                    "nodes": json.dumps(
                        next_nodes,
                        ensure_ascii=False,
                    ),
                    "edges": json.dumps(
                        next_edges,
                        ensure_ascii=False,
                    ),
                })

            if total_paths >= max_paths:
                exhausted = True
                break

            if hop < 3:
                stack.append(
                    (nxt, next_nodes, next_edges)
                )

        if exhausted:
            break

    summary = {
        "ticker": ticker,
        "date": str(day.date()),
        "root_found": True,
        "hop1_paths": counts[1],
        "hop2_paths": counts[2],
        "hop3_paths": counts[3],
        "unique_reached_nodes": len(reached),
        "relations_reached": len(relation_set),
        "max_original_degree": max_original_degree,
        "edges_cut_by_degree_limit": cut_edges,
        "path_budget_exhausted": exhausted,
    }

    return summary, examples, relation_set


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v14a_graph_audit"
        ),
    )

    p.add_argument(
        "--max-degree",
        type=int,
        default=25,
    )

    p.add_argument(
        "--max-paths",
        type=int,
        default=20000,
    )

    p.add_argument(
        "--dates",
        nargs="+",
        default=DEFAULT_DATES,
    )

    args = p.parse_args()

    if args.max_degree < 1 or args.max_paths < 1:
        raise ValueError(
            "max-degree and max-paths must be positive"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph, original_rows = load_graph(args.kg)

    relation_counts = (
        graph.groupby("relation")
        .size()
        .rename("unique_triples")
        .reset_index()
        .sort_values(
            "unique_triples",
            ascending=False,
        )
    )

    relation_counts.to_csv(
        args.out / "relation_inventory.csv",
        index=False,
    )

    dates = pd.to_datetime(
        args.dates,
        errors="raise",
    )

    if any(pd.Timestamp(d).year > 2021 for d in dates):
        raise RuntimeError(
            "Audit dates must be <= 2021"
        )

    if any(pd.Timestamp(d).year < 2018 for d in dates):
        raise RuntimeError(
            "Audit dates must be >= 2018"
        )

    rows = []
    examples = []
    coverage_rows = []

    for day in dates:
        day = pd.Timestamp(day).normalize()

        # Strictly before decision date.
        active = graph.loc[
            graph["first_observed"] < day
        ].copy()

        adjacency = build_adjacency(active)

        log()
        log(
            "[AUDIT DATE]",
            str(day.date()),
            "active unique triples=",
            len(active),
            "nodes=",
            len(adjacency),
        )

        for ticker in TICKERS:
            summary, sample, relation_set = (
                audit_ticker(
                    ticker,
                    day,
                    adjacency,
                    args.max_degree,
                    args.max_paths,
                )
            )

            rows.append(summary)
            examples.extend(sample)

            for rel in sorted(relation_set):
                coverage_rows.append({
                    "date": str(day.date()),
                    "ticker": ticker,
                    "relation": rel,
                })

            log(
                "[TICKER]",
                ticker,
                "1-hop", summary["hop1_paths"],
                "2-hop", summary["hop2_paths"],
                "3-hop", summary["hop3_paths"],
                "relations", summary["relations_reached"],
                "budget_exhausted",
                summary["path_budget_exhausted"],
            )

    audit = pd.DataFrame(rows)

    audit.to_csv(
        args.out / "multihop_summary.csv",
        index=False,
    )

    pd.DataFrame(
        examples,
        columns=[
            "date", "ticker", "hop", "nodes", "edges"
        ],
    ).to_csv(
        args.out / "path_examples.csv",
        index=False,
    )

    pd.DataFrame(
        coverage_rows,
        columns=["date", "ticker", "relation"],
    ).to_csv(
        args.out / "reachable_relation_coverage.csv",
        index=False,
    )

    diagnostics = {
        "original_rows": int(original_rows),
        "unique_directed_triples": int(len(graph)),
        "relation_count": int(graph["relation"].nunique()),
        "audit_dates": [
            str(pd.Timestamp(d).date())
            for d in dates
        ],
        "max_degree": args.max_degree,
        "max_paths": args.max_paths,
        "root_found_fraction": float(
            audit["root_found"].mean()
        ),
        "two_hop_positive_fraction": float(
            (audit["hop2_paths"] > 0).mean()
        ),
        "three_hop_positive_fraction": float(
            (audit["hop3_paths"] > 0).mean()
        ),
        "budget_exhaustion_fraction": float(
            audit["path_budget_exhausted"].mean()
        ),
        "interpretation": (
            "Bounded sampled simple paths, not exact "
            "whole-graph path counts."
        ),
        "pit_definition": (
            "first_observed < decision date; "
            "source publication timing not certified"
        ),
    }

    (
        args.out / "audit_manifest.json"
    ).write_text(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 72)
    log("GRAPH AUDIT SUMMARY")
    log("=" * 72)

    log(
        audit.to_string(index=False)
    )

    log()
    log("[MANIFEST]")
    log(
        json.dumps(
            diagnostics,
            indent=2,
            ensure_ascii=False,
        )
    )

    log()
    log("V14-A GRAPH AUDIT: PASS")


if __name__ == "__main__":
    main()
