#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V14-A: Temporal Multi-hop GraphRAG

G0: Market
G1: Market + V13 1-hop KG RAG
G2: G1 + 2-hop path representation
G3: G2 + 3-hop path representation
G4: G3 + Dynamic KGT 200D

Historical data:
  2018-2019: initial training
  2020: monthly expanding walk-forward
  2021: fixed validation
  2022-2023: never evaluated

IMPORTANT:
  KG first-observed timestamps are not certified
  public disclosure timestamps.
  Numbered relations have unknown semantics.
  KGT embeddings are not assumed to share a
  semantic vector space with KG paths.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from unified_event_memory.finrl_x.aegis_rl import (
    run_v8_phase2b as v8b,
    run_v13_unified_sec_kg_rag as v13,
    run_v14_0_ablation as base,
)

TICKERS = list(base.TICKERS)
ARMS = ["G0", "G1", "G2", "G3", "G4"]

# Each extra hop produces:
# 2*R relation-direction frequencies
# 32 path-signature hashing bins
# 10 graph statistics
HASH_DIM = 32
STATS_DIM = 10

COMPARISONS = [
    ("G0", "G1"),
    ("G1", "G2"),
    ("G2", "G3"),
    ("G3", "G4"),
    ("G1", "G3"),
    ("G0", "G3"),
]


def log(*args):
    print(*args, flush=True)


def choose_column(df, names, field):
    lookup = {
        str(c).lower(): c
        for c in df.columns
    }
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise RuntimeError(
        f"Cannot find {field}: {list(df.columns)}"
    )


def load_unique_triples(path):
    df = pd.read_csv(
        path,
        low_memory=False,
    )

    h = choose_column(
        df,
        ["head", "head_id", "source", "src",
         "subject", "entity1", "source_entity"],
        "head",
    )
    r = choose_column(
        df,
        ["relation", "rel", "predicate",
         "relation_type", "edge_type"],
        "relation",
    )
    t = choose_column(
        df,
        ["tail", "tail_id", "target", "dst",
         "object", "entity2", "target_entity"],
        "tail",
    )
    d = choose_column(
        df,
        ["date", "event_date", "timestamp",
         "observed_date", "snapshot_date",
         "as_of_date", "time", "effective_date"],
        "date",
    )

    df = df[[h, r, t, d]].copy()
    df.columns = ["head", "relation", "tail", "date"]

    for col in ["head", "relation", "tail"]:
        df[col] = (
            df[col].astype("string").str.strip()
        )

    df["date"] = (
        pd.to_datetime(
            df["date"],
            utc=True,
            errors="coerce",
        )
        .dt.tz_convert(None)
        .dt.normalize()
    )

    good = (
        df["date"].notna()
        & df["head"].notna()
        & df["relation"].notna()
        & df["tail"].notna()
        & df["head"].str.len().gt(0)
        & df["relation"].str.len().gt(0)
        & df["tail"].str.len().gt(0)
    )

    if not bool(good.all()):
        raise RuntimeError(
            f"Invalid KG rows: {int((~good).sum())}"
        )

    unique = (
        df.groupby(
            ["head", "relation", "tail"],
            as_index=False,
        )["date"]
        .min()
        .rename(
            columns={"date": "first_observed"}
        )
        .sort_values(
            ["first_observed", "head",
             "relation", "tail"]
        )
        .reset_index(drop=True)
    )

    # The requested experiment ends in 2021.
    # Future records must not enter retrieval.
    unique = unique.loc[
        unique["first_observed"]
        < pd.Timestamp("2022-01-01")
    ].copy()

    if unique.empty:
        raise RuntimeError("Empty temporal KG")

    relations = sorted(
        unique["relation"].unique().tolist()
    )

    log("[KG UNIQUE HISTORICAL EDGES]", len(unique))
    log("[RELATIONS]", len(relations))

    return unique, relations


class TemporalGraph:
    """
    Incremental adjacency index.
    Every inserted edge has first_observed < day.

    Each original directed edge is traversable
    in either neighborhood direction, but its
    original direction is recorded explicitly.
    """

    def __init__(self, triples):
        self.rows = list(
            triples[
                ["head", "relation",
                 "tail", "first_observed"]
            ].itertuples(
                index=False, name=None
            )
        )
        self.pointer = 0
        self.adj = defaultdict(list)
        self.dirty = set()
        self.previous_day = None
        self.active_edges = 0

    def advance(self, day):
        day = pd.Timestamp(day).normalize()

        if (
            self.previous_day is not None
            and day < self.previous_day
        ):
            raise RuntimeError(
                "Graph dates must be increasing"
            )

        self.previous_day = day

        while self.pointer < len(self.rows):
            h, rel, t, observed = (
                self.rows[self.pointer]
            )
            observed = pd.Timestamp(observed)

            if not observed < day:
                break

            h = str(h)
            t = str(t)
            rel = str(rel)

            # neighbor, relation, traversal direction,
            # timestamp, original head, original tail
            self.adj[h].append(
                (t, rel, "out", observed, h, t)
            )
            self.adj[t].append(
                (h, rel, "in", observed, h, t)
            )

            self.dirty.add(h)
            self.dirty.add(t)

            self.pointer += 1
            self.active_edges += 1

    def neighbors(self, node, limit):
        edges = self.adj.get(node, [])

        if node in self.dirty:
            edges.sort(
                key=lambda e: (
                    -e[3].value,
                    e[1],
                    e[0],
                    e[2],
                )
            )
            self.dirty.discard(node)

        return edges[:limit], len(edges)


def path_hash(signature):
    digest = hashlib.blake2b(
        signature.encode("utf-8"),
        digest_size=8,
    ).digest()

    value = int.from_bytes(
        digest,
        "little",
        signed=False,
    )

    bucket = value % HASH_DIM
    sign = 1.0 if ((value >> 32) & 1) else -1.0

    return bucket, sign


def path_feature(
    paths,
    hop,
    relations,
    current_day,
    budget_used,
    degree_cut_count,
):
    """
    Structural path encoding, not semantic LLM RAG.

    Features:
      2R directional final-edge relation weights
      32 hashed relation-direction path signatures
      10 statistics

    Ranking weights depend only on historical
    observed edge recency and path length.
    """
    r_index = {
        r: i for i, r in enumerate(relations)
    }
    R = len(relations)

    out = np.zeros(
        2*R + HASH_DIM + STATS_DIM,
        dtype=np.float32,
    )

    if not paths:
        return out

    relation_vector = np.zeros(
        2*R,
        dtype=np.float64,
    )
    hash_vector = np.zeros(
        HASH_DIM,
        dtype=np.float64,
    )

    endpoints = set()
    other_companies = set()
    ages = []
    weights = []
    inbound = 0
    source_signatures = set()

    for nodes, edges in paths:
        if len(edges) != hop:
            raise RuntimeError(
                "Path depth mismatch"
            )

        for e in edges:
            if not e[3] < current_day:
                raise RuntimeError(
                    "Temporal lookahead detected"
                )

        # Recency uses oldest edge in the path:
        # a path is only as fresh as its oldest link.
        edge_ages = [
            max(
                0,
                (current_day - e[3]).days,
            )
            for e in edges
        ]

        oldest_age = max(edge_ages)

        weight = float(
            np.exp(
                -oldest_age / 365.0
            )
        )

        last = edges[-1]

        relation_id = r_index[last[1]]

        direction_idx = (
            0 if last[2] == "out" else 1
        )

        relation_vector[
            relation_id*2 + direction_idx
        ] += weight

        signature = "|".join(
            f"{e[1]}:{e[2]}"
            for e in edges
        )

        bucket, sign = path_hash(signature)

        hash_vector[bucket] += (
            sign * weight
        )

        source_signatures.add(signature)
        endpoints.add(nodes[-1])

        if nodes[-1].startswith("COMPANY:"):
            other_companies.add(nodes[-1])

        inbound += sum(
            e[2] == "in"
            for e in edges
        )

        ages.append(oldest_age)
        weights.append(weight)

    count = len(paths)

    # normalize structural histograms to keep
    # sample count from dominating their scale
    relation_vector /= max(
        float(np.sum(weights)), 1e-12
    )

    hash_vector /= max(
        float(np.sum(weights)), 1e-12
    )

    stats = np.array(
        [
            np.log1p(count),
            np.log1p(len(endpoints)),
            np.log1p(len(other_companies)),
            np.log1p(len(source_signatures)),
            float(np.mean(ages)) / 365.0,
            float(np.min(ages)) / 365.0,
            float(np.mean(weights)),
            float(inbound) / (count * hop),
            float(budget_used),
            np.log1p(degree_cut_count),
        ],
        dtype=np.float64,
    )

    out[:] = np.concatenate(
        [
            relation_vector,
            hash_vector,
            stats,
        ],
    ).astype(np.float32)

    return out


def retrieve_company_paths(
    graph,
    ticker,
    day,
    max_degree,
    max_paths,
):
    """
    Enumerate bounded simple paths to depth 3.

    Each edge direction is preserved.
    No repeated entity within a path.
    No future edge may be traversed.

    G2 and G3 use only paths at their
    respective hop depth.
    """
    root = f"COMPANY:{ticker}"

    by_hop = {
        1: [],
        2: [],
        3: [],
    }

    degree_cuts = {
        1: 0,
        2: 0,
        3: 0,
    }

    if root not in graph.adj:
        return by_hop, degree_cuts, False

    # Breadth first, so the path budget
    # cannot omit shallow paths in favor of
    # deep paths.
    frontier = [(root, [root], [])]

    total = 0
    exhausted = False

    for hop in (1, 2, 3):
        next_frontier = []

        for node, nodes, edges in frontier:
            neighbors, degree = (
                graph.neighbors(
                    node,
                    max_degree,
                )
            )

            degree_cuts[hop] += max(
                0,
                degree - len(neighbors),
            )

            for edge in neighbors:
                nxt, rel, direction, observed, h, t = edge

                if nxt in nodes:
                    continue

                if not observed < day:
                    raise RuntimeError(
                        "Future KG edge in retrieval"
                    )

                new_nodes = nodes + [nxt]
                new_edges = edges + [edge]

                by_hop[hop].append(
                    (new_nodes, new_edges)
                )

                total += 1

                if total >= max_paths:
                    exhausted = True
                    break

                if hop < 3:
                    next_frontier.append(
                        (
                            nxt,
                            new_nodes,
                            new_edges,
                        )
                    )

            if exhausted:
                break

        if exhausted:
            break

        frontier = next_frontier

    return by_hop, degree_cuts, exhausted


def build_multihop_features(
    dates,
    triples,
    relations,
    max_degree,
    max_paths,
    outdir,
):
    n = len(dates)
    R = len(relations)
    dim = 2*R + HASH_DIM + STATS_DIM

    hop2 = np.zeros(
        (n, 5, dim),
        dtype=np.float32,
    )
    hop3 = np.zeros(
        (n, 5, dim),
        dtype=np.float32,
    )

    graph = TemporalGraph(triples)
    audits = []
    examples = []

    # Sample detailed paths only for reproducible
    # audit dates, not every trading session.
    sample_days = {
        "2018-01-08",
        "2020-03-02",
        "2020-12-01",
        "2021-06-01",
    }

    for i, day in enumerate(dates):
        day = pd.Timestamp(day).normalize()
        graph.advance(day)

        if i % 100 == 0:
            log(
                "[MULTIHOP]",
                i, "/", n,
                str(day.date()),
                "active_edges=",
                graph.active_edges,
            )

        for j, ticker in enumerate(TICKERS):
            paths, cuts, exhausted = (
                retrieve_company_paths(
                    graph,
                    ticker,
                    day,
                    max_degree,
                    max_paths,
                )
            )

            hop2[i, j] = path_feature(
                paths[2],
                2,
                relations,
                day,
                exhausted,
                cuts[2],
            )

            hop3[i, j] = path_feature(
                paths[3],
                3,
                relations,
                day,
                exhausted,
                cuts[3],
            )

            audits.append({
                "date": str(day.date()),
                "ticker": ticker,
                "active_edges": graph.active_edges,
                "hop1_paths": len(paths[1]),
                "hop2_paths": len(paths[2]),
                "hop3_paths": len(paths[3]),
                "hop2_degree_cuts": cuts[2],
                "hop3_degree_cuts": cuts[3],
                "path_budget_exhausted": exhausted,
            })

            if str(day.date()) in sample_days:
                for hop in (1, 2, 3):
                    for nodes, edges in paths[hop][:5]:
                        examples.append({
                            "date": str(day.date()),
                            "ticker": ticker,
                            "hop": hop,
                            "nodes": json.dumps(
                                nodes,
                                ensure_ascii=False,
                            ),
                            "edges": json.dumps(
                                [
                                    {
                                        "original_head": e[4],
                                        "relation": e[1],
                                        "original_tail": e[5],
                                        "traversal": e[2],
                                        "first_observed": str(
                                            e[3].date()
                                        ),
                                    }
                                    for e in edges
                                ],
                                ensure_ascii=False,
                            ),
                        })

    pd.DataFrame(audits).to_csv(
        outdir / "multihop_retrieval_audit.csv",
        index=False,
    )

    pd.DataFrame(
        examples,
        columns=[
            "date", "ticker", "hop",
            "nodes", "edges",
        ],
    ).to_csv(
        outdir / "multihop_path_examples.csv",
        index=False,
    )

    if not (
        np.isfinite(hop2).all()
        and np.isfinite(hop3).all()
    ):
        raise RuntimeError(
            "Nonfinite multi-hop features"
        )

    log(
        "[2-HOP]", hop2.shape,
        "[3-HOP]", hop3.shape,
    )

    return (
        hop2,
        hop3,
        pd.DataFrame(audits),
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--temporal-kg",
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
            "v14a_multihop"
        ),
    )

    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1000.0,
    )

    p.add_argument(
        "--kg-top-k",
        type=int,
        default=12,
    )

    p.add_argument(
        "--max-degree",
        type=int,
        default=12,
    )

    p.add_argument(
        "--max-paths",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # Same original loader options as V14-0.
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke/runs/"
            "ACTIVE__A2C__seed42/model.zip"
        ),
    )
    p.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )
    p.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )
    p.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )
    p.add_argument(
        "--sec-signals",
        default=(
            "outputs/unified_event_memory_v2/"
            "sec_numeric_signals.csv"
        ),
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )
    p.add_argument(
        "--embed",
        default="bge-m3:latest",
    )
    p.add_argument(
        "--device",
        default="cpu",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.0003,
    )
    p.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )
    p.add_argument(
        "--cost-scale",
        type=float,
        default=0.0,
    )
    p.add_argument(
        "--tier",
        default="conservative",
    )
    p.add_argument(
        "--max-days",
        type=int,
        default=0,
    )
    p.add_argument(
        "--split",
        default="train",
    )
    p.add_argument(
        "--algo",
        default="A2C",
    )

    args = p.parse_args()

    if (
        args.max_degree < 1
        or args.max_paths < 1
        or args.ridge_alpha <= 0
    ):
        raise ValueError(
            "Invalid graph or Ridge parameters"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("===== FINRL-X MARKET + DYNAMIC KGT =====")

    train, val, dates, contexts, n_train = (
        v8b.load_data(args)
    )

    dates = base.normalize_dates(dates)

    market = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    )
    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    )

    if not (
        len(dates) == len(market) == len(prices)
    ):
        raise RuntimeError(
            "Dates / market / prices mismatch"
        )

    if (
        dates.min().year != 2018
        or dates.max().year != 2021
    ):
        raise RuntimeError(
            "Unexpected evaluation dates"
        )

    market_base, kgt200 = (
        base.validate_market_layout(market)
    )

    log("===== V13 DIRECT 1-HOP KG RAG =====")

    company_edges = v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg1, kg_relations = v13.build_kg_rag(
        dates,
        company_edges,
        args.kg_top_k,
        args.out,
    )

    log(
        "[1-HOP KG RAG]",
        kg1.shape,
    )

    log("===== FULL TEMPORAL GRAPH =====")

    triples, relations = load_unique_triples(
        args.temporal_kg
    )

    log("===== BUILD 2-HOP AND 3-HOP =====")

    hop2, hop3, graph_audit = (
        build_multihop_features(
            dates,
            triples,
            relations,
            args.max_degree,
            args.max_paths,
            args.out,
        )
    )

    if not (
        kg1.shape[:2]
        == hop2.shape[:2]
        == hop3.shape[:2]
        == market_base.shape[:2]
    ):
        raise RuntimeError(
            "Company feature date alignment failed"
        )

    log("===== COMPOSE FIVE ARMS =====")

    datasets = {
        "G0": base.compose(
            market_base
        ),
        "G1": base.compose(
            market_base, kg1
        ),
        "G2": base.compose(
            market_base, kg1, hop2
        ),
        "G3": base.compose(
            market_base, kg1, hop2, hop3
        ),
        "G4": base.compose(
            market_base, kgt200,
            kg1, hop2, hop3
        ),
    }

    for arm, x in datasets.items():
        log(
            "[ARM]", arm, x.shape
        )
        if not np.isfinite(x).all():
            raise RuntimeError(
                f"Nonfinite features in {arm}"
            )

    targets, maturity = base.build_targets(
        prices,
        dates,
    )

    log("===== 2020 MONTHLY WALK-FORWARD =====")

    # Temporarily use the G0-G4 arm list in
    # the already validated V14-0 evaluator.
    old_arms = list(base.ARMS)

    try:
        base.ARMS = list(ARMS)

        wf_daily, wf_pred, folds = (
            base.run_2020_expanding(
                datasets,
                targets,
                maturity,
                dates,
                args.ridge_alpha,
            )
        )

        log("===== 2021 FIXED VALIDATION =====")

        val_daily, val_pred, val_info = (
            base.run_2021_fixed(
                datasets,
                targets,
                maturity,
                dates,
                args.ridge_alpha,
            )
        )
    finally:
        base.ARMS = old_arms

    folds.to_csv(
        args.out / "walk_forward_folds.csv",
        index=False,
    )

    daily = pd.concat(
        [wf_daily, val_daily],
        ignore_index=True,
    )
    predictions = pd.concat(
        [wf_pred, val_pred],
        ignore_index=True,
    )

    daily.to_csv(
        args.out / "daily_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        args.out / "predictions.csv",
        index=False,
    )

    metrics = base.summarize(daily)
    metrics.to_csv(
        args.out / "method_metrics.csv",
        index=False,
    )

    log("===== BOOTSTRAP =====")

    boot = []

    for period in [
        "2020_WALK_FORWARD",
        "2021_VALIDATION",
    ]:
        for i, (a, b) in enumerate(COMPARISONS):
            for j, metric in enumerate(
                ["mse", "mae", "rank_ic"]
            ):
                boot.append(
                    base.moving_block_comparison(
                        daily,
                        period,
                        a,
                        b,
                        metric,
                        seed=(
                            args.seed
                            + i*10
                            + j
                            + (
                                1000
                                if period
                                == "2021_VALIDATION"
                                else 0
                            )
                        ),
                    )
                )

    pd.DataFrame(boot).to_csv(
        args.out / "paired_bootstrap.csv",
        index=False,
    )

    coverage = {
        "root_found_fraction": float(
            (
                graph_audit["hop1_paths"] > 0
            ).mean()
        ),
        "hop2_positive_fraction": float(
            (
                graph_audit["hop2_paths"] > 0
            ).mean()
        ),
        "hop3_positive_fraction": float(
            (
                graph_audit["hop3_paths"] > 0
            ).mean()
        ),
        "budget_exhaustion_fraction": float(
            graph_audit[
                "path_budget_exhausted"
            ].mean()
        ),
        "mean_hop2_paths": float(
            graph_audit["hop2_paths"].mean()
        ),
        "mean_hop3_paths": float(
            graph_audit["hop3_paths"].mean()
        ),
    }

    manifest = {
        "experiment": "FinRL-X V14-A",
        "market_dim": 106,
        "dynamic_kgt_dim": 200,
        "original_1hop_kg_dim": int(
            kg1.shape[2]
        ),
        "hop2_dim": int(hop2.shape[2]),
        "hop3_dim": int(hop3.shape[2]),
        "full_graph_triples_before_2022": int(
            len(triples)
        ),
        "relation_count": len(relations),
        "relations": relations,
        "max_degree": args.max_degree,
        "max_paths": args.max_paths,
        "ridge_alpha": args.ridge_alpha,
        "2020_folds": len(folds),
        "2021": val_info,
        "coverage": coverage,
        "test_2022_2023_used": False,
        "arms": {
            "G0": "Market",
            "G1": "Market + direct KG RAG",
            "G2": "G1 + 2-hop",
            "G3": "G2 + 3-hop",
            "G4": "G3 + 200D KGT",
        },
        "limitations": [
            "KG first-observed time is not certified publication time.",
            "Paths use bounded graph search, not exhaustive enumeration.",
            "High-degree nodes are truncated by recent observation date.",
            "KG edge types with numbered labels have unknown semantic meanings.",
            "Path signature hashing is structural, not learned semantic retrieval.",
            "G1 reproduces V13 direct-company KG feature definition.",
            "KGT historical source timestamps still require verification.",
            "2021 is previously examined validation data.",
            "No SEC fusion or RL portfolio execution in V14-A.",
        ],
    }

    (
        args.out / "experiment_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 76)
    log("V14-A METRICS")
    log("=" * 76)
    log(metrics.to_string(index=False))

    log()
    log("[MULTIHOP COVERAGE]")
    log(
        json.dumps(
            coverage,
            indent=2,
        )
    )

    log()
    log("V14-A: PASS")
    log("2022-2023 TEST NOT USED")


if __name__ == "__main__":
    main()
