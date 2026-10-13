#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from node2vec import Node2Vec
from scipy.linalg import orthogonal_procrustes


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "kg"
    / "data"
    / "finreflectkg_5stocks_2018_2023.parquet"
)

OUTPUT = (
    ROOT
    / "kg"
    / "output"
    / "finreflectkg_company_node2vec_32d.csv"
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

YEARS = [
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
]

COMPANY_TYPES = {
    "ORG",
    "COMP",
}


EMBED_DIM = 32
WALK_LENGTH = 20
NUM_WALKS = 100
WINDOW = 10
SEED = 42


# ============================================================
# Helpers
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return None

    x = str(x).strip().lower()

    if not x:
        return None

    if x in {
        "nan",
        "none",
        "null",
    }:
        return None

    return x


def build_graph(df):
    """
    Company-company graph only.

    Node types:
        ORG
        COMP

    Multiple observations of the same pair increase edge weight.
    """

    G = nx.Graph()

    for row in df.itertuples(index=False):

        if (
            row.entity_type not in COMPANY_TYPES
            or row.target_type not in COMPANY_TYPES
        ):
            continue

        u = clean_text(row.entity)
        v = clean_text(row.target)

        if (
            u is None
            or v is None
            or u == v
        ):
            continue

        if G.has_edge(u, v):
            G[u][v]["weight"] += 1.0
        else:
            G.add_edge(
                u,
                v,
                weight=1.0,
            )

    return G


def train_node2vec(G):
    node2vec = Node2Vec(
        G,
        dimensions=EMBED_DIM,
        walk_length=WALK_LENGTH,
        num_walks=NUM_WALKS,
        workers=1,
        seed=SEED,
        weight_key="weight",
        quiet=False,
    )

    return node2vec.fit(
        window=WINDOW,
        min_count=1,
        batch_words=128,
        seed=SEED,
    )


# ============================================================
# Load
# ============================================================

print("=" * 100)
print("LOAD FINREFLECTKG")
print("=" * 100)

df = pd.read_parquet(INPUT)

df["entity_type"] = (
    df["entity_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["target_type"] = (
    df["target_type"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["year"] = pd.to_numeric(
    df["year"],
    errors="coerce",
)

df = df.dropna(
    subset=[
        "entity",
        "target",
        "year",
    ]
).copy()

df["year"] = df["year"].astype(int)


company_df = df[
    df["entity_type"].isin(COMPANY_TYPES)
    &
    df["target_type"].isin(COMPANY_TYPES)
].copy()


print("All triples:", len(df))
print("Company-company triples:", len(company_df))

print("\nRelationship counts inside company-company graph:")
print(
    company_df["relationship"]
    .value_counts()
    .head(30)
)


# ============================================================
# Train yearly cumulative graphs
# ============================================================

rows = []

previous_aligned = None


for year in YEARS:

    print("\n")
    print("=" * 100)
    print(f"CUMULATIVE GRAPH THROUGH {year}")
    print("=" * 100)

    ydf = company_df[
        company_df["year"] <= year
    ].copy()

    G = build_graph(ydf)

    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())

    if G.number_of_nodes() < 2:
        raise RuntimeError(
            f"Graph too small for {year}"
        )

    if G.number_of_edges() < 1:
        raise RuntimeError(
            f"No edges for {year}"
        )


    model = train_node2vec(G)


    # --------------------------------------------------------
    # Raw embeddings
    # --------------------------------------------------------

    current = {
        str(node): np.asarray(
            model.wv[str(node)],
            dtype=np.float64,
        )
        for node in model.wv.index_to_key
    }


    # --------------------------------------------------------
    # Align current year to previous year's aligned space
    #
    # Find nodes existing in both years.
    #
    # Solve:
    #
    #     current @ R ~= previous
    #
    # with orthogonal Procrustes.
    # --------------------------------------------------------

    if previous_aligned is None:

        aligned = {
            node: vec.copy()
            for node, vec in current.items()
        }

        anchor_count = 0

        print(
            "Alignment: base year, no rotation required"
        )

    else:

        common_nodes = sorted(
            set(current)
            &
            set(previous_aligned)
        )

        anchor_count = len(common_nodes)

        print(
            "Alignment anchors:",
            anchor_count,
        )

        if anchor_count < EMBED_DIM:

            raise RuntimeError(
                f"Only {anchor_count} common nodes "
                f"for year {year}; insufficient for robust alignment."
            )


        A = np.vstack(
            [
                current[node]
                for node in common_nodes
            ]
        )

        B = np.vstack(
            [
                previous_aligned[node]
                for node in common_nodes
            ]
        )


        # Center anchor embeddings before alignment.
        A_mean = A.mean(
            axis=0,
            keepdims=True,
        )

        B_mean = B.mean(
            axis=0,
            keepdims=True,
        )


        R, _ = orthogonal_procrustes(
            A - A_mean,
            B - B_mean,
        )


        aligned = {}

        for node, vec in current.items():

            vec_aligned = (
                (vec - A_mean.ravel())
                @ R
                + B_mean.ravel()
            )

            aligned[node] = vec_aligned


    # --------------------------------------------------------
    # Extract EXACT portfolio company node
    # --------------------------------------------------------

    for ticker in TARGET_TICKERS:

        node = ticker.lower()

        if node in aligned:

            emb = aligned[node]
            available = 1

        else:

            emb = np.zeros(
                EMBED_DIM,
                dtype=np.float64,
            )

            available = 0


        out = {
            "ticker":
                ticker,

            "year":
                year,

            "n2v_available":
                available,

            "graph_nodes":
                G.number_of_nodes(),

            "graph_edges":
                G.number_of_edges(),

            "alignment_anchors":
                anchor_count,
        }


        for i, value in enumerate(emb):

            out[
                f"n2v_{i:02d}"
            ] = float(value)


        rows.append(out)


        print(
            f"{ticker:5s}",
            "FOUND"
            if available
            else "MISSING",
        )


    previous_aligned = aligned


# ============================================================
# Save
# ============================================================

result = pd.DataFrame(rows)

result = result.sort_values(
    [
        "ticker",
        "year",
    ]
).reset_index(drop=True)


result.to_csv(
    OUTPUT,
    index=False,
)


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)

print(
    "Output:",
    OUTPUT
)

print(
    "Shape:",
    result.shape
)

print("\nAvailability:")

print(
    result.groupby("ticker")[
        "n2v_available"
    ].sum()
)

print("\nGraph / alignment summary:")

print(
    result[
        [
            "year",
            "graph_nodes",
            "graph_edges",
            "alignment_anchors",
        ]
    ]
    .drop_duplicates()
    .to_string(index=False)
)

print("\nTicker-year availability:")

print(
    result[
        [
            "ticker",
            "year",
            "n2v_available",
        ]
    ].to_string(index=False)
)
