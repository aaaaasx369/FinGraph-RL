from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

PORTFOLIO = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--edges",
        default="kg/data/edges_kg_v1.csv",
    )

    parser.add_argument(
        "--dim-per-relation",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--output",
        default="kg/data/kg_structural_embeddings.npz",
    )

    parser.add_argument(
        "--csv-output",
        default="kg/data/kg_nodes.csv",
    )

    args = parser.parse_args()

    # ========================================================
    # Load edges
    # ========================================================

    edges = pd.read_csv(
        args.edges
    )

    edges = edges[
        edges["relation"].isin(
            RELATIONS
        )
    ].copy()

    edges["source"] = (
        edges["source"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    edges["target"] = (
        edges["target"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # ========================================================
    # Node universe
    #
    # IMPORTANT:
    # include all graph nodes PLUS all portfolio stocks,
    # even if a portfolio stock currently has degree zero.
    # ========================================================

    graph_nodes = (
        set(
            edges["source"]
            .loc[
                edges["source"].ne("")
            ]
            .tolist()
        )
        |
        set(
            edges["target"]
            .loc[
                edges["target"].ne("")
            ]
            .tolist()
        )
    )

    nodes = sorted(
        graph_nodes
        |
        set(PORTFOLIO)
    )

    node_to_idx = {
        node: i
        for i, node in enumerate(nodes)
    }

    n = len(nodes)

    print("=" * 70)
    print("KG STRUCTURAL EMBEDDING")
    print("=" * 70)

    print(
        "Nodes:",
        n,
    )

    print(
        "Edges:",
        len(edges),
    )

    print()

    # ========================================================
    # Build one embedding block per relation type
    # ========================================================

    relation_embeddings = []

    for relation in RELATIONS:

        print(
            "Relation:",
            relation,
        )

        sub = edges[
            edges["relation"]
            == relation
        ]

        A = np.zeros(
            (n, n),
            dtype=np.float32,
        )

        for _, row in sub.iterrows():

            source = str(
                row["source"]
            ).strip()

            target = str(
                row["target"]
            ).strip()

            if (
                source not in node_to_idx
                or
                target not in node_to_idx
            ):
                continue

            i = node_to_idx[source]
            j = node_to_idx[target]

            try:
                confidence = float(
                    row.get(
                        "confidence",
                        1.0,
                    )
                )
            except Exception:
                confidence = 1.0

            if np.isnan(confidence):
                confidence = 1.0

            confidence = max(
                0.0,
                confidence,
            )

            A[i, j] = max(
                A[i, j],
                confidence,
            )

            # Symmetric relations
            if relation in {
                "COMPETES_WITH",
                "SAME_SECTOR",
            }:

                A[j, i] = max(
                    A[j, i],
                    confidence,
                )

        # ----------------------------------------------------
        # Preserve both outgoing and incoming structure.
        #
        # [A | A^T]
        # ----------------------------------------------------

        X = np.concatenate(
            [
                A,
                A.T,
            ],
            axis=1,
        )

        active_nodes = int(
            (
                np.abs(X)
                .sum(axis=1)
                > 0
            ).sum()
        )

        print(
            f"  raw edges   : {len(sub)}"
        )

        print(
            f"  active nodes: {active_nodes}/{n}"
        )

        # ----------------------------------------------------
        # Completely empty relation
        # ----------------------------------------------------

        if np.allclose(
            X,
            0,
        ):

            Z = np.zeros(
                (
                    n,
                    args.dim_per_relation,
                ),
                dtype=np.float32,
            )

            relation_embeddings.append(
                Z
            )

            print(
                "  relation empty -> zero block"
            )

            print()

            continue

        # ----------------------------------------------------
        # SVD dimensionality
        # ----------------------------------------------------

        max_components = min(
            args.dim_per_relation,
            X.shape[0] - 1,
            X.shape[1] - 1,
        )

        max_components = max(
            1,
            max_components,
        )

        svd = TruncatedSVD(
            n_components=max_components,
            random_state=42,
        )

        Z_small = svd.fit_transform(
            X
        )

        # Pad to fixed dimension if necessary
        Z = np.zeros(
            (
                n,
                args.dim_per_relation,
            ),
            dtype=np.float32,
        )

        Z[
            :,
            :Z_small.shape[1],
        ] = Z_small

        # ----------------------------------------------------
        # IMPORTANT:
        # normalize only non-zero rows.
        #
        # Isolated nodes remain exactly zero.
        # ----------------------------------------------------

        norms = np.linalg.norm(
            Z,
            axis=1,
            keepdims=True,
        )

        nonzero_mask = (
            norms[:, 0] > 1e-12
        )

        Z[
            nonzero_mask
        ] = (
            Z[
                nonzero_mask
            ]
            /
            norms[
                nonzero_mask
            ]
        )

        relation_embeddings.append(
            Z.astype(
                np.float32
            )
        )

        print(
            "  SVD dims    :",
            Z_small.shape[1],
        )

        print()

    # ========================================================
    # Concatenate relation blocks
    #
    # 4 relations × 8 dims = 32 dimensions
    # ========================================================

    kg_embedding = np.concatenate(
        relation_embeddings,
        axis=1,
    ).astype(
        np.float32
    )

    # Normalize final embedding,
    # but keep isolated nodes exactly zero.

    norms = np.linalg.norm(
        kg_embedding,
        axis=1,
        keepdims=True,
    )

    nonzero_mask = (
        norms[:, 0] > 1e-12
    )

    kg_embedding[
        nonzero_mask
    ] = (
        kg_embedding[
            nonzero_mask
        ]
        /
        norms[
            nonzero_mask
        ]
    )

    # ========================================================
    # Node degree / QA metadata
    # ========================================================

    degree_rows = []

    for node in nodes:

        outgoing = edges[
            edges["source"] == node
        ]

        incoming = edges[
            edges["target"] == node
        ]

        row = {
            "node": node,
            "out_degree": len(outgoing),
            "in_degree": len(incoming),
            "total_degree": (
                len(outgoing)
                +
                len(incoming)
            ),
            "portfolio_member": (
                1
                if node in PORTFOLIO
                else 0
            ),
        }

        for relation in RELATIONS:

            rel_edges = edges[
                edges["relation"]
                == relation
            ]

            degree = (
                (
                    rel_edges["source"]
                    == node
                )
                |
                (
                    rel_edges["target"]
                    == node
                )
            ).sum()

            row[
                f"degree_{relation.lower()}"
            ] = int(
                degree
            )

        degree_rows.append(
            row
        )

    node_df = pd.DataFrame(
        degree_rows
    )

    # ========================================================
    # Save
    # ========================================================

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        nodes=np.asarray(
            nodes
        ),
        embeddings=kg_embedding,
        relations=np.asarray(
            RELATIONS
        ),
    )

    node_df.to_csv(
        args.csv_output,
        index=False,
    )

    # ========================================================
    # Final QA
    # ========================================================

    print("=" * 70)
    print("FINAL KG EMBEDDINGS")
    print("=" * 70)

    print(
        "Embedding shape:",
        kg_embedding.shape,
    )

    print(
        "Dimension:",
        kg_embedding.shape[1],
    )

    print()
    print("PORTFOLIO")

    for ticker in PORTFOLIO:

        i = node_to_idx[
            ticker
        ]

        z = kg_embedding[
            i
        ]

        total_degree = int(
            node_df.loc[
                node_df["node"]
                == ticker,
                "total_degree",
            ].iloc[0]
        )

        print(
            f"{ticker:5s} "
            f"degree={total_degree:3d} "
            f"norm={np.linalg.norm(z):.6f} "
            f"nonzero={np.count_nonzero(z):2d}"
        )

    print()
    print(
        "Zero-vector portfolio nodes:"
    )

    zero_portfolio = []

    for ticker in PORTFOLIO:

        i = node_to_idx[
            ticker
        ]

        if np.allclose(
            kg_embedding[i],
            0,
        ):
            zero_portfolio.append(
                ticker
            )

    if zero_portfolio:
        print(
            ", ".join(
                zero_portfolio
            )
        )
    else:
        print("None")

    print()
    print(
        "Saved embeddings ->",
        output,
    )

    print(
        "Saved node QA    ->",
        args.csv_output,
    )


if __name__ == "__main__":
    main()
