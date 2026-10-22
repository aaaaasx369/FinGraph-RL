from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


# ============================================================
# Configuration
# ============================================================

PORTFOLIO = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
]

# COMPETES_WITH and SAME_SECTOR are naturally symmetric.
SYMMETRIC_RELATIONS = {
    "COMPETES_WITH",
    "SAME_SECTOR",
}

OOV_NODE = "__OOV__"


# ============================================================
# Helpers
# ============================================================

def clean_text(x) -> str:
    if pd.isna(x):
        return ""

    return str(x).strip()


def normalize_ticker(x) -> str:
    return clean_text(x).upper()


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = True,
) -> Optional[str]:

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower_map:
            return lower_map[
                candidate.lower()
            ]

    if required:
        raise ValueError(
            "Could not detect required column.\n"
            f"Candidates: {candidates}\n"
            f"Columns: {df.columns.tolist()}"
        )

    return None


def assign_split(
    dates: pd.Series,
) -> pd.Series:

    years = dates.dt.year

    result = pd.Series(
        "other",
        index=dates.index,
        dtype="object",
    )

    result[
        (years >= 2018)
        &
        (years <= 2023)
    ] = "train"

    result[
        years == 2024
    ] = "val"

    result[
        years == 2025
    ] = "test"

    result[
        years == 2026
    ] = "holdout"

    return result


# ============================================================
# Load and normalize KG edges
# ============================================================

def load_edges(
    path: Path,
    undated_policy: str,
    train_start: pd.Timestamp,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    print()
    print("=" * 72)
    print("RAW KG EDGE SCHEMA")
    print("=" * 72)

    print(
        df.columns.tolist()
    )

    source_col = find_column(
        df,
        [
            "source",
            "source_ticker",
            "src",
            "src_ticker",
            "from",
            "from_ticker",
            "head",
            "subject",
        ],
    )

    target_col = find_column(
        df,
        [
            "target",
            "target_ticker",
            "dst",
            "dst_ticker",
            "to",
            "to_ticker",
            "tail",
            "object",
        ],
    )

    relation_col = find_column(
        df,
        [
            "relation",
            "relation_type",
            "edge_type",
            "type",
            "predicate",
        ],
    )

    date_col = find_column(
        df,
        [
            "available_date",
            "filing_date",
            "published_at",
            "event_date",
            "date",
            "source_date",
            "evidence_date",
        ],
        required=False,
    )

    print(
        "source column:",
        source_col,
    )

    print(
        "target column:",
        target_col,
    )

    print(
        "relation column:",
        relation_col,
    )

    print(
        "date column:",
        date_col,
    )

    out = pd.DataFrame()

    out["source"] = (
        df[source_col]
        .map(normalize_ticker)
    )

    out["target"] = (
        df[target_col]
        .map(normalize_ticker)
    )

    out["relation"] = (
        df[relation_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    if date_col is None:

        out[
            "available_date"
        ] = pd.NaT

    else:

        out[
            "available_date"
        ] = pd.to_datetime(
            df[date_col],
            errors="coerce",
        )

    # Keep only four official relation types.
    out = out[
        out[
            "relation"
        ].isin(
            RELATIONS
        )
    ].copy()

    out = out[
        (out["source"] != "")
        &
        (out["target"] != "")
    ].copy()

    out = out[
        out["source"]
        !=
        out["target"]
    ].copy()

    print()
    print(
        "Relation counts before "
        "date policy:"
    )

    print(
        out[
            "relation"
        ].value_counts()
    )

    print()
    print(
        "Undated edges:",
        out[
            "available_date"
        ].isna().sum(),
    )

    # --------------------------------------------------------
    # Important:
    #
    # Undated edges are dangerous.
    #
    # By default EXCLUDE them because we cannot prove
    # they were known at a historical date.
    #
    # Optional train_start is provided only for controlled
    # experiments.
    # --------------------------------------------------------

    if undated_policy == "exclude":

        undated = out[
            out[
                "available_date"
            ].isna()
        ]

        if len(undated):

            print()
            print(
                "WARNING:"
            )

            print(
                f"Excluding {len(undated)} "
                "undated KG edges to avoid "
                "future leakage."
            )

            print(
                undated[
                    "relation"
                ].value_counts()
            )

        out = out[
            out[
                "available_date"
            ].notna()
        ].copy()

    elif undated_policy == "train_start":

        print()
        print(
            "WARNING: assigning undated "
            "edges to train start."
        )

        out[
            "available_date"
        ] = (
            out[
                "available_date"
            ]
            .fillna(
                train_start
            )
        )

    else:

        raise ValueError(
            "undated_policy must be "
            "'exclude' or 'train_start'"
        )

    # Earliest evidence wins for duplicate relation pair.
    out = (
        out
        .sort_values(
            "available_date"
        )
        .drop_duplicates(
            subset=[
                "source",
                "target",
                "relation",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Usable temporal edges:",
        len(out),
    )

    print()

    if len(out):

        print(
            out.groupby(
                "relation"
            )[
                "available_date"
            ]
            .agg(
                [
                    "count",
                    "min",
                    "max",
                ]
            )
        )

    return out


# ============================================================
# Node universe
#
# The embedding coordinate system itself must not depend on
# validation/test/holdout nodes.
#
# Build vocabulary from TRAIN edges only.
#
# Future unseen companies -> __OOV__
# ============================================================

def build_train_node_universe(
    edges: pd.DataFrame,
    train_end: pd.Timestamp,
) -> Tuple[
    List[str],
    Dict[str, int],
]:

    train_edges = edges[
        edges[
            "available_date"
        ]
        <=
        train_end
    ].copy()

    nodes = set(
        PORTFOLIO
    )

    nodes.update(
        train_edges[
            "source"
        ].tolist()
    )

    nodes.update(
        train_edges[
            "target"
        ].tolist()
    )

    nodes.discard(
        OOV_NODE
    )

    nodes = sorted(nodes)

    nodes.append(
        OOV_NODE
    )

    node_to_idx = {
        node: i
        for i, node
        in enumerate(nodes)
    }

    return (
        nodes,
        node_to_idx,
    )


# ============================================================
# Temporal relation adjacency
#
# Directed relation:
#   source -> target = +1
#   target -> source = -1
#
# This preserves direction in one relation matrix.
#
# Symmetric relation:
#   both directions = +1
#
# Rows = company
# Columns = neighbor node
# ============================================================

def build_relation_matrix(
    edges: pd.DataFrame,
    relation: str,
    cutoff: pd.Timestamp,
    nodes: List[str],
    node_to_idx: Dict[str, int],
) -> csr_matrix:

    n = len(nodes)

    subset = edges[
        (
            edges[
                "relation"
            ]
            ==
            relation
        )
        &
        (
            edges[
                "available_date"
            ]
            <=
            cutoff
        )
    ]

    if len(subset) == 0:

        return csr_matrix(
            (n, n),
            dtype=np.float32,
        )

    rows = []
    cols = []
    vals = []

    oov_idx = node_to_idx[
        OOV_NODE
    ]

    symmetric = (
        relation
        in
        SYMMETRIC_RELATIONS
    )

    for edge in subset.itertuples(
        index=False
    ):

        source = edge.source
        target = edge.target

        s_idx = node_to_idx.get(
            source,
            oov_idx,
        )

        t_idx = node_to_idx.get(
            target,
            oov_idx,
        )

        # Avoid OOV self-loop caused by two unseen nodes.
        if (
            s_idx == oov_idx
            and
            t_idx == oov_idx
        ):
            continue

        if symmetric:

            rows.extend(
                [
                    s_idx,
                    t_idx,
                ]
            )

            cols.extend(
                [
                    t_idx,
                    s_idx,
                ]
            )

            vals.extend(
                [
                    1.0,
                    1.0,
                ]
            )

        else:

            # source role
            rows.append(
                s_idx
            )

            cols.append(
                t_idx
            )

            vals.append(
                1.0
            )

            # target role
            rows.append(
                t_idx
            )

            cols.append(
                s_idx
            )

            vals.append(
                -1.0
            )

    matrix = csr_matrix(
        (
            np.asarray(
                vals,
                dtype=np.float32,
            ),
            (
                np.asarray(
                    rows,
                    dtype=np.int32,
                ),
                np.asarray(
                    cols,
                    dtype=np.int32,
                ),
            ),
        ),
        shape=(
            n,
            n,
        ),
        dtype=np.float32,
    )

    # Multiple identical evidence edges should not scale
    # arbitrarily after duplicate removal.
    if matrix.nnz:

        matrix.data = np.clip(
            matrix.data,
            -1.0,
            1.0,
        )

    return matrix


# ============================================================
# Fit TRAIN-ONLY SVD bases
#
# One fixed basis per relation.
#
# 4 relations × 8 dims = 32 KG dimensions.
# ============================================================

def fit_relation_bases(
    edges: pd.DataFrame,
    nodes: List[str],
    node_to_idx: Dict[str, int],
    train_end: pd.Timestamp,
    dims_per_relation: int,
    model_dir: Path,
):

    models = {}

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for relation in RELATIONS:

        print()
        print(
            "FIT:",
            relation,
        )

        A_train = build_relation_matrix(
            edges=edges,
            relation=relation,
            cutoff=train_end,
            nodes=nodes,
            node_to_idx=node_to_idx,
        )

        print(
            "  train adjacency:",
            A_train.shape,
            "nnz=",
            A_train.nnz,
        )

        if A_train.nnz == 0:

            print(
                "  no train edges -> "
                "zero embedding block"
            )

            models[
                relation
            ] = None

            continue

        max_dims = min(
            dims_per_relation,
            max(
                1,
                A_train.shape[1] - 1,
            ),
        )

        svd = TruncatedSVD(
            n_components=
                max_dims,
            algorithm="randomized",
            n_iter=10,
            random_state=42,
        )

        svd.fit(
            A_train
        )

        print(
            "  dims:",
            max_dims,
        )

        print(
            "  explained variance:",
            float(
                svd
                .explained_variance_ratio_
                .sum()
            ),
        )

        models[
            relation
        ] = svd

        joblib.dump(
            svd,
            model_dir
            /
            (
                "temporal_kg_"
                f"{relation.lower()}_"
                "train_2018_2023.joblib"
            ),
        )

    return models


# ============================================================
# Project one temporal snapshot
# ============================================================

def project_snapshot(
    edges: pd.DataFrame,
    date: pd.Timestamp,
    nodes: List[str],
    node_to_idx: Dict[str, int],
    models,
    dims_per_relation: int,
) -> Dict[str, np.ndarray]:

    result = {
        ticker: []
        for ticker in PORTFOLIO
    }

    for relation in RELATIONS:

        A = build_relation_matrix(
            edges=edges,
            relation=relation,
            cutoff=date,
            nodes=nodes,
            node_to_idx=node_to_idx,
        )

        model = models[
            relation
        ]

        if model is None:

            block = np.zeros(
                (
                    len(PORTFOLIO),
                    dims_per_relation,
                ),
                dtype=np.float32,
            )

        else:

            portfolio_indices = [
                node_to_idx[
                    ticker
                ]
                for ticker
                in PORTFOLIO
            ]

            portfolio_rows = A[
                portfolio_indices,
                :
            ]

            transformed = (
                model
                .transform(
                    portfolio_rows
                )
                .astype(
                    np.float32
                )
            )

            block = np.zeros(
                (
                    len(PORTFOLIO),
                    dims_per_relation,
                ),
                dtype=np.float32,
            )

            use_dims = min(
                transformed.shape[1],
                dims_per_relation,
            )

            block[
                :,
                :use_dims
            ] = transformed[
                :,
                :use_dims
            ]

        for i, ticker in enumerate(
            PORTFOLIO
        ):

            result[
                ticker
            ].extend(
                block[
                    i
                ].tolist()
            )

    return {
        ticker:
            np.asarray(
                values,
                dtype=np.float32,
            )
        for ticker, values
        in result.items()
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--m1",
        default=(
            "kg/data/"
            "m1_finrl_features_2018_2026.csv"
        ),
    )

    parser.add_argument(
        "--edges",
        default=(
            "kg/data/"
            "edges_kg_v1.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "kg/data/"
            "m2_finrl_features_2018_2026.csv"
        ),
    )

    parser.add_argument(
        "--model-dir",
        default=(
            "kg/data/models/"
            "temporal_kg"
        ),
    )

    parser.add_argument(
        "--dims-per-relation",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--train-start",
        default="2018-01-01",
    )

    parser.add_argument(
        "--train-end",
        default="2023-12-31",
    )

    parser.add_argument(
        "--undated-policy",
        choices=[
            "exclude",
            "train_start",
        ],
        default="exclude",
    )

    args = parser.parse_args()

    m1_path = Path(
        args.m1
    )

    edges_path = Path(
        args.edges
    )

    output_path = Path(
        args.output
    )

    model_dir = Path(
        args.model_dir
    )

    if not m1_path.exists():

        raise FileNotFoundError(
            m1_path
        )

    if not edges_path.exists():

        raise FileNotFoundError(
            edges_path
        )

    train_start = pd.Timestamp(
        args.train_start
    )

    train_end = pd.Timestamp(
        args.train_end
    )

    # ========================================================
    # Load M1
    # ========================================================

    print("=" * 72)
    print("LOAD M1")
    print("=" * 72)

    m1 = pd.read_csv(
        m1_path
    )

    if (
        "ticker"
        not in m1.columns
        and
        "tic"
        in m1.columns
    ):

        m1 = m1.rename(
            columns={
                "tic":
                    "ticker"
            }
        )

    if "date" not in m1.columns:

        raise ValueError(
            "M1 has no date column."
        )

    m1[
        "date"
    ] = pd.to_datetime(
        m1[
            "date"
        ],
        errors="coerce",
    )

    m1[
        "ticker"
    ] = (
        m1[
            "ticker"
        ]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    m1 = m1.dropna(
        subset=[
            "date",
            "ticker",
        ]
    ).copy()

    print(
        "rows:",
        len(m1),
    )

    print(
        "range:",
        m1[
            "date"
        ].min(),
        "->",
        m1[
            "date"
        ].max(),
    )

    print()

    print(
        m1.groupby(
            "ticker"
        ).size()
    )

    # ========================================================
    # Load temporal edges
    # ========================================================

    print()
    print("=" * 72)
    print("LOAD TEMPORAL KG")
    print("=" * 72)

    edges = load_edges(
        path=edges_path,
        undated_policy=
            args.undated_policy,
        train_start=
            train_start,
    )

    # ========================================================
    # Build TRAIN-ONLY node vocabulary
    # ========================================================

    nodes, node_to_idx = (
        build_train_node_universe(
            edges=edges,
            train_end=
                train_end,
        )
    )

    print()
    print("=" * 72)
    print("TRAIN NODE UNIVERSE")
    print("=" * 72)

    print(
        "nodes:",
        len(nodes),
    )

    print(
        "portfolio:",
        PORTFOLIO,
    )

    print(
        "OOV index:",
        node_to_idx[
            OOV_NODE
        ],
    )

    # Save vocabulary for reproducibility.
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        model_dir
        /
        "train_node_vocabulary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            nodes,
            f,
            indent=2,
        )

    # ========================================================
    # Fit relation-specific TRAIN-only bases
    # ========================================================

    print()
    print("=" * 72)
    print("FIT TRAIN-ONLY KG BASES")
    print("=" * 72)

    models = fit_relation_bases(
        edges=edges,
        nodes=nodes,
        node_to_idx=
            node_to_idx,
        train_end=
            train_end,
        dims_per_relation=
            args.dims_per_relation,
        model_dir=
            model_dir,
    )

    # ========================================================
    # Build temporal features
    #
    # Compute once per unique market date.
    # ========================================================

    print()
    print("=" * 72)
    print("BUILD AS-OF KG FEATURES")
    print("=" * 72)

    market_dates = sorted(
        m1[
            "date"
        ]
        .dropna()
        .unique()
    )

    kg_rows = []

    kg_dim = (
        len(RELATIONS)
        *
        args.dims_per_relation
    )

    kg_cols = []

    for relation in RELATIONS:

        prefix = (
            "kg_"
            +
            relation.lower()
        )

        for d in range(
            args.dims_per_relation
        ):

            kg_cols.append(
                f"{prefix}_{d:02d}"
            )

    print(
        "market dates:",
        len(
            market_dates
        ),
    )

    print(
        "KG dims:",
        kg_dim,
    )

    for i, date_value in enumerate(
        market_dates,
        1,
    ):

        date = pd.Timestamp(
            date_value
        )

        snapshot = project_snapshot(
            edges=edges,
            date=date,
            nodes=nodes,
            node_to_idx=
                node_to_idx,
            models=models,
            dims_per_relation=
                args.dims_per_relation,
        )

        # Latest edge actually available by this day.
        eligible = edges[
            edges[
                "available_date"
            ]
            <=
            date
        ]

        latest_edge_date = (
            eligible[
                "available_date"
            ].max()
            if len(eligible)
            else pd.NaT
        )

        for ticker in PORTFOLIO:

            vector = snapshot[
                ticker
            ]

            row = {
                "date":
                    date,

                "ticker":
                    ticker,

                "kg_latest_edge_date":
                    latest_edge_date,
            }

            for j, col in enumerate(
                kg_cols
            ):

                row[col] = float(
                    vector[j]
                )

            kg_rows.append(
                row
            )

        if (
            i == 1
            or
            i % 250 == 0
            or
            i == len(
                market_dates
            )
        ):

            print(
                f"[{i}/"
                f"{len(market_dates)}] "
                f"{date.date()}"
            )

    kg_daily = pd.DataFrame(
        kg_rows
    )

    # ========================================================
    # Merge into M1
    # ========================================================

    print()
    print("=" * 72)
    print("MERGE M1 + TEMPORAL KG")
    print("=" * 72)

    out = m1.merge(
        kg_daily,
        on=[
            "date",
            "ticker",
        ],
        how="left",
        validate="one_to_one",
    )

    for col in kg_cols:

        out[
            col
        ] = (
            out[
                col
            ]
            .fillna(0.0)
            .astype(
                np.float32
            )
        )

    # ========================================================
    # Leakage checks
    # ========================================================

    out[
        "kg_latest_edge_date"
    ] = pd.to_datetime(
        out[
            "kg_latest_edge_date"
        ],
        errors="coerce",
    )

    leakage = out[
        (
            out[
                "kg_latest_edge_date"
            ].notna()
        )
        &
        (
            out[
                "kg_latest_edge_date"
            ]
            >
            out[
                "date"
            ]
        )
    ]

    if len(leakage):

        print(
            leakage[
                [
                    "ticker",
                    "date",
                    "kg_latest_edge_date",
                ]
            ].head(
                20
            )
        )

        raise AssertionError(
            "TEMPORAL KG LEAKAGE: "
            f"{len(leakage)} rows"
        )

    # ========================================================
    # KG norm / coverage
    # ========================================================

    kg_matrix = out[
        kg_cols
    ].to_numpy(
        dtype=np.float32
    )

    norms = np.linalg.norm(
        kg_matrix,
        axis=1,
    )

    out[
        "kg_embedding_norm"
    ] = norms

    out[
        "kg_embedding_nonzero"
    ] = (
        norms
        >
        1e-12
    ).astype(int)

    out[
        "kg_edge_age_days"
    ] = (
        out[
            "date"
        ]
        -
        out[
            "kg_latest_edge_date"
        ]
    ).dt.days

    out[
        "split"
    ] = assign_split(
        out[
            "date"
        ]
    )

    # ========================================================
    # Save
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # Final QA
    # ========================================================

    print()
    print("=" * 72)
    print("M2 TEMPORAL KG BUILD COMPLETE")
    print("=" * 72)

    print(
        "ROWS:",
        len(out),
    )

    print(
        "DATE RANGE:",
        out[
            "date"
        ].min(),
        "->",
        out[
            "date"
        ].max(),
    )

    print()

    print(
        "KG DIMS:",
        len(
            kg_cols
        ),
    )

    print()

    print(
        "SPLITS:"
    )

    print(
        out.groupby(
            "split"
        ).size()
    )

    print()

    print(
        "TEMPORAL LEAKAGE ROWS:",
        len(
            leakage
        ),
    )

    print()

    print(
        "NONZERO KG ROWS:"
    )

    print(
        out.groupby(
            "ticker"
        )[
            "kg_embedding_nonzero"
        ].agg(
            [
                "sum",
                "count",
            ]
        )
    )

    print()

    print(
        "KG NORM RANGE:"
    )

    print(
        out.groupby(
            "ticker"
        )[
            "kg_embedding_norm"
        ].agg(
            [
                "min",
                "max",
                "mean",
            ]
        )
    )

    print()

    print(
        "EDGE DATE RANGE:"
    )

    print(
        edges[
            "available_date"
        ].min(),
        "->",
        edges[
            "available_date"
        ].max(),
    )

    print()

    print(
        "RELATIONS USED:"
    )

    print(
        edges[
            "relation"
        ].value_counts()
    )

    print()

    print(
        "OUTPUT:",
        output_path,
    )


if __name__ == "__main__":
    main()
