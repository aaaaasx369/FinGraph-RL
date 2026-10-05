from __future__ import annotations

import os
import sys
from pathlib import Path
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# ============================================================
# Paths
# ============================================================

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

sys.path.insert(
    0,
    str(HERE)
)

import DKG

from DKG.utils.model_utils import get_embedding
from DKG.model import (
    DynamicGraphModel,
    EmbeddingUpdater,
    Combiner,
    EdgeModel,
    InterEventTimeModel,
)
from DKG.model.time_interval_transform import TimeIntervalTransform
from DKG.train import unpack_checkpoint


GRAPH_NAME = "FinDKG"
DATA_ROOT = HERE / "data"

CHECKPOINT = (
    HERE
    / "result"
    / "FinDKG"
    / "DKG"
    / "FinDKG_KGTransformer_train2020_final.pt"
)

TIME_MAP = (
    PROJECT_ROOT
    / "kg"
    / "data"
    / "findkg"
    / "time_map.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "kg"
    / "data"
    / "findkg"
    / "train2020"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "findkg_weekly_dynamic_embeddings_2018_2022.csv"
)

OUTPUT_NPZ = (
    OUTPUT_DIR
    / "findkg_weekly_dynamic_embeddings_2018_2022.npz"
)


PORTFOLIO = {
    "AAPL": 9,
    "GOOGL": 11,
    "AMZN": 20,
    "MSFT": 30,
    "NVDA": 221,
}

MODEL_TYPE = "KGT+RNN"

INTER_EVENT_TIME_DTYPE = torch.float32


# ============================================================
# Helper
# ============================================================

def to_int(x):

    if torch.is_tensor(x):
        return int(
            x.detach()
            .cpu()
            .reshape(-1)[-1]
            .item()
        )

    a = np.asarray(x).reshape(-1)

    return int(
        a[-1]
    )


def get_structural_matrix(
    dynamic_entity_emb,
):

    x = (
        dynamic_entity_emb
        .structural
        .detach()
        .cpu()
    )

    # Expected FinDKG shape:
    # [nodes, rnn_layers, 200]

    if x.ndim == 3:
        x = x[:, -1, :]

    elif x.ndim != 2:
        raise RuntimeError(
            f"Unexpected structural embedding shape: "
            f"{tuple(x.shape)}"
        )

    return x.numpy().astype(
        np.float32
    )


# ============================================================
# Load graph
# ============================================================

print("=" * 78)
print("LOAD FINDKG-FINRL GRAPH")
print("=" * 78)

G = DKG.data.load_temporal_knowledge_graph(
    GRAPH_NAME,
    data_root=str(DATA_ROOT),
)

print("nodes:", G.number_of_nodes())
print("edges:", G.number_of_edges())
print("relations:", G.num_relations)

print(
    "times:",
    len(G.train_times),
    len(G.val_times),
    len(G.test_times),
)


collate_fn = partial(
    DKG.utils.collate_fn,
    G=G,
)

train_loader = DataLoader(
    G.train_times,
    shuffle=False,
    collate_fn=collate_fn,
)

val_loader = DataLoader(
    G.val_times,
    shuffle=False,
    collate_fn=collate_fn,
)

test_loader = DataLoader(
    G.test_times,
    shuffle=False,
    collate_fn=collate_fn,
)


# ============================================================
# Load time mapping
# ============================================================

print()
print("=" * 78)
print("LOAD TIME MAP")
print("=" * 78)

time_df = pd.read_csv(
    TIME_MAP
)

print(
    "columns:",
    time_df.columns.tolist()
)

time_id_col = None
date_col = None

for c in time_df.columns:

    lc = str(c).lower()

    if lc in {
        "timeid",
        "time_id",
    }:
        time_id_col = c

    if lc in {
        "date_wk",
        "date",
    }:
        date_col = c


if time_id_col is None:
    raise RuntimeError(
        "Could not detect TimeID column "
        f"in {TIME_MAP}"
    )

if date_col is None:
    raise RuntimeError(
        "Could not detect date column "
        f"in {TIME_MAP}"
    )


time_df[time_id_col] = pd.to_numeric(
    time_df[time_id_col],
    errors="coerce",
)

time_df[date_col] = pd.to_datetime(
    time_df[date_col],
    errors="coerce",
)

time_df = time_df.dropna(
    subset=[
        time_id_col,
        date_col,
    ]
)

time_lookup = dict(
    zip(
        time_df[
            time_id_col
        ].astype(int),
        time_df[
            date_col
        ],
    )
)

print(
    "time IDs:",
    min(time_lookup),
    "->",
    max(time_lookup),
)

print(
    "date:",
    min(time_lookup.values()),
    "->",
    max(time_lookup.values()),
)


# ============================================================
# Load checkpoint FIRST
#
# We use its saved args so model dimensions/config match exactly.
# ============================================================

print()
print("=" * 78)
print("LOAD TRAIN2020 FINAL CHECKPOINT")
print("=" * 78)

if not CHECKPOINT.exists():

    raise FileNotFoundError(
        CHECKPOINT
    )


checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
)

args = checkpoint[
    "args"
]

# CUDA_VISIBLE_DEVICES=1 makes physical GPU 1 appear as cuda:0.

# Export/replay on CPU.
# Model parameters are frozen; this avoids 8GB GPU OOM.
args.cuda = False
args.gpu = -1
args.device = torch.device("cpu")

print(
    "checkpoint:",
    CHECKPOINT
)

print(
    "device:",
    args.device
)

print(
    "static dim:",
    args.static_entity_embed_dim
)

print(
    "structural dynamic dim:",
    args.structural_dynamic_entity_embed_dim
)

print(
    "temporal dynamic dim:",
    args.temporal_dynamic_entity_embed_dim
)

print(
    "RNN layers:",
    args.num_rnn_layers
)


# ============================================================
# Rebuild exact KGTransformer architecture
# ============================================================

print()
print("=" * 78)
print("REBUILD KGTRANSFORMER")
print("=" * 78)


node_latest_event_time = torch.zeros(
    G.number_of_nodes(),
    G.number_of_nodes() + 1,
    2,
    dtype=INTER_EVENT_TIME_DTYPE,
)


time_interval_transform = (
    TimeIntervalTransform(
        log_transform=True
    )
)


embedding_updater = EmbeddingUpdater(
    G.number_of_nodes(),
    args.static_entity_embed_dim,
    args.structural_dynamic_entity_embed_dim,
    args.temporal_dynamic_entity_embed_dim,
    node_latest_event_time,
    G.num_relations,
    args.rel_embed_dim,
    num_node_types=
        G.num_node_types,
    num_heads=
        args.num_attn_heads,
    graph_structural_conv=
        MODEL_TYPE,
    graph_temporal_conv=
        MODEL_TYPE,
    num_gconv_layers=
        args.num_gconv_layers,
    num_rnn_layers=
        args.num_rnn_layers,
    time_interval_transform=
        time_interval_transform,
    dropout=
        args.dropout,
    activation=
        args.embedding_updater_activation,
    graph_name=
        GRAPH_NAME,
).to(
    args.device
)


combiner = Combiner(
    args.static_entity_embed_dim,
    args.structural_dynamic_entity_embed_dim,
    args.static_dynamic_combine_mode,
    args.combiner_gconv,
    G.num_relations,
    args.dropout,
    args.combiner_activation,
).to(
    args.device
)


edge_model = EdgeModel(
    G.number_of_nodes(),
    G.num_relations,
    args.rel_embed_dim,
    combiner,
    dropout=args.dropout,
).to(
    args.device
)


inter_event_time_model = (
    InterEventTimeModel(
        dynamic_entity_embed_dim=
            args.temporal_dynamic_entity_embed_dim,
        static_entity_embed_dim=
            args.static_entity_embed_dim,
        num_rels=
            G.num_relations,
        rel_embed_dim=
            args.rel_embed_dim,
        num_mix_components=
            args.num_mix_components,
        time_interval_transform=
            time_interval_transform,
        inter_event_time_mode=
            args.inter_event_time_mode,
        dropout=
            args.dropout,
    )
)


model = DynamicGraphModel(
    embedding_updater,
    combiner,
    edge_model,
    inter_event_time_model,
    node_latest_event_time,
).to(
    args.device
)


# ============================================================
# Load trained parameters + trained initial embeddings
# ============================================================

(
    model,
    _,
    _,
    static_entity_embeds,
    init_dynamic_entity_embeds,
    _val_dynamic_entity_embeds,
    init_dynamic_relation_embeds,
    _val_dynamic_relation_embeds,
    _checkpoint_node_latest_event_time,
    _loss_weights,
    _saved_args,
) = unpack_checkpoint(
    checkpoint,
    model,
)


model.eval()

print(
    "checkpoint loaded successfully"
)


# ============================================================
# Replay chronologically FROM SCRATCH
#
# Critical:
# do NOT start from checkpoint validation state.
#
# Train -> Val -> Test
#
# This produces an as-of embedding after each week's edges.
# ============================================================

model.node_latest_event_time.zero_()
node_latest_event_time.zero_()


dynamic_entity_emb = deepcopy(
    init_dynamic_entity_embeds
)

dynamic_relation_emb = deepcopy(
    init_dynamic_relation_embeds
)


rows = []


def replay_split(
    split_name,
    loader,
    graph_times,
    dynamic_entity_emb,
    dynamic_relation_emb,
):

    graph_times = list(
        graph_times
    )

    print()
    print("=" * 78)
    print(
        "REPLAY",
        split_name.upper()
    )
    print("=" * 78)

    with torch.no_grad():

        for i, batch in enumerate(
            tqdm(
                loader,
                desc=split_name,
            )
        ):

            (
                prior_G,
                batch_G,
                cumul_G,
                batch_times,
            ) = batch


            (
                dynamic_entity_emb,
                dynamic_relation_emb,
            ) = model.embedding_updater.forward(
                prior_G,
                batch_G,
                cumul_G,
                static_entity_embeds,
                dynamic_entity_emb,
                dynamic_relation_emb,
                args.device,
            )


            try:
                time_id = to_int(
                    graph_times[i]
                )

            except Exception:
                time_id = to_int(
                    batch_times
                )


            if time_id not in time_lookup:

                raise RuntimeError(
                    f"No real date for "
                    f"time_id={time_id}"
                )


            date = time_lookup[
                time_id
            ]


            emb_matrix = (
                get_structural_matrix(
                    dynamic_entity_emb
                )
            )


            for (
                ticker,
                entity_id,
            ) in PORTFOLIO.items():

                if (
                    entity_id
                    >=
                    emb_matrix.shape[0]
                ):

                    raise RuntimeError(
                        f"{ticker}: entity "
                        f"{entity_id} outside "
                        f"{emb_matrix.shape}"
                    )


                v = emb_matrix[
                    entity_id
                ]


                row = {
                    "date":
                        date,

                    "time_id":
                        time_id,

                    "split":
                        split_name,

                    "ticker":
                        ticker,

                    "entity_id":
                        entity_id,
                }


                for j, value in enumerate(
                    v
                ):

                    row[
                        f"findkg_emb_{j:03d}"
                    ] = float(
                        value
                    )


                rows.append(
                    row
                )


    return (
        dynamic_entity_emb,
        dynamic_relation_emb,
    )


dynamic_entity_emb, dynamic_relation_emb = (
    replay_split(
        "train",
        train_loader,
        G.train_times,
        dynamic_entity_emb,
        dynamic_relation_emb,
    )
)


dynamic_entity_emb, dynamic_relation_emb = (
    replay_split(
        "val",
        val_loader,
        G.val_times,
        dynamic_entity_emb,
        dynamic_relation_emb,
    )
)


dynamic_entity_emb, dynamic_relation_emb = (
    replay_split(
        "test",
        test_loader,
        G.test_times,
        dynamic_entity_emb,
        dynamic_relation_emb,
    )
)


# ============================================================
# Final dataframe
# ============================================================

df = pd.DataFrame(
    rows
)


df["date"] = pd.to_datetime(
    df["date"]
)


df = df.sort_values(
    [
        "date",
        "ticker",
    ]
).reset_index(
    drop=True
)


emb_cols = sorted(
    [
        c
        for c in df.columns
        if c.startswith(
            "findkg_emb_"
        )
    ]
)


X = df[
    emb_cols
].to_numpy(
    dtype=np.float32
)


df[
    "findkg_embedding_norm"
] = np.linalg.norm(
    X,
    axis=1,
)


# ============================================================
# QA
# ============================================================

print()
print("=" * 78)
print("EXPORT QA")
print("=" * 78)

print(
    "rows:",
    len(df)
)

print(
    "embedding dims:",
    len(emb_cols)
)

print(
    "date:",
    df["date"].min(),
    "->",
    df["date"].max(),
)

print()
print(
    "rows by split:"
)

print(
    df.groupby(
        "split"
    ).size()
)

print()
print(
    "rows by ticker:"
)

print(
    df.groupby(
        "ticker"
    ).size()
)

print()
print(
    "missing embedding values:",
    int(
        df[
            emb_cols
        ]
        .isna()
        .sum()
        .sum()
    )
)

print()
print(
    "embedding norms:"
)

print(
    df.groupby(
        "ticker"
    )[
        "findkg_embedding_norm"
    ].agg(
        [
            "min",
            "max",
            "mean",
        ]
    )
)


# ============================================================
# Strong QA
# ============================================================

expected_rows = (
    (
        len(G.train_times)
        +
        len(G.val_times)
        +
        len(G.test_times)
    )
    *
    len(PORTFOLIO)
)


if len(df) != expected_rows:

    raise AssertionError(
        f"Expected {expected_rows} rows, "
        f"got {len(df)}"
    )


if len(emb_cols) != 200:

    raise AssertionError(
        f"Expected 200 dimensions, "
        f"got {len(emb_cols)}"
    )


if (
    df[
        emb_cols
    ]
    .isna()
    .any()
    .any()
):

    raise AssertionError(
        "NaN found in embeddings"
    )


if df.duplicated(
    [
        "date",
        "ticker",
    ]
).any():

    raise AssertionError(
        "Duplicate date/ticker rows"
    )


# ============================================================
# Save
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    OUTPUT_CSV,
    index=False,
)


np.savez_compressed(
    OUTPUT_NPZ,
    embeddings=X,
    dates=
        df["date"]
        .astype(str)
        .to_numpy(),
    time_ids=
        df["time_id"]
        .to_numpy(),
    splits=
        df["split"]
        .to_numpy(),
    tickers=
        df["ticker"]
        .to_numpy(),
    entity_ids=
        df["entity_id"]
        .to_numpy(),
)


print()
print("=" * 78)
print("EXPORT COMPLETE")
print("=" * 78)

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "NPZ:",
    OUTPUT_NPZ
)
