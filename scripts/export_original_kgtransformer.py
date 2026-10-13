#!/usr/bin/env python3
"""
Export original Financial-TKG KGTransformer embeddings.

Input:
  Trained original KGTransformer checkpoint
  FinDKG-full chronological train/valid/test events

Output:
  weekly_embeddings_64d.csv
  original_daily_embeddings.csv
  original_embedding_manifest.json

No KG model training.
Original model weights are frozen.
PCA fitted using 2018-2021 only.
2022 remains original model validation period.
2023 remains test period.
"""

import argparse
import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
RUN = BASE / "original_run"

sys.path.insert(0, str(RUN.resolve()))
sys.path.insert(0, str((ROOT / "scripts").resolve()))

os.environ.setdefault("DGLBACKEND", "pytorch")

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

import DKG
import DKG.utils as utils
from DKG import data

from DKG.model import (
    DynamicGraphModel,
    EmbeddingUpdater,
    Combiner,
    EdgeModel,
    InterEventTimeModel,
)

from DKG.model.time_interval_transform import (
    TimeIntervalTransform,
)

import v36_finrl_kg_ablation as v36

TICKERS = {
    "AAPL": 9,
    "MSFT": 30,
    "NVDA": 221,
    "AMZN": 20,
    "GOOGL": 11,
}

CHECKPOINT = (
    RUN
    / "result/FinDKG-full/DKG/"
    "FinDKG-full_KGTransformer_"
    "overall_best_checkpoint_opt_edge.pt"
)

FEATURES_11 = [
    f"dkg_emb_{i:02d}" for i in range(11)
]


def fail(message):
    raise RuntimeError(
        "ORIGINAL KG EXPORT STOP: " + str(message)
    )


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint():
    if not CHECKPOINT.exists():
        fail("Original checkpoint is missing")

    # Load only your own trusted training checkpoint.
    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model",
        "args",
        "static_entity_emb",
        "init_dynamic_entity_emb",
        "init_dynamic_relation_emb",
        "val_dynamic_entity_emb",
        "val_dynamic_relation_emb",
    }

    missing = required - set(ckpt)

    if missing:
        fail(f"Missing checkpoint fields: {missing}")

    return ckpt


def construct_original_model(ckpt, G):
    args = ckpt["args"]

    args.device = torch.device("cpu")
    args.cuda = False
    args.gpu = -1

    model_type = "KGT+RNN"

    if G.num_nodes() != 13645:
        fail("Unexpected FinDKG-full node count")

    if G.num_relations != 15:
        fail("Unexpected relation count")

    latest = torch.zeros(
        G.num_nodes(),
        G.num_nodes() + 1,
        2,
        dtype=torch.float32,
    )

    interval_transform = TimeIntervalTransform(
        log_transform=True
    )

    updater = EmbeddingUpdater(
        G.num_nodes(),
        args.static_entity_embed_dim,
        args.structural_dynamic_entity_embed_dim,
        args.temporal_dynamic_entity_embed_dim,
        latest,
        G.num_relations,
        args.rel_embed_dim,
        num_node_types=G.num_node_types,
        num_heads=args.num_attn_heads,
        graph_structural_conv=model_type,
        graph_temporal_conv=model_type,
        num_gconv_layers=args.num_gconv_layers,
        num_rnn_layers=args.num_rnn_layers,
        time_interval_transform=interval_transform,
        dropout=args.dropout,
        activation=args.embedding_updater_activation,
        graph_name=args.graph,
    )

    combiner = Combiner(
        args.static_entity_embed_dim,
        args.structural_dynamic_entity_embed_dim,
        args.static_dynamic_combine_mode,
        args.combiner_gconv,
        G.num_relations,
        args.dropout,
        args.combiner_activation,
    )

    edge_model = EdgeModel(
        G.num_nodes(),
        G.num_relations,
        args.rel_embed_dim,
        combiner,
        dropout=args.dropout,
    )

    time_model = InterEventTimeModel(
        dynamic_entity_embed_dim=(
            args.temporal_dynamic_entity_embed_dim
        ),
        static_entity_embed_dim=(
            args.static_entity_embed_dim
        ),
        num_rels=G.num_relations,
        rel_embed_dim=args.rel_embed_dim,
        num_mix_components=args.num_mix_components,
        time_interval_transform=interval_transform,
        inter_event_time_mode=args.inter_event_time_mode,
        dropout=args.dropout,
    )

    model = DynamicGraphModel(
        updater,
        combiner,
        edge_model,
        time_model,
        latest,
    )

    # Strict loading: prevents silently using random parameters.
    model.load_state_dict(
        ckpt["model"],
        strict=True,
    )

    model.eval()

    return model, args


def replay():
    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    ckpt = load_checkpoint()

    dataset = RUN / "FinDKG_dataset"

    G = data.load_temporal_knowledge_graph(
        "FinDKG-full",
        data_root=str(dataset),
    )

    model, args = construct_original_model(
        ckpt, G
    )

    date_map = pd.read_csv(
        dataset / "FinDKG-full/time2id.txt"
    )

    date_map["DATE_WK"] = pd.to_datetime(
        date_map["DATE_WK"]
    ).dt.normalize()

    lookup = dict(zip(
        date_map["TimeID"].astype(int),
        date_map["DATE_WK"],
    ))

    static_emb = ckpt["static_entity_emb"]
    entity_state = ckpt["init_dynamic_entity_emb"]
    relation_state = ckpt["init_dynamic_relation_emb"]

    # Explicit reset to the beginning of historical replay.
    model.node_latest_event_time.zero_()

    collate = partial(
        utils.collate_fn,
        G=G,
    )

    datasets = [
        ("train", G.train_times),
        ("valid", G.val_times),
        ("test", G.test_times),
    ]

    rows = []

    print(
        "=== ORIGINAL KGTRANSFORMER REPLAY ===",
        flush=True,
    )

    with torch.no_grad():
        for split, times in datasets:
            loader = DataLoader(
                times,
                shuffle=False,
                collate_fn=collate,
            )

            for i, batch in enumerate(loader):
                (
                    prior_G,
                    batch_G,
                    cumul_G,
                    batch_times,
                ) = batch

                # State BEFORE this week's edges.
                # This avoids exposing current-batch events
                # in the prediction made for that batch.
                tid = int(times[i])

                if tid not in lookup:
                    fail(f"Unknown time_id={tid}")

                event_date = lookup[tid]

                state = entity_state.structural

                if state.ndim != 3:
                    fail(
                        "Wrong original structural shape: "
                        + str(tuple(state.shape))
                    )

                if (
                    state.shape[0] != 13645
                    or state.shape[-1] != 64
                ):
                    fail(
                        "Unexpected original entity state: "
                        + str(tuple(state.shape))
                    )

                for ticker, entity_id in TICKERS.items():
                    vector = (
                        state[entity_id, -1, :]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )

                    if not np.isfinite(vector).all():
                        fail(
                            f"Nonfinite state: {ticker} {tid}"
                        )

                    row = {
                        "time_id": tid,
                        "week_date": event_date,
                        "split": split,
                        "tic": ticker,
                    }

                    row.update({
                        f"emb_{k:02d}": float(v)
                        for k, v in enumerate(vector)
                    })

                    rows.append(row)

                # Original architecture / original update.
                entity_state, relation_state = (
                    model.embedding_updater.forward(
                        prior_G,
                        batch_G,
                        cumul_G,
                        static_emb,
                        entity_state,
                        relation_state,
                        args.device,
                    )
                )

                if i % 20 == 0:
                    print(
                        f"{split} {i+1}/{len(loader)} "
                        f"time={tid} "
                        f"date={event_date.date()}",
                        flush=True,
                    )

    weekly = pd.DataFrame(rows)

    if weekly.duplicated(
        ["time_id", "tic"]
    ).any():
        fail("Duplicate weekly entity embeddings")

    weekly.to_csv(
        BASE / "weekly_embeddings_64d.csv",
        index=False,
    )

    print(
        "Original weekly embeddings:",
        len(weekly),
        flush=True,
    )

    return weekly


def make_daily(weekly):
    emb64 = [
        f"emb_{i:02d}" for i in range(64)
    ]

    # PCA may only learn from training-period KG states.
    train = weekly[
        weekly["week_date"] < "2022-01-01"
    ].copy()

    if len(train) < 11:
        fail("Insufficient training observations for PCA")

    # Center and scale using training observations only.
    X_train = train[emb64].to_numpy(dtype=float)

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale < 1e-8] = 1.0

    normalized_train = (
        X_train - mean
    ) / scale

    pca = PCA(
        n_components=11,
        svd_solver="full",
    )

    pca.fit(normalized_train)

    X_all = weekly[emb64].to_numpy(dtype=float)

    reduced = pca.transform(
        (X_all - mean) / scale
    )

    weekly11 = weekly[
        ["time_id", "week_date", "split", "tic"]
    ].copy()

    for i, feature in enumerate(FEATURES_11):
        weekly11[feature] = reduced[:, i]

    weekly11.to_csv(
        BASE / "weekly_embeddings_11d.csv",
        index=False,
    )

    # Save the exact train-only transformation.
    np.savez(
        BASE / "train_only_pca.npz",
        mean=mean,
        scale=scale,
        components=pca.components_,
        center=pca.mean_,
        explained_variance_ratio=(
            pca.explained_variance_ratio_
        ),
    )

    market_path = (
        ROOT
        / "outputs/v20/fusion/dataset_conservative.csv"
    )

    market = pd.read_csv(
        market_path,
        low_memory=False,
    )

    date_col, ticker_col = v36.identify_keys(
        market
    )

    days = market[
        [date_col, ticker_col]
    ].copy()

    days["date"] = pd.to_datetime(
        days[date_col]
    ).dt.normalize()

    days["tic"] = (
        days[ticker_col]
        .astype(str)
        .str.upper()
    )

    days = days[["date", "tic"]]

    if days.duplicated(["date", "tic"]).any():
        fail("Duplicate market ticker-date")

    parts = []

    for ticker in TICKERS:
        left = (
            days[days["tic"] == ticker]
            .sort_values("date")
            .copy()
        )

        right = (
            weekly11[
                weekly11["tic"] == ticker
            ]
            .sort_values("week_date")
            .copy()
        )

        # Use the latest strictly EARLIER KG snapshot.
        # Never use future KG snapshots.
        joined = pd.merge_asof(
            left,
            right.drop(columns=["tic"]),
            left_on="date",
            right_on="week_date",
            direction="backward",
            allow_exact_matches=False,
        )

        joined["tic"] = ticker
        parts.append(joined)

    daily = pd.concat(
        parts, ignore_index=True
    )

    # Before the first snapshot, represent missing
    # historical KG signal with zero after PCA.
    daily[FEATURES_11] = (
        daily[FEATURES_11].fillna(0.0)
    )

    if len(daily) != len(market):
        fail("Daily/market row count mismatch")

    if not np.isfinite(
        daily[FEATURES_11].to_numpy(dtype=float)
    ).all():
        fail("Nonfinite daily KG features")

    # Leakage check:
    known = daily["week_date"].notna()

    if not (
        daily.loc[known, "week_date"]
        < daily.loc[known, "date"]
    ).all():
        fail("A future/same-day KG snapshot was used")

    output = daily[
        ["date", "tic"] + FEATURES_11
    ].copy()

    output["date"] = (
        output["date"]
        .dt.strftime("%Y-%m-%d")
    )

    output.to_csv(
        BASE / "original_daily_embeddings.csv",
        index=False,
    )

    manifest = {
        "model": "Financial-TKG_KGTransformer",
        "checkpoint_path": str(
            CHECKPOINT.resolve()
        ),
        "checkpoint_sha256": checksum(CHECKPOINT),
        "original_model_loaded_strictly": True,
        "replay": "Original EmbeddingUpdater.forward",
        "model_parameter_updates_during_replay": False,
        "weekly_state_dim": 64,
        "finrl_feature_dim": 11,
        "pca_fit_period": "2018-2021",
        "kg_validation_period": "2022",
        "kg_test_period": "2023",
        "weekly_rows": len(weekly),
        "daily_rows": len(output),
        "explained_variance_ratio": (
            pca.explained_variance_ratio_.tolist()
        ),
        "strict_prior_snapshot": True,
        "kg_selection_uses_validation": True,
        "four_requested_relations_complete": False,
        "strict_pit_certified": False,
        "training_period_embedded_with_final_weights": True,
        "notes": (
            "Training-period replay uses a checkpoint "
            "selected after full 2018-2021 training and "
            "2022 validation. This is not strict "
            "walk-forward for 2018-2021; 2023 remains "
            "the main out-of-sample trading test."
        ),
    }

    (BASE / "original_embedding_manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== ORIGINAL DAILY EMBEDDINGS READY ==="
    )
    print("Weekly states:", len(weekly))
    print("Daily features:", len(output))
    print("Dimensions:", len(FEATURES_11))
    print(
        "PCA explained variance:",
        round(
            float(
                np.sum(
                    pca.explained_variance_ratio_
                )
            ), 6
        ),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--skip-replay",
        action="store_true",
    )
    args = p.parse_args()

    BASE.mkdir(parents=True, exist_ok=True)

    if args.skip_replay:
        path = BASE / "weekly_embeddings_64d.csv"

        if not path.exists():
            fail("No weekly replay output exists")

        weekly = pd.read_csv(path)
        weekly["week_date"] = pd.to_datetime(
            weekly["week_date"]
        )
    else:
        weekly = replay()

    make_daily(weekly)


if __name__ == "__main__":
    main()
