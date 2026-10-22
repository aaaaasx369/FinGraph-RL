#!/usr/bin/env python3
import ast
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import dgl

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "references/FinDKG"))
sys.path.insert(0, str(ROOT / "unified_event_memory"))

from DKG.model import MultiAspectEmbedding
from kgtransformer_event_loader import UnifiedTemporalLoader

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1048576), b""):
            h.update(part)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/unified_kgt_edge_training_v2/best_validation.pt"
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument(
        "--out",
        default="outputs/unified_kgt_finrl_x_v1"
    )
    args = parser.parse_args()

    checkpoint_path = ROOT / args.checkpoint
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    kg_dir = ROOT / "outputs/unified_temporal_kg_v3"
    entity_path = kg_dir / "entity2id.json"
    relation_path = kg_dir / "relation2id.json"

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    for filename, path in (
        ("entity2id.json", entity_path),
        ("relation2id.json", relation_path),
    ):
        expected = ckpt["vocab_sha256"][filename]
        if sha256(path) != expected:
            raise RuntimeError(
                f"Checkpoint vocabulary mismatch: {filename}"
            )

    with open(entity_path, encoding="utf-8") as f:
        entity2id = json.load(f)

    company_ids = {
        tic: int(entity2id[f"COMPANY:{tic}"])
        for tic in TICKERS
    }

    # Reuse EXACT initialization from the already-tested
    # original KGT smoke script. The script remains unchanged.
    smoke_file = (
        ROOT / "unified_event_memory/smoke_original_kgt.py"
    )
    smoke_src = smoke_file.read_text(encoding="utf-8")
    marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

    if smoke_src.count(marker) != 1:
        raise RuntimeError("Verified initialization marker missing")

    prefix = smoke_src.split(marker, 1)[0]

    # Execute the verified setup in a contained namespace.
    # It initializes the actual CUDA compatibility updater.
    ns = {
        "__file__": str(
            ROOT / "unified_event_memory/smoke_original_kgt.py"
        ),
        "__name__": "_kgt_embedding_setup",
    }
    exec(
        compile(prefix, str(smoke_file), "exec"),
        ns,
        ns,
    )

    updater = ns["updater"]
    static_entity = ns["static_entity"]
    memory_entity = ns["dynamic_entity"]
    memory_relation = ns["dynamic_relation"]
    time_state = ns["node_latest_event_time"]
    device = ns["device"]

    updater.load_state_dict(ckpt["updater"], strict=True)

    with torch.no_grad():
        static_entity.structural.copy_(
            ckpt["static_structural"].to(
                static_entity.structural.device
            )
        )
        static_entity.temporal.copy_(
            ckpt["static_temporal"].to(
                static_entity.temporal.device
            )
        )

    updater.eval()
    for p in updater.parameters():
        p.requires_grad_(False)

    loader = UnifiedTemporalLoader()

    if loader.num_nodes != int(ckpt["entities"]):
        raise RuntimeError("Entity count mismatch")
    if loader.num_relations != int(ckpt["relations"]):
        raise RuntimeError("Relation count mismatch")

    dates = pd.bdate_range(args.start, args.end)
    if len(dates) == 0:
        raise RuntimeError("No dates in requested period")

    # Preload only event-day metadata and the event records,
    # not thousands of large DGL graphs.
    event_iter = iter(
        loader.iter_event_days(
            "2018-01-01",
            args.end,
        )
    )
    pending = next(event_iter, None)

    embedding_dim = int(ckpt["embedding_dim"])
    feature_cols = [f"kg_{i:03d}" for i in range(embedding_dim)]
    rows = []
    event_days_used = 0
    events_used = 0
    last_event = ""

    try:
        with torch.no_grad():
            for day_index, day in enumerate(dates):
                current = day.strftime("%Y-%m-%d")

                # Strict as-of:
                # NEVER consume graph events dated today or later.
                while (
                    pending is not None
                    and str(pending[0])[:10] < current
                ):
                    event_day, event_rows = pending
                    graph = loader.graph_from_events(
                        event_day, event_rows
                    )

                    memory_entity, memory_relation = (
                        updater.forward(
                            prior_G=graph,
                            batch_G=graph,
                            cumul_G=graph,
                            static_entity_emb=static_entity,
                            dynamic_entity_emb=memory_entity,
                            dynamic_relation_emb=memory_relation,
                            device=device,
                        )
                    )

                    event_days_used += 1
                    events_used += len(event_rows)
                    last_event = str(event_day)[:10]
                    pending = next(event_iter, None)

                for tic in TICKERS:
                    nid = company_ids[tic]

                    # 200-D structural static + evolving memory.
                    static_vec = (
                        static_entity.structural[nid]
                        .detach().cpu().float().numpy()
                    )

                    dynamic_vec = (
                        memory_entity.structural[nid, -1, :]
                        .detach().cpu().float().numpy()
                    )

                    vec = static_vec + dynamic_vec

                    if vec.shape != (embedding_dim,):
                        raise RuntimeError(
                            f"Unexpected vector shape: {vec.shape}"
                        )
                    if not np.isfinite(vec).all():
                        raise RuntimeError("Nonfinite embedding")

                    record = {
                        "date": current,
                        "tic": tic,
                        "last_kg_event_date": last_event,
                        "kg_updated": int(bool(last_event)),
                    }

                    record.update({
                        col: float(value)
                        for col, value in zip(feature_cols, vec)
                    })
                    rows.append(record)

                if (day_index + 1) % 100 == 0:
                    print(
                        f"[EXPORT] {current} "
                        f"dates={day_index+1}/{len(dates)} "
                        f"replayed_days={event_days_used} "
                        f"events={events_used}",
                        flush=True,
                    )

        dynamic_df = pd.DataFrame(rows)
        dynamic_path = out / "daily_dynamic_kgt.csv"
        dynamic_df.to_csv(dynamic_path, index=False)

        # A true non-temporal control: company vectors fixed
        # across ALL dates. Uses the trained static embeddings
        # only, never the future dynamic memory.
        static_rows = []
        for tic in TICKERS:
            nid = company_ids[tic]
            vec = (
                static_entity.structural[nid]
                .detach().cpu().float().numpy()
            )

            for day in dates:
                record = {
                    "date": day.strftime("%Y-%m-%d"),
                    "tic": tic,
                }
                record.update({
                    col: float(value)
                    for col, value in zip(feature_cols, vec)
                })
                static_rows.append(record)

        static_df = pd.DataFrame(static_rows)
        static_path = out / "daily_static_kgt.csv"
        static_df.to_csv(static_path, index=False)

        if dynamic_df.duplicated(["date", "tic"]).any():
            raise RuntimeError("Duplicate dynamic date/ticker")
        if static_df.duplicated(["date", "tic"]).any():
            raise RuntimeError("Duplicate static date/ticker")

        print("\n" + "=" * 72)
        print("DAILY KGT EXPORT COMPLETED")
        print("=" * 72)
        print("Date rows:", len(dates))
        print("Dynamic rows:", len(dynamic_df))
        print("Static rows:", len(static_df))
        print("Replayed event days:", event_days_used)
        print("Replayed events:", events_used)
        print("Company IDs:", company_ids)
        print("Dynamic CSV:", dynamic_path)
        print("Static CSV:", static_path)

        # Audit the amount of time variation.
        for tic in TICKERS:
            x = dynamic_df[
                dynamic_df["tic"] == tic
            ][feature_cols].to_numpy()

            changes = int(
                np.any(
                    np.abs(np.diff(x, axis=0)) > 1e-8,
                    axis=1,
                ).sum()
            )

            print(f"{tic}: changed date transitions = {changes}")

        metadata = {
            "base_checkpoint": str(checkpoint_path),
            "base_sha256": sha256(checkpoint_path),
            "entity2id_sha256": sha256(entity_path),
            "relation2id_sha256": sha256(relation_path),
            "company_ids": company_ids,
            "dimension": embedding_dim,
            "embedding_definition": "static_structural + dynamic_structural_last",
            "event_date_policy": "strictly_prior_to_feature_date",
            "calendar": "weekdays_proxy_market_merge_required",
            "replayed_event_days": event_days_used,
            "replayed_events": events_used,
            "trained_weights_frozen": True,
            "sec_fusion_used": False,
            "strict_pit_certified": False,
        }
        (out / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        print("DAILY DYNAMIC AND STATIC KG EXPORT PASS")

    finally:
        loader.close()


if __name__ == "__main__":
    main()
