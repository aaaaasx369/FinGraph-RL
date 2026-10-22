#!/usr/bin/env python3

import gc
import sys
import time
import traceback
from pathlib import Path

import torch
import dgl

ROOT = Path(__file__).resolve().parents[1]
FINDKG = ROOT / "references" / "FinDKG"
GRAPH_PATH = (
    ROOT / "outputs"
    / "unified_kgtransformer_loader_v1"
    / "sample_20180311.bin"
)

sys.path.insert(0, str(FINDKG))

from DKG import settings
from DKG.model import MultiAspectEmbedding

import importlib.util

compat_path = (
    ROOT
    / "unified_event_memory"
    / "fintkg_compat"
    / "embedding.py"
)

spec = importlib.util.spec_from_file_location(
    "DKG.model.embedding_compat",
    str(compat_path)
)

compat_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compat_module
spec.loader.exec_module(compat_module)

EmbeddingUpdater = compat_module.EmbeddingUpdater

print("Using isolated CUDA-compatible EmbeddingUpdater")

from DKG.model.time_interval_transform import TimeIntervalTransform

# Match known original configuration.
N = 14196
R = 19
DIM = 200
LAYERS = 1
HEADS = 8

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("ORIGINAL FINDKG KGT+RNN FORWARD SMOKE TEST")
print("=" * 70)

print("Device:", device)
print("PyTorch:", torch.__version__)
print("DGL:", dgl.__version__)
print("Time dtype:", settings.INTER_EVENT_TIME_DTYPE)

if not GRAPH_PATH.exists():
    raise FileNotFoundError(GRAPH_PATH)

graphs, _ = dgl.load_graphs(str(GRAPH_PATH))
g = graphs[0]

# DGL save/load does not preserve arbitrary Python attributes.
g.num_all_nodes = N
g.num_relations = R

print("Batch nodes:", g.num_nodes())
print("Batch edges:", g.num_edges())

assert g.ndata[dgl.NID].max().item() < N
assert g.edata["rel_type"].max().item() < R
assert "time" in g.edata

torch.manual_seed(41)

print("\nCreating original pairwise time state...")

t0 = time.time()

node_latest_event_time = torch.zeros(
    N,
    N + 1,
    2,
    dtype=settings.INTER_EVENT_TIME_DTYPE,
    device=device,
)

print(
    "Time state MiB:",
    round(
        node_latest_event_time.numel()
        * node_latest_event_time.element_size()
        / 1024**2,
        2
    )
)

time_transform = TimeIntervalTransform(
    log_transform=True
)

print("\nInitializing original EmbeddingUpdater...")

# Use KEYWORD ARGUMENTS to avoid the positional
# mismatch in the original train.py.
updater = EmbeddingUpdater(
    num_nodes=N,
    in_dim=DIM,
    structural_hid_dim=DIM,
    temporal_hid_dim=DIM,
    node_latest_event_time=node_latest_event_time,
    num_rels=R,
    rel_embed_dim=DIM,
    graph_structural_conv="KGT+RNN",
    graph_temporal_conv="KGT+RNN",
    num_gconv_layers=2,
    num_rnn_layers=LAYERS,
    num_node_types=1,
    num_heads=HEADS,
    time_interval_transform=time_transform,
    dropout=0.0,
    activation=None,
    graph_name="FinDKG",
).to(device)

print("Model initialization: PASS")

# Original dynamic embedding shapes.
static_entity = MultiAspectEmbedding(
    structural=(torch.randn(N, DIM, device=device) * 0.02),
    temporal=(torch.randn(N, DIM, device=device) * 0.02),
)

dynamic_entity = MultiAspectEmbedding(
    structural=torch.zeros(N, LAYERS, DIM),
    temporal=torch.zeros(N, LAYERS, DIM, 2),
)

dynamic_relation = MultiAspectEmbedding(
    structural=torch.zeros(R, LAYERS, DIM, 2),
    temporal=torch.zeros(R, LAYERS, DIM, 2),
)

if device.type == "cuda":
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)



import csv
import json
from kgtransformer_event_loader import UnifiedTemporalLoader

OUT = ROOT / "outputs/unified_kgt_replay_v1"
OUT.mkdir(parents=True, exist_ok=True)

loader = UnifiedTemporalLoader()

portfolio_ids = loader.portfolio_ids

previous_entity = dynamic_entity
previous_relation = dynamic_relation

updater.eval()

summary = []
features = []

previous_day = None
total_events = 0
day_count = 0

try:
    with torch.no_grad():

        for day, events in loader.iter_event_days(
            "2018-01-01",
            "2023-12-31"
        ):

            if previous_day is not None:
                assert day > previous_day

            graph = loader.graph_from_events(day, events)

            # Observe only current event pairs.
            src_local, dst_local = graph.edges()
            ids = graph.ndata[dgl.NID]

            h = ids[src_local].long().to(device)
            t = ids[dst_local].long().to(device)

            before_forward = (
                node_latest_event_time[t, h, 0]
                .detach()
                .clone()
            )

            before_reverse = (
                node_latest_event_time[h, t, 1]
                .detach()
                .clone()
            )

            next_entity, next_relation = updater.forward(
                prior_G=graph,
                batch_G=graph,
                cumul_G=graph,
                static_entity_emb=static_entity,
                dynamic_entity_emb=previous_entity,
                dynamic_relation_emb=previous_relation,
                device=device,
            )

            after_forward = node_latest_event_time[t, h, 0]
            after_reverse = node_latest_event_time[h, t, 1]

            changed_time = int(
                (after_forward != before_forward).sum().item()
                +
                (after_reverse != before_reverse).sum().item()
            )

            # Check only the entity states updated this day.
            changed_nodes = graph.ndata[dgl.NID].long().cpu()

            updated_struct = next_entity.structural[
                changed_nodes
            ]

            updated_temp = next_entity.temporal[
                changed_nodes
            ]

            assert torch.isfinite(updated_struct).all()
            assert torch.isfinite(updated_temp).all()

            assert torch.isfinite(
                next_relation.structural
            ).all()

            assert torch.isfinite(
                next_relation.temporal
            ).all()

            day_count += 1
            total_events += len(events)

            summary.append({
                "date": day,
                "events": len(events),
                "nodes": graph.num_nodes(),
                "changed_time_pairs": changed_time,
            })

            # Five-ticker latest memory snapshot.
            # On days with no company-specific events,
            # its state is retained from previous updates.
            for ticker, eid in portfolio_ids.items():

                structural = (
                    next_entity.structural[eid, -1]
                    .detach()
                    .cpu()
                )

                temporal = (
                    next_entity.temporal[eid, -1]
                    .mean(dim=-1)
                    .detach()
                    .cpu()
                )

                vector = torch.cat(
                    [structural, temporal],
                    dim=0
                )

                assert vector.numel() == 2 * DIM
                assert torch.isfinite(vector).all()

                features.append({
                    "date": day,
                    "ticker": ticker,
                    "entity_id": eid,
                    "embedding": vector.tolist(),
                })

            previous_entity = next_entity
            previous_relation = next_relation
            previous_day = day

            if day_count == 1 or day_count % 25 == 0:
                print(
                    f"Day {day_count:3d} | "
                    f"{day} | "
                    f"edges={len(events):5d} | "
                    f"nodes={graph.num_nodes():4d} | "
                    f"total={total_events:,}",
                    flush=True
                )

    assert day_count == 572, (
        f"Expected 572 event days, got {day_count}"
    )

    assert total_events == 224403, (
        f"Expected 224403 events, got {total_events}"
    )

    with (OUT / "replay_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    # Store embeddings in structured JSONL,
    # not a very wide 400-column CSV.
    with (OUT / "portfolio_embeddings.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in features:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    metadata = {
        "model": "original FinDKG KGT+RNN",
        "trained": False,
        "checkpoint_loaded": False,
        "event_days": day_count,
        "events": total_events,
        "entities": N,
        "relations": R,
        "portfolio_tickers": list(portfolio_ids),
        "embedding_dimensions": 2 * DIM,
        "embedding_description": (
            "Untrained structural hidden state "
            "concatenated with mean of two temporal "
            "directional hidden states"
        ),
        "snapshot_frequency": (
            "event days only; not trading days"
        ),
        "sec_numeric_features_used": False,
        "persistent_state_fusion_used": False,
        "strict_pit_certified": False,
    }

    (OUT / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("FULL KGT MEMORY REPLAY PASS")
    print("=" * 70)
    print("Event days:", day_count)
    print("Total events:", total_events)
    print("Portfolio snapshots:", len(features))
    print("Output:", OUT)

    if device.type == "cuda":
        print(
            "Peak CUDA allocated MiB:",
            round(
                torch.cuda.max_memory_allocated(device)
                / 1024**2,
                2
            )
        )
        print(
            "Peak CUDA reserved MiB:",
            round(
                torch.cuda.max_memory_reserved(device)
                / 1024**2,
                2
            )
        )

    print("WARNING: Embeddings are UNTRAINED.")
    print("Do not use these as validated FinRL features.")

finally:
    loader.close()
