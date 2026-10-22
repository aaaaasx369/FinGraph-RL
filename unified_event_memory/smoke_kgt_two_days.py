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


from kgtransformer_event_loader import UnifiedTemporalLoader

loader = UnifiedTemporalLoader()

try:
    batches = list(
        loader.iter_event_days("2018-03-11", "2018-03-25")
    )[:2]

    assert len(batches) == 2, "Need two nonempty event days"

    updater.eval()

    previous_entity = dynamic_entity
    previous_relation = dynamic_relation
    previous_day = None

    for step, (day, events) in enumerate(batches, 1):
        graph = loader.graph_from_events(day, events)

        # Observe only time-state entries associated with
        # the current event edges. Avoid cloning the full
        # 14,196 x 14,197 x 2 GPU tensor.
        src_local, dst_local = graph.edges()

        src_global = graph.ndata[dgl.NID][src_local]
        dst_global = graph.ndata[dgl.NID][dst_local]

        pair_indices = torch.stack(
            [src_global, dst_global],
            dim=1
        ).unique(dim=0)

        h = pair_indices[:, 0].long().to(device)
        t = pair_indices[:, 1].long().to(device)

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

        with torch.no_grad():
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
            + (after_reverse != before_reverse).sum().item()
        )

        changed_entity = int(
            (
                next_entity.structural
                != previous_entity.structural
            ).any(dim=2).any(dim=1).sum().item()
        )

        print("=" * 65)
        print("REPLAY STEP:", step)
        print("Date:", day)
        print("Edges:", graph.num_edges())
        print("Nodes:", graph.num_nodes())
        print("Changed time-state entries:", changed_time)
        print("Changed entity states:", changed_entity)

        assert changed_time > 0
        assert changed_entity > 0

        for emb in (
            next_entity.structural,
            next_entity.temporal,
            next_relation.structural,
            next_relation.temporal,
        ):
            assert torch.isfinite(emb).all()

        if previous_day is not None:
            assert day > previous_day

        previous_day = day
        previous_entity = next_entity
        previous_relation = next_relation

    if device.type == "cuda":
        print(
            "Peak CUDA allocated MiB:",
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        )

    print("=" * 65)
    print("TWO-DAY MEMORY REPLAY PASS")
    print("NOTE: Untrained parameters; diagnostic replay only.")
    print("NOTE: Persistent state and SEC features not fused.")

finally:
    loader.close()
