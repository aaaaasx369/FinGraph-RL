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



print("\n" + "=" * 70)
print("ORIGINAL KGT+RNN BACKWARD SMOKE TEST")
print("=" * 70)

updater.train()

optimizer = torch.optim.AdamW(
    updater.parameters(),
    lr=1e-4
)
optimizer.zero_grad(set_to_none=True)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

try:
    # Real KGT+RNN forward, with gradients enabled.
    updated_entity, updated_relation = updater.forward(
        prior_G=g,
        batch_G=g,
        cumul_G=g,
        static_entity_emb=static_entity,
        dynamic_entity_emb=dynamic_entity,
        dynamic_relation_emb=dynamic_relation,
        device=device,
    )

    # Only gather nodes participating in this batch.
    ids = g.ndata[dgl.NID].long().cpu()

    structural = updated_entity.structural[ids]
    temporal = updated_entity.temporal[ids]

    # DIAGNOSTIC LOSS ONLY.
    # This is NOT FinDKG's original Edge Loss.
    loss = (
        structural.square().mean()
        + temporal.square().mean()
    )

    print("Diagnostic loss:", loss.item())

    assert torch.isfinite(loss)
    assert loss.requires_grad, "Loss is detached"

    loss.backward()

    total_params = 0
    grad_params = 0
    nonzero_grad_params = 0

    structural_grad = 0
    temporal_grad = 0

    for name, p in updater.named_parameters():
        total_params += 1

        if p.grad is None:
            continue

        grad_params += 1

        assert torch.isfinite(p.grad).all(), (
            f"Non-finite gradient: {name}"
        )

        if torch.count_nonzero(p.grad).item() > 0:
            nonzero_grad_params += 1

            if "graph_structural_conv" in name:
                structural_grad += 1

            if "graph_temporal_conv" in name:
                temporal_grad += 1

    print("Total parameter tensors:", total_params)
    print("Parameter tensors with gradients:", grad_params)
    print("Nonzero-gradient tensors:", nonzero_grad_params)
    print("Structural nonzero-gradient tensors:", structural_grad)
    print("Temporal nonzero-gradient tensors:", temporal_grad)

    assert structural_grad > 0, (
        "No structural pathway gradients"
    )
    assert temporal_grad > 0, (
        "No temporal pathway gradients"
    )

    optimizer.step()

    print("\nBACKWARD + OPTIMIZER STEP PASS")

    if device.type == "cuda":
        print(
            "Peak CUDA allocated MiB:",
            round(
                torch.cuda.max_memory_allocated(device)
                / 1024**2, 2
            )
        )
        print(
            "Peak CUDA reserved MiB:",
            round(
                torch.cuda.max_memory_reserved(device)
                / 1024**2, 2
            )
        )

    print("NOTE: Diagnostic loss, not original FinDKG training.")
    print("NOTE: No SEC feature fusion or persistent state fusion.")

except Exception:
    print("\nBACKWARD FAILED — FULL TRACEBACK")
    traceback.print_exc()
    raise
