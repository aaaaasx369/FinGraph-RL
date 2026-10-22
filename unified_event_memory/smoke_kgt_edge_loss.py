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



import ast
import math
from pathlib import Path

from DKG.model import Combiner, EdgeModel

print()
print("=" * 70)
print("ORIGINAL FINDKG EDGE LOSS TRAINING SMOKE")
print("=" * 70)

# ======================================================
# 1. Read original experiment configuration
# ======================================================

RUNNER = ROOT / "references/FinDKG/train_DKG_run.py"


def read_original_setting(name):
    """
    Resolve original configuration from:
    1. Explicit assignments in train_DKG_run.py
    2. argparse defaults in the runner or argument parser

    Never silently invent configuration values.
    """

    import ast
    import re

    runner = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(runner)

    found = []

    # 1. Explicit literal assignments
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue

            if not (
                isinstance(target.value, ast.Name)
                and target.value.id == "args"
                and target.attr == name
            ):
                continue

            try:
                found.append(ast.literal_eval(node.value))
            except (ValueError, TypeError):
                pass

    if found:
        if len(set(map(repr, found))) != 1:
            raise RuntimeError(
                f"Conflicting settings for {name}: {found}"
            )
        return found[0]

    # 2. Search the actual argparse add_argument calls
    candidates = [
        RUNNER,
        ROOT / "references/FinDKG/DKG/train.py",
    ]

    for path in candidates:
        if not path.exists():
            continue

        parsed = ast.parse(
            path.read_text(encoding="utf-8")
        )

        for node in ast.walk(parsed):
            if not isinstance(node, ast.Call):
                continue

            if not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"
            ):
                continue

            flags = []

            for a in node.args:
                try:
                    flags.append(ast.literal_eval(a))
                except (ValueError, TypeError):
                    continue

            expected = "--" + name.replace("_", "-")

            if not any(
                flag in (expected, "--" + name)
                for flag in flags
            ):
                continue

            for kw in node.keywords:
                if kw.arg == "default":
                    try:
                        return ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass

    raise RuntimeError(
        f"Original setting {name} still unresolved. "
        "Inspect the original parser/configuration rather "
        "than guessing a value."
    )


combine_mode = read_original_setting(
    "static_dynamic_combine_mode"
)

combiner_gconv = read_original_setting(
    "combiner_gconv"
)

print("Combination mode:", combine_mode)
print("Combiner graph conv:", combiner_gconv)
print("Entities:", N)
print("Relations:", R)
print("Hidden dimension:", DIM)
print("Device:", device)

# ======================================================
# 2. Original model components
# ======================================================

combiner = Combiner(
    static_emb_dim=DIM,
    dynamic_emb_dim=DIM,
    static_dynamic_combine_mode=combine_mode,
    graph_conv=combiner_gconv,
    num_rels=R,
    dropout=0.0,
    activation=None,
).to(device)

edge_model = EdgeModel(
    num_entities=N,
    num_rels=R,
    rel_embed_dim=DIM,
    combiner=combiner,
    dropout=0.0,
).to(device)

# The previous Forward test used random tensors
# without trainable static embedding parameters.
# Enable gradients for the actual loss test.

static_entity.structural.requires_grad_(True)
static_entity.temporal.requires_grad_(True)

# First event batch: dynamic states are initially zero.
# This tests the original edge prediction components.
# It does NOT yet establish gradient flow through KGT+RNN.

optimizer = torch.optim.AdamW(
    list(combiner.parameters())
    + list(edge_model.parameters())
    + [static_entity.structural],
    lr=1e-4,
    weight_decay=1e-5,
)

# Deduplicate parameters because EdgeModel can hold
# a reference to Combiner.

unique_params = {}
for group in optimizer.param_groups:
    for p in group["params"]:
        unique_params[id(p)] = p

optimizer = torch.optim.AdamW(
    list(unique_params.values()),
    lr=1e-4,
    weight_decay=1e-5,
)

optimizer.zero_grad(set_to_none=True)

combiner.train()
edge_model.train()

# ======================================================
# 3. Move the event graph to CUDA
# ======================================================

batch_G = g.to(device)

batch_G.num_relations = R
batch_G.num_all_nodes = N

global_ids_gpu = batch_G.ndata[dgl.NID].long()

# Static embedding is on GPU.
static_batch_emb = static_entity.structural[
    global_ids_gpu
]

# Dynamic embedding is on CPU.
global_ids_cpu = global_ids_gpu.cpu()

dynamic_batch_emb = (
    dynamic_entity.structural[
        global_ids_cpu
    ][:, -1, :]
    .to(device)
)

dynamic_rel_emb = (
    dynamic_relation.structural[:, -1, :, :]
    .to(device)
)

print()
print("Batch nodes:", batch_G.num_nodes())
print("Batch edges:", batch_G.num_edges())

print("Static batch:", tuple(static_batch_emb.shape))
print("Dynamic batch:", tuple(dynamic_batch_emb.shape))
print("Relation embeddings:", tuple(dynamic_rel_emb.shape))

# ======================================================
# 4. Original Combiner + EdgeModel
# ======================================================

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

print()
print("Running original Combiner...")

combined_emb = combiner(
    static_batch_emb,
    dynamic_batch_emb,
    batch_G
)

print("Combined embedding:", tuple(combined_emb.shape))

print()
print("Running ORIGINAL EdgeModel...")

edge_LL = edge_model(
    batch_G,
    combined_emb,
    static_emb=static_batch_emb,
    dynamic_emb=dynamic_batch_emb,
    dynamic_relation_emb=dynamic_rel_emb,
    eid=None,
)

loss = -edge_LL

if loss.ndim != 0:
    raise RuntimeError(
        f"Original EdgeModel returned non-scalar "
        f"loss shape: {tuple(loss.shape)}"
    )

assert torch.isfinite(loss).item()
assert loss.requires_grad

print("Original edge log-likelihood:", edge_LL.item())
print("Original edge loss:", loss.item())

# ======================================================
# 5. Backward
# ======================================================

print()
print("Running backward...")

loss.backward()

total_grad_tensors = 0
nonzero_grad_tensors = 0

for p in unique_params.values():

    if p.grad is None:
        continue

    total_grad_tensors += 1

    if not torch.isfinite(p.grad).all().item():
        raise RuntimeError(
            "Nonfinite gradient detected"
        )

    if torch.count_nonzero(p.grad).item() > 0:
        nonzero_grad_tensors += 1

print("Parameters with gradients:", total_grad_tensors)
print("Nonzero-gradient tensors:", nonzero_grad_tensors)

if nonzero_grad_tensors == 0:
    raise RuntimeError(
        "No trainable parameters received nonzero gradients"
    )

# ======================================================
# 6. Verify optimizer actually changes parameters
# ======================================================

candidate = None

for p in unique_params.values():
    if (
        p.grad is not None
        and torch.count_nonzero(p.grad).item() > 0
    ):
        candidate = p
        break

assert candidate is not None

before = candidate.detach().clone()

optimizer.step()

parameter_change = (
    candidate.detach() - before
).abs().max().item()

print("Max parameter change:", parameter_change)

assert math.isfinite(parameter_change)
assert parameter_change > 0

# ======================================================
# 7. Report
# ======================================================

print()
print("=" * 70)
print("ORIGINAL EDGE LOSS BACKWARD PASS")
print("=" * 70)

print("Loss:", float(loss.detach().cpu()))
print("Nonzero gradient tensors:", nonzero_grad_tensors)
print("Parameter update:", parameter_change)

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

print()
print("ORIGINAL EDGE LOSS TRAINING SMOKE PASS")
print("NOTE: One optimizer step only.")
print("NOTE: KGT+RNN updater gradients not tested here.")
print("NOTE: No checkpoint saved.")
print("NOTE: No SEC feature/state fusion.")
