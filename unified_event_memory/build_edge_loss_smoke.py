#!/usr/bin/env python3

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "unified_event_memory/smoke_original_kgt.py"
OUTPUT = ROOT / "unified_event_memory/smoke_kgt_edge_loss.py"

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

source = SOURCE.read_text(encoding="utf-8")

marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if source.count(marker) != 1:
    raise RuntimeError(
        "Cannot find expected smoke-test marker"
    )

prefix = source.split(marker, 1)[0]

body = r'''
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
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))

    values = []

    for node in ast.walk(tree):

        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
        )

        for target in targets:

            if not isinstance(target, ast.Attribute):
                continue

            if target.attr != name:
                continue

            if not (
                isinstance(target.value, ast.Name)
                and target.value.id == "args"
            ):
                continue

            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue

            values.append(value)

    if not values:
        raise RuntimeError(
            f"Cannot determine original args.{name} "
            f"from {RUNNER}. "
            "Do not silently substitute a different "
            "model configuration."
        )

    if any(v != values[0] for v in values):
        raise RuntimeError(
            f"Conflicting original settings for {name}: "
            f"{values}"
        )

    return values[0]


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
'''

result = prefix + "\n" + body

compile(
    result,
    str(OUTPUT),
    "exec"
)

OUTPUT.write_text(
    result,
    encoding="utf-8"
)

print("Generated:", OUTPUT)
