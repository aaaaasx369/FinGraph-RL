#!/usr/bin/env python3

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "unified_event_memory"

FORWARD = BASE / "smoke_original_kgt.py"
EDGE = BASE / "smoke_kgt_edge_loss.py"
OUTPUT = BASE / "smoke_temporal_edge_gradient.py"

for p in (FORWARD, EDGE):
    if not p.exists():
        raise FileNotFoundError(p)

# Reuse the tested KGT+RNN initialization.
forward_source = FORWARD.read_text(encoding="utf-8")

marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if forward_source.count(marker) != 1:
    raise RuntimeError("Forward initialization marker missing")

prefix = forward_source.split(marker, 1)[0]

# Reuse the working original configuration resolver.
edge_source = EDGE.read_text(encoding="utf-8")
tree = ast.parse(edge_source)

functions = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "read_original_setting"
]

if len(functions) != 1:
    raise RuntimeError(
        "Cannot locate working read_original_setting"
    )

config_function = ast.get_source_segment(
    edge_source, functions[0]
)

body = r'''
import ast
from DKG.model import Combiner, EdgeModel
from kgtransformer_event_loader import UnifiedTemporalLoader

RUNNER = ROOT / "references/FinDKG/train_DKG_run.py"

print()
print("=" * 72)
print("CROSS-DAY ORIGINAL EDGE LOSS -> KGT+RNN GRADIENT TEST")
print("=" * 72)

torch.manual_seed(41)

combine_mode = read_original_setting(
    "static_dynamic_combine_mode"
)

combiner_gconv = read_original_setting(
    "combiner_gconv"
)

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

updater.train()
combiner.train()
edge_model.train()

static_entity.structural.requires_grad_(True)

# Deduplicate shared parameters.
params = {}

for p in list(updater.parameters()) + \
         list(edge_model.parameters()) + \
         list(combiner.parameters()) + \
         [static_entity.structural]:
    params[id(p)] = p

optimizer = torch.optim.AdamW(
    list(params.values()),
    lr=1e-4,
    weight_decay=1e-5,
)

optimizer.zero_grad(set_to_none=True)

loader = UnifiedTemporalLoader()

try:
    # Use exact days rather than accidentally selecting
    # an unrelated intervening SEC event.
    batches = {}

    for day, events in loader.iter_event_days(
        "2018-03-04",
        "2018-03-11"
    ):
        if day in ("2018-03-04", "2018-03-11"):
            batches[day] = events

    assert len(batches) == 2, (
        "The two required event dates were not found"
    )

    g1 = loader.graph_from_events(
        "2018-03-04",
        batches["2018-03-04"]
    )

    g2 = loader.graph_from_events(
        "2018-03-11",
        batches["2018-03-11"]
    )

    ids1 = set(g1.ndata[dgl.NID].tolist())
    ids2 = set(g2.ndata[dgl.NID].tolist())

    shared_ids = ids1 & ids2

    print("First day edges:", g1.num_edges())
    print("Second day edges:", g2.num_edges())
    print("Shared entities:", len(shared_ids))

    assert len(shared_ids) > 0, (
        "No shared entities; unsuitable gradient test"
    )

    # ---------------------------------------------
    # STEP 1: Update memory from first-day events.
    # Crucially, gradients are ENABLED.
    # ---------------------------------------------

    print()
    print("Step 1: KGT+RNN forward on 2018-03-04")

    memory1_entity, memory1_relation = updater.forward(
        prior_G=g1,
        batch_G=g1,
        cumul_G=g1,
        static_entity_emb=static_entity,
        dynamic_entity_emb=dynamic_entity,
        dynamic_relation_emb=dynamic_relation,
        device=device,
    )

    assert memory1_entity.structural.requires_grad, (
        "Memory is detached from KGT+RNN"
    )

    # ---------------------------------------------
    # STEP 2: Predict second-day events using the
    # PRE-EVENT memory obtained after first day.
    # Do NOT update memory with g2 before loss.
    # ---------------------------------------------

    print("Step 2: Original Edge Loss on 2018-03-11")

    batch = g2.to(device)

    batch.num_relations = R
    batch.num_all_nodes = N

    nid_gpu = batch.ndata[dgl.NID].long()
    nid_cpu = nid_gpu.cpu()

    static_batch = (
        static_entity.structural[nid_gpu]
    )

    dynamic_batch = (
        memory1_entity.structural[nid_cpu][:, -1, :]
        .to(device)
    )

    relation_batch = (
        memory1_relation.structural[:, -1, :, :]
        .to(device)
    )

    combined = combiner(
        static_batch,
        dynamic_batch,
        batch
    )

    edge_ll = edge_model(
        batch,
        combined,
        static_emb=static_batch,
        dynamic_emb=dynamic_batch,
        dynamic_relation_emb=relation_batch,
        eid=None,
    )

    loss = -edge_ll

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.requires_grad

    print("Original Edge Loss:", loss.item())

    # ---------------------------------------------
    # STEP 3: Backward to original KGTransformer.
    # ---------------------------------------------

    print("Step 3: Backward through historical memory")

    loss.backward()

    structural_kgt_nonzero = 0
    temporal_kgt_nonzero = 0
    structural_rnn_nonzero = 0
    nonzero_other = 0

    for name, p in updater.named_parameters():
        if p.grad is None:
            continue

        assert torch.isfinite(p.grad).all(), name

        has_grad = bool(
            torch.count_nonzero(p.grad).item() > 0
        )

        if not has_grad:
            continue

        if (
            "graph_structural_conv.graph_conv" in name
        ):
            structural_kgt_nonzero += 1

        elif "graph_temporal_conv.graph_conv" in name:
            temporal_kgt_nonzero += 1

        elif "graph_structural_conv.rnn_structural" in name:
            structural_rnn_nonzero += 1

        else:
            nonzero_other += 1

    print()
    print("Structural KGT nonzero tensors:",
          structural_kgt_nonzero)
    print("Temporal KGT nonzero tensors:",
          temporal_kgt_nonzero)
    print("Structural RNN nonzero tensors:",
          structural_rnn_nonzero)
    print("Other updater nonzero tensors:",
          nonzero_other)

    assert structural_kgt_nonzero > 0, (
        "Edge Loss did not reach structural KGTransformer"
    )

    assert structural_rnn_nonzero > 0, (
        "Edge Loss did not reach structural RNN"
    )

    # Under edge-only training, temporal KGT gradients
    # are not guaranteed, since the original EdgeModel
    # predicts edges from STRUCTURAL embeddings only.

    candidate = None

    for name, p in updater.named_parameters():

        if "graph_structural_conv.graph_conv" not in name:
            continue

        if p.grad is not None and \
           torch.count_nonzero(p.grad).item() > 0:
            candidate = p
            break

    assert candidate is not None

    before = candidate.detach().clone()

    optimizer.step()

    parameter_change = (
        candidate.detach() - before
    ).abs().max().item()

    print("Structural KGT parameter change:",
          parameter_change)

    assert parameter_change > 0

    print()
    print("=" * 72)
    print("CROSS-DAY ORIGINAL EDGE LOSS TRAINING PASS")
    print("=" * 72)

    print("Original Edge Loss:", float(loss.detach().cpu()))
    print("Structural KGT gradients:", structural_kgt_nonzero)
    print("Structural RNN gradients:", structural_rnn_nonzero)
    print("KGT parameter updated:", parameter_change)

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

    print("NOTE: Two selected event dates; intermediate")
    print("      event days are intentionally not replayed.")
    print("NOTE: Gradient connectivity test only.")
    print("NOTE: Not a complete historical training run.")
    print("NOTE: SEC features/state not yet fused.")

finally:
    loader.close()
'''

result = (
    prefix
    + "\n"
    + config_function
    + "\n"
    + body
)

compile(result, str(OUTPUT), "exec")

OUTPUT.write_text(
    result,
    encoding="utf-8"
)

print("Generated:", OUTPUT)
