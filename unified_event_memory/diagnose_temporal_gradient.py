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
    torch.autograd.set_detect_anomaly(True)

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


    print("\\n" + "=" * 70)
    print("AUTOGRAD GRADIENT PATH DIAGNOSTICS")
    print("=" * 70)

    def probe(label, target, retain_graph=True):
        print(f"\\n[PROBE] {label}", flush=True)
        print("  shape:", tuple(target.shape))
        print("  device:", target.device)
        print("  requires_grad:", target.requires_grad)
        print("  grad_fn:", type(target.grad_fn).__name__
              if target.grad_fn is not None else None)

        if not target.requires_grad:
            print("  RESULT: NO GRADIENT PATH")
            return False

        try:
            grad = torch.autograd.grad(
                outputs=loss,
                inputs=target,
                retain_graph=retain_graph,
                allow_unused=True,
            )[0]

            if grad is None:
                print("  RESULT: DISCONNECTED")
                return False

            finite = bool(torch.isfinite(grad).all())
            nonzero = int(torch.count_nonzero(grad).item())

            print("  finite:", finite)
            print("  nonzero elements:", nonzero)
            print("  RESULT:", "PASS" if finite and nonzero else "ZERO/NONFINITE")

            return finite and nonzero

        except Exception as exc:
            print("  RESULT: FAILED")
            print("  exception:", repr(exc))
            return False

    # Probe A: EdgeModel -> current-day dynamic inputs
    probe(
        "A: Edge Loss -> current-day dynamic_batch",
        dynamic_batch
    )

    # Probe B: EdgeModel -> prior-day global CPU memory
    probe(
        "B: Edge Loss -> previous-day structural memory",
        memory1_entity.structural
    )

    # Probe C: EdgeModel -> structural RNN weights
    structural_rnn_param = next(
        p for name, p in updater.named_parameters()
        if "graph_structural_conv.rnn_structural" in name
        and p.requires_grad
    )

    probe(
        "C: Edge Loss -> historical structural RNN parameter",
        structural_rnn_param
    )

    # Probe D: EdgeModel -> structural KGTransformer weights
    kgt_matches = [
        (name, p)
        for name, p in updater.named_parameters()
        if "graph_structural_conv.graph_conv" in name
        and p.requires_grad
    ]

    if not kgt_matches:
        raise RuntimeError("No structural KGT parameters found")

    for name, p in kgt_matches[:3]:
        probe(
            "D: Edge Loss -> " + name,
            p
        )

    print("\\nDIAGNOSTIC COMPLETE")
    print("No optimizer step performed.")
    print("Full training is not yet validated.")

    # Stop diagnostic run before the original optimizer logic.
    raise SystemExit(0)


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
