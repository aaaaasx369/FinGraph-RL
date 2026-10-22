#!/usr/bin/env python3

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "unified_event_memory"

FORWARD = BASE / "smoke_original_kgt.py"
EDGE = BASE / "smoke_kgt_edge_loss.py"
OUTPUT = BASE / "train_unified_kgt_edge.py"

for p in (FORWARD, EDGE):
    if not p.exists():
        raise FileNotFoundError(p)

# Reuse the exact KGT initialization that passed
# the CUDA forward/backward smoke tests.
forward_source = FORWARD.read_text(encoding="utf-8")

marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if forward_source.count(marker) != 1:
    raise RuntimeError(
        "Cannot locate verified KGT initialization"
    )

prefix = forward_source.split(marker, 1)[0]

# Reuse the configuration resolver that worked
# in the original EdgeModel smoke test.
edge_source = EDGE.read_text(encoding="utf-8")
tree = ast.parse(edge_source)

matches = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "read_original_setting"
]

if len(matches) != 1:
    raise RuntimeError(
        "Cannot locate verified config resolver"
    )

config_function = ast.get_source_segment(
    edge_source,
    matches[0],
)

body = r'''
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime

from DKG.model import Combiner, EdgeModel
from kgtransformer_event_loader import UnifiedTemporalLoader


RUNNER = ROOT / "references/FinDKG/train_DKG_run.py"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--epochs", type=int, default=3
)
parser.add_argument(
    "--lr", type=float, default=1e-4
)
parser.add_argument(
    "--weight-decay", type=float, default=1e-5
)
parser.add_argument(
    "--tbptt", type=int, default=2
)
parser.add_argument(
    "--grad-clip", type=float, default=1.0
)
parser.add_argument(
    "--max-train-days", type=int, default=0
)
parser.add_argument(
    "--max-val-days", type=int, default=0
)
parser.add_argument(
    "--log-every", type=int, default=20
)
parser.add_argument(
    "--out",
    default="outputs/unified_kgt_edge_training_v1"
)

args = parser.parse_args()

if args.epochs < 1 or args.tbptt < 1:
    parser.error("--epochs and --tbptt must be >= 1")

if args.lr <= 0 or args.grad_clip <= 0:
    parser.error("--lr and --grad-clip must be positive")

OUT = Path(args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)

torch.manual_seed(41)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(41)

combine_mode = read_original_setting(
    "static_dynamic_combine_mode"
)

combiner_gconv = read_original_setting(
    "combiner_gconv"
)

print()
print("=" * 76)
print("UNIFIED KGTRANSFORMER EDGE TRAINING")
print("=" * 76)
print("Device:", device)
print("Entities:", N)
print("Relations:", R)
print("Static/dynamic combine mode:", combine_mode)
print("Combiner graph conv:", combiner_gconv)
print("TBPTT:", args.tbptt)
print("Epochs:", args.epochs)
print("Output:", OUT)

# ------------------------------------------------
# Original FinDKG Combiner + EdgeModel
# ------------------------------------------------

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

# Edge prediction uses the structural static
# representation. Preserve the validated
# embedding initialization.
static_entity.structural.requires_grad_(True)

# Build a deduplicated parameter set.
param_map = {}

for p in (
    list(updater.parameters())
    + list(combiner.parameters())
    + list(edge_model.parameters())
    + [static_entity.structural]
):
    if p.requires_grad:
        param_map[id(p)] = p

trainable_params = list(param_map.values())

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=args.lr,
    weight_decay=args.weight_decay,
)

# Verify vocabulary consistency.
loader = UnifiedTemporalLoader()

assert loader.num_nodes == N
assert loader.num_relations == R

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()

vocab_hashes = {
    name: file_hash(KG_DIR / name)
    for name in ("entity2id.json", "relation2id.json")
} if "KG_DIR" in globals() else {
    name: file_hash(
        ROOT / "outputs/unified_temporal_kg_v3" / name
    )
    for name in ("entity2id.json", "relation2id.json")
}

# ------------------------------------------------
# Original Edge Loss with safe CPU/GPU indexing.
# The mathematical EdgeModel objective is unchanged.
# ------------------------------------------------

def compute_original_edge_loss(
    batch_graph,
    entity_memory,
    relation_memory
):
    graph = batch_graph.to(device)
    graph.num_relations = R
    graph.num_all_nodes = N

    global_ids_gpu = graph.ndata[dgl.NID].long()
    global_ids_cpu = global_ids_gpu.cpu()

    static_batch = static_entity.structural[
        global_ids_gpu
    ]

    dynamic_batch = entity_memory.structural[
        global_ids_cpu
    ][:, -1, :].to(device)

    relation_batch = relation_memory.structural[
        :, -1, :, :
    ].to(device)

    combined = combiner(
        static_batch,
        dynamic_batch,
        graph,
    )

    edge_ll = edge_model(
        graph,
        combined,
        static_emb=static_batch,
        dynamic_emb=dynamic_batch,
        dynamic_relation_emb=relation_batch,
        eid=None,
    )

    loss = -edge_ll

    if loss.ndim != 0:
        raise RuntimeError(
            f"Expected scalar Edge Loss, got {loss.shape}"
        )

    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite Edge Loss")

    return loss


def detach_memory(entity_memory, relation_memory):
    # Preserve numerical state but truncate
    # historical autograd connections.
    entity_memory = MultiAspectEmbedding(
        structural=entity_memory.structural.detach(),
        temporal=entity_memory.temporal.detach(),
    )

    relation_memory = MultiAspectEmbedding(
        structural=relation_memory.structural.detach(),
        temporal=relation_memory.temporal.detach(),
    )

    return entity_memory, relation_memory


def reset_memory():
    node_latest_event_time.zero_()

    entity = MultiAspectEmbedding(
        structural=torch.zeros(
            N, LAYERS, DIM,
            dtype=torch.float32,
        ),
        temporal=torch.zeros(
            N, LAYERS, DIM, 2,
            dtype=torch.float32,
        ),
    )

    relation = MultiAspectEmbedding(
        structural=torch.zeros(
            R, LAYERS, DIM, 2,
            dtype=torch.float32,
        ),
        temporal=torch.zeros(
            R, LAYERS, DIM, 2,
            dtype=torch.float32,
        ),
    )

    return entity, relation


def run_period(
    start,
    end,
    entity_memory,
    relation_memory,
    training,
    max_days=0,
    epoch=0,
):
    if training:
        updater.train()
        combiner.train()
        edge_model.train()
        optimizer.zero_grad(set_to_none=True)
    else:
        updater.eval()
        combiner.eval()
        edge_model.eval()

    count_days = 0
    count_events = 0

    weighted_loss = 0.0
    unweighted_loss = 0.0

    window_losses = []
    optimizer_steps = 0
    last_date = None

    iterator = loader.iter_event_days(start, end)

    for day, events in iterator:

        if max_days > 0 and count_days >= max_days:
            break

        if last_date is not None:
            assert day > last_date, (
                f"Nonchronological event day: {day}"
            )

        graph = loader.graph_from_events(
            day,
            events,
        )

        # The loss is computed BEFORE updating
        # memory with the current event batch.
        if training:
            loss = compute_original_edge_loss(
                graph,
                entity_memory,
                relation_memory,
            )

            window_losses.append(loss)

        else:
            with torch.no_grad():
                loss = compute_original_edge_loss(
                    graph,
                    entity_memory,
                    relation_memory,
                )

        loss_value = float(loss.detach().cpu())

        unweighted_loss += loss_value
        weighted_loss += loss_value * len(events)

        count_days += 1
        count_events += len(events)

        # Update memory only after predicting
        # the current batch.
        if training:
            entity_memory, relation_memory = (
                updater.forward(
                    prior_G=graph,
                    batch_G=graph,
                    cumul_G=graph,
                    static_entity_emb=static_entity,
                    dynamic_entity_emb=entity_memory,
                    dynamic_relation_emb=relation_memory,
                    device=device,
                )
            )
        else:
            with torch.no_grad():
                entity_memory, relation_memory = (
                    updater.forward(
                        prior_G=graph,
                        batch_G=graph,
                        cumul_G=graph,
                        static_entity_emb=static_entity,
                        dynamic_entity_emb=entity_memory,
                        dynamic_relation_emb=relation_memory,
                        device=device,
                    )
                )

        # Truncated BPTT:
        # one optimizer update per window.
        if training and len(window_losses) >= args.tbptt:

            window_loss = torch.stack(
                window_losses
            ).mean()

            window_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=args.grad_clip,
                error_if_nonfinite=True,
            )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Keep memory VALUES, remove
            # only old graph connections.
            entity_memory, relation_memory = (
                detach_memory(
                    entity_memory,
                    relation_memory,
                )
            )

            window_losses.clear()
            optimizer_steps += 1

        if (
            count_days == 1
            or count_days % args.log_every == 0
        ):
            phase = "TRAIN" if training else "VAL"

            print(
                f"[{phase}] "
                f"epoch={epoch} "
                f"day={count_days} "
                f"date={day} "
                f"edges={len(events)} "
                f"loss={loss_value:.6f}",
                flush=True,
            )

        last_date = day

    # Flush final incomplete TBPTT window.
    if training and window_losses:

        window_loss = torch.stack(
            window_losses
        ).mean()

        window_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=args.grad_clip,
            error_if_nonfinite=True,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        entity_memory, relation_memory = (
            detach_memory(
                entity_memory,
                relation_memory,
            )
        )

        optimizer_steps += 1
        window_losses.clear()

    if count_days == 0:
        raise RuntimeError(
            f"No event days in {start} to {end}"
        )

    metrics = {
        "start": start,
        "end": end,
        "event_days": count_days,
        "events": count_events,
        "mean_event_day_loss": (
            unweighted_loss / count_days
        ),
        "edge_weighted_mean_loss": (
            weighted_loss / count_events
        ),
        "optimizer_steps": optimizer_steps,
        "last_event_date": last_date,
    }

    return entity_memory, relation_memory, metrics


def save_checkpoint(epoch, metrics, best=False):
    name = (
        "best_validation.pt"
        if best else
        f"epoch_{epoch:03d}.pt"
    )

    path = OUT / name
    temp = OUT / (name + ".tmp")

    checkpoint = {
        "epoch": epoch,
        "updater": updater.state_dict(),
        "combiner": combiner.state_dict(),
        "edge_model": edge_model.state_dict(),
        "static_structural": (
            static_entity.structural.detach().cpu()
        ),
        "static_temporal": (
            static_entity.temporal.detach().cpu()
        ),
        "optimizer": optimizer.state_dict(),
        "metrics": metrics,
        "config": vars(args),
        "vocab_sha256": vocab_hashes,
        "entities": N,
        "relations": R,
        "embedding_dim": DIM,
        "objective": "original_findkg_edge_only",
        "sec_numeric_fused": False,
        "persistent_state_fused": False,
        "strict_pit_certified": False,
        "checkpoint_type": (
            "best_val" if best else "epoch"
        ),
    }

    # GPU model states are serialized by torch;
    # loading later should use map_location.
    torch.save(checkpoint, temp)
    os.replace(temp, path)

    return str(path)


history_path = OUT / "training_history.csv"
history = []
best_val = float("inf")
best_epoch = None

try:
    for epoch in range(1, args.epochs + 1):

        print()
        print("=" * 76)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 76)

        start_time = time.time()

        # Start each epoch at 2018 with empty
        # dynamic and timestamp memory, but
        # retain learned model parameters.
        entity_memory, relation_memory = reset_memory()

        entity_memory, relation_memory, train_metrics = (
            run_period(
                "2018-01-01",
                "2021-12-31",
                entity_memory,
                relation_memory,
                training=True,
                max_days=args.max_train_days,
                epoch=epoch,
            )
        )

        # Validation uses the PREVIOUS training
        # memory, not a randomly reset state.
        entity_memory, relation_memory = (
            detach_memory(
                entity_memory,
                relation_memory,
            )
        )

        # No model parameter updates in validation.
        entity_memory, relation_memory, val_metrics = (
            run_period(
                "2022-01-01",
                "2022-12-31",
                entity_memory,
                relation_memory,
                training=False,
                max_days=args.max_val_days,
                epoch=epoch,
            )
        )

        elapsed = time.time() - start_time

        row = {
            "epoch": epoch,
            "train_days": train_metrics["event_days"],
            "train_events": train_metrics["events"],
            "train_loss": (
                train_metrics["mean_event_day_loss"]
            ),
            "val_days": val_metrics["event_days"],
            "val_events": val_metrics["events"],
            "val_loss": (
                val_metrics["mean_event_day_loss"]
            ),
            "optimizer_steps": (
                train_metrics["optimizer_steps"]
            ),
            "elapsed_seconds": round(elapsed, 2),
        }

        history.append(row)

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=list(row),
            )
            writer.writeheader()
            writer.writerows(history)

        metrics = {
            "train": train_metrics,
            "validation": val_metrics,
            "elapsed_seconds": elapsed,
        }

        epoch_checkpoint = save_checkpoint(
            epoch,
            metrics,
            best=False,
        )

        # Lower validation NLL is better.
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_epoch = epoch

            best_path = save_checkpoint(
                epoch,
                metrics,
                best=True,
            )

            print("New best checkpoint:", best_path)

        print()
        print(f"[EPOCH {epoch}]")
        print(f"Train loss: {row['train_loss']:.6f}")
        print(f"Val loss:   {row['val_loss']:.6f}")
        print(f"Steps:      {row['optimizer_steps']}")
        print(f"Seconds:    {elapsed:.2f}")
        print("Saved:", epoch_checkpoint)

        if device.type == "cuda":
            print(
                "Peak GPU allocated MiB:",
                round(
                    torch.cuda.max_memory_allocated(device)
                    / 1024**2,
                    2,
                )
            )

    result = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "train_period": [
            "2018-01-01",
            "2021-12-31",
        ],
        "validation_period": [
            "2022-01-01",
            "2022-12-31",
        ],
        "test_period": [
            "2023-01-01",
            "2023-12-31",
        ],
        "test_used": False,
        "partial_training": (
            args.max_train_days > 0
        ),
        "partial_validation": (
            args.max_val_days > 0
        ),
        "edge_only": True,
        "sec_numeric_fused": False,
        "persistent_state_fused": False,
        "pit_certified": False,
        "history": history,
    }

    (OUT / "training_result.json").write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 76)
    print("UNIFIED KGT EDGE TRAINING PASS")
    print("=" * 76)
    print("Best epoch:", best_epoch)
    print("Best validation loss:", best_val)
    print("Output:", OUT)

    if args.max_train_days or args.max_val_days:
        print(
            "WARNING: Partial data run. "
            "Not a full research training result."
        )

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
    encoding="utf-8",
)

print("TRAINER GENERATED:", OUTPUT)
print("Original FinDKG files remain unchanged.")
