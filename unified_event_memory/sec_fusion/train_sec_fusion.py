#!/usr/bin/env python3

import gc
import sys
import time
import traceback
from pathlib import Path

import torch
import dgl

ROOT = Path(__file__).resolve().parents[2]
FINDKG = ROOT / "references" / "FinDKG"
GRAPH_PATH = (
    ROOT / "outputs"
    / "unified_kgtransformer_loader_v1"
    / "sample_20180311.bin"
)

sys.path.insert(0, str(FINDKG))


# Make original FinDKG and Unified KG modules importable.
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FINDKG_ROOT = _PROJECT_ROOT / "references" / "FinDKG"
_UNIFIED_ROOT = _PROJECT_ROOT / "unified_event_memory"

for _path in (
    _FINDKG_ROOT,
    _UNIFIED_ROOT,
):
    if not _path.exists():
        raise FileNotFoundError(_path)

    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

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

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import random
import time
from datetime import date
from pathlib import Path

import pandas as pd

from DKG.model import Combiner, EdgeModel
from kgtransformer_event_loader import UnifiedTemporalLoader
from sec_fusion.model import SECGraphFusion


RUNNER = ROOT / "references/FinDKG/train_DKG_run.py"

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

FEATURES = [
    "impact_norm",
    "risk_norm",
    "confidence_norm",
]

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base-checkpoint",
    default=(
        "outputs/unified_kgt_edge_training_v2/"
        "best_validation.pt"
    ),
)

parser.add_argument(
    "--sec-features",
    default=(
        "outputs/unified_sec_fusion_v1/"
        "sec_asof_features.csv"
    ),
)

parser.add_argument(
    "--sec-metadata",
    default=(
        "outputs/unified_sec_fusion_v1/"
        "metadata.json"
    ),
)

parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--max-train-days", type=int, default=0)
parser.add_argument("--max-val-days", type=int, default=0)
parser.add_argument("--log-every", type=int, default=20)
parser.add_argument("--seed", type=int, default=41)

parser.add_argument(
    "--out",
    default="outputs/unified_sec_fusion_training_v1",
)

args = parser.parse_args()

if args.epochs < 1:
    parser.error("--epochs must be positive")

if args.lr <= 0 or args.grad_clip <= 0:
    parser.error("--lr and --grad-clip must be positive")

random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

OUT = (ROOT / args.out).resolve()
OUT.mkdir(parents=True, exist_ok=True)


def resolve_path(p):
    path = Path(p)
    return path if path.is_absolute() else ROOT / path


BASE_CHECKPOINT = resolve_path(args.base_checkpoint)
SEC_CSV = resolve_path(args.sec_features)
SEC_METADATA = resolve_path(args.sec_metadata)
KG_DIR = ROOT / "outputs/unified_temporal_kg_v3"

for p in (
    BASE_CHECKPOINT,
    SEC_CSV,
    SEC_METADATA,
    KG_DIR / "entity2id.json",
    KG_DIR / "relation2id.json",
):
    if not p.exists():
        raise FileNotFoundError(
            f"Required input missing: {p}"
        )


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


with open(
    KG_DIR / "entity2id.json",
    encoding="utf-8",
) as f:
    entity2id = json.load(f)

company_ids = {
    ticker: int(entity2id[f"COMPANY:{ticker}"])
    for ticker in TICKERS
}

base = torch.load(
    BASE_CHECKPOINT,
    map_location=device,
)

if int(base["entities"]) != N:
    raise RuntimeError("Checkpoint entity count mismatch")

if int(base["relations"]) != R:
    raise RuntimeError("Checkpoint relation count mismatch")

if int(base["embedding_dim"]) != DIM:
    raise RuntimeError("Checkpoint embedding dimension mismatch")

# Enforce exactly the same vocabulary mapping
# as the trained KGT checkpoint.
if "vocab_sha256" not in base:
    raise RuntimeError(
        "Base checkpoint has no vocabulary fingerprint"
    )

for filename in ("entity2id.json", "relation2id.json"):
    expected = base["vocab_sha256"].get(filename)

    actual = sha256(KG_DIR / filename)

    if expected != actual:
        raise RuntimeError(
            f"Vocabulary fingerprint mismatch: {filename}"
        )

# Original KGT and EdgeModel configuration.
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

# Restore verified trained components.
updater.load_state_dict(
    base["updater"],
    strict=True,
)

combiner.load_state_dict(
    base["combiner"],
    strict=True,
)

edge_model.load_state_dict(
    base["edge_model"],
    strict=True,
)

with torch.no_grad():
    static_entity.structural.copy_(
        base["static_structural"].to(device)
    )

    static_entity.temporal.copy_(
        base["static_temporal"].to(device)
    )

# Freeze the complete original FinDKG model.
for module in (updater, combiner, edge_model):
    module.eval()

    for p in module.parameters():
        p.requires_grad_(False)

static_entity.structural.requires_grad_(False)
static_entity.temporal.requires_grad_(False)

# New trainable SEC Fusion module.
fusion = SECGraphFusion(
    kgt_dim=DIM,
    sec_hidden_dim=64,
    dropout=0.1,
).to(device)

optimizer = torch.optim.AdamW(
    fusion.parameters(),
    lr=args.lr,
    weight_decay=args.weight_decay,
)

# ---------------------------------------------------
# SEC AS-OF FEATURES
# ---------------------------------------------------

sec_metadata = json.loads(
    SEC_METADATA.read_text(encoding="utf-8")
)

for ticker in TICKERS:
    expected = company_ids[ticker]
    actual = int(
        sec_metadata["company_ids"][ticker]
    )

    if expected != actual:
        raise RuntimeError(
            f"SEC/KGT ID mismatch for {ticker}: "
            f"{actual} != {expected}"
        )

sec_df = pd.read_csv(SEC_CSV)

required = {
    "date",
    "ticker",
    "company_id",
    "sec_mask",
    "sec_age_days",
    "sec_event_date",
    *FEATURES,
}

missing_columns = required - set(sec_df.columns)

if missing_columns:
    raise RuntimeError(
        f"SEC CSV missing columns: "
        f"{sorted(missing_columns)}"
    )

if sec_df.duplicated(["ticker", "date"]).any():
    raise RuntimeError(
        "Duplicate ticker/date SEC feature rows"
    )

# Check all portfolio mappings, not just masked rows.
for ticker in TICKERS:
    sub = sec_df.loc[sec_df["ticker"] == ticker]

    if sub.empty:
        raise RuntimeError(
            f"No SEC feature dates for {ticker}"
        )

    ids = set(
        pd.to_numeric(
            sub["company_id"],
            errors="raise",
        ).astype(int)
    )

    if ids != {company_ids[ticker]}:
        raise RuntimeError(
            f"Incorrect company IDs in SEC CSV: {ticker}"
        )

# Only keep valid observed SEC feature rows.
# Masked rows are not observations.
observed = sec_df[
    sec_df["sec_mask"].astype(int) == 1
].copy()

observed["date"] = pd.to_datetime(
    observed["date"],
    errors="raise",
).dt.date

observed["sec_event_date"] = pd.to_datetime(
    observed["sec_event_date"],
    errors="raise",
).dt.date

if not (
    observed["sec_event_date"]
    < observed["date"]
).all():
    raise RuntimeError(
        "SEC AS-OF CHECK FAILED: future/same-day signal"
    )

max_age = int(sec_metadata["max_age_days"])

if not observed[
    "sec_age_days"
].between(1, max_age).all():
    raise RuntimeError(
        "SEC age exceeds configured maximum"
    )

if not observed.empty:
    x = observed[FEATURES].to_numpy(dtype="float32")

    if not bool(
        torch.isfinite(torch.from_numpy(x)).all()
    ):
        raise RuntimeError("Nonfinite SEC numeric feature")

# Indexed by ticker for efficient as-of lookup.
# Existing Feature Builder uses weekdays. For an
# event on a weekend, we select the most recent
# *earlier generated date*. This is conservative;
# never use a future weekday's feature row.

sec_index = {}

for ticker in TICKERS:
    rows = observed[
        observed["ticker"] == ticker
    ].sort_values("date")

    sec_index[ticker] = (
        rows["date"].tolist(),
        rows.to_dict("records"),
    )

print()
print("=" * 76)
print("REAL SEC NUMERIC + FROZEN KGT EDGE TRAINING")
print("=" * 76)
print("Device:", device)
print("Base checkpoint:", BASE_CHECKPOINT)
print("SEC features:", SEC_CSV)
print("SEC observed feature rows:", len(observed))
print("Portfolio IDs:", company_ids)
print("Training: 2018-2021")
print("Validation: 2022")
print("Fusion dimension:", DIM)
print("Output:", OUT)

# Save initial memory templates. The verified
# KGT prefix already created these objects.
initial_entity_structural = (
    dynamic_entity.structural.detach().clone()
)
initial_entity_temporal = (
    dynamic_entity.temporal.detach().clone()
)
initial_relation_structural = (
    dynamic_relation.structural.detach().clone()
)
initial_relation_temporal = (
    dynamic_relation.temporal.detach().clone()
)


def reset_memory():
    # Restore all dynamic representations and
    # pairwise time state for historical replay.
    node_latest_event_time.zero_()

    entity = MultiAspectEmbedding(
        structural=initial_entity_structural.clone(),
        temporal=initial_entity_temporal.clone(),
    )

    relation = MultiAspectEmbedding(
        structural=initial_relation_structural.clone(),
        temporal=initial_relation_temporal.clone(),
    )

    return entity, relation


def asof_for_graph(day, graph):
    """
    Return:
        SEC numeric [num_local_nodes, 3]
        mask        [num_local_nodes, 1]
        age         [num_local_nodes, 1]

    Strictly exclude SEC source events whose
    source date is >= the graph event date.
    """

    if isinstance(day, str):
        current = date.fromisoformat(day[:10])
    elif isinstance(day, date):
        current = day
    else:
        current = pd.Timestamp(day).date()

    ids = graph.ndata[dgl.NID].long().cpu().tolist()
    lookup = {
        int(nid): i
        for i, nid in enumerate(ids)
    }

    features = torch.zeros(
        len(ids), 3,
        dtype=torch.float32,
    )

    masks = torch.zeros(
        len(ids), 1,
        dtype=torch.float32,
    )

    ages = torch.full(
        (len(ids), 1),
        -1.0,
        dtype=torch.float32,
    )

    for ticker in TICKERS:
        global_id = company_ids[ticker]

        if global_id not in lookup:
            continue

        dates, rows = sec_index[ticker]

        # Latest precomputed SEC feature row
        # at or before the KG event date.
        pos = bisect.bisect_right(
            dates, current
        ) - 1

        if pos < 0:
            continue

        row = rows[pos]
        sec_date = row["sec_event_date"]

        # Recompute age for the actual graph
        # day, including weekend event dates.
        age = (current - sec_date).days

        if not (0 < age <= max_age):
            continue

        index = lookup[global_id]

        features[index] = torch.tensor(
            [
                float(row[f])
                for f in FEATURES
            ],
            dtype=torch.float32,
        )

        masks[index] = 1.0
        ages[index] = float(age)

    return (
        features.to(device),
        masks.to(device),
        ages.to(device),
    )


def edge_losses(
    graph,
    entity_memory,
    relation_memory,
    day,
    use_fusion,
):
    """
    Original FinDKG Edge Loss, with an optional
    SEC residual fusion on structural dynamic
    node embeddings.

    Base KGT, combiner, edge_model remain frozen.
    """

    batch = graph.to(device)
    batch.num_relations = R
    batch.num_all_nodes = N

    ids_gpu = batch.ndata[dgl.NID].long()
    ids_cpu = ids_gpu.cpu()

    static_batch = static_entity.structural[
        ids_gpu
    ]

    kgt_batch = (
        entity_memory.structural[
            ids_cpu
        ][:, -1, :]
        .to(device)
    )

    relation_batch = (
        relation_memory.structural[
            :, -1, :, :
        ].to(device)
    )

    sec_values, sec_masks, sec_ages = (
        asof_for_graph(day, graph)
    )

    if use_fusion:
        dynamic_batch = fusion(
            kgt_batch,
            sec_values,
            sec_masks,
            sec_ages,
        )
    else:
        dynamic_batch = kgt_batch

    combined = combiner(
        static_batch,
        dynamic_batch,
        batch,
    )

    ll = edge_model(
        batch,
        combined,
        static_emb=static_batch,
        dynamic_emb=dynamic_batch,
        dynamic_relation_emb=relation_batch,
        eid=None,
    )

    loss = -ll

    if loss.ndim != 0 or not torch.isfinite(loss).item():
        raise RuntimeError(
            f"Invalid edge loss at {day}"
        )

    active_entities = int(
        sec_masks.sum().item()
    )

    return loss, active_entities


def run_period(
    loader,
    start,
    end,
    training,
    epoch,
    max_days,
    log_every,
    score_start=None,
):
    # Fixed-weight KGT historical replay
    # starts from identical initial memory.
    entity_memory, relation_memory = reset_memory()

    updater.eval()
    combiner.eval()
    edge_model.eval()

    if training:
        fusion.train()
    else:
        fusion.eval()

    count = 0
    score_count = 0
    events = 0
    active_days = 0
    steps = 0

    total_loss = 0.0
    active_loss = 0.0

    baseline_loss = 0.0
    baseline_active_loss = 0.0

    previous_day = None

    for day, event_rows in loader.iter_event_days(
        start, end
    ):
        if max_days > 0 and score_count >= max_days:
            break

        day_string = str(day)[:10]

        if (
            previous_day is not None
            and day_string <= previous_day
        ):
            raise RuntimeError(
                "Nonchronological KG events"
            )

        graph = loader.graph_from_events(
            day,
            event_rows,
        )

        # Evaluate original baseline at the exact
        # same pre-event memory and event graph.
        with torch.no_grad():
            base_loss, _ = edge_losses(
                graph,
                entity_memory,
                relation_memory,
                day_string,
                use_fusion=False,
            )

        if training:
            optimizer.zero_grad(set_to_none=True)

            fused_loss, active = edge_losses(
                graph,
                entity_memory,
                relation_memory,
                day_string,
                use_fusion=True,
            )

            # Only update Fusion when a graph batch
            # contains at least one company with
            # an available SEC signal.
            if active > 0:
                fused_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    fusion.parameters(),
                    args.grad_clip,
                    error_if_nonfinite=True,
                )

                optimizer.step()
                steps += 1

        else:
            with torch.no_grad():
                fused_loss, active = edge_losses(
                    graph,
                    entity_memory,
                    relation_memory,
                    day_string,
                    use_fusion=True,
                )

        fused_value = float(
            fused_loss.detach().cpu()
        )

        base_value = float(
            base_loss.detach().cpu()
        )

        should_score = (
            score_start is None
            or day_string >= score_start
        )

        if should_score:
            score_count += 1
            events += len(event_rows)
            total_loss += fused_value
            baseline_loss += base_value

            if active > 0:
                active_days += 1
                active_loss += fused_value
                baseline_active_loss += base_value

        count += 1

        # IMPORTANT: update base KGT memory AFTER
        # computing current-day loss.
        #
        # Frozen base model, no gradient and no
        # mutation by SEC fusion.
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

        if (
            count == 1
            or count % log_every == 0
        ):
            label = (
                "TRAIN" if training
                else "VAL"
            )

            print(
                f"[{label}] "
                f"epoch={epoch} "
                f"day={count} "
                f"date={day_string} "
                f"edges={len(event_rows)} "
                f"active_companies={active} "
                f"base_loss={base_value:.5f} "
                f"fused_loss={fused_value:.5f}",
                flush=True,
            )

        previous_day = day_string

    if score_count == 0:
        raise RuntimeError(
            f"No scored KG event days in "
            f"{start} to {end}"
        )

    return {
        "replayed_event_days": count,
        "event_days": score_count,
        "events": events,
        "sec_active_days": active_days,
        "optimizer_steps": steps,
        "base_mean_loss": baseline_loss / score_count,
        "fused_mean_loss": total_loss / score_count,
        "base_active_mean_loss": (
            baseline_active_loss / active_days
            if active_days else None
        ),
        "fused_active_mean_loss": (
            active_loss / active_days
            if active_days else None
        ),
        "last_event_date": previous_day,
    }


def save_checkpoint(epoch, train, val, best):
    name = (
        "best_fusion.pt" if best
        else f"fusion_epoch_{epoch:03d}.pt"
    )

    path = OUT / name
    temp = OUT / (name + ".tmp")

    checkpoint = {
        "epoch": epoch,
        "fusion_state_dict": fusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_metrics": train,
        "validation_metrics": val,
        "config": vars(args),
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_sha256": sha256(
            BASE_CHECKPOINT
        ),
        "sec_features_sha256": sha256(SEC_CSV),
        "entity2id_sha256": sha256(
            KG_DIR / "entity2id.json"
        ),
        "relation2id_sha256": sha256(
            KG_DIR / "relation2id.json"
        ),
        "feature_order": FEATURES,
        "company_ids": company_ids,
        "embedding_dim": DIM,
        "trainable_module": "SECGraphFusion",
        "base_kgt_frozen": True,
        "sec_asof_strict_pit_certified": False,
        "uses_2023_for_selection": False,
    }

    torch.save(checkpoint, temp)
    os.replace(temp, path)

    return path


def write_csv(path, rows):
    if not rows:
        return

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


loader = UnifiedTemporalLoader()

if loader.num_nodes != N:
    raise RuntimeError(
        "KGT loader entity count mismatch"
    )

if loader.num_relations != R:
    raise RuntimeError(
        "KGT loader relation count mismatch"
    )

history = []
best_val = float("inf")
best_epoch = None

try:
    for epoch in range(1, args.epochs + 1):
        print()
        print("=" * 76)
        print(f"FUSION EPOCH {epoch}/{args.epochs}")
        print("=" * 76)

        t0 = time.time()

        train = run_period(
            loader,
            "2018-01-01",
            "2021-12-31",
            training=True,
            epoch=epoch,
            max_days=args.max_train_days,
            log_every=args.log_every,
        )

        if train["optimizer_steps"] == 0:
            raise RuntimeError(
                "No training graph batches had usable "
                "SEC company features. Fusion was NOT "
                "trained. Check SEC coverage and dates."
            )

        val = run_period(
            loader,
            "2018-01-01",
            "2022-12-31",
            training=False,
            epoch=epoch,
            max_days=args.max_val_days,
            log_every=args.log_every,
            score_start="2022-01-01",
        )

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train["fused_mean_loss"],
            "train_baseline_loss": train["base_mean_loss"],
            "train_active_days": train["sec_active_days"],
            "optimizer_steps": train["optimizer_steps"],
            "val_loss": val["fused_mean_loss"],
            "val_baseline_loss": val["base_mean_loss"],
            "val_active_days": val["sec_active_days"],
            "val_active_loss": val["fused_active_mean_loss"],
            "val_active_baseline_loss": val["base_active_mean_loss"],
            "elapsed_seconds": round(elapsed, 2),
        }

        history.append(row)

        write_csv(
            OUT / "training_history.csv",
            history,
        )

        save_checkpoint(epoch, train, val, best=False)

        # Select using the 2022-only SEC-active
        # event-day loss, not historical replay loss.
        # This targets the cases where SEC Fusion
        # can actually affect predictions.
        selection_loss = val["fused_active_mean_loss"]

        if selection_loss is None:
            raise RuntimeError(
                "No SEC-active events in 2022. "
                "Cannot select SEC Fusion using "
                "the intended validation metric."
            )

        if selection_loss < best_val:
            best_val = selection_loss
            best_epoch = epoch

            save_checkpoint(
                epoch,
                train,
                val,
                best=True,
            )

            print(
                "NEW BEST FUSION CHECKPOINT:",
                OUT / "best_fusion.pt",
            )

        print()
        print(
            "Epoch:", epoch,
            "Train active days:", train["sec_active_days"],
            "Optimizer steps:", train["optimizer_steps"],
            "2022 Val loss:", val["fused_mean_loss"],
            "2022 Active Val loss:", val["fused_active_mean_loss"],
            "2022 Active Baseline:", val["base_active_mean_loss"],
            "Seconds:", round(elapsed, 2),
        )

    print()
    print("=" * 76)
    print("FUSION TRAINING LOOP COMPLETED")
    print("=" * 76)

    print("Best epoch:", best_epoch)
    print("Best 2022 SEC-active validation loss:", best_val)
    print("Best checkpoint:", OUT / "best_fusion.pt")

    result = {
        "best_epoch": best_epoch,
        "best_2022_active_validation_loss": best_val,
        "best_checkpoint": str(OUT / "best_fusion.pt"),
        "base_checkpoint": str(BASE_CHECKPOINT),
        "train_period": "2018-2021",
        "validation_period": "2022",
        "test_used": False,
        "base_kgt_frozen": True,
        "strict_pit_certified": False,
        "history": history,
    }

    (OUT / "training_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("SEC FUSION TRAINING AND VALIDATION PASS")

finally:
    loader.close()
