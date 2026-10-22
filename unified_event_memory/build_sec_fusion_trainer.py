#!/usr/bin/env python3

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "unified_event_memory"

FORWARD = BASE / "smoke_original_kgt.py"
EDGE = BASE / "smoke_kgt_edge_loss.py"
OUTPUT = BASE / "sec_fusion/train_sec_fusion.py"

for path in (FORWARD, EDGE):
    if not path.exists():
        raise FileNotFoundError(path)

source = FORWARD.read_text(encoding="utf-8")

marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if source.count(marker) != 1:
    raise RuntimeError("Verified KGT prefix not found")

prefix = source.split(marker, 1)[0]

edge_source = EDGE.read_text(encoding="utf-8")
tree = ast.parse(edge_source)

matches = [
    n for n in tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "read_original_setting"
]

if len(matches) != 1:
    raise RuntimeError(
        "Original configuration resolver not found"
    )

resolver = ast.get_source_segment(
    edge_source,
    matches[0],
)

body = r'''
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
        if max_days > 0 and count >= max_days:
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

        total_loss += fused_value
        baseline_loss += base_value

        if active > 0:
            active_days += 1
            active_loss += fused_value
            baseline_active_loss += base_value

        events += len(event_rows)
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

    if count == 0:
        raise RuntimeError(
            f"No KG events in {start} to {end}"
        )

    return {
        "event_days": count,
        "events": events,
        "sec_active_days": active_days,
        "optimizer_steps": steps,
        "base_mean_loss": baseline_loss / count,
        "fused_mean_loss": total_loss / count,
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
            max_days=0,
            log_every=args.log_every,
        )

        # Validation replay includes 2018-2021 history
        # to rebuild memory at fixed KGT weights.
        # For model selection, score ONLY 2022 below.
        #
        # The replay result above includes 2018-2021
        # losses too, so it must not be used directly
        # as the validation metric.
        #
        # A split-specific scoring function is used
        # in the next section of this script.

        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "train_loss": train["fused_mean_loss"],
            "train_baseline_loss": train["base_mean_loss"],
            "train_active_days": train["sec_active_days"],
            "optimizer_steps": train["optimizer_steps"],
            "replay_loss": val["fused_mean_loss"],
            "elapsed_seconds": round(elapsed, 2),
        }

        history.append(row)

        write_csv(
            OUT / "training_history.csv",
            history,
        )

        save_checkpoint(epoch, train, val, best=False)

        print()
        print(
            "Epoch:", epoch,
            "Train active days:", train["sec_active_days"],
            "Optimizer steps:", train["optimizer_steps"],
            "Replay loss:", val["fused_mean_loss"],
            "Seconds:", round(elapsed, 2),
        )

    print()
    print("=" * 76)
    print("FUSION TRAINING LOOP COMPLETED")
    print("=" * 76)
    print(
        "NOTE: Replay metric above is NOT a "
        "2022-only validation metric."
    )
    print(
        "NOTE: Do not select best checkpoint "
        "from combined replay loss."
    )

finally:
    loader.close()
'''

result = prefix + "\n" + resolver + "\n" + body

compile(result, str(OUTPUT), "exec")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(result, encoding="utf-8")

print("TRAINER GENERATED:", OUTPUT)
