#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V5: Evidence-Market Alignment

Commands:
    prepare   Generate PIT-filtered historical samples
    pretrain  Train V4 GatedEvidenceExtractor
    finetune  Initialize V4 Portfolio RL with pretrained weights

Research design:
    Train: 2018–2020
    Validation: 2021
    Test: 2022–2023 (never used during alignment)

Targets:
    Next-trading-day stock return relative to
    the five-stock cross-sectional median.

This is a supervised auxiliary task.
Predictive improvement is not guaranteed.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v2 as v2,
    run_active_rag_v3 as v3,
    run_active_rag_v4 as v4,
)

TICKERS = list(v2.TICKERS)

FEATURE_DIM = 1265
EVIDENCE_DIM = 1024
EXTRA_DIM = 10
OBS_DIM = FEATURE_DIM + 2 * EVIDENCE_DIM + EXTRA_DIM

DEFAULT_ROOT = (
    "results/aegis_rl_v1/evidence_alignment_v5"
)


def log(*args):
    print(*args, flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_observation(features, evidence):
    """
    V4 observation format:

    Market/KGT 1265D
    KG evidence 1024D
    SEC evidence 1024D
    Portfolio weights 5D
    Environment extras 5D
    """
    market = np.asarray(
        features, dtype=np.float32
    ).reshape(-1)

    ev = np.asarray(
        evidence, dtype=np.float32
    )

    if market.shape != (FEATURE_DIM,):
        raise RuntimeError(
            f"Invalid market feature shape: {market.shape}"
        )

    if ev.shape != (2, EVIDENCE_DIM):
        raise RuntimeError(
            f"Invalid evidence shape: {ev.shape}"
        )

    extras = np.array(
        [
            0.2, 0.2, 0.2, 0.2, 0.2,
            1.0, 1.0, 0.0, 1.0, 0.0,
        ],
        dtype=np.float32,
    )

    observation = np.concatenate(
        [market, ev.reshape(-1), extras]
    )

    if observation.shape != (OBS_DIM,):
        raise RuntimeError(
            "Observation dimension mismatch"
        )

    return np.nan_to_num(
        observation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


class StopRouter:
    """
    Never retrieve during environment construction.

    Historical evidence is retrieved explicitly
    using the indexed retriever.
    """

    def __init__(self):
        self.pending = []
        self.training = False

    def select(self, state):
        return (
            "STOP",
            TICKERS[0],
            v2.RELATIONS[0],
        )

    def update(self, reward):
        pass


def load_research_data(args):
    log("[LOAD] Historical market data")

    _, _, schema, market, _, _ = (
        v2.v2.read_original_data(args)
    )

    market = market.copy()
    market["date"] = pd.to_datetime(market["date"])

    dates = sorted(market["date"].unique())

    kgt, columns = v2.legacy.load_kgt(
        args.kgt, dates
    )

    if len(columns) != 200:
        raise RuntimeError(
            f"Expected 200D KGT, got {len(columns)}"
        )

    kg = v2.legacy.load_kg(args.kg)
    sec = v2.legacy.load_sec(args.sec)

    llm = v2.CachedOllama(
        args.llm,
        args.embed,
        str(args.root / "llm_cache.sqlite"),
        "cpu",
        48,
        args.ollama_host,
    )

    retriever = v2.IndexedTemporalRetrieval(
        kg, sec, llm, args.top_k
    )

    return (
        schema,
        market,
        kgt,
        retriever,
    )


def get_environment(args, split, schema, market, kgt, retriever):
    """
    Build an environment only to obtain the same
    feature matrix, price array, and date ordering
    as the original Portfolio RL.
    """
    router = StopRouter()

    # The STOP router avoids expensive LLM generation.
    env = v3.ContextualActiveRAGEnv(
        market[market.split.eq(split)],
        schema["FULL_DYNAMIC"],
        kgt,
        retriever,
        retriever.llm,
        "NEVER",
        1,
        args.transaction_cost,
        args.cost_scale,
        split,
        router=router,
    )

    return env


def prepare(args):
    if args.split not in ("train", "val"):
        raise RuntimeError(
            "Alignment data may only be prepared "
            "from train or validation split"
        )

    schema, market, kgt, retriever = (
        load_research_data(args)
    )

    env = get_environment(
        args,
        args.split,
        schema,
        market,
        kgt,
        retriever,
    )

    features = np.asarray(env.features)
    prices = np.asarray(env.price)

    if features.ndim != 3:
        raise RuntimeError(
            f"Unexpected feature tensor: {features.shape}"
        )

    if features.shape[1:] != (5, 253):
        raise RuntimeError(
            "Expected (days, 5, 253) features, "
            f"got {features.shape}"
        )

    if prices.ndim != 2 or prices.shape[1] != 5:
        raise RuntimeError(
            f"Expected (days,5) price matrix: {prices.shape}"
        )

    n = min(
        len(features),
        len(prices),
        len(env.dates),
    )

    # Never create a label across split boundaries.
    valid_days = list(
        range(0, n - 1, args.stride)
    )

    if args.max_days > 0:
        valid_days = valid_days[:args.max_days]

    observations = []
    ticker_indices = []
    targets = []
    dates = []
    meta = []

    log(
        f"[PREPARE] split={args.split}, "
        f"days={len(valid_days)}, "
        f"retrievals={len(valid_days)*len(TICKERS)}"
    )

    for count, day_idx in enumerate(valid_days):
        day = pd.Timestamp(env.dates[day_idx])

        if args.split == "train" and day.year > 2020:
            raise RuntimeError(
                f"Train split leakage: {day}"
            )

        if args.split == "val" and day.year != 2021:
            raise RuntimeError(
                f"Validation split mismatch: {day}"
            )

        today_price = prices[day_idx]
        next_price = prices[day_idx + 1]

        if not (
            np.isfinite(today_price).all()
            and np.isfinite(next_price).all()
            and (today_price > 0).all()
        ):
            continue

        next_returns = (
            next_price / today_price - 1.0
        )

        # Cross-sectional excess return:
        # remove broad common market movement.
        median_return = float(
            np.median(next_returns)
        )

        market_state = (
            features[day_idx].reshape(-1)
        )

        for ticker_idx, ticker in enumerate(TICKERS):
            # Fixed deterministic question.
            # No post-decision LLM generation.
            question = (
                f"{ticker} financial risk supply chain "
                "customers competitors SEC filings"
            )

            vector, provenance = retriever.retrieve(
                "BOTH",
                ticker,
                "SUPPLIES_TO",
                day,
                question,
            )

            vector = np.asarray(
                vector,
                dtype=np.float32,
            )

            if vector.shape != (2, EVIDENCE_DIM):
                raise RuntimeError(
                    f"Incorrect evidence vector: "
                    f"{vector.shape}"
                )

            # PIT audit every retrieved document.
            for item in provenance:
                available = pd.Timestamp(
                    item["available_at"]
                )

                if available >= day:
                    raise RuntimeError(
                        "LOOK-AHEAD VIOLATION: "
                        f"{available} >= {day}"
                    )

            label = float(
                next_returns[ticker_idx]
                > median_return
            )

            observations.append(
                make_observation(
                    market_state,
                    vector,
                )
            )

            ticker_indices.append(
                ticker_idx
            )

            targets.append(label)
            dates.append(str(day.date()))

            meta.append({
                "date": str(day.date()),
                "ticker": ticker,
                "label": label,
                "next_return": float(
                    next_returns[ticker_idx]
                ),
                "KG_hits": sum(
                    x["source"] == "KG"
                    for x in provenance
                ),
                "SEC_hits": sum(
                    x["source"] == "SEC"
                    for x in provenance
                ),
            })

        if (count + 1) % 10 == 0:
            log(
                f"[PREPARE] {count+1}/"
                f"{len(valid_days)} days"
            )

    if not observations:
        raise RuntimeError(
            "No usable training samples"
        )

    args.root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Include sampling controls in dataset filename,
    # preventing smoke/full datasets being confused.
    tag = (
        f"{args.split}_stride{args.stride}"
        f"_days{args.max_days}"
    )

    path = args.root / f"{tag}.npz"

    np.savez_compressed(
        path,
        x=np.stack(observations),
        ticker=np.asarray(
            ticker_indices,
            dtype=np.int64,
        ),
        y=np.asarray(
            targets,
            dtype=np.float32,
        ),
        dates=np.asarray(dates),
    )

    pd.DataFrame(meta).to_csv(
        args.root / f"{tag}_audit.csv",
        index=False,
    )

    log("[SAVED]", path)
    log("[SAMPLES]", len(observations))
    log("[POSITIVE RATE]", np.mean(targets))


class AlignmentNetwork(nn.Module):
    """
    V4 feature extractor + supervised prediction head.

    All V4 encoder/gate parameters receive gradients
    through the auxiliary prediction loss.
    """

    def __init__(self):
        super().__init__()

        observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        self.extractor = (
            v4.GatedEvidenceExtractor(
                observation_space,
                evidence_dim=EVIDENCE_DIM,
            )
        )

        self.head = nn.Sequential(
            nn.Linear(
                self.extractor.features_dim + 5,
                64,
            ),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, observation, ticker):
        h = self.extractor(observation)

        onehot = F.one_hot(
            ticker.long(),
            num_classes=5,
        ).float()

        logits = self.head(
            torch.cat(
                [h, onehot],
                dim=1,
            )
        )

        return logits.flatten()


def load_dataset(path):
    data = np.load(path)

    x = torch.tensor(
        data["x"],
        dtype=torch.float32,
    )

    ticker = torch.tensor(
        data["ticker"],
        dtype=torch.long,
    )

    y = torch.tensor(
        data["y"],
        dtype=torch.float32,
    )

    if x.ndim != 2 or x.shape[1] != OBS_DIM:
        raise RuntimeError(
            f"Wrong alignment observation shape: {x.shape}"
        )

    return x, ticker, y


@torch.no_grad()
def evaluate_alignment(model, dataset, batch_size=128):
    model.eval()

    x, ticker, y = dataset

    probabilities = []
    losses = []

    for a in range(0, len(x), batch_size):
        xb = x[a:a+batch_size]
        tb = ticker[a:a+batch_size]
        yb = y[a:a+batch_size]

        logits = model(xb, tb)

        loss = F.binary_cross_entropy_with_logits(
            logits, yb
        )

        losses.append(
            (float(loss.item()), len(xb))
        )

        probabilities.extend(
            torch.sigmoid(logits).cpu().tolist()
        )

    prob = np.asarray(probabilities)
    truth = y.numpy()

    accuracy = float(
        np.mean((prob >= 0.5) == truth)
    )

    mean_loss = sum(
        value * count for value, count in losses
    ) / max(1, sum(count for _, count in losses))

    return {
        "loss": mean_loss,
        "accuracy": accuracy,
        "majority_accuracy": float(
            max(truth.mean(), 1-truth.mean())
        ),
    }


def pretrain(args):
    if not args.train_data or not args.val_data:
        raise RuntimeError(
            "--train-data and --val-data required"
        )

    set_seed(args.seed)

    train = load_dataset(args.train_data)
    val = load_dataset(args.val_data)

    # Validate date separation from NPZ metadata.
    train_dates = pd.to_datetime(
        np.load(args.train_data)["dates"]
    )

    val_dates = pd.to_datetime(
        np.load(args.val_data)["dates"]
    )

    if not all(d.year <= 2020 for d in train_dates):
        raise RuntimeError("Train date leakage")

    if not all(d.year == 2021 for d in val_dates):
        raise RuntimeError("Validation date leakage")

    model = AlignmentNetwork()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    loader = DataLoader(
        TensorDataset(*train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    best = float("inf")
    best_path = args.root / "aligned_extractor.pt"
    args.root.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        loss_total = 0.0
        count = 0

        for x, ticker, y in loader:
            logits = model(x, ticker)

            loss = F.binary_cross_entropy_with_logits(
                logits, y
            )

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            loss_total += float(loss.item()) * len(x)
            count += len(x)

        metrics = evaluate_alignment(model, val)

        row = {
            "epoch": epoch,
            "train_loss": loss_total / max(count,1),
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "majority_accuracy":
                metrics["majority_accuracy"],
        }

        history.append(row)

        log("[EPOCH]", row)

        if metrics["loss"] < best:
            best = metrics["loss"]

            torch.save({
                "extractor":
                    model.extractor.state_dict(),
                "observation_dim": OBS_DIM,
                "evidence_dim": EVIDENCE_DIM,
                "train_data": str(args.train_data),
                "val_data": str(args.val_data),
                "best_val_loss": best,
                "epoch": epoch,
                "seed": args.seed,
            }, best_path)

    pd.DataFrame(history).to_csv(
        args.root / "alignment_history.csv",
        index=False,
    )

    log("[BEST EXTRACTOR]", best_path)
    log("[BEST VALIDATION LOSS]", best)


def finetune(args, remaining_args):
    """
    Start a new V4 RL training run initialized
    from aligned feature extractor weights.

    A fresh output folder is mandatory.
    No old V4 checkpoint may be silently reused.
    """

    if not args.align_weights:
        raise RuntimeError(
            "--align-weights is required"
        )

    checkpoint = torch.load(
        args.align_weights,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["observation_dim"] != OBS_DIM:
        raise RuntimeError(
            "Pretrained observation mismatch"
        )

    state = checkpoint["extractor"]

    # Install V4 FLARE + critic + gated policy builder.
    v4.install_v4()

    original_builder = v2.create_portfolio_model

    def aligned_builder(agent, env, seed, runner_args):
        model = original_builder(
            agent, env, seed, runner_args
        )

        extractors = [
            module
            for module in model.policy.modules()
            if (
                type(module).__name__
                == "GatedEvidenceExtractor"
                and hasattr(module, "gate")
                and hasattr(module, "kg_encoder")
                and hasattr(module, "sec_encoder")
            )
        ]

        # Some off-policy architectures can reference
        # multiple extractor modules.
        unique = list({
            id(module): module
            for module in extractors
        }.values())

        if not unique:
            raise RuntimeError(
                "FATAL: Gated Fusion not installed. "
                "Refusing to train."
            )

        for extractor in unique:
            extractor.load_state_dict(
                state, strict=True
            )

        log(
            "[ALIGNMENT WEIGHTS LOADED]",
            agent,
            "extractors:",
            len(unique),
            "val_loss:",
            checkpoint["best_val_loss"],
        )

        return model

    # V3 main delegates model creation to V2 main.
    v2.create_portfolio_model = aligned_builder

    # Ensure V3 can parse only its own arguments.
    sys.argv = [
        sys.argv[0],
        *remaining_args,
    ]

    if "--out" not in remaining_args:
        raise RuntimeError(
            "Use an explicit NEW --out directory"
        )

    out = Path(
        remaining_args[
            remaining_args.index("--out") + 1
        ]
    )

    # Prevent skipping pretrained initialization
    # through previously completed/checkpointed runs.
    if (out / "runs").exists():
        raise RuntimeError(
            "Output already contains runs. "
            "Choose a new --out directory."
        )

    v3.main()


def main():
    parser = argparse.ArgumentParser(
        description="FinRL-X Evidence Alignment V5"
    )

    parser.add_argument(
        "command",
        choices=["prepare","pretrain","finetune"],
    )

    parser.add_argument(
        "--root", type=Path, default=Path(DEFAULT_ROOT)
    )

    parser.add_argument(
        "--split", default="train"
    )

    parser.add_argument(
        "--stride", type=int, default=5
    )

    parser.add_argument(
        "--max-days", type=int, default=0
    )

    parser.add_argument(
        "--seed", type=int, default=42
    )

    parser.add_argument(
        "--epochs", type=int, default=30
    )

    parser.add_argument(
        "--batch-size", type=int, default=64
    )

    parser.add_argument(
        "--lr", type=float, default=1e-4
    )

    parser.add_argument(
        "--train-data", type=Path
    )

    parser.add_argument(
        "--val-data", type=Path
    )

    parser.add_argument(
        "--align-weights", type=Path
    )

    parser.add_argument(
        "--kgt",
        default=(
            "outputs/aegis_kgt_daily_2018_2023/"
            "daily_dynamic_kgt.csv"
        ),
    )

    parser.add_argument(
        "--kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        ),
    )

    parser.add_argument(
        "--sec",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )

    parser.add_argument(
        "--llm", default="qwen2.5:3b"
    )

    parser.add_argument(
        "--embed", default="bge-m3:latest"
    )

    parser.add_argument(
        "--top-k", type=int, default=3
    )

    parser.add_argument(
        "--transaction-cost", type=float, default=0.001
    )

    parser.add_argument(
        "--cost-scale", type=float, default=1.0
    )

    parser.add_argument(
        "--reward-scaling", type=float, default=1.0
    )

    parser.add_argument(
        "--tier", default="conservative"
    )

    args, remaining = parser.parse_known_args()

    args.root = args.root.resolve()

    if args.command == "prepare":
        prepare(args)

    elif args.command == "pretrain":
        pretrain(args)

    elif args.command == "finetune":
        finetune(args, remaining)


if __name__ == "__main__":
    main()
