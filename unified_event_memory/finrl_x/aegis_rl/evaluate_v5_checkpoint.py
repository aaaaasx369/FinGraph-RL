#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate already-trained FinRL-X V5 checkpoint.

No training, no weight updates.

Loads:
    - V5 model.zip
    - V5 router.pt
    - Market + Dynamic KGT 200D
    - Temporal KG-RAG
    - SEC-RAG
    - V4 FLARE / Evidence Critic

Evaluates:
    Test: 2022-2023

Outputs:
    test_queries.csv
    test_result.json
    evaluation_status.json

Does not overwrite existing training checkpoints.
"""

import sys
import json
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import A2C

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v2 as v2,
    run_active_rag_v3 as v3,
    run_active_rag_v4 as v4,
)


def log(*args):
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        *args,
        flush=True,
    )


class InitializationRouter:
    """
    Prevent retrieval during environment __init__.

    After initialization, delegate all actions
    to the actual trained V5 retrieval router.
    """

    def __init__(self):
        self.delegate = None
        self.pending = []
        self.training = False

    def select(self, features):
        if self.delegate is None:
            return (
                "STOP",
                v2.TICKERS[0],
                v2.RELATIONS[0],
            )

        return self.delegate.select(features)

    def update(self, reward):
        if self.delegate is not None:
            return self.delegate.update(reward)

        self.pending.clear()


def save_json(path, obj):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=(
            "results/aegis_rl_v1/"
            "active_rag_v5_smoke"
        ),
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
        "--llm",
        default="qwen2.5:3b",
    )

    parser.add_argument(
        "--embed",
        default="bge-m3:latest",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--cost-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--tier",
        default="conservative",
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if test_result.json exists",
    )

    args = parser.parse_args()

    run_root = Path(args.root).resolve()

    folder = (
        run_root
        / "runs"
        / "ACTIVE__A2C__seed42"
    )

    model_path = folder / "model.zip"
    router_path = folder / "router.pt"

    test_csv = folder / "test_queries.csv"
    result_path = folder / "test_result.json"
    status_path = folder / "evaluation_status.json"

    for path in [model_path, router_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required checkpoint missing: {path}"
            )

    if result_path.exists() and not args.force:
        log("[ALREADY COMPLETED]", result_path)

        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )

        return

    log("=" * 72)
    log("FINRL-X V5 CHECKPOINT TEST EVALUATION")
    log("=" * 72)

    log("[MODEL]", model_path)
    log("[ROUTER]", router_path)

    save_json(
        status_path,
        {
            "status": "loading",
            "model": str(model_path),
            "router": str(router_path),
            "split": "test",
        },
    )

    # -------------------------------------------------
    # 1. Install V4 FLARE + Self-RAG adapted components
    # -------------------------------------------------

    log("[INSTALL] V4 Retrieval Extensions")

    v4.install_v4()

    # -------------------------------------------------
    # 2. Load market and Dynamic KGT
    # -------------------------------------------------

    log("[LOAD] Market Data")

    _, _, schema, market, _, _ = (
        v2.v2.read_original_data(args)
    )

    market = market.copy()

    market["date"] = pd.to_datetime(
        market["date"]
    )

    dates = sorted(
        market["date"].unique()
    )

    log("[LOAD] Dynamic KGT")

    kgt, kgt_cols = v2.legacy.load_kgt(
        args.kgt,
        dates,
    )

    if len(kgt_cols) != 200:
        raise RuntimeError(
            "Expected 200D Dynamic KGT, "
            f"got {len(kgt_cols)}"
        )

    base_cols = schema["FULL_DYNAMIC"]

    # -------------------------------------------------
    # 3. Load KG / SEC
    # -------------------------------------------------

    log("[LOAD] KG and SEC")

    kg = v2.legacy.load_kg(
        args.kg
    )

    sec = v2.legacy.load_sec(
        args.sec
    )

    log(
        "[DATA]",
        "KG rows:", len(kg),
        "SEC chunks:", len(sec),
        "KGT dims:", len(kgt_cols),
    )

    # -------------------------------------------------
    # 4. Ollama / Embedding cache
    # -------------------------------------------------

    llm = v2.CachedOllama(
        args.llm,
        args.embed,
        run_root / "llm_cache.sqlite",
        args.device,
        48,
        args.ollama_host,
    )

    retriever = v2.IndexedTemporalRetrieval(
        kg,
        sec,
        llm,
        args.top_k,
    )

    # -------------------------------------------------
    # 5. Load trained V5 Router
    # -------------------------------------------------

    log("[LOAD] Trained Retrieval Router")

    router = v3.ContextualRouter(
        state_dim=14,
        lr=1e-4,
        entropy_weight=0.01,
        training=False,
    )

    router.load(
        router_path
    )

    router.training = False

    log(
        "[ROUTER]",
        "updates:",
        router.update_count,
    )

    # -------------------------------------------------
    # 6. Construct environment safely
    # -------------------------------------------------

    test_market = market[
        market.split.eq("test")
    ].copy()

    if test_market.empty:
        raise RuntimeError(
            "Test market split is empty"
        )

    log(
        "[BUILD] Test Environment",
        "market rows:",
        len(test_market),
    )

    init_router = InitializationRouter()

    env = v3.ContextualActiveRAGEnv(
        test_market,
        base_cols,
        kgt,
        retriever,
        llm,
        "ACTIVE",
        args.max_search_steps,
        args.transaction_cost,
        args.cost_scale,
        "test",
        router=init_router,
    )

    # Enable trained router only after __init__.
    init_router.delegate = router

    # -------------------------------------------------
    # 7. Load frozen V5 A2C Portfolio Actor
    # -------------------------------------------------

    log("[LOAD] Frozen V5 A2C")

    model = A2C.load(
        str(model_path),
        device=args.device,
    )

    extractor = (
        model.policy.features_extractor
    )

    required = [
        "market_encoder",
        "kg_encoder",
        "sec_encoder",
        "market_evidence",
        "gate",
    ]

    missing = [
        name
        for name in required
        if not hasattr(extractor, name)
    ]

    if missing:
        raise RuntimeError(
            "FATAL: Gated Fusion was not "
            f"loaded correctly: {missing}"
        )

    expected_obs = (
        model.observation_space.shape
    )

    actual_obs = (
        env.observation_space.shape
    )

    if expected_obs != actual_obs:
        raise RuntimeError(
            "Observation shape mismatch: "
            f"model={expected_obs}, "
            f"environment={actual_obs}"
        )

    log(
        "[FUSION]",
        type(extractor).__name__,
        "observation:",
        expected_obs,
    )

    if expected_obs != (3323,):
        raise RuntimeError(
            "Expected 3323D observation"
        )

    # Ensure evaluation mode.
    model.policy.set_training_mode(False)

    save_json(
        status_path,
        {
            "status": "evaluating",
            "split": "test",
            "model": str(model_path),
            "router": str(router_path),
            "observation_shape": expected_obs,
            "router_updates": router.update_count,
        },
    )

    # -------------------------------------------------
    # 8. Test Evaluation
    # -------------------------------------------------

    log("[EVAL START] 2022-2023 Test")

    start = time.time()

    metrics, history = v2.evaluate_model(
        model,
        env,
    )

    elapsed = time.time() - start

    # -------------------------------------------------
    # 9. Save results
    # -------------------------------------------------

    if history.empty:
        raise RuntimeError(
            "Evaluation returned empty history"
        )

    history.to_csv(
        test_csv,
        index=False,
    )

    output = {
        "mode": "ACTIVE",
        "agent": "A2C",
        "seed": 42,
        "split": "test",
        "timesteps_trained": 250,
        **metrics,
        "evaluation_seconds": elapsed,
        "evaluation_days": len(history),
    }

    save_json(
        result_path,
        output,
    )

    save_json(
        status_path,
        {
            "status": "completed",
            **output,
        },
    )

    log("=" * 72)
    log("[EVAL COMPLETE] V5 A2C TEST")
    log("=" * 72)

    print(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )

    log("[SAVED]", test_csv)
    log("[SAVED]", result_path)

    log("FINRL-X V5 TEST EVALUATION: PASS")


if __name__ == "__main__":
    main()
