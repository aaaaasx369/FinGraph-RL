#!/usr/bin/env python3
"""Joint SEC+KG RAG -> existing V16 PPO. Sparse E2E smoke."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ollama

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_v16_dual_rag_finrl as v16
import run_v17_unified_llm_rag as v17

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]


def save_jsonl(path, records):
    path.write_text(
        "".join(
            json.dumps(x, ensure_ascii=False, default=str) + "\n"
            for x in records
        ),
        encoding="utf-8",
    )


def chosen_dates(dates, n_train, per_split):
    picks = []
    for lo, hi in [(0, n_train), (n_train, len(dates))]:
        indices = np.arange(lo, hi)
        if len(indices) == 0:
            raise ValueError("Empty train or validation split")
        n = min(per_split, len(indices))
        positions = np.linspace(
            0, len(indices) - 1, n, dtype=int
        )
        picks.extend(indices[positions].tolist())
    return sorted(set(picks))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        default="results/aegis_rl_v1/v17_joint_rag_ppo_smoke",
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--joint-per-split", type=int, default=2)
    p.add_argument("--max-age-days", type=int, default=120)
    p.add_argument("--sec-top-k", type=int, default=5)
    p.add_argument("--kg-top-k", type=int, default=12)
    p.add_argument("--timesteps", type=int, default=1024)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--all-arms", action="store_true")

    opt, v16_options = p.parse_known_args()

    if (
        opt.joint_per_split < 1
        or opt.timesteps < 128
        or opt.max_age_days < 1
    ):
        raise ValueError("Invalid experiment settings")

    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    # Reuse the real V16 argument defaults and data paths.
    original_argv = sys.argv
    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"] + v16_options
        args = v16.parse_args()
    finally:
        sys.argv = original_argv

    args.out = out
    args.algo = "PPO"
    args.timesteps = opt.timesteps
    args.seeds = [opt.seed]
    args.sec_top_k = opt.sec_top_k
    args.kg_top_k = opt.kg_top_k

    print("========== LOAD FINRL-X ==========", flush=True)

    # Existing loader needs A2C during initialization.
    args.algo = "A2C"
    train, val, dates, contexts, n_train = (
        v16.v8b.load_data(args)
    )
    args.algo = "PPO"

    dates = v16.base.normalize_dates(dates)

    market_flat = np.concatenate(
        [train["market"], val["market"]], axis=0
    )
    prices = np.concatenate(
        [train["prices"], val["prices"]], axis=0
    )

    market, _ = v16.base.validate_market_layout(
        market_flat
    )

    assert market.shape == (len(dates), 5, 106)
    assert len(prices) == len(dates)
    assert 0 < n_train < len(dates)
    assert dates.min().year == 2018
    assert dates.max().year == 2021
    assert dates[n_train].year == 2021

    print(
        "[DATA]",
        "train=", n_train,
        "val=", len(dates) - n_train,
        flush=True,
    )

    print("========== V16 RETRIEVAL FEATURES ==========")

    sec, sec_rows = v16.v13.load_sec(
        dates,
        contexts,
        args.pit_sec,
        args.sec_top_k,
        out / "sec_retrieval_audit.csv",
    )

    kg_events = v16.v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg, relations = v16.v13.build_kg_rag(
        dates,
        kg_events,
        args.kg_top_k,
        out,
    )

    assert sec.shape == (len(dates), 5, 64)
    assert kg.shape[:2] == (len(dates), 5)

    print("========== V17 JOINT RAG ==========", flush=True)

    # Existing V17 point-in-time retrieval functions.
    sec_v17 = v17.load_sec(args.pit_sec)
    kg_v17 = v17.load_kg(args.temporal_kg)

    client = ollama.Client(
        host=opt.ollama_host,
        timeout=180,
    )

    models = [x.model for x in client.list().models]
    if opt.llm not in models:
        raise RuntimeError(
            f"Missing Ollama model: {opt.llm}; available={models}"
        )

    selected = chosen_dates(
        dates, n_train, opt.joint_per_split
    )

    print(
        "[RAG DATES]",
        [str(pd.Timestamp(dates[i]).date()) for i in selected],
        flush=True,
    )

    cache_path = out / "joint_rag_cache.jsonl"
    cache = {}

    if cache_path.exists():
        for line in cache_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                item = json.loads(line)
                cache[(item["date"], item["ticker"])] = item

    audit = []
    generated = {}

    for i in selected:
        day = pd.Timestamp(dates[i])

        for j, ticker in enumerate(TICKERS):
            key = (str(day.date()), ticker)

            sec_hits = v17.sec_retrieve(
                sec_v17,
                ticker,
                day,
                opt.sec_top_k,
            )

            kg_hits = v17.kg_retrieve(
                kg_v17,
                ticker,
                day,
                opt.kg_top_k,
            )

            if any(
                pd.Timestamp(x["available_from"]) >= day
                for x in sec_hits
            ):
                raise RuntimeError("SEC PIT leakage")

            if any(
                pd.Timestamp(x["first_observed"]) >= day
                for x in kg_hits
            ):
                raise RuntimeError("KG temporal leakage")

            if key in cache:
                record = cache[key]

            else:
                try:
                    # Actual V17 joint Qwen generation:
                    # BOTH SEC and KG retrieved context.
                    result = v17.call_llm(
                        client,
                        opt.llm,
                        ticker,
                        day,
                        sec_hits,
                        kg_hits,
                    )

                    status = "GENERATED"
                    error = ""

                except Exception as exc:
                    result = None
                    status = "ABSTAIN_ERROR"
                    error = str(exc)[:500]

                record = {
                    "date": key[0],
                    "ticker": ticker,
                    "status": status,
                    "error": error,
                    "result": result,
                    "sec_evidence": sec_hits,
                    "kg_evidence": kg_hits,
                }

                cache[key] = record
                save_jsonl(
                    cache_path, list(cache.values())
                )

            audit.append(record)
            generated[(i, j)] = record

            print(
                "[JOINT]",
                key[0],
                ticker,
                record["status"],
                flush=True,
            )

    save_jsonl(
        out / "joint_rag_audit.jsonl",
        audit,
    )

    print("========== BUILD JOINT FEATURES ==========")

    # Per stock: impact, risk, confidence,
    # valid SEC citation mask, KG citation mask,
    # and feature freshness.
    joint = np.zeros(
        (len(dates), 5, 6),
        dtype=np.float32,
    )

    latest = [None] * 5
    last_day = [None] * 5

    for t, day in enumerate(dates):
        # Do not carry a signal from the training
        # split across the validation boundary.
        if t == n_train:
            latest = [None] * 5
            last_day = [None] * 5

        for j in range(5):
            if (t, j) in generated:
                record = generated[(t, j)]
                result = record.get("result")

                # V17 verifies citation IDs, not semantic
                # entailment. Do not infer financial impact
                # solely from opaque KG edges.
                if (
                    record["status"] == "GENERATED"
                    and result is not None
                    and result.get("sec_evidence_ids")
                ):
                    latest[j] = result
                    last_day[j] = pd.Timestamp(day)
                else:
                    latest[j] = None
                    last_day[j] = None

            if latest[j] is None:
                continue

            age = (
                pd.Timestamp(day) - last_day[j]
            ).days

            if age > opt.max_age_days:
                continue

            result = latest[j]

            joint[t, j] = [
                float(result["impact"]),
                float(result["risk"]),
                float(result["confidence"]),
                1.0,
                float(bool(result.get("kg_evidence_ids"))),
                1.0 - min(
                    age / opt.max_age_days, 1.0
                ),
            ]

    if not np.isfinite(joint).all():
        raise RuntimeError("Nonfinite Joint RAG features")

    coverage = float(joint[:, :, 3].mean())

    print(
        "[JOINT COVERAGE]",
        f"{coverage:.2%}",
        flush=True,
    )

    # The same compose() and V16 PPO environment.
    datasets = {
        "MARKET": v16.base.compose(market),
        "MARKET_SEC": v16.base.compose(market, sec),
        "MARKET_KG": v16.base.compose(market, kg),
        "MARKET_SEC_KG": v16.base.compose(
            market, sec, kg
        ),
        "MARKET_SEC_KG_JOINT_RAG": v16.base.compose(
            market, sec, kg, joint
        ),
    }

    arms = (
        list(datasets)
        if opt.all_arms
        else [
            "MARKET_SEC_KG",
            "MARKET_SEC_KG_JOINT_RAG",
        ]
    )

    print("========== TRAIN V16 PPO ==========", flush=True)

    results = []

    for arm in arms:
        features = datasets[arm]

        if not np.isfinite(features).all():
            raise RuntimeError(
                f"Nonfinite features: {arm}"
            )

        print(
            "[TRAIN]",
            arm,
            "shape=", features.shape,
            flush=True,
        )

        random.seed(opt.seed)
        np.random.seed(opt.seed)

        train_env = v16.DualRAGPortfolioEnv(
            features[:n_train],
            prices[:n_train],
            dates[:n_train],
            args.transaction_cost,
            args.reward_scale,
        )

        val_env = v16.DualRAGPortfolioEnv(
            features[n_train:],
            prices[n_train:],
            dates[n_train:],
            args.transaction_cost,
            args.reward_scale,
        )

        model = v16.train_policy(
            "PPO",
            train_env,
            opt.seed,
            opt.timesteps,
            args.learning_rate,
        )

        model.save(
            str(out / f"{arm}_PPO_seed{opt.seed}")
        )

        history, metrics = v16.evaluate(
            model,
            val_env,
            arm,
            opt.seed,
            "PPO",
        )

        history.to_csv(
            out / f"{arm}_validation_2021.csv",
            index=False,
        )

        results.append(metrics)

        pd.DataFrame(results).to_csv(
            out / "portfolio_metrics.csv",
            index=False,
        )

        print(
            "[VALIDATION]",
            arm,
            "multiple=",
            metrics["final_multiple"],
            "sharpe=",
            metrics["sharpe"],
            flush=True,
        )

    summary = pd.DataFrame(results)
    summary.to_csv(
        out / "portfolio_summary.csv",
        index=False,
    )

    manifest = {
        "experiment": "V17 Joint SEC-KG RAG to V16 PPO",
        "train": "2018-2020",
        "validation": "2021",
        "test_accessed": False,
        "llm": opt.llm,
        "timesteps": opt.timesteps,
        "seed": opt.seed,
        "joint_rag_coverage": coverage,
        "joint_decision_dates": [
            str(pd.Timestamp(dates[i]).date())
            for i in selected
        ],
        "limitations": [
            "Sparse historical RAG refresh.",
            "Signals have a bounded carry-forward period.",
            "LLM citations are ID-validated, not independently entailment-verified.",
            "Opaque KG relations have no verified economic meaning.",
            "PIT of KG first-observed timestamps is not certified.",
            "Uses existing V16 experimental SB3 portfolio environment.",
            "Validation is not a blind test.",
            "No 2022-2023 test data used.",
        ],
    }

    (out / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== JOINT RAG PPO SUMMARY ==========")
    print(summary.to_string(index=False))
    print("\nJOINT RAG PPO SMOKE: PASS", flush=True)


if __name__ == "__main__":
    main()
