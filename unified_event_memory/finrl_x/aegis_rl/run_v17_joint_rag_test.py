#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V17 Frozen PPO Test Runner

2018-2020: trained earlier
2021:      validation completed earlier
2022-2023: provisional out-of-sample evaluation

Reuses:
- V16 original data loader
- V16 market layout and feature encoding
- V16 SEC/KG retrieval encoders
- V16 DualRAGPortfolioEnv and evaluate()
- V17 SEC/KG point-in-time retrieval
- V17 joint Qwen generation

Does NOT train or update PPO parameters.

The test data loader MUST truly return
2022-2023 dates. Otherwise fail closed.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ollama
from stable_baselines3 import PPO

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_v16_dual_rag_finrl as v16
import run_v17_unified_llm_rag as v17


TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

ARMS = [
    "MARKET",
    "MARKET_SEC",
    "MARKET_KG",
    "MARKET_SEC_KG",
    "MARKET_SEC_KG_JOINT_RAG",
]

TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2023-12-31")


def say(*args):
    print(*args, flush=True)


def stable_hash(data):
    return hashlib.sha256(
        json.dumps(
            data,
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def save_jsonl(path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(
            json.dumps(
                item,
                ensure_ascii=False,
                default=str,
            ) + "\n"
            for item in records
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def get_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--models-dir",
        default=(
            "results/aegis_rl_v1/"
            "v17_joint_rag_ppo_smoke"
        ),
    )
    p.add_argument(
        "--out",
        default=(
            "results/aegis_rl_v1/"
            "v17_joint_rag_test_2022_2023"
        ),
    )
    p.add_argument(
        "--ollama-host",
        default="http://140.113.87.164:11434",
    )
    p.add_argument(
        "--llm",
        default="qwen2.5:3b",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 2026],
    )
    p.add_argument(
        "--joint-refresh-dates",
        type=int,
        default=2,
    )
    p.add_argument(
        "--max-age-days",
        type=int,
        default=120,
    )
    p.add_argument(
        "--sec-top-k",
        type=int,
        default=5,
    )
    p.add_argument(
        "--kg-top-k",
        type=int,
        default=12,
    )
    p.add_argument(
        "--market-only-check",
        action="store_true",
        help="Test loader and market-only model first",
    )

    args, remaining = p.parse_known_args()

    if (
        args.joint_refresh_dates < 1
        or args.max_age_days < 1
        or args.sec_top_k < 1
        or args.kg_top_k < 1
    ):
        raise ValueError("Invalid RAG settings")

    return args, remaining


def load_test_data(extra, out):
    """
    Ask the original loader for its test split.
    Never relabel validation observations as test.
    """

    original_argv = sys.argv

    try:
        sys.argv = [
            "run_v16_dual_rag_finrl.py"
        ] + extra

        base_args = v16.parse_args()

    finally:
        sys.argv = original_argv

    base_args.algo = "A2C"
    base_args.split = "test"
    base_args.out = out / "loader"

    base_args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    if hasattr(base_args, "max_days"):
        base_args.max_days = 0

    say("[LOADER] Requesting split=test")

    try:
        train, val, dates, contexts, n_train = (
            v16.v8b.load_data(base_args)
        )

    except Exception as exc:
        raise RuntimeError(
            "The existing v8b.load_data(args) "
            "could not load the test split. "
            "Do not replace it with 2021 data. "
            "Original loader error: "
            + repr(exc)
        ) from exc

    dates = v16.base.normalize_dates(dates)

    market_flat = np.concatenate(
        [
            train["market"],
            val["market"],
        ],
        axis=0,
    )

    prices = np.concatenate(
        [
            train["prices"],
            val["prices"],
        ],
        axis=0,
    )

    market, _ = (
        v16.base.validate_market_layout(
            market_flat
        )
    )

    if not (
        len(dates)
        == len(prices)
        == len(market)
    ):
        raise RuntimeError(
            "Test date/market/price mismatch"
        )

    say(
        "[LOADER RETURNED]",
        len(dates),
        str(dates.min()),
        str(dates.max()),
    )

    # The data source must be an actual test
    # dataset, not the old 2018-2021 split.
    if not (
        dates.min() >= TEST_START
        and dates.max() <= TEST_END
        and dates.min().year == 2022
        and dates.max().year == 2023
    ):
        raise RuntimeError(
            "TEST_LOADER_NOT_READY: "
            "Requested split=test but the "
            "loader did not return actual "
            "2022-2023 observations. "
            f"Returned range: {dates.min()} "
            f"to {dates.max()}. "
            "The original test data loader "
            "must be fixed before continuing."
        )

    if market.shape[1:] != (5, 106):
        raise RuntimeError(
            "Unexpected V16 market dimension: "
            + str(market.shape)
        )

    if len(dates) < 300:
        raise RuntimeError(
            "Too few 2022-2023 test dates"
        )

    if not np.isfinite(
        market
    ).all():
        raise RuntimeError(
            "Nonfinite test market features"
        )

    if not np.isfinite(
        prices
    ).all():
        raise RuntimeError(
            "Nonfinite test prices"
        )

    return (
        base_args,
        dates,
        contexts,
        market,
        prices,
    )


def refresh_positions(dates, count):
    n = min(
        len(dates),
        count,
    )

    return sorted(set(
        np.linspace(
            0,
            len(dates) - 1,
            n,
            dtype=int,
        ).tolist()
    ))


def build_joint_features(
    args,
    base_args,
    dates,
    out,
):
    sec_raw = v17.load_sec(
        base_args.pit_sec
    )

    kg_raw = v17.load_kg(
        base_args.temporal_kg
    )

    client = ollama.Client(
        host=args.ollama_host,
        timeout=180,
    )

    models = {
        model.model
        for model in client.list().models
    }

    if args.llm not in models:
        raise RuntimeError(
            "Ollama model not available: "
            + args.llm
        )

    positions = refresh_positions(
        dates,
        args.joint_refresh_dates,
    )

    cache_path = out / "test_joint_cache.jsonl"

    cache = {}

    if cache_path.exists():
        for line in cache_path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue

            record = json.loads(line)
            cache[record["cache_key"]] = record

    generated = {}
    audit = []

    for index in positions:
        day = pd.Timestamp(
            dates[index]
        )

        for j, ticker in enumerate(TICKERS):
            sec_hits = v17.sec_retrieve(
                sec_raw,
                ticker,
                day,
                args.sec_top_k,
            )

            kg_hits = v17.kg_retrieve(
                kg_raw,
                ticker,
                day,
                args.kg_top_k,
            )

            for hit in sec_hits:
                if (
                    pd.Timestamp(
                        hit["available_from"]
                    ) >= day
                ):
                    raise RuntimeError(
                        "SEC PIT violation"
                    )

            for hit in kg_hits:
                if (
                    pd.Timestamp(
                        hit["first_observed"]
                    ) >= day
                ):
                    raise RuntimeError(
                        "KG temporal violation"
                    )

            identity = stable_hash({
                "ticker": ticker,
                "date": str(day.date()),
                "llm": args.llm,
                "host": args.ollama_host,
                "sec": sec_hits,
                "kg": kg_hits,
                "sec_top_k": args.sec_top_k,
                "kg_top_k": args.kg_top_k,
            })

            if identity in cache:
                record = cache[identity]
            else:
                try:
                    result = v17.call_llm(
                        client,
                        args.llm,
                        ticker,
                        day,
                        sec_hits,
                        kg_hits,
                    )

                    status = "GENERATED"
                    error = None

                except Exception as exc:
                    result = None
                    status = "ABSTAIN_ERROR"
                    error = str(exc)[:500]

                record = {
                    "cache_key": identity,
                    "date": str(day.date()),
                    "ticker": ticker,
                    "status": status,
                    "error": error,
                    "result": result,
                    "sec_evidence": sec_hits,
                    "kg_evidence": kg_hits,
                }

                cache[identity] = record
                save_jsonl(
                    cache_path,
                    list(cache.values()),
                )

            generated[(index, j)] = record
            audit.append(record)

            say(
                "[TEST RAG]",
                day.date(),
                ticker,
                record["status"],
            )

    save_jsonl(
        out / "test_joint_rag_audit.jsonl",
        audit,
    )

    # Identical six-dimensional feature layout
    # to the existing train/val Joint RAG model.
    joint = np.zeros(
        (len(dates), 5, 6),
        dtype=np.float32,
    )

    latest = [None] * 5
    last_day = [None] * 5

    for i, day in enumerate(dates):
        day = pd.Timestamp(day)

        for j in range(5):
            if (i, j) in generated:
                record = generated[(i, j)]
                result = record.get("result")

                if (
                    record["status"] == "GENERATED"
                    and result is not None
                    and result.get(
                        "sec_evidence_ids"
                    )
                ):
                    latest[j] = result
                    last_day[j] = day
                else:
                    latest[j] = None
                    last_day[j] = None

            if latest[j] is None:
                continue

            age = (
                day - last_day[j]
            ).days

            if age > args.max_age_days:
                continue

            result = latest[j]

            joint[i, j] = [
                float(result["impact"]),
                float(result["risk"]),
                float(result["confidence"]),
                1.0,
                float(bool(
                    result.get(
                        "kg_evidence_ids"
                    )
                )),
                1.0 - min(
                    age / args.max_age_days,
                    1.0,
                ),
            ]

    coverage = float(
        joint[:, :, 3].mean()
    )

    say(
        "[JOINT TEST COVERAGE]",
        f"{coverage:.2%}",
    )

    return joint, coverage


def main():
    args, extra = get_args()

    out = Path(args.out)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_dir = Path(
        args.models_dir
    )

    (
        base_args,
        dates,
        contexts,
        market,
        prices,
    ) = load_test_data(
        extra,
        out,
    )

    say(
        "[TEST PERIOD]",
        dates.min(),
        dates.max(),
    )

    if args.market_only_check:
        datasets = {
            "MARKET": v16.base.compose(
                market
            )
        }

        arms = ["MARKET"]

    else:
        say("[SEC RETRIEVAL]")

        sec, sec_rows = v16.v13.load_sec(
            dates,
            contexts,
            base_args.pit_sec,
            args.sec_top_k,
            out / "test_sec_retrieval_audit.csv",
        )

        say("[KG RETRIEVAL]")

        kg_events = (
            v16.v13.kg_events_from_csv(
                base_args.temporal_kg,
                TEST_END,
            )
        )

        kg, relations = (
            v16.v13.build_kg_rag(
                dates,
                kg_events,
                args.kg_top_k,
                out,
            )
        )

        if not (
            sec.shape == (
                len(dates), 5, 64
            )
            and kg.shape[:2] == (
                len(dates), 5
            )
        ):
            raise RuntimeError(
                "Test SEC/KG dimension mismatch"
            )

        joint, coverage = (
            build_joint_features(
                args,
                base_args,
                dates,
                out,
            )
        )

        datasets = {
            "MARKET":
                v16.base.compose(market),

            "MARKET_SEC":
                v16.base.compose(
                    market, sec
                ),

            "MARKET_KG":
                v16.base.compose(
                    market, kg
                ),

            "MARKET_SEC_KG":
                v16.base.compose(
                    market, sec, kg
                ),

            "MARKET_SEC_KG_JOINT_RAG":
                v16.base.compose(
                    market, sec, kg, joint
                ),
        }

        arms = ARMS

    results = []

    for arm in arms:
        features = datasets[arm]

        if not np.isfinite(
            features
        ).all():
            raise RuntimeError(
                "Nonfinite test features: "
                + arm
            )

        say(
            "[TEST ARM]",
            arm,
            "shape=",
            features.shape,
        )

        for seed in args.seeds:
            checkpoint = (
                model_dir
                / f"{arm}_PPO_seed{seed}.zip"
            )

            if not checkpoint.exists():
                raise FileNotFoundError(
                    "Frozen PPO checkpoint missing: "
                    + str(checkpoint)
                )

            # Evaluation only.
            model = PPO.load(
                str(checkpoint),
                device="cpu",
            )

            env = v16.DualRAGPortfolioEnv(
                features,
                prices,
                dates,
                base_args.transaction_cost,
                base_args.reward_scale,
            )

            model_obs = (
                model.observation_space.shape
            )

            env_obs = (
                env.observation_space.shape
            )

            if model_obs != env_obs:
                raise RuntimeError(
                    f"Observation mismatch for "
                    f"{arm}: model={model_obs}, "
                    f"test={env_obs}. "
                    "Do not change model dimensions "
                    "or retrain on the test set."
                )

            random.seed(seed)
            np.random.seed(seed)

            history, metrics = (
                v16.evaluate(
                    model,
                    env,
                    arm,
                    seed,
                    "PPO",
                )
            )

            history.to_csv(
                out
                / f"{arm}_PPO_seed{seed}_test.csv",
                index=False,
            )

            results.append(metrics)

            pd.DataFrame(
                results
            ).to_csv(
                out / "test_metrics.csv",
                index=False,
            )

            say(
                "[TEST]",
                arm,
                seed,
                "multiple=",
                round(
                    metrics["final_multiple"],
                    6,
                ),
                "sharpe=",
                round(
                    metrics["sharpe"],
                    6,
                ),
            )

    result = pd.DataFrame(
        results
    )

    summary = (
        result.groupby("arm")
        .agg(
            seeds=("seed", "nunique"),
            mean_multiple=(
                "final_multiple", "mean"
            ),
            std_multiple=(
                "final_multiple", "std"
            ),
            mean_sharpe=(
                "sharpe", "mean"
            ),
            mean_mdd=(
                "max_drawdown", "mean"
            ),
            mean_turnover=(
                "mean_turnover", "mean"
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "test_summary.csv",
        index=False,
    )

    manifest = {
        "experiment": (
            "V17 Joint SEC-KG RAG "
            "Frozen PPO Test"
        ),
        "test_range": [
            str(dates.min().date()),
            str(dates.max().date()),
        ],
        "seeds": args.seeds,
        "model_directory": str(
            model_dir
        ),
        "trained_on_test": False,
        "test_type": (
            "PROVISIONAL_OUT_OF_SAMPLE"
        ),
        "joint_refresh_dates": (
            args.joint_refresh_dates
        ),
        "max_age_days": (
            args.max_age_days
        ),
        "limitations": [
            "Prior test-period inspection "
            "prevents strict blind-test claims.",
            "LLM citations are ID-validated only.",
            "KG first-observed does not prove "
            "public availability.",
            "Opaque KG semantic claims "
            "are not independently verified.",
            "Test features must match the "
            "training encoding and parameters.",
            "No frozen weights were updated.",
        ],
    }

    (
        out / "manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    say()
    say(
        "========== 2022-2023 TEST SUMMARY =========="
    )
    say(
        summary.to_string(
            index=False
        )
    )
    say()
    say(
        "FROZEN JOINT RAG PPO TEST: PASS"
    )


if __name__ == "__main__":
    main()
