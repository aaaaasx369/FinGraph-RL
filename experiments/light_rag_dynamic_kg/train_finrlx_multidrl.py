#!/usr/bin/env python3

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import A2C, PPO, DDPG
from stable_baselines3.common.noise import NormalActionNoise

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_v16_dual_rag_finrl as v16,
)

BASE = ROOT / "results/light_rag_dynamic_kg"
RAG_DIR = BASE / "ablation_v1"

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

ARMS = ["M0", "M1", "M2", "M3"]
ALGOS = ["A2C", "PPO", "DDPG"]

# Impact, Risk, Confidence, SEC-verified,
# KG-structural, Freshness, 3 known-value masks.
D = 9


def parse():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--timesteps", type=int, default=20000
    )
    ap.add_argument(
        "--seeds", nargs="+",
        type=int, default=[42]
    )
    ap.add_argument(
        "--algos", nargs="+",
        choices=ALGOS, default=ALGOS
    )
    ap.add_argument(
        "--arms", nargs="+",
        choices=ARMS, default=ARMS
    )
    ap.add_argument(
        "--max-age-days", type=int, default=35
    )
    ap.add_argument(
        "--min-rag-coverage", type=float, default=0.15
    )
    ap.add_argument(
        "--learning-rate", type=float, default=0.0003
    )
    ap.add_argument(
        "--device", default="auto"
    )
    ap.add_argument(
        "--out",
        default=str(BASE / "full_pipeline" / "multidrl"),
    )
    ap.add_argument(
        "--dry-run", action="store_true"
    )

    return ap.parse_args()


def finite_number(value, lo, hi):
    if value is None or isinstance(value, bool):
        return None

    try:
        x = float(value)
    except (ValueError, TypeError):
        return None

    if not np.isfinite(x) or not lo <= x <= hi:
        return None

    return x


def read_signal(ticker, date, source):
    suffix = {
        "KG": "A_KG_RAG",
        "SEC": "B_SEC_RAG",
    }[source]

    path = RAG_DIR / (
        f"{ticker}_{date}_{suffix}.json"
    )

    if not path.exists():
        return None

    record = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        record.get("ticker") != ticker
        or record.get("date") != date
    ):
        raise RuntimeError(
            f"RAG identity mismatch: {path}"
        )

    allowed = set(record.get("allowed_sources", []))

    if source not in allowed:
        raise RuntimeError(
            f"RAG source violation: {path}"
        )

    evidence = record.get("evidence") or []

    if any(
        item.get("type") != source
        for item in evidence
    ):
        raise RuntimeError(
            f"Mixed-source evidence in {path}"
        )

    # Conservatively require every selected source
    # to predate the decision date.
    cutoff = pd.Timestamp(date)

    for item in evidence:
        timestamp = pd.to_datetime(
            item.get("available_at"),
            utc=True,
            errors="coerce",
        )

        if (
            pd.isna(timestamp)
            or timestamp.tz_localize(None) >= cutoff
        ):
            raise RuntimeError(
                f"PIT violation: {path}"
            )

    v = record.get("validation") or {}
    generated = record.get("generated") or {}

    semantic_ok = (
        v.get("semantic_verified") is True
        and v.get("quote_checks_passed") is True
        and v.get("status") == "supported"
        and not v.get("rejected")
    )

    if not semantic_ok:
        return {
            "valid": False,
            "reason": "semantic_unverified",
        }

    impact = finite_number(
        v.get("impact"), -1, 1
    )
    risk = finite_number(
        v.get("risk"), 0, 1
    )
    confidence = finite_number(
        v.get("confidence"), 0, 1
    )

    values = [impact, risk, confidence]

    if all(value is None for value in values):
        return {
            "valid": False,
            "reason": "all_scores_missing",
        }

    vector = np.zeros(D, dtype=np.float32)

    for j, value in enumerate(values):
        if value is not None:
            vector[j] = value
            vector[6+j] = 1.0

    vector[3] = float(source == "SEC")
    vector[4] = float(source == "KG")
    vector[5] = 1.0

    return {
        "valid": True,
        "vector": vector,
        "reason": "verified",
    }


def build_rag_tensor(dates, source, max_age_days):
    """
    Only current/past refreshes are usable.
    No future backfill.

    An explicit failed refresh clears prior state.
    A valid historical signal can carry forward
    only within its allowed age.
    """

    n = len(dates)
    result = np.zeros(
        (n, 5, D), dtype=np.float32
    )

    latest = [None] * 5
    latest_day = [None] * 5

    date_index = pd.DatetimeIndex(dates)

    for i, day in enumerate(date_index):
        date_str = day.strftime("%Y-%m-%d")

        # Reset state between Train and Validation.
        if (
            i > 0
            and date_index[i-1].year <= 2020
            and day.year >= 2021
        ):
            latest = [None] * 5
            latest_day = [None] * 5

        for j, ticker in enumerate(TICKERS):
            current = read_signal(
                ticker, date_str, source
            )

            if current is not None:
                if current["valid"]:
                    latest[j] = current["vector"]
                    latest_day[j] = day
                else:
                    latest[j] = None
                    latest_day[j] = None

            if latest[j] is None:
                continue

            age = int(
                (day - latest_day[j]).days
            )

            if age < 0:
                raise RuntimeError(
                    "Future RAG data detected"
                )

            if age > max_age_days:
                continue

            vector = latest[j].copy()

            vector[5] = (
                1.0 - age / max_age_days
            )

            result[i, j] = vector

    return result


def make_model(
    algo, env, seed, lr, device,
):
    common = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        learning_rate=lr,
        device=device,
        verbose=0,
        policy_kwargs={
            "net_arch": [128, 128]
        },
    )

    if algo == "A2C":
        return A2C(
            **common,
            n_steps=20,
            gamma=0.99,
            gae_lambda=0.95,
        )

    if algo == "PPO":
        return PPO(
            **common,
            n_steps=128,
            batch_size=64,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
        )

    if algo == "DDPG":
        if env.action_space.shape != (5,):
            raise RuntimeError(
                "DDPG requires 5D action space"
            )

        noise = NormalActionNoise(
            mean=np.zeros(5),
            sigma=np.full(5, 0.1),
        )

        return DDPG(
            **common,
            action_noise=noise,
            learning_starts=500,
            buffer_size=50000,
            batch_size=128,
            gamma=0.99,
        )

    raise ValueError(algo)


def coverage(tensor):
    # At least one scored numerical signal known.
    return float(
        (tensor[:, :, 6:9].sum(axis=-1) > 0).mean()
    )


def main():
    opt = parse()

    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    # Original V16 parser is preserved.
    old_argv = sys.argv

    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        args = v16.parse_args()
    finally:
        sys.argv = old_argv

    args.out = out
    args.seeds = opt.seeds
    args.timesteps = opt.timesteps

    print("========== ORIGINAL V16 DATA ==========")

    train, val, dates, contexts, n_train = (
        v16.v8b.load_data(args)
    )

    dates = v16.base.normalize_dates(dates)

    market_flat = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    )
    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    )

    market, kgt = (
        v16.base.validate_market_layout(
            market_flat
        )
    )

    if not (
        market.shape == (len(dates), 5, 106)
        and len(prices) == len(dates)
        and dates[0].year == 2018
        and dates[n_train].year == 2021
        and dates[-1].year == 2021
    ):
        raise RuntimeError(
            "Original FinRL-X split mismatch"
        )

    print("[MARKET]", market.shape)
    print("[KGT]", kgt.shape)
    print("[TRAIN DAYS]", n_train)
    print("[VAL DAYS]", len(dates)-n_train)

    print("\n========== ORIGINAL SEC64 ==========")

    sec, _ = v16.v13.load_sec(
        dates,
        contexts,
        args.pit_sec,
        args.sec_top_k,
        out / "original_sec_retrieval_audit.csv",
    )

    print("[SEC]", sec.shape)

    print("\n========== ORIGINAL TEMPORAL KG ==========")

    kg_events = v16.v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg, _ = v16.v13.build_kg_rag(
        dates,
        kg_events,
        args.kg_top_k,
        out,
    )

    print("[KG]", kg.shape)

    if (
        sec.shape != (len(dates), 5, 64)
        or kg.shape[:2] != (len(dates), 5)
    ):
        raise RuntimeError(
            "Original SEC/KG dimension mismatch"
        )

    base_features = v16.base.compose(
        market, sec, kg
    )

    if base_features.shape[:2] != (
        len(dates), 5
    ):
        raise RuntimeError(
            "Base feature layout mismatch"
        )

    print("[ORIGINAL FINRL-X]", base_features.shape)

    print("\n========== NEW KG LIGHTRAG ==========")

    rag_kg = build_rag_tensor(
        dates, "KG", opt.max_age_days
    )

    print("\n========== NEW SEC GENERATIVE RAG ==========")

    rag_sec = build_rag_tensor(
        dates, "SEC", opt.max_age_days
    )

    print("[KG RAG]", rag_kg.shape)
    print("[SEC RAG]", rag_sec.shape)

    cov = {
        "kg_train": coverage(rag_kg[:n_train]),
        "kg_val": coverage(rag_kg[n_train:]),
        "sec_train": coverage(rag_sec[:n_train]),
        "sec_val": coverage(rag_sec[n_train:]),
    }

    print("[VERIFIED COVERAGE]", cov)

    # Fixed observation shape across all arms.
    zeros = np.zeros_like(rag_kg)

    def compose(extra_kg, extra_sec):
        return np.concatenate(
            [
                base_features,
                extra_kg,
                extra_sec,
            ],
            axis=2,
        ).astype(np.float32)

    datasets = {
        "M0": compose(zeros, zeros),
        "M1": compose(rag_kg, zeros),
        "M2": compose(zeros, rag_sec),
        "M3": compose(rag_kg, rag_sec),
    }

    for arm, matrix in datasets.items():
        if not np.isfinite(matrix).all():
            raise RuntimeError(
                f"Nonfinite features: {arm}"
            )
        print(
            "[ARM]", arm,
            "FEATURES", matrix.shape,
            "OBS", matrix.shape[1] * matrix.shape[2] + 5,
        )

    manifest = {
        "train": "2018-2020",
        "validation": "2021",
        "test": "2022-2023 (not loaded here)",
        "arms": ARMS,
        "algorithms": ALGOS,
        "seeds": opt.seeds,
        "timesteps": opt.timesteps,
        "base_shape": list(base_features.shape),
        "rag_kg_shape": list(rag_kg.shape),
        "rag_sec_shape": list(rag_sec.shape),
        "coverage": cov,
        "max_age_days": opt.max_age_days,
        "semantic_requirement": "verified",
    }

    (out / "manifest.json").write_text(
        json.dumps(
            manifest, indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if opt.dry_run:
        print("[DRY RUN] No training.")
        return

    threshold = opt.min_rag_coverage

    for arm in opt.arms:
        checks = []

        if arm in ("M1", "M3"):
            checks.extend([
                cov["kg_train"] >= threshold,
                cov["kg_val"] >= threshold,
            ])

        if arm in ("M2", "M3"):
            checks.extend([
                cov["sec_train"] >= threshold,
                cov["sec_val"] >= threshold,
            ])

        if checks and not all(checks):
            raise RuntimeError(
                f"{arm}: verified RAG coverage "
                f"below {threshold:.1%}. "
                f"Generate and verify historical "
                f"RAG features before training."
            )

    results = []

    for arm in opt.arms:
        x = datasets[arm]

        for algo in opt.algos:
            for seed in opt.seeds:
                random.seed(seed)
                np.random.seed(seed)

                print(
                    "\n[TRAIN]",
                    arm, algo, "seed", seed,
                    flush=True,
                )

                train_env = v16.DualRAGPortfolioEnv(
                    x[:n_train],
                    prices[:n_train],
                    dates[:n_train],
                    args.transaction_cost,
                    args.reward_scale,
                )

                val_env = v16.DualRAGPortfolioEnv(
                    x[n_train:],
                    prices[n_train:],
                    dates[n_train:],
                    args.transaction_cost,
                    args.reward_scale,
                )

                model = make_model(
                    algo,
                    train_env,
                    seed,
                    opt.learning_rate,
                    opt.device,
                )

                model.learn(
                    total_timesteps=opt.timesteps,
                    progress_bar=False,
                )

                name = (
                    f"{arm}_{algo}_seed{seed}"
                )

                model.save(str(out / name))

                history, metrics = v16.evaluate(
                    model,
                    val_env,
                    arm,
                    seed,
                    algo,
                )

                history.to_csv(
                    out / f"{name}_validation.csv",
                    index=False,
                )

                results.append(metrics)

                pd.DataFrame(results).to_csv(
                    out / "validation_metrics.csv",
                    index=False,
                )

                print("[FINISHED]", name, metrics)

    print("\n========== FINISHED ==========")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
