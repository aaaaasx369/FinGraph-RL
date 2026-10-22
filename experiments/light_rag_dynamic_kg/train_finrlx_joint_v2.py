#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_v16_dual_rag_finrl as v16,
)

from experiments.light_rag_dynamic_kg import (
    train_finrlx_multidrl as legacy,
)

BASE = ROOT / "results/light_rag_dynamic_kg"
GEN = BASE / "ablation_v1"
VER = BASE / "verified_v1"

TICKERS = list(v16.TICKERS)

GROUPS = [
    "A_KG_RAG",
    "B_SEC_RAG",
    "C_KG_SEC_RAG",
]

CONDITIONS = {
    "M0": [],
    "M1": ["A_KG_RAG"],
    "M2": ["B_SEC_RAG"],
    "M3": ["C_KG_SEC_RAG"],
}

FEATURE_DIM = 9
TOTAL_EXTRA = 27


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def signal_path(ticker, date, group, directory):
    return directory / f"{ticker}_{date}_{group}.json"


def checked_signal(
    ticker,
    date,
    group,
    allow_auto_judge,
):
    source_path = signal_path(
        ticker, date, group, GEN
    )
    verified_path = signal_path(
        ticker, date, group, VER
    )

    original = read_json(source_path)
    checked = read_json(verified_path)

    if original is None or checked is None:
        return None

    # Prevent mixing outdated judge results
    # with regenerated source JSON.
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(
            original,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    if fingerprint != checked.get("input_sha256"):
        raise RuntimeError(
            f"Stale verification: {verified_path}"
        )

    for row in (original, checked):
        if (
            row.get("ticker") != ticker
            or row.get("date") != date
            or row.get("group") != group
        ):
            raise RuntimeError(
                f"Signal identity mismatch: {source_path}"
            )

    # Unreviewed automated judge output is not
    # automatically human certified.
    human = checked.get("human_verified") is True
    auto = (
        checked.get("automated_semantic_pass") is True
        and allow_auto_judge
    )

    if not (human or auto):
        return {
            "valid": False,
            "reason": "not_admitted",
        }

    source_types = {
        item.get("type")
        for item in original.get("evidence", [])
    }

    if group == "A_KG_RAG" and source_types != {"KG"}:
        return {"valid": False, "reason": "wrong_sources"}

    if group == "B_SEC_RAG" and source_types != {"SEC"}:
        return {"valid": False, "reason": "wrong_sources"}

    if group == "C_KG_SEC_RAG":
        if not source_types or not source_types <= {
            "KG", "SEC"
        }:
            return {
                "valid": False,
                "reason": "wrong_sources",
            }

    # Repeat conservative temporal checks
    # before admitting any historical signal.
    cutoff = pd.Timestamp(date)

    for item in original.get("evidence", []):
        available = pd.to_datetime(
            item.get("available_at"),
            utc=True,
            errors="coerce",
        )

        if (
            pd.isna(available)
            or available.tz_localize(None) >= cutoff
        ):
            raise RuntimeError(
                "PIT violation: " + str(source_path)
            )

    vector = np.zeros(FEATURE_DIM, dtype=np.float32)

    scores = checked.get("accepted_scores") or {}
    masks = checked.get("accepted_masks") or {}

    for idx, key in enumerate(
        ("impact", "risk", "confidence")
    ):
        if masks.get(key) == 1:
            value = scores.get(key)

            if value is None:
                raise RuntimeError(
                    "Accepted mask without score"
                )

            value = float(value)

            if not np.isfinite(value):
                raise RuntimeError(
                    "Nonfinite accepted score"
                )

            vector[idx] = value
            vector[6 + idx] = 1.0

    # Existing V18-compatible 9D layout.
    # source 0=KG-only / source 1=SEC-only
    # Joint stores both source availability flags.
    vector[3] = float("SEC" in source_types)
    vector[4] = float("KG" in source_types)
    vector[5] = 1.0

    if not vector[6:9].any():
        return {
            "valid": False,
            "reason": "no_accepted_scores",
        }

    fallback = (
        "BOTH"
        if source_types == {"KG", "SEC"}
        else "SEC_ONLY_FALLBACK"
        if source_types == {"SEC"}
        else "KG_ONLY_FALLBACK"
    )

    return {
        "valid": True,
        "vector": vector,
        "source_status": fallback,
    }


def build_tensor(
    dates,
    group,
    max_age_days,
    allow_auto_judge,
):
    dates = pd.DatetimeIndex(dates)

    tensor = np.zeros(
        (len(dates), 5, FEATURE_DIM),
        dtype=np.float32,
    )

    latest = [None] * 5
    refreshed = [None] * 5

    counts = {
        "BOTH": 0,
        "SEC_ONLY_FALLBACK": 0,
        "KG_ONLY_FALLBACK": 0,
        "REJECTED": 0,
    }

    for i, day in enumerate(dates):
        date = day.strftime("%Y-%m-%d")

        # Preserve V18's Train/Val separation.
        if (
            i > 0
            and dates[i - 1].year <= 2020
            and day.year >= 2021
        ):
            latest = [None] * 5
            refreshed = [None] * 5

        for j, ticker in enumerate(TICKERS):
            update = checked_signal(
                ticker,
                date,
                group,
                allow_auto_judge,
            )

            if update is not None:
                if update["valid"]:
                    latest[j] = update["vector"].copy()
                    refreshed[j] = day

                    counts[
                        update["source_status"]
                    ] += 1
                else:
                    # Explicit rejected refresh invalidates
                    # the previous evidence state.
                    latest[j] = None
                    refreshed[j] = None
                    counts["REJECTED"] += 1

            if latest[j] is None:
                continue

            age = (day - refreshed[j]).days

            if age < 0:
                raise RuntimeError("Future information")

            if age > max_age_days:
                continue

            row = latest[j].copy()

            # Same freshness convention as V18.
            row[5] = (
                1.0 - min(age / max_age_days, 1.0)
            )

            tensor[i, j] = row

    return tensor, counts


def coverage(tensor):
    return float(
        (tensor[:, :, 6:9].sum(axis=-1) > 0).mean()
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--timesteps", type=int, default=20000
    )
    p.add_argument(
        "--seeds", nargs="+",
        type=int, default=[42]
    )
    p.add_argument(
        "--algos", nargs="+",
        choices=["A2C", "PPO", "DDPG"],
        default=["A2C", "PPO", "DDPG"],
    )
    p.add_argument(
        "--arms", nargs="+",
        choices=list(CONDITIONS),
        default=list(CONDITIONS),
    )
    p.add_argument(
        "--max-age-days", type=int, default=35
    )
    p.add_argument(
        "--min-coverage", type=float, default=0.15
    )
    p.add_argument(
        "--allow-automated-judge",
        action="store_true",
        help="Exploratory mode only; not human certification.",
    )
    p.add_argument(
        "--learning-rate", type=float, default=3e-4
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--out",
        default=str(BASE / "full_pipeline/joint_v2"),
    )

    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv

    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        opt = v16.parse_args()
    finally:
        sys.argv = old_argv

    opt.out = out

    train, val, dates, contexts, n_train = (
        v16.v8b.load_data(opt)
    )

    dates = v16.base.normalize_dates(dates)

    flat = np.concatenate(
        [train["market"], val["market"]],
        axis=0,
    )

    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0,
    )

    market, kgt = v16.base.validate_market_layout(flat)

    if not (
        market.shape == (len(dates), 5, 106)
        and dates[0].year == 2018
        and dates[n_train].year == 2021
        and dates[-1].year == 2021
        and len(prices) == len(dates)
    ):
        raise RuntimeError(
            "Unexpected native FinRL-X date split"
        )

    sec, _ = v16.v13.load_sec(
        dates,
        contexts,
        opt.pit_sec,
        opt.sec_top_k,
        out / "original_sec_audit.csv",
    )

    events = v16.v13.kg_events_from_csv(
        opt.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg, _ = v16.v13.build_kg_rag(
        dates,
        events,
        opt.kg_top_k,
        out,
    )

    base_features = v16.base.compose(
        market, sec, kg
    )

    print("[MARKET]", market.shape)
    print("[ORIGINAL SEC]", sec.shape)
    print("[ORIGINAL KG]", kg.shape)
    print("[ORIGINAL BASE]", base_features.shape)

    rag = {}
    counts = {}

    for group in GROUPS:
        rag[group], counts[group] = build_tensor(
            dates,
            group,
            args.max_age_days,
            args.allow_automated_judge,
        )

    zeros = np.zeros(
        (len(dates), 5, TOTAL_EXTRA),
        dtype=np.float32,
    )

    datasets = {}

    for arm in CONDITIONS:
        extra = zeros.copy()

        for k, group in enumerate(GROUPS):
            if group in CONDITIONS[arm]:
                extra[
                    :, :,
                    k * FEATURE_DIM:(k + 1) * FEATURE_DIM
                ] = rag[group]

        datasets[arm] = np.concatenate(
            [base_features, extra],
            axis=2,
        ).astype(np.float32)

    cov = {}

    for arm in CONDITIONS:
        print(
            "[DATASET]", arm,
            datasets[arm].shape,
        )

    for group in GROUPS:
        cov[group] = {
            "train": coverage(
                rag[group][:n_train]
            ),
            "val": coverage(
                rag[group][n_train:]
            ),
            "counts": counts[group],
        }

    print(
        "[RAG COVERAGE]",
        json.dumps(cov, indent=2),
    )

    manifest = {
        "train": "2018-2020",
        "validation": "2021",
        "test": "not loaded",
        "per_company_base_dim": base_features.shape[2],
        "per_company_new_rag_dim": TOTAL_EXTRA,
        "per_company_total_dim":
            datasets["M0"].shape[2],
        "rag_groups": GROUPS,
        "conditions": CONDITIONS,
        "coverage": cov,
        "automated_judge_allowed":
            args.allow_automated_judge,
        "human_certified": False,
        "missingness": "zero_placeholder_plus_masks",
        "max_age_days": args.max_age_days,
    }

    (out / "manifest.json").write_text(
        json.dumps(
            manifest, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    if args.dry_run:
        print("[DRY RUN] No RL training")
        return

    for arm in args.arms:
        for group in CONDITIONS[arm]:
            c = cov[group]

            if (
                c["train"] < args.min_coverage
                or c["val"] < args.min_coverage
            ):
                raise RuntimeError(
                    f"{arm}/{group}: insufficient "
                    f"verified RAG coverage: {c}"
                )

    metrics_all = []

    for arm in args.arms:
        x = datasets[arm]

        if not np.isfinite(x).all():
            raise RuntimeError(
                f"Nonfinite features: {arm}"
            )

        for algo in args.algos:
            for seed in args.seeds:
                random.seed(seed)
                np.random.seed(seed)

                print(
                    "[TRAIN]", arm, algo, seed,
                    flush=True,
                )

                train_env = v16.DualRAGPortfolioEnv(
                    x[:n_train],
                    prices[:n_train],
                    dates[:n_train],
                    opt.transaction_cost,
                    opt.reward_scale,
                )

                val_env = v16.DualRAGPortfolioEnv(
                    x[n_train:],
                    prices[n_train:],
                    dates[n_train:],
                    opt.transaction_cost,
                    opt.reward_scale,
                )

                model = legacy.make_model(
                    algo,
                    train_env,
                    seed,
                    args.learning_rate,
                    args.device,
                )

                model.learn(
                    total_timesteps=args.timesteps,
                    progress_bar=False,
                )

                name = f"{arm}_{algo}_seed{seed}"

                model.save(str(out / name))

                history, metrics = v16.evaluate(
                    model, val_env, arm, seed, algo
                )

                history.to_csv(
                    out / f"{name}_validation.csv",
                    index=False,
                )

                metrics_all.append(metrics)

                pd.DataFrame(metrics_all).to_csv(
                    out / "validation_metrics.csv",
                    index=False,
                )

                print(
                    "[FINISHED]", name, metrics,
                    flush=True,
                )

    print("[ALL FINISHED]")


if __name__ == "__main__":
    main()
