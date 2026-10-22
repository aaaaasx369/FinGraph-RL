#!/usr/bin/env python3
"""
Option B: Native FinRL-X three-way experiment.

M0: Original 319D
M1: Original 319D + KG Structural 28D
M2: M1 + train-only PCA16D + summary_valid_mask

Default: DRY RUN ONLY.
--train: enabled only after coverage/PCA checks pass.

No LightRAG calls, no modification of original V16.
"""

import argparse
import hashlib
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

DUAL = ROOT / (
    "results/light_rag_dynamic_kg/kg_dual_channel_v1/"
    "kg_dual_channel_finrl_aligned.npz"
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
ARMS = ["M0", "M1", "M2"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_native(out):
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

    dates = pd.DatetimeIndex(
        v16.base.normalize_dates(dates)
    )

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
        raise RuntimeError("Unexpected native V16 split")

    if dates.has_duplicates:
        raise RuntimeError("Duplicate trading dates")

    if not dates.is_monotonic_increasing:
        raise RuntimeError("Trading dates not sorted")

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

    base = v16.base.compose(
        market, sec, kg
    ).astype(np.float32)

    if base.shape != (len(dates), 5, 319):
        raise RuntimeError(
            f"Original FinRL-X must be [N,5,319], got {base.shape}"
        )

    if not np.isfinite(base).all():
        raise RuntimeError("Nonfinite native features")

    if not np.isfinite(prices).all():
        raise RuntimeError("Nonfinite prices")

    return opt, dates, prices, base, n_train


def load_dual(dates, n_train):
    if not DUAL.exists():
        raise FileNotFoundError(DUAL)

    # This NPZ is a locally produced trusted research artifact.
    # The date array is object dtype in the existing pipeline.
    with np.load(DUAL, allow_pickle=True) as z:
        required = {
            "structured",
            "summary_embeddings",
            "dates",
            "tickers",
            "feature_names",
            "n_train",
        }

        if not required.issubset(z.files):
            raise RuntimeError(
                f"Missing keys: {required - set(z.files)}"
            )

        s = z["structured"].copy().astype(np.float32)
        e = z["summary_embeddings"].copy().astype(np.float32)
        stored_dates = pd.DatetimeIndex(
            pd.to_datetime(z["dates"].astype(str))
        )
        tickers = [str(t) for t in z["tickers"]]
        names = [str(t) for t in z["feature_names"]]
        stored_n = int(z["n_train"])

    if s.shape != (len(dates), 5, 28):
        raise RuntimeError(f"Structured shape mismatch: {s.shape}")

    if e.shape != (len(dates), 5, 1024):
        raise RuntimeError(f"Embedding shape mismatch: {e.shape}")

    if stored_n != n_train:
        raise RuntimeError("Native and Dual train boundaries differ")

    if tickers != TICKERS:
        raise RuntimeError(
            f"Ticker order mismatch: {tickers} != {TICKERS}"
        )

    if not stored_dates.equals(dates):
        differing = np.flatnonzero(
            stored_dates.to_numpy() != dates.to_numpy()
        )
        raise RuntimeError(
            "Dual Channel dates differ from V16; "
            f"first mismatch indices={differing[:10].tolist()}"
        )

    if not np.isfinite(s).all() or not np.isfinite(e).all():
        raise RuntimeError("Nonfinite Dual Channel data")

    required_names = [
        "impact", "risk", "confidence",
        "impact_valid", "risk_valid", "confidence_valid",
        "retrieval_valid", "summary_present",
    ]

    if not all(k in names for k in required_names):
        raise RuntimeError("Missing structural mask columns")

    # Do not treat invalid numeric candidates as real values.
    for value, flag in [
        ("impact", "impact_valid"),
        ("risk", "risk_valid"),
        ("confidence", "confidence_valid"),
    ]:
        vi, mi = names.index(value), names.index(flag)
        if np.any((s[:, :, mi] <= 0) & (np.abs(s[:, :, vi]) > 1e-7)):
            raise RuntimeError(
                f"{value} has nonzero values without valid mask"
            )

    retrieval = s[:, :, names.index("retrieval_valid")] > 0.5
    summary = s[:, :, names.index("summary_present")] > 0.5
    nonzero = np.linalg.norm(e, axis=2) > 1e-8
    valid = retrieval & summary & nonzero

    if np.any(summary & ~retrieval):
        raise RuntimeError("Summary present without retrieval flag")

    # Missing embedding is never supplied as valid evidence.
    e[~valid] = 0.0

    return s, e, valid, names


def train_only_pca(e, valid, n_train, components=16):
    vectors = e[:n_train][valid[:n_train]]
    unique = np.unique(vectors, axis=0)

    result = {
        "requested_components": components,
        "valid_training_stock_days": int(len(vectors)),
        "unique_training_embeddings": int(len(unique)),
        "max_nonzero_components": max(0, len(unique) - 1),
        "ready": False,
    }

    if len(unique) < components + 1:
        result["reason"] = (
            f"Need at least {components+1} distinct "
            f"training embeddings, found {len(unique)}"
        )
        return None, result

    mean = unique.mean(axis=0, dtype=np.float64)
    centered = unique.astype(np.float64) - mean

    # Only the training set contributes to PCA.
    _, singular, vt = np.linalg.svd(
        centered, full_matrices=False
    )

    rank = int(np.sum(
        singular > singular[0] * 1e-7
    ))

    result["numeric_rank"] = rank

    if rank < components:
        result["reason"] = (
            f"Training matrix rank {rank} < {components}"
        )
        return None, result

    basis = vt[:components]

    # Fix component signs for reproducibility.
    for i in range(components):
        j = np.argmax(np.abs(basis[i]))
        if basis[i, j] < 0:
            basis[i] *= -1

    projected = (
        (e.astype(np.float64) - mean) @ basis.T
    ).astype(np.float32)

    # Normalize PCA scores using training-only valid values.
    tr = projected[:n_train][valid[:n_train]]
    scale = tr.std(axis=0, dtype=np.float64)
    scale = np.maximum(scale, 1e-6).astype(np.float32)

    projected = projected / scale
    projected[~valid] = 0.0

    if not np.isfinite(projected).all():
        raise RuntimeError("Nonfinite PCA projection")

    result["ready"] = True
    result["reason"] = "Train-only PCA successful"

    return projected, result


def coverage(mask, n_train):
    return {
        "train": float(mask[:n_train].mean()),
        "validation": float(mask[n_train:].mean()),
        "train_stock_days": int(mask[:n_train].sum()),
        "validation_stock_days": int(mask[n_train:].sum()),
    }


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--train", action="store_true")
    p.add_argument(
        "--out",
        default="results/light_rag_dynamic_kg/option_b_threeway",
    )
    p.add_argument("--timesteps", type=int, default=20000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument(
        "--algos", nargs="+",
        choices=["PPO", "A2C", "DDPG"],
        default=["PPO"],
    )
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--device", default="auto")
    p.add_argument("--min-train-coverage", type=float, default=0.10)
    p.add_argument("--min-val-coverage", type=float, default=0.05)

    args = p.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("OPTION B: NATIVE V16 + KG LIGHTRAG")
    print("MODE:", "TRAIN" if args.train else "DRY RUN")
    print("=" * 75)

    print("\n[1] Loading original V16")

    opt, dates, prices, base, n_train = load_native(out)

    print("BASE:", base.shape)
    print("TRAIN DAYS:", n_train)
    print("VAL DAYS:", len(dates) - n_train)

    print("\n[2] Loading existing KG Dual Channel")

    structural, embeddings, valid, feature_names = (
        load_dual(dates, n_train)
    )

    print("STRUCTURAL:", structural.shape)
    print("EMBEDDINGS:", embeddings.shape)

    cov = coverage(valid, n_train)

    print("SUMMARY COVERAGE:", cov)

    print("\n[3] Training-only PCA feasibility")

    pca, pca_report = train_only_pca(
        embeddings, valid, n_train, components=16
    )

    print(json.dumps(pca_report, indent=2))

    print("\n[4] Constructing experiment conditions")

    # M0: unchanged original FinRL-X.
    m0 = base.copy()

    # M1: structural retrieval evidence.
    m1 = np.concatenate(
        [base, structural], axis=2
    ).astype(np.float32)

    datasets = {
        "M0": m0,
        "M1": m1,
    }

    if pca is not None:
        mask = valid.astype(np.float32)[..., None]

        m2 = np.concatenate(
            [base, structural, pca, mask], axis=2
        ).astype(np.float32)

        datasets["M2"] = m2

    for arm, x in datasets.items():
        if not np.array_equal(x[:, :, :319], base):
            raise RuntimeError(
                f"Original 319D modified in {arm}"
            )

        if not np.isfinite(x).all():
            raise RuntimeError(f"Nonfinite data in {arm}")

        print(arm, x.shape)

    if "M2" not in datasets:
        print("M2: NOT_READY (PCA16 insufficient evidence)")

    # Baseline and augmented conditions must share dates/prices.
    # Audit representation only: upstream PIT certification
    # must be checked independently.
    report = {
        "mode": "train" if args.train else "dry_run",
        "base_dim": 319,
        "structural_dim": 28,
        "pca_dim": 16,
        "summary_mask_dim": 1,
        "conditions": {
            "M0": [len(dates), 5, 319],
            "M1": [len(dates), 5, 347],
            "M2": [len(dates), 5, 364]
            if pca is not None else "NOT_READY",
        },
        "dates": {
            "first": str(dates[0].date()),
            "last": str(dates[-1].date()),
            "train_days": n_train,
            "val_days": len(dates) - n_train,
        },
        "coverage": cov,
        "pca": pca_report,
        "source_npz": str(DUAL),
        "source_sha256": digest(DUAL),
        "external_PIT_certified": False,
        "summary_semantics_certified": False,
        "historical_batch_complete": False,
    }

    report_path = out / "preflight_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n[SAVED]", report_path)

    if not args.train:
        print("\nDRY RUN COMPLETE")
        print("No RL model was trained.")
        print("No original V16 model was modified.")
        return

    # Do not train while coverage and PCA are insufficient.
    if pca is None:
        print(
            "[SKIP] M2 NOT_READY: PCA16 unavailable. "
            "Training available models only.",
            flush=True,
        )

    if cov["train"] < args.min_train_coverage:
        raise RuntimeError("Insufficient training RAG coverage")

    if cov["validation"] < args.min_val_coverage:
        raise RuntimeError("Insufficient validation RAG coverage")

    metrics_all = []

    for arm in ARMS:
        if arm not in datasets:
            print(f"[SKIP ARM] {arm}: dataset not ready")
            continue
        x = datasets[arm]

        for algo in args.algos:
            for seed in args.seeds:
                print("[TRAIN]", arm, algo, seed, flush=True)

                random.seed(seed)
                np.random.seed(seed)

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
                    algo, train_env, seed,
                    args.learning_rate,
                    args.device,
                )

                model.learn(
                    total_timesteps=args.timesteps,
                    progress_bar=False,
                )

                name = f"{arm}_{algo}_seed{seed}"
                model_path = out / f"{name}.zip"

                if model_path.exists():
                    raise FileExistsError(
                        f"Refusing to overwrite {model_path}"
                    )

                model.save(str(model_path))

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

                print("[FINISHED]", name, metrics, flush=True)

    print("OPTION B TRAINING COMPLETE")


if __name__ == "__main__":
    main()
