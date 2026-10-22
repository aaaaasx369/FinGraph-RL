#!/usr/bin/env python3
"""
V20 Frozen Independent Fusion Blind Test.

Inputs:
    Market: [N,5,106]
    SEC:    [N,5,64]
    KG:     [N,5,149]
    Joint:  [N,5,9] zeros

Final:
    [N,5,328]

No training, no LLM calls, no test-set model selection.

IMPORTANT:
Matching shapes and timestamp checks do not alone establish
feature normalization equivalence or complete PIT lineage.
"""

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_v16_dual_rag_finrl as v16

ROOT = Path("results/aegis_rl_v1")
OUT = ROOT / "v20_blind"
FROZEN = ROOT / "v19_final/frozen_model.json"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

OUT.mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_npz(path, feature_key):
    path = Path(path)

    require(path.exists(), f"Missing input: {path}")

    with np.load(path, allow_pickle=False) as z:
        required = {"dates", "tickers", feature_key}

        require(
            required.issubset(z.files),
            f"{path}: expected {sorted(required)}, got {z.files}"
        )

        x = np.asarray(z[feature_key], dtype=np.float32)
        dates = pd.DatetimeIndex(
            pd.to_datetime(
                z["dates"].astype(str),
                errors="raise"
            )
        )
        tickers = [
            str(t) for t in z["tickers"].astype(str)
        ]

        prices = (
            np.asarray(z["prices"], dtype=np.float32)
            if "prices" in z.files
            else None
        )

    require(
        tickers == TICKERS,
        f"{path}: ticker order mismatch: {tickers}"
    )

    require(
        len(dates) == len(x),
        f"{path}: date count != feature length"
    )

    require(
        dates.is_unique and dates.is_monotonic_increasing,
        f"{path}: dates duplicated or unordered"
    )

    require(
        x.ndim == 3 and x.shape[1] == 5,
        f"{path}: invalid shape {x.shape}"
    )

    require(
        np.isfinite(x).all(),
        f"{path}: nonfinite features"
    )

    return x, dates, prices


def discover(args):
    print("========== V20 SOURCE DISCOVERY ==========")

    print("\nV16 source files:")
    for module in [
        v16,
        v16.v8b,
        v16.v13,
        v16.base
    ]:
        path = inspect.getsourcefile(module)
        print(
            module.__name__,
            "->",
            path
        )

    print("\nV16 data loader:")
    print(inspect.getsource(v16.v8b.load_data)[:16000])

    print("\nV16 argument defaults:")
    old = sys.argv
    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        options = v16.parse_args()
    finally:
        sys.argv = old

    selected = {
        k: str(v)
        for k, v in vars(options).items()
        if any(
            term in k.lower()
            for term in (
                "path", "data", "market",
                "sec", "kg", "train",
                "val", "test", "date"
            )
        )
    }

    print(json.dumps(
        selected,
        indent=2,
        ensure_ascii=False
    ))

    print("\nExisting potential source files:")

    search_roots = [
        Path("results/aegis_rl_v1"),
        Path("unified_event_memory"),
        Path("data"),
        Path("datasets"),
    ]

    candidates = []

    for root in search_roots:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in (
                ".npz", ".parquet", ".csv"
            ):
                continue

            name = path.name.lower()

            if any(
                term in name
                for term in (
                    "test", "2022", "2023",
                    "market", "sec_pit",
                    "kg", "feature"
                )
            ):
                candidates.append(str(path))

    candidates = sorted(set(candidates))

    for name in candidates[:150]:
        print(name)

    write_json(
        OUT / "discovery.json",
        {
            "v16_args": selected,
            "source_candidates": candidates,
            "note": (
                "Discovery identifies candidates only; "
                "it does not certify feature equivalence."
            )
        }
    )

    print("\nDISCOVERY COMPLETE")
    print("Candidates:", len(candidates))
    print("Test data: NOT EVALUATED")


def verify_pit(path, allowed_dates):
    path = Path(path)

    require(
        path.exists(),
        f"Missing PIT audit: {path}"
    )

    audit = pd.read_csv(path, dtype=str).fillna("")

    required = {
        "decision_date",
        "ticker",
        "source_type",
        "source_id",
        "available_at",
    }

    require(
        required.issubset(audit.columns),
        f"PIT audit missing {required-set(audit.columns)}"
    )

    require(
        len(audit) > 0,
        "PIT audit is empty"
    )

    require(
        audit[list(required)].ne("").all().all(),
        "PIT audit has missing required fields"
    )

    decision = pd.to_datetime(
        audit["decision_date"],
        utc=True,
        errors="raise"
    )

    available = pd.to_datetime(
        audit["available_at"],
        utc=True,
        errors="raise"
    )

    require(
        bool((available < decision).all()),
        "PIT VIOLATION: source available_at >= decision_date"
    )

    require(
        set(audit["ticker"]).issubset(set(TICKERS)),
        "PIT audit has unexpected tickers"
    )

    used_dates = set(
        decision.dt.strftime("%Y-%m-%d")
    )

    require(
        used_dates.issubset(
            set(allowed_dates.strftime("%Y-%m-%d"))
        ),
        "PIT audit contains dates outside test data"
    )

    return {
        "rows": len(audit),
        "sha256": sha256(path),
        "source_types": (
            audit["source_type"].value_counts().to_dict()
        ),
        "earliest_source_time": str(available.min()),
        "latest_source_time": str(available.max()),
        "warning": (
            "Only supplied lineage rows were checked. "
            "Completeness of lineage is not established."
        )
    }


def build(args):
    freeze = json.loads(FROZEN.read_text(encoding="utf-8"))

    require(
        freeze["selected_arm"] == "INDEPENDENT",
        "Frozen model is not Independent Fusion"
    )

    market, mdates, prices = load_npz(
        args.market, "market"
    )

    sec, sdates, _ = load_npz(
        args.sec, "sec"
    )

    kg, kdates, _ = load_npz(
        args.kg, "kg"
    )

    require(
        mdates.equals(sdates) and mdates.equals(kdates),
        "Market/SEC/KG dates are not identical"
    )

    require(
        market.shape[2] == 106,
        f"Expected market dim 106, got {market.shape[2]}"
    )

    require(
        sec.shape[2] == 64,
        f"Expected SEC dim 64, got {sec.shape[2]}"
    )

    require(
        kg.shape[2] == 149,
        f"Expected KG dim 149, got {kg.shape[2]}"
    )

    require(
        prices is not None,
        "Market NPZ must include prices"
    )

    require(
        prices.shape == (len(mdates), 5),
        "Price shape invalid"
    )

    require(
        bool(np.isfinite(prices).all())
        and bool((prices > 0).all()),
        "Invalid prices"
    )

    require(
        mdates.min() >= pd.Timestamp("2022-01-01")
        and mdates.max() <= pd.Timestamp("2023-12-31"),
        "Test dates outside 2022–2023"
    )

    # The frozen model has 9 extra dimensions.
    # Independent Fusion adds no Joint RAG information.
    zero_joint = np.zeros(
        (len(mdates), 5, 9),
        dtype=np.float32
    )

    features = v16.base.compose(
        market,
        sec,
        kg,
        zero_joint
    )

    require(
        features.shape == (len(mdates), 5, 328),
        f"Unexpected final feature shape: {features.shape}"
    )

    # Structural timestamp audit. Must correspond to actual
    # retrieval events, not a manually invented audit.
    pit = verify_pit(args.pit_audit, mdates)

    provenance_path = Path(args.provenance)
    require(
        provenance_path.exists(),
        "Missing source provenance JSON"
    )

    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )

    required_attestations = [
        "market_feature_equivalent_to_training",
        "sec_retrieval_pit_verified",
        "kg_retrieval_pit_verified",
        "normalization_frozen_from_training",
        "complete_lineage_verified",
    ]

    require(
        all(
            provenance.get(k) is True
            for k in required_attestations
        ),
        (
            "Feature provenance has not passed all required "
            "independent checks. Do not evaluate the sealed test."
        )
    )

    target = OUT / "test_features_2022_2023.npz"

    require(
        not target.exists(),
        "Sealed test tensor already exists; refusing overwrite"
    )

    np.savez_compressed(
        target,
        features=features,
        prices=prices,
        dates=mdates.strftime("%Y-%m-%d").to_numpy(),
        tickers=np.array(TICKERS)
    )

    report = {
        "test_file": str(target),
        "test_sha256": sha256(target),
        "market_sha256": sha256(args.market),
        "sec_sha256": sha256(args.sec),
        "kg_sha256": sha256(args.kg),
        "provenance_sha256": sha256(provenance_path),
        "pit": pit,
        "dates": [
            str(mdates.min().date()),
            str(mdates.max().date())
        ],
        "n_days": len(mdates),
        "shape": list(features.shape),
        "model_hash": freeze["model_sha256"],
        "joint_features": "zero 9D Independent Fusion",
        "limitations": [
            "Test input sources require independently verified lineage.",
            "PIT audit alone does not certify all source availability.",
            "Source-feature equivalence relies on verified provenance.",
        ],
    }

    write_json(OUT / "test_build_manifest.json", report)

    print("\n========== TEST FEATURES BUILT ==========")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("PPO TEST: NOT YET EVALUATED")


def evaluate(args):
    from stable_baselines3 import PPO

    manifest = json.loads(
        (OUT / "test_build_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    freeze = json.loads(
        FROZEN.read_text(encoding="utf-8")
    )

    model_path = Path(freeze["model_path"])

    require(
        sha256(model_path) == freeze["model_sha256"],
        "Frozen PPO model hash mismatch"
    )

    test_path = Path(manifest["test_file"])

    require(
        sha256(test_path) == manifest["test_sha256"],
        "Test data hash mismatch"
    )

    result_dir = OUT / "sealed_results"

    require(
        not result_dir.exists(),
        "Sealed result directory already exists"
    )

    with np.load(
        test_path,
        allow_pickle=False
    ) as z:
        features = np.asarray(
            z["features"],
            dtype=np.float32
        )
        prices = np.asarray(
            z["prices"],
            dtype=np.float32
        )
        dates = pd.DatetimeIndex(
            pd.to_datetime(z["dates"].astype(str))
        )

        tickers = list(z["tickers"].astype(str))

    require(
        tickers == TICKERS,
        "Test stock order changed"
    )

    require(
        features.ndim == 3
        and features.shape[1:] == (5, 328),
        "Invalid test feature tensor"
    )

    model = PPO.load(
        str(model_path),
        device="cpu"
    )

    old_argv = sys.argv

    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        options = v16.parse_args()
    finally:
        sys.argv = old_argv

    env = v16.DualRAGPortfolioEnv(
        features,
        prices,
        dates,
        options.transaction_cost,
        options.reward_scale
    )

    require(
        env.observation_space.shape
        == model.observation_space.shape,
        "PPO model/environment observation mismatch"
    )

    # Prevent casual repeated sealed-test evaluation.
    result_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    (result_dir / "STARTED").write_text(
        "Frozen PPO sealed test started\n",
        encoding="utf-8"
    )

    history, metrics = v16.evaluate(
        model,
        env,
        "INDEPENDENT_FROZEN_TEST",
        int(freeze["selected_seed"]),
        "PPO"
    )

    history.to_csv(
        result_dir / "history.csv",
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        result_dir / "metrics.csv",
        index=False
    )

    report = {
        "frozen_arm": freeze["selected_arm"],
        "seed": freeze["selected_seed"],
        "model_sha256": freeze["model_sha256"],
        "test_sha256": manifest["test_sha256"],
        "test_start": str(dates.min().date()),
        "test_end": str(dates.max().date()),
        "test_days": len(dates),
        "metrics": metrics,
    }

    write_json(
        result_dir / "report.json",
        report
    )

    print("\n========== V20 BLIND TEST RESULT ==========")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print("\nBLIND TEST: COMPLETE")


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="mode",
        required=True
    )

    sub.add_parser("discover")

    b = sub.add_parser("build")
    b.add_argument("--market", required=True)
    b.add_argument("--sec", required=True)
    b.add_argument("--kg", required=True)
    b.add_argument("--pit-audit", required=True)
    b.add_argument("--provenance", required=True)

    sub.add_parser("evaluate")

    args = parser.parse_args()

    {
        "discover": discover,
        "build": build,
        "evaluate": evaluate
    }[args.mode](args)


if __name__ == "__main__":
    main()
