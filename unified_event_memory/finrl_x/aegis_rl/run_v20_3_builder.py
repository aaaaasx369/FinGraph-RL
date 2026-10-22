#!/usr/bin/env python3
"""
V20.3 Independent Fusion Test Feature Builder.

INSPECT: Verify training feature contract without accessing Test.
BUILD:   Attempt genuine 2022-2023 feature construction.

Does NOT evaluate the sealed PPO.
Does NOT retrain a model.
Does NOT invent missing market observations.
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

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

OUT.mkdir(parents=True, exist_ok=True)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def get_args():
    old = sys.argv
    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        args = v16.parse_args()
    finally:
        sys.argv = old
    return args


def load_training(args):
    args.algo = "A2C"

    train, val, dates, contexts, n_train = (
        v16.v8b.load_data(args)
    )

    args.algo = "PPO"

    dates = v16.base.normalize_dates(dates)

    market_flat = np.concatenate(
        [train["market"], val["market"]],
        axis=0
    )

    prices = np.concatenate(
        [train["prices"], val["prices"]],
        axis=0
    )

    market, _ = v16.base.validate_market_layout(
        market_flat
    )

    check(
        market.shape == (len(dates), 5, 106),
        f"Training market contract changed: {market.shape}"
    )

    check(
        prices.shape == (len(dates), 5),
        "Training prices inconsistent"
    )

    check(
        dates.min().year == 2018
        and dates.max().year == 2021
        and dates[n_train].year == 2021,
        "Unexpected training/validation split"
    )

    return train, val, dates, prices, market, n_train


def inspect_contract(args):
    print("\n========== TRAINING CONTRACT ==========")

    frozen = json.loads(
        FROZEN.read_text(encoding="utf-8")
    )

    check(
        frozen["selected_arm"] == "INDEPENDENT",
        "Frozen model is not Independent Fusion"
    )

    check(
        frozen["selected_seed"] == 42,
        "Unexpected frozen seed"
    )

    check(
        sha256(frozen["model_path"])
        == frozen["model_sha256"],
        "Frozen model hash mismatch"
    )

    train, val, dates, prices, market, n_train = (
        load_training(args)
    )

    print("Dates:", dates.min(), dates.max())
    print("Train days:", n_train)
    print("Validation days:", len(dates) - n_train)
    print("Market:", market.shape)
    print("Prices:", prices.shape)

    print("\n========== TEST SPLIT LOADER ==========")

    loader = v16.v8b.v7b.load_split

    print("Loader module:", loader.__module__)
    print("Loader signature:", inspect.signature(loader))

    source = inspect.getsource(loader)

    print("\nLoader source:")
    print(source[:16000])

    print("\n========== SOURCE EXISTENCE ==========")

    for label, path in [
        ("SEC PIT", args.pit_sec),
        ("Temporal KG", args.temporal_kg),
        ("Daily KGT", args.kgt),
    ]:
        p = Path(path)
        print(
            label,
            "EXISTS" if p.exists() else "MISSING",
            p,
        )

    print("\n========== V20.3 INSPECTION COMPLETE ==========")
    print("Test data not loaded.")
    print("No PPO evaluation.")

    return dates, prices, market


def load_test_market(args):
    """
    Reuse the EXACT upstream split loader.
    Reject unsupported test splits rather than changing
    training data transformations.
    """
    try:
        test = v16.v8b.v7b.load_split(
            args, "test"
        )
    except Exception as exc:
        raise RuntimeError(
            "Existing upstream loader could not load a genuine "
            "2022-2023 test split. Do not create fake NPZ files. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc

    check(
        isinstance(test, dict),
        "Test loader must return a data dictionary"
    )

    required = {"market", "prices", "dates"}

    check(
        required.issubset(test),
        f"Test data missing keys: {required-set(test)}"
    )

    dates = v16.base.normalize_dates(
        test["dates"]
    )

    market_flat = np.asarray(
        test["market"],
        dtype=np.float32
    )

    prices = np.asarray(
        test["prices"],
        dtype=np.float32
    )

    check(
        len(dates) > 300,
        "Unexpectedly short test interval"
    )

    check(
        dates.min() >= pd.Timestamp("2022-01-01")
        and dates.max() <= pd.Timestamp("2023-12-31"),
        f"Test dates invalid: {dates.min()} ... {dates.max()}"
    )

    check(
        dates.is_unique
        and dates.is_monotonic_increasing,
        "Test dates are duplicated or unordered"
    )

    check(
        prices.shape == (len(dates), 5),
        f"Test prices invalid: {prices.shape}"
    )

    check(
        np.isfinite(prices).all()
        and (prices > 0).all(),
        "Test prices are missing or invalid"
    )

    market, _ = v16.base.validate_market_layout(
        market_flat
    )

    check(
        market.shape == (len(dates), 5, 106),
        f"Test market contract mismatch: {market.shape}"
    )

    return dates, prices, market


def build(args):
    print("\n========== VALIDATE FROZEN MODEL ==========")

    frozen = json.loads(
        FROZEN.read_text(encoding="utf-8")
    )

    check(
        frozen["selected_arm"] == "INDEPENDENT",
        "Selected model changed"
    )

    check(
        sha256(frozen["model_path"])
        == frozen["model_sha256"],
        "Frozen model hash mismatch"
    )

    print("\n========== LOAD TRAINING HISTORY ==========")

    _, _, train_dates, train_prices, train_market, _ = (
        load_training(args)
    )

    print("\n========== LOAD GENUINE TEST SPLIT ==========")

    test_dates, test_prices, test_market = (
        load_test_market(args)
    )

    check(
        train_dates.max() < test_dates.min(),
        "Train/test chronological overlap"
    )

    print("Test days:", len(test_dates))
    print("Market shape:", test_market.shape)

    print("\n========== HISTORICAL SEC CONTEXT ==========")

    # Use full chronological price history for the same
    # rolling-context method as training.
    all_prices = np.concatenate(
        [train_prices, test_prices],
        axis=0
    )

    all_dates = train_dates.append(test_dates)

    check(
        all_dates.is_unique
        and all_dates.is_monotonic_increasing,
        "Combined history is not chronological"
    )

    context_all = v16.v8b.v8.market_features(
        all_prices
    )

    check(
        context_all.shape == (len(all_dates), 5, 5),
        "SEC retrieval context shape mismatch"
    )

    context_test = context_all[len(train_dates):]

    print("Test context:", context_test.shape)

    print("\n========== BUILD SEC 64D ==========")

    sec, documents = v16.v13.load_sec(
        test_dates,
        context_test,
        args.pit_sec,
        args.sec_top_k,
        OUT / "v20_3_sec_retrieval_audit.csv",
    )

    check(
        sec.shape == (len(test_dates), 5, 64),
        f"SEC feature mismatch: {sec.shape}"
    )

    print("SEC documents:", documents)
    print("SEC shape:", sec.shape)

    print("\n========== FREEZE KG RELATION VOCAB ==========")

    # Training vocabulary must be defined from data
    # available within the original 2018-2021 period.
    historical_events = (
        v16.v13.kg_events_from_csv(
            args.temporal_kg,
            pd.Timestamp("2021-12-31")
        )
    )

    train_relations = sorted(
        historical_events["relation"].unique().tolist()
    )

    print("Frozen relation count:", len(train_relations))

    # This must agree with the exact 149D training design:
    # 64 + 64 + number_of_relations + 2.
    expected_kg_dim = (
        64 + 64 + len(train_relations) + 2
    )

    check(
        expected_kg_dim == 149,
        (
            "Training KG relation vocabulary incompatible "
            f"with frozen 149D feature contract: {expected_kg_dim}"
        )
    )

    print("\n========== BUILD TEST KG 149D ==========")

    all_events = v16.v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2023-12-31")
    )

    # Preserve old relation coordinates. New relation types
    # cannot be inserted into the frozen 149D representation.
    new_relations = sorted(
        set(all_events["relation"]) - set(train_relations)
    )

    if new_relations:
        print(
            "[WARNING] Test-only KG relations excluded:",
            new_relations
        )

    all_events = all_events[
        all_events["relation"].isin(train_relations)
    ].copy()

    kg, relation_names = v16.v13.build_kg_rag(
        test_dates,
        all_events,
        args.kg_top_k,
        OUT,
    )

    check(
        relation_names == train_relations,
        "KG relation order changed"
    )

    check(
        kg.shape == (len(test_dates), 5, 149),
        f"KG feature mismatch: {kg.shape}"
    )

    print("KG shape:", kg.shape)

    print("\n========== INDEPENDENT FUSION ==========")

    zero_joint = np.zeros(
        (len(test_dates), 5, 9),
        dtype=np.float32
    )

    features = v16.base.compose(
        test_market,
        sec,
        kg,
        zero_joint
    )

    check(
        features.shape == (len(test_dates), 5, 328),
        f"Unexpected PPO feature shape: {features.shape}"
    )

    check(
        np.isfinite(features).all(),
        "Nonfinite test features"
    )

    print("Final features:", features.shape)

    # These are STAGING artifacts, not certified sealed-test
    # inputs. PIT source lineage and normalization must still
    # be independently verified before evaluation.
    target = OUT / "v20_3_test_features_UNCERTIFIED.npz"

    check(
        not target.exists(),
        "Staging artifact already exists; refusing overwrite"
    )

    np.savez_compressed(
        target,
        features=features,
        prices=test_prices,
        dates=test_dates.strftime("%Y-%m-%d").to_numpy(),
        tickers=np.array(TICKERS),
    )

    report = {
        "experiment": "V20.3 Test Feature Staging",
        "status": "UNCERTIFIED",
        "test_start": str(test_dates.min().date()),
        "test_end": str(test_dates.max().date()),
        "test_days": len(test_dates),
        "features_shape": list(features.shape),
        "ticker_order": TICKERS,
        "frozen_relations": train_relations,
        "test_only_relations_excluded": new_relations,
        "sec_document_count": documents,
        "test_staging_file": str(target),
        "staging_sha256": sha256(target),
        "frozen_model_sha256": frozen["model_sha256"],
        "blocking_checks": [
            "Confirm actual 2022-2023 upstream market data coverage.",
            "Confirm SEC retrieval has 2022-2023 documents.",
            "Verify SEC disclosure timestamps, not filing year alone.",
            "Verify KG first-observed times are valid PIT availability.",
            "Confirm frozen market normalization and feature order.",
            "Confirm Test timestamps cover intended trading decisions.",
            "Do not evaluate a model using uncertified staging data.",
        ],
    }

    save_json(
        OUT / "v20_3_build_report.json",
        report
    )

    print("\n========== V20.3 BUILD COMPLETE ==========")
    print("File:", target)
    print("Shape:", features.shape)
    print("Status: UNCERTIFIED")
    print("PPO test performance: NOT ACCESSED")


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        choices=["inspect", "build"],
        required=True
    )

    opt = p.parse_args()
    args = get_args()

    if opt.mode == "inspect":
        inspect_contract(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
