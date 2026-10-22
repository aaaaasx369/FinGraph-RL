#!/usr/bin/env python3
"""
V20.7
Build 2022-2023 Independent Fusion feature tensors.

No PPO training.
No model evaluation.
No fabricated inputs.
Outputs are UNCERTIFIED until PIT and training parity audits pass.
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from unified_event_memory.finrl_x.aegis_rl import (
    run_aegis_evidence_portfolio as original
)

import run_v16_dual_rag_finrl as v16


OUT = Path(
    "results/aegis_rl_v1/v20_blind"
)

OUT.mkdir(parents=True, exist_ok=True)

FROZEN = Path(
    "results/aegis_rl_v1/v19_final/"
    "frozen_model.json"
)

TICKERS = [
    "AAPL",
    "AMZN",
    "GOOGL",
    "MSFT",
    "NVDA",
]

TRAIN_END = pd.Timestamp("2020-12-31")
VAL_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2023-12-31")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def hash_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def save_npz(path, **arrays):
    path = Path(path)

    require(
        not path.exists(),
        f"Output already exists; refusing overwrite: {path}"
    )

    np.savez_compressed(path, **arrays)


def get_v16_options():
    original_argv = sys.argv

    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        options = v16.parse_args()
    finally:
        sys.argv = original_argv

    return options


def load_original_market():
    print(
        "\n========== ORIGINAL MARKET ==========",
        flush=True
    )

    schema, extra = original.load_schema()

    require(
        len(schema["FULL_DYNAMIC"]) == 53,
        "Original market schema is not 53D"
    )

    print(
        "Schema source:",
        original.SCHEMA_DIR,
        flush=True
    )

    df = original.load_market(
        "conservative",
        schema,
        extra,
    )

    required = {
        "date",
        "ticker",
        "split",
        "market_close",
    }

    require(
        required.issubset(df.columns),
        "Original market columns incomplete"
    )

    df = df.copy()
    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    require(
        not df.duplicated(
            ["date", "ticker"]
        ).any(),
        "Duplicate market rows"
    )

    for split, lo, hi in [
        ("train", "2018-01-01", "2020-12-31"),
        ("val", "2021-01-01", "2021-12-31"),
        ("test", "2022-01-01", "2023-12-31"),
    ]:
        sub = df[df["split"] == split]

        require(
            not sub.empty,
            f"Missing {split} split"
        )

        require(
            sub["date"].between(lo, hi).all(),
            f"Invalid {split} dates"
        )

        print(
            split,
            "rows=", len(sub),
            "days=", sub["date"].nunique(),
            "range=",
            sub["date"].min().date(),
            sub["date"].max().date(),
            flush=True
        )

    return df, schema


def make_market_arrays(df, schema):
    print(
        "\n========== MARKET 106D ==========",
        flush=True
    )

    dates = pd.DatetimeIndex(
        sorted(df["date"].unique())
    )

    feature_names = list(
        schema["FULL_DYNAMIC"]
    )

    require(
        len(feature_names) == 53,
        "53D feature schema changed"
    )

    n = len(dates)

    base = np.empty(
        (n, 5, 53),
        dtype=np.float32
    )

    prices = np.empty(
        (n, 5),
        dtype=np.float64
    )

    panel = df.set_index(
        ["date", "ticker"]
    )

    for j, ticker in enumerate(TICKERS):
        keys = pd.MultiIndex.from_arrays(
            [
                dates,
                np.repeat(ticker, n),
            ],
            names=["date", "ticker"]
        )

        selected = panel.reindex(keys)

        require(
            not selected[feature_names].isna().any().any(),
            f"Missing market features for {ticker}"
        )

        require(
            selected["market_close"].notna().all(),
            f"Missing prices for {ticker}"
        )

        base[:, j, :] = selected[
            feature_names
        ].to_numpy(dtype=np.float32)

        prices[:, j] = selected[
            "market_close"
        ].to_numpy(dtype=np.float64)

    require(
        np.isfinite(base).all(),
        "Nonfinite market features"
    )

    require(
        np.isfinite(prices).all()
        and (prices > 0).all(),
        "Invalid market prices"
    )

    # Exact structure from V16:
    # 53 original + 53 contemporaneous relative.
    cross_company_mean = base.mean(
        axis=1,
        keepdims=True
    )

    relative = base - cross_company_mean

    market106 = np.concatenate(
        [base, relative],
        axis=2
    ).astype(np.float32)

    require(
        market106.shape == (n, 5, 106),
        f"Market shape mismatch: {market106.shape}"
    )

    # Independently verify reconstruction.
    require(
        np.array_equal(
            market106[:, :, :53],
            base
        ),
        "Market base changed"
    )

    require(
        np.allclose(
            market106[:, :, 53:],
            base - base.mean(
                axis=1,
                keepdims=True
            ),
            atol=1e-6,
            rtol=1e-6
        ),
        "Relative market reconstruction failed"
    )

    print(
        "Full market shape:",
        market106.shape,
        flush=True
    )

    test_mask = (
        (dates >= TEST_START)
        & (dates <= TEST_END)
    )

    test_dates = dates[test_mask]
    test_market = market106[test_mask]
    test_prices = prices[test_mask]

    require(
        len(test_dates) > 300,
        "Test market coverage unexpectedly short"
    )

    require(
        test_dates.min().year == 2022
        and test_dates.max().year == 2023,
        "Test does not cover both 2022 and 2023"
    )

    print(
        "Test market shape:",
        test_market.shape,
        flush=True
    )

    return {
        "all_dates": dates,
        "all_prices": prices,
        "all_market": market106,
        "test_dates": test_dates,
        "test_market": test_market,
        "test_prices": test_prices,
        "feature_names": feature_names,
    }


def build_sec(data, args):
    print(
        "\n========== SEC 64D ==========",
        flush=True
    )

    all_dates = data["all_dates"]
    all_prices = data["all_prices"]
    test_dates = data["test_dates"]

    context = (
        v16.v8b.v8.market_features(
            all_prices
        )
    )

    require(
        context.shape == (
            len(all_dates), 5, 5
        ),
        f"Invalid SEC context: {context.shape}"
    )

    test_mask = (
        (all_dates >= TEST_START)
        & (all_dates <= TEST_END)
    )

    test_context = context[test_mask]

    sec, n_docs = v16.v13.load_sec(
        test_dates,
        test_context,
        args.pit_sec,
        args.sec_top_k,
        OUT / "v20_7_sec_retrieval_audit.csv",
    )

    require(
        sec.shape == (
            len(test_dates), 5, 64
        ),
        f"SEC shape mismatch: {sec.shape}"
    )

    require(
        np.isfinite(sec).all(),
        "SEC has NaN/Inf"
    )

    print(
        "SEC documents:", n_docs,
        flush=True
    )

    print(
        "SEC feature shape:", sec.shape,
        flush=True
    )

    print(
        "SEC nonzero ratio:",
        float(np.mean(sec != 0)),
        flush=True
    )

    return sec, n_docs


def build_kg(data, args):
    print(
        "\n========== KG 149D ==========",
        flush=True
    )

    # Historical relation coordinates must be
    # established before the test period.
    historical = (
        v16.v13.kg_events_from_csv(
            args.temporal_kg,
            VAL_END
        )
    )

    relations = sorted(
        historical["relation"].unique().tolist()
    )

    require(
        len(relations) == 19,
        (
            "Historical KG vocabulary is not 19 relations: "
            f"{len(relations)}"
        )
    )

    all_events = (
        v16.v13.kg_events_from_csv(
            args.temporal_kg,
            TEST_END
        )
    )

    new_relations = sorted(
        set(all_events["relation"])
        - set(relations)
    )

    if new_relations:
        print(
            "New Test-only relations excluded:",
            new_relations,
            flush=True
        )

    # Preserve original relation coordinates.
    existing = all_events[
        all_events["relation"].isin(
            relations
        )
    ].copy()

    kg, actual_relations = (
        v16.v13.build_kg_rag(
            data["test_dates"],
            existing,
            args.kg_top_k,
            OUT,
        )
    )

    require(
        actual_relations == relations,
        "KG relation order mismatch"
    )

    require(
        kg.shape == (
            len(data["test_dates"]), 5, 149
        ),
        f"KG shape mismatch: {kg.shape}"
    )

    require(
        np.isfinite(kg).all(),
        "KG contains NaN/Inf"
    )

    print(
        "KG feature shape:", kg.shape,
        flush=True
    )

    print(
        "KG nonzero ratio:",
        float(np.mean(kg != 0)),
        flush=True
    )

    return kg, relations, new_relations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["market", "all"],
        default="market"
    )

    options = parser.parse_args()

    require(
        FROZEN.exists(),
        "Frozen model manifest missing"
    )

    frozen = json.loads(
        FROZEN.read_text(encoding="utf-8")
    )

    require(
        frozen["selected_arm"] == "INDEPENDENT",
        "Selected arm is no longer INDEPENDENT"
    )

    require(
        int(frozen["selected_seed"]) == 42,
        "Frozen seed changed"
    )

    require(
        hash_file(frozen["model_path"])
        == frozen["model_sha256"],
        "Frozen model SHA256 mismatch"
    )

    df, schema = load_original_market()

    data = make_market_arrays(
        df,
        schema
    )

    dates = data["test_dates"]
    market = data["test_market"]
    prices = data["test_prices"]

    arrays = {
        "dates": dates.strftime(
            "%Y-%m-%d"
        ).to_numpy(),
        "tickers": np.array(TICKERS),
    }

    market_path = (
        OUT / "market_2022_2023_UNCERTIFIED.npz"
    )

    save_npz(
        market_path,
        **arrays,
        market=market,
        prices=prices,
    )

    market_report = {
        "status": "UNCERTIFIED",
        "source": str(
            original.SCHEMA_DIR
            / "dataset_conservative.csv"
        ),
        "schema": str(
            original.SCHEMA_DIR
            / "feature_sets.json"
        ),
        "test_start": str(dates.min().date()),
        "test_end": str(dates.max().date()),
        "test_days": len(dates),
        "tickers": TICKERS,
        "market_shape": list(market.shape),
        "market_feature_names": (
            data["feature_names"]
            + [
                "relative_" + name
                for name in data["feature_names"]
            ]
        ),
        "market_file": str(market_path),
        "market_sha256": hash_file(market_path),
        "model_sha256": frozen["model_sha256"],
        "limitations": [
            "Precomputed market feature PIT provenance not yet certified.",
            "Market feature parity with V18.1 training observations not yet independently verified.",
            "Same-day feature availability relative to trade execution is unresolved.",
        ],
    }

    write_json(
        OUT / "v20_7_market_report.json",
        market_report,
    )

    print(
        "\nMARKET 106D STAGING COMPLETE",
        flush=True
    )

    if options.mode == "market":
        print(
            "SEC/KG not requested.",
            flush=True
        )
        print(
            "PPO Test: NOT ACCESSED",
            flush=True
        )
        return

    args = get_v16_options()

    sec, n_docs = build_sec(
        data, args
    )

    kg, relations, new_relations = build_kg(
        data, args
    )

    zero_joint = np.zeros(
        (len(dates), 5, 9),
        dtype=np.float32
    )

    # Explicit order avoids ambiguity between
    # 3-argument and 4-argument compose functions.
    features = np.concatenate(
        [
            market,
            sec,
            kg,
            zero_joint
        ],
        axis=2
    ).astype(np.float32)

    require(
        features.shape == (
            len(dates), 5, 328
        ),
        f"Final shape mismatch: {features.shape}"
    )

    require(
        np.isfinite(features).all(),
        "Final feature tensor invalid"
    )

    sec_path = (
        OUT / "sec_2022_2023_UNCERTIFIED.npz"
    )

    kg_path = (
        OUT / "kg_2022_2023_UNCERTIFIED.npz"
    )

    final_path = (
        OUT / "test_2022_2023_UNCERTIFIED.npz"
    )

    save_npz(
        sec_path,
        **arrays,
        sec=sec
    )

    save_npz(
        kg_path,
        **arrays,
        kg=kg
    )

    save_npz(
        final_path,
        **arrays,
        features=features,
        prices=prices,
    )

    report = {
        "status": "UNCERTIFIED",
        "test_days": len(dates),
        "dates": [
            str(dates.min().date()),
            str(dates.max().date())
        ],
        "ticker_order": TICKERS,
        "market_shape": list(market.shape),
        "sec_shape": list(sec.shape),
        "kg_shape": list(kg.shape),
        "joint_shape": list(zero_joint.shape),
        "final_shape": list(features.shape),
        "sec_documents": n_docs,
        "kg_relation_names": relations,
        "test_only_relations_excluded": new_relations,
        "frozen_model_sha256": (
            frozen["model_sha256"]
        ),
        "inputs": {
            "original_market_dataset": str(
                original.SCHEMA_DIR
                / "dataset_conservative.csv"
            ),
            "sec_pit": args.pit_sec,
            "temporal_kg": args.temporal_kg,
        },
        "files": {
            "market": str(market_path),
            "sec": str(sec_path),
            "kg": str(kg_path),
            "combined": str(final_path),
        },
        "sha256": {
            "market": hash_file(market_path),
            "sec": hash_file(sec_path),
            "kg": hash_file(kg_path),
            "combined": hash_file(final_path),
        },
        "blocking_checks": [
            "Verify Market 106D parity with original training pipeline.",
            "Verify feature generation uses information available at the portfolio decision time.",
            "Verify SEC filing availability and retrieval lineage.",
            "Verify KG source timestamps and relation-coordinate parity.",
            "Confirm test data has not influenced hyperparameter or model selection.",
        ],
    }

    write_json(
        OUT / "v20_7_build_report.json",
        report,
    )

    print(
        "\n========== V20.7 BUILD COMPLETE ==========",
        flush=True
    )

    print(
        "Market:", market.shape,
        flush=True
    )

    print(
        "SEC:", sec.shape,
        flush=True
    )

    print(
        "KG:", kg.shape,
        flush=True
    )

    print(
        "Joint:", zero_joint.shape,
        flush=True
    )

    print(
        "PPO input:", features.shape,
        flush=True
    )

    print(
        "\nOUTPUT:", final_path,
        flush=True
    )

    print(
        "STATUS: UNCERTIFIED",
        flush=True
    )

    print(
        "PPO TEST PERFORMANCE: NOT ACCESSED",
        flush=True
    )


if __name__ == "__main__":
    main()
