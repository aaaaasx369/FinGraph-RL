#!/usr/bin/env python3

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RAG_ROOT = ROOT / "results/light_rag_dynamic_kg"

OUT = RAG_ROOT / "finrl_bridge"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

# Proposed NEW schema, not the existing V18.1 Joint 9D.
FEATURES = [
    "impact",
    "risk",
    "confidence",
    "impact_valid",
    "risk_valid",
    "confidence_valid",
    "evidence_age",
    "evidence_coverage",
    "grounding_flag",
]

CONDITIONS = {
    "M0": [],
    "M1": ["KG"],
    "M2": ["SEC"],
    "M3": ["KG", "SEC"],
}


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def finite_float(value, low, high):
    if value is None or isinstance(value, bool):
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(value):
        return None

    if not low <= value <= high:
        return None

    return value


def normalize_signal(record):
    """
    Accept A/B/C ablation JSON.

    Keep numerical values as candidates only.
    Do not certify grounding merely because a quote
    was found in a source document.
    """

    validation = record.get("validation", {})

    generated = record.get("generated") or {}

    if not isinstance(validation, dict):
        validation = {}

    if not isinstance(generated, dict):
        generated = {}

    ranges = {
        "impact": (-1, 1),
        "risk": (0, 1),
        "confidence": (0, 1),
    }

    scores = {}

    for name, (low, high) in ranges.items():
        scores[name] = finite_float(
            validation.get(name),
            low,
            high,
        )

    status = str(
        validation.get("status", "unknown")
    )

    quote_passed = bool(
        validation.get(
            "quote_checks_passed",
            False,
        )
    )

    semantic_verified = bool(
        validation.get(
            "semantic_verified",
            False,
        )
    )

    # Source-count based coverage is only a retrieval
    # diagnostic, NOT semantic evidence coverage.
    kg_count = int(record.get("kg_count") or 0)
    sec_count = int(record.get("sec_count") or 0)

    return {
        "scores": scores,
        "status": status,
        "quote_passed": quote_passed,
        "semantic_verified": semantic_verified,
        "kg_count": kg_count,
        "sec_count": sec_count,
        "generated": generated,
    }


def make_features(info, source):
    scores = dict(info["scores"])

    # Enforce semantic verification for PPO-valid
    # RAG features.
    verified = (
        info["semantic_verified"]
        and info["quote_passed"]
        and info["status"] == "supported"
    )

    if not verified:
        scores = {
            "impact": None,
            "risk": None,
            "confidence": None,
        }

    values = []

    for key in ("impact", "risk", "confidence"):
        value = scores[key]
        values.append(0.0 if value is None else value)

    for key in ("impact", "risk", "confidence"):
        values.append(
            0.0 if scores[key] is None else 1.0
        )

    # Age=0 on the date the signal is generated.
    # A trading-day aligned forward-fill pipeline
    # must compute future ages separately.
    values.append(0.0)

    observed = (
        info["kg_count"] if source == "KG"
        else info["sec_count"]
    )

    budget = 12 if source == "KG" else 5

    values.append(
        min(float(observed) / budget, 1.0)
    )

    values.append(1.0 if verified else 0.0)

    return values


def scan_rag_outputs():
    paths = sorted(
        (RAG_ROOT / "ablation_v1").glob(
            "*_A_KG_RAG.json"
        )
    )

    paths += sorted(
        (RAG_ROOT / "ablation_v1").glob(
            "*_B_SEC_RAG.json"
        )
    )

    paths += sorted(
        (RAG_ROOT / "ablation_v1").glob(
            "*_C_KG_SEC_RAG.json"
        )
    )

    rows = []
    audit = []

    for path in paths:
        record = load_json(path)

        if not isinstance(record, dict):
            continue

        ticker = record.get("ticker")
        date = record.get("date")
        group = record.get("group")

        if ticker not in TICKERS:
            continue

        if not isinstance(date, str):
            continue

        date_value = pd.to_datetime(
            date, errors="coerce"
        )

        if pd.isna(date_value):
            continue

        if not (
            pd.Timestamp("2018-01-01")
            <= date_value
            < pd.Timestamp("2022-01-01")
        ):
            continue

        # C must remain a separate joint condition.
        # Do not split its scores and mislabel them
        # as independent KG or SEC signals.
        group_to_source = {
            "A_KG_RAG": "KG",
            "B_SEC_RAG": "SEC",
            "C_KG_SEC_RAG": "KG_SEC",
        }

        source = group_to_source.get(group)

        if source is None:
            continue

        info = normalize_signal(record)

        features = make_features(
            info,
            "SEC" if source == "SEC" else "KG",
        )

        row = {
            "ticker": ticker,
            "date": date_value.strftime("%Y-%m-%d"),
            "source": source,
        }

        row.update(dict(zip(FEATURES, features)))

        rows.append(row)

        audit.append({
            "ticker": ticker,
            "date": row["date"],
            "source": source,
            "raw_impact": info["scores"]["impact"],
            "raw_risk": info["scores"]["risk"],
            "raw_confidence": info["scores"]["confidence"],
            "kg_count": info["kg_count"],
            "sec_count": info["sec_count"],
            "quote_passed": info["quote_passed"],
            "semantic_verified": info["semantic_verified"],
            "grounding_status": info["status"],
            "path": str(path),
        })

    return pd.DataFrame(rows), pd.DataFrame(audit)


def inspect_finrl_code():
    """
    Discover candidate native FinRL training files.
    Never assume their training API signatures.
    """

    roots = [
        ROOT / "experiments",
        ROOT / "unified_event_memory",
        ROOT / "finrl_x",
    ]

    keywords = (
        "ppo",
        "ddpg",
        "a2c",
        "joint",
        "observation",
        "feature_extractor",
        "train",
    )

    candidates = []

    for base in roots:
        if not base.exists():
            continue

        for path in base.rglob("*.py"):
            if (
                "light_rag_dynamic_kg"
                in path.parts
            ):
                continue

            if not any(
                word in path.name.lower()
                for word in keywords
            ):
                continue

            try:
                text = path.read_text(
                    encoding="utf-8"
                )
            except (OSError, UnicodeDecodeError):
                continue

            relevant = [
                "GatedEvidenceExtractor",
                "Joint",
                "joint",
                "PPO(",
                "A2C(",
                "DDPG(",
                "observation_space",
                "action_space",
            ]

            hits = [
                word for word in relevant
                if word in text
            ]

            if not hits:
                continue

            functions = []

            try:
                tree = ast.parse(text)

                functions = [
                    node.name
                    for node in ast.walk(tree)
                    if isinstance(
                        node,
                        (ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and any(
                        key in node.name.lower()
                        for key in (
                            "train",
                            "feature",
                            "joint",
                            "data",
                            "env",
                            "replay",
                        )
                    )
                ][:25]

            except SyntaxError:
                pass

            candidates.append({
                "path": str(
                    path.relative_to(ROOT)
                ),
                "hits": ",".join(hits),
                "functions": ",".join(functions),
            })

    return pd.DataFrame(candidates)


def main():
    features, audit = scan_rag_outputs()

    print("\n" + "=" * 76)
    print("GENERATIVE RAG FEATURE AUDIT")
    print("=" * 76)

    if features.empty:
        print("[WARNING] No valid A/B/C output JSON found.")
    else:
        features = features.sort_values(
            ["date", "ticker", "source"]
        )

        feature_path = OUT / "rag_feature_candidates.csv"

        features.to_csv(
            feature_path,
            index=False,
        )

        audit.to_csv(
            OUT / "rag_evidence_audit.csv",
            index=False,
        )

        print("[RAG ROWS]", len(features))
        print("[TICKERS]", features.ticker.unique().tolist())
        print("[DATES]", features.date.nunique())

        print("\n[FEATURE EXAMPLE]")
        print(features.head(12).to_string(index=False))

        print("\n[RAW EVIDENCE AUDIT]")
        print(audit.head(12).to_string(index=False))

        valid_rows = (
            features[
                [
                    "impact_valid",
                    "risk_valid",
                    "confidence_valid",
                ]
            ].sum(axis=1) > 0
        )

        print(
            "\n[ROWS WITH VERIFIED NUMERIC SIGNALS]",
            int(valid_rows.sum()),
        )

        print("[SAVED]", feature_path)

    print("\n" + "=" * 76)
    print("NATIVE FINRL-X TRAINING CODE AUDIT")
    print("=" * 76)

    candidates = inspect_finrl_code()

    if candidates.empty:
        print("[WARNING] No matching native code files found.")
    else:
        candidates = candidates.sort_values(
            "path"
        )

        candidates.to_csv(
            OUT / "finrl_training_candidates.csv",
            index=False,
        )

        print(
            candidates.to_string(
                index=False,
                max_rows=80,
                max_colwidth=95,
            )
        )

    print("\n" + "=" * 76)
    print("TRAIN / VALIDATION / TEST POLICY")
    print("=" * 76)

    print("TRAIN: 2018-01-01 .. 2020-12-31")
    print("VAL:   2021-01-01 .. 2021-12-31")
    print("TEST:  2022-01-01 .. 2023-12-31")
    print("ALGORITHMS: A2C, PPO, DDPG")
    print("CONDITIONS:  M0, M1, M2, M3")
    print("TOTAL: 12 combinations per seed")

    print("\n[IMPORTANT]")
    print(
        "No training started: native observation/"
        "environment interface is not yet verified."
    )
    print(
        "Sparse one-date RAG signals must not be "
        "treated as full historical coverage."
    )
    print(
        "Quote matching alone does not certify "
        "semantic grounding."
    )

    print("\n[OUTPUT DIRECTORY]", OUT)


if __name__ == "__main__":
    main()
