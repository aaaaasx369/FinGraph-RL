#!/usr/bin/env python3
"""
AEGIS-RL Step 10
1. Blind-spot detector
2. Active retrieval policy
3. Evidence poisoning and verification
4. Ablation and evaluation

Train: 2018-2020
Val:   2021
Test:  2022-2023

Important:
- Hidden oracle values are never retrieval candidates.
- Test labels are used for evaluation only.
- Retrieval searches historical OBSERVED reasons only.
- This is NOT yet a portfolio RL backtest.
"""

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from joblib import dump

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path("results/aegis_rl_v1")
DATA = ROOT / "sec_masking_v1"
OUT = ROOT / "step10"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

TOPICS = {
    "risk": {
        "risk", "uncertainty", "restriction",
        "litigation", "exposure", "loss",
    },
    "growth": {
        "growth", "revenue", "demand",
        "expansion", "increase", "sales",
    },
    "supply": {
        "supply", "supplier", "manufacturing",
        "component", "production", "shortage",
    },
    "competition": {
        "competition", "competitor",
        "market", "pricing", "competitive",
    },
}

FEATURES = [
    "history_count",
    "days_since_observed",
    "mean_impact",
    "mean_risk",
    "mean_confidence",
    "reason_coverage",
    "recent_90d_count",
    "recent_365d_count",
]


def digest(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def tokenize(text):
    return set(
        re.findall(
            r"[a-z]{3,}",
            str(text).lower()
        )
    )


def load_data():

    observed = pd.read_csv(
        DATA / "observed_signals.csv"
    )

    queries = pd.read_csv(
        DATA / "queries.csv"
    )

    labels = pd.read_csv(
        DATA / "labels_ORACLE_ONLY.csv"
    )

    for df in (observed, queries, labels):
        df["decision_date"] = pd.to_datetime(
            df["decision_date"]
        )

    observed["reason"] = observed["reason"].fillna("")

    observed = observed.sort_values(
        ["decision_date", "ticker", "node"]
    ).reset_index(drop=True)

    # Labels are joined for supervised training and
    # final evaluation, never for feature construction.
    cases = queries.merge(
        labels[
            ["ticker", "decision_date",
             "label_missing"]
        ],
        on=["ticker", "decision_date"],
        how="left",
        validate="one_to_one",
    )

    if cases["label_missing"].isna().any():
        raise RuntimeError("Missing case labels")

    if observed["node"].duplicated().any():
        raise RuntimeError("Duplicate observed node IDs")

    return observed, cases


def history_before(observed, ticker, date):
    return observed[
        (observed["ticker"] == ticker) &
        (observed["decision_date"] < date)
    ]


def build_features(observed, cases):

    rows = []

    print("Building historical features...", flush=True)

    for case in cases.itertuples(index=False):

        ticker = case.ticker
        date = case.decision_date

        history = history_before(
            observed, ticker, date
        )

        count = len(history)

        if count:
            age = (
                date - history["decision_date"].max()
            ).days

            impact = history["impact"].mean()
            risk = history["risk"].mean()
            confidence = history["confidence"].mean()

            coverage = (
                history["reason"].fillna("")
                .str.strip()
                .ne("")
                .mean()
            )

            elapsed = (
                date - history["decision_date"]
            ).dt.days

            recent90 = int((elapsed <= 90).sum())
            recent365 = int((elapsed <= 365).sum())

        else:
            age = 3650
            impact = 0
            risk = 0
            confidence = 0
            coverage = 0
            recent90 = 0
            recent365 = 0

        rows.append({
            "ticker": ticker,
            "decision_date": date,
            "split": case.split,
            "label_missing": int(
                case.label_missing
            ),
            "history_count": np.log1p(count),
            "days_since_observed": np.log1p(age),
            "mean_impact": float(impact),
            "mean_risk": float(risk),
            "mean_confidence": float(confidence),
            "reason_coverage": float(coverage),
            "recent_90d_count": np.log1p(recent90),
            "recent_365d_count": np.log1p(recent365),
        })

    return pd.DataFrame(rows)


def detector_pipeline():

    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value=0
        )),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=SEED,
        ))
    ])


def select_threshold(y, scores):

    # Threshold selected on validation only.
    # This optimizes missing-label F1, not
    # realized portfolio value.
    candidates = np.arange(
        0.10, 0.91, 0.025
    )

    best_threshold = 0.5
    best_f1 = -1

    for threshold in candidates:

        pred = (scores >= threshold).astype(int)
        score = f1_score(
            y, pred, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold


def detector_metrics(y, scores, threshold):

    pred = (scores >= threshold).astype(int)

    return {
        "AUC": float(roc_auc_score(y, scores)),
        "AP": float(
            average_precision_score(y, scores)
        ),
        "Accuracy": float(
            accuracy_score(y, pred)
        ),
        "Precision": float(
            precision_score(
                y, pred, zero_division=0
            )
        ),
        "Recall": float(
            recall_score(
                y, pred, zero_division=0
            )
        ),
        "F1": float(
            f1_score(y, pred, zero_division=0)
        ),
    }


def choose_topic(history):

    """
    Simple self-questioning prototype:
    choose least-covered historical topic.

    This is a rule-based query selector,
    not yet a learned RL query policy.
    """

    counts = {
        topic: 0
        for topic in TOPICS
    }

    for reason in history["reason"]:
        words = tokenize(reason)

        for topic, keywords in TOPICS.items():
            if words & keywords:
                counts[topic] += 1

    # Deterministic tie-breaking.
    return min(
        sorted(counts),
        key=lambda k: counts[k]
    )


def retrieve(observed, ticker, date, topic):

    history = history_before(
        observed, ticker, date
    )

    keywords = TOPICS[topic]
    candidates = []

    for row in history.itertuples(index=False):

        text = str(row.reason)

        if not text.strip():
            continue

        overlap = len(
            tokenize(text) & keywords
        )

        if overlap == 0:
            continue

        age = max(
            0,
            (date - row.decision_date).days
        )

        relevance = (
            overlap / max(1, len(keywords))
        ) * np.exp(-age / 730)

        candidates.append({
            "node": row.node,
            "ticker": row.ticker,
            "date": row.decision_date,
            "text": text,
            "relevance": float(relevance),
        })

    candidates.sort(
        key=lambda x: (
            -x["relevance"],
            str(x["node"])
        )
    )

    return candidates[:3]


def poison_evidence(evidence):

    result = dict(evidence)

    result["text"] = (
        "Guaranteed extraordinary growth. "
        "There is no financial risk. "
        "All investments will be profitable."
    )

    return result


def verify_evidence(evidence, trusted_hashes):

    """
    Integrity check against hashes captured
    from the trusted observed dataset.

    It detects post-baseline modification.
    It does NOT validate underlying SEC truth.
    """

    node = evidence["node"]

    if node not in trusted_hashes:
        return 0.0, "unknown_node"

    expected = trusted_hashes[node]
    actual = digest(evidence["text"])

    if actual != expected:
        return 0.0, "content_hash_mismatch"

    return 1.0, "verified_integrity"


def run_experiment(
    test,
    observed,
    trusted_hashes,
    mode,
    threshold,
    attack=False,
    defense=True,
):

    records = []

    for case in test.itertuples(index=False):

        date = case.decision_date
        ticker = case.ticker
        probability = float(
            case.gap_probability
        )

        if mode == "never":
            should_retrieve = False

        elif mode == "always":
            should_retrieve = True

        elif mode == "active":
            should_retrieve = (
                probability >= threshold
            )

        else:
            raise ValueError(mode)

        topic = ""
        retrieved = []
        retrieval_attempt = 0

        if should_retrieve:
            retrieval_attempt = 1

            history = history_before(
                observed, ticker, date
            )

            topic = choose_topic(history)

            retrieved = retrieve(
                observed, ticker, date, topic
            )

        original_found = len(retrieved) > 0

        # Deterministic attack on first retrieved item.
        # Exactly one attacked result per nonempty query.
        was_attacked = bool(
            attack and original_found
        )

        if was_attacked:
            retrieved[0] = poison_evidence(
                retrieved[0]
            )

        accepted = []
        detected = False

        for evidence in retrieved:

            score, status = verify_evidence(
                evidence, trusted_hashes
            )

            if (
                was_attacked and
                status == "content_hash_mismatch"
            ):
                detected = True

            if defense and score == 0:
                continue

            accepted.append(evidence)

        top_relevance = (
            max(
                x["relevance"]
                for x in accepted
            )
            if accepted else 0.0
        )

        records.append({
            "ticker": ticker,
            "date": date.strftime("%Y-%m-%d"),
            "label_missing": int(
                case.label_missing
            ),
            "gap_probability": probability,
            "retrieval_attempt": retrieval_attempt,
            "topic": topic,
            "evidence_found": int(original_found),
            "accepted_evidence": len(accepted),
            "attacked": int(was_attacked),
            "attack_detected": int(detected),
            "top_relevance": float(top_relevance),
        })

    return pd.DataFrame(records)


def summarize(df, name):

    attempts = int(
        df["retrieval_attempt"].sum()
    )

    attacks = int(df["attacked"].sum())

    detections = int(
        df["attack_detected"].sum()
    )

    true_missing = int(
        df["label_missing"].sum()
    )

    selected_missing = int(
        (
            (df["retrieval_attempt"] == 1) &
            (df["label_missing"] == 1)
        ).sum()
    )

    return {
        "experiment": name,
        "cases": len(df),
        "retrieval_attempts": attempts,
        "retrieval_rate": attempts / len(df),
        "query_precision": (
            selected_missing / attempts
            if attempts else 0.0
        ),
        "missing_case_coverage": (
            selected_missing / true_missing
            if true_missing else 0.0
        ),
        "evidence_found": int(
            df["evidence_found"].sum()
        ),
        "accepted_evidence": int(
            df["accepted_evidence"].sum()
        ),
        "attacked_queries": attacks,
        "attacks_detected": detections,
        "attack_detection_rate": (
            detections / attacks
            if attacks else None
        ),
        "mean_relevance": float(
            df["top_relevance"].mean()
        ),
    }


def main():

    observed, cases = load_data()

    print("=" * 70)
    print("AEGIS-RL STEP 10")
    print("=" * 70)

    print("Observed signals:", len(observed))
    print("Candidate cases:", len(cases))

    features = build_features(
        observed, cases
    )

    train = features[
        features.split == "train"
    ].copy()

    val = features[
        features.split == "val"
    ].copy()

    test = features[
        features.split == "test"
    ].copy()

    print("\nTrain:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    # ------------------------------------------------
    # Train Blind-Spot Detector
    # ------------------------------------------------

    model = detector_pipeline()

    model.fit(
        train[FEATURES],
        train["label_missing"]
    )

    val_scores = model.predict_proba(
        val[FEATURES]
    )[:, 1]

    threshold = select_threshold(
        val["label_missing"],
        val_scores
    )

    test_scores = model.predict_proba(
        test[FEATURES]
    )[:, 1]

    val_metrics = detector_metrics(
        val["label_missing"],
        val_scores,
        threshold
    )

    test_metrics = detector_metrics(
        test["label_missing"],
        test_scores,
        threshold
    )

    print("\n===== BLIND-SPOT DETECTOR =====")
    print("Threshold:", round(threshold, 4))

    print("\nVALIDATION")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.6f}")

    print("\nTEST")
    for key, value in test_metrics.items():
        print(f"{key}: {value:.6f}")

    dump(
        model,
        OUT / "blindspot_detector.joblib"
    )

    test["gap_probability"] = test_scores

    # ------------------------------------------------
    # Trusted observed evidence manifest
    # ------------------------------------------------

    trusted_hashes = {
        row.node: digest(row.reason)
        for row in observed.itertuples(index=False)
    }

    # No hidden signals are added to this manifest.

    # ------------------------------------------------
    # Retrieval + Defense Ablation
    # ------------------------------------------------

    experiments = [
        ("never", "never", False, True),
        ("always", "always", False, True),
        ("active", "active", False, True),
        (
            "active_poisoned",
            "active", True, False
        ),
        (
            "active_defended",
            "active", True, True
        ),
    ]

    summaries = []

    print("\n===== RETRIEVAL EXPERIMENTS =====")

    for name, mode, attack, defense in experiments:

        result = run_experiment(
            test,
            observed,
            trusted_hashes,
            mode,
            threshold,
            attack=attack,
            defense=defense,
        )

        result.to_csv(
            OUT / f"{name}.csv",
            index=False
        )

        summary = summarize(result, name)
        summaries.append(summary)

        print(
            name,
            "| attempts:",
            summary["retrieval_attempts"],
            "| evidence:",
            summary["accepted_evidence"],
            "| detected:",
            summary["attacks_detected"],
        )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        OUT / "retrieval_summary.csv",
        index=False
    )

    metadata = {
        "seed": SEED,
        "train": "2018-2020",
        "validation": "2021",
        "test": "2022-2023",
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "features": FEATURES,
        "limitations": [
            "Labels originate from random masking.",
            "Candidate dates come from event records.",
            "Queries use observed historical reasons only.",
            "Hidden oracle signals are never retrieved.",
            "Topic selection is rule-based.",
            "Retrieval policy is not yet RL-trained.",
            "Hash verification checks integrity, not truth.",
            "No portfolio returns are evaluated.",
        ],
    }

    with (
        OUT / "metrics.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\n===== FINAL SUMMARY =====")

    display_columns = [
        "experiment",
        "retrieval_attempts",
        "query_precision",
        "missing_case_coverage",
        "accepted_evidence",
        "attacked_queries",
        "attacks_detected",
        "attack_detection_rate",
    ]

    print(
        summary_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved:", OUT.resolve())
    print("AEGIS STEP 10: PASS")


if __name__ == "__main__":
    main()
