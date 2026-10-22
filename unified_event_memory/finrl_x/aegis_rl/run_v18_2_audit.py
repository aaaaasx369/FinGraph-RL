#!/usr/bin/env python3
"""
V18.2 Reproducibility and Grounding Quality Audit.

Read-only for existing V18/V18.1/B7/B8 artifacts.
Does not train PPO, call LLMs, or access 2022-2023 test data.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/aegis_rl_v1")
OUT = ROOT / "v18_2_audit"

RUNS = {
    "V18": ROOT / "v18_threeway_ppo_full",
    "V18_1": ROOT / "v18_1/ppo_full",
}

B7 = ROOT / "v18_b7_matched_20/audit.jsonl"
B8 = ROOT / "v18_b8_matched_20/audit.jsonl"

SEEDS = [42, 123, 2026]
ARMS = [
    "INDEPENDENT",
    "UNVERIFIED_JOINT",
    "GROUNDED_JOINT",
]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def save_csv(path, rows, columns=None):
    if columns is None:
        columns = list(rows[0]) if rows else []

    with path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_runs():
    runs = {}

    for name, folder in RUNS.items():
        manifest_path = folder / "manifest.json"
        metrics_path = folder / "portfolio_summary.csv"
        tensor_path = folder / "joint_feature_tensors.npz"

        for path in [
            manifest_path,
            metrics_path,
            tensor_path,
        ]:
            if not path.exists():
                raise FileNotFoundError(path)

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        metrics = pd.read_csv(metrics_path)

        if int(manifest["timesteps"]) != 30000:
            raise RuntimeError(
                f"{name}: expected 30000 timesteps"
            )

        if set(map(int, manifest["seeds"])) != set(SEEDS):
            raise RuntimeError(
                f"{name}: unexpected seeds"
            )

        if len(metrics) != 9:
            raise RuntimeError(
                f"{name}: expected 9 model results"
            )

        metrics["seed"] = metrics["seed"].astype(int)

        actual = set(
            zip(metrics["arm"], metrics["seed"])
        )

        expected = {
            (arm, seed)
            for arm in ARMS
            for seed in SEEDS
        }

        if actual != expected:
            raise RuntimeError(
                f"{name}: incomplete or duplicate arm/seed"
            )

        with np.load(
            tensor_path, allow_pickle=False
        ) as z:
            tensors = {
                key: np.array(z[key], copy=True)
                for key in ["original", "grounded", "independent"]
            }

        for key, arr in tensors.items():
            if not np.isfinite(arr).all():
                raise RuntimeError(
                    f"{name} {key}: nonfinite features"
                )

        runs[name] = {
            "folder": folder,
            "manifest": manifest,
            "metrics": metrics,
            "tensors": tensors,
            "hashes": {
                "manifest": sha256(manifest_path),
                "portfolio_summary": sha256(metrics_path),
                "joint_tensors": sha256(tensor_path),
            },
        }

    return runs


def reproducibility(runs):
    print("\n========== EXPERIMENT CONFIGURATION ==========")

    for name, run in runs.items():
        m = run["manifest"]

        print(
            name,
            "timesteps=", m.get("timesteps"),
            "seeds=", m.get("seeds"),
            "max_age=", m.get("max_age_days"),
            "joint_dim=", m.get(
                "joint_feature_dimension", "not declared"
            ),
        )

    print("\n========== TENSOR COMPARISON ==========")

    comparisons = []

    for key in [
        "original",
        "grounded",
        "independent",
    ]:
        a = runs["V18"]["tensors"][key]
        b = runs["V18_1"]["tensors"][key]

        if a.shape[:2] != b.shape[:2]:
            raise RuntimeError(
                f"Date/company tensor shape mismatch: {key}"
            )

        n = min(a.shape[-1], b.shape[-1])

        common_equal = np.array_equal(
            a[..., :n], b[..., :n]
        )

        max_diff = float(np.max(
            np.abs(a[..., :n] - b[..., :n])
        ))

        extra_a = a[..., n:]
        extra_b = b[..., n:]

        extra_b_nonzero = int(
            np.count_nonzero(extra_b)
        )

        print(
            key,
            "V18=", a.shape,
            "V18.1=", b.shape,
            "common_equal=", common_equal,
            "max_diff=", max_diff,
            "new_nonzero=", extra_b_nonzero,
        )

        comparisons.append({
            "tensor": key,
            "v18_shape": str(a.shape),
            "v18_1_shape": str(b.shape),
            "common_dimensions_equal": common_equal,
            "max_common_abs_difference": max_diff,
            "new_dimensions_nonzero_count": extra_b_nonzero,
        })

    save_csv(
        OUT / "tensor_comparison.csv",
        comparisons
    )

    print("\n========== MODEL ARTIFACT HASHES ==========")

    hashes = []

    for name, run in runs.items():
        for seed in SEEDS:
            for arm in ARMS:
                path = run["folder"] / (
                    f"{arm}_PPO_seed{seed}.zip"
                )

                if not path.exists():
                    raise FileNotFoundError(path)

                hashes.append({
                    "version": name,
                    "arm": arm,
                    "seed": seed,
                    "model_sha256": sha256(path),
                    "model_bytes": path.stat().st_size,
                    "model_path": str(path),
                })

    save_csv(
        OUT / "model_hashes.csv",
        hashes
    )

    print("Model artifacts checked:", len(hashes))

    return comparisons


def metrics_analysis(runs):
    print("\n========== PER-SEED RESULTS ==========")

    rows = []

    for name, run in runs.items():
        df = run["metrics"]

        for _, r in df.iterrows():
            rows.append({
                "version": name,
                "arm": r["arm"],
                "seed": int(r["seed"]),
                "final_multiple": float(
                    r["final_multiple"]
                ),
                "sharpe": float(r["sharpe"]),
                "max_drawdown": float(
                    r["max_drawdown"]
                ),
                "mean_turnover": float(
                    r["mean_turnover"]
                ),
            })

    save_csv(OUT / "all_seed_metrics.csv", rows)

    summary = (
        pd.DataFrame(rows)
        .groupby(["version", "arm"])
        .agg(
            n_seeds=("seed", "nunique"),
            mean_multiple=("final_multiple", "mean"),
            std_multiple=("final_multiple", "std"),
            mean_sharpe=("sharpe", "mean"),
            mean_mdd=("max_drawdown", "mean"),
            mean_turnover=("mean_turnover", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "version_summary.csv",
        index=False
    )

    print(summary.to_string(index=False))

    print("\n========== PAIRED DELTAS ==========")

    paired_rows = []

    for version, run in runs.items():
        df = run["metrics"].set_index(
            ["arm", "seed"]
        )

        for seed in SEEDS:
            base = df.loc[("INDEPENDENT", seed)]
            unverified = df.loc[
                ("UNVERIFIED_JOINT", seed)
            ]
            grounded = df.loc[
                ("GROUNDED_JOINT", seed)
            ]

            for contrast, left, right in [
                (
                    "GROUNDED_MINUS_UNVERIFIED",
                    grounded, unverified
                ),
                (
                    "GROUNDED_MINUS_INDEPENDENT",
                    grounded, base
                ),
                (
                    "UNVERIFIED_MINUS_INDEPENDENT",
                    unverified, base
                ),
            ]:
                paired_rows.append({
                    "version": version,
                    "seed": seed,
                    "contrast": contrast,
                    "delta_multiple": float(
                        left["final_multiple"]
                        - right["final_multiple"]
                    ),
                    "delta_sharpe": float(
                        left["sharpe"] - right["sharpe"]
                    ),
                    "delta_mdd": float(
                        left["max_drawdown"]
                        - right["max_drawdown"]
                    ),
                })

    save_csv(
        OUT / "paired_seed_differences.csv",
        paired_rows
    )

    paired = pd.DataFrame(paired_rows)

    paired_summary = (
        paired.groupby(["version", "contrast"])
        .agg(
            mean_delta_multiple=("delta_multiple", "mean"),
            std_delta_multiple=("delta_multiple", "std"),
            mean_delta_sharpe=("delta_sharpe", "mean"),
            mean_delta_mdd=("delta_mdd", "mean"),
            positive_multiple_seeds=(
                "delta_multiple",
                lambda x: int((x > 0).sum())
            ),
        )
        .reset_index()
    )

    paired_summary.to_csv(
        OUT / "paired_summary.csv",
        index=False
    )

    print(paired_summary.to_string(index=False))

    print("\n========== CROSS-VERSION INDEPENDENT ==========")

    a = runs["V18"]["metrics"].set_index(
        ["arm", "seed"]
    )

    b = runs["V18_1"]["metrics"].set_index(
        ["arm", "seed"]
    )

    for seed in SEEDS:
        old = a.loc[("INDEPENDENT", seed)]
        new = b.loc[("INDEPENDENT", seed)]

        print(
            "Seed", seed,
            "V18=", round(float(old["final_multiple"]), 6),
            "V18.1=", round(float(new["final_multiple"]), 6),
            "delta=", round(float(
                new["final_multiple"]
                - old["final_multiple"]
            ), 6),
        )


def make_grounding_review():
    print("\n========== GROUNDING LABEL TEMPLATE ==========")

    b7 = read_jsonl(B7)
    b8 = read_jsonl(B8)

    claims = []

    for row in b7:
        for claim in row.get("case", {}).get("claims", []):
            check = claim.get("final_check") or {}

            claims.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "claim_id": claim.get("claim_id", ""),
                "sec_evidence_id": claim.get(
                    "sec_evidence_id", ""
                ),
                "quote": claim.get("quote", ""),
                "final_claim": claim.get(
                    "final_claim", ""
                ),
                "alignment_score": (
                    check.get("alignment_score")
                    if check.get("alignment_score") is not None
                    else ""
                ),
                "auto_status": check.get(
                    "verification_status", ""
                ),
                "human_supported": "",
                "human_notes": "",
            })

    events = []

    for row in b8:
        for decision in row.get("claim_decisions", []):
            b = decision.get("b8") or {}
            c = b.get("classification") or {}

            events.append({
                "date": row["date"],
                "ticker": row["ticker"],
                "claim_id": decision.get(
                    "claim_id", ""
                ),
                "quote": decision.get("quote", ""),
                "final_claim": decision.get(
                    "final_claim", ""
                ),
                "predicted_occurrence": c.get(
                    "event_occurrence", ""
                ),
                "predicted_outcome": c.get(
                    "outcome_status", ""
                ),
                "predicted_potential": c.get(
                    "potential_business_impact", ""
                ),
                "predicted_realized": c.get(
                    "realized_business_impact", ""
                ),
                "human_occurrence": "",
                "human_outcome": "",
                "human_potential": "",
                "human_realized": "",
                "human_notes": "",
            })

    for filename, records in [
        ("grounding_review.csv", claims),
        ("event_review.csv", events),
    ]:
        path = OUT / filename

        if path.exists():
            print(
                "[PRESERVED]",
                path,
                "(existing annotations not overwritten)"
            )
            continue

        if records:
            save_csv(path, records)

        print("[CREATED]", path, "rows=", len(records))

    return len(claims), len(events)


def confusion(y_true, y_pred):
    tp = sum(
        a == 1 and b == 1
        for a, b in zip(y_true, y_pred)
    )
    tn = sum(
        a == 0 and b == 0
        for a, b in zip(y_true, y_pred)
    )
    fp = sum(
        a == 0 and b == 1
        for a, b in zip(y_true, y_pred)
    )
    fn = sum(
        a == 1 and b == 0
        for a, b in zip(y_true, y_pred)
    )

    precision = (
        tp / (tp + fp) if tp + fp else 0.0
    )
    recall = (
        tp / (tp + fn) if tp + fn else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (
            (tp + tn) / len(y_true)
            if y_true else 0.0
        ),
    }


def human_evaluation():
    path = OUT / "grounding_review.csv"

    with path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        rows = list(csv.DictReader(f))

    labeled = [
        row for row in rows
        if row["human_supported"].strip() in {"0", "1"}
    ]

    if not labeled:
        print("\n[QUALITY] No human labels yet.")
        return False

    scored = []

    for row in labeled:
        try:
            score = float(row["alignment_score"])
        except (TypeError, ValueError):
            continue

        if math.isfinite(score):
            scored.append(
                (score, int(row["human_supported"]))
            )

    if not scored:
        print("[QUALITY] No numeric labeled AlignScores.")
        return False

    results = []

    for i in range(21):
        threshold = i / 20

        gold = [g for _, g in scored]
        predicted = [
            int(score >= threshold)
            for score, _ in scored
        ]

        results.append({
            "threshold": threshold,
            "n": len(scored),
            **confusion(gold, predicted),
        })

    save_csv(
        OUT / "alignscore_threshold_sweep.csv",
        results
    )

    print("\n========== HUMAN GROUNDING QUALITY ==========")
    print("Labeled:", len(labeled), "/", len(rows))
    print("Numeric scores:", len(scored))
    print(
        "At threshold 0.5:",
        next(
            row for row in results
            if row["threshold"] == 0.5
        )
    )

    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--evaluate-labels",
        action="store_true"
    )
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    runs = load_runs()
    comparisons = reproducibility(runs)
    metrics_analysis(runs)
    n_claims, n_events = make_grounding_review()

    evaluated = human_evaluation()

    report = {
        "version": "V18.2",
        "training": False,
        "test_2022_2023_accessed": False,
        "models_checked": 18,
        "tensor_comparison": comparisons,
        "claims_for_review": n_claims,
        "events_for_review": n_events,
        "human_quality_evaluated": evaluated,
        "source_hashes": {
            name: run["hashes"]
            for name, run in runs.items()
        },
        "limitations": [
            "The audit cannot prove deterministic replay.",
            "Checkpoint hashes identify artifacts, not causal explanations.",
            "AlignScore needs independent human labels.",
            "Threshold selection on the evaluation set is optimistic.",
            "Grounded risk and confidence remain uncalibrated.",
            "The 2022-2023 sealed test is not used.",
        ],
    }

    (OUT / "audit_manifest.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n========== V18.2 COMPLETE ==========")
    print("Reproducibility artifact audit: COMPLETE")
    print("Grounding label templates: READY")
    print(
        "Human grounding evaluation:",
        "AVAILABLE" if evaluated else "PENDING LABELS"
    )
    print("Blind test: NOT ACCESSED")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
