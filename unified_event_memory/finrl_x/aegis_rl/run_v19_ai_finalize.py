#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/aegis_rl_v1")
AI = ROOT / "v19_final/ai_annotation"
OUT = ROOT / "v19_final"
PPO = ROOT / "v18_1/ppo_full"

ARMS = [
    "INDEPENDENT",
    "UNVERIFIED_JOINT",
    "GROUNDED_JOINT",
]
SEEDS = {42, 123, 2026}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(
        json.dumps(
            data, ensure_ascii=False, indent=2,
            default=str
        ),
        encoding="utf-8"
    )


def normalize_binary(value):
    s = str(value).strip().lower()

    if s in ("true", "1", "yes", "supported"):
        return 1

    if s in ("false", "0", "no", "unsupported"):
        return 0

    return None


def normalize_category(value):
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "null"):
        return None
    return s.replace("-", "_").replace(" ", "_")


def main():
    grounding_path = AI / "grounding_ai_labeled.csv"
    event_path = AI / "event_ai_labeled.csv"

    for p in (grounding_path, event_path):
        if not p.exists():
            raise FileNotFoundError(p)

    g = pd.read_csv(
        grounding_path, dtype=str
    ).fillna("")

    e = pd.read_csv(
        event_path, dtype=str
    ).fillna("")

    print("========== AI ANNOTATION ==========")
    print("Grounding rows:", len(g))
    print("Event rows:", len(e))

    required_g = {
        "date", "ticker", "claim_id",
        "ai_supported", "alignment_score",
    }

    required_e = {
        "date", "ticker", "claim_id",
        "ai_occurrence", "ai_outcome",
        "ai_potential", "ai_realized",
    }

    if not required_g.issubset(g.columns):
        raise RuntimeError(
            f"Missing grounding columns: {required_g-set(g.columns)}"
        )

    if not required_e.issubset(e.columns):
        raise RuntimeError(
            f"Missing event columns: {required_e-set(e.columns)}"
        )

    if len(g) != 40 or len(e) != 19:
        raise RuntimeError(
            "Expected 40 grounding and 19 event rows. "
            "Check whether an earlier --limit run overwrote outputs."
        )

    if g.duplicated(["date","ticker","claim_id"]).any():
        raise RuntimeError("Duplicate grounding claim keys")

    if e.duplicated(["date","ticker","claim_id"]).any():
        raise RuntimeError("Duplicate event claim keys")

    # AlignScore vs AI Judge.
    g["score"] = pd.to_numeric(
        g["alignment_score"], errors="coerce"
    )

    g["ai_label"] = g["ai_supported"].map(
        normalize_binary
    )

    g["align_label"] = g["score"].map(
        lambda x: (
            int(x >= 0.5)
            if pd.notna(x) and np.isfinite(x)
            else None
        )
    )

    comparable = g[
        g.ai_label.notna() &
        g.align_label.notna()
    ].copy()

    comparable["agree"] = (
        comparable.ai_label.astype(int) ==
        comparable.align_label.astype(int)
    )

    disagreement = comparable[
        ~comparable.agree
    ].copy()

    disagreement.to_csv(
        OUT / "ai_alignscore_disagreements.csv",
        index=False, encoding="utf-8-sig"
    )

    # B8 vs independent AI Judge, conditional on both answering.
    comparisons = {}

    for field in (
        "occurrence", "outcome", "potential", "realized"
    ):
        prediction_col = f"predicted_{field}"
        judge_col = f"ai_{field}"

        if prediction_col not in e.columns:
            raise RuntimeError(
                f"Missing B8 prediction: {prediction_col}"
            )

        predicted = e[prediction_col].map(
            normalize_category
        )
        judged = e[judge_col].map(
            normalize_category
        )

        usable = predicted.notna() & judged.notna()

        agreement = (
            float((predicted[usable] == judged[usable]).mean())
            if usable.any()
            else None
        )

        comparisons[field] = {
            "n_comparable": int(usable.sum()),
            "n_total": len(e),
            "agreement": agreement,
        }

        conflicting = e[
            usable & (predicted != judged)
        ].copy()

        conflicting.to_csv(
            OUT / f"ai_b8_{field}_disagreements.csv",
            index=False, encoding="utf-8-sig"
        )

    report = {
        "evaluation_type": "AI_JUDGE_AGREEMENT",
        "human_ground_truth": False,
        "grounding_total": len(g),
        "grounding_comparable": len(comparable),
        "grounding_agreement": (
            float(comparable.agree.mean())
            if len(comparable) else None
        ),
        "grounding_disagreements": len(disagreement),
        "grounding_ai_abstentions_or_unparsed": int(
            g.ai_label.isna().sum()
        ),
        "grounding_missing_scores": int(
            g.align_label.isna().sum()
        ),
        "b8_agreement": comparisons,
        "judge_source": "Ollama AI annotation",
        "limitations": [
            "AI judge is not human ground truth.",
            "Agreement is not precision, recall, or F1 against independent truth.",
            "Using another Qwen model may introduce correlated errors.",
            "Agreement is conditional on parsable, answered records.",
            "No calibrated AlignScore threshold is established by this report.",
        ],
    }

    quality_path = OUT / "ai_quality_report.json"
    save_json(quality_path, report)

    print("\n========== AI QUALITY REPORT ==========")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    # PPO selection uses 2021 validation ONLY.
    metrics_path = PPO / "portfolio_summary.csv"
    manifest_path = PPO / "manifest.json"
    replay_path = OUT / "replay_report.json"

    if not replay_path.exists():
        raise FileNotFoundError(
            "Replay report required before freeze"
        )

    replay = json.loads(replay_path.read_text())

    if not (
        replay.get("feature_tensors_exactly_equal")
        and replay.get("metric_differences_within_tolerance")
        and replay.get("max_abs_multiple_difference") == 0
        and replay.get("max_abs_sharpe_difference") == 0
    ):
        raise RuntimeError("Replay has not passed strict checks")

    manifest = json.loads(manifest_path.read_text())

    if (
        int(manifest.get("timesteps", -1)) != 30000
        or set(manifest.get("seeds", [])) != SEEDS
        or int(manifest.get("joint_feature_dimension", -1)) != 9
    ):
        raise RuntimeError("Unexpected PPO training manifest")

    metrics = pd.read_csv(metrics_path)

    required_m = {
        "arm", "seed", "sharpe", "final_multiple"
    }
    if not required_m.issubset(metrics.columns):
        raise RuntimeError("Missing validation metrics")

    pairs = list(
        zip(metrics.arm, metrics.seed.astype(int))
    )

    if len(pairs) != 9 or set(pairs) != {
        (arm, seed)
        for arm in ARMS
        for seed in SEEDS
    }:
        raise RuntimeError("Incomplete 3-arm, 3-seed validation")

    means = metrics.groupby("arm")["sharpe"].mean()

    chosen_arm = sorted(
        ARMS,
        key=lambda arm: (-means[arm], arm)
    )[0]

    subset = metrics[
        metrics.arm == chosen_arm
    ].copy()

    median_sharpe = subset.sharpe.median()

    subset["distance"] = (
        subset.sharpe - median_sharpe
    ).abs()

    chosen = subset.sort_values(
        ["distance", "seed"]
    ).iloc[0]

    seed = int(chosen.seed)
    model_path = (
        PPO / f"{chosen_arm}_PPO_seed{seed}.zip"
    ).resolve()

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    frozen = {
        "selection_rule": (
            "Highest mean 2021 validation Sharpe across arms; "
            "within selected arm choose median-Sharpe seed."
        ),
        "selected_arm": chosen_arm,
        "selected_seed": seed,
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "training_manifest_sha256": sha256(manifest_path),
        "validation_metrics_sha256": sha256(metrics_path),
        "replay_report_sha256": sha256(replay_path),
        "ai_quality_report_sha256": sha256(quality_path),
        "feature_dimensions": [5,328],
        "additional_feature_dimensions": 9,
        "train": "2018-2020",
        "validation": "2021",
        "test": "2022-2023",
        "max_age_days": 120,
        "tickers": [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
        ],
        "alignscore_threshold_used_in_training": 0.5,
        "threshold_human_calibrated": False,
        "human_grounding_verified": False,
        "test_accessed_by_this_script": False,
    }

    freeze_path = OUT / "frozen_model.json"

    if freeze_path.exists():
        existing = json.loads(freeze_path.read_text())
        if (
            existing.get("model_sha256") != frozen["model_sha256"]
            or existing.get("selected_arm") != chosen_arm
            or existing.get("selected_seed") != seed
        ):
            raise RuntimeError(
                "Existing freeze selects a different model. "
                "Refusing to replace it."
            )
        print("\nExisting frozen model preserved.")
    else:
        save_json(freeze_path, frozen)

    print("\n========== FROZEN MODEL ==========")
    print("Arm:", chosen_arm)
    print("Seed:", seed)
    print("Mean validation Sharpe:", means[chosen_arm])
    print("Model:", model_path)
    print("Hash:", frozen["model_sha256"])
    print("Quality: AI agreement only, not human gold labels")
    print("2022-2023 TEST: NOT ACCESSED")
    print("\nV19 AI FINALIZATION: COMPLETE")


if __name__ == "__main__":
    main()
