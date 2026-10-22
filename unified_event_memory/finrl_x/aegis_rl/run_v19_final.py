#!/usr/bin/env python3
"""
V19 research finalization
Stages:
  status
  replay
  quality
  freeze
  certify-test
  blind-test

Never generates human labels or synthetic test features.
Never trains on the test set.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("results/aegis_rl_v1")
V18 = ROOT / "v18_threeway_ppo_full"
V181 = ROOT / "v18_1/ppo_full"
V19 = ROOT / "v19_final"
REVIEW = ROOT / "v18_2_audit"

HERE = Path(__file__).resolve().parent
TRAIN_SCRIPT = HERE / "run_v18_1_missingness_ppo.py"

ARMS = [
    "INDEPENDENT",
    "UNVERIFIED_JOINT",
    "GROUNDED_JOINT",
]
SEEDS = [42, 123, 2026]
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

V19.mkdir(parents=True, exist_ok=True)


def fail(message):
    raise RuntimeError(message)


def read_json(path):
    path = Path(path)
    if not path.exists():
        fail(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)
    return h.hexdigest()


def df_csv(path):
    if not Path(path).exists():
        fail(f"Missing CSV: {path}")
    return pd.read_csv(path)


def manifest_check(path):
    m = read_json(path)

    if int(m["timesteps"]) != 30000:
        fail("Expected 30000 training steps")

    if set(map(int, m["seeds"])) != set(SEEDS):
        fail("Unexpected seed set")

    if int(m.get("joint_feature_dimension", -1)) != 9:
        fail("Expected 9D V18.1 joint features")

    if int(m["max_age_days"]) != 120:
        fail("Expected 120-day freshness limit")

    return m


def metrics_check(path):
    d = df_csv(path)
    required = {
        "arm", "seed", "final_multiple",
        "sharpe", "max_drawdown"
    }
    if not required.issubset(d.columns):
        fail(f"Metrics missing: {required - set(d.columns)}")

    d["seed"] = d["seed"].astype(int)

    pairs = list(zip(d.arm, d.seed))
    wanted = {
        (arm, seed)
        for arm in ARMS
        for seed in SEEDS
    }
    if len(pairs) != 9 or set(pairs) != wanted:
        fail("Expected exactly 9 distinct arm/seed results")

    numeric = d[
        ["final_multiple", "sharpe", "max_drawdown"]
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric).all():
        fail("Non-finite portfolio metric")

    return d


def test_not_accessed():
    # This is a protocol assertion, not a claim about all prior activity.
    if (V19 / "blind_test" / "test_report.json").exists():
        fail("Blind test already evaluated in V19")


def status(args):
    print("\n========== V19 STATUS ==========")

    files = {
        "V18.1 manifest": V181 / "manifest.json",
        "V18.1 metrics": V181 / "portfolio_summary.csv",
        "Grounding labels": REVIEW / "grounding_review.csv",
        "Event labels": REVIEW / "event_review.csv",
        "Quality report": V19 / "quality.json",
        "Frozen model": V19 / "frozen_model.json",
        "Test certificate": V19 / "test_certificate.json",
        "Blind-test report": V19 / "blind_test/test_report.json",
    }

    for label, path in files.items():
        print(
            f"{label:26s}",
            "EXISTS" if path.exists() else "PENDING"
        )


def replay(args):
    manifest_check(V181 / "manifest.json")
    original = metrics_check(
        V181 / "portfolio_summary.csv"
    )

    run_dir = V19 / "replay"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not args.inspect_existing and any(
        run_dir.glob("*_PPO_seed*.zip")
    ):
        fail(
            "Replay models already exist. "
            "Use --inspect-existing instead of overwriting."
        )

    if not args.inspect_existing:
        if not TRAIN_SCRIPT.exists():
            fail(f"Missing training script: {TRAIN_SCRIPT}")

        command = [
            sys.executable,
            "-u",
            str(TRAIN_SCRIPT),
            "--timesteps", "30000",
            "--seeds", "42", "123", "2026",
            "--max-age-days", "120",
            "--out", str(run_dir),
        ]

        print("[REPLAY]", " ".join(command), flush=True)

        with (V19 / "replay.log").open(
            "w", encoding="utf-8"
        ) as logfile:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={
                    **os.environ,
                    "PYTHONPATH": (
                        str(Path.cwd())
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", "")
                    )
                },
            )

            for line in process.stdout:
                print(line, end="", flush=True)
                logfile.write(line)
                logfile.flush()

            rc = process.wait()

            if rc:
                fail(f"Replay training failed: exit={rc}")

    replay_manifest = manifest_check(
        run_dir / "manifest.json"
    )
    reproduced = metrics_check(
        run_dir / "portfolio_summary.csv"
    )

    with np.load(
        V181 / "joint_feature_tensors.npz",
        allow_pickle=False,
    ) as a, np.load(
        run_dir / "joint_feature_tensors.npz",
        allow_pickle=False,
    ) as b:
        tensor_equal = all(
            np.array_equal(a[k], b[k])
            for k in ("independent", "original", "grounded")
        )

    if not tensor_equal:
        fail(
            "REPLAY FAILED: Joint feature tensors differ. "
            "Stop before freeze."
        )

    original = original.set_index(["arm", "seed"])
    reproduced = reproduced.set_index(["arm", "seed"])

    rows = []

    for arm in ARMS:
        for seed in SEEDS:
            old = original.loc[(arm, seed)]
            new = reproduced.loc[(arm, seed)]

            rows.append({
                "arm": arm,
                "seed": seed,
                "original_multiple": float(
                    old.final_multiple
                ),
                "replay_multiple": float(
                    new.final_multiple
                ),
                "delta_multiple": float(
                    new.final_multiple - old.final_multiple
                ),
                "delta_sharpe": float(
                    new.sharpe - old.sharpe
                ),
                "delta_mdd": float(
                    new.max_drawdown - old.max_drawdown
                ),
            })

    d = pd.DataFrame(rows)
    d.to_csv(V19 / "replay_differences.csv", index=False)

    # Diagnostic tolerance, not proof of deterministic replay.
    tol = float(args.tolerance)
    within_tolerance = bool(
        (d[[
            "delta_multiple", "delta_sharpe", "delta_mdd"
        ]].abs() <= tol).all().all()
    )

    report = {
        "feature_tensors_exactly_equal": tensor_equal,
        "metric_tolerance": tol,
        "metric_differences_within_tolerance": within_tolerance,
        "max_abs_multiple_difference": float(
            d.delta_multiple.abs().max()
        ),
        "max_abs_sharpe_difference": float(
            d.delta_sharpe.abs().max()
        ),
        "original_manifest_sha256": digest(
            V181 / "manifest.json"
        ),
        "replay_manifest_sha256": digest(
            run_dir / "manifest.json"
        ),
        "interpretation": (
            "One repeat estimates stability. Metric agreement "
            "within tolerance is not proof of bitwise determinism."
        ),
    }

    write_json(V19 / "replay_report.json", report)

    print("\n========== REPLAY REPORT ==========")
    print(d.to_string(index=False))
    print(json.dumps(report, indent=2))

    if not within_tolerance:
        print(
            "\n[WARNING] Replay metrics differ beyond "
            "the diagnostic tolerance."
        )


def binary_metrics(gold, pred):
    gold = np.asarray(gold, dtype=int)
    pred = np.asarray(pred, dtype=int)

    tp = int(((gold == 1) & (pred == 1)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())

    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0

    return {
        "n": len(gold),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "accuracy": (tp + tn) / len(gold) if len(gold) else 0.0
    }


def quality(args):
    gpath = REVIEW / "grounding_review.csv"
    epath = REVIEW / "event_review.csv"

    claims = df_csv(gpath).fillna("")
    events = df_csv(epath).fillna("")

    if len(claims) == 0:
        fail("No grounding claims")

    human = claims.human_supported.astype(str).str.strip()

    if not human.isin(["0", "1"]).all():
        missing = int((~human.isin(["0", "1"])).sum())
        fail(
            f"{missing} claims lack human 0/1 labels. "
            f"Fill {gpath} before running quality."
        )

    claims["gold"] = human.astype(int)
    claims["score"] = pd.to_numeric(
        claims.alignment_score,
        errors="coerce"
    )

    scored = claims[np.isfinite(claims.score)].copy()

    if len(scored) < 10:
        fail(
            "Fewer than 10 numeric labeled AlignScores. "
            "Insufficient for even an exploratory threshold report."
        )

    # Group by date to reduce leakage between closely related claims.
    unique_dates = sorted(scored.date.astype(str).unique())

    if len(unique_dates) < 3:
        fail(
            "Need at least 3 unique dates for a separate "
            "calibration/holdout split. Current matched sample "
            "is too small; collect more labeled cases."
        )

    holdout_date = unique_dates[-1]

    calibration = scored[
        scored.date.astype(str) != holdout_date
    ]
    holdout = scored[
        scored.date.astype(str) == holdout_date
    ]

    if len(calibration) < 10 or len(holdout) < 2:
        fail(
            "Not enough calibrated/held-out numeric labels. "
            "Collect more independently annotated samples."
        )

    # Selection restricted to the calibration group.
    sweep = []

    for threshold in np.linspace(0, 1, 101):
        pred = (
            calibration.score.to_numpy() >= threshold
        ).astype(int)

        result = binary_metrics(
            calibration.gold.to_numpy(),
            pred
        )

        sweep.append({
            "threshold": float(threshold),
            **result
        })

    sweeps = pd.DataFrame(sweep)
    sweeps.to_csv(
        V19 / "alignscore_calibration_sweep.csv",
        index=False
    )

    ranked = sweeps.sort_values(
        ["f1", "threshold"],
        ascending=[False, True],
    )

    chosen = float(ranked.iloc[0].threshold)

    holdout_report = binary_metrics(
        holdout.gold.to_numpy(),
        (holdout.score.to_numpy() >= chosen).astype(int)
    )

    original_report = binary_metrics(
        holdout.gold.to_numpy(),
        (holdout.score.to_numpy() >= 0.5).astype(int)
    )

    # Evaluate B8 human annotations independently.
    event_fields = {
        "occurrence": (
            "predicted_occurrence", "human_occurrence"
        ),
        "outcome": (
            "predicted_outcome", "human_outcome"
        ),
        "potential": (
            "predicted_potential", "human_potential"
        ),
        "realized": (
            "predicted_realized", "human_realized"
        ),
    }

    event_quality = {}

    for name, (pred_col, gold_col) in event_fields.items():
        pred = events[pred_col].astype(str).str.strip().str.lower()
        gold = events[gold_col].astype(str).str.strip().str.lower()

        valid = gold.ne("")
        if not valid.all():
            fail(
                f"Human event labels incomplete: {gold_col}. "
                "All event-review rows must be labeled."
            )

        answered = valid & pred.ne("")
        abstained = valid & pred.eq("")

        event_quality[name] = {
            "n_labeled": int(valid.sum()),
            "n_answered": int(answered.sum()),
            "n_abstained": int(abstained.sum()),
            "coverage": (
                float(answered.sum() / valid.sum())
                if valid.sum() else 0.0
            ),
            "accuracy_on_answered": (
                float((pred[answered] == gold[answered]).mean())
                if answered.sum() else None
            ),
            "predicted_distribution": (
                pred[answered].value_counts().to_dict()
            ),
            "human_distribution": (
                gold[valid].value_counts().to_dict()
            ),
        }

    result = {
        "human_labeled_claims": len(claims),
        "numeric_scored_claims": len(scored),
        "excluded_without_numeric_score": (
            len(claims) - len(scored)
        ),
        "calibration_dates": unique_dates[:-1],
        "holdout_date": holdout_date,
        "chosen_threshold": chosen,
        "holdout_quality_at_chosen_threshold": holdout_report,
        "holdout_quality_at_original_0_5": original_report,
        "b8_quality": event_quality,
        "quality_limitations": [
            "Small date-grouped human-labeled sample.",
            "Threshold is exploratory; not large-scale calibration.",
            "Threshold selection did not retrain PPO or regenerate RAG.",
            "Claims with missing scores were excluded from numeric evaluation.",
        ],
    }

    write_json(V19 / "quality.json", result)

    print("\n========== GROUNDING QUALITY ==========")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def freeze(args):
    test_not_accessed()

    manifest_check(V181 / "manifest.json")
    df = metrics_check(V181 / "portfolio_summary.csv")

    replay_report = read_json(
        V19 / "replay_report.json"
    )
    quality_report = read_json(
        V19 / "quality.json"
    )

    if not replay_report["feature_tensors_exactly_equal"]:
        fail("Cannot freeze: feature replay mismatch")

    if not replay_report["metric_differences_within_tolerance"]:
        if not args.accept_replay_variance:
            fail(
                "Replay variance exceeds tolerance. "
                "Investigate or explicitly document the limitation."
            )

    freeze_path = V19 / "frozen_model.json"

    if freeze_path.exists():
        fail("Frozen model already exists; refusing overwrite")

    # Arm chosen by 2021 validation mean Sharpe.
    means = df.groupby("arm").sharpe.mean()

    arm = sorted(
        ARMS,
        key=lambda name: (
            -float(means.loc[name]),
            name
        )
    )[0]

    # Representative seed: closest to arm's median validation Sharpe.
    subset = df[df.arm == arm].copy()
    median = float(subset.sharpe.median())
    subset["distance"] = (subset.sharpe - median).abs()
    chosen_row = subset.sort_values(
        ["distance", "seed"]
    ).iloc[0]

    seed = int(chosen_row.seed)

    model_path = (
        V181 / f"{arm}_PPO_seed{seed}.zip"
    ).resolve()

    if not model_path.exists():
        fail(f"Selected model missing: {model_path}")

    locked = {
        "selection_rule": (
            "Select highest mean 2021 validation Sharpe among "
            "three arms; within selected arm pick seed nearest "
            "the median validation Sharpe; deterministic tie break."
        ),
        "selected_arm": arm,
        "selected_seed": seed,
        "model_path": str(model_path),
        "model_sha256": digest(model_path),
        "validation_metrics_sha256": digest(
            V181 / "portfolio_summary.csv"
        ),
        "training_manifest_sha256": digest(
            V181 / "manifest.json"
        ),
        "quality_report_sha256": digest(
            V19 / "quality.json"
        ),
        "replay_report_sha256": digest(
            V19 / "replay_report.json"
        ),
        "calibrated_alignscore_threshold": (
            quality_report["chosen_threshold"]
        ),
        "joint_feature_dimension": 9,
        "per_stock_feature_dimension": 328,
        "max_age_days": 120,
        "stock_order": TICKERS,
        "training_period": "2018-2020",
        "validation_period": "2021",
        "sealed_test_period": "2022-2023",
        "test_accessed_by_this_pipeline": False,
        "replay_variance_accepted": bool(
            not replay_report[
                "metric_differences_within_tolerance"
            ]
        ),
        "important": (
            "The exploratory calibrated AlignScore threshold "
            "has NOT been used to regenerate training features "
            "or retrain this model. Test features must replicate "
            "the actual frozen training feature algorithm and "
            "its original grounding settings."
        ),
    }

    write_json(freeze_path, locked)

    print("\n========== MODEL FROZEN ==========")
    print(json.dumps(locked, indent=2, ensure_ascii=False))


def load_test_arrays(path):
    path = Path(path)
    if not path.exists():
        fail(f"Missing test NPZ: {path}")

    with np.load(path, allow_pickle=False) as z:
        required = {"features", "prices", "dates"}
        if not required.issubset(set(z.files)):
            fail(
                f"Test NPZ missing: "
                f"{required - set(z.files)}"
            )

        features = np.array(
            z["features"], dtype=np.float32
        )
        prices = np.array(
            z["prices"], dtype=np.float32
        )
        dates = pd.to_datetime(
            z["dates"].astype(str),
            errors="raise"
        )

    if features.ndim != 3 or features.shape[1:] != (5, 328):
        fail(f"Bad test feature shape: {features.shape}")

    if prices.shape != (len(features), 5):
        fail(f"Bad prices: {prices.shape}")

    if not len(dates) == len(features):
        fail("Date count mismatch")

    if dates.has_duplicates or not dates.is_monotonic_increasing:
        fail("Test dates duplicated or unsorted")

    if (
        dates.min() < pd.Timestamp("2022-01-01")
        or dates.max() > pd.Timestamp("2023-12-31")
    ):
        fail("Outside sealed test period")

    if not np.isfinite(features).all():
        fail("Non-finite features")

    if not np.isfinite(prices).all() or not (prices > 0).all():
        fail("Invalid price values")

    # Diagnostic check only, not proof of data provenance.
    if np.all(features == 0):
        fail("All-zero test features")

    return features, prices, dates


def certify_test(args):
    test_not_accessed()

    frozen_path = V19 / "frozen_model.json"
    frozen = read_json(frozen_path)

    cert_path = V19 / "test_certificate.json"

    if cert_path.exists():
        fail("Existing test certificate; refusing overwrite")

    test_path = Path(args.test_npz)
    features, prices, dates = load_test_arrays(test_path)

    provenance_path = Path(args.provenance)
    provenance = read_json(provenance_path)

    required = {
        "source_market",
        "source_sec",
        "source_kg",
        "source_joint",
        "build_script_sha256",
        "train_feature_contract_sha256",
        "pit_audit_csv",
        "universe",
        "date_range",
        "feature_order",
        "grounding_settings",
    }

    missing = required - set(provenance)
    if missing:
        fail(
            "PIT provenance incomplete: "
            + ", ".join(sorted(missing))
        )

    if provenance["universe"] != TICKERS:
        fail("Test stock order differs from training")

    if provenance["date_range"] != "2022-2023":
        fail("Unexpected test range declaration")

    if not isinstance(provenance["feature_order"], list):
        fail("feature_order must be an ordered list")

    if len(provenance["feature_order"]) != 328:
        fail("Expected 328 exact feature names per stock")

    if len(set(provenance["feature_order"])) != 328:
        fail("Duplicate feature names")

    # The value must be independently bound to the actual
    # training pipeline. The pipeline does not infer this hash.
    if not provenance["train_feature_contract_sha256"]:
        fail("Missing independently verified feature contract")

    audit_path = Path(provenance["pit_audit_csv"])
    audit = df_csv(audit_path)

    columns = {
        "decision_date",
        "source_type",
        "source_id",
        "available_at",
    }

    if not columns.issubset(audit.columns):
        fail(
            "PIT audit requires: "
            + ", ".join(sorted(columns))
        )

    if audit.empty:
        fail("PIT audit empty")

    decisions = pd.to_datetime(
        audit.decision_date,
        errors="raise",
        utc=True
    )
    available = pd.to_datetime(
        audit.available_at,
        errors="raise",
        utc=True
    )

    if decisions.isna().any() or available.isna().any():
        fail("Missing PIT timestamps")

    if not (available < decisions).all():
        bad = int((available >= decisions).sum())
        fail(f"LOOK-AHEAD DETECTED: {bad} source records")

    # Every row in PIT audit must be within the tested interval.
    test_days = set(dates.strftime("%Y-%m-%d"))
    audit_days = set(
        pd.to_datetime(
            audit.decision_date, utc=True
        ).dt.strftime("%Y-%m-%d")
    )

    if not audit_days.issubset(test_days):
        fail("PIT audit includes dates outside test NPZ")

    # This checks row-level timestamps, not complete feature lineage.
    certificate = {
        "test_npz": str(test_path.resolve()),
        "test_npz_sha256": digest(test_path),
        "provenance": str(provenance_path.resolve()),
        "provenance_sha256": digest(provenance_path),
        "pit_audit_sha256": digest(audit_path),
        "frozen_model_sha256": frozen["model_sha256"],
        "date_start": str(dates.min().date()),
        "date_end": str(dates.max().date()),
        "n_days": len(dates),
        "pit_audit_rows": len(audit),
        "feature_shape": list(features.shape),
        "checks": {
            "shape": True,
            "date_range": True,
            "finite_values": True,
            "stock_order_declaration": True,
            "timestamp_order": True,
            "full_feature_lineage_verified": False,
        },
        "warning": (
            "This is a structural/timestamp certificate, not "
            "proof that the independently supplied test features "
            "were constructed identically to training, or that "
            "all contributing evidence was disclosed before trading."
        ),
    }

    write_json(cert_path, certificate)

    print("\n========== TEST CERTIFICATION ==========")
    print(json.dumps(certificate, indent=2, ensure_ascii=False))


def blind_test(args):
    frozen = read_json(V19 / "frozen_model.json")
    cert = read_json(V19 / "test_certificate.json")

    output = V19 / "blind_test"

    if output.exists() and any(output.iterdir()):
        fail("Blind-test output already exists; refusing rerun")

    model_path = Path(frozen["model_path"])

    if digest(model_path) != frozen["model_sha256"]:
        fail("Frozen model hash mismatch")

    if cert["frozen_model_sha256"] != frozen["model_sha256"]:
        fail("Test certificate/frozen model mismatch")

    test_path = Path(cert["test_npz"])

    if digest(test_path) != cert["test_npz_sha256"]:
        fail("Test NPZ changed after certification")

    if digest(cert["provenance"]) != cert["provenance_sha256"]:
        fail("Test provenance changed after certification")

    features, prices, dates = load_test_arrays(test_path)

    # This is intentionally a read-only evaluation.
    from stable_baselines3 import PPO

    sys.path.insert(0, str(HERE))
    import run_v16_dual_rag_finrl as v16

    previous_argv = sys.argv
    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        settings = v16.parse_args()
    finally:
        sys.argv = previous_argv

    model = PPO.load(str(model_path), device="cpu")

    env = v16.DualRAGPortfolioEnv(
        features,
        prices,
        dates,
        settings.transaction_cost,
        settings.reward_scale,
    )

    if (
        model.observation_space.shape
        != env.observation_space.shape
    ):
        fail(
            "Model/environment shape mismatch: "
            f"{model.observation_space.shape} vs "
            f"{env.observation_space.shape}"
        )

    # Reserve output BEFORE computing results.
    output.mkdir(parents=True, exist_ok=False)
    (output / "STARTED").write_text(
        "Sealed test execution started. "
        "Do not rerun for model selection.\n"
    )

    history, result = v16.evaluate(
        model,
        env,
        frozen["selected_arm"] + "_SEALED_TEST",
        frozen["selected_seed"],
        "PPO",
    )

    history.to_csv(
        output / "test_history.csv",
        index=False
    )

    pd.DataFrame([result]).to_csv(
        output / "test_metrics.csv",
        index=False
    )

    report = {
        "arm": frozen["selected_arm"],
        "seed": frozen["selected_seed"],
        "model_sha256": frozen["model_sha256"],
        "test_npz_sha256": cert["test_npz_sha256"],
        "dates": [
            str(dates.min().date()),
            str(dates.max().date())
        ],
        "n_days": len(dates),
        "metrics": result,
        "limitations": [
            "Depends on externally supplied feature provenance.",
            "Timestamp checks alone do not prove full PIT safety.",
            "The model must not be reselected after this evaluation.",
        ],
    }

    write_json(output / "test_report.json", report)
    (output / "COMPLETE").write_text("PASS\n")

    print("\n========== SEALED TEST RESULTS ==========")
    print(pd.DataFrame([result]).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    sub.add_parser("status")

    p = sub.add_parser("replay")
    p.add_argument(
        "--inspect-existing",
        action="store_true"
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.01
    )

    sub.add_parser("quality")

    p = sub.add_parser("freeze")
    p.add_argument(
        "--accept-replay-variance",
        action="store_true"
    )

    p = sub.add_parser("certify-test")
    p.add_argument("--test-npz", required=True)
    p.add_argument("--provenance", required=True)

    sub.add_parser("blind-test")

    args = parser.parse_args()

    actions = {
        "status": status,
        "replay": replay,
        "quality": quality,
        "freeze": freeze,
        "certify-test": certify_test,
        "blind-test": blind_test,
    }

    actions[args.command](args)


if __name__ == "__main__":
    main()
