#!/usr/bin/env python3
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_v16_dual_rag_finrl as v16

ROOT = Path("results/aegis_rl_v1")
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

SOURCES = {
    "original": ROOT / "v17_joint_rag_ppo_smoke/joint_rag_cache.jsonl",
    "b7": ROOT / "v18_b7_matched_20/audit.jsonl",
    "b8": ROOT / "v18_b8_matched_20/audit.jsonl",
}

ARMS = [
    "INDEPENDENT",
    "UNVERIFIED_JOINT",
    "GROUNDED_JOINT",
]


def load(path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["date"], r["ticker"])
        if key in result:
            raise RuntimeError(f"Duplicate: {key}")
        result[key] = r
    return result


def get_hits(row, source):
    for location in (
        row.get(f"{source}_evidence"),
        row.get(f"{source}_hits"),
        row.get("case", {}).get(f"{source}_hits"),
    ):
        if isinstance(location, list):
            return location
    raise RuntimeError(f"Missing {source} evidence list")


def check_ids(a, b):
    def ids(items):
        return [
            str(item.get("id", item.get("evidence_id")))
            for item in items
        ]
    return ids(a) == ids(b)


def impact(label):
    values = {
        "negative": -1.0,
        "neutral": 0.0,
        "positive": 1.0,
    }
    return values.get(str(label).lower())


def accepted_claim_ids(row):
    accepted = set()
    for c in row.get("case", {}).get("claims", []):
        f = c.get("final_check") or {}
        if (
            f.get("independent_verified") is True
            and f.get("verification_status") == "ALIGNSCORE_PASS"
            and f.get("deterministic_ok") is True
        ):
            accepted.add(c["claim_id"])
    return accepted


def grounded_features(b7, b8):
    accepted = accepted_claim_ids(b7)
    decisions = b8.get("claim_decisions", [])

    potentials = []
    classified = 0

    for d in decisions:
        if d.get("claim_id") not in accepted:
            continue

        item = d.get("b8") or {}
        if item.get("status") != "CLASSIFIED":
            continue

        cls = item.get("classification")
        if not isinstance(cls, dict):
            continue

        classified += 1
        value = impact(cls.get("potential_business_impact"))

        if value is not None:
            potentials.append(value)

    sec_verified = bool(accepted)
    kg_present = bool(get_hits(b7, "kg"))

    # 6D representation:
    # impact, risk-placeholder, confidence-placeholder,
    # verified SEC mask, structural KG mask, freshness.
    #
    # Missing financial direction is represented by impact=0;
    # downstream interpretation must also use audit masks.
    available = sec_verified

    vector = np.array([
        float(np.mean(potentials)) if potentials else 0.0,
        0.0,
        0.0,
        float(sec_verified),
        float(kg_present and available),
        1.0 if available else 0.0,
    ], dtype=np.float32)

    metadata = {
        "verified_claims": len(accepted),
        "classified_claims": classified,
        "known_potential_claims": len(potentials),
        "impact_known_mask": int(bool(potentials)),
        "sec_verified_mask": int(sec_verified),
        "kg_structural_mask": int(kg_present),
    }

    return vector, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1024)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--max-age-days", type=int, default=120)
    parser.add_argument(
        "--out",
        default="results/aegis_rl_v1/v18_threeway_ppo"
    )
    parser.add_argument("--dry-run", action="store_true")
    opt = parser.parse_args()

    if opt.timesteps < 128 or opt.max_age_days < 1:
        raise ValueError("Invalid settings")

    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)

    print("========== LOAD MATCHED RAG ==========", flush=True)

    data = {name: load(path) for name, path in SOURCES.items()}
    keys = set(data["original"])

    if not (
        keys == set(data["b7"]) == set(data["b8"])
        and len(keys) == 20
    ):
        raise RuntimeError("Matched 20 cases required")

    for key in sorted(keys):
        a = data["original"][key]
        b = data["b7"][key]

        for source in ("sec", "kg"):
            if not check_ids(
                get_hits(a, source),
                get_hits(b, source),
            ):
                raise RuntimeError(
                    f"Evidence ID mismatch: {key} {source}"
                )

    print("Matched cases: 20")
    print("Evidence IDs: PASS")

    print("\n========== LOAD V16 DATA ==========", flush=True)

    old_argv = sys.argv
    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        args = v16.parse_args()
    finally:
        sys.argv = old_argv

    args.out = out
    args.seeds = opt.seeds
    args.timesteps = opt.timesteps
    args.algo = "A2C"

    train, val, dates, contexts, n_train = v16.v8b.load_data(args)
    args.algo = "PPO"

    dates = v16.base.normalize_dates(dates)

    market_flat = np.concatenate(
        [train["market"], val["market"]], axis=0
    )
    prices = np.concatenate(
        [train["prices"], val["prices"]], axis=0
    )

    market, _ = v16.base.validate_market_layout(market_flat)

    if (
        market.shape != (len(dates), 5, 106)
        or len(prices) != len(dates)
        or dates.min().year != 2018
        or dates.max().year != 2021
        or dates[n_train].year != 2021
    ):
        raise RuntimeError("Unexpected V16 dataset split")

    sec, _ = v16.v13.load_sec(
        dates,
        contexts,
        args.pit_sec,
        args.sec_top_k,
        out / "sec_retrieval_audit.csv",
    )

    kg_events = v16.v13.kg_events_from_csv(
        args.temporal_kg,
        pd.Timestamp("2021-12-31"),
    )

    kg, _ = v16.v13.build_kg_rag(
        dates,
        kg_events,
        args.kg_top_k,
        out,
    )

    print("\n========== BUILD MATCHED FEATURES ==========")

    shape = (len(dates), 5, 6)
    original_joint = np.zeros(shape, dtype=np.float32)
    grounded_joint = np.zeros(shape, dtype=np.float32)
    zero_joint = np.zeros(shape, dtype=np.float32)

    # Store exact refresh records; do not generate new RAG outputs.
    refresh = {}

    for key in sorted(keys):
        original = data["original"][key]
        b7 = data["b7"][key]
        b8 = data["b8"][key]

        g_vector, metadata = grounded_features(b7, b8)
        result = original.get("result")

        if not (
            original.get("status") == "GENERATED"
            and isinstance(result, dict)
        ):
            o_vector = np.zeros(6, dtype=np.float32)
        elif result.get("sec_evidence_ids"):
            o_vector = np.array([
                float(result.get("impact", 0)),
                float(result.get("risk", 0)),
                float(result.get("confidence", 0)),
                1.0,
                float(bool(result.get("kg_evidence_ids"))),
                1.0,
            ], dtype=np.float32)
        else:
            o_vector = np.zeros(6, dtype=np.float32)

        refresh[key] = (o_vector, g_vector, metadata)

    latest_original = [None] * 5
    latest_grounded = [None] * 5
    last_original = [None] * 5
    last_grounded = [None] * 5

    metadata_rows = []

    for t, date in enumerate(dates):
        day = pd.Timestamp(date)
        day_str = str(day.date())

        if t == n_train:
            latest_original = [None] * 5
            latest_grounded = [None] * 5
            last_original = [None] * 5
            last_grounded = [None] * 5

        for j, ticker in enumerate(TICKERS):
            key = (day_str, ticker)

            if key in refresh:
                o, g, meta = refresh[key]

                # A rejected or missing refresh clears previous state.
                latest_original[j] = (
                    o.copy() if o[3] > 0 else None
                )
                latest_grounded[j] = (
                    g.copy() if g[3] > 0 else None
                )

                last_original[j] = (
                    day if latest_original[j] is not None else None
                )
                last_grounded[j] = (
                    day if latest_grounded[j] is not None else None
                )

                metadata_rows.append({
                    "date": day_str,
                    "ticker": ticker,
                    **meta,
                    "original_available": int(o[3] > 0),
                    "grounded_available": int(g[3] > 0),
                })

            for latest, last, tensor in (
                (latest_original, last_original, original_joint),
                (latest_grounded, last_grounded, grounded_joint),
            ):
                if latest[j] is None:
                    continue

                age = (day - last[j]).days
                if age > opt.max_age_days:
                    continue

                vector = latest[j].copy()
                vector[5] = 1.0 - min(
                    age / opt.max_age_days, 1.0
                )
                tensor[t, j] = vector

    if not (
        np.isfinite(original_joint).all()
        and np.isfinite(grounded_joint).all()
    ):
        raise RuntimeError("Nonfinite features")

    features = {
        "INDEPENDENT": v16.base.compose(
            market, sec, kg, zero_joint
        ),
        "UNVERIFIED_JOINT": v16.base.compose(
            market, sec, kg, original_joint
        ),
        "GROUNDED_JOINT": v16.base.compose(
            market, sec, kg, grounded_joint
        ),
    }

    dimensions = {name: x.shape for name, x in features.items()}
    if len(set(dimensions.values())) != 1:
        raise RuntimeError(f"Feature shapes differ: {dimensions}")

    pd.DataFrame(metadata_rows).to_csv(
        out / "refresh_signal_audit.csv",
        index=False,
    )

    np.savez_compressed(
        out / "joint_feature_tensors.npz",
        original=original_joint,
        grounded=grounded_joint,
        independent=zero_joint,
    )

    coverage = {
        "original_sec_mask": float(np.mean(original_joint[:, :, 3])),
        "grounded_sec_mask": float(np.mean(grounded_joint[:, :, 3])),
        "grounded_kg_mask": float(np.mean(grounded_joint[:, :, 4])),
        "known_impact_refreshes": int(
            sum(x["impact_known_mask"] for x in metadata_rows)
        ),
        "total_refreshes": len(metadata_rows),
    }

    print("Feature shapes:", dimensions)
    print("Coverage:", coverage)

    manifest = {
        "experiment": "V18 Three-Way Matched PPO",
        "arms": ARMS,
        "timesteps": opt.timesteps,
        "seeds": opt.seeds,
        "train": "2018-2020",
        "validation": "2021",
        "test_accessed": False,
        "max_age_days": opt.max_age_days,
        "coverage": coverage,
        "limitations": [
            "Same feature dimension does not mean same semantics.",
            "Grounded financial impact coverage is sparse.",
            "Risk/confidence zero values are placeholders.",
            "AlignScore threshold is not manually calibrated.",
            "KG timestamps are not certified disclosure timestamps.",
            "Only evidence IDs, not full evidence payloads, were matched.",
        ],
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )

    if opt.dry_run:
        print("\nV18 FEATURE PREFLIGHT: PASS")
        print("PPO TRAINING: SKIPPED")
        return

    print("\n========== TRAIN THREE-WAY PPO ==========")

    all_results = []

    for seed in opt.seeds:
        for arm in ARMS:
            random.seed(seed)
            np.random.seed(seed)

            x = features[arm]

            train_env = v16.DualRAGPortfolioEnv(
                x[:n_train],
                prices[:n_train],
                dates[:n_train],
                args.transaction_cost,
                args.reward_scale,
            )

            val_env = v16.DualRAGPortfolioEnv(
                x[n_train:],
                prices[n_train:],
                dates[n_train:],
                args.transaction_cost,
                args.reward_scale,
            )

            model = v16.train_policy(
                "PPO",
                train_env,
                seed,
                opt.timesteps,
                args.learning_rate,
            )

            model.save(str(out / f"{arm}_PPO_seed{seed}"))

            history, metrics = v16.evaluate(
                model, val_env, arm, seed, "PPO"
            )

            history.to_csv(
                out / f"{arm}_seed{seed}_validation_2021.csv",
                index=False,
            )

            all_results.append(metrics)

            pd.DataFrame(all_results).to_csv(
                out / "portfolio_metrics.csv",
                index=False,
            )

            print(
                "[RESULT]",
                arm,
                seed,
                "multiple=", metrics["final_multiple"],
                "sharpe=", metrics["sharpe"],
                flush=True,
            )

    df = pd.DataFrame(all_results)
    df.to_csv(out / "portfolio_summary.csv", index=False)

    summary = df.groupby("arm", as_index=False).agg(
        n_seeds=("seed", "nunique"),
        mean_multiple=("final_multiple", "mean"),
        std_multiple=("final_multiple", "std"),
        mean_sharpe=("sharpe", "mean"),
        mean_mdd=("max_drawdown", "mean"),
        mean_turnover=("mean_turnover", "mean"),
    )

    summary.to_csv(out / "multiseed_summary.csv", index=False)

    print("\n========== MULTI-SEED SUMMARY ==========")
    print(summary.to_string(index=False))
    print("\nV18 THREE-WAY PPO: COMPLETE")


if __name__ == "__main__":
    main()
