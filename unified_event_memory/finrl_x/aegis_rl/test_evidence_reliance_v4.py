#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V4 Evidence Reliance Test

Uses:
    Frozen V4 A2C model
    V4 FLARE follow-up retrieval
    V4 evidence critic
    V4 trainable gated fusion
    V4 recorded retrieval traces

Four evidence interventions:
    REAL
    ZERO
    SHUFFLED
    LAGGED

No RL training.

This is an evidence sensitivity experiment,
not proof of causal or economic improvement.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v4 as v4
)

from unified_event_memory.finrl_x.aegis_rl import (
    test_evidence_reliance_v3 as replay
)


def log(*args):
    print(*args, flush=True)


def analyze_real_weights(out, split):
    """
    Compare actual normalized portfolio weights
    rather than only raw A2C actions.
    """

    tickers = [
        "AAPL",
        "AMZN",
        "GOOGL",
        "MSFT",
        "NVDA",
    ]

    weight_cols = [
        f"weight_{ticker}"
        for ticker in tickers
    ]

    modes = [
        "REAL",
        "ZERO",
        "SHUFFLED",
        "LAGGED",
    ]

    frames = {}

    for mode in modes:
        path = (
            out
            / f"{split}_{mode.lower()}_portfolio.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing replay output: {path}"
            )

        df = pd.read_csv(path)

        required = (
            {"date"} | set(weight_cols)
        )

        missing = required - set(df.columns)

        if missing:
            raise RuntimeError(
                f"{mode}: missing columns {missing}"
            )

        df = df.copy()

        df["date"] = pd.to_datetime(
            df["date"]
        )

        if df["date"].duplicated().any():
            raise RuntimeError(
                f"{mode}: duplicate dates"
            )

        df = (
            df.set_index("date")
            .sort_index()
        )

        frames[mode] = df

    real = frames["REAL"]

    rows = []
    daily = []

    for mode in modes:
        other = frames[mode]

        if not real.index.equals(
            other.index
        ):
            raise RuntimeError(
                f"Date alignment failed: {mode}"
            )

        real_weights = real[
            weight_cols
        ].to_numpy(dtype=float)

        other_weights = other[
            weight_cols
        ].to_numpy(dtype=float)

        if not (
            np.isfinite(real_weights).all()
            and np.isfinite(other_weights).all()
        ):
            raise RuntimeError(
                f"Nonfinite weights: {mode}"
            )

        # Five-stock weight L1 distance.
        distances = np.abs(
            real_weights - other_weights
        ).sum(axis=1)

        rows.append({
            "mode": mode,
            "days": len(distances),
            "mean_weight_l1": float(
                distances.mean()
            ),
            "median_weight_l1": float(
                np.median(distances)
            ),
            "max_weight_l1": float(
                distances.max()
            ),
            "fraction_weight_changed": float(
                (distances > 1e-6).mean()
            ),
        })

        for date, distance in zip(
            real.index,
            distances,
        ):
            daily.append({
                "date": str(date.date()),
                "mode": mode,
                "weight_l1_vs_real": float(
                    distance
                ),
            })

    summary = pd.DataFrame(rows)

    summary.to_csv(
        out / f"{split}_weight_reliance_summary.csv",
        index=False,
    )

    pd.DataFrame(daily).to_csv(
        out / f"{split}_daily_weight_differences.csv",
        index=False,
    )

    log()
    log("===== ACTUAL PORTFOLIO WEIGHT RELIANCE =====")
    log(summary.to_string(index=False))

    return summary


def verify_real_replay(out, split, original_root):
    """
    Compare V4 REAL replay with original V4
    evaluation summary when available.

    A mismatch must be flagged explicitly.
    """

    result_path = (
        original_root
        / "runs"
        / "ACTIVE__A2C__seed42"
        / "result.json"
    )

    summary_path = (
        out
        / f"{split}_reliance_summary.csv"
    )

    if not result_path.exists():
        log("[WARNING] Original result.json missing")
        return

    import json

    original = json.loads(
        result_path.read_text(
            encoding="utf-8"
        )
    )

    original_rows = original.get(
        "rows",
        []
    )

    matches = [
        row
        for row in original_rows
        if row.get("split") == split
    ]

    if not matches:
        log("[WARNING] No matching original split")
        return

    summary = pd.read_csv(
        summary_path
    )

    real = summary.loc[
        summary["mode"] == "REAL"
    ]

    if len(real) != 1:
        raise RuntimeError(
            "Missing REAL replay result"
        )

    original_return = float(
        matches[0]["total_return"]
    )

    replay_return = float(
        real.iloc[0]["total_return"]
    )

    difference = (
        replay_return - original_return
    )

    log()
    log("===== REAL REPLAY CONSISTENCY =====")

    log(
        "Original V4 return:",
        f"{original_return:.9f}"
    )

    log(
        "Replayed REAL return:",
        f"{replay_return:.9f}"
    )

    log(
        "Difference:",
        f"{difference:.9f}"
    )

    tolerance = 1e-4

    if abs(difference) > tolerance:
        log(
            "[WARNING] REAL replay differs from "
            "original V4 evaluation."
        )
        log(
            "Interpret evidence interventions "
            "as comparisons within this replay, "
            "not a verified reproduction "
            "of the original result."
        )
    else:
        log("REAL REPLAY CONSISTENCY: PASS")


def main():
    argv = sys.argv[1:]

    if "--split" in argv:
        split = argv[
            argv.index("--split") + 1
        ]
    else:
        split = "val"

    if "--root" in argv:
        original_root = Path(
            argv[
                argv.index("--root") + 1
            ]
        ).resolve()
    else:
        original_root = Path(
            "results/aegis_rl_v1/"
            "active_rag_v4_smoke"
        ).resolve()

        argv += [
            "--root",
            str(original_root),
        ]

    if "--out" in argv:
        out = Path(
            argv[
                argv.index("--out") + 1
            ]
        ).resolve()
    else:
        out = Path(
            "results/aegis_rl_v1/"
            "evidence_reliance_v4"
        ).resolve()

        argv += [
            "--out",
            str(out),
        ]

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        original_root
        / "runs"
        / "ACTIVE__A2C__seed42"
        / "model.zip"
    )

    trace_path = (
        original_root
        / "runs"
        / "ACTIVE__A2C__seed42"
        / f"{split}_queries.csv"
    )

    for path in [
        model_path,
        trace_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    log("===== FINRL-X V4 EVIDENCE RELIANCE =====")
    log("Model:", model_path)
    log("Trace:", trace_path)
    log("Split:", split)
    log("Output:", out)

    # Install the actual V4 retrieval routine
    # on the V3 contextual environment.
    #
    # No portfolio training occurs here.
    v4.install_v4()

    # The existing replay implementation
    # uses v3.ContextualActiveRAGEnv.
    # install_v4 modifies that class's _search()
    # so it uses FLARE follow-up + evidence critic.
    #
    # It also loads the saved V4 A2C policy
    # including GatedEvidenceExtractor.

    sys.argv = [
        sys.argv[0],
        *argv,
    ]

    replay.main()

    analyze_real_weights(
        out,
        split,
    )

    verify_real_replay(
        out,
        split,
        original_root,
    )

    log()
    log("V4 EVIDENCE RELIANCE ANALYSIS: COMPLETE")


if __name__ == "__main__":
    main()
