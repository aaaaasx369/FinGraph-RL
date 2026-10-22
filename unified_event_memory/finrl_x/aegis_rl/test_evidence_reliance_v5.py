#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
V5 Evidence Reliance with REAL replay parity guard.

- Loads V5 frozen A2C model
- Uses original V5 test_queries.csv
- Installs V4 FLARE / Evidence Critic
- Runs REAL / ZERO / SHUFFLED / LAGGED
- Aborts immediately if REAL does not reproduce V5 Test
- Never trains or modifies the saved model
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import A2C

from unified_event_memory.finrl_x.aegis_rl import (
    test_evidence_reliance_v3 as replay,
    test_evidence_reliance_v4 as v4_replay,
)

EXPECTED_RETURN = -0.08991709305265161
RETURN_TOLERANCE = 1e-6


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="results/aegis_rl_v1/active_rag_v5_smoke",
    )
    parser.add_argument(
        "--out",
        default="results/aegis_rl_v1/evidence_reliance_v5",
    )
    parser.add_argument(
        "--split",
        choices=["test"],
        default="test",
    )
    parser.add_argument(
        "--lag-days",
        type=int,
        default=90,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()
    run = root / "runs" / "ACTIVE__A2C__seed42"

    paths = {
        "model": run / "model.zip",
        "router": run / "router.pt",
        "trace": run / "test_queries.csv",
        "original": run / "test_result.json",
    }

    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{name}: {path}"
            )

    original = json.loads(
        paths["original"].read_text(
            encoding="utf-8"
        )
    )

    if original.get("split") != "test":
        raise RuntimeError("Original split is not test")

    if original.get("agent") != "A2C":
        raise RuntimeError("Original agent is not A2C")

    if original.get("seed") != 42:
        raise RuntimeError("Original seed mismatch")

    original_return = float(
        original["total_return"]
    )

    if abs(original_return - EXPECTED_RETURN) > 1e-8:
        raise RuntimeError(
            "Unexpected original V5 Test return"
        )

    model = A2C.load(
        str(paths["model"]),
        device=args.device,
    )

    extractor = model.policy.features_extractor

    if type(extractor).__name__ != "GatedEvidenceExtractor":
        raise RuntimeError(
            "V5 GatedEvidenceExtractor not loaded"
        )

    if model.observation_space.shape != (3323,):
        raise RuntimeError(
            "V5 observation dimension mismatch"
        )

    print("===== V5 PRECHECK =====", flush=True)
    print("Model:", paths["model"], flush=True)
    print("Trace:", paths["trace"], flush=True)
    print(
        "Router checkpoint:",
        paths["router"],
        flush=True,
    )
    print(
        "Router mode: FrozenRouter(test_queries.csv)",
        flush=True,
    )
    print(
        "Original return:",
        original_return,
        flush=True,
    )
    print(
        "Gated Fusion:",
        type(extractor).__name__,
        flush=True,
    )

    out.mkdir(parents=True, exist_ok=True)

    # V3's main() calls run_episode() once per mode.
    # The first invocation is REAL.
    original_run_episode = replay.run_episode
    evaluated = {"count": 0}

    def guarded_run_episode(model, env):
        metrics, history, allocations = (
            original_run_episode(model, env)
        )

        evaluated["count"] += 1

        if evaluated["count"] == 1:
            replay_return = float(
                metrics["total_return"]
            )

            difference = (
                replay_return - original_return
            )

            print()
            print(
                "===== V5 REAL REPLAY CONSISTENCY =====",
                flush=True,
            )
            print(
                "Original:",
                f"{original_return:.12f}",
                flush=True,
            )
            print(
                "Replay:",
                f"{replay_return:.12f}",
                flush=True,
            )
            print(
                "Difference:",
                f"{difference:.12f}",
                flush=True,
            )

            checks = {
                "return": (
                    abs(difference)
                    <= RETURN_TOLERANCE
                ),
                "days": (
                    int(metrics["steps"])
                    == int(original["evaluation_days"])
                ),
            }

            for name in ("KG_hits", "SEC_hits"):
                if name in metrics:
                    checks[name] = (
                        int(metrics[name])
                        == int(original[name])
                    )

            for name, passed in checks.items():
                print(
                    f"[CHECK] {name}: "
                    f"{'PASS' if passed else 'FAIL'}",
                    flush=True,
                )

            if not all(checks.values()):
                raise RuntimeError(
                    "V5 REAL REPLAY FAILED. "
                    "Stopping before ZERO/SHUFFLED/LAGGED. "
                    "Check retrieval trace, cache, and "
                    "environment configuration."
                )

            print(
                "V5 REAL REPLAY CONSISTENCY: PASS",
                flush=True,
            )

        return metrics, history, allocations

    replay.run_episode = guarded_run_episode

    # v4_replay.main() installs V4 _search()
    # and delegates replay execution to V3.
    sys.argv = [
        sys.argv[0],
        "--root", str(root),
        "--out", str(out),
        "--split", "test",
        "--lag-days", str(args.lag_days),
        "--device", args.device,
    ]

    try:
        v4_replay.main()
    finally:
        replay.run_episode = original_run_episode

    if evaluated["count"] != 4:
        raise RuntimeError(
            "Expected four evidence interventions, "
            f"got {evaluated['count']}"
        )

    print()
    print("=" * 72)
    print("V5 EVIDENCE RELIANCE TEST: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
