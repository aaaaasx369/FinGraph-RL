#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_data, permute_observation_feature
from src.env import PortfolioEnv
from src.train import load_model
from src.evaluate import evaluate_model


def mean_reallocated_pct(base_w: np.ndarray, pert_w: np.ndarray) -> float:
    """
    0.5 * L1 distance is the fraction of capital that must be reallocated
    to transform one portfolio into the other.
    """
    n = min(len(base_w), len(pert_w))
    if n == 0:
        return np.nan
    return float((0.5 * np.abs(base_w[:n] - pert_w[:n]).sum(axis=1)).mean() * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--agent", default="PPO")
    ap.add_argument("--mode", default="fusion")
    ap.add_argument("--model", required=True, help="Path to SB3 model.zip or path without .zip")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--permute-gate", action="store_true",
                    help="Also permute raw RAG used by soft/veto gate. Default isolates policy influence.")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "permutation_importance.csv"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ["market_csv", "rag_csv"]:
        p = Path(cfg["data"][k])
        if not p.is_absolute():
            cfg["data"][k] = str(ROOT / p)

    _, test, _, _ = prepare_data(cfg)
    env = PortfolioEnv(test, cfg, mode=args.mode)
    model = load_model(args.agent, args.model, env=env, device=cfg["rl"].get("device", "auto"))

    _, base_w, _ = evaluate_model(model, test, cfg, args.mode)

    rows = []
    specs = (
        [("technical", i, name) for i, name in enumerate(test.tech_names)]
        + [("rag", i, name) for i, name in enumerate(test.rag_names)]
    )

    for group, idx, name in specs:
        vals = []
        for r in range(args.repeats):
            pert = permute_observation_feature(
                test,
                feature_group=group,
                feature_index=idx,
                seed=1000 + r,
                permute_gate_signal=(args.permute_gate and group == "rag"),
            )
            _, pert_w, _ = evaluate_model(model, pert, cfg, args.mode)
            vals.append(mean_reallocated_pct(base_w, pert_w))
        rows.append({
            "group": group,
            "feature": name,
            "mean_reallocated_weight_pct": float(np.mean(vals)),
            "std_reallocated_weight_pct": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "repeats": len(vals),
            "permute_gate_signal": bool(args.permute_gate and group == "rag"),
        })

    out = pd.DataFrame(rows).sort_values(
        "mean_reallocated_weight_pct", ascending=False
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
