#!/usr/bin/env python3

import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg.train_option_b_threeway import (
    load_native,
    load_dual,
)

from experiments.light_rag_dynamic_kg import (
    train_finrlx_multidrl as legacy,
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_v16_dual_rag_finrl as v16,
)


OUT = ROOT / (
    "results/light_rag_dynamic_kg/option_b_smoke"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
TIMESTEPS = 2000
ALGORITHM = "PPO"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    print("=" * 75)
    print("OPTION B M0/M1 PPO SMOKE")
    print("=" * 75)

    print("\n[1] Loading original V16")

    opt, dates, prices, base, n_train = load_native(OUT)

    print("BASE:", base.shape)

    print("\n[2] Loading KG Structural")

    structural, embeddings, valid, names = (
        load_dual(dates, n_train)
    )

    conditions = {
        "M0": base,
        "M1": np.concatenate(
            [base, structural],
            axis=2,
        ).astype(np.float32),
    }

    metrics_all = []

    for arm, x in conditions.items():

        print(f"\n========== TRAIN {arm} ==========")

        seed_everything(SEED)

        train_env = v16.DualRAGPortfolioEnv(
            x[:n_train],
            prices[:n_train],
            dates[:n_train],
            opt.transaction_cost,
            opt.reward_scale,
        )

        val_env = v16.DualRAGPortfolioEnv(
            x[n_train:],
            prices[n_train:],
            dates[n_train:],
            opt.transaction_cost,
            opt.reward_scale,
        )

        model = legacy.make_model(
            ALGORITHM,
            train_env,
            SEED,
            3e-4,
            "cpu",
        )

        print("OBS:", train_env.observation_space)
        print("TRAIN TIMESTEPS:", TIMESTEPS)

        model.learn(
            total_timesteps=TIMESTEPS,
            progress_bar=False,
        )

        model_path = OUT / f"{arm}_PPO_seed{SEED}"

        if model_path.with_suffix(".zip").exists():
            raise FileExistsError(
                f"Model already exists: {model_path}.zip"
            )

        model.save(str(model_path))

        history, metrics = v16.evaluate(
            model,
            val_env,
            arm,
            SEED,
            ALGORITHM,
        )

        history.to_csv(
            OUT / f"{arm}_validation.csv",
            index=False,
        )

        metrics_all.append(metrics)

        print("\nMETRICS:")
        print(json.dumps(
            metrics,
            indent=2,
            default=float,
        ))

    summary = pd.DataFrame(metrics_all)
    summary.to_csv(
        OUT / "smoke_summary.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("SMOKE TEST COMPLETE")
    print("=" * 75)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
