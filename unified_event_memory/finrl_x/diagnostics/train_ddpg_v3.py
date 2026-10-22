#!/usr/bin/env python3

import argparse
import json
import sys
import traceback
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from gymnasium import spaces
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

ROOT = Path("/home/doralin/projects/FinRL-Trading")
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.diagnostics import (
    train_ddpg_v2 as v2
)
from unified_event_memory.finrl_x import (
    run_dynamic_kg_drl as runner
)

TICKERS = list(runner.TICKERS)
KG_COLS = list(runner.KG_COLS)

DEFAULT_OUT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/"
    "results/ddpg_v3_normalization"
)

CONFIGS = {
    "v3a": {
        "learning_rate": 1e-4,
        "reward_scale": 100.0,
    },
    "v3b": {
        "learning_rate": 1e-4,
        "reward_scale": 10.0,
    },
    "v3c": {
        "learning_rate": 5e-5,
        "reward_scale": 10.0,
    },
}


def log(*args):
    print(*args, flush=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("SAVED:", path)


class ObservationNormalizer(gym.ObservationWrapper):
    """
    Normalize by observation row, pooling all five
    companies.

    mean/std shape: [n_rows, 1]

    Same fitted statistics are applied to every
    ticker and to 2022 validation observations.
    """

    def __init__(self, env, mean, std, clip=5.0):
        super().__init__(env)

        self.mean = np.asarray(
            mean, dtype=np.float32
        )
        self.std = np.asarray(
            std, dtype=np.float32
        )
        self.clip = float(clip)

        if len(env.observation_space.shape) != 2:
            raise RuntimeError(
                "Expected 2D portfolio observation"
            )

        shape = env.observation_space.shape

        if self.mean.shape != (shape[0], 1):
            raise RuntimeError(
                f"Mean shape mismatch: "
                f"{self.mean.shape} vs {shape}"
            )

        if self.std.shape != self.mean.shape:
            raise RuntimeError(
                "Mean/std shape mismatch"
            )

        self.observation_space = spaces.Box(
            low=-self.clip,
            high=self.clip,
            shape=shape,
            dtype=np.float32,
        )

    def observation(self, observation):
        x = np.asarray(
            observation, dtype=np.float32
        )

        z = (x - self.mean) / self.std
        z = np.clip(
            z, -self.clip, self.clip
        ).astype(np.float32)

        if not np.isfinite(z).all():
            raise RuntimeError(
                "Nonfinite normalized observation"
            )

        return z


def collect_train_observations(df, features):
    """
    Collect states for the 2018-2021 training
    period only.

    No validation/test observations are used.
    """
    env = runner.environment(
        df.copy(), features
    )

    observations = []

    try:
        result = env.reset()
        obs = result[0] if isinstance(
            result, tuple
        ) else result

        n_dates = df.date.nunique()

        for i in range(n_dates):
            observations.append(
                np.asarray(
                    obs, dtype=np.float64
                ).copy()
            )

            if i >= n_dates - 1:
                break

            action = np.full(
                len(TICKERS),
                1.0 / len(TICKERS),
                dtype=np.float32,
            )

            result = env.step(action)

            if len(result) == 5:
                obs, _, terminated, truncated, _ = result
                done = terminated or truncated
            elif len(result) == 4:
                obs, _, done, _ = result
            else:
                raise RuntimeError(
                    "Unsupported environment step"
                )

            if done:
                raise RuntimeError(
                    "Training environment ended early"
                )

    finally:
        env.close()

    return np.stack(observations, axis=0)


def fit_normalizer(train_df, features, out):
    """
    Train-only normalization.

    [dates, rows, stocks] -> statistics [rows, 1]

    Uses pooled companies, NOT separate company
    scalers, so Frozen KG company distinctions
    remain available.
    """
    log("\n===== FIT TRAIN-ONLY NORMALIZER =====")

    observations = collect_train_observations(
        train_df, features
    )

    if not np.isfinite(observations).all():
        raise RuntimeError(
            "Nonfinite raw train observations"
        )

    # Mean across dates and companies.
    mean = observations.mean(
        axis=(0, 2)
    )[:, None]

    std = observations.std(
        axis=(0, 2)
    )[:, None]

    std = np.where(
        (~np.isfinite(std)) | (std < 1e-8),
        1.0,
        std,
    )

    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    out.mkdir(
        parents=True, exist_ok=True
    )

    np.savez_compressed(
        out / "observation_scaler.npz",
        mean=mean,
        std=std,
        clip=np.array([5.0]),
    )

    # Verify scaler actually normalizes observation.
    z = np.clip(
        (observations - mean[None, :, :])
        / std[None, :, :],
        -5.0,
        5.0,
    )

    if not np.isfinite(z).all():
        raise RuntimeError(
            "Nonfinite normalized train observations"
        )

    kg_start = len(features) + len(TICKERS) - len(KG_COLS)

    if kg_start < 0:
        raise RuntimeError(
            "Invalid KG observation offset"
        )

    if len(KG_COLS) > 0 and all(
        c in features for c in KG_COLS
    ):
        kg = z[:, -len(KG_COLS):, :]

        # Across-company variation must not be
        # destroyed by per-stock normalization.
        cross_company_std = kg.std(
            axis=2
        ).mean()

        log(
            "Normalized KG cross-company std:",
            float(cross_company_std)
        )

    log(
        "Train observation shape:",
        observations.shape
    )

    log(
        "Train normalized mean abs:",
        float(np.abs(z.mean()))
    )

    log(
        "Train normalized max abs:",
        float(np.max(np.abs(z)))
    )

    return mean, std


def make_env(
    df,
    features,
    mean,
    std,
    reward_scale=None,
):
    env = runner.environment(
        df.copy(), features
    )

    if reward_scale is not None:
        env = v2.LogReturnReward(
            env, reward_scale
        )

    env = ObservationNormalizer(
        env=env,
        mean=mean,
        std=std,
        clip=5.0,
    )

    return env


def build_probe(
    train_df,
    features,
    mean,
    std,
    max_samples=24,
):
    env = make_env(
        train_df,
        features,
        mean,
        std,
        reward_scale=None,
    )

    samples = []

    try:
        obs, _ = env.reset()
        n_dates = train_df.date.nunique()

        stride = max(
            1, n_dates // max_samples
        )

        for i in range(n_dates):
            if (
                i % stride == 0
                and len(samples) < max_samples
            ):
                samples.append(
                    np.asarray(
                        obs, dtype=np.float32
                    ).copy()
                )

            if i >= n_dates - 1:
                break

            action = np.full(
                len(TICKERS),
                1.0 / len(TICKERS),
                dtype=np.float32,
            )

            obs, _, terminated, truncated, _ = (
                env.step(action)
            )

            if terminated or truncated:
                raise RuntimeError(
                    "Probe environment ended early"
                )

    finally:
        env.close()

    return np.stack(samples)


def train(
    train_df,
    features,
    use_kg,
    seed,
    timesteps,
    learning_rate,
    reward_scale,
    mean,
    std,
    check_freq,
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def factory():
        return make_env(
            train_df,
            features,
            mean,
            std,
            reward_scale=reward_scale,
        )

    vec_env = DummyVecEnv([factory])

    policy_kwargs = {}

    if use_kg:
        policy_kwargs = {
            "features_extractor_class":
                runner.KGProjectionExtractor,
            "features_extractor_kwargs": {
                "kg_dim": len(KG_COLS),
                "projected_dim": 16,
                "stock_dim": len(TICKERS),
            },
        }

    action_noise = NormalActionNoise(
        mean=np.zeros(len(TICKERS)),
        sigma=0.1 * np.ones(len(TICKERS)),
    )

    model = DDPG(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        batch_size=256,
        buffer_size=100000,
        learning_starts=5000,
        train_freq=(1, "step"),
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        action_noise=action_noise,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="cpu",
        verbose=0,
    )

    probe = build_probe(
        train_df,
        features,
        mean,
        std,
    )

    monitor = v2.SaturationMonitor(
        probe=probe,
        check_freq=check_freq,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=monitor,
    )

    return model, monitor, vec_env


def evaluate_validation(
    model,
    df,
    features,
    mean,
    std,
):
    """
    Evaluation only on 2022 validation.

    Uses the original portfolio accounting
    but train-fitted normalized observations.
    """
    env = make_env(
        df,
        features,
        mean,
        std,
        reward_scale=None,
    )

    rows = []

    try:
        obs, _ = env.reset()
        dates = sorted(df.date.unique())

        for i in range(len(dates) - 1):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            action = np.asarray(
                action,
                dtype=np.float64,
            ).reshape(-1)

            # Original FinRL environment applies
            # Softmax to raw actions.
            weights = v2.softmax(action)

            row = {"date": dates[i]}

            for j, tic in enumerate(TICKERS):
                row[f"action_{tic}"] = float(
                    action[j]
                )
                row[f"weight_{tic}"] = float(
                    weights[j]
                )

            rows.append(row)

            obs, _, terminated, truncated, _ = (
                env.step(
                    action.astype(np.float32)
                )
            )

            if terminated or truncated:
                raise RuntimeError(
                    "Validation ended early"
                )

        assets = np.asarray(
            env.unwrapped.asset_memory,
            dtype=np.float64,
        )

    finally:
        env.close()

    if len(assets) < 3:
        raise RuntimeError(
            "Not enough validation returns"
        )

    returns = assets[1:] / assets[:-1] - 1

    vol = returns.std(ddof=1)

    sharpe = (
        float(
            np.sqrt(252)
            * returns.mean()
            / vol
        )
        if vol > 0
        else float("nan")
    )

    peak = np.maximum.accumulate(
        assets
    )

    summary = {
        "val_final_multiple": float(
            assets[-1] / assets[0]
        ),
        "val_sharpe": sharpe,
        "val_mdd": float(
            np.min(assets / peak - 1)
        ),
    }

    allocation = pd.DataFrame(rows)

    weight_cols = [
        f"weight_{tic}"
        for tic in TICKERS
    ]

    w = allocation[
        weight_cols
    ].to_numpy(np.float64)

    unique = np.unique(
        np.round(w, 6),
        axis=0,
    ).shape[0]

    change = (
        0.5 * np.abs(
            np.diff(w, axis=0)
        ).sum(axis=1)
    )

    summary["val_unique_allocations"] = int(
        unique
    )

    summary["val_mean_weight_change"] = (
        float(change.mean())
        if change.size
        else 0.0
    )

    return summary, allocation


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--check-freq",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--group",
        choices=[
            "clean",
            "frozen_dynamic",
            "real_dynamic",
        ],
        default="real_dynamic",
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=[
            "v3a",
            "v3b",
            "v3c",
        ],
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    if args.timesteps < 6000:
        raise ValueError(
            "Timesteps must be at least 6000"
        )

    args.out.mkdir(
        parents=True, exist_ok=True
    )

    log("\n===== LOAD DATA =====")

    panels, features_by_group = (
        v2.build_panels()
    )

    panel = panels[args.group]
    features = features_by_group[
        args.group
    ]

    train_df, val_df, test_df = (
        v2.split_panel(panel)
    )

    # No test-set usage in this sweep.
    del test_df

    rows = []

    for config_name in args.configs:
        config = CONFIGS[config_name]

        run_dir = (
            args.out
            / f"seed_{args.seed}"
            / args.group
            / config_name
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        log("\n" + "=" * 70)
        log(
            "EXPERIMENT:",
            args.group,
            config_name,
        )
        log("CONFIG:", config)
        log("=" * 70)

        mean, std = fit_normalizer(
            train_df,
            features,
            run_dir,
        )

        model, monitor, vec_env = train(
            train_df=train_df,
            features=features,
            use_kg=(
                args.group != "clean"
            ),
            seed=args.seed,
            timesteps=args.timesteps,
            learning_rate=config[
                "learning_rate"
            ],
            reward_scale=config[
                "reward_scale"
            ],
            mean=mean,
            std=std,
            check_freq=args.check_freq,
        )

        saturation = pd.DataFrame(
            monitor.rows
        )

        save_csv(
            saturation,
            run_dir / "saturation_training.csv"
        )

        model.save(
            str(run_dir / "ddpg_v3")
        )

        val_summary, allocation = (
            evaluate_validation(
                model,
                val_df,
                features,
                mean,
                std,
            )
        )

        save_csv(
            allocation,
            run_dir / "validation_allocations.csv",
        )

        last = saturation.iloc[-1]

        result = {
            "seed": args.seed,
            "group": args.group,
            "config": config_name,
            "timesteps": args.timesteps,
            "learning_rate": config[
                "learning_rate"
            ],
            "reward_scale": config[
                "reward_scale"
            ],
            **val_summary,
            "final_train_sat99": float(
                last["saturation_099"]
            ),
            "final_train_sat999": float(
                last["saturation_0999"]
            ),
            "final_train_pre_abs": float(
                last["pre_tanh_mean_abs"]
            ),
            "final_train_action_std": float(
                last["action_std_mean"]
            ),
        }

        rows.append(result)

        save_csv(
            pd.DataFrame([result]),
            run_dir / "summary.csv",
        )

        metadata = {
            "normalization": (
                "Per observation row, "
                "pooled across dates and tickers; "
                "fit on 2018-2021 only"
            ),
            "normalization_clip": 5.0,
            "reward_definition": (
                "reward_scale * "
                "log(V_t / V_(t-1))"
            ),
            "train": "2018-2021",
            "validation": "2022",
            "test_used_for_selection": False,
            "seed": args.seed,
            "group": args.group,
            "config": config_name,
            "timesteps": args.timesteps,
            **config,
        }

        with (
            run_dir / "metadata.json"
        ).open("w") as f:
            json.dump(
                metadata,
                f,
                indent=2,
            )

        log("RESULT:", result)

        vec_env.close()
        del model

    summary = pd.DataFrame(rows)

    summary_path = (
        args.out
        / f"seed_{args.seed}"
        / args.group
        / "v3_comparison.csv"
    )

    save_csv(
        summary,
        summary_path,
    )

    log("\n===== V3 VALIDATION COMPARISON =====")
    log(
        summary.to_string(index=False)
    )

    log("\n===== FINISHED =====")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
