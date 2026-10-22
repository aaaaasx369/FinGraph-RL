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

from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise

ROOT = Path("/home/doralin/projects/FinRL-Trading")
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x import (
    run_dynamic_kg_drl as runner
)

from unified_event_memory.finrl_x import (
    run_temporal_ablation as ablation
)

TICKERS = list(runner.TICKERS)
KG_COLS = list(runner.KG_COLS)

DEFAULT_OUT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/"
    "results/ddpg_saturation_fix_v2"
)


def log(*args):
    print(*args, flush=True)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log("SAVED:", path)


def prepare_index(df):
    """
    FinRL requires the same integer index
    for all stocks on the same trading date.
    """
    df = runner.prepare_panel(df).copy()

    df.index = (
        pd.factorize(
            df["date"],
            sort=True,
        )[0]
    )

    counts = df.groupby(level=0)["tic"].nunique()

    if not (counts == len(TICKERS)).all():
        raise RuntimeError(
            "Incomplete date-indexed stock panel"
        )

    return df


class LogReturnReward(gym.Wrapper):
    """
    Replace the original portfolio-value reward
    with scaled portfolio log return.

    Original environment:
        reward ~= portfolio value, e.g. 1,000,000

    New reward:
        100 * log(V_t / V_{t-1})

    Portfolio accounting and trading transitions
    remain unchanged.
    """

    def __init__(self, env, reward_scale=100.0):
        super().__init__(env)
        self.reward_scale = float(reward_scale)

    def step(self, action):
        previous_value = float(
            self.env.unwrapped.portfolio_value
        )

        obs, raw_reward, terminated, truncated, info = (
            self.env.step(action)
        )

        current_value = float(
            self.env.unwrapped.portfolio_value
        )

        if (
            previous_value <= 0
            or current_value <= 0
        ):
            raise RuntimeError(
                "Nonpositive portfolio value"
            )

        reward = self.reward_scale * np.log(
            current_value / previous_value
        )

        if not np.isfinite(reward):
            raise RuntimeError(
                "Nonfinite log return reward"
            )

        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )


def build_panels():
    log("\n===== BUILD ORIGINAL PANELS =====")

    market, raw_kg = (
        ablation.load_market_and_dynamic(
            ROOT / "outputs/unified_kgt_finrl_x_v1"
        )
    )

    frozen_raw, frozen_date = (
        ablation.make_frozen_dynamic(raw_kg)
    )

    panels = {
        "clean": market.copy(),
        "frozen_dynamic": runner.attach_kg(
            market, frozen_raw
        ),
        "real_dynamic": runner.attach_kg(
            market, raw_kg
        ),
    }

    features = {
        "clean": list(runner.exp.BASE_FEATURES),
        "frozen_dynamic": (
            list(runner.exp.BASE_FEATURES)
            + KG_COLS
        ),
        "real_dynamic": (
            list(runner.exp.BASE_FEATURES)
            + KG_COLS
        ),
    }

    for name in panels:
        panels[name] = prepare_index(
            panels[name]
        )

        log(
            name,
            "rows =", len(panels[name]),
            "features =", len(features[name]),
        )

    log("Frozen reference:", frozen_date)

    return panels, features


def make_env(df, features, reward_scale=None):
    env = runner.environment(
        df.copy(),
        features,
    )

    if reward_scale is not None:
        env = LogReturnReward(
            env,
            reward_scale,
        )

    return env


def split_panel(df):
    train = df.loc[
        (df.date >= "2018-01-01")
        & (df.date < "2022-01-01")
    ].copy()

    val = df.loc[
        (df.date >= "2022-01-01")
        & (df.date < "2023-01-01")
    ].copy()

    test = df.loc[
        (df.date >= "2023-01-01")
        & (df.date < "2024-01-01")
    ].copy()

    return (
        prepare_index(train),
        prepare_index(val),
        prepare_index(test),
    )


class SaturationMonitor(BaseCallback):
    """
    Inspect trained Actor on a fixed collection
    of observations during learning.
    """

    def __init__(
        self,
        probe,
        check_freq=5000,
        verbose=0,
    ):
        super().__init__(verbose)

        self.probe = np.asarray(
            probe,
            dtype=np.float32,
        )

        self.check_freq = int(check_freq)
        self.rows = []

    def _on_step(self):
        if self.num_timesteps % self.check_freq != 0:
            return True

        actor = self.model.policy.actor

        with torch.no_grad():
            tensor, _ = (
                self.model.policy.obs_to_tensor(
                    self.probe
                )
            )

            # SB3 DDPG Actor features
            features = actor.extract_features(
                tensor,
                actor.features_extractor,
            )

            pre_tanh = None

            for layer in actor.mu.children():
                features = layer(features)

                if isinstance(
                    layer,
                    torch.nn.Linear,
                ):
                    pre_tanh = features

            # The final mu output is after Tanh.
            post_tanh = features

            post = (
                post_tanh.detach()
                .cpu()
                .numpy()
            )

            pre = (
                pre_tanh.detach()
                .cpu()
                .numpy()
            )

        actions = []

        for obs in self.probe:
            action, _ = self.model.predict(
                obs,
                deterministic=True,
            )

            actions.append(
                np.asarray(action).reshape(-1)
            )

        actions = np.stack(actions)

        unique_actions = len(
            np.unique(
                np.round(actions, 6),
                axis=0,
            )
        )

        record = {
            "step": self.num_timesteps,
            "pre_tanh_mean_abs": float(
                np.abs(pre).mean()
            ),
            "pre_tanh_max_abs": float(
                np.abs(pre).max()
            ),
            "saturation_099": float(
                np.mean(
                    np.abs(post) >= 0.99
                )
            ),
            "saturation_0999": float(
                np.mean(
                    np.abs(post) >= 0.999
                )
            ),
            "action_boundary_fraction": float(
                np.mean(
                    (actions <= 1e-6)
                    | (actions >= 1 - 1e-6)
                )
            ),
            "unique_actions_6dp": int(
                unique_actions
            ),
            "action_std_mean": float(
                actions.std(axis=0).mean()
            ),
        }

        self.rows.append(record)

        log(
            "[SATURATION]",
            "step =", record["step"],
            "pre =", round(
                record["pre_tanh_mean_abs"], 3
            ),
            "sat99 =", round(
                record["saturation_099"], 4
            ),
            "unique =", unique_actions,
            "action_std =", round(
                record["action_std_mean"], 6
            ),
        )

        return True


def build_probe(df, features, max_samples=24):
    """
    Construct probe observations from the
    original environment with no training.
    """

    env = make_env(
        df,
        features,
        reward_scale=None,
    )

    obs, _ = env.reset()

    samples = []

    n_dates = df["date"].nunique()

    for i in range(n_dates):
        if (
            i % max(1, n_dates // max_samples) == 0
            and len(samples) < max_samples
        ):
            samples.append(
                np.asarray(
                    obs,
                    dtype=np.float32,
                ).copy()
            )

        if i >= n_dates - 1:
            break

        action = np.full(
            len(TICKERS),
            1 / len(TICKERS),
            dtype=np.float32,
        )

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            break

    env.close()

    if not samples:
        raise RuntimeError("No probe states")

    return np.stack(samples)


def train_ddpg(
    train_df,
    features,
    use_kg,
    seed,
    timesteps,
    reward_scale,
    learning_rate,
    check_freq,
):
    log("\n===== TRAIN DDPG V2 =====")

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    def factory():
        return make_env(
            train_df,
            features,
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

    noise = NormalActionNoise(
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
        action_noise=noise,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device="cpu",
        verbose=0,
    )

    probe = build_probe(
        train_df,
        features,
    )

    monitor = SaturationMonitor(
        probe=probe,
        check_freq=check_freq,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=monitor,
    )

    return model, monitor, vec_env


def softmax(action):
    a = np.asarray(
        action,
        dtype=np.float64,
    )

    x = np.exp(a - np.max(a))

    return x / x.sum()


def evaluate(model, panel, features):
    env = make_env(
        panel,
        features,
        reward_scale=None,
    )

    obs, _ = env.reset()

    allocation_rows = []

    dates = sorted(panel.date.unique())

    for i in range(len(dates) - 1):
        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        action = np.asarray(
            action
        ).reshape(-1)

        weights = softmax(action)

        row = {
            "date": dates[i],
        }

        for j, tic in enumerate(TICKERS):
            row[f"action_{tic}"] = float(
                action[j]
            )
            row[f"weight_{tic}"] = float(
                weights[j]
            )

        allocation_rows.append(row)

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        if terminated or truncated:
            break

    assets = np.asarray(
        env.unwrapped.asset_memory,
        dtype=np.float64,
    )

    returns = assets[1:] / assets[:-1] - 1

    final_multiple = (
        assets[-1] / assets[0]
    )

    sharpe = (
        float(
            np.sqrt(252)
            * returns.mean()
            / returns.std(ddof=1)
        )
        if len(returns) > 1
        and returns.std(ddof=1) > 0
        else float("nan")
    )

    peak = np.maximum.accumulate(assets)

    mdd = float(
        np.min(
            assets / peak - 1
        )
    )

    alloc = pd.DataFrame(
        allocation_rows
    )

    cols = [
        f"weight_{tic}"
        for tic in TICKERS
    ]

    w = alloc[cols].to_numpy(
        dtype=np.float64
    )

    unique_allocations = len(
        np.unique(
            np.round(w, 6),
            axis=0,
        )
    )

    allocation_change = (
        0.5 * np.abs(
            np.diff(w, axis=0)
        ).sum(axis=1)
    )

    summary = {
        "final_multiple": float(
            final_multiple
        ),
        "final_value": float(
            assets[-1]
        ),
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "unique_allocations_6dp":
            unique_allocations,
        "mean_target_weight_change": (
            float(allocation_change.mean())
            if len(allocation_change)
            else 0.0
        ),
    }

    env.close()

    return summary, alloc


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
        default=30000,
    )

    parser.add_argument(
        "--reward-scale",
        type=float,
        default=100.0,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--check-freq",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        choices=[
            "clean",
            "frozen_dynamic",
            "real_dynamic",
        ],
        default=[
            "clean",
            "frozen_dynamic",
            "real_dynamic",
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
            "Use at least 6000 timesteps"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    panels, features_by_group = build_panels()

    results = []

    for group in args.groups:
        log("\n" + "=" * 70)
        log("GROUP:", group)
        log("=" * 70)

        group_dir = (
            args.out
            / f"seed_{args.seed}"
            / group
        )

        group_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        train, val, test = split_panel(
            panels[group]
        )

        features = features_by_group[group]

        model, monitor, vec_env = train_ddpg(
            train_df=train,
            features=features,
            use_kg=(group != "clean"),
            seed=args.seed,
            timesteps=args.timesteps,
            reward_scale=args.reward_scale,
            learning_rate=args.learning_rate,
            check_freq=args.check_freq,
        )

        monitor_df = pd.DataFrame(
            monitor.rows
        )

        save_csv(
            monitor_df,
            group_dir / "saturation_training.csv"
        )

        model.save(
            str(group_dir / "ddpg_v2")
        )

        val_summary, val_alloc = evaluate(
            model,
            val,
            features,
        )

        test_summary, test_alloc = evaluate(
            model,
            test,
            features,
        )

        save_csv(
            val_alloc,
            group_dir / "validation_allocations.csv"
        )

        save_csv(
            test_alloc,
            group_dir / "test_allocations.csv"
        )

        result = {
            "seed": args.seed,
            "group": group,
            "timesteps": args.timesteps,
            "learning_rate": args.learning_rate,
            "reward_scale": args.reward_scale,
            "val_sharpe": val_summary["sharpe"],
            "val_multiple":
                val_summary["final_multiple"],
            "test_sharpe": test_summary["sharpe"],
            "test_multiple":
                test_summary["final_multiple"],
            "test_mdd":
                test_summary["max_drawdown"],
            "test_unique_allocations":
                test_summary[
                    "unique_allocations_6dp"
                ],
            "test_mean_weight_change":
                test_summary[
                    "mean_target_weight_change"
                ],
            "final_saturation_099": (
                float(
                    monitor_df[
                        "saturation_099"
                    ].iloc[-1]
                )
                if not monitor_df.empty
                else float("nan")
            ),
        }

        results.append(result)

        save_csv(
            pd.DataFrame([result]),
            group_dir / "summary.csv"
        )

        log("\nRESULT:", result)

        vec_env.close()

        del model

    summary = pd.DataFrame(results)

    summary_path = (
        args.out
        / f"seed_{args.seed}"
        / "comparison.csv"
    )

    save_csv(
        summary,
        summary_path
    )

    config = {
        "seed": args.seed,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "reward_scale": args.reward_scale,
        "groups": args.groups,
        "method": "scaled log return reward",
        "model": "SB3 DDPG",
        "kg": "Original Unified Dynamic KGT",
        "training_period": "2018-2021",
        "validation_period": "2022",
        "test_period": "2023",
    }

    with (
        args.out
        / f"seed_{args.seed}"
        / "experiment_config.json"
    ).open("w") as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    log("\n===== DDPG V2 COMPLETED =====")
    log(summary.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
