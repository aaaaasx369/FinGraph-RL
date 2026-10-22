#!/usr/bin/env python3

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from stable_baselines3 import DDPG
from stable_baselines3.common.utils import polyak_update

ROOT = Path("/home/doralin/projects/FinRL-Trading")
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.diagnostics import (
    train_ddpg_v2 as v2,
    train_ddpg_v3 as v3,
)

DEFAULT_OUT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/"
    "results/ddpg_v4_actor_regularization"
)

CONFIGS = {
    "v4a": {
        "actor_lr": 1e-4,
        "critic_lr": 1e-4,
        "action_penalty": 0.0,
    },
    "v4b": {
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "action_penalty": 0.0,
    },
    "v4c": {
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "action_penalty": 0.1,
    },
}


def grad_norm(parameters):
    values = []

    for p in parameters:
        if p.grad is not None:
            values.append(
                p.grad.detach().norm(2).square()
            )

    if not values:
        return 0.0

    return float(torch.stack(values).sum().sqrt().item())


class V4DDPG(DDPG):
    """
    DDPG with:
      - separate actor/critic learning rates
      - optional actor output regularization
      - Q-value and gradient diagnostics

    No alteration to the environment's
    portfolio accounting.
    """

    actor_lr_v4 = 1e-4
    critic_lr_v4 = 1e-4
    action_penalty_v4 = 0.0

    def _setup_model(self):
        super()._setup_model()
        self.v4_diagnostics = []

    def train(self, gradient_steps, batch_size=100):
        self.policy.set_training_mode(True)

        self._update_learning_rate([
            self.actor.optimizer,
            self.critic.optimizer,
        ])

        # SB3 normally assigns the same learning
        # rate to both optimizers. Override after
        # its scheduler update.
        for group in self.actor.optimizer.param_groups:
            group["lr"] = self.actor_lr_v4

        for group in self.critic.optimizer.param_groups:
            group["lr"] = self.critic_lr_v4

        actor_losses = []
        critic_losses = []

        for _ in range(gradient_steps):
            self._n_updates += 1

            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )

            with torch.no_grad():
                noise = (
                    replay_data.actions.clone()
                    .normal_(0, self.target_policy_noise)
                )

                noise = noise.clamp(
                    -self.target_noise_clip,
                    self.target_noise_clip,
                )

                next_actions = (
                    self.actor_target(
                        replay_data.next_observations
                    ) + noise
                ).clamp(-1, 1)

                next_q = torch.cat(
                    self.critic_target(
                        replay_data.next_observations,
                        next_actions,
                    ),
                    dim=1,
                )

                next_q, _ = torch.min(
                    next_q,
                    dim=1,
                    keepdim=True,
                )

                target_q = (
                    replay_data.rewards
                    + (1 - replay_data.dones)
                    * self.gamma
                    * next_q
                )

            current_q = self.critic(
                replay_data.observations,
                replay_data.actions,
            )

            critic_loss = sum(
                F.mse_loss(q, target_q)
                for q in current_q
            )

            self.critic.optimizer.zero_grad()
            critic_loss.backward()

            critic_gradient = grad_norm(
                self.critic.parameters()
            )

            self.critic.optimizer.step()

            critic_losses.append(
                float(critic_loss.item())
            )

            if self._n_updates % self.policy_delay == 0:
                actor_actions = self.actor(
                    replay_data.observations
                )

                q_actor = self.critic.q1_forward(
                    replay_data.observations,
                    actor_actions,
                )

                base_actor_loss = -q_actor.mean()

                penalty = actor_actions.square().mean()

                actor_loss = (
                    base_actor_loss
                    + self.action_penalty_v4 * penalty
                )

                self.actor.optimizer.zero_grad()
                actor_loss.backward()

                actor_gradient = grad_norm(
                    self.actor.parameters()
                )

                self.actor.optimizer.step()

                actor_losses.append(
                    float(actor_loss.item())
                )

                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )

                polyak_update(
                    self.actor.parameters(),
                    self.actor_target.parameters(),
                    self.tau,
                )

                if hasattr(
                    self,
                    "critic_batch_norm_stats",
                ) and hasattr(
                    self,
                    "critic_batch_norm_stats_target",
                ):
                    polyak_update(
                        self.critic_batch_norm_stats,
                        self.critic_batch_norm_stats_target,
                        1.0,
                    )

                if hasattr(
                    self,
                    "actor_batch_norm_stats",
                ) and hasattr(
                    self,
                    "actor_batch_norm_stats_target",
                ):
                    polyak_update(
                        self.actor_batch_norm_stats,
                        self.actor_batch_norm_stats_target,
                        1.0,
                    )

                if self._n_updates % 1000 == 0:
                    record = {
                        "gradient_update": int(
                            self._n_updates
                        ),
                        "timesteps": int(
                            self.num_timesteps
                        ),
                        "critic_loss": float(
                            critic_loss.item()
                        ),
                        "actor_loss": float(
                            actor_loss.item()
                        ),
                        "base_actor_loss": float(
                            base_actor_loss.item()
                        ),
                        "actor_action_penalty": float(
                            penalty.item()
                        ),
                        "target_q_mean": float(
                            target_q.mean().item()
                        ),
                        "target_q_abs_max": float(
                            target_q.abs().max().item()
                        ),
                        "critic_q_mean": float(
                            current_q[0].mean().item()
                        ),
                        "actor_q_mean": float(
                            q_actor.mean().item()
                        ),
                        "critic_gradient_norm": (
                            critic_gradient
                        ),
                        "actor_gradient_norm": (
                            actor_gradient
                        ),
                        "actor_lr": float(
                            self.actor_lr_v4
                        ),
                        "critic_lr": float(
                            self.critic_lr_v4
                        ),
                    }

                    self.v4_diagnostics.append(record)

                    print(
                        "[V4 GRADIENT]",
                        "step =", self.num_timesteps,
                        "Q =", round(
                            record["critic_q_mean"], 4
                        ),
                        "actor_grad =", round(
                            actor_gradient, 4
                        ),
                        "critic_grad =", round(
                            critic_gradient, 4
                        ),
                        flush=True,
                    )

        self.logger.record(
            "train/n_updates",
            self._n_updates,
            exclude="tensorboard",
        )

        if actor_losses:
            self.logger.record(
                "train/actor_loss",
                np.mean(actor_losses),
            )

        if critic_losses:
            self.logger.record(
                "train/critic_loss",
                np.mean(critic_losses),
            )


def save_csv(df, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(path, index=False)

    print("SAVED:", path, flush=True)


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
        "--configs",
        nargs="+",
        choices=list(CONFIGS),
        default=list(CONFIGS),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    if args.timesteps < 6000:
        raise ValueError(
            "Need at least 6000 timesteps."
        )

    panels, feature_map = v2.build_panels()

    panel = panels["real_dynamic"]
    features = feature_map["real_dynamic"]

    train_df, val_df, test_df = (
        v2.split_panel(panel)
    )

    # Model selection strictly uses 2022.
    del test_df

    results = []

    for name in args.configs:
        config = CONFIGS[name]

        out = (
            args.out
            / f"seed_{args.seed}"
            / "real_dynamic"
            / name
        )

        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "\n===== RUN",
            name,
            config,
            "=====",
            flush=True,
        )

        mean, std = v3.fit_normalizer(
            train_df,
            features,
            out,
        )

        # Reuse V3's training pipeline, but
        # replace its DDPG class with V4DDPG.
        V4DDPG.actor_lr_v4 = config["actor_lr"]
        V4DDPG.critic_lr_v4 = config["critic_lr"]
        V4DDPG.action_penalty_v4 = config[
            "action_penalty"
        ]

        original_ddpg = v3.DDPG
        v3.DDPG = V4DDPG

        try:
            model, monitor, vec_env = v3.train(
                train_df=train_df,
                features=features,
                use_kg=True,
                seed=args.seed,
                timesteps=args.timesteps,
                learning_rate=config["critic_lr"],
                reward_scale=10.0,
                mean=mean,
                std=std,
                check_freq=args.check_freq,
            )
        finally:
            v3.DDPG = original_ddpg

        model.save(
            str(out / "ddpg_v4")
        )

        saturation = pd.DataFrame(
            monitor.rows
        )

        gradients = pd.DataFrame(
            model.v4_diagnostics
        )

        save_csv(
            saturation,
            out / "saturation_training.csv",
        )

        save_csv(
            gradients,
            out / "critic_actor_gradients.csv",
        )

        validation, allocation = (
            v3.evaluate_validation(
                model,
                val_df,
                features,
                mean,
                std,
            )
        )

        save_csv(
            allocation,
            out / "validation_allocations.csv",
        )

        last = saturation.iloc[-1]

        result = {
            "seed": args.seed,
            "config": name,
            "timesteps": args.timesteps,
            "reward_scale": 10.0,
            **config,
            **validation,
            "train_sat99": float(
                last["saturation_099"]
            ),
            "train_sat999": float(
                last["saturation_0999"]
            ),
            "train_pre_tanh_abs": float(
                last["pre_tanh_mean_abs"]
            ),
            "train_action_std": float(
                last["action_std_mean"]
            ),
        }

        results.append(result)

        save_csv(
            pd.DataFrame([result]),
            out / "summary.csv",
        )

        with (
            out / "config.json"
        ).open("w") as f:
            json.dump(
                {
                    **config,
                    "seed": args.seed,
                    "timesteps": args.timesteps,
                    "reward_scale": 10.0,
                    "normalizer":
                        "Train-only pooled normalization",
                    "validation": "2022",
                    "test_evaluated": False,
                },
                f,
                indent=2,
            )

        print(
            "FINISHED:",
            name,
            result,
            flush=True,
        )

        vec_env.close()
        del model

    summary = pd.DataFrame(results)

    save_csv(
        summary,
        args.out
        / f"seed_{args.seed}"
        / "v4_comparison.csv",
    )

    print(
        "\n===== V4 VALIDATION COMPARISON =====",
        flush=True,
    )

    print(
        summary.to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
