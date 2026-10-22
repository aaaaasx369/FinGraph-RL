#!/usr/bin/env python3

from pathlib import Path

from unified_event_memory.finrl_x.diagnostics import (
    train_ddpg_v4 as v4
)

v4.CONFIGS.clear()

v4.CONFIGS.update({
    "v5a": {
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "action_penalty": 0.001,
    },
    "v5b": {
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "action_penalty": 0.005,
    },
    "v5c": {
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "action_penalty": 0.010,
    },
    "v5d": {
        "actor_lr": 5e-5,
        "critic_lr": 1e-4,
        "action_penalty": 0.020,
    },
})

v4.DEFAULT_OUT = Path(
    "/mnt/sda1/doralin/FinRL-Trading/"
    "results/ddpg_v5_regularization_sweep"
)

if __name__ == "__main__":
    v4.main()
