from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO, A2C, DDPG, SAC
from stable_baselines3.common.utils import set_random_seed

from .env import PortfolioEnv


ALGOS = {
    "PPO": PPO,
    "A2C": A2C,
    "DDPG": DDPG,
    "SAC": SAC,
}


def build_model(agent: str, env, cfg: dict, seed: int):
    agent = agent.upper()
    if agent not in ALGOS:
        raise ValueError(f"Unknown agent {agent}. Choose from {list(ALGOS)}")

    common = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=float(cfg["rl"]["learning_rate"]),
        gamma=float(cfg["rl"]["gamma"]),
        seed=int(seed),
        verbose=1,
        device=cfg["rl"].get("device", "auto"),
    )

    if agent == "PPO":
        c = cfg["rl"]["ppo"]
        return PPO(
            **common,
            n_steps=int(c["n_steps"]),
            batch_size=int(c["batch_size"]),
            ent_coef=float(c.get("ent_coef", 0.0)),
        )
    if agent == "A2C":
        c = cfg["rl"]["a2c"]
        return A2C(
            **common,
            n_steps=int(c["n_steps"]),
            ent_coef=float(c.get("ent_coef", 0.0)),
        )
    if agent == "DDPG":
        c = cfg["rl"]["ddpg"]
        return DDPG(
            **common,
            buffer_size=int(c["buffer_size"]),
            learning_starts=int(c["learning_starts"]),
            batch_size=int(c["batch_size"]),
        )
    if agent == "SAC":
        c = cfg["rl"]["sac"]
        return SAC(
            **common,
            buffer_size=int(c["buffer_size"]),
            learning_starts=int(c["learning_starts"]),
            batch_size=int(c["batch_size"]),
        )


def train_model(agent: str, train_data, cfg: dict, mode: str, seed: int, out_dir: Path):
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = PortfolioEnv(train_data, cfg, mode=mode)
    model = build_model(agent, env, cfg, seed)
    model.learn(total_timesteps=int(cfg["rl"]["total_timesteps"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model"
    model.save(str(model_path))
    return model


def load_model(agent: str, model_path: str, env=None, device="auto"):
    cls = ALGOS[agent.upper()]
    return cls.load(model_path, env=env, device=device)
