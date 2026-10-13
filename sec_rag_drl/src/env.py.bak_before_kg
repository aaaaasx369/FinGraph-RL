from __future__ import annotations
from typing import Literal
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data import PortfolioData
from .gating import soft_confidence_gate, apply_negative_veto


Mode = Literal[
    "technical",
    "fusion",
    "gate_only",
    "hybrid_soft",
    "hybrid_veto",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - np.max(x)
    e = np.exp(z)
    return (e / np.sum(e)).astype(np.float64)


class PortfolioEnv(gym.Env):
    """
    Long-only, fully invested portfolio environment with optional CASH.

    Timeline:
      observation at date t
      -> model proposes target weights
      -> optional RAG gate modifies target weights
      -> pay transaction cost
      -> realize t -> t+1 close-to-close returns
    """
    metadata = {"render_modes": []}

    def __init__(self, data: PortfolioData, cfg: dict, mode: Mode = "technical"):
        super().__init__()
        self.data = data
        self.cfg = cfg
        self.mode = mode

        self.n_stocks = len(data.tickers)
        self.include_cash = bool(cfg["portfolio"].get("include_cash_asset", True))
        if not self.include_cash:
            raise ValueError(
                "This reference implementation keeps include_cash_asset=true "
                "because vetoed weight is deliberately moved to cash."
            )

        self.n_assets = self.n_stocks + 1
        self.use_rag_state = mode in {"fusion", "hybrid_soft", "hybrid_veto"}
        self.use_soft_gate = mode in {"gate_only", "hybrid_soft", "hybrid_veto"}
        self.use_veto = mode == "hybrid_veto"

        per_stock_features = data.tech.shape[2] + (data.rag_state.shape[2] if self.use_rag_state else 0)
        obs_dim = self.n_stocks * per_stock_features + self.n_assets

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Symmetric action space as recommended for continuous control.
        # Actions are logits transformed to simplex weights with softmax.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.initial_cash = float(cfg["portfolio"]["initial_cash"])
        self.tc_rate = float(cfg["portfolio"]["transaction_cost_bps"]) / 10000.0
        self.reward_scale = float(cfg["portfolio"].get("reward_scale", 100.0))
        self.gate_cfg = cfg["gate"]

        self.t = 0
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.n_assets, dtype=np.float64)
        self.current_weights[-1] = 1.0  # start fully in cash
        self.history = []

    def _get_obs(self):
        pieces = [self.data.tech[self.t].reshape(-1)]
        if self.use_rag_state:
            pieces.append(self.data.rag_state[self.t].reshape(-1))
        pieces.append(self.current_weights.astype(np.float32))
        return np.concatenate(pieces).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.portfolio_value = self.initial_cash
        self.current_weights = np.zeros(self.n_assets, dtype=np.float64)
        self.current_weights[-1] = 1.0
        self.history = []
        return self._get_obs(), {}

    def _apply_integration(self, raw_target: np.ndarray):
        stock_w = raw_target[:self.n_stocks].copy()
        cash_w = float(raw_target[-1])
        rag = self.data.rag_raw[self.t]

        veto_score = np.zeros(self.n_stocks, dtype=float)

        if self.use_soft_gate:
            stock_w, cash_w = soft_confidence_gate(
                stock_w, cash_w, rag, self.gate_cfg
            )

        if self.use_veto:
            stock_w, cash_w, veto_score = apply_negative_veto(
                stock_w, cash_w, rag, self.gate_cfg
            )

        final = np.concatenate([stock_w, [cash_w]])
        final = np.maximum(final, 0.0)
        final = final / final.sum()
        return final, veto_score

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        raw_target = _softmax(action)
        target, veto_score = self._apply_integration(raw_target)

        # Turnover uses current drifted holdings -> new target.
        turnover = 0.5 * np.abs(target - self.current_weights).sum()
        tx_cost = self.tc_rate * turnover

        p0 = self.data.close[self.t]
        p1 = self.data.close[self.t + 1]
        stock_ret = p1 / p0 - 1.0

        gross_ret = float(np.dot(target[:self.n_stocks], stock_ret))
        net_ret = gross_ret - tx_cost

        old_value = self.portfolio_value
        self.portfolio_value *= max(1.0 + net_ret, 1e-12)

        # Holdings drift after returns; cash return is zero.
        end_values = np.concatenate([
            target[:self.n_stocks] * (1.0 + stock_ret),
            [target[-1]],
        ])
        denom = end_values.sum()
        if denom <= 0:
            drifted = np.zeros(self.n_assets)
            drifted[-1] = 1.0
        else:
            drifted = end_values / denom
        self.current_weights = drifted

        reward = np.log(max(1.0 + net_ret, 1e-12)) * self.reward_scale

        info = {
            "date": str(self.data.dates[self.t].date()),
            "portfolio_value": float(self.portfolio_value),
            "gross_return": gross_ret,
            "net_return": net_ret,
            "turnover": float(turnover),
            "transaction_cost_return": float(tx_cost),
            "raw_target_weights": raw_target.astype(np.float32),
            "target_weights": target.astype(np.float32),
            "veto_score": veto_score.astype(np.float32),
        }
        self.history.append(info)

        self.t += 1
        terminated = self.t >= len(self.data.dates) - 1
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        return obs, float(reward), terminated, truncated, info
