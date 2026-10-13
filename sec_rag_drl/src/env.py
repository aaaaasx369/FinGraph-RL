from __future__ import annotations

from typing import Literal

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .data import PortfolioData
from .gating import (
    soft_confidence_gate,
    apply_negative_veto,
)


Mode = Literal[
    "technical",
    "fusion",
    "gate_only",
    "hybrid_soft",
    "hybrid_veto",

    "fusion_kg",
    "hybrid_soft_kg",
    "hybrid_veto_kg",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) - np.max(x)
    e = np.exp(z)

    return (
        e / np.sum(e)
    ).astype(np.float64)


class PortfolioEnv(gym.Env):
    """
    Long-only, fully invested portfolio environment
    with optional CASH.

    Timeline
    --------
    observation at date t
        -> model proposes target weights
        -> optional RAG gate modifies target weights
        -> pay transaction cost
        -> realize t -> t+1 close-to-close returns

    Observation variants
    --------------------

    technical:
        technical features only

    fusion:
        technical
        + SEC-RAG state

    fusion_kg:
        technical
        + SEC-RAG state
        + FinReflectKG state

    hybrid_soft:
        fusion observation
        + SEC-RAG soft gate after policy action

    hybrid_soft_kg:
        fusion_kg observation
        + SEC-RAG soft gate after policy action

    hybrid_veto:
        fusion observation
        + SEC-RAG soft gate
        + negative veto

    hybrid_veto_kg:
        fusion_kg observation
        + SEC-RAG soft gate
        + negative veto
    """

    metadata = {
        "render_modes": []
    }


    def __init__(
        self,
        data: PortfolioData,
        cfg: dict,
        mode: Mode = "technical",
    ):
        super().__init__()

        self.data = data
        self.cfg = cfg
        self.mode = mode


        # ----------------------------------------------------
        # Assets
        # ----------------------------------------------------

        self.n_stocks = len(
            data.tickers
        )

        self.include_cash = bool(
            cfg["portfolio"].get(
                "include_cash_asset",
                True,
            )
        )

        if not self.include_cash:
            raise ValueError(
                "This reference implementation "
                "requires include_cash_asset=true "
                "because vetoed weight is moved to cash."
            )

        self.n_assets = (
            self.n_stocks + 1
        )


        # ----------------------------------------------------
        # Mode behavior
        # ----------------------------------------------------

        self.use_rag_state = mode in {
            "fusion",
            "hybrid_soft",
            "hybrid_veto",
            "fusion_kg",
            "hybrid_soft_kg",
            "hybrid_veto_kg",
        }

        self.use_kg_state = mode in {
            "fusion_kg",
            "hybrid_soft_kg",
            "hybrid_veto_kg",
        }

        self.use_soft_gate = mode in {
            "gate_only",
            "hybrid_soft",
            "hybrid_veto",
            "hybrid_soft_kg",
            "hybrid_veto_kg",
        }

        self.use_veto = mode in {
            "hybrid_veto",
            "hybrid_veto_kg",
        }


        # ----------------------------------------------------
        # Observation dimension
        # ----------------------------------------------------

        per_stock_features = (
            data.tech.shape[2]
        )

        if self.use_rag_state:
            per_stock_features += (
                data.rag_state.shape[2]
            )

        if self.use_kg_state:
            per_stock_features += (
                data.kg_state.shape[2]
            )

        obs_dim = (
            self.n_stocks
            * per_stock_features
            + self.n_assets
        )


        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # Action space
        # ----------------------------------------------------

        # Continuous logits -> softmax -> simplex weights.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32,
        )


        # ----------------------------------------------------
        # Portfolio config
        # ----------------------------------------------------

        self.initial_cash = float(
            cfg["portfolio"][
                "initial_cash"
            ]
        )

        self.tc_rate = (
            float(
                cfg["portfolio"][
                    "transaction_cost_bps"
                ]
            )
            / 10000.0
        )

        self.reward_scale = float(
            cfg["portfolio"].get(
                "reward_scale",
                100.0,
            )
        )

        self.gate_cfg = cfg["gate"]


        # ----------------------------------------------------
        # Runtime state
        # ----------------------------------------------------

        self.t = 0

        self.portfolio_value = (
            self.initial_cash
        )

        self.current_weights = np.zeros(
            self.n_assets,
            dtype=np.float64,
        )

        # Start fully in cash.
        self.current_weights[-1] = 1.0

        self.history = []


    # ========================================================
    # Observation
    # ========================================================

    def _get_obs(self):

        pieces = [
            self.data.tech[
                self.t
            ].reshape(-1)
        ]


        if self.use_rag_state:

            pieces.append(
                self.data.rag_state[
                    self.t
                ].reshape(-1)
            )


        if self.use_kg_state:

            pieces.append(
                self.data.kg_state[
                    self.t
                ].reshape(-1)
            )


        # Current drifted portfolio weights
        # are part of state.
        pieces.append(
            self.current_weights.astype(
                np.float32
            )
        )


        return np.concatenate(
            pieces
        ).astype(np.float32)


    # ========================================================
    # Reset
    # ========================================================

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):

        super().reset(
            seed=seed
        )

        self.t = 0

        self.portfolio_value = (
            self.initial_cash
        )

        self.current_weights = np.zeros(
            self.n_assets,
            dtype=np.float64,
        )

        self.current_weights[-1] = 1.0

        self.history = []

        return (
            self._get_obs(),
            {},
        )


    # ========================================================
    # RAG integration layer
    # ========================================================

    def _apply_integration(
        self,
        raw_target: np.ndarray,
    ):

        stock_w = raw_target[
            :self.n_stocks
        ].copy()

        cash_w = float(
            raw_target[-1]
        )


        # IMPORTANT:
        # gates still use only the original
        # raw SEC-RAG signals.
        #
        # KG affects policy observation,
        # not the gate itself.
        rag = self.data.rag_raw[
            self.t
        ]


        veto_score = np.zeros(
            self.n_stocks,
            dtype=float,
        )


        if self.use_soft_gate:

            stock_w, cash_w = (
                soft_confidence_gate(
                    stock_w,
                    cash_w,
                    rag,
                    self.gate_cfg,
                )
            )


        if self.use_veto:

            (
                stock_w,
                cash_w,
                veto_score,
            ) = apply_negative_veto(
                stock_w,
                cash_w,
                rag,
                self.gate_cfg,
            )


        final = np.concatenate(
            [
                stock_w,
                [cash_w],
            ]
        )


        # Numerical safety.
        final = np.maximum(
            final,
            0.0,
        )


        total = final.sum()

        if total <= 0:

            final = np.zeros(
                self.n_assets,
                dtype=np.float64,
            )

            final[-1] = 1.0

        else:

            final = (
                final / total
            )


        return (
            final,
            veto_score,
        )


    # ========================================================
    # Step
    # ========================================================

    def step(
        self,
        action,
    ):

        action = np.asarray(
            action,
            dtype=np.float64,
        )


        # ----------------------------------------------------
        # Policy logits -> raw portfolio target
        # ----------------------------------------------------

        raw_target = _softmax(
            action
        )


        # ----------------------------------------------------
        # Optional SEC-RAG post-allocation integration
        # ----------------------------------------------------

        (
            target,
            veto_score,
        ) = self._apply_integration(
            raw_target
        )


        # ----------------------------------------------------
        # Transaction cost
        # ----------------------------------------------------

        # Turnover measured from current drifted holdings.
        turnover = (
            0.5
            * np.abs(
                target
                - self.current_weights
            ).sum()
        )

        tx_cost = (
            self.tc_rate
            * turnover
        )


        # ----------------------------------------------------
        # Market return t -> t+1
        # ----------------------------------------------------

        p0 = self.data.close[
            self.t
        ]

        p1 = self.data.close[
            self.t + 1
        ]


        stock_ret = (
            p1 / p0
            - 1.0
        )


        gross_ret = float(
            np.dot(
                target[
                    :self.n_stocks
                ],
                stock_ret,
            )
        )


        net_ret = (
            gross_ret
            - tx_cost
        )


        # ----------------------------------------------------
        # Portfolio value
        # ----------------------------------------------------

        self.portfolio_value *= max(
            1.0 + net_ret,
            1e-12,
        )


        # ----------------------------------------------------
        # Drift holdings after realized returns
        # ----------------------------------------------------

        end_values = np.concatenate(
            [
                target[
                    :self.n_stocks
                ]
                * (
                    1.0
                    + stock_ret
                ),
                [
                    target[-1]
                ],
            ]
        )


        denom = end_values.sum()


        if denom <= 0:

            drifted = np.zeros(
                self.n_assets,
                dtype=np.float64,
            )

            drifted[-1] = 1.0

        else:

            drifted = (
                end_values
                / denom
            )


        self.current_weights = drifted


        # ----------------------------------------------------
        # Reward
        # ----------------------------------------------------

        reward = (
            np.log(
                max(
                    1.0 + net_ret,
                    1e-12,
                )
            )
            * self.reward_scale
        )


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        info = {
            "date":
                str(
                    self.data.dates[
                        self.t
                    ].date()
                ),

            "portfolio_value":
                float(
                    self.portfolio_value
                ),

            "gross_return":
                gross_ret,

            "net_return":
                net_ret,

            "turnover":
                float(
                    turnover
                ),

            "transaction_cost_return":
                float(
                    tx_cost
                ),

            "raw_target_weights":
                raw_target.astype(
                    np.float32
                ),

            "target_weights":
                target.astype(
                    np.float32
                ),

            "veto_score":
                veto_score.astype(
                    np.float32
                ),
        }


        self.history.append(
            info
        )


        # ----------------------------------------------------
        # Advance time
        # ----------------------------------------------------

        self.t += 1


        terminated = (
            self.t
            >= len(
                self.data.dates
            ) - 1
        )

        truncated = False


        if terminated:

            obs = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )

        else:

            obs = self._get_obs()


        return (
            obs,
            float(reward),
            terminated,
            truncated,
            info,
        )
