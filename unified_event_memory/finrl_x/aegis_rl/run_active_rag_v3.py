#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X Active RAG V3

Architecture
------------
Portfolio observation + Dynamic KGT
       |
       v
Contextual Conditional Retrieval Policy
       |
       +--> STOP
       +--> KG
       +--> SEC
       +--> BOTH
       |
       v
Conditional ticker / relation selection
       |
       v
Ollama query generation + KG / SEC retrieval
       |
       v
Updated evidence and retrieval context
       |
       v
Optional next retrieval step
       |
       v
A2C / PPO / DDPG Portfolio Allocation
       |
       v
Portfolio Reward - Retrieval Cost
       |
       v
Online Policy Gradient Router Update

Additional evaluation
---------------------
Frozen-policy counterfactual replay:
    STOP / KG / SEC / BOTH

Important:
    Counterfactual replay is an intervention-based
    sensitivity analysis, NOT an unbiased causal
    estimate of retrieval value.

This runner extends the already-tested V2 runner.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import A2C, PPO, DDPG


# --------------------------------------------------
# 1. IMPORT EXISTING TESTED V2
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(
    0,
    str(ROOT),
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v2 as v2
)


SOURCES = tuple(v2.SOURCES)
TICKERS = tuple(v2.TICKERS)
RELATIONS = tuple(v2.RELATIONS)

# Keep the existing 14-dimensional router features.
BASE_STATE_DIM = 14

# Context extracted from current RL observation.
OBS_CONTEXT_DIM = 256

# Explicit summaries of retrieved semantic evidence.
EVIDENCE_CONTEXT_DIM = 32

STATE_DIM = (
    BASE_STATE_DIM
    + OBS_CONTEXT_DIM
    + EVIDENCE_CONTEXT_DIM
)


def log(*args):
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        *args,
        flush=True,
    )


def numeric_vector(x):
    array = np.asarray(
        x,
        dtype=np.float32,
    ).reshape(-1)

    return np.nan_to_num(
        array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def resize_context(values, length):
    """
    Convert variable-sized observation vectors
    to a fixed-sized contextual representation.

    All observation positions contribute through
    deterministic piecewise pooling.

    No learned embedding is fitted on test data.
    """
    x = numeric_vector(values)

    if len(x) == 0:
        return np.zeros(
            length,
            dtype=np.float32,
        )

    if len(x) == length:
        return x.copy()

    if len(x) < length:
        output = np.zeros(
            length,
            dtype=np.float32,
        )
        output[:len(x)] = x
        return output

    edges = np.linspace(
        0,
        len(x),
        length + 1,
        dtype=int,
    )

    output = np.zeros(
        length,
        dtype=np.float32,
    )

    for i in range(length):
        a = edges[i]
        b = max(a + 1, edges[i + 1])

        output[i] = float(
            np.mean(x[a:b])
        )

    return output


def summarize_evidence(evidence):
    """
    Compact summaries of KG and SEC embeddings.

    Input:
        two semantic evidence vectors

    Output:
        32-dimensional context

    The exact evidence is still passed to the
    original portfolio observation.
    """
    array = np.asarray(
        evidence,
        dtype=np.float32,
    )

    if array.ndim != 2 or array.shape[0] != 2:
        return np.zeros(
            EVIDENCE_CONTEXT_DIM,
            dtype=np.float32,
        )

    parts = []

    for source_vector in array:
        x = numeric_vector(source_vector)

        parts.append(
            resize_context(x, 16)
        )

    return np.concatenate(
        parts
    ).astype(np.float32)


# --------------------------------------------------
# 2. CONDITIONAL RETRIEVAL POLICY
# --------------------------------------------------

class ConditionalRetrievalPolicy(nn.Module):
    """
    P(source | context)

    P(ticker | context, source)

    P(relation | context, source, ticker)

    Unlike independent heads, ticker/relation
    explicitly depend on preceding decisions.
    """

    def __init__(
        self,
        input_dim=STATE_DIM,
        hidden_dim=128,
    ):
        super().__init__()

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        self.source_head = nn.Linear(
            hidden_dim,
            len(SOURCES),
        )

        self.ticker_head = nn.Sequential(
            nn.Linear(
                hidden_dim + len(SOURCES),
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                len(TICKERS),
            ),
        )

        self.relation_head = nn.Sequential(
            nn.Linear(
                hidden_dim
                + len(SOURCES)
                + len(TICKERS),
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                len(RELATIONS),
            ),
        )

    def encode(self, state):
        return self.encoder(state)

    def source_logits(self, h):
        return self.source_head(h)

    def ticker_logits(self, h, source_index):
        source_onehot = F.one_hot(
            source_index,
            num_classes=len(SOURCES),
        ).float()

        return self.ticker_head(
            torch.cat(
                [h, source_onehot],
                dim=-1,
            )
        )

    def relation_logits(
        self,
        h,
        source_index,
        ticker_index,
    ):
        source_onehot = F.one_hot(
            source_index,
            num_classes=len(SOURCES),
        ).float()

        ticker_onehot = F.one_hot(
            ticker_index,
            num_classes=len(TICKERS),
        ).float()

        return self.relation_head(
            torch.cat(
                [
                    h,
                    source_onehot,
                    ticker_onehot,
                ],
                dim=-1,
            )
        )


# --------------------------------------------------
# 3. LEARNED CONTEXTUAL ROUTER
# --------------------------------------------------

class ContextualRouter:
    """
    Online categorical policy gradient.

    Each algorithm/seed has its own router.

    During training:
        sample categorical actions

    During evaluation:
        use deterministic conditional argmax

    Log action probabilities and entropy.
    """

    def __init__(
        self,
        state_dim,
        lr,
        entropy_weight,
        training=True,
    ):
        # V2 passes state_dim=14.
        # V3 uses additional context instead.
        self.input_dim = STATE_DIM

        self.net = ConditionalRetrievalPolicy(
            self.input_dim
        )

        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=lr,
        )

        self.entropy_weight = float(
            entropy_weight
        )

        self.training = bool(training)

        self.pending = []

        self.running_baseline = 0.0
        self.update_count = 0

        self.action_log = []

        # Last actual conditional distribution.
        self.last_distribution = {}

    def select(self, features):
        x = numeric_vector(features)

        if len(x) != self.input_dim:
            raise RuntimeError(
                "Router context dimension mismatch: "
                f"got={len(x)}, "
                f"expected={self.input_dim}"
            )

        state = torch.from_numpy(
            x
        ).unsqueeze(0)

        h = self.net.encode(state)

        src_logits = self.net.source_logits(h)

        src_dist = (
            torch.distributions.Categorical(
                logits=src_logits
            )
        )

        if self.training:
            src = src_dist.sample()
        else:
            src = torch.argmax(
                src_logits,
                dim=-1,
            )

        source = SOURCES[
            int(src.item())
        ]

        log_probs = [
            src_dist.log_prob(src)
        ]

        entropies = [
            src_dist.entropy()
        ]

        source_probs = (
            src_dist.probs
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        # STOP has no meaningful ticker/relation.
        if source == "STOP":
            ticker = TICKERS[0]
            relation = RELATIONS[0]

            ticker_probs = None
            relation_probs = None

        else:
            ticker_logits = (
                self.net.ticker_logits(
                    h,
                    src,
                )
            )

            ticker_dist = (
                torch.distributions.Categorical(
                    logits=ticker_logits
                )
            )

            if self.training:
                tick = ticker_dist.sample()
            else:
                tick = torch.argmax(
                    ticker_logits,
                    dim=-1,
                )

            ticker = TICKERS[
                int(tick.item())
            ]

            log_probs.append(
                ticker_dist.log_prob(tick)
            )

            entropies.append(
                ticker_dist.entropy()
            )

            ticker_probs = (
                ticker_dist.probs
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            relation_logits = (
                self.net.relation_logits(
                    h,
                    src,
                    tick,
                )
            )

            relation_dist = (
                torch.distributions.Categorical(
                    logits=relation_logits
                )
            )

            if self.training:
                rel = relation_dist.sample()
            else:
                rel = torch.argmax(
                    relation_logits,
                    dim=-1,
                )

            relation = RELATIONS[
                int(rel.item())
            ]

            log_probs.append(
                relation_dist.log_prob(rel)
            )

            entropies.append(
                relation_dist.entropy()
            )

            relation_probs = (
                relation_dist.probs
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

        if self.training:
            self.pending.append(
                (
                    torch.stack(
                        log_probs
                    ).sum(),
                    torch.stack(
                        entropies
                    ).mean(),
                )
            )

        def entropy(probs):
            if probs is None:
                return None

            probs = np.clip(
                probs,
                1e-12,
                1.0,
            )

            return float(
                -np.sum(
                    probs * np.log(probs)
                )
            )

        self.last_distribution = {
            "source": {
                key: float(p)
                for key, p in zip(
                    SOURCES,
                    source_probs,
                )
            },
            "ticker": (
                None
                if ticker_probs is None
                else {
                    key: float(p)
                    for key, p in zip(
                        TICKERS,
                        ticker_probs,
                    )
                }
            ),
            "relation": (
                None
                if relation_probs is None
                else {
                    key: float(p)
                    for key, p in zip(
                        RELATIONS,
                        relation_probs,
                    )
                }
            ),
            "source_entropy": entropy(
                source_probs
            ),
            "ticker_entropy": entropy(
                ticker_probs
            ),
            "relation_entropy": entropy(
                relation_probs
            ),
        }

        self.action_log.append({
            "source": source,
            "ticker": ticker,
            "relation": relation,
            **self.last_distribution,
        })

        return (
            source,
            ticker,
            relation,
        )

    def update(self, reward):
        if not self.training:
            self.pending.clear()
            return

        if not self.pending:
            return

        reward = float(reward)

        advantage = (
            reward
            - self.running_baseline
        )

        # Match the existing V2 reward scaling.
        advantage = float(
            np.clip(
                advantage * 100.0,
                -5.0,
                5.0,
            )
        )

        logprob = torch.stack([
            x[0]
            for x in self.pending
        ]).sum()

        entropy = torch.stack([
            x[1]
            for x in self.pending
        ]).mean()

        loss = (
            -advantage * logprob
            -self.entropy_weight * entropy
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.net.parameters(),
            1.0,
        )

        self.optimizer.step()

        self.running_baseline = (
            0.98 * self.running_baseline
            + 0.02 * reward
        )

        self.update_count += 1

        self.pending.clear()

    def save(self, path):
        torch.save({
            "network": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "input_dim": self.input_dim,
            "running_baseline":
                self.running_baseline,
            "update_count":
                self.update_count,
        }, path)

    def load(self, path):
        data = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )

        if data.get(
            "input_dim"
        ) != self.input_dim:
            raise RuntimeError(
                "Incompatible V3 router checkpoint"
            )

        self.net.load_state_dict(
            data["network"]
        )

        self.optimizer.load_state_dict(
            data["optimizer"]
        )

        self.running_baseline = (
            data["running_baseline"]
        )

        self.update_count = (
            data["update_count"]
        )


# --------------------------------------------------
# 4. CONTEXTUAL ACTIVE RAG ENVIRONMENT
# --------------------------------------------------

class ContextualActiveRAGEnv(
    v2.ActiveRAGEnvironment
):
    """
    Uses original portfolio environment
    and original KG/SEC retrieval functions.

    Router now observes:
        - V2 base state 14D
        - pooled RL observation 256D
        - explicit evidence summary 32D

    This includes information from the current
    portfolio observation and evidence.

    Future returns must NOT enter _obs().
    """

    def _router_state(self):
        base = numeric_vector(
            super()._router_state()
        )

        if len(base) != BASE_STATE_DIM:
            raise RuntimeError(
                f"Expected 14D base state; "
                f"got {len(base)}"
            )

        observation = numeric_vector(
            self._obs()
        )

        context = resize_context(
            observation,
            OBS_CONTEXT_DIM,
        )

        evidence = summarize_evidence(
            self.evidence
        )

        state = np.concatenate([
            base,
            context,
            evidence,
        ])

        if len(state) != STATE_DIM:
            raise RuntimeError(
                "Invalid V3 contextual state"
            )

        return state.astype(np.float32)


# --------------------------------------------------
# 5. FORCED SOURCE WRAPPER
# --------------------------------------------------

class ForcedSourceRouter:
    """
    Freeze a trained router and intervene only
    on its source decision.

    Ticker and relation are obtained from
    the trained router, conditional on the
    actual counterfactual observation.

    IMPORTANT:
    This is an intervention rollout.
    It does not identify unbiased causal effects.
    """

    def __init__(
        self,
        learned_router,
        forced_source,
    ):
        self.learned_router = (
            learned_router
        )

        self.forced_source = (
            forced_source
        )

        self.pending = []
        self.action_log = []

    def select(self, features):
        source, ticker, relation = (
            self.learned_router.select(
                features
            )
        )

        forced = self.forced_source

        self.action_log.append({
            "model_source": source,
            "forced_source": forced,
            "ticker": ticker,
            "relation": relation,
        })

        return (
            forced,
            ticker,
            relation,
        )

    def update(self, reward):
        self.pending.clear()


# --------------------------------------------------
# 6. POLICY REPLAY
# --------------------------------------------------

_ORIGINAL_EVALUATE = (
    v2.evaluate_model
)

COUNTERFACTUAL_ENABLED = False
COUNTERFACTUAL_ROOT = None

COUNTERFACTUAL_RECORDS = []
REPLAY_INDEX = 0


def evaluate_with_counterfactual(
    model,
    env,
):
    """
    First evaluate learned active policy.

    Then replay:
        STOP
        KG
        SEC
        BOTH

    All runs use:
        - same trained portfolio actor
        - same initial capital
        - same market window
        - same transaction-cost model

    Portfolio paths can differ due to retrieved
    evidence. This is intended.
    """
    global REPLAY_INDEX

    metrics, history = (
        _ORIGINAL_EVALUATE(
            model,
            env,
        )
    )

    if not COUNTERFACTUAL_ENABLED:
        return metrics, history

    if getattr(
        env,
        "mode",
        None,
    ) != "ACTIVE":
        return metrics, history

    REPLAY_INDEX += 1

    original_router = env.router

    original_training = (
        original_router.training
    )

    original_router.training = False

    split_name = str(
        getattr(
            env,
            "split",
            "unknown",
        )
    )

    report_rows = [{
        "replay_index": REPLAY_INDEX,
        "split": split_name,
        "intervention": "LEARNED_ACTIVE",
        **metrics,
    }]

    try:
        for forced_source in (
            "STOP",
            "KG",
            "SEC",
            "BOTH",
        ):
            log(
                "[COUNTERFACTUAL]",
                split_name,
                forced_source,
            )

            forced_router = ForcedSourceRouter(
                original_router,
                forced_source,
            )

            env.router = forced_router

            cf_metrics, cf_history = (
                _ORIGINAL_EVALUATE(
                    model,
                    env,
                )
            )

            row = {
                "replay_index": REPLAY_INDEX,
                "split": split_name,
                "intervention": forced_source,
                **cf_metrics,
            }

            report_rows.append(row)

            # Save replay traces separately.
            if COUNTERFACTUAL_ROOT is not None:
                COUNTERFACTUAL_ROOT.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                cf_history.to_csv(
                    COUNTERFACTUAL_ROOT
                    / (
                        f"replay_{REPLAY_INDEX}_"
                        f"{split_name}_"
                        f"{forced_source}.csv"
                    ),
                    index=False,
                )

            log(
                "[COUNTERFACTUAL RESULT]",
                forced_source,
                "return:",
                round(
                    cf_metrics["total_return"],
                    6,
                ),
                "query_rate:",
                round(
                    cf_metrics["query_rate"],
                    4,
                ),
                "KG:",
                cf_metrics["KG_hits"],
                "SEC:",
                cf_metrics["SEC_hits"],
            )

    finally:
        env.router = original_router
        original_router.training = (
            original_training
        )

    frame = pd.DataFrame(
        report_rows
    )

    stop_rows = frame[
        frame["intervention"] == "STOP"
    ]

    if not stop_rows.empty:
        stop_return = float(
            stop_rows.iloc[0][
                "total_return"
            ]
        )

        frame[
            "return_difference_vs_stop"
        ] = (
            frame["total_return"]
            - stop_return
        )

    COUNTERFACTUAL_RECORDS.extend(
        report_rows
    )

    if COUNTERFACTUAL_ROOT is not None:
        frame.to_csv(
            COUNTERFACTUAL_ROOT
            / (
                f"replay_{REPLAY_INDEX}_"
                f"{split_name}_summary.csv"
            ),
            index=False,
        )

    print(
        "\nCOUNTERFACTUAL REPLAY SUMMARY",
        flush=True,
    )

    print(
        frame[
            [
                "intervention",
                "total_return",
                "sharpe",
                "max_drawdown",
                "query_rate",
                "KG_hits",
                "SEC_hits",
            ]
        ].to_string(
            index=False
        ),
        flush=True,
    )

    return metrics, history


# --------------------------------------------------
# 7. OPTIONAL ROUTER PROBABILITY DIAGNOSTIC
# --------------------------------------------------

def inspect_router(checkpoint):
    data = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    dim = int(
        data["input_dim"]
    )

    model = ConditionalRetrievalPolicy(
        input_dim=dim
    )

    model.load_state_dict(
        data["network"]
    )

    model.eval()

    # This is only a reference state.
    # It is NOT a historical observation.
    x = torch.zeros(
        1,
        dim,
    )

    with torch.no_grad():
        h = model.encode(x)

        source_prob = F.softmax(
            model.source_logits(h),
            dim=-1,
        ).flatten()

        src = torch.argmax(
            source_prob
        ).reshape(1)

        ticker_prob = F.softmax(
            model.ticker_logits(
                h,
                src,
            ),
            dim=-1,
        ).flatten()

        tick = torch.argmax(
            ticker_prob
        ).reshape(1)

        relation_prob = F.softmax(
            model.relation_logits(
                h,
                src,
                tick,
            ),
            dim=-1,
        ).flatten()

    print(
        "Router updates:",
        data["update_count"],
    )

    for name, labels, prob in [
        (
            "SOURCE",
            SOURCES,
            source_prob,
        ),
        (
            "TICKER GIVEN SOURCE",
            TICKERS,
            ticker_prob,
        ),
        (
            "RELATION GIVEN SOURCE/TICKER",
            RELATIONS,
            relation_prob,
        ),
    ]:
        entropy = float(
            -torch.sum(
                prob
                * torch.log(
                    prob + 1e-12
                )
            )
        )

        print(
            f"\n===== {name} ====="
        )

        print(
            "Entropy:",
            round(entropy, 6),
        )

        for label, p in zip(
            labels,
            prob.tolist(),
        ):
            print(
                f"{label:20s}",
                f"{p:.6f}",
            )


# --------------------------------------------------
# 8. MAIN
# --------------------------------------------------

def main():
    global COUNTERFACTUAL_ENABLED
    global COUNTERFACTUAL_ROOT

    # V3-specific flags are removed before
    # handing the remaining CLI args to
    # the tested V2 parser.

    args = sys.argv[1:]

    COUNTERFACTUAL_ENABLED = (
        "--counterfactual"
        in args
    )

    args = [
        x
        for x in args
        if x != "--counterfactual"
    ]

    inspect_path = None

    if "--inspect-router" in args:
        idx = args.index(
            "--inspect-router"
        )

        inspect_path = args[
            idx + 1
        ]

        del args[
            idx:idx + 2
        ]

        inspect_router(
            inspect_path
        )

        return

    # Default output folder if --out
    # is not explicitly supplied.
    if "--out" not in args:
        args.extend([
            "--out",
            (
                "results/aegis_rl_v1/"
                "active_rag_v3"
            ),
        ])

    out_index = (
        args.index("--out") + 1
    )

    out = Path(
        args[out_index]
    ).resolve()

    COUNTERFACTUAL_ROOT = (
        out / "counterfactual"
    )

    # Replace only the Router,
    # Active Environment and Evaluator.
    #
    # V2 still performs:
    #   data loading
    #   historical indexing
    #   Ollama requests
    #   temporal KG/SEC retrieval
    #   portfolio training
    #   checkpointing
    #   result writing

    v2.LearnedRouter = ContextualRouter

    v2.ActiveRAGEnvironment = (
        ContextualActiveRAGEnv
    )

    v2.evaluate_model = (
        evaluate_with_counterfactual
    )

    log(
        "[V3]",
        "Contextual router:",
        STATE_DIM,
        "dimensions",
    )

    log(
        "[V3]",
        "Conditional source/ticker/relation: ON",
    )

    log(
        "[V3]",
        "Counterfactual replay:",
        COUNTERFACTUAL_ENABLED,
    )

    # Restore only options understood by V2.
    sys.argv = [
        sys.argv[0],
        *args,
    ]

    v2.main()

    if COUNTERFACTUAL_RECORDS:
        out.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            COUNTERFACTUAL_RECORDS
        ).to_csv(
            out
            / "counterfactual_summary.csv",
            index=False,
        )

        log(
            "[V3]",
            "Counterfactual results saved",
        )


if __name__ == "__main__":
    main()
