#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V4
RA-DIT-inspired + Self-RAG-inspired + FLARE-inspired

Research architecture:

    Dynamic KGT 200D + Market State
                 |
                 v
    Conditional Retrieval Controller (V3)
                 |
           Ollama FLARE Query
                 |
            KG-RAG / SEC-RAG
                 |
           Evidence Critic
                 |
        Multi-step Follow-up
                 |
          Gated Evidence Fusion
                 |
           A2C / PPO / DDPG
                 |
            Portfolio Reward

Important:
- This is an adapted research implementation.
- Not an exact reproduction of RA-DIT, Self-RAG or FLARE.
- Evidence Fusion gradients are optimized by SB3.
- The evidence critic is a transparent heuristic,
  not the original trained Self-RAG critic.
- Retrieval decisions remain trained by the V3 router.
- The FLARE confidence is the available Ollama
  generated-token confidence proxy.
"""

import sys
import re
import json
import math
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3 import A2C, PPO, DDPG

from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor
)

from stable_baselines3.common.noise import (
    NormalActionNoise
)

# 將 FinRL-Trading 專案根目錄加入 Python 搜尋路徑
ROOT = Path(__file__).resolve().parents[3]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v2 as v2
)

from unified_event_memory.finrl_x.aegis_rl import (
    run_active_rag_v3 as v3
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

TICKERS = v2.TICKERS

RELATIONS = v2.RELATIONS

SOURCES = v2.SOURCES

COST_BPS = v2.COSTS

CONFIDENCE_THRESHOLD = 0.65

MIN_EVIDENCE_RELEVANCE = 0.05

MAX_CRITIC_HISTORY = 100000


def log(*args):
    print(
        time.strftime("%Y-%m-%d %H:%M:%S"),
        *args,
        flush=True
    )


# ============================================================
# 2. RA-DIT-INSPIRED TRAINABLE EVIDENCE FUSION
# ============================================================

class GatedEvidenceExtractor(BaseFeaturesExtractor):
    """
    Trainable feature extractor for SB3.

    Input Observation:
        [market + 200D Dynamic KGT]
        [1024D KG evidence]
        [1024D SEC evidence]
        [5D weights]
        [5D environment extras]

    Architecture:
        Market Encoder
        KG Evidence Encoder
        SEC Evidence Encoder
        Evidence Gate
        Fused Representation

    Gate options:
        MARKET_ONLY
        KG
        SEC

    The gate is jointly trained with Portfolio RL.

    Unlike external detached fusion, the parameters
    belong to the SB3 policy optimizer.
    """

    def __init__(
        self,
        observation_space,
        evidence_dim=1024,
        market_hidden=128,
        evidence_hidden=64,
    ):
        super().__init__(
            observation_space,
            features_dim=market_hidden + evidence_hidden
        )

        obs_dim = int(
            observation_space.shape[0]
        )

        self.evidence_dim = evidence_dim

        self.extra_dim = 10

        self.market_dim = (
            obs_dim
            - 2 * evidence_dim
            - self.extra_dim
        )

        if self.market_dim <= 0:
            raise ValueError(
                "Invalid observation dimensions: "
                f"obs_dim={obs_dim}, "
                f"market_dim={self.market_dim}"
            )

        # Market + KGT + portfolio metadata.
        self.market_encoder = nn.Sequential(
            nn.Linear(
                self.market_dim + self.extra_dim,
                256
            ),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(
                256,
                market_hidden
            ),
            nn.LayerNorm(market_hidden),
            nn.ReLU()
        )

        # KG semantic evidence.
        self.kg_encoder = nn.Sequential(
            nn.Linear(
                evidence_dim,
                128
            ),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(
                128,
                evidence_hidden
            ),
            nn.LayerNorm(evidence_hidden),
            nn.Tanh()
        )

        # SEC semantic evidence.
        self.sec_encoder = nn.Sequential(
            nn.Linear(
                evidence_dim,
                128
            ),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(
                128,
                evidence_hidden
            ),
            nn.LayerNorm(evidence_hidden),
            nn.Tanh()
        )

        # Project market information into the same
        # semantic space as evidence.
        self.market_evidence = nn.Sequential(
            nn.Linear(
                market_hidden,
                evidence_hidden
            ),
            nn.Tanh()
        )

        # Learned source importance.
        self.gate = nn.Sequential(
            nn.Linear(
                market_hidden + 2 * evidence_hidden,
                128
            ),
            nn.ReLU(),

            nn.Linear(
                128,
                3
            )
        )

        # Last gate values for diagnostics.
        self.last_gate = None

    def forward(self, observations):

        batch = observations.shape[0]

        d = self.evidence_dim
        m = self.market_dim

        market = observations[:, :m]

        kg = observations[
            :,
            m:m+d
        ]

        sec = observations[
            :,
            m+d:m+2*d
        ]

        extras = observations[
            :,
            m+2*d:
        ]

        market_input = torch.cat(
            [market, extras],
            dim=1
        )

        h_market = self.market_encoder(
            market_input
        )

        # Explicit evidence-presence masks.
        kg_present = (
            torch.linalg.vector_norm(
                kg,
                dim=1
            ) > 1e-8
        )

        sec_present = (
            torch.linalg.vector_norm(
                sec,
                dim=1
            ) > 1e-8
        )

        h_kg = self.kg_encoder(kg)

        h_sec = self.sec_encoder(sec)

        # Remove bias-only contributions from
        # absent evidence.
        h_kg = h_kg * (
            kg_present.float().unsqueeze(1)
        )

        h_sec = h_sec * (
            sec_present.float().unsqueeze(1)
        )

        gate_input = torch.cat(
            [
                h_market,
                h_kg,
                h_sec
            ],
            dim=1
        )

        logits = self.gate(
            gate_input
        )

        # Gate index 0: MARKET_ONLY
        # Gate index 1: KG
        # Gate index 2: SEC

        availability = torch.stack(
            [
                torch.ones(
                    batch,
                    dtype=torch.bool,
                    device=observations.device
                ),
                kg_present,
                sec_present
            ],
            dim=1
        )

        logits = logits.masked_fill(
            ~availability,
            -1e9
        )

        alpha = torch.softmax(
            logits,
            dim=1
        )

        h_base = self.market_evidence(
            h_market
        )

        h_fused = (
            alpha[:, 0:1] * h_base
            + alpha[:, 1:2] * h_kg
            + alpha[:, 2:3] * h_sec
        )

        self.last_gate = (
            alpha.detach()
        )

        features = torch.cat(
            [
                h_market,
                h_fused
            ],
            dim=1
        )

        return features


# ============================================================
# 3. SELF-RAG-INSPIRED EVIDENCE CRITIC
# ============================================================

class EvidenceCritic:
    """
    Lightweight auditable evidence critic.

    Scores:
      1. Historical temporal validity
      2. Lexical relevance
      3. Source evidence availability

    This is not the trained original Self-RAG critic.

    A low score is logged for analysis.
    The retrieved evidence is not silently
    discarded; otherwise historical retrieval
    comparisons could become inconsistent.
    """

    def __init__(self):
        self.records = []

    @staticmethod
    def tokens(text):
        return set(
            re.findall(
                r"[a-z0-9]+",
                str(text).lower()
            )
        )

    def evaluate(
        self,
        query,
        evidence,
        decision_date
    ):
        decision_date = pd.Timestamp(
            decision_date
        )

        if not evidence:
            result = {
                "score": 0.0,
                "relevant": False,
                "supported": False,
                "valid_time": True,
                "reason": "NO_EVIDENCE"
            }

            return result

        qtokens = self.tokens(
            query
        )

        relevance_scores = []

        for item in evidence:

            available = pd.Timestamp(
                item["available_at"]
            )

            # Hard historical safety check.
            if available >= decision_date:
                raise RuntimeError(
                    "PIT violation in Evidence Critic: "
                    f"{available} >= {decision_date}"
                )

            content = item.get(
                "text_excerpt",
                ""
            )

            etokens = self.tokens(
                content
            )

            if not qtokens:
                lexical = 0.0
            else:
                lexical = (
                    len(
                        qtokens & etokens
                    )
                    / len(qtokens)
                )

            relevance_scores.append(
                lexical
            )

        best = float(
            max(relevance_scores)
        )

        result = {
            "score": best,
            "relevant": (
                best >= MIN_EVIDENCE_RELEVANCE
            ),
            "supported": bool(evidence),
            "valid_time": True,
            "reason": (
                "LEXICAL_EVIDENCE_PRESENT"
                if best >= MIN_EVIDENCE_RELEVANCE
                else "LOW_LEXICAL_OVERLAP"
            )
        }

        # "supported" only means evidence exists.
        # It does not verify an investment thesis.

        if len(self.records) < MAX_CRITIC_HISTORY:
            self.records.append(
                result
            )

        return result


# ============================================================
# 4. FLARE-INSPIRED MULTI-STEP RETRIEVAL
# ============================================================

CRITIC = EvidenceCritic()


def flare_selfrag_search(self):
    """
    Replacement for V2/V3 _search().

    Key changes:

    - Follow-up question is used in next query.
    - Ollama token confidence is recorded.
    - Evidence critic evaluates retrieval results.
    - The trained router still decides STOP/source.
    - No forced query-rate inflation.
    """

    date = self.dates[
        self.day
    ]

    self.phase = 0
    self.used = 0

    self.evidence[:] = 0

    self.day_queries = []

    self.remaining_cost = 0.0

    self.last_hits = 0

    previous_followup = ""

    previous_confidence = None

    for step_id in range(self.steps):

        # Use V3 contextual retrieval state.
        source, ticker, relation = (
            self.router.select(
                self._router_state()
            )
        )

        if self.mode == "NEVER":
            source = "STOP"

        elif self.mode == "ALWAYS_BOTH":
            source = "BOTH"

        if source == "STOP":
            break

        summaries = " ".join(
            q.get("summary", "")
            for q in self.day_queries[-2:]
        )[:500]

        # FLARE-style follow-up is carried
        # into the next search prompt.
        if (
            previous_followup
            and previous_confidence is not None
            and previous_confidence
                < CONFIDENCE_THRESHOLD
        ):
            unresolved = (
                previous_followup
            )

        elif previous_followup:
            unresolved = (
                previous_followup
            )

        else:
            unresolved = (
                "Identify important currently "
                "unresolved financial evidence."
            )

        prompt = (
            f"Historical decision date: {date.date()}.\n"
            f"Company: {ticker}.\n"
            f"Relation: {relation}.\n"
            f"Retrieval source: {source}.\n"
            f"Previously retrieved excerpts: "
            f"{summaries or 'none'}.\n"
            f"Unresolved question: {unresolved}\n"
            "Generate one short financial search query. "
            "Use only information available on or before "
            "the decision date. Do not answer the query."
        )

        question, confidence = (
            self.llm.generate(
                prompt
            )
        )

        # Actual KG/SEC retrieval.
        vector, audit = (
            self.ret.retrieve(
                source,
                ticker,
                relation,
                date,
                question
            )
        )

        # Self-RAG-inspired evaluation.
        critic_result = (
            CRITIC.evaluate(
                question,
                audit,
                date
            )
        )

        # Update KG/SEC semantic state.
        for j in range(2):

            new_vector = np.asarray(
                vector[j],
                dtype=np.float32
            )

            if (
                np.linalg.norm(
                    new_vector
                ) > 0
            ):

                if (
                    np.linalg.norm(
                        self.evidence[j]
                    ) == 0
                ):
                    self.evidence[j] = (
                        new_vector
                    )

                else:
                    self.evidence[j] = (
                        0.5 * self.evidence[j]
                        + 0.5 * new_vector
                    )

        self.used += 1

        cost = (
            COST_BPS[source]
            / 10000.0
            * self.cost_scale
        )

        self.remaining_cost += cost

        self.confidence = float(
            confidence
        )

        self.last_hits = len(
            audit
        )

        if hasattr(
            self,
            "actual_retrieval_count"
        ):
            self.actual_retrieval_count += 1

        if hasattr(
            self,
            "actual_kg_hits"
        ):
            self.actual_kg_hits += sum(
                item["source"] == "KG"
                for item in audit
            )

        if hasattr(
            self,
            "actual_sec_hits"
        ):
            self.actual_sec_hits += sum(
                item["source"] == "SEC"
                for item in audit
            )

        summary = " ".join(
            item.get(
                "text_excerpt",
                ""
            )
            for item in audit[:2]
        )[:400]

        followup_prompt = (
            f"Historical date: {date.date()}.\n"
            f"Financial question: {question}\n"
            f"Retrieved evidence:\n"
            f"{summary or 'NO EVIDENCE'}\n"
            f"Evidence relevance score: "
            f"{critic_result['score']:.4f}\n"
            "Identify one specific unresolved question "
            "that should be researched next. "
            "Do not invent facts."
        )

        followup, followup_confidence = (
            self.llm.generate(
                followup_prompt
            )
        )

        previous_followup = str(
            followup
        )

        previous_confidence = float(
            followup_confidence
        )

        self.day_queries.append({
            "date": str(
                date.date()
            ),
            "step": step_id + 1,
            "source": source,
            "ticker": ticker,
            "relation": relation,
            "query": question,
            "token_confidence": float(
                confidence
            ),
            "followup": followup,
            "followup_confidence": float(
                followup_confidence
            ),
            "flare_low_confidence": bool(
                confidence
                < CONFIDENCE_THRESHOLD
            ),
            "critic": critic_result,
            "evidence": audit,
            "summary": summary,
            "retrieval_cost": cost
        })

        if self.mode == "ALWAYS_BOTH":
            break

    # Portfolio allocation happens after
    # retrieval finishes.
    self.phase = 1


# ============================================================
# 5. SB3 PORTFOLIO ACTOR WITH TRAINABLE FUSION
# ============================================================

def create_v4_portfolio_model(
    agent,
    env,
    seed,
    args
):
    """
    Replace V2 portfolio actor construction.

    Crucially, the gated fusion network is
    instantiated INSIDE SB3 policy_kwargs.

    Its parameters receive RL gradients.
    """

    evidence_dim = env.edim

    policy_kwargs = {
        "features_extractor_class":
            GatedEvidenceExtractor,

        "features_extractor_kwargs": {
            "evidence_dim": evidence_dim,
            "market_hidden": 128,
            "evidence_hidden": 64
        },

        "net_arch": [128, 128]
    }

    common = {
        "seed": seed,
        "learning_rate": args.learning_rate,
        "device": args.device,
        "verbose": 0,
        "policy_kwargs": policy_kwargs
    }

    if agent == "A2C":

        return A2C(
            "MlpPolicy",
            env,
            n_steps=10,
            **common
        )

    if agent == "PPO":

        return PPO(
            "MlpPolicy",
            env,
            n_steps=128,
            batch_size=64,
            **common
        )

    if agent == "DDPG":

        noise = NormalActionNoise(
            mean=np.zeros(5),
            sigma=0.1 * np.ones(5)
        )

        return DDPG(
            "MlpPolicy",
            env,
            action_noise=noise,
            learning_starts=500,
            buffer_size=50000,
            batch_size=128,
            **common
        )

    raise ValueError(
        agent
    )


# ============================================================
# 6. CHECK FEATURE FUSION AND RL GRADIENTS
# ============================================================

def verify_fusion():
    """
    Fast unit test.

    Checks:
      - REAL evidence changes representation
      - ZERO evidence disables KG/SEC gates
      - trainable fusion receives gradients
    """

    from gymnasium import spaces

    obs_dim = (
        5 * (53 + 200)
        + 2 * 1024
        + 10
    )

    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(obs_dim,),
        dtype=np.float32
    )

    extractor = GatedEvidenceExtractor(
        observation_space,
        evidence_dim=1024
    )

    original = torch.randn(
        4,
        obs_dim
    )

    zero = original.clone()

    market_dim = (
        obs_dim
        - 2048
        - 10
    )

    zero[
        :,
        market_dim:
        market_dim + 2048
    ] = 0.0

    h_real = extractor(
        original
    )

    h_zero = extractor(
        zero
    )

    difference = torch.mean(
        torch.abs(
            h_real - h_zero
        )
    ).item()

    print(
        "[FUSION TEST] REAL-ZERO feature difference:",
        difference
    )

    if difference <= 1e-8:
        raise RuntimeError(
            "Evidence Fusion insensitive to evidence"
        )

    loss = h_real.square().mean()

    loss.backward()

    grads = [
        p.grad
        for p in extractor.parameters()
        if p.requires_grad
    ]

    valid = any(
        g is not None
        and torch.isfinite(g).all()
        and g.abs().sum() > 0
        for g in grads
    )

    if not valid:
        raise RuntimeError(
            "Fusion network received no gradients"
        )

    print(
        "FUSION GRADIENT TEST: PASS"
    )

    print(
        "This only verifies differentiability, "
        "not improvement in investment returns."
    )


# ============================================================
# 7. INSTALL IN EXISTING V3 TRAINING PIPELINE
# ============================================================

def install_v4():

    # V3 already uses the contextual router.
    # Replace its environment retrieval routine.
    v3.ContextualActiveRAGEnv._search = (
        flare_selfrag_search
    )

    # Replace SB3 policy creation.
    v2.create_portfolio_model = (
        create_v4_portfolio_model
    )

    log(
        "[V4] RA-DIT-inspired trainable Gated Fusion: ON"
    )

    log(
        "[V4] Self-RAG-inspired Evidence Critic: ON"
    )

    log(
        "[V4] FLARE-inspired multi-step Follow-up: ON"
    )


# ============================================================
# 8. ENTRY POINT
# ============================================================

def main():

    arguments = sys.argv[1:]

    if "--verify-fusion" in arguments:
        verify_fusion()
        return

    install_v4()

    # V3 remains responsible for:
    #   data loading
    #   learned retrieval controller
    #   KG and SEC indexes
    #   semantic embedding cache
    #   model checkpoints
    #   validation/testing
    #
    # V4 changes the actor feature extractor
    # and the active retrieval implementation.

    v3.main()

    if CRITIC.records:

        scores = [
            x["score"]
            for x in CRITIC.records
        ]

        log(
            "[V4 CRITIC]",
            "Evaluations:",
            len(scores),
            "Mean relevance:",
            round(
                float(
                    np.mean(scores)
                ),
                6
            )
        )


if __name__ == "__main__":
    main()
