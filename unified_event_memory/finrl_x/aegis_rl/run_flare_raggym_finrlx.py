#!/usr/bin/env python3
"""
FinRL-X: FLARE-inspired + RAG-Gym-inspired Active RAG

Extends:
  run_self_questioning_v1.py

Features:
  - Structured financial question generation
  - Dynamic KG 200D retained
  - Temporal KG-RAG
  - Temporal SEC-RAG
  - Multiple retrieval decisions before allocation
  - Agent-controlled stopping
  - Accumulated retrieved evidence
  - Retrieval costs deducted from portfolio return
  - A2C / PPO / DDPG training
  - Query/evidence audit in existing CSV outputs

Important:
  This is an adaptation of FLARE/RAG-Gym ideas.
  It does not run their original language models
  or implement RAG-Gym process reward modeling.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from unified_event_memory.finrl_x.aegis_rl import (
    run_self_questioning_v1 as base
)


class MultiStepFinancialRAGEnv(
    base.SelfQuestioningEnv
):
    """
    Two stages per market day:

    Stage 0:
       Choose source / company / relation.
       Retrieve evidence or STOP.
       Repeat up to max_search_steps.

    Stage 1:
       Observe accumulated evidence.
       Allocate portfolio.
       Advance to next trading date.

    Action shape remains 18:
       0:4   retrieval source / stop
       4:9   company
       9:13  relation
       13:18 portfolio allocation

    Retrieval cost is accumulated on each
    actual retrieval and deducted once
    at portfolio allocation.
    """

    max_search_steps = 3

    def __init__(self, *args, **kwargs):
        self.search_count = 0
        self.search_history = []
        self.seen_queries = set()
        self.last_evidence_count = 0

        super().__init__(
            *args,
            **kwargs,
        )

    def reset(self, *, seed=None, options=None):
        self.search_count = 0
        self.search_history = []
        self.seen_queries = set()
        self.last_evidence_count = 0

        return super().reset(
            seed=seed,
            options=options,
        )

    def _obs(self):
        observation = super()._obs()

        # FLARE-inspired information-need proxy.
        #
        # The original environment supplies:
        #   phase, recent volatility, coverage gap
        #
        # Replace the final gap indicator with
        # a continuous coverage estimate.
        #
        # This is NOT token-level uncertainty
        # and not calibrated predictive entropy.

        evidence_coverage = (
            self.last_evidence_count
        )

        gap = (
            1.0
            / (1.0 + evidence_coverage)
        )

        observation[-1] = float(gap)

        return observation

    def _start_new_day(self):
        self.search_count = 0
        self.search_history = []
        self.seen_queries = set()
        self.last_evidence_count = 0
        self.current_cost = 0.0
        self.current_query = None

    def _query_decision(self, action):
        action = np.asarray(
            action,
            dtype=float,
        ).reshape(-1)

        source_index = int(
            np.argmax(
                action[0:4]
            )
        )

        company_index = int(
            np.argmax(
                action[4:9]
            )
        )

        relation_index = int(
            np.argmax(
                action[9:13]
            )
        )

        source = base.ACTIONS[
            source_index
        ]

        if self.mode == "NEVER":
            source = "SKIP"

        if self.mode == "ALWAYS_BOTH":
            source = "BOTH"

        ticker = base.TICKERS[
            company_index
        ]

        relation = base.RELATIONS[
            relation_index
        ]

        return (
            source,
            ticker,
            relation,
        )

    def _finish_search(self):
        day = self.dates[self.idx]

        self.current_query = {
            "date": str(day.date()),
            "mode": self.mode,

            # Source actions, not just one action.
            "action": (
                "SKIP"
                if not self.search_history
                else "+".join(
                    x["action"]
                    for x in self.search_history
                )
            ),

            "ticker": (
                self.search_history[-1]["ticker"]
                if self.search_history
                else ""
            ),

            "relation": (
                self.search_history[-1]["relation"]
                if self.search_history
                else ""
            ),

            "query": (
                self.search_history[-1]["query"]
                if self.search_history
                else ""
            ),

            "query_steps": list(
                self.search_history
            ),

            "evidence": [
                item
                for step in self.search_history
                for item in step["evidence"]
            ],

            "retrieval_cost_fraction":
                float(
                    self.current_cost
                ),

            "n_queries": len(
                self.search_history
            ),

            "n_evidence":
                int(
                    self.last_evidence_count
                ),
        }

        self.phase = 1

    def step(self, action):
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action.shape != (18,):
            raise ValueError(
                "Expected action shape (18,)"
            )

        # Portfolio allocation stage.
        if self.phase == 1:
            outcome = super().step(
                action
            )

            # Parent advanced to next day.
            self._start_new_day()

            obs, reward, done, truncated, info = outcome

            return (
                self._obs(),
                reward,
                done,
                truncated,
                info,
            )

        # Retrieval stage.
        day = self.dates[self.idx]

        (
            source,
            ticker,
            relation,
        ) = self._query_decision(
            action
        )

        if source == "SKIP":
            # SKIP now means:
            # stop searching and allocate.
            self._finish_search()

            return (
                self._obs(),
                0.0,
                False,
                False,
                {
                    "phase": "allocation_ready",
                    "stopped": True,
                },
            )

        fingerprint = (
            source,
            ticker,
            relation,
        )

        if fingerprint in self.seen_queries:
            # Avoid paying for the same query twice.
            # The agent still consumes a step.

            self.search_count += 1

            if (
                self.search_count
                >= self.max_search_steps
            ):
                self._finish_search()

            return (
                self._obs(),
                0.0,
                False,
                False,
                {
                    "phase": "retrieval",
                    "duplicate": True,
                },
            )

        self.seen_queries.add(
            fingerprint
        )

        evidence, provenance = (
            self.retrievers.run(
                source,
                ticker,
                relation,
                day,
            )
        )

        # Critical: reject future evidence.
        for item in provenance:
            available = pd.Timestamp(
                item["available_at"]
            )

            if available >= day:
                raise RuntimeError(
                    "Point-in-Time violation: "
                    f"{available} >= {day}"
                )

        # Combine evidence from multiple
        # retrieval actions. Maximum pooling
        # preserves the strongest signal in
        # each KG/SEC evidence dimension.
        self.retrieved = np.maximum(
            self.retrieved,
            evidence,
        )

        self.last_evidence_count += (
            len(provenance)
        )

        query_cost = (
            base.ACTION_COST_BPS[source]
            / 10000.0
        )

        self.current_cost += (
            query_cost
        )

        record = {
            "step": self.search_count + 1,
            "date": str(day.date()),
            "action": source,
            "ticker": ticker,
            "relation": relation,
            "query": base.query_text(
                ticker,
                relation,
            ),
            "evidence": provenance,
            "n_evidence": len(
                provenance
            ),
            "cost_fraction":
                query_cost,
        }

        self.search_history.append(
            record
        )

        self.search_count += 1

        # For ALWAYS_BOTH, perform one retrieval
        # per day, matching the original baseline.
        limit = (
            1
            if self.mode == "ALWAYS_BOTH"
            else self.max_search_steps
        )

        if self.search_count >= limit:
            self._finish_search()

        return (
            self._obs(),
            0.0,
            False,
            False,
            {
                "phase": (
                    "allocation_ready"
                    if self.phase == 1
                    else "retrieval"
                ),
                "n_queries": len(
                    self.search_history
                ),
                "n_evidence": len(
                    provenance
                ),
                "source": source,
            },
        )


def main():
    # Parse only this extension's arguments.
    # Pass the remaining CLI arguments to
    # the previously implemented FinRL-X runner.
    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--max-search-steps",
        type=int,
        default=3,
    )

    extension, remaining = (
        parser.parse_known_args()
    )

    if extension.max_search_steps < 1:
        raise ValueError(
            "max-search-steps must be >= 1"
        )

    if "--smoke" in remaining:
        # The original smoke test expects
        # allocation phase after one query.
        extension.max_search_steps = 1

    MultiStepFinancialRAGEnv.max_search_steps = (
        extension.max_search_steps
    )

    # Replace only the Gym environment.
    # Keep the original data loaders,
    # KG/SEC retrievers and 3 RL algorithms.
    base.SelfQuestioningEnv = (
        MultiStepFinancialRAGEnv
    )

    sys.argv = [
        sys.argv[0],
        *remaining,
    ]

    base.main()


if __name__ == "__main__":
    main()
