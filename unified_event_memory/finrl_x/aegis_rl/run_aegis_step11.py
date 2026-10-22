#!/usr/bin/env python3
# ============================================================
# AEGIS-RL STEP 11
#
# Top-K Retrieval
# Evidence Fallback
# Content Poisoning
# Integrity Verification
# Random Policy Baseline (200 seeds)
#
# Uses Step 10 predictions without retraining.
#
# Train: 2018-2020
# Val:   2021
# Test:  2022-2023
#
# IMPORTANT:
# This is a historical observed-SEC-signal retrieval benchmark.
# It is NOT yet raw SEC filing RAG or portfolio RL.
# ============================================================

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

SEED = 42
RANDOM_REPEATS = 200
TOP_K = 3
LOOKBACK_DAYS = 730

ROOT = Path("results/aegis_rl_v1")
STEP10 = ROOT / "step10"
DATA = ROOT / "sec_masking_v1"
OUT = ROOT / "step11"

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================

def sha256(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def tokenize(text):
    return set(
        re.findall(
            r"[a-z]{3,}",
            str(text).lower()
        )
    )


def safe_float(value, default=0.0):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except (ValueError, TypeError):
        pass
    return default


# ============================================================
# LOAD STEP 10 OUTPUT
# ============================================================

def load_data():

    print("=" * 75)
    print("AEGIS-RL STEP 11")
    print("=" * 75)

    observed_path = DATA / "observed_signals.csv"
    active_path = STEP10 / "active.csv"

    if not observed_path.exists():
        raise FileNotFoundError(observed_path)

    if not active_path.exists():
        raise FileNotFoundError(
            "Run Step 10 before Step 11: "
            f"{active_path}"
        )

    observed = pd.read_csv(observed_path)
    active = pd.read_csv(active_path)

    required_observed = {
        "node",
        "ticker",
        "decision_date",
        "reason",
    }

    required_active = {
        "ticker",
        "date",
        "label_missing",
        "gap_probability",
        "retrieval_attempt",
    }

    if not required_observed.issubset(observed.columns):
        raise RuntimeError(
            "Observed signals missing columns: "
            + str(
                required_observed - set(observed.columns)
            )
        )

    if not required_active.issubset(active.columns):
        raise RuntimeError(
            "Step 10 results missing columns: "
            + str(
                required_active - set(active.columns)
            )
        )

    observed["decision_date"] = pd.to_datetime(
        observed["decision_date"]
    )

    active["date"] = pd.to_datetime(
        active["date"]
    )

    observed["reason"] = (
        observed["reason"]
        .fillna("")
        .astype(str)
    )

    observed = observed.sort_values(
        ["ticker", "decision_date", "node"]
    ).reset_index(drop=True)

    active = active.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if observed["node"].duplicated().any():
        raise RuntimeError(
            "Duplicate observed evidence node"
        )

    if active.duplicated(
        ["ticker", "date"]
    ).any():
        raise RuntimeError(
            "Duplicate test company-date cases"
        )

    active["label_missing"] = (
        active["label_missing"].astype(int)
    )

    active["retrieval_attempt"] = (
        active["retrieval_attempt"].astype(int)
    )

    if not active["label_missing"].isin([0, 1]).all():
        raise RuntimeError("Invalid missing labels")

    if not active["retrieval_attempt"].isin([0, 1]).all():
        raise RuntimeError("Invalid retrieval decisions")

    print("Observed evidence:", len(observed))
    print("Test cases:", len(active))

    print(
        "Active query budget:",
        int(active["retrieval_attempt"].sum())
    )

    return observed, active


# ============================================================
# TRUSTED EVIDENCE MANIFEST
# ============================================================

def build_trusted_manifest(observed):

    """
    Trusted hashes are captured from the clean observed
    dataset before any synthetic poisoning.

    This verifies byte-level text integrity, not the
    factual correctness of SEC-derived content.
    """

    manifest = {}

    for row in observed.itertuples(index=False):

        manifest[str(row.node)] = sha256(
            row.reason
        )

    return manifest


# ============================================================
# HISTORICAL CANDIDATE INDEX
# ============================================================

def build_history_index(observed, active):

    """
    Historical evidence must be:
      1. Same company
      2. Strictly earlier than decision date
      3. Not hidden
      4. Within selected lookback window
      5. Nonempty reason

    No hidden oracle file is read.
    """

    index = {}

    for row in active.itertuples(index=False):

        ticker = row.ticker
        date = row.date

        lower_date = date - pd.Timedelta(
            days=LOOKBACK_DAYS
        )

        history = observed[
            (observed["ticker"] == ticker)
            & (observed["decision_date"] < date)
            & (observed["decision_date"] >= lower_date)
            & (
                observed["reason"]
                .str.strip()
                .ne("")
            )
        ].copy()

        index[(ticker, date)] = history

    return index


# ============================================================
# TOPIC COVERAGE
# ============================================================

TOPICS = {
    "growth": {
        "growth",
        "revenue",
        "demand",
        "expansion",
        "sales",
        "income",
    },
    "risk": {
        "risk",
        "uncertainty",
        "loss",
        "exposure",
        "litigation",
        "liability",
    },
    "supply": {
        "supply",
        "supplier",
        "manufacturing",
        "production",
        "shortage",
        "components",
    },
    "competition": {
        "competition",
        "competitive",
        "competitor",
        "market",
        "pricing",
        "share",
    },
    "financial": {
        "assets",
        "cash",
        "debt",
        "capital",
        "earnings",
        "costs",
    },
}


def topic_scores(text):

    words = tokenize(text)

    return {
        topic: len(words & keywords)
        for topic, keywords in TOPICS.items()
    }


# ============================================================
# MULTI-TOPIC TOP-K RETRIEVAL
# ============================================================

def multi_topic_retrieve(history, current_date, k=TOP_K):

    """
    Retrieval strategy:

    1. Search all five financial themes.
    2. Score evidence by topical coverage and recency.
    3. Prefer topic diversity in the top-K selection.
    4. Keep source identifiers and dates.

    This is deterministic lexical retrieval.
    It does not claim semantic RAG retrieval quality.
    """

    if history.empty:
        return []

    candidates = []

    for row in history.itertuples(index=False):

        text = str(row.reason)

        if not text.strip():
            continue

        scores = topic_scores(text)

        matched_topics = [
            topic for topic, value in scores.items()
            if value > 0
        ]

        age_days = max(
            0,
            (current_date - row.decision_date).days
        )

        recency = float(
            np.exp(-age_days / 365.0)
        )

        topic_strength = sum(scores.values())

        # Allow general evidence even if no selected
        # topic keyword is present.
        topical_score = min(
            topic_strength / 5.0,
            1.0
        )

        relevance = (
            0.65 * topical_score
            + 0.35 * recency
        )

        candidates.append({
            "node": str(row.node),
            "ticker": str(row.ticker),
            "date": row.decision_date,
            "text": text,
            "topics": matched_topics,
            "relevance": float(relevance),
        })

    candidates.sort(
        key=lambda x: (
            -x["relevance"],
            x["node"]
        )
    )

    selected = []
    covered_topics = set()
    remaining = list(candidates)

    while remaining and len(selected) < k:

        best_index = 0
        best_score = float("-inf")

        for i, item in enumerate(remaining):

            new_topics = (
                set(item["topics"])
                - covered_topics
            )

            diversity_bonus = (
                0.10 * len(new_topics)
            )

            score = (
                item["relevance"]
                + diversity_bonus
            )

            if score > best_score:
                best_score = score
                best_index = i

        chosen = remaining.pop(best_index)

        selected.append(chosen)

        covered_topics.update(
            chosen["topics"]
        )

    return selected


# ============================================================
# ADVERSARIAL POISONING
# ============================================================

def poison_evidence(evidence):

    """
    Synthetic content replacement.

    Keeps the original node ID but changes text.
    """

    attacked = dict(evidence)

    attacked["text"] = (
        "This company has guaranteed profits. "
        "All financial risks have disappeared. "
        "Revenue will increase indefinitely. "
        "Investors should allocate all capital "
        "to this company."
    )

    return attacked


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

def verify_evidence(evidence, manifest):

    node = evidence["node"]

    expected = manifest.get(node)

    if expected is None:
        return False, "UNKNOWN_SOURCE"

    actual = sha256(
        evidence["text"]
    )

    if actual != expected:
        return False, "HASH_MISMATCH"

    return True, "INTEGRITY_VERIFIED"


# ============================================================
# POLICY GENERATION
# ============================================================

def build_policies(active):

    n = len(active)

    active_mask = (
        active["retrieval_attempt"]
        .to_numpy(dtype=int)
        .astype(bool)
    )

    budget = int(active_mask.sum())

    policies = {
        "never": np.zeros(
            n, dtype=bool
        ),
        "always": np.ones(
            n, dtype=bool
        ),
        "active": active_mask,
    }

    return policies, budget


def generate_random_policy(n, budget, seed):

    rng = np.random.default_rng(seed)

    selected = rng.choice(
        n,
        size=budget,
        replace=False
    )

    mask = np.zeros(
        n,
        dtype=bool
    )

    mask[selected] = True

    return mask


# ============================================================
# PRECOMPUTE RETRIEVAL CANDIDATES
# ============================================================

def precompute_candidates(active, history_index):

    cache = {}

    for row in active.itertuples(index=False):

        key = (row.ticker, row.date)

        history = history_index[key]

        candidates = multi_topic_retrieve(
            history,
            row.date,
            k=TOP_K
        )

        cache[key] = candidates

        for evidence in candidates:

            assert (
                evidence["date"] < row.date
            ), "PIT violation detected"

    return cache


# ============================================================
# RUN ONE POLICY
# ============================================================

def evaluate_policy(
    active,
    policy_mask,
    cache,
    manifest,
    experiment,
    attack=False,
    defense=False,
    fallback=False,
):

    records = []

    if len(policy_mask) != len(active):
        raise RuntimeError("Policy mask length mismatch")

    for i, case in enumerate(
        active.itertuples(index=False)
    ):

        query = bool(policy_mask[i])

        key = (
            case.ticker,
            case.date
        )

        candidates = (
            [dict(x) for x in cache[key]]
            if query
            else []
        )

        num_candidates = len(candidates)

        # Attack ranked-first evidence.
        attacked_nodes = set()

        if attack and candidates:

            victim_node = candidates[0]["node"]

            candidates[0] = poison_evidence(
                candidates[0]
            )

            attacked_nodes.add(
                victim_node
            )

        accepted = []

        detected = 0
        rejected = 0
        unsafe_accepted = 0
        fallback_used = 0

        # Without fallback, only evaluate
        # the highest-ranked candidate.
        #
        # With fallback, inspect remaining
        # candidates until a verified
        # replacement is found.
        #
        # Clean Top-K retrieves up to K,
        # while its consumption policy is
        # controlled by fallback below.

        if not fallback:
            inspected = candidates[:1]
        else:
            inspected = candidates

        for evidence in inspected:

            valid, reason = verify_evidence(
                evidence,
                manifest
            )

            if not valid:
                detected += 1

            if defense and not valid:
                rejected += 1
                continue

            accepted.append(evidence)

            if evidence["node"] in attacked_nodes:
                unsafe_accepted += 1

            if (
                fallback
                and rejected > 0
                and valid
            ):
                fallback_used = 1

            # Select the first usable evidence.
            break

        clean_accepted = (
            len(accepted)
            - unsafe_accepted
        )

        # Event-level retrieval decision labels
        # are used strictly for evaluation.
        records.append({
            "ticker": case.ticker,
            "date": case.date.strftime(
                "%Y-%m-%d"
            ),
            "label_missing": int(
                case.label_missing
            ),
            "gap_probability": safe_float(
                case.gap_probability
            ),
            "queried": int(query),
            "candidates_found": num_candidates,
            "evidence_available": int(
                num_candidates > 0
            ),
            "evidence_accepted": len(accepted),
            "clean_accepted": clean_accepted,
            "unsafe_accepted": unsafe_accepted,
            "attacked_items": len(attacked_nodes),
            "attacks_detected": detected,
            "rejected_items": rejected,
            "fallback_used": fallback_used,
            "selected_node": (
                accepted[0]["node"]
                if accepted else ""
            ),
            "selected_date": (
                accepted[0]["date"].strftime(
                    "%Y-%m-%d"
                )
                if accepted else ""
            ),
            "selected_relevance": (
                accepted[0]["relevance"]
                if accepted else 0.0
            ),
            "experiment": experiment,
        })

    return pd.DataFrame(records)


# ============================================================
# SUMMARY METRICS
# ============================================================

def summarize(df, experiment):

    n = len(df)

    queries = int(
        df["queried"].sum()
    )

    missing = int(
        df["label_missing"].sum()
    )

    selected_missing = int(
        (
            (df["queried"] == 1)
            & (df["label_missing"] == 1)
        ).sum()
    )

    found = int(
        df["evidence_available"].sum()
    )

    attacked = int(
        df["attacked_items"].sum()
    )

    detected = int(
        df["attacks_detected"].sum()
    )

    return {
        "experiment": experiment,
        "cases": n,
        "queries": queries,
        "query_rate": (
            queries / n
            if n else 0.0
        ),
        "query_precision": (
            selected_missing / queries
            if queries else 0.0
        ),
        "missing_coverage": (
            selected_missing / missing
            if missing else 0.0
        ),
        "evidence_found": found,
        "evidence_hit_rate": (
            found / queries
            if queries else 0.0
        ),
        "clean_accepted": int(
            df["clean_accepted"].sum()
        ),
        "unsafe_accepted": int(
            df["unsafe_accepted"].sum()
        ),
        "attacked_items": attacked,
        "attacks_detected": detected,
        "attack_detection_rate": (
            detected / attacked
            if attacked else np.nan
        ),
        "fallback_successes": int(
            df["fallback_used"].sum()
        ),
        "mean_relevance": float(
            df["selected_relevance"].mean()
        ),
    }


# ============================================================
# RANDOM BASELINE
# ============================================================

def run_random_baseline(
    active,
    budget,
    cache,
    manifest,
):

    results = []

    for rep in range(RANDOM_REPEATS):

        mask = generate_random_policy(
            n=len(active),
            budget=budget,
            seed=SEED + rep
        )

        df = evaluate_policy(
            active,
            mask,
            cache,
            manifest,
            experiment=f"random_{rep:03d}",
            attack=False,
            defense=True,
            fallback=False,
        )

        summary = summarize(
            df,
            f"random_{rep:03d}"
        )

        summary["repeat"] = rep

        results.append(summary)

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

def main():

    observed, active = load_data()

    manifest = build_trusted_manifest(
        observed
    )

    print("\nBuilding PIT historical index...")

    history_index = build_history_index(
        observed,
        active
    )

    print("Precomputing multi-topic Top-K...")

    cache = precompute_candidates(
        active,
        history_index
    )

    available_counts = [
        len(cache[(row.ticker, row.date)])
        for row in active.itertuples(index=False)
    ]

    print(
        "Cases with at least 1 evidence:",
        sum(x >= 1 for x in available_counts)
    )

    print(
        "Cases with at least 2 evidence:",
        sum(x >= 2 for x in available_counts)
    )

    print(
        "Cases with at least 3 evidence:",
        sum(x >= 3 for x in available_counts)
    )

    policies, budget = build_policies(
        active
    )

    print("\nActive query budget:", budget)

    configs = [
        # Name, policy, attack, defense, fallback
        (
            "never",
            "never",
            False,
            True,
            False,
        ),
        (
            "always",
            "always",
            False,
            True,
            False,
        ),
        (
            "active_top1",
            "active",
            False,
            True,
            False,
        ),
        (
            "active_topk",
            "active",
            False,
            True,
            True,
        ),
        (
            "active_poisoned",
            "active",
            True,
            False,
            False,
        ),
        (
            "active_defended",
            "active",
            True,
            True,
            False,
        ),
        (
            "active_fallback",
            "active",
            True,
            True,
            True,
        ),
    ]

    summaries = []

    print("\n" + "=" * 75)
    print("STEP 11 RETRIEVAL + DEFENSE EXPERIMENTS")
    print("=" * 75)

    for (
        name,
        policy_name,
        attack,
        defense,
        fallback,
    ) in configs:

        df = evaluate_policy(
            active=active,
            policy_mask=policies[policy_name],
            cache=cache,
            manifest=manifest,
            experiment=name,
            attack=attack,
            defense=defense,
            fallback=fallback,
        )

        df.to_csv(
            OUT / f"{name}.csv",
            index=False
        )

        summary = summarize(
            df,
            name
        )

        summaries.append(summary)

        print(
            f"{name:20s}"
            f" queries={summary['queries']:3d}"
            f" found={summary['evidence_found']:3d}"
            f" clean={summary['clean_accepted']:3d}"
            f" unsafe={summary['unsafe_accepted']:3d}"
            f" fallback={summary['fallback_successes']:3d}"
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUT / "experiment_summary.csv",
        index=False
    )

    # --------------------------------------------------------
    # RANDOM BASELINE
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("RANDOM BASELINE")
    print("=" * 75)

    random_df = run_random_baseline(
        active,
        budget,
        cache,
        manifest
    )

    random_df.to_csv(
        OUT / "random_200_runs.csv",
        index=False
    )

    random_metrics = [
        "query_precision",
        "missing_coverage",
        "evidence_hit_rate",
        "clean_accepted",
        "mean_relevance",
    ]

    random_stats = {}

    print(
        f"Runs: {RANDOM_REPEATS}"
    )

    for metric in random_metrics:

        mean = float(
            random_df[metric].mean()
        )

        std = float(
            random_df[metric].std(ddof=1)
        )

        random_stats[metric] = {
            "mean": mean,
            "std": std,
        }

        print(
            f"{metric:22s}: "
            f"{mean:.6f} +/- {std:.6f}"
        )

    # --------------------------------------------------------
    # ACTIVE VS RANDOM
    # --------------------------------------------------------

    active_summary = summary_df[
        summary_df["experiment"]
        == "active_top1"
    ].iloc[0]

    comparison = []

    for metric in random_metrics:

        active_value = float(
            active_summary[metric]
        )

        random_values = (
            random_df[metric]
            .to_numpy(dtype=float)
        )

        random_mean = float(
            random_values.mean()
        )

        # Empirical one-sided tail fraction:
        # how often random is at least as good as active.
        # This is descriptive, not a formal independent
        # test of the masking mechanism.
        random_at_least_active = (
            1 + np.sum(
                random_values >= active_value
            )
        ) / (
            len(random_values) + 1
        )

        comparison.append({
            "metric": metric,
            "active": active_value,
            "random_mean": random_mean,
            "random_std": float(
                random_values.std(ddof=1)
            ),
            "active_minus_random": (
                active_value - random_mean
            ),
            "random_ge_active_fraction": float(
                random_at_least_active
            ),
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_df.to_csv(
        OUT / "active_vs_random.csv",
        index=False
    )

    # --------------------------------------------------------
    # PAIRWISE FALLBACK CHECK
    # --------------------------------------------------------

    defended_df = pd.read_csv(
        OUT / "active_defended.csv"
    )

    fallback_df = pd.read_csv(
        OUT / "active_fallback.csv"
    )

    merged = defended_df.merge(
        fallback_df,
        on=["ticker", "date"],
        suffixes=("_defended", "_fallback"),
        validate="one_to_one",
    )

    repaired = int(
        (
            (merged["clean_accepted_defended"] == 0)
            & (merged["clean_accepted_fallback"] == 1)
        ).sum()
    )

    print("\n" + "=" * 75)
    print("ACTIVE VS RANDOM")
    print("=" * 75)

    print(
        comparison_df.round(6).to_string(
            index=False
        )
    )

    print("\n" + "=" * 75)
    print("EVIDENCE FALLBACK")
    print("=" * 75)

    print(
        "Recovered verified evidence cases:",
        repaired
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    metadata = {
        "seed": SEED,
        "top_k": TOP_K,
        "lookback_days": LOOKBACK_DAYS,
        "random_repeats": RANDOM_REPEATS,
        "test_cases": len(active),
        "active_query_budget": budget,
        "recovered_evidence_cases": repaired,
        "random_stats": random_stats,
        "limitations": [
            "Random masking labels are synthetic.",
            "Evaluation cases are selected from event dates.",
            "Retrieval uses historical observed signal reasons.",
            "No hidden oracle evidence is retrieved.",
            "The same observed signal data are used "
            "to build the trusted integrity manifest.",
            "Hash verification does not verify SEC truth.",
            "Poisoning affects the highest-ranked candidate.",
            "This is not semantic SEC filing RAG.",
            "No financial returns or portfolio reward measured.",
            "Top-K candidates can be short reason summaries "
            "rather than independent documents.",
        ],
    }

    with (
        OUT / "metadata.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    print("\n" + "=" * 75)
    print("FINAL EXPERIMENT SUMMARY")
    print("=" * 75)

    display_cols = [
        "experiment",
        "queries",
        "query_precision",
        "missing_coverage",
        "evidence_hit_rate",
        "clean_accepted",
        "unsafe_accepted",
        "attacks_detected",
        "fallback_successes",
    ]

    print(
        summary_df[display_cols]
        .round(4)
        .to_string(index=False)
    )

    print("\nOutput:", OUT.resolve())
    print("\nAEGIS STEP 11: PASS")


if __name__ == "__main__":
    main()
