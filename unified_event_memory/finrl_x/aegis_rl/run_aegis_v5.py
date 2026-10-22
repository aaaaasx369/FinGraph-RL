#!/usr/bin/env python3
"""
AEGIS-RL V5
Cost-Sensitive Learned Active KG-SEC Retrieval

Train: 2018-2020 (existing V4 models)
Validation: 2021 (cost calibration)
Test: 2022-2023 (held out)

Outputs:
- validation_cost_sweep.csv
- test_cost_sensitivity.csv
- policy_benchmark.csv
- random_200_seeds.csv
- test_policy_decisions.csv
- learned_policy_by_year.csv
- metadata.json

Important:
Utility = V4 constructed retrieval proxy.
It is NOT realized investment profit.

KGT is not retrained.
Test labels/rewards are never used to select lambda.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

ROOT = Path.cwd()

V4 = ROOT / "results/aegis_rl_v1/learned_policy_v4"
OUT = ROOT / "results/aegis_rl_v1/cost_sensitive_v5"

OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
RANDOM_REPEATS = 200

ACTIONS = ["SKIP", "KG", "SEC", "BOTH"]

COST = np.array([
    0.00,
    0.03,
    0.08,
    0.11,
], dtype=float)

# Predefined cost sweep.
# Lambda is selected using 2021 validation only.
LAMBDAS = np.linspace(0, 50, 501)

# Target budget, chosen before inspecting 2022-2023.
TARGET_QUERY_RATE = 0.50

FEATURES = [
    "kg_count_365d_log",
    "sec_count_365d_log",
    "kg_age_log",
    "sec_age_log",
    "kg_relation_diversity",
    "sec_document_count_log",
    "sec_mean_text_length_log",
    "sec_form_diversity",
    "kg_recent_90d_log",
    "sec_recent_90d_log",
    "kg_sec_count_ratio",
]


def load_data():

    path = V4 / "historical_states.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = (
        ["date", "tic", "split"]
        + FEATURES
        + [f"reward_{a}" for a in ACTIONS]
    )

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing historical columns: {missing}"
        )

    df["date"] = pd.to_datetime(df["date"])

    if df.duplicated(["date", "tic"]).any():
        raise RuntimeError(
            "Duplicate company-date states"
        )

    if df[required[3:]].isna().any().any():
        raise RuntimeError(
            "Missing state or reward values"
        )

    return df.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)


def load_models():

    models = {}

    for action in ACTIONS:

        path = V4 / f"policy_{action}.joblib"

        if not path.exists():
            raise FileNotFoundError(path)

        models[action] = load(path)

    return models


def predicted_v4_reward(frame, models):

    pred = np.column_stack([
        models[action].predict(
            frame[FEATURES]
        )
        for action in ACTIONS
    ])

    if not np.isfinite(pred).all():
        raise RuntimeError(
            "Nonfinite predicted reward"
        )

    return pred


def actual_v4_reward(frame):

    return frame[
        [f"reward_{a}" for a in ACTIONS]
    ].to_numpy(dtype=float)


def adjusted_rewards(rewards_v4, lam):

    # V4 rewards already subtract one unit of COST:
    #
    # reward_v4 = evidence_utility - COST
    #
    # Thus:
    # reward_lambda = reward_v4 - (lambda-1)*COST

    return (
        rewards_v4
        - (lam - 1.0) * COST[None, :]
    )


def choose_actions(pred_v4, lam):

    adjusted = adjusted_rewards(
        pred_v4, lam
    )

    # np.argmax resolves ties in ACTIONS order:
    # SKIP first.
    return np.argmax(
        adjusted, axis=1
    )


def summarize_policy(
    actual_v4,
    action_ids,
    lam,
    name,
):

    matrix = adjusted_rewards(
        actual_v4, lam
    )

    n = len(action_ids)

    realized = matrix[
        np.arange(n), action_ids
    ]

    oracle = matrix.max(axis=1)

    cost = COST[action_ids]

    result = {
        "policy": name,
        "cases": n,
        "lambda": float(lam),
        "mean_net_proxy_reward": float(
            realized.mean()
        ),
        "mean_proxy_regret": float(
            (oracle - realized).mean()
        ),
        "query_rate": float(
            np.mean(action_ids != 0)
        ),
        "mean_base_cost": float(
            cost.mean()
        ),
        "mean_weighted_cost": float(
            (lam * cost).mean()
        ),
    }

    for i, action in enumerate(ACTIONS):
        result[action] = int(
            np.sum(action_ids == i)
        )

    return result


def calibrate_lambda(
    val_pred,
    val_actual,
):

    rows = []

    for lam in LAMBDAS:

        decisions = choose_actions(
            val_pred, lam
        )

        row = summarize_policy(
            val_actual,
            decisions,
            lam,
            "Learned_V5_Validation",
        )

        row["budget_gap"] = abs(
            row["query_rate"]
            - TARGET_QUERY_RATE
        )

        rows.append(row)

    sweep = pd.DataFrame(rows)

    sweep.to_csv(
        OUT / "validation_cost_sweep.csv",
        index=False,
    )

    # Primary criterion: get closest to the
    # predefined validation query budget.
    #
    # Secondary criterion: highest validation
    # net proxy reward.
    #
    # No test information is involved.
    ordered = sweep.sort_values(
        [
            "budget_gap",
            "mean_net_proxy_reward",
            "lambda",
        ],
        ascending=[True, False, True],
    )

    selected = ordered.iloc[0]

    return (
        float(selected["lambda"]),
        sweep,
        selected,
    )


def load_v4_test_actions(test):

    path = V4 / "test_policy_predictions.csv"

    old = pd.read_csv(path)

    required = ["date", "tic", "action"]

    if not set(required).issubset(old.columns):
        raise RuntimeError(
            "Invalid V4 prediction file"
        )

    old["date"] = pd.to_datetime(
        old["date"]
    )

    merged = test[
        ["date", "tic"]
    ].merge(
        old[required],
        on=["date", "tic"],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if merged["action"].isna().any():
        raise RuntimeError(
            "V4 test action alignment failed"
        )

    action_map = {
        a: i for i, a in enumerate(ACTIONS)
    }

    return merged["action"].map(
        action_map
    ).to_numpy(dtype=int)


def evaluate_test(
    test_actual,
    learned_ids,
    v4_ids,
    lam,
):

    n = len(learned_ids)

    policies = {
        "Never": np.full(
            n, 0, dtype=int
        ),
        "Always_KG": np.full(
            n, 1, dtype=int
        ),
        "Always_SEC": np.full(
            n, 2, dtype=int
        ),
        "Always_BOTH": np.full(
            n, 3, dtype=int
        ),
        "Learned_V4": v4_ids,
        "Learned_V5": learned_ids,
    }

    records = [
        summarize_policy(
            test_actual,
            ids,
            lam,
            name,
        )
        for name, ids in policies.items()
    ]

    return pd.DataFrame(records)


def random_matched_budget(
    test_actual,
    learned_ids,
    lam,
):

    records = []

    for repeat in range(RANDOM_REPEATS):

        rng = np.random.default_rng(
            SEED + repeat
        )

        # Exactly the same action counts
        # and exactly the same mean cost
        # as the learned policy.
        random_ids = rng.permutation(
            learned_ids
        )

        result = summarize_policy(
            test_actual,
            random_ids,
            lam,
            f"Random_{repeat:03d}",
        )

        result["seed"] = SEED + repeat

        records.append(result)

    df = pd.DataFrame(records)

    df.to_csv(
        OUT / "random_200_seeds.csv",
        index=False,
    )

    return df


def test_cost_sweep(test_pred, test_actual):

    """
    Sensitivity analysis only.

    Lambda selection must NOT use these
    test results. The chosen lambda is
    already fixed on validation.
    """

    records = []

    for lam in LAMBDAS:

        decisions = choose_actions(
            test_pred, lam
        )

        records.append(
            summarize_policy(
                test_actual,
                decisions,
                lam,
                "Learned_V5_Sensitivity",
            )
        )

    result = pd.DataFrame(records)

    result.to_csv(
        OUT / "test_cost_sensitivity.csv",
        index=False,
    )

    return result


def main():

    print("=" * 76)
    print("AEGIS-RL V5 COST-SENSITIVE RETRIEVAL")
    print("=" * 76)

    df = load_data()
    models = load_models()

    train = df[
        df.split == "train"
    ].copy()

    val = df[
        df.split == "val"
    ].copy()

    test = df[
        df.split == "test"
    ].copy()

    if train.empty or val.empty or test.empty:
        raise RuntimeError(
            "Train, val or test split missing"
        )

    print("Train:", len(train))
    print("Validation:", len(val))
    print("Test:", len(test))

    assert val.date.dt.year.eq(2021).all()
    assert test.date.dt.year.isin([2022, 2023]).all()

    val_pred = predicted_v4_reward(
        val, models
    )

    val_actual = actual_v4_reward(val)

    print("\n===== VALIDATION COST CALIBRATION =====")

    selected_lambda, sweep, selected = (
        calibrate_lambda(
            val_pred,
            val_actual,
        )
    )

    print(
        "Target query rate:",
        TARGET_QUERY_RATE
    )

    print(
        "Selected lambda:",
        round(selected_lambda, 4)
    )

    print(
        "Validation query rate:",
        round(
            float(selected["query_rate"]),
            6,
        )
    )

    print(
        "Validation net proxy reward:",
        round(
            float(
                selected["mean_net_proxy_reward"]
            ),
            6,
        )
    )

    print("\n===== TEST EVALUATION =====")

    test_pred = predicted_v4_reward(
        test, models
    )

    test_actual = actual_v4_reward(test)

    learned_ids = choose_actions(
        test_pred,
        selected_lambda,
    )

    v4_ids = load_v4_test_actions(
        test
    )

    benchmark = evaluate_test(
        test_actual,
        learned_ids,
        v4_ids,
        selected_lambda,
    )

    random_df = random_matched_budget(
        test_actual,
        learned_ids,
        selected_lambda,
    )

    random_mean = float(
        random_df[
            "mean_net_proxy_reward"
        ].mean()
    )

    random_std = float(
        random_df[
            "mean_net_proxy_reward"
        ].std(ddof=1)
    )

    random_row = {
        "policy": "Random_MatchedBudget",
        "cases": len(test),
        "lambda": selected_lambda,
        "mean_net_proxy_reward": random_mean,
        "mean_proxy_regret": float(
            random_df[
                "mean_proxy_regret"
            ].mean()
        ),
        "query_rate": float(
            random_df[
                "query_rate"
            ].mean()
        ),
        "mean_base_cost": float(
            random_df[
                "mean_base_cost"
            ].mean()
        ),
        "mean_weighted_cost": float(
            random_df[
                "mean_weighted_cost"
            ].mean()
        ),
    }

    for action in ACTIONS:
        random_row[action] = int(
            np.sum(
                learned_ids
                == ACTIONS.index(action)
            )
        )

    benchmark = pd.concat(
        [
            benchmark,
            pd.DataFrame([random_row]),
        ],
        ignore_index=True,
    )

    benchmark.to_csv(
        OUT / "policy_benchmark.csv",
        index=False,
    )

    print(
        benchmark.round(6).to_string(
            index=False
        )
    )

    learned_reward = float(
        benchmark.loc[
            benchmark.policy == "Learned_V5",
            "mean_net_proxy_reward",
        ].iloc[0]
    )

    random_rewards = random_df[
        "mean_net_proxy_reward"
    ].to_numpy()

    random_ge_learned = float(
        (
            1 + np.sum(
                random_rewards >= learned_reward
            )
        ) / (
            RANDOM_REPEATS + 1
        )
    )

    print("\n===== LEARNED VS RANDOM =====")

    print(
        "Learned reward:",
        round(learned_reward, 6)
    )

    print(
        "Random mean:",
        round(random_mean, 6)
    )

    print(
        "Random std:",
        round(random_std, 6)
    )

    print(
        "Learned minus random:",
        round(
            learned_reward - random_mean,
            6,
        )
    )

    print(
        "Random >= Learned fraction:",
        round(random_ge_learned, 6)
    )

    # --------------------------------------------------
    # Detailed daily decisions
    # --------------------------------------------------

    test_out = test[
        ["date", "tic", "split"] + FEATURES
    ].copy()

    test_out["action"] = [
        ACTIONS[i]
        for i in learned_ids
    ]

    test_out["v4_action"] = [
        ACTIONS[i]
        for i in v4_ids
    ]

    test_out["selected_lambda"] = (
        selected_lambda
    )

    pred_matrix = adjusted_rewards(
        test_pred,
        selected_lambda,
    )

    actual_matrix = adjusted_rewards(
        test_actual,
        selected_lambda,
    )

    for i, action in enumerate(ACTIONS):

        test_out[
            f"predicted_net_{action}"
        ] = pred_matrix[:, i]

        test_out[
            f"actual_proxy_net_{action}"
        ] = actual_matrix[:, i]

    test_out.to_csv(
        OUT / "test_policy_decisions.csv",
        index=False,
    )

    # --------------------------------------------------
    # Yearly breakdown
    # --------------------------------------------------

    yearly = []

    for year in [2022, 2023]:

        mask = (
            test.date.dt.year.to_numpy()
            == year
        )

        summary = summarize_policy(
            test_actual[mask],
            learned_ids[mask],
            selected_lambda,
            "Learned_V5",
        )

        summary["year"] = year

        yearly.append(summary)

    yearly_df = pd.DataFrame(yearly)

    yearly_df.to_csv(
        OUT / "learned_policy_by_year.csv",
        index=False,
    )

    print("\n===== LEARNED V5 BY YEAR =====")

    print(
        yearly_df[
            [
                "year",
                "query_rate",
                "mean_net_proxy_reward",
                "SKIP",
                "KG",
                "SEC",
                "BOTH",
            ]
        ].round(6).to_string(
            index=False
        )
    )

    # This is a reporting-only sensitivity sweep.
    test_cost_sweep(
        test_pred,
        test_actual,
    )

    metadata = {
        "version": "AEGIS-RL V5",
        "train": "2018-2020",
        "validation": "2021",
        "test": "2022-2023",
        "target_validation_query_rate": (
            TARGET_QUERY_RATE
        ),
        "selected_lambda": (
            selected_lambda
        ),
        "validation_query_rate": float(
            selected["query_rate"]
        ),
        "test_query_rate": float(
            np.mean(
                learned_ids != 0
            )
        ),
        "random_repeats": (
            RANDOM_REPEATS
        ),
        "action_base_costs": {
            action: float(COST[i])
            for i, action in enumerate(ACTIONS)
        },
        "random_ge_learned_fraction": (
            random_ge_learned
        ),
        "notes": [
            "Uses V4 models without retraining.",
            "Lambda calibrated on 2021 only.",
            "2022-2023 never used to select lambda.",
            "Random policy matches exact action counts.",
            "Rewards are constructed retrieval proxies.",
            "No new SEC or KG documents are retrieved.",
            "No A2C PPO DDPG trading experiments.",
            "No masking-based epistemic evaluation.",
            "KGT embeddings not used in policy fitting.",
            "No claim of financial returns.",
        ],
    }

    with (
        OUT / "metadata.json"
    ).open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n===== OUTPUT =====")
    print("Directory:", OUT)
    print("AEGIS V5: PASS")


if __name__ == "__main__":
    main()
