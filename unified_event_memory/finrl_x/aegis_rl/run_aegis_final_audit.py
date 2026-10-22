#!/usr/bin/env python3
"""
AEGIS-RL Final Audit

Stages:
  1. Load existing 180-run portfolio results
  2. Validate 2018-2020 / 2021 / 2022-2023 splits
  3. Select portfolio algorithm on 2021 validation
  4. Calibrate four-action retrieval cost on 2021
  5. Evaluate held-out 2022-2023 performance
  6. Run Real / Zero / Lagged / Shuffled evidence tests
  7. Run Clean / Synthetic Attack / Defense comparisons
  8. Audit evidence provenance and write summaries

Existing portfolio models are reused.
No model retraining is performed.

Research status: EXPLORATORY.
Strict PIT certification is NOT automatic.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, PPO, DDPG

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OLD = ROOT / "results/aegis_rl_v1/evidence_rag_portfolio"
V4 = ROOT / "results/aegis_rl_v1/learned_policy_v4"
V5 = ROOT / "results/aegis_rl_v1/cost_sensitive_v5"

OUT_DEFAULT = (
    ROOT / "results/aegis_rl_v1/aegis_final_audit"
)

SCRIPT = (
    ROOT
    / "unified_event_memory/finrl_x/aegis_rl"
    / "run_aegis_evidence_portfolio.py"
)

ACTIONS = ["SKIP", "KG", "SEC", "BOTH"]

PROXY_COST = {
    "SKIP": 0.0,
    "KG": 0.03,
    "SEC": 0.08,
    "BOTH": 0.11,
}

RAG_COLS = (
    [f"rag_kg_{i}" for i in range(8)]
    + [f"rag_sec_{i}" for i in range(8)]
)

POLICY_COLS = [
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

LAMBDA_GRID = [
    0.0, 0.5, 1.0, 2.0, 4.0,
    8.0, 12.0, 16.0, 22.0, 29.2,
]

MODEL_CLASSES = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
}

EXPECTED_ARMS = [
    "CLEAN",
    "CLEAN_RAG",
    "FULL_STATIC",
    "FULL_STATIC_RAG",
    "FULL_DYNAMIC_PLACEBO",
    "FULL_DYNAMIC_PLACEBO_RAG",
    "FULL_DYNAMIC",
    "FULL_DYNAMIC_RAG",
    "DYNAMIC_ALWAYS_BOTH",
    "DYNAMIC_RANDOM",
    "DYNAMIC_ACTIVE",
    "DYNAMIC_ACTIVE_DEFENSE",
]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, indent=2,
            default=str,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required input: {path}"
        )
    return path


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name, path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(
        spec
    )
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_frame(path):
    return pd.read_csv(require(path))


def normalize_keys(df, ticker_col="ticker"):
    x = df.copy()
    x["date"] = pd.to_datetime(
        x["date"], errors="raise", utc=True
    ).dt.tz_convert(None).dt.normalize()
    x[ticker_col] = (
        x[ticker_col].astype(str).str.upper()
    )
    return x


def check_unique(df, keys, label):
    duplicates = df.duplicated(keys)
    if duplicates.any():
        raise RuntimeError(
            f"{label}: duplicate keys "
            f"{df.loc[duplicates, keys].head()}"
        )


def load_completed_runs():
    results = read_frame(
        OLD / "all_results.csv"
    )

    keys = ["arm", "agent", "seed", "split"]
    required_cols = keys + [
        "total_return",
        "sharpe",
        "max_drawdown",
    ]

    missing = [
        c for c in required_cols
        if c not in results.columns
    ]
    if missing:
        raise RuntimeError(
            f"Missing result columns: {missing}"
        )

    check_unique(
        results,
        keys,
        "portfolio results",
    )

    counts = results.groupby(
        ["arm", "agent", "seed"]
    )["split"].agg(
        lambda x: set(x)
    )

    if not all(
        x == {"val", "test"}
        for x in counts
    ):
        raise RuntimeError(
            "Incomplete validation/test results"
        )

    for arm in EXPECTED_ARMS:
        for agent in MODEL_CLASSES:
            s = set(
                results.loc[
                    (results.arm == arm)
                    & (results.agent == agent),
                    "seed"
                ]
            )
            if len(s) != 5:
                raise RuntimeError(
                    f"Incomplete 5-seed run: "
                    f"{arm}/{agent}"
                )

    print(
        "Completed portfolio models:",
        len(counts),
    )
    return results


def select_models(results, mode):
    """
    All selection uses 2021 validation ONLY.
    One selected algorithm per seed.
    No test performance used.
    """

    val = results[
        results.split.eq("val")
        & results.arm.eq("DYNAMIC_ACTIVE")
    ].copy()

    if mode == "sharpe":
        metric = "sharpe"
    else:
        metric = "total_return"

    val = val.sort_values(
        ["seed", metric, "agent"],
        ascending=[True, False, True],
    )

    selected = (
        val.groupby("seed")
        .head(1)
        .reset_index(drop=True)
    )

    if len(selected) != 5:
        raise RuntimeError(
            "Expected five selected portfolio models"
        )

    return selected[
        ["seed", "agent", "total_return", "sharpe"]
    ].copy()


def load_market_and_helpers(aegis, tier):
    v21, v18, original = (
        aegis.load_original_helpers()
    )

    schema, extra = aegis.load_schema()

    market = aegis.load_market(
        tier, schema, extra
    )

    market = normalize_keys(
        market,
        "ticker",
    )

    # No 2024+ observations.
    if market.date.max() >= pd.Timestamp(
        "2024-01-01"
    ):
        raise RuntimeError(
            "Unexpected post-test market dates"
        )

    for name, lo, hi in [
        ("train", "2018-01-01", "2021-01-01"),
        ("val", "2021-01-01", "2022-01-01"),
        ("test", "2022-01-01", "2024-01-01"),
    ]:
        x = market[market.split.eq(name)]
        if x.empty:
            raise RuntimeError(f"Empty {name}")
        if (
            x.date.min() < pd.Timestamp(lo)
            or x.date.max() >= pd.Timestamp(hi)
        ):
            raise RuntimeError(
                f"Wrong {name} temporal split"
            )

    return v18, original, market, schema


def load_evidence():
    features = normalize_keys(
        read_frame(OLD / "retrieved_features.csv")
    )
    actions = normalize_keys(
        read_frame(OLD / "retrieval_actions.csv")
    )

    check_unique(
        features,
        ["date", "ticker"],
        "retrieved features",
    )
    check_unique(
        actions,
        ["date", "ticker"],
        "retrieval actions",
    )

    for action in ACTIONS:
        missing = [
            f"{action}_{x}"
            for x in RAG_COLS
            if f"{action}_{x}" not in features.columns
        ]
        if missing:
            raise RuntimeError(
                f"Missing {action} evidence columns: "
                f"{missing}"
            )

    for col in [
        f"DEFENSE_{x}" for x in RAG_COLS
    ]:
        if col not in features:
            raise RuntimeError(
                f"Missing defense feature: {col}"
            )

    if not set(
        actions.active_action.unique()
    ).issubset(set(ACTIONS)):
        raise RuntimeError(
            "Invalid existing actions"
        )

    return features, actions


def audit_evidence(market, out):
    audit = read_frame(
        OLD / "retrieval_audit.csv"
    )
    audit = normalize_keys(
        audit,
        "ticker",
    )

    audit["available_at"] = pd.to_datetime(
        audit["available_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    failures = (
        audit["available_at"].isna()
        | (
            audit["available_at"]
            >= audit["date"]
        )
    )

    invalid_ids = (
        audit["evidence_id"].isna()
        | audit["evidence_id"].astype(str).eq("")
    )

    bad_hashes = (
        audit["sha256"].isna()
        | ~audit["sha256"]
        .astype(str)
        .str.fullmatch("[0-9a-f]{64}")
    )

    summary = {
        "audited_evidence_rows": int(len(audit)),
        "date_gate_violations": int(failures.sum()),
        "missing_ids": int(invalid_ids.sum()),
        "invalid_sha256": int(bad_hashes.sum()),
        "date_gate_pass": not bool(failures.any()),
        "strict_pit_certified": False,
        "not_certified_reasons": [
            "Historical KG source availability is not "
            "independently verified against original publication.",
            "Legacy 9D/42D/53D feature provenance "
            "has not been independently recertified.",
            "Existing dynamic KG features may have "
            "train-period representation leakage.",
            "Existing V4 training-period policy predictions "
            "are not out-of-fold.",
            "SEC dates are checked at day granularity; "
            "intraday publication/execution ordering "
            "is not independently verified.",
        ],
    }

    if failures.any():
        audit.loc[failures].to_csv(
            out / "pit_violations.csv",
            index=False,
        )

    if (
        failures.any()
        or invalid_ids.any()
        or bad_hashes.any()
    ):
        raise RuntimeError(
            "Evidence integrity / PIT date gate failed"
        )

    write_json(
        out / "pit_audit.json",
        summary,
    )

    return summary


def load_policy_scores(market):
    states = read_frame(
        V4 / "historical_states.csv"
    )

    if "tic" not in states:
        raise RuntimeError(
            "V4 historical_states.csv missing tic"
        )

    states["ticker"] = (
        states["tic"].astype(str).str.upper()
    )

    states = normalize_keys(states)

    check_unique(
        states,
        ["date", "ticker"],
        "V4 historical states",
    )

    for col in POLICY_COLS:
        if col not in states:
            raise RuntimeError(
                f"Missing V4 policy feature: {col}"
            )

    frame = (
        market[["date", "ticker", "split"]]
        .merge(
            states[
                ["date", "ticker"] + POLICY_COLS
            ],
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
    )

    if frame[POLICY_COLS].isna().any().any():
        raise RuntimeError(
            "Missing history-only policy features"
        )

    X = frame[POLICY_COLS]

    scores = np.column_stack([
        joblib.load(
            V4 / f"policy_{action}.joblib"
        ).predict(X)
        for action in ACTIONS
    ])

    if not np.isfinite(scores).all():
        raise RuntimeError(
            "Nonfinite policy scores"
        )

    return frame, scores


def actions_at_lambda(scores, lam):
    costs = np.array(
        [PROXY_COST[a] for a in ACTIONS]
    )

    # Same V5 adjustment:
    # V4 reward already includes base action cost.
    adjusted = (
        scores
        - (float(lam) - 1.0) * costs[None, :]
    )

    indices = adjusted.argmax(axis=1)
    return np.array(ACTIONS)[indices]


def get_base_panel(market, schema, features, manifest):
    # Identical evidence scaling to original training.
    scale = np.array(
        manifest["evidence_scaler"],
        dtype=float,
    )

    if len(scale) != 16:
        raise RuntimeError(
            "Invalid original evidence scaler"
        )

    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise RuntimeError(
            "Nonfinite evidence scaler"
        )

    keys = market[
        ["date", "ticker"]
    ].copy()

    ev = keys.merge(
        features,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    if ev[f"SKIP_{RAG_COLS[0]}"].isna().any():
        raise RuntimeError(
            "Missing evidence rows"
        )

    base_features = schema["FULL_DYNAMIC"]

    def encode_action(action):
        cols = [
            f"{action}_{x}" for x in RAG_COLS
        ]
        arr = ev[cols].to_numpy(
            dtype=float
        )
        return np.clip(
            arr / scale[None, :],
            -5, 5,
        )

    blocks = {
        a: encode_action(a)
        for a in ACTIONS
    }

    blocks["DEFENSE"] = np.clip(
        ev[
            [f"DEFENSE_{x}" for x in RAG_COLS]
        ].to_numpy(float)
        / scale[None, :],
        -5, 5,
    )

    return base_features, blocks, scale


def make_panel(
    market, base_features, matrix
):
    if matrix.shape != (len(market), 16):
        raise RuntimeError(
            "Evidence matrix shape mismatch"
        )

    df = market.copy()

    for j, col in enumerate(RAG_COLS):
        df[col] = matrix[:, j]

    cols = base_features + RAG_COLS

    if not np.isfinite(
        df[cols].to_numpy(float)
    ).all():
        raise RuntimeError(
            "Nonfinite model observation"
        )

    return df, cols


def matrix_for_actions(blocks, decisions):
    out = np.zeros_like(
        blocks["SKIP"]
    )

    for action in ACTIONS:
        mask = decisions == action
        out[mask] = blocks[action][mask]

    return out


def group_shuffle(matrix, market, seed):
    """
    Shuffles evidence within the same ticker
    and split, without modifying market features.

    Diagnostic only: it is a negative control,
    not a historically deployable retrieval policy.
    """

    rng = np.random.default_rng(seed)

    shuffled = matrix.copy()
    work = market[
        ["ticker", "split"]
    ].copy()

    for _, idx in work.groupby(
        ["ticker", "split"]
    ).indices.items():
        ids = np.asarray(idx, dtype=int)
        perm = rng.permutation(ids)
        shuffled[ids] = matrix[perm]

    return shuffled


def lag_evidence(matrix, market, lag=20):
    """
    Apply 20-trading-day lag separately
    by ticker and split.

    This keeps different evaluation periods
    from borrowing test observations.
    """

    lagged = np.zeros_like(matrix)

    work = market[
        ["ticker", "split", "date"]
    ].copy()

    for _, idx in work.groupby(
        ["ticker", "split"]
    ).indices.items():
        ids = np.asarray(idx, dtype=int)
        order = np.argsort(
            work.iloc[ids]["date"].to_numpy()
        )
        ids = ids[order]

        if len(ids) > lag:
            lagged[ids[lag:]] = matrix[ids[:-lag]]

    return lagged


def synthetic_attack(matrix, decisions):
    """
    Explicit feature-level stress test.

    The attacker changes first accessible
    nonzero evidence component.

    This is NOT a realistic semantic attack
    and is NOT a replacement for poisoning
    raw SEC text before retrieval.
    """

    attacked = matrix.copy()
    changed = np.zeros(
        len(matrix),
        dtype=bool,
    )

    for i, action in enumerate(decisions):
        if action == "SKIP":
            continue

        allowed = (
            range(8) if action == "KG"
            else range(8, 16) if action == "SEC"
            else range(16)
        )

        allowed = list(allowed)
        if not allowed:
            continue

        j = allowed[0]
        old = attacked[i, j]

        attacked[i, j] = (
            -5.0 if old > 0 else 5.0
        )

        changed[i] = True

    return attacked, changed


def model_path(seed, agent):
    return require(
        OLD
        / "runs"
        / f"DYNAMIC_ACTIVE__{agent}__seed{seed}"
        / "model.zip"
    )


def load_model(seed, agent, device):
    path = model_path(seed, agent)
    return MODEL_CLASSES[agent].load(
        str(path),
        device=device,
    )


def env_args(args):
    return SimpleNamespace(
        transaction_cost=args.transaction_cost,
        reward_scaling=args.reward_scaling,
        learning_rate=3e-4,
        verbose=0,
    )


def replay(
    model,
    frame,
    cols,
    v18,
    original,
    args,
    split,
):
    portion = frame[
        frame.split.eq(split)
    ].copy()

    if portion.empty:
        raise RuntimeError(
            f"Empty replay split: {split}"
        )

    env = v18.create_env(
        original,
        portion,
        cols,
        env_args(args),
    )

    metrics, equity, weights, returns = (
        original.evaluate_model(
            model,
            env,
        )
    )

    metrics = {
        k: (
            v.item() if isinstance(v, np.generic)
            else v
        )
        for k, v in metrics.items()
    }

    # Numeric allocations: remove date/time fields.
    if hasattr(weights, "copy"):
        wd = pd.DataFrame(weights).copy()
    else:
        wd = pd.DataFrame(weights)

    numeric = wd.select_dtypes(
        include=[np.number]
    ).copy()

    for col in numeric.columns:
        if (
            "date" in str(col).lower()
            or "index" == str(col).lower()
        ):
            numeric = numeric.drop(
                columns=col
            )

    weight_array = numeric.to_numpy(
        dtype=float
    )

    if weight_array.size == 0:
        raise RuntimeError(
            "No numeric weights in evaluation"
        )

    if not np.isfinite(weight_array).all():
        raise RuntimeError(
            "Nonfinite allocation weights"
        )

    return metrics, weight_array


def mean_weight_difference(a, b):
    if a.shape != b.shape:
        raise RuntimeError(
            "Weight array shape mismatch"
        )

    return float(
        np.abs(a - b).mean()
    )


def average_cost(decisions):
    return float(np.mean(
        [PROXY_COST[a] for a in decisions]
    ))


def query_rate(decisions):
    return float(
        np.mean(decisions != "SKIP")
    )


def action_distribution(decisions):
    total = max(1, len(decisions))
    return {
        a: float(
            np.sum(decisions == a) / total
        )
        for a in ACTIONS
    }


def action_by_split(market, scores, lam):
    decisions = actions_at_lambda(
        scores, lam
    )

    if len(decisions) != len(market):
        raise RuntimeError(
            "Policy dimension mismatch"
        )

    return decisions


def baseline_validation_selection(
    results, out, metric
):
    selected = select_models(
        results,
        metric,
    )

    selected.to_csv(
        out / "2021_selected_agents.csv",
        index=False,
    )

    print("\n2021-selected portfolio models:")
    print(selected.to_string(index=False))

    return selected


def perform_lambda_sweep(
    selected, market, scores, blocks,
    base_features, v18, original,
    args, out,
):
    """
    Optimize lambda on 2021 validation only.

    For every candidate, replay the SAME
    pretrained DYNAMIC_ACTIVE agents.

    Objective:
       2021 validation Sharpe
       - cost_weight * average cost proxy

    Cost proxy is not an API fee.
    """

    rows = []
    model_cache = {}

    for sel in selected.itertuples(index=False):
        model_cache[int(sel.seed)] = load_model(
            int(sel.seed),
            str(sel.agent),
            args.device,
        )

    for lam in LAMBDA_GRID:
        decision = action_by_split(
            market, scores, lam
        )

        matrix = matrix_for_actions(
            blocks, decision
        )

        panel, cols = make_panel(
            market,
            base_features,
            matrix,
        )

        val_mask = (
            market.split.eq("val")
        ).to_numpy()

        val_actions = decision[val_mask]
        avg_cost = average_cost(val_actions)

        for sel in selected.itertuples(index=False):
            seed = int(sel.seed)
            model = model_cache[seed]

            metric, _ = replay(
                model,
                panel,
                cols,
                v18,
                original,
                args,
                "val",
            )

            rows.append({
                "lambda": lam,
                "seed": seed,
                "agent": sel.agent,
                "val_return": float(
                    metric["total_return"]
                ),
                "val_sharpe": float(
                    metric["sharpe"]
                ),
                "avg_proxy_cost": avg_cost,
                "query_rate": query_rate(
                    val_actions
                ),
                **{
                    f"share_{a}": v
                    for a, v in action_distribution(
                        val_actions
                    ).items()
                },
            })

        print(
            "[VAL SWEEP]",
            "lambda", lam,
            "query_rate", round(
                query_rate(val_actions), 4
            ),
            flush=True,
        )

    raw = pd.DataFrame(rows)
    raw.to_csv(
        out / "validation_lambda_runs.csv",
        index=False,
    )

    summary = (
        raw.groupby("lambda")
        .agg(
            val_return=("val_return", "mean"),
            val_sharpe=("val_sharpe", "mean"),
            avg_proxy_cost=("avg_proxy_cost", "mean"),
            query_rate=("query_rate", "mean"),
            kg_share=("share_KG", "mean"),
            sec_share=("share_SEC", "mean"),
            both_share=("share_BOTH", "mean"),
            skip_share=("share_SKIP", "mean"),
        )
        .reset_index()
    )

    summary["objective"] = (
        summary.val_sharpe
        - args.cost_weight
        * summary.avg_proxy_cost
    )

    # Deterministic tie-break: lower cost,
    # then smaller lambda.
    summary = summary.sort_values(
        [
            "objective",
            "avg_proxy_cost",
            "lambda",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    )

    summary.to_csv(
        out / "validation_lambda_summary.csv",
        index=False,
    )

    best = summary.iloc[0]

    print("\nVALIDATION SELECTED LAMBDA")
    print(
        best.to_string()
    )

    return (
        float(best["lambda"]),
        model_cache,
    )


def policy_test(
    selected, models,
    market, scores, blocks,
    base_features, v18, original,
    args, out, best_lambda
):
    rows = []

    old_cfg = json.loads(
        require(V5 / "metadata.json").read_text()
    )
    old_lambda = float(
        old_cfg["selected_lambda"]
    )

    schedules = {
        "Never": np.array(
            ["SKIP"] * len(market)
        ),
        "Always_BOTH": np.array(
            ["BOTH"] * len(market)
        ),
        "Existing_V5": action_by_split(
            market, scores, old_lambda
        ),
        "Cost_Calibrated": action_by_split(
            market, scores, best_lambda
        ),
    }

    # Matched-budget random control:
    # shuffle selected actions separately
    # inside each time period.
    rng = np.random.default_rng(
        args.random_seed
    )

    selected_actions = schedules[
        "Cost_Calibrated"
    ]

    random_actions = selected_actions.copy()

    for split, ids in market.groupby(
        "split"
    ).indices.items():
        ids = np.asarray(ids, dtype=int)
        random_actions[ids] = rng.permutation(
            selected_actions[ids]
        )

    schedules[
        "Random_MatchedBudget"
    ] = random_actions

    action_rows = []

    for policy, decision in schedules.items():
        for i, (date, ticker, split) in enumerate(
            market[
                ["date", "ticker", "split"]
            ].itertuples(index=False, name=None)
        ):
            action_rows.append({
                "policy": policy,
                "date": date,
                "ticker": ticker,
                "split": split,
                "action": decision[i],
                "proxy_cost": PROXY_COST[
                    decision[i]
                ],
            })

        matrix = matrix_for_actions(
            blocks, decision
        )

        panel, cols = make_panel(
            market,
            base_features,
            matrix,
        )

        test_mask = (
            market.split.eq("test")
        ).to_numpy()

        test_decision = decision[test_mask]

        for sel in selected.itertuples(index=False):
            model = models[int(sel.seed)]

            metrics, weights = replay(
                model,
                panel,
                cols,
                v18,
                original,
                args,
                "test",
            )

            rows.append({
                "policy": policy,
                "seed": int(sel.seed),
                "agent": sel.agent,
                "total_return": float(
                    metrics["total_return"]
                ),
                "sharpe": float(
                    metrics["sharpe"]
                ),
                "max_drawdown": float(
                    metrics["max_drawdown"]
                ),
                "query_rate": query_rate(
                    test_decision
                ),
                "avg_proxy_cost": average_cost(
                    test_decision
                ),
                **{
                    f"share_{a}": v
                    for a, v in action_distribution(
                        test_decision
                    ).items()
                },
            })

        print(
            "[TEST POLICY]", policy,
            flush=True,
        )

    action_df = pd.DataFrame(action_rows)
    action_df.to_csv(
        out / "all_policy_actions.csv",
        index=False,
    )

    df = pd.DataFrame(rows)
    df.to_csv(
        out / "heldout_policy_results.csv",
        index=False,
    )

    summary = (
        df.groupby("policy")
        .agg(
            return_mean=("total_return", "mean"),
            return_std=("total_return", "std"),
            sharpe_mean=("sharpe", "mean"),
            sharpe_std=("sharpe", "std"),
            mdd_mean=("max_drawdown", "mean"),
            query_rate=("query_rate", "mean"),
            avg_proxy_cost=("avg_proxy_cost", "mean"),
            sec_share=("share_SEC", "mean"),
            both_share=("share_BOTH", "mean"),
        )
        .reset_index()
    )

    summary["illustrative_net_return"] = (
        summary.return_mean
        - args.cost_return_equivalent
        * summary.avg_proxy_cost
    )

    summary.to_csv(
        out / "heldout_policy_summary.csv",
        index=False,
    )

    paired_rows = []
    baseline = df[
        df.policy.eq("Cost_Calibrated")
    ].set_index("seed")

    for policy in df.policy.unique():
        if policy == "Cost_Calibrated":
            continue

        other = df[
            df.policy.eq(policy)
        ].set_index("seed")

        for seed in baseline.index.intersection(
            other.index
        ):
            paired_rows.append({
                "seed": int(seed),
                "baseline": policy,
                "delta_return": (
                    float(
                        baseline.loc[
                            seed, "total_return"
                        ]
                    )
                    - float(
                        other.loc[
                            seed, "total_return"
                        ]
                    )
                ),
                "delta_sharpe": (
                    float(
                        baseline.loc[
                            seed, "sharpe"
                        ]
                    )
                    - float(
                        other.loc[
                            seed, "sharpe"
                        ]
                    )
                ),
                "delta_mdd": (
                    float(
                        baseline.loc[
                            seed, "max_drawdown"
                        ]
                    )
                    - float(
                        other.loc[
                            seed, "max_drawdown"
                        ]
                    )
                ),
            })

    pd.DataFrame(paired_rows).to_csv(
        out / "policy_paired_differences.csv",
        index=False,
    )

    return schedules, df


def reliance_tests(
    selected, models,
    market, blocks,
    base_features, v18, original,
    args, out, decisions
):
    """
    Keep original trained DYNAMIC_ACTIVE agent
    unchanged and intervene on evidence only.

    Real, Zero, Lagged, Shuffled
    are evaluation-time perturbations.
    """

    real = matrix_for_actions(
        blocks,
        decisions,
    )

    zero = np.zeros_like(real)

    lagged = lag_evidence(
        real,
        market,
        lag=args.lag,
    )

    shuffled = group_shuffle(
        real,
        market,
        args.random_seed,
    )

    variants = {
        "REAL": real,
        "ZERO": zero,
        "LAGGED": lagged,
        "SHUFFLED": shuffled,
    }

    records = []
    reference_weights = {}

    # REAL first establishes the reference.
    for variant, matrix in variants.items():
        panel, cols = make_panel(
            market,
            base_features,
            matrix,
        )

        for sel in selected.itertuples(index=False):
            seed = int(sel.seed)
            model = models[seed]

            metrics, weights = replay(
                model, panel, cols,
                v18, original, args,
                "test",
            )

            if variant == "REAL":
                reference_weights[seed] = weights
                delta_weights = 0.0
            else:
                delta_weights = (
                    mean_weight_difference(
                        reference_weights[seed],
                        weights,
                    )
                )

            records.append({
                "variant": variant,
                "seed": seed,
                "agent": sel.agent,
                "total_return": float(
                    metrics["total_return"]
                ),
                "sharpe": float(
                    metrics["sharpe"]
                ),
                "max_drawdown": float(
                    metrics["max_drawdown"]
                ),
                "mean_abs_weight_change":
                    delta_weights,
            })

        print(
            "[RELIANCE]", variant,
            flush=True,
        )

    df = pd.DataFrame(records)
    df.to_csv(
        out / "rag_reliance_results.csv",
        index=False,
    )

    summary = (
        df.groupby("variant")
        .agg(
            return_mean=("total_return", "mean"),
            sharpe_mean=("sharpe", "mean"),
            mdd_mean=("max_drawdown", "mean"),
            mean_weight_change=(
                "mean_abs_weight_change",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "rag_reliance_summary.csv",
        index=False,
    )

    return df


def defense_tests(
    selected, models,
    market, blocks,
    base_features, v18, original,
    args, out, decisions
):
    clean = matrix_for_actions(
        blocks,
        decisions,
    )

    attacked, changed = synthetic_attack(
        clean,
        decisions,
    )

    # Existing original clean manifest plus
    # filtering of synthetically tampered
    # top-1 evidence. This is not semantic defense.
    defense = clean.copy()

    active_defense = blocks["DEFENSE"]

    defense[changed] = (
        active_defense[changed]
    )

    variants = {
        "CLEAN": clean,
        "ATTACK_NO_DEFENSE": attacked,
        "ATTACK_WITH_DEFENSE": defense,
    }

    records = []

    for scenario, matrix in variants.items():
        panel, cols = make_panel(
            market,
            base_features,
            matrix,
        )

        for sel in selected.itertuples(index=False):
            model = models[int(sel.seed)]

            metrics, weights = replay(
                model,
                panel,
                cols,
                v18,
                original,
                args,
                "test",
            )

            records.append({
                "scenario": scenario,
                "seed": int(sel.seed),
                "agent": sel.agent,
                "total_return": float(
                    metrics["total_return"]
                ),
                "sharpe": float(
                    metrics["sharpe"]
                ),
                "max_drawdown": float(
                    metrics["max_drawdown"]
                ),
                "attacked_state_count": int(
                    changed[
                        market.split.eq("test")
                        .to_numpy()
                    ].sum()
                ),
            })

        print(
            "[DEFENSE]", scenario,
            flush=True,
        )

    df = pd.DataFrame(records)
    df.to_csv(
        out / "defense_threeway_results.csv",
        index=False,
    )

    summary = (
        df.groupby("scenario")
        .agg(
            return_mean=("total_return", "mean"),
            sharpe_mean=("sharpe", "mean"),
            mdd_mean=("max_drawdown", "mean"),
            attacked_states=(
                "attacked_state_count",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "defense_threeway_summary.csv",
        index=False,
    )

    return df


def save_research_report(
    out,
    audit,
    best_lambda,
    selected,
    args,
):
    policy = read_frame(
        out / "heldout_policy_summary.csv"
    )
    reliance = read_frame(
        out / "rag_reliance_summary.csv"
    )
    defense = read_frame(
        out / "defense_threeway_summary.csv"
    )

    text = [
        "# AEGIS-RL Final Research Audit",
        "",
        "## Experimental design",
        "",
        "- Portfolio training: 2018–2020",
        "- Validation/model selection: 2021",
        "- Held-out evaluation: 2022–2023",
        "- Original models reused; no retraining",
        "- Original Dynamic KG feature dimensions preserved",
        "",
        "## Active policy calibration",
        "",
        f"- Selected lambda: {best_lambda}",
        f"- Validation cost weight: {args.cost_weight}",
        "- Actions: SKIP, KG, SEC, BOTH",
        "- Cost numbers are normalized proxies",
        "- Policy calibrated on 2021 only",
        "- 2022–2023 is reserved for evaluation",
        "",
        policy.to_markdown(index=False),
        "",
        "## RAG evidence reliance",
        "",
        reliance.to_markdown(index=False),
        "",
        "## Synthetic defense controls",
        "",
        defense.to_markdown(index=False),
        "",
        "## PIT audit",
        "",
        f"- Date gate pass: {audit['date_gate_pass']}",
        f"- Strict PIT certified: "
        f"{audit['strict_pit_certified']}",
        "",
        "## Interpretive limitations",
        "",
        "- Replayed retrieval policies are evaluated "
        "using agents originally trained under V5.",
        "- A policy selected by validation does not "
        "guarantee better test performance.",
        "- Evidence interventions can create "
        "out-of-distribution observations.",
        "- Mean allocation change establishes "
        "sensitivity, not causal economic value.",
        "- Evidence cost is a proxy, not real billing.",
        "- Defense is synthetic feature-level corruption, "
        "not a full adversarial SEC-text benchmark.",
        "- Original model training and source data "
        "need independent full PIT certification.",
        "- These results are exploratory, "
        "not a final causal proof.",
        "",
    ]

    (out / "RESEARCH_REPORT.md").write_text(
        "\n".join(text),
        encoding="utf-8",
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--tier",
        default="conservative",
        choices=["conservative", "inclusive"],
    )

    p.add_argument(
        "--out",
        default=str(OUT_DEFAULT),
    )

    p.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="cpu",
    )

    p.add_argument(
        "--selection-metric",
        choices=["sharpe", "return"],
        default="sharpe",
    )

    p.add_argument(
        "--cost-weight",
        type=float,
        default=2.0,
        help="Dimensionless validation Sharpe penalty "
             "per normalized retrieval cost",
    )

    p.add_argument(
        "--cost-return-equivalent",
        type=float,
        default=0.0,
        help="Illustrative portfolio-return penalty "
             "per mean normalized retrieval cost; "
             "0 = do not invent monetary costs",
    )

    p.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
    )

    p.add_argument(
        "--lag",
        type=int,
        default=20,
    )

    p.add_argument(
        "--random-seed",
        type=int,
        default=2026,
    )

    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("AEGIS-RL FINAL AUDIT")
    print("=" * 72)

    for path in [
        SCRIPT,
        OLD / "all_results.csv",
        OLD / "retrieved_features.csv",
        OLD / "retrieval_actions.csv",
        OLD / "retrieval_audit.csv",
        OLD / "manifest.json",
        V4 / "historical_states.csv",
        V5 / "metadata.json",
    ]:
        require(path)

    aegis = load_module(
        SCRIPT,
        "aegis_portfolio_original",
    )

    results = load_completed_runs()

    manifest = json.loads(
        (OLD / "manifest.json").read_text()
    )

    if manifest["tier"] != args.tier:
        raise RuntimeError(
            "Tier differs from original training manifest"
        )

    if (
        abs(
            float(manifest["transaction_cost"])
            - args.transaction_cost
        ) > 1e-12
    ):
        raise RuntimeError(
            "Transaction cost mismatch"
        )

    if (
        abs(
            float(manifest["reward_scaling"])
            - args.reward_scaling
        ) > 1e-12
    ):
        raise RuntimeError(
            "Reward scaling mismatch"
        )

    v18, original, market, schema = (
        load_market_and_helpers(
            aegis, args.tier
        )
    )

    evidence, original_actions = (
        load_evidence()
    )

    audit = audit_evidence(
        market, out
    )

    states, scores = load_policy_scores(
        market
    )

    # Key order must be identical.
    a = market[["date", "ticker"]].reset_index(drop=True)
    b = states[["date", "ticker"]].reset_index(drop=True)

    if not a.equals(b):
        raise RuntimeError(
            "Market and policy state key order mismatch"
        )

    base_cols, blocks, scales = get_base_panel(
        market,
        schema,
        evidence,
        manifest,
    )

    # Reproduce original V5 actions on held-out dates.
    v5_cfg = json.loads(
        (V5 / "metadata.json").read_text()
    )

    old_lambda = float(
        v5_cfg["selected_lambda"]
    )

    replayed_v5 = actions_at_lambda(
        scores,
        old_lambda,
    )

    original_actions = (
        market[["date", "ticker"]]
        .merge(
            original_actions[
                ["date", "ticker", "active_action"]
            ],
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
    )

    test_mask = (
        market.split.eq("test")
    ).to_numpy()

    mismatch = (
        replayed_v5[test_mask]
        != original_actions.loc[
            test_mask,
            "active_action",
        ].to_numpy()
    )

    if mismatch.any():
        raise RuntimeError(
            "Original test V5 policy actions "
            "do not match replayed model: "
            f"{int(mismatch.sum())} mismatches"
        )

    print(
        "Original V5 action reproduction: PASS"
    )

    selected = baseline_validation_selection(
        results,
        out,
        args.selection_metric,
    )

    best_lambda, models = perform_lambda_sweep(
        selected,
        market,
        scores,
        blocks,
        base_cols,
        v18,
        original,
        args,
        out,
    )

    selected_actions = action_by_split(
        market,
        scores,
        best_lambda,
    )

    schedules, test_results = policy_test(
        selected,
        models,
        market,
        scores,
        blocks,
        base_cols,
        v18,
        original,
        args,
        out,
        best_lambda,
    )

    reliance = reliance_tests(
        selected,
        models,
        market,
        blocks,
        base_cols,
        v18,
        original,
        args,
        out,
        selected_actions,
    )

    defense = defense_tests(
        selected,
        models,
        market,
        blocks,
        base_cols,
        v18,
        original,
        args,
        out,
        selected_actions,
    )

    write_json(
        out / "audit_manifest.json",
        {
            "research_status":
                "EXPLORATORY_COUNTERFACTUAL_REPLAY",
            "portfolio_models_retrained": False,
            "base_feature_dimensions": 53,
            "evidence_dimensions": 16,
            "time_splits": {
                "train": "2018-2020",
                "validation": "2021",
                "test": "2022-2023",
            },
            "selected_lambda_2021": best_lambda,
            "selected_agents": selected.to_dict(
                orient="records"
            ),
            "cost_weight": args.cost_weight,
            "cost_return_equivalent":
                args.cost_return_equivalent,
            "model_source": str(OLD),
            "source_manifest_sha256": sha_file(
                OLD / "manifest.json"
            ),
            "strict_pit_certified": False,
            "date_gate_pass": audit["date_gate_pass"],
            "defense_scenario":
                "SYNTHETIC_FEATURE_CORRUPTION",
            "not_jointly_trained": True,
        },
    )

    save_research_report(
        out, audit,
        best_lambda, selected,
        args,
    )

    print("\n" + "=" * 72)
    print("FINAL AUDIT SUMMARY")
    print("=" * 72)

    print("\nRetrieval policies:")
    print(
        read_frame(
            out / "heldout_policy_summary.csv"
        ).to_string(index=False)
    )

    print("\nRAG reliance:")
    print(
        read_frame(
            out / "rag_reliance_summary.csv"
        ).to_string(index=False)
    )

    print("\nDefense:")
    print(
        read_frame(
            out / "defense_threeway_summary.csv"
        ).to_string(index=False)
    )

    print("\nSaved to:", out)
    print("AEGIS FINAL AUDIT: PASS")


if __name__ == "__main__":
    main()
