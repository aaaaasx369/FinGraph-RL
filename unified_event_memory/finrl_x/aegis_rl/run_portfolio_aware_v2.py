#!/usr/bin/env python3
"""
AEGIS-RL Portfolio-Aware Retrieval V2

1. Reuse historical KG/SEC evidence.
2. Improve evidence relevance using historical retrieval scores.
3. Generate four-action portfolio utility targets.
4. Fit a cost-sensitive contextual bandit on 2018-2020.
5. Select cost parameter on 2021 validation.
6. Freeze policy before 2022-2023.
7. Train new portfolio agents using the frozen policy.
8. Compare NEVER, ALWAYS_BOTH, RANDOM, ACTIVE.
9. Save full data, actions, models, and metrics.

Research classification:
Model-based offline contextual bandit.

NOT:
- joint end-to-end retrieval/portfolio RL
- causal identification of retrieval value
- automatically PIT-certified

No 2022-2023 outcomes are used for
policy fitting or hyperparameter selection.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from stable_baselines3 import A2C, PPO, DDPG

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

OLD = ROOT / "results/aegis_rl_v1/evidence_rag_portfolio"
OUT_DEFAULT = ROOT / "results/aegis_rl_v1/portfolio_aware_v2"

AEGIS_SCRIPT = (
    ROOT / "unified_event_memory/finrl_x/aegis_rl"
    / "run_aegis_evidence_portfolio.py"
)

ACTIONS = ["SKIP", "KG", "SEC", "BOTH"]
ALGORITHMS = ["A2C", "PPO", "DDPG"]
SEEDS = [42, 123, 2026, 7, 999]

COSTS = np.array(
    [0.0, 0.03, 0.08, 0.11],
    dtype=np.float64,
)

RAG_COLS = (
    [f"rag_kg_{i}" for i in range(8)]
    + [f"rag_sec_{i}" for i in range(8)]
)

LAMBDA_GRID = [
    0.0, 0.1, 0.5, 1.0,
    2.0, 4.0, 8.0, 16.0,
]

POLICY_NAMES = [
    "NEVER",
    "ALWAYS_BOTH",
    "RANDOM_MATCHED",
    "ACTIVE_V2",
]

MODEL_MAP = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            default=str,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def file_sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(block)

    return h.hexdigest()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def args_for_env(args):
    return SimpleNamespace(
        transaction_cost=args.transaction_cost,
        reward_scaling=args.reward_scaling,
        learning_rate=args.learning_rate,
        verbose=0,
    )


def normalize_dates(df):
    out = df.copy()

    out["date"] = pd.to_datetime(
        out["date"],
        errors="raise",
        utc=True,
    ).dt.tz_convert(None).dt.normalize()

    out["ticker"] = (
        out["ticker"].astype(str).str.upper()
    )

    return out


def read_original_data(args):
    if not AEGIS_SCRIPT.exists():
        raise FileNotFoundError(AEGIS_SCRIPT)

    mod = load_module(
        AEGIS_SCRIPT,
        "aegis_evidence_v1",
    )

    v21, v18, original = (
        mod.load_original_helpers()
    )

    schema, dynamic_extra = mod.load_schema()

    market = mod.load_market(
        args.tier,
        schema,
        dynamic_extra,
    )

    market = normalize_dates(market)

    evidence = pd.read_csv(
        OLD / "retrieved_features.csv"
    )
    evidence = normalize_dates(evidence)

    manifest = json.loads(
        (OLD / "manifest.json").read_text()
    )

    if manifest["tier"] != args.tier:
        raise RuntimeError("Input tier mismatch")

    for k, v in [
        ("transaction_cost", args.transaction_cost),
        ("reward_scaling", args.reward_scaling),
    ]:
        if not np.isclose(
            float(manifest[k]), float(v)
        ):
            raise RuntimeError(
                f"Original environment parameter differs: {k}"
            )

    if market.duplicated(
        ["date", "ticker"]
    ).any():
        raise RuntimeError(
            "Duplicate market keys"
        )

    if evidence.duplicated(
        ["date", "ticker"]
    ).any():
        raise RuntimeError(
            "Duplicate evidence keys"
        )

    market = market.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    evidence = market[
        ["date", "ticker"]
    ].merge(
        evidence,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )

    if evidence[
        "SKIP_" + RAG_COLS[0]
    ].isna().any():
        raise RuntimeError(
            "Missing evidence dates/tickers"
        )

    return (
        v18,
        original,
        schema,
        market,
        evidence,
        manifest,
    )


def audit_provenance(market, out):
    path = OLD / "retrieval_audit.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    audit = pd.read_csv(path)
    audit = normalize_dates(audit)

    audit["available_at"] = pd.to_datetime(
        audit["available_at"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    invalid_time = (
        audit["available_at"].isna()
        | audit["available_at"].ge(
            audit["date"]
        )
    )

    invalid_hash = (
        ~audit["sha256"]
        .astype(str)
        .str.fullmatch("[0-9a-f]{64}")
    )

    invalid_id = (
        audit["evidence_id"].isna()
        | audit["evidence_id"]
        .astype(str)
        .eq("")
    )

    report = {
        "audit_rows": int(len(audit)),
        "date_violations": int(invalid_time.sum()),
        "hash_violations": int(invalid_hash.sum()),
        "id_violations": int(invalid_id.sum()),
        "strict_pit_certified": False,
        "notes": [
            "Day-level available_at gate only.",
            "Original KG embedding provenance not fully audited.",
            "Original 9D/42D/53D features not recertified.",
            "Teacher models use historical training labels.",
            "Train-period teacher signals are in-sample.",
            "Timestamp granularity may miss intraday execution leakage.",
        ],
    }

    save_json(
        out / "provenance_audit.json",
        report,
    )

    if any([
        invalid_time.any(),
        invalid_hash.any(),
        invalid_id.any(),
    ]):
        raise RuntimeError(
            "Evidence provenance audit failed"
        )

    return audit


def construct_evidence(
    evidence,
    audit,
    manifest,
    market,
):
    """
    Fixed 16D evidence encoder enhancement.

    Reweights existing KG and SEC features by
    retrieved-source relevance and freshness.

    We do not change model input dimension.

    This prevents adding an untrained 24D block
    to an existing 16D portfolio model.
    """

    key = ["date", "ticker"]

    a = audit.copy()

    a["score"] = (
        pd.to_numeric(
            a["score"], errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    a["age_days"] = (
        a["date"]
        - a["available_at"]
    ).dt.total_seconds() / 86400.0

    a["fresh_score"] = (
        np.exp(
            -a["age_days"].clip(lower=0)
            / 180.0
        )
    )

    a["relevance"] = (
        a["score"] * a["fresh_score"]
    )

    aggregate = (
        a.groupby(key + ["source"])
        .agg(
            relevance=("relevance", "mean"),
            evidence_count=("evidence_id", "nunique"),
        )
        .reset_index()
    )

    meta = market[key].copy()

    for source in ["KG", "SEC"]:
        sub = aggregate[
            aggregate.source.eq(source)
        ][key + ["relevance", "evidence_count"]]

        sub = sub.rename(columns={
            "relevance": f"rel_{source}",
            "evidence_count": f"count_{source}",
        })

        meta = meta.merge(
            sub,
            on=key,
            how="left",
            validate="one_to_one",
        )

    for col in [
        "rel_KG",
        "rel_SEC",
        "count_KG",
        "count_SEC",
    ]:
        meta[col] = (
            meta[col]
            .fillna(0)
            .astype(float)
        )

    raw_scale = np.array(
        manifest["evidence_scaler"],
        dtype=float,
    )

    if (
        len(raw_scale) != 16
        or (raw_scale <= 0).any()
        or not np.isfinite(raw_scale).all()
    ):
        raise RuntimeError(
            "Invalid original evidence scale"
        )

    # Original learned portfolio models expect
    # original 16D scales, not newly fitted ones.
    # New agents will be trained under these same
    # fixed scales for comparability.
    blocks = {}

    for action in ACTIONS:
        cols = [
            f"{action}_{c}"
            for c in RAG_COLS
        ]

        values = evidence[cols].to_numpy(
            dtype=float
        ).copy()

        # Relevance weighting is causal in the
        # sense it uses only retrieved past evidence.
        # It does not use future stock returns.
        kg_rel = (
            meta["rel_KG"].to_numpy()
        )
        sec_rel = (
            meta["rel_SEC"].to_numpy()
        )

        # Stable nonnegative bounded weights.
        wkg = np.clip(
            0.5 + kg_rel,
            0.5,
            1.5,
        )

        wsec = np.clip(
            0.5 + sec_rel,
            0.5,
            1.5,
        )

        values[:, :8] *= wkg[:, None]
        values[:, 8:] *= wsec[:, None]

        # Preserve original evidence normalization.
        blocks[action] = np.clip(
            values / raw_scale,
            -5,
            5,
        ).astype(np.float32)

    blocks["SKIP"][:] = 0.0

    if any(
        not np.isfinite(x).all()
        for x in blocks.values()
    ):
        raise RuntimeError(
            "Nonfinite relevance-encoded evidence"
        )

    return blocks, meta


def build_context(market, meta):
    """
    Date-level predecision context.

    5-company summary of historical evidence
    availability/relevance and lagged returns.

    The action is one of four actions per
    portfolio rebalance day, applying to all
    five stocks simultaneously.
    """

    work = market[
        [
            "date",
            "ticker",
            "split",
            "market_close",
        ]
    ].merge(
        meta,
        on=["date", "ticker"],
        validate="one_to_one",
    )

    work = work.sort_values(
        ["ticker", "date"]
    )

    work["prior_return"] = (
        work.groupby("ticker")[
            "market_close"
        ].pct_change()
    ).fillna(0.0)

    work["prior_abs_return"] = (
        work["prior_return"].abs()
    )

    work["prior_vol_20"] = (
        work.groupby("ticker")[
            "prior_return"
        ].transform(
            lambda x: x.rolling(
                20,
                min_periods=2,
            ).std()
        )
    ).fillna(0.0)

    columns = [
        "rel_KG",
        "rel_SEC",
        "count_KG",
        "count_SEC",
        "prior_return",
        "prior_abs_return",
        "prior_vol_20",
    ]

    aggregate = (
        work.groupby("date")[columns]
        .agg(["mean", "std", "max"])
    )

    aggregate.columns = [
        f"{a}_{b}"
        for a, b in aggregate.columns
    ]

    aggregate = aggregate.reset_index()

    aggregate = aggregate.sort_values(
        "date"
    ).reset_index(drop=True)

    X = aggregate.drop(
        columns=["date"]
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)

    return (
        aggregate["date"],
        X,
    )


def make_feature_panel(
    market,
    base_cols,
    matrix,
):
    if matrix.shape != (
        len(market),
        len(RAG_COLS),
    ):
        raise RuntimeError(
            "Evidence tensor shape mismatch"
        )

    panel = market.copy()

    for j, col in enumerate(RAG_COLS):
        panel[col] = matrix[:, j]

    return (
        panel,
        base_cols + RAG_COLS,
    )


def action_matrix(
    blocks,
    action_for_date,
    market,
):
    dates = market["date"].to_numpy()

    if isinstance(action_for_date, pd.Series):
        action_for_date = action_for_date.to_dict()

    selected = np.array([
        action_for_date[pd.Timestamp(x)]
        for x in dates
    ])

    out = np.zeros_like(
        blocks["SKIP"]
    )

    for action in ACTIONS:
        mask = selected == action
        out[mask] = blocks[action][mask]

    return out


def make_env(
    v18,
    original,
    panel,
    cols,
    args,
):
    return v18.create_env(
        original,
        panel,
        cols,
        args_for_env(args),
    )


def load_teacher():
    """
    A2C seed 42 is a fixed pretrained teacher.

    This is a model-based teacher, not
    ground-truth causal retrieval utility.
    """
    path = (
        OLD
        / "runs/DYNAMIC_ACTIVE__A2C__seed42/model.zip"
    )

    if not path.exists():
        raise FileNotFoundError(path)

    return A2C.load(
        str(path),
        device="cpu",
    )


def teacher_action_targets(
    teacher,
    v18,
    original,
    market,
    base_cols,
    blocks,
    args,
):
    """
    For each historical date and retrieval action:

    Build the SAME 5-stock observation state.
    Predict portfolio weights using fixed teacher.
    Calculate realized NEXT-DAY portfolio return.

    The next-day return is a TRAINING LABEL only.
    It is NEVER included in the policy features.

    This is a one-step counterfactual estimate,
    NOT a full sequential off-policy causal value.
    """

    date_order = sorted(
        market.date.unique()
    )

    all_rows = []

    # Construct one observation environment per
    # action, once, rather than per trading date.
    envs = {}

    for action in ACTIONS:
        panel, cols = make_feature_panel(
            market,
            base_cols,
            blocks[action],
        )

        envs[action] = make_env(
            v18,
            original,
            panel,
            cols,
            args,
        )

    reference = envs["SKIP"]

    for day_index in range(
        len(date_order) - 1
    ):
        day = pd.Timestamp(
            date_order[day_index]
        )

        next_returns = (
            reference.price_matrix[
                day_index + 1
            ]
            /
            reference.price_matrix[
                day_index
            ]
            - 1.0
        )

        row = {"date": day}

        for action in ACTIONS:
            env = envs[action]

            # Use the SAME prior allocation,
            # for all four counterfactuals.
            prev_weights = np.ones(5) / 5.0

            observation = np.concatenate([
                env.feature_tensor[
                    day_index
                ].reshape(-1),
                prev_weights,
            ]).astype(np.float32)

            predicted_action, _ = teacher.predict(
                observation,
                deterministic=True,
            )

            target_weights = (
                original.PortfolioEnv
                .action_to_weights(
                    predicted_action
                )
            )

            turnover_cost = (
                args.transaction_cost
                *
                np.abs(
                    target_weights - prev_weights
                ).sum()
            )

            next_return = (
                np.dot(
                    target_weights,
                    next_returns,
                )
                - turnover_cost
            )

            row[action] = float(
                next_return
            )

        all_rows.append(row)

    labels = pd.DataFrame(
        all_rows
    )

    # Training data must not look into validation
    # outcomes, even through a last-day label.
    labels["next_date"] = labels[
        "date"
    ].map({
        pd.Timestamp(date_order[i]):
            pd.Timestamp(date_order[i + 1])
        for i in range(len(date_order) - 1)
    })

    return labels


def policy_model(alpha):
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "regressor",
            Ridge(alpha=alpha),
        ),
    ])


def train_bandit(
    contexts,
    labels,
    args,
    out,
):
    frame = contexts.merge(
        labels,
        on="date",
        how="left",
        validate="one_to_one",
    )

    train = (
        frame.date < pd.Timestamp(
            "2021-01-01"
        )
    ) & (
        frame.next_date < pd.Timestamp(
            "2021-01-01"
        )
    )

    feature_cols = [
        c for c in contexts.columns
        if c != "date"
    ]

    x_train = frame.loc[
        train,
        feature_cols,
    ].to_numpy(float)

    y_train = frame.loc[
        train,
        ACTIONS,
    ].to_numpy(float)

    if len(y_train) < 100:
        raise RuntimeError(
            "Too few valid training labels"
        )

    model = policy_model(
        args.ridge_alpha
    )

    model.fit(
        x_train,
        y_train,
    )

    joblib.dump(
        model,
        out / "portfolio_bandit.joblib",
    )

    predicted = model.predict(
        contexts[
            feature_cols
        ].to_numpy(float)
    )

    if predicted.shape != (
        len(contexts),
        4,
    ):
        raise RuntimeError(
            "Invalid four-action predictions"
        )

    if not np.isfinite(
        predicted
    ).all():
        raise RuntimeError(
            "Nonfinite bandit predictions"
        )

    return model, predicted


def predict_policy(scores, lam):
    value = scores - lam * COSTS[None, :]

    return np.array(ACTIONS)[
        value.argmax(axis=1)
    ]


def policy_to_dict(dates, actions):
    return {
        pd.Timestamp(day): action
        for day, action in zip(
            dates,
            actions,
        )
    }


def evaluate_fixed_teacher(
    teacher,
    action_schedule,
    dates,
    market,
    blocks,
    base_cols,
    v18,
    original,
    args,
    split,
):
    matrix = action_matrix(
        blocks,
        policy_to_dict(
            dates,
            action_schedule,
        ),
        market,
    )

    panel, cols = make_feature_panel(
        market,
        base_cols,
        matrix,
    )

    subset = panel[
        panel.split.eq(split)
    ].copy()

    env = make_env(
        v18,
        original,
        subset,
        cols,
        args,
    )

    metrics, equity, weights, returns = (
        original.evaluate_model(
            teacher,
            env,
        )
    )

    return metrics


def calibrate_lambda(
    teacher,
    dates,
    predictions,
    market,
    blocks,
    base_cols,
    v18,
    original,
    args,
    out,
):
    rows = []

    date_split = (
        market[
            ["date", "split"]
        ]
        .drop_duplicates()
        .set_index("date")[
            "split"
        ]
    )

    val_mask = np.array([
        date_split.loc[
            pd.Timestamp(d)
        ] == "val"
        for d in dates
    ])

    for lam in LAMBDA_GRID:
        selected = predict_policy(
            predictions,
            lam,
        )

        metrics = evaluate_fixed_teacher(
            teacher,
            selected,
            dates,
            market,
            blocks,
            base_cols,
            v18,
            original,
            args,
            "val",
        )

        validation_actions = selected[
            val_mask
        ]

        mean_cost = float(np.mean([
            COSTS[ACTIONS.index(a)]
            for a in validation_actions
        ]))

        # Unit-consistent exploratory objective:
        # Sharpe minus dimensionless cost aversion.
        objective = (
            float(metrics["sharpe"])
            - args.cost_weight * mean_cost
        )

        rows.append({
            "lambda": lam,
            "val_return":
                float(metrics["total_return"]),
            "val_sharpe":
                float(metrics["sharpe"]),
            "query_rate":
                float(np.mean(
                    validation_actions != "SKIP"
                )),
            "mean_proxy_cost": mean_cost,
            "objective": objective,
            **{
                f"share_{a}": float(
                    np.mean(
                        validation_actions == a
                    )
                )
                for a in ACTIONS
            },
        })

        print(
            "[VALIDATION]",
            "lambda", lam,
            "sharpe",
            round(float(metrics["sharpe"]), 5),
            "cost",
            round(mean_cost, 5),
            flush=True,
        )

    summary = pd.DataFrame(
        rows
    )

    summary = summary.sort_values(
        [
            "objective",
            "mean_proxy_cost",
            "lambda",
        ],
        ascending=[False, True, True],
    )

    summary.to_csv(
        out / "lambda_validation.csv",
        index=False,
    )

    return float(
        summary.iloc[0]["lambda"]
    )


def build_schedules(
    dates,
    scores,
    lam,
    market,
    seed,
):
    active = predict_policy(
        scores,
        lam,
    )

    schedule = {
        "NEVER":
            np.array(
                ["SKIP"] * len(dates)
            ),
        "ALWAYS_BOTH":
            np.array(
                ["BOTH"] * len(dates)
            ),
        "ACTIVE_V2":
            active,
    }

    # Budget-matched random action permutation.
    # Match action counts within each period;
    # permute timing, not total source usage.
    rng = np.random.default_rng(
        seed
    )

    random_action = active.copy()

    date_to_split = (
        market[
            ["date", "split"]
        ]
        .drop_duplicates()
        .set_index("date")["split"]
    )

    labels = np.array([
        date_to_split.loc[
            pd.Timestamp(d)
        ]
        for d in dates
    ])

    for split in ["train", "val", "test"]:
        ix = np.where(
            labels == split
        )[0]

        random_action[ix] = (
            rng.permutation(
                active[ix]
            )
        )

    schedule["RANDOM_MATCHED"] = (
        random_action
    )

    return schedule


def action_audit(
    dates,
    schedules,
    market,
    out,
):
    date_to_split = (
        market[
            ["date", "split"]
        ]
        .drop_duplicates()
        .set_index("date")["split"]
    )

    rows = []

    for name, actions in schedules.items():
        for day, a in zip(dates, actions):
            rows.append({
                "policy": name,
                "date": day,
                "split": date_to_split.loc[
                    pd.Timestamp(day)
                ],
                "action": a,
                "proxy_cost": COSTS[
                    ACTIONS.index(a)
                ],
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        out / "action_audit.csv",
        index=False,
    )

    return result


def prepare_policy_panels(
    market,
    dates,
    schedules,
    blocks,
    base_cols,
):
    out = {}

    for policy, actions in schedules.items():
        matrix = action_matrix(
            blocks,
            policy_to_dict(
                dates,
                actions,
            ),
            market,
        )

        out[policy] = (
            make_feature_panel(
                market,
                base_cols,
                matrix,
            )
        )

    return out


def train_portfolio(
    panel,
    cols,
    agent,
    seed,
    v18,
    original,
    args,
    folder,
):
    train = panel[
        panel.split.eq("train")
    ].copy()

    original.set_seed(
        seed
    )

    vec_env = (
        original.DummyVecEnv([
            lambda: make_env(
                v18,
                original,
                train,
                cols,
                args,
            )
        ])
    )

    try:
        model = original.build_model(
            agent=agent,
            env=vec_env,
            seed=seed,
            learning_rate=args.learning_rate,
            verbose=0,
        )

        model.learn(
            total_timesteps=args.timesteps,
            progress_bar=False,
        )

        model.save(
            str(folder / "model")
        )

    finally:
        vec_env.close()

    return model


def evaluate_portfolio(
    model,
    panel,
    cols,
    split,
    v18,
    original,
    args,
    folder,
):
    df = panel[
        panel.split.eq(split)
    ].copy()

    env = make_env(
        v18,
        original,
        df,
        cols,
        args,
    )

    metrics, equity, weights, returns = (
        original.evaluate_model(
            model,
            env,
        )
    )

    equity.to_csv(
        folder / f"{split}_equity.csv",
        index=False,
    )

    weights.to_csv(
        folder / f"{split}_weights.csv",
        index=False,
    )

    returns.to_csv(
        folder / f"{split}_returns.csv",
        index=False,
    )

    return {
        k: (
            v.item()
            if isinstance(v, np.generic)
            else v
        )
        for k, v in metrics.items()
    }


def train_experiments(
    panels,
    schedules,
    dates,
    market,
    v18,
    original,
    args,
    out,
    config_hash,
):
    action_lookup = (
        market[
            ["date", "split"]
        ]
        .drop_duplicates()
        .set_index("date")["split"]
    )

    results = []

    for policy in POLICY_NAMES:
        panel, cols = panels[policy]
        acts = schedules[policy]

        test_actions = np.array([
            a for d, a in zip(dates, acts)
            if action_lookup.loc[
                pd.Timestamp(d)
            ] == "test"
        ])

        query_rate = float(
            np.mean(
                test_actions != "SKIP"
            )
        )

        proxy_cost = float(
            np.mean([
                COSTS[ACTIONS.index(a)]
                for a in test_actions
            ])
        )

        for agent in args.agents:
            for seed in args.seeds:
                run_id = (
                    f"{policy}__{agent}__seed{seed}"
                )

                folder = (
                    out / "runs" / run_id
                )

                folder.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                result_path = (
                    folder / "result.json"
                )

                if result_path.exists():
                    old = json.loads(
                        result_path.read_text()
                    )

                    if (
                        old.get("config_hash")
                        != config_hash
                    ):
                        raise RuntimeError(
                            "Resume configuration mismatch: "
                            + run_id
                        )

                    if not (
                        folder / "model.zip"
                    ).exists():
                        raise RuntimeError(
                            "Missing model for " + run_id
                        )

                    results.extend(
                        old["rows"]
                    )

                    print(
                        "[RESUME]",
                        run_id,
                        flush=True,
                    )
                    continue

                print(
                    "[TRAIN]",
                    run_id,
                    flush=True,
                )

                set_seed(seed)

                model = train_portfolio(
                    panel,
                    cols,
                    agent,
                    seed,
                    v18,
                    original,
                    args,
                    folder,
                )

                rows = []

                for split in [
                    "val",
                    "test",
                ]:
                    metrics = evaluate_portfolio(
                        model,
                        panel,
                        cols,
                        split,
                        v18,
                        original,
                        args,
                        folder,
                    )

                    rows.append({
                        "policy": policy,
                        "agent": agent,
                        "seed": seed,
                        "split": split,
                        "query_rate_test":
                            query_rate,
                        "proxy_cost_test":
                            proxy_cost,
                        **metrics,
                    })

                save_json(
                    result_path,
                    {
                        "config_hash":
                            config_hash,
                        "rows": rows,
                    },
                )

                results.extend(rows)

                pd.DataFrame(
                    results
                ).to_csv(
                    out / "all_results.csv",
                    index=False,
                )

    return pd.DataFrame(results)


def final_summaries(results, out):
    results.to_csv(
        out / "all_results.csv",
        index=False,
    )

    test = results[
        results.split.eq("test")
    ]

    summary = (
        test.groupby(
            ["policy", "agent"]
        )
        .agg(
            return_mean=(
                "total_return", "mean"
            ),
            return_std=(
                "total_return", "std"
            ),
            sharpe_mean=(
                "sharpe", "mean"
            ),
            sharpe_std=(
                "sharpe", "std"
            ),
            mdd_mean=(
                "max_drawdown", "mean"
            ),
            query_rate=(
                "query_rate_test", "mean"
            ),
            proxy_cost=(
                "proxy_cost_test", "mean"
            ),
            n=("seed", "count"),
        )
        .reset_index()
    )

    summary.to_csv(
        out / "test_summary.csv",
        index=False,
    )

    # Paired comparison, same agent and seed.
    pairs = []

    for agent in test.agent.unique():
        subset = test[
            test.agent.eq(agent)
        ]

        active = subset[
            subset.policy.eq("ACTIVE_V2")
        ].set_index("seed")

        for baseline in [
            "NEVER",
            "ALWAYS_BOTH",
            "RANDOM_MATCHED",
        ]:
            other = subset[
                subset.policy.eq(baseline)
            ].set_index("seed")

            common = active.index.intersection(
                other.index
            )

            for seed in common:
                pairs.append({
                    "agent": agent,
                    "seed": int(seed),
                    "candidate": "ACTIVE_V2",
                    "baseline": baseline,
                    "delta_return": (
                        float(
                            active.loc[
                                seed,
                                "total_return",
                            ]
                        )
                        -
                        float(
                            other.loc[
                                seed,
                                "total_return",
                            ]
                        )
                    ),
                    "delta_sharpe": (
                        float(
                            active.loc[
                                seed,
                                "sharpe",
                            ]
                        )
                        -
                        float(
                            other.loc[
                                seed,
                                "sharpe",
                            ]
                        )
                    ),
                    "delta_mdd": (
                        float(
                            active.loc[
                                seed,
                                "max_drawdown",
                            ]
                        )
                        -
                        float(
                            other.loc[
                                seed,
                                "max_drawdown",
                            ]
                        )
                    ),
                })

    pd.DataFrame(pairs).to_csv(
        out / "paired_results.csv",
        index=False,
    )

    print("\n===== TEST SUMMARY =====")
    print(
        summary.to_string(
            index=False
        )
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--tier",
        default="conservative",
        choices=[
            "conservative",
            "inclusive",
        ],
    )

    p.add_argument(
        "--out",
        default=str(OUT_DEFAULT),
    )

    p.add_argument(
        "--agents",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
    )

    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=50000,
    )

    p.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
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
        "--ridge-alpha",
        type=float,
        default=10.0,
    )

    p.add_argument(
        "--cost-weight",
        type=float,
        default=2.0,
    )

    p.add_argument(
        "--random-seed",
        type=int,
        default=2026,
    )

    p.add_argument(
        "--prepare-only",
        action="store_true",
    )

    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("AEGIS PORTFOLIO-AWARE RETRIEVAL V2")
    print("=" * 76)

    (
        v18,
        original,
        schema,
        market,
        evidence,
        manifest,
    ) = read_original_data(
        args
    )

    audit = audit_provenance(
        market,
        out,
    )

    blocks, relevance = (
        construct_evidence(
            evidence,
            audit,
            manifest,
            market,
        )
    )

    dates, context_X = (
        build_context(
            market,
            relevance,
        )
    )

    contexts = pd.concat(
        [
            dates.rename("date"),
            context_X,
        ],
        axis=1,
    )

    contexts.to_csv(
        out / "policy_contexts.csv",
        index=False,
    )

    teacher = load_teacher()

    base_cols = schema[
        "FULL_DYNAMIC"
    ]

    print(
        "\nBuilding four-action "
        "portfolio utility targets...",
        flush=True,
    )

    labels = teacher_action_targets(
        teacher,
        v18,
        original,
        market,
        base_cols,
        blocks,
        args,
    )

    labels.to_csv(
        out / "portfolio_utility_targets.csv",
        index=False,
    )

    print(
        "\nTraining contextual bandit...",
        flush=True,
    )

    policy, scores = train_bandit(
        contexts,
        labels,
        args,
        out,
    )

    score_df = pd.DataFrame(
        scores,
        columns=[
            f"value_{a}"
            for a in ACTIONS
        ],
    )

    score_df.insert(
        0,
        "date",
        dates.to_numpy(),
    )

    score_df.to_csv(
        out / "predicted_action_values.csv",
        index=False,
    )

    print(
        "\nCalibrating retrieval cost "
        "using 2021 validation...",
        flush=True,
    )

    selected_lambda = calibrate_lambda(
        teacher,
        dates,
        scores,
        market,
        blocks,
        base_cols,
        v18,
        original,
        args,
        out,
    )

    print(
        "\nSelected lambda:",
        selected_lambda,
        flush=True,
    )

    schedules = build_schedules(
        dates,
        scores,
        selected_lambda,
        market,
        args.random_seed,
    )

    action_audit(
        dates,
        schedules,
        market,
        out,
    )

    panels = prepare_policy_panels(
        market,
        dates,
        schedules,
        blocks,
        base_cols,
    )

    config = {
        "status":
            "EXPLORATORY",
        "policy_class":
            "model_based_contextual_bandit",
        "teacher":
            "DYNAMIC_ACTIVE A2C seed42",
        "teacher_retrained":
            False,
        "source_tier":
            args.tier,
        "base_dimensions":
            len(base_cols),
        "evidence_dimensions":
            len(RAG_COLS),
        "selected_lambda_2021":
            selected_lambda,
        "ridge_alpha":
            args.ridge_alpha,
        "cost_weight":
            args.cost_weight,
        "seeds":
            args.seeds,
        "agents":
            args.agents,
        "timesteps":
            args.timesteps,
        "learning_rate":
            args.learning_rate,
        "transaction_cost":
            args.transaction_cost,
        "reward_scaling":
            args.reward_scaling,
        "source_manifest_sha256":
            file_sha256(
                OLD / "manifest.json"
            ),
        "source_script_sha256":
            file_sha256(
                AEGIS_SCRIPT
            ),
        "strict_pit_certified":
            False,
        "limitations": [
            "Teacher estimates one-step utility.",
            "Teacher is trained on historical train period.",
            "Counterfactual returns are model-based labels.",
            "Ridge action values are not causal effects.",
            "Policy actions are portfolio-day level.",
            "Costs are dimensionless proxies.",
            "Retrieval costs not deducted in PortfolioEnv.",
            "Historical KG representation provenance unverified.",
            "Evidence encoder is a lexical/relevance heuristic.",
            "Joint retrieval/portfolio RL is not implemented.",
        ],
    }

    config_hash = hashlib.sha256(
        json.dumps(
            config,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]

    config["config_hash"] = (
        config_hash
    )

    manifest_path = (
        out / "experiment_manifest.json"
    )

    if manifest_path.exists():
        old_config = json.loads(
            manifest_path.read_text()
        )

        if (
            old_config.get("config_hash")
            != config_hash
        ):
            raise RuntimeError(
                "Existing output has a different configuration. "
                "Use a new --out directory."
            )

    save_json(
        manifest_path,
        config,
    )

    if args.prepare_only:
        print(
            "\nPORTFOLIO-AWARE PREPARATION: PASS"
        )
        return

    print(
        "\nTraining portfolio agents "
        "under new frozen policies...",
        flush=True,
    )

    results = train_experiments(
        panels,
        schedules,
        dates,
        market,
        v18,
        original,
        args,
        out,
        config_hash,
    )

    final_summaries(
        results,
        out,
    )

    print(
        "\nAEGIS PORTFOLIO-AWARE V2: PASS"
    )


if __name__ == "__main__":
    main()
