#!/usr/bin/env python3
"""
AEGIS Evidence-level RAG + Portfolio DRL

12 experimental arms:
  CLEAN / CLEAN_RAG
  FULL_STATIC / FULL_STATIC_RAG
  FULL_DYNAMIC_PLACEBO / FULL_DYNAMIC_PLACEBO_RAG
  FULL_DYNAMIC / FULL_DYNAMIC_RAG
  DYNAMIC_ALWAYS_BOTH
  DYNAMIC_RANDOM
  DYNAMIC_ACTIVE
  DYNAMIC_ACTIVE_DEFENSE

Original feature schema:
  CLEAN = 9D
  FULL_STATIC = 42D
  FULL_DYNAMIC_PLACEBO = 53D
  FULL_DYNAMIC = 53D

Additional fixed evidence block:
  KG evidence = 8D
  SEC evidence = 8D
  Total = 16D

Time:
  2018-2020 train
  2021 validation
  2022-2023 test

Research status: EXPLORATORY / INTEGRATION
"""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SCHEMA_DIR = ROOT / "outputs/v20/fusion"
KG_PATH = (
    ROOT / "outputs/unified_temporal_kg_v3"
    / "unified_temporal_edges.csv"
)
SEC_PATH = (
    ROOT / "rag/data/processed/sec"
    / "sec_chunks.jsonl"
)
V4_PATH = (
    ROOT / "results/aegis_rl_v1/learned_policy_v4"
)
V5_PATH = (
    ROOT / "results/aegis_rl_v1/cost_sensitive_v5"
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
ALGORITHMS = ["A2C", "PPO", "DDPG"]

BASES = [
    "CLEAN",
    "FULL_STATIC",
    "FULL_DYNAMIC_PLACEBO",
    "FULL_DYNAMIC",
]

ARMS = [
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

ACTION_COST = {
    "SKIP": 0.0,
    "KG": 0.03,
    "SEC": 0.08,
    "BOTH": 0.11,
}

SPLITS = {
    "train": ("2018-01-01", "2021-01-01"),
    "val": ("2021-01-01", "2022-01-01"),
    "test": ("2022-01-01", "2024-01-01"),
}

POLICY_FEATURES = [
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

KG_F = [f"rag_kg_{i}" for i in range(8)]
SEC_F = [f"rag_sec_{i}" for i in range(8)]
RAG_F = KG_F + SEC_F

TOPIC_REGEX = [
    r"\brevenue\b|\bsales\b|\bdemand\b",
    r"\brisk\b|\buncertainty\b|\bloss\b",
    r"\bsupplier\b|\bsupply\b|\bshortage\b",
    r"\bcompetitive\b|\bcompetition\b",
    r"\blitigation\b|\bregulation\b",
    r"\bgrowth\b|\bprofit\b|\bmargin\b",
    r"\bdebt\b|\bliquidity\b|\bcash\b",
    r"\bcustomer\b|\bcontract\b|\bclient\b",
]
TOPICS = [re.compile(x, re.I) for x in TOPIC_REGEX]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(text):
    return hashlib.sha256(
        str(text).encode("utf-8")
    ).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(
            obj, indent=2, ensure_ascii=False,
            default=str, allow_nan=False
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def load_original_helpers():
    v21 = load_module(
        ROOT / "scripts/run_v21_dynamic_ablation.py",
        "aegis_v21_helper",
    )
    v18, original, tickers = v21.load_helpers()
    if set(tickers) != set(TICKERS):
        raise RuntimeError("Ticker universe mismatch")
    return v21, v18, original


def load_schema():
    obj = json.loads(
        (SCHEMA_DIR / "feature_sets.json").read_text()
    )
    clean = list(obj["CLEAN"])
    static = list(obj["FULL_STATIC"])
    dynamic = list(obj["FULL_DYNAMIC"])

    if (len(clean), len(static), len(dynamic)) != (9, 42, 53):
        raise RuntimeError("Expected feature schema 9/42/53")
    if dynamic[:42] != static:
        raise RuntimeError("Dynamic schema prefix != static schema")

    extra = dynamic[42:]
    schema = {
        "CLEAN": clean,
        "FULL_STATIC": static,
        "FULL_DYNAMIC_PLACEBO": (
            static + ["placebo_" + c for c in extra]
        ),
        "FULL_DYNAMIC": dynamic,
    }
    return schema, extra


def load_market(tier, schema, extra):
    path = SCHEMA_DIR / f"dataset_{tier}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ticker"] = df["ticker"].astype(str).str.upper()

    df = df[
        df["ticker"].isin(TICKERS)
        & df["date"].between("2018-01-01", "2023-12-31")
    ].copy()

    if df.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Duplicate market rows")

    for c in extra:
        df["placebo_" + c] = 0.0

    all_cols = list(dict.fromkeys(
        sum(schema.values(), [])
    ))
    missing = [c for c in all_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")

    for c in all_cols:
        df[c] = pd.to_numeric(df[c], errors="raise")

    if not np.isfinite(df[all_cols].to_numpy(float)).all():
        raise RuntimeError("Nonfinite base features")

    if not (df["market_close"] > 0).all():
        raise RuntimeError("Invalid market prices")

    complete = df.groupby("date")["ticker"].nunique()
    if not complete.eq(5).all():
        raise RuntimeError("Incomplete five-stock panel")

    df["split"] = np.select(
        [
            df.date < pd.Timestamp("2021-01-01"),
            df.date < pd.Timestamp("2022-01-01"),
        ],
        ["train", "val"],
        default="test",
    )
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    for key in SPLITS:
        if not df.split.eq(key).any():
            raise RuntimeError(f"Missing split: {key}")

    return df


def load_kg():
    df = pd.read_csv(
        KG_PATH,
        usecols=[
            "event_id", "head", "tail", "relation",
            "available_at", "confidence"
        ],
    )
    df["date"] = pd.to_datetime(
        df["available_at"], utc=True, errors="coerce"
    ).dt.tz_convert(None).dt.normalize()

    df = df.dropna(subset=["date"]).copy()
    df["confidence"] = pd.to_numeric(
        df["confidence"], errors="coerce"
    ).fillna(0).clip(0, 1)

    indices = {}
    for tic in TICKERS:
        node = f"COMPANY:{tic}"
        x = df[
            df["head"].eq(node) | df["tail"].eq(node)
        ].sort_values("date").reset_index(drop=True)
        indices[tic] = x
    return indices


def load_sec():
    rows = []
    with SEC_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            tic = str(x.get("ticker", "")).upper()
            if tic not in TICKERS:
                continue
            day = pd.to_datetime(
                x.get("published_at"), utc=True,
                errors="coerce"
            )
            if pd.isna(day):
                continue
            txt = x.get("text", "")
            if not isinstance(txt, str) or not txt.strip():
                continue
            rows.append({
                "ticker": tic,
                "date": day.tz_convert(None).normalize(),
                "evidence_id": str(x.get("chunk_id", "")),
                "doc_id": str(x.get("doc_id", "")),
                "text": txt,
                "form": str(x.get("form", "")),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No SEC evidence")
    df = df.drop_duplicates(["ticker", "evidence_id"])
    return {
        tic: df[df.ticker.eq(tic)]
        .sort_values("date").reset_index(drop=True)
        for tic in TICKERS
    }


def choose_evidence(tic, day, kg_index, sec_index):
    """
    Date-gated historical retrieval.

    Fixed generic finance query:
      revenue, risk, supply, competition, regulation,
      growth, debt and customers.

    Returns actual event/chunk provenance.
    """

    kg = kg_index[tic]
    sec = sec_index[tic]
    cutoff = day - pd.Timedelta(days=365)

    kg = kg[
        (kg.date < day) & (kg.date >= cutoff)
    ]
    sec = sec[
        (sec.date < day) & (sec.date >= cutoff)
    ]

    kg_items = []
    for r in kg.itertuples(index=False):
        text = f"{r.head} {r.relation} {r.tail}"
        age = (day - r.date).days
        score = (
            np.exp(-age / 365.0)
            + float(r.confidence) * 0.2
        )
        kg_items.append({
            "source": "KG",
            "id": str(r.event_id),
            "date": r.date,
            "text": text,
            "score": float(score),
            "document": str(r.event_id),
            "confidence": float(r.confidence),
        })

    sec_items = []
    for r in sec.itertuples(index=False):
        hits = sum(bool(p.search(r.text)) for p in TOPICS)
        age = (day - r.date).days
        score = hits / 8 + 0.25 * np.exp(-age / 365.0)
        sec_items.append({
            "source": "SEC",
            "id": str(r.evidence_id),
            "date": r.date,
            "text": r.text,
            "score": float(score),
            "document": str(r.doc_id),
        })

    kg_items.sort(key=lambda x: (-x["score"], x["id"]))
    sec_items.sort(key=lambda x: (-x["score"], x["id"]))

    kg_top = kg_items[:3]
    sec_top = []
    used_docs = set()
    for x in sec_items:
        if x["document"] in used_docs:
            continue
        used_docs.add(x["document"])
        sec_top.append(x)
        if len(sec_top) == 3:
            break

    for x in kg_top + sec_top:
        if not x["date"] < day:
            raise RuntimeError("Evidence date leakage")

    return {"KG": kg_top, "SEC": sec_top}


def encode_kg(items, day):
    if not items:
        return np.zeros(8, dtype=float)

    ages = np.array([
        (day - x["date"]).days for x in items
    ], dtype=float)
    texts = " ".join(x["text"].lower() for x in items)
    relations = [
        x["text"].split(" ", 1)[-1].split(" ")[0]
        for x in items
    ]

    return np.array([
        len(items) / 3,
        float(np.exp(-ages.min() / 365.0)),
        float(np.mean([x["confidence"] for x in items])),
        len(set(relations)) / 3,
        float("supplies_to" in texts or "supplier" in texts),
        float("customer_of" in texts or "customer" in texts),
        float("competes_with" in texts),
        float("has_sec_signal" in texts),
    ], dtype=float)


def encode_sec(items, day):
    if not items:
        return np.zeros(8, dtype=float)

    # Content-derived, not just document counts.
    scores = []
    weights = []
    for x in items:
        age = max(1, (day - x["date"]).days)
        weights.append(np.exp(-age / 365.0))
        scores.append([
            float(bool(p.search(x["text"])))
            for p in TOPICS
        ])

    mat = np.array(scores, dtype=float)
    w = np.array(weights, dtype=float)
    return np.average(mat, weights=w, axis=0)


def load_active_policy():
    state = pd.read_csv(
        V4_PATH / "historical_states.csv"
    )
    state["date"] = pd.to_datetime(
        state["date"]
    ).dt.normalize()
    state["ticker"] = state["tic"].astype(str).str.upper()

    cfg = json.loads(
        (V5_PATH / "metadata.json").read_text()
    )
    lam = float(cfg["selected_lambda"])

    actions = ["SKIP", "KG", "SEC", "BOTH"]
    costs = np.array([
        ACTION_COST[a] for a in actions
    ])

    preds = np.column_stack([
        load(V4_PATH / f"policy_{a}.joblib")
        .predict(state[POLICY_FEATURES])
        for a in actions
    ])
    adjusted = preds - (lam - 1) * costs[None, :]
    state["active_action"] = np.array(actions)[
        adjusted.argmax(axis=1)
    ]

    return state[
        ["date", "ticker", "active_action"]
    ]


def build_decisions(market, policy):
    keys = market[
        ["date", "ticker", "split"]
    ].copy()
    keys = keys.merge(
        policy,
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
    )
    if keys.active_action.isna().any():
        raise RuntimeError("Missing V5 policy decisions")

    rng = np.random.default_rng(2026)
    keys["random_action"] = "SKIP"

    # Match V5 action counts separately within
    # each split. Random seed fixed in advance.
    for split, idx in keys.groupby("split").groups.items():
        chosen = keys.loc[idx, "active_action"].to_numpy()
        keys.loc[idx, "random_action"] = rng.permutation(chosen)

    return keys


def build_evidence(market, decisions, out, max_days=0):
    kg_idx = load_kg()
    sec_idx = load_sec()

    keys = market[
        ["date", "ticker"]
    ].drop_duplicates()

    if max_days:
        dates = sorted(keys.date.unique())[:max_days]
        keys = keys[keys.date.isin(dates)]

    keys = keys.merge(
        decisions[
            ["date", "ticker", "active_action", "random_action"]
        ],
        on=["date", "ticker"],
        validate="one_to_one",
    )

    cache = []
    audit = []
    defense_audit = []

    # Trusted manifest is built from local clean
    # evidence. Hashes catch content modification,
    # not false original statements.
    for i, row in enumerate(keys.itertuples(index=False)):
        day = row.date
        tic = row.ticker
        candidates = choose_evidence(
            tic, day, kg_idx, sec_idx
        )

        # Different actions select which original
        # evidence is revealed to the encoder.
        masks = {
            "SKIP": [],
            "KG": candidates["KG"],
            "SEC": candidates["SEC"],
            "BOTH": (
                candidates["KG"][:2]
                + candidates["SEC"][:2]
            ),
        }

        item = {
            "date": day,
            "ticker": tic,
            "active_action": row.active_action,
            "random_action": row.random_action,
        }

        for action in ["SKIP", "KG", "SEC", "BOTH"]:
            selection = masks[action]
            kg = [x for x in selection if x["source"] == "KG"]
            sec = [x for x in selection if x["source"] == "SEC"]
            features = np.concatenate([
                encode_kg(kg, day),
                encode_sec(sec, day),
            ])
            for k, v in zip(RAG_F, features):
                item[f"{action}_{k}"] = float(v)

        # Attack scenario: alter first selected evidence.
        # Trusted manifest + fallback to another clean
        # candidate from the same authorized sources.
        active = masks[row.active_action]
        poisoned = [dict(x) for x in active]
        if poisoned:
            poisoned[0]["text"] = (
                "Guaranteed profit and zero financial risk"
            )

        approved = {
            (x["source"], x["id"]): sha(x["text"])
            for x in active
        }

        valid = []
        blocked = 0
        for x in poisoned:
            if (
                approved.get((x["source"], x["id"]))
                != sha(x["text"])
            ):
                blocked += 1
                continue
            valid.append(x)

        # For feature comparability, use only validated
        # remaining evidence after corruption.
        kg_valid = [x for x in valid if x["source"] == "KG"]
        sec_valid = [x for x in valid if x["source"] == "SEC"]
        defended = np.concatenate([
            encode_kg(kg_valid, day),
            encode_sec(sec_valid, day),
        ])
        for k, v in zip(RAG_F, defended):
            item[f"DEFENSE_{k}"] = float(v)

        defense_audit.append({
            "date": day,
            "ticker": tic,
            "action": row.active_action,
            "attacked": int(bool(poisoned)),
            "blocked": blocked,
            "remaining_evidence": len(valid),
        })

        for source in ["KG", "SEC"]:
            for rank, x in enumerate(candidates[source], 1):
                audit.append({
                    "date": day,
                    "ticker": tic,
                    "source": source,
                    "rank": rank,
                    "evidence_id": x["id"],
                    "available_at": x["date"],
                    "score": x["score"],
                    "sha256": sha(x["text"]),
                })

        cache.append(item)
        if (i + 1) % 500 == 0:
            print("[RETRIEVE]", i + 1, "/", len(keys), flush=True)

    features = pd.DataFrame(cache)
    pd.DataFrame(audit).to_csv(
        out / "retrieval_audit.csv", index=False
    )
    pd.DataFrame(defense_audit).to_csv(
        out / "defense_audit.csv", index=False
    )
    features.to_csv(
        out / "retrieved_features.csv", index=False
    )
    return features


def make_arms(market, schema, evidence):
    panels = {}
    zero = np.zeros((len(market), 16), dtype=float)

    # All arms share the same dates and five stocks.
    for arm in ARMS:
        if arm.startswith("DYNAMIC_"):
            base = "FULL_DYNAMIC"
        else:
            base = arm.replace("_RAG", "")

        df = market.copy()
        cols = schema[base]

        ev = df[["date", "ticker"]].merge(
            evidence,
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
        )
        if ev["SKIP_" + RAG_F[0]].isna().any():
            raise RuntimeError("Missing retrieval feature rows")

        matrix = zero.copy()

        if arm.endswith("_RAG"):
            matrix = ev[
                ["SEC_" + x for x in RAG_F]
            ].to_numpy(float)
        elif arm == "DYNAMIC_ALWAYS_BOTH":
            matrix = ev[
                ["BOTH_" + x for x in RAG_F]
            ].to_numpy(float)
        elif arm in (
            "DYNAMIC_RANDOM",
            "DYNAMIC_ACTIVE",
            "DYNAMIC_ACTIVE_DEFENSE",
        ):
            action_col = (
                "random_action" if arm == "DYNAMIC_RANDOM"
                else "active_action"
            )
            matrix = np.zeros(
                (len(ev), len(RAG_F)), dtype=float
            )
            for a in ACTION_COST:
                mask = ev[action_col].eq(a).to_numpy()
                prefix = (
                    "DEFENSE"
                    if arm == "DYNAMIC_ACTIVE_DEFENSE"
                    else a
                )
                matrix[mask] = ev.loc[
                    mask,
                    [prefix + "_" + x for x in RAG_F]
                ].to_numpy(float)

        for j, col in enumerate(RAG_F):
            df[col] = matrix[:, j]

        selected = cols + RAG_F
        if not np.isfinite(df[selected].to_numpy(float)).all():
            raise RuntimeError(f"Nonfinite features: {arm}")

        panels[arm] = (df, selected, len(cols))

    return panels


def normalize_evidence(panels):
    # Shared train-only scaling; zero remains zero.
    train_arrays = []
    for arm, (df, cols, _) in panels.items():
        train_arrays.append(
            df.loc[df.split.eq("train"), RAG_F].to_numpy(float)
        )

    stacked = np.vstack(train_arrays)
    scale = np.std(stacked, axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0

    for arm in panels:
        df, cols, base_dim = panels[arm]
        df[RAG_F] = (df[RAG_F] / scale).clip(-5, 5)
        panels[arm] = (df, cols, base_dim)

    return scale.tolist()


def metrics_native(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, (int, float)):
            if not np.isfinite(v):
                raise RuntimeError(f"Nonfinite metric: {k}")
        out[k] = v
    return out


def save_table(data, path):
    if hasattr(data, "to_csv"):
        data.to_csv(path, index=False)
    else:
        pd.DataFrame(data).to_csv(path, index=False)


def run_one(
    arm, agent, seed, panel, features,
    base_dim, v18, original, args, out, run_hash
):
    folder = out / "runs" / f"{arm}__{agent}__seed{seed}"
    status = folder / "result.json"

    if status.exists():
        prev = json.loads(status.read_text())
        if prev["run_hash"] != run_hash:
            raise RuntimeError("Resume config mismatch")
        needed = [
            folder / "model.zip",
            folder / "val_returns.csv",
            folder / "test_returns.csv",
        ]
        if not all(p.exists() for p in needed):
            raise RuntimeError("Incomplete run: " + str(folder))
        print("[RESUME]", folder.name, flush=True)
        return prev["rows"]

    folder.mkdir(parents=True, exist_ok=True)
    train = panel[panel.split.eq("train")].copy()
    val = panel[panel.split.eq("val")].copy()
    test = panel[panel.split.eq("test")].copy()

    original.set_seed(seed)
    vec = original.DummyVecEnv([
        lambda: v18.create_env(original, train, features, args)
    ])
    try:
        model = original.build_model(
            agent=agent,
            env=vec,
            seed=seed,
            learning_rate=args.learning_rate,
            verbose=0,
        )
        model.learn(total_timesteps=args.timesteps)
        model.save(str(folder / "model"))

        rows = []
        for split, frame in [("val", val), ("test", test)]:
            env = v18.create_env(
                original, frame, features, args
            )
            metrics, equity, weights, returns = (
                original.evaluate_model(model, env)
            )
            metrics = metrics_native(metrics)
            rows.append({
                "arm": arm,
                "agent": agent,
                "seed": seed,
                "split": split,
                "base_feature_dims": base_dim,
                "evidence_dims": len(RAG_F),
                **metrics,
            })
            save_table(equity, folder / f"{split}_equity.csv")
            save_table(weights, folder / f"{split}_weights.csv")
            save_table(returns, folder / f"{split}_returns.csv")
            print(
                "[RESULT]", arm, agent, seed, split,
                "return", metrics.get("total_return"),
                "sharpe", metrics.get("sharpe"),
                flush=True,
            )

        save_json(status, {
            "run_hash": run_hash,
            "rows": rows,
        })
        return rows
    finally:
        vec.close()


def summarize(out):
    rows = []
    for p in sorted((out / "runs").glob("*/result.json")):
        x = json.loads(p.read_text())
        rows.extend(x["rows"])
    if not rows:
        return

    df = pd.DataFrame(rows)
    df.to_csv(out / "all_results.csv", index=False)

    metrics = [
        c for c in ["total_return", "sharpe", "max_drawdown"]
        if c in df.columns
    ]
    summary = df.groupby(
        ["arm", "agent", "split"]
    )[metrics].agg(["mean", "std", "count"])
    summary.columns = [
        f"{a}_{b}" for a, b in summary.columns
    ]
    summary.reset_index().to_csv(
        out / "summary.csv", index=False
    )

    comparison_pairs = [
        ("CLEAN_RAG", "CLEAN"),
        ("FULL_STATIC_RAG", "FULL_STATIC"),
        ("FULL_DYNAMIC_PLACEBO_RAG", "FULL_DYNAMIC_PLACEBO"),
        ("FULL_DYNAMIC_RAG", "FULL_DYNAMIC"),
        ("FULL_DYNAMIC", "FULL_DYNAMIC_PLACEBO"),
        ("DYNAMIC_ACTIVE", "FULL_DYNAMIC"),
        ("DYNAMIC_ACTIVE", "DYNAMIC_ALWAYS_BOTH"),
        ("DYNAMIC_ACTIVE", "DYNAMIC_RANDOM"),
        ("DYNAMIC_ACTIVE_DEFENSE", "DYNAMIC_ACTIVE"),
    ]

    comparisons = []
    for candidate, baseline in comparison_pairs:
        a = df[df.arm.eq(candidate)].set_index(
            ["agent", "seed", "split"]
        )
        b = df[df.arm.eq(baseline)].set_index(
            ["agent", "seed", "split"]
        )
        for index in a.index.intersection(b.index):
            x, y = a.loc[index], b.loc[index]
            if isinstance(x, pd.DataFrame):
                raise RuntimeError("Duplicated results")
            row = {
                "candidate": candidate,
                "baseline": baseline,
                "agent": index[0],
                "seed": index[1],
                "split": index[2],
            }
            for metric in metrics:
                row["delta_" + metric] = (
                    float(x[metric]) - float(y[metric])
                )
            comparisons.append(row)

    if comparisons:
        pd.DataFrame(comparisons).to_csv(
            out / "paired_comparison.csv", index=False
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", default="conservative",
                   choices=["conservative", "inclusive"])
    p.add_argument("--arms", nargs="+", choices=ARMS,
                   default=ARMS)
    p.add_argument("--agents", nargs="+", choices=ALGORITHMS,
                   default=ALGORITHMS)
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 123, 2026, 7, 999])
    p.add_argument("--timesteps", type=int, default=50000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--out", default=(
        "results/aegis_rl_v1/evidence_rag_portfolio"
    ))
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--verbose", type=int, default=0)
    args = p.parse_args()

    if args.timesteps <= 0:
        p.error("--timesteps must be positive")
    if len(args.seeds) != len(set(args.seeds)):
        p.error("Duplicate seeds")

    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    v21, v18, original = load_original_helpers()
    schema, dynamic_extra = load_schema()
    market = load_market(args.tier, schema, dynamic_extra)

    print("Market:", len(market), "rows")
    print(market.groupby("split").size())

    policy = load_active_policy()
    decisions = build_decisions(market, policy)

    decisions.to_csv(
        out / "retrieval_actions.csv", index=False
    )
    print("\nActive actions:")
    print(pd.crosstab(
        decisions.split, decisions.active_action
    ))

    evidence = build_evidence(market, decisions, out)
    panels = make_arms(market, schema, evidence)
    evidence_scaler = normalize_evidence(panels)

    manifest = {
        "tier": args.tier,
        "arms": args.arms,
        "agents": args.agents,
        "seeds": args.seeds,
        "timesteps": args.timesteps,
        "learning_rate": args.learning_rate,
        "transaction_cost": args.transaction_cost,
        "reward_scaling": args.reward_scaling,
        "splits": SPLITS,
        "schema": schema,
        "rag_columns": RAG_F,
        "evidence_scaler": evidence_scaler,
        "strict_pit_certified": False,
        "retrieval_cost_in_env_reward": False,
        "limitations": [
            "V5 policy selected from constructed proxy rewards.",
            "V5 train actions are in-sample.",
            "Source metadata only passes coarse date-level gate.",
            "Original feature source provenance not re-certified.",
            "KG evidence and SEC chunks use lexical/time ranking.",
            "RAG encoder is lightweight fixed 16D lexical encoder.",
            "No LLM answer generation or semantic verification.",
            "Original KG 11D features remain as supplied.",
            "Active+Defense includes synthetic top-1 corruption.",
            "SHA256 detects alteration, not misinformation.",
            "No real adversarial attack corpus.",
            "Active currently may choose no SEC on test.",
            "Retrieval cost must be reported separately.",
        ],
    }

    config_hash = sha(json.dumps(manifest, sort_keys=True))
    manifest["config_hash"] = config_hash
    save_json(out / "manifest.json", manifest)

    for arm in args.arms:
        df, features, base_dim = panels[arm]
        print(
            "[ARM]", arm,
            "base", base_dim,
            "+ evidence", 16,
            "= total", len(features),
        )
        train = df[df.split.eq("train")]
        # Check the exact original environment.
        result = v18.env_smoke(
            original, train, features, args
        )
        print("[ENV CHECK]", arm, result, flush=True)

    if args.prepare_only:
        print("AEGIS EVIDENCE PREPARATION: PASS")
        return

    for seed in args.seeds:
        for arm in args.arms:
            df, features, base_dim = panels[arm]
            for agent in args.agents:
                print(
                    "\nTRAIN:", arm, agent, seed,
                    flush=True
                )
                run_one(
                    arm, agent, seed, df, features,
                    base_dim, v18, original,
                    args, out, config_hash
                )
                summarize(out)

    print("\nAEGIS EVIDENCE PORTFOLIO: PASS")


if __name__ == "__main__":
    main()
