#!/usr/bin/env python3
"""
AEGIS-RL Step 16
Unified KG + SEC-RAG Portfolio Feature Fusion

Time split:
    Train: 2018-2020
    Val:   2021
    Test:  2022-2023

Outputs:
    master_features.csv
    clean.csv
    kg_only.csv
    sec_only.csv
    kg_sec.csv
    active.csv
    validation_report.json

This is a FEATURE DATASET, not DRL training.

CAUTION:
- Historical retrieval uses publication date < decision date.
- AEGIS KGT checkpoint was trained on 2018-2020.
- SEC-derived signals require provenance verification.
- V5 policy fitted on 2018-2020; historical
  train actions are in-sample, not out-of-fold.
- No claim of strict full PIT or portfolio gains.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

ROOT = Path.cwd()

AEGIS = (
    ROOT
    / "results/aegis_rl_v1"
)

OUT = AEGIS / "portfolio_fusion_v1"
OUT.mkdir(parents=True, exist_ok=True)

EMB_FILE = (
    ROOT
    / "outputs/aegis_kgt_daily_2018_2023"
    / "daily_dynamic_kgt.csv"
)

KG_FILE = (
    ROOT
    / "outputs/unified_temporal_kg_v3"
    / "unified_temporal_edges.csv"
)

SIGNAL_FILE = (
    ROOT
    / "outputs/unified_temporal_kg_v3"
    / "sec_signal_nodes.csv"
)

CHUNK_FILE = (
    ROOT
    / "rag/data/processed/sec"
    / "sec_chunks.jsonl"
)

V4_DIR = AEGIS / "learned_policy_v4"
V5_DIR = AEGIS / "cost_sensitive_v5"

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]

ACTIONS = ["SKIP", "KG", "SEC", "BOTH"]

COST = np.array(
    [0.0, 0.03, 0.08, 0.11]
)

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

WORDS = {
    "growth": re.compile(
        r"\b(growth|revenue|demand|sales|earnings)\b",
        re.I
    ),
    "risk": re.compile(
        r"\b(risk|loss|litigation|uncertainty|liability)\b",
        re.I
    ),
    "supply": re.compile(
        r"\b(supply|supplier|manufacturing|shortage)\b",
        re.I
    ),
    "competition": re.compile(
        r"\b(competition|competitor|competitive)\b",
        re.I
    ),
}


def split_date(d):
    if d.year <= 2020:
        return "train"
    if d.year == 2021:
        return "val"
    return "test"


def read_embedding():
    df = pd.read_csv(EMB_FILE)
    df["date"] = pd.to_datetime(df["date"])

    cols = sorted(
        c for c in df.columns
        if re.fullmatch(r"kg_\d{3}", c)
    )

    assert len(cols) == 200
    assert not df.duplicated(
        ["date", "tic"]
    ).any()

    assert np.isfinite(
        df[cols].to_numpy(dtype=float)
    ).all()

    return df, cols


def read_kg():
    df = pd.read_csv(
        KG_FILE,
        usecols=[
            "head", "tail", "relation",
            "available_at"
        ]
    )

    df["date"] = pd.to_datetime(
        df["available_at"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None).dt.normalize()

    df = df.dropna(subset=["date"])

    return df


def read_signals():
    df = pd.read_csv(SIGNAL_FILE)

    df["date"] = pd.to_datetime(
        df["available_at"],
        errors="coerce",
        utc=True
    ).dt.tz_convert(None).dt.normalize()

    df["tic"] = (
        df["company"]
        .astype(str)
        .str.replace(
            "COMPANY:", "",
            regex=False
        )
    )

    df = df[
        df["tic"].isin(TICKERS)
    ].copy()

    return df


def read_chunks():

    rows = []

    with CHUNK_FILE.open(
        "r", encoding="utf-8"
    ) as f:

        for line in f:

            if not line.strip():
                continue

            obj = json.loads(line)

            if not isinstance(obj, dict):
                continue

            tic = str(
                obj.get("ticker", "")
            ).upper()

            if tic not in TICKERS:
                continue

            date = pd.to_datetime(
                obj.get("published_at"),
                errors="coerce",
                utc=True
            )

            if pd.isna(date):
                continue

            text = obj.get("text", "")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            row = {
                "tic": tic,
                "date": date.tz_convert(
                    None
                ).normalize(),
                "doc_id": str(
                    obj.get("doc_id", "")
                ),
                "chunk_id": str(
                    obj.get("chunk_id", "")
                ),
                "form": str(
                    obj.get("form", "")
                ),
                "text_length": len(text),
            }

            for name, pattern in WORDS.items():
                row[f"text_{name}"] = int(
                    bool(pattern.search(text))
                )

            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(
            "No valid SEC chunks"
        )

    df = df.drop_duplicates(
        "chunk_id"
    )

    print("SEC chunks:", len(df))

    return df


def history(df, tic, date, days=365):
    return df[
        (df["tic"] == tic)
        & (df["date"] < date)
        & (
            df["date"] >=
            date - pd.Timedelta(days=days)
        )
    ]


def build_source_features(
    emb, kg, signals, chunks
):

    kg_by_tic = {}

    for tic in TICKERS:

        node = f"COMPANY:{tic}"

        related = kg[
            (kg["head"] == node)
            | (kg["tail"] == node)
        ].copy()

        related["tic"] = tic
        kg_by_tic[tic] = related

    signal_by_tic = {
        tic: signals[
            signals["tic"] == tic
        ].copy()
        for tic in TICKERS
    }

    chunk_by_tic = {
        tic: chunks[
            chunks["tic"] == tic
        ].copy()
        for tic in TICKERS
    }

    records = []

    for i, row in enumerate(
        emb.itertuples(index=False)
    ):

        tic = row.tic
        date = row.date

        k = history(
            kg_by_tic[tic],
            tic,
            date
        )

        s = history(
            signal_by_tic[tic],
            tic,
            date
        )

        c = history(
            chunk_by_tic[tic],
            tic,
            date
        )

        result = {
            "date": date,
            "tic": tic,
            "split": split_date(date),
            "hist_kg_count": len(k),
            "hist_kg_relations": (
                k["relation"].nunique()
                if len(k) else 0
            ),
            "hist_sec_chunks": len(c),
            "hist_sec_documents": (
                c["doc_id"].nunique()
                if len(c) else 0
            ),
            "hist_sec_signals": len(s),
            "hist_impact": (
                s["impact"].mean()
                if len(s) else 0.0
            ),
            "hist_risk": (
                s["risk"].mean()
                if len(s) else 0.0
            ),
            "hist_confidence": (
                s["confidence"].mean()
                if len(s) else 0.0
            ),
        }

        for name in WORDS:

            col = f"text_{name}"

            result[
                f"hist_{col}"
            ] = (
                c[col].mean()
                if len(c) else 0.0
            )

        records.append(result)

        if (i + 1) % 1000 == 0:
            print(
                f"[FUSION] {i+1}/{len(emb)}",
                flush=True
            )

    return pd.DataFrame(records)


def load_policy_predictions(master):

    historical = pd.read_csv(
        V4_DIR / "historical_states.csv"
    )

    historical["date"] = pd.to_datetime(
        historical["date"]
    )

    meta = json.loads(
        (
            V5_DIR / "metadata.json"
        ).read_text()
    )

    lam = float(
        meta["selected_lambda"]
    )

    models = {
        action: load(
            V4_DIR / f"policy_{action}.joblib"
        )
        for action in ACTIONS
    }

    pred = np.column_stack([
        models[a].predict(
            historical[POLICY_FEATURES]
        )
        for a in ACTIONS
    ])

    adjusted = (
        pred
        - (lam - 1.0) * COST[None, :]
    )

    ids = adjusted.argmax(axis=1)

    actions = historical[
        ["date", "tic"]
    ].copy()

    actions["action"] = [
        ACTIONS[j] for j in ids
    ]

    actions["policy_lambda"] = lam

    # Where available, use actual saved V5
    # test decisions as an alignment check.
    saved_path = (
        V5_DIR / "test_policy_decisions.csv"
    )

    if saved_path.exists():

        saved = pd.read_csv(saved_path)

        saved["date"] = pd.to_datetime(
            saved["date"]
        )

        check = actions.merge(
            saved[
                ["date", "tic", "action"]
            ],
            on=["date", "tic"],
            how="inner",
            suffixes=(
                "_recomputed",
                "_saved"
            ),
        )

        if not (
            check["action_recomputed"]
            == check["action_saved"]
        ).all():
            raise RuntimeError(
                "V5 test actions do not match"
            )

    result = master.merge(
        actions,
        on=["date", "tic"],
        how="left",
        validate="one_to_one"
    )

    if result["action"].isna().any():
        raise RuntimeError(
            "Missing policy actions"
        )

    return result


def export_ablations(
    master, embeddings, emb_cols
):

    df = master.merge(
        embeddings[
            ["date", "tic"] + emb_cols
        ],
        on=["date", "tic"],
        how="left",
        validate="one_to_one"
    )

    if df[emb_cols].isna().any().any():
        raise RuntimeError(
            "KGT merge failed"
        )

    kg_features = [
        "hist_kg_count",
        "hist_kg_relations",
    ]

    sec_features = [
        col for col in df.columns
        if col.startswith("hist_sec_")
        or col in [
            "hist_impact",
            "hist_risk",
            "hist_confidence",
        ]
        or col.startswith("hist_text_")
    ]

    feature_cols = (
        emb_cols
        + kg_features
        + sec_features
    )

    base_columns = [
        "date", "tic", "split",
        "action", "policy_lambda"
    ]

    experiments = {}

    # Clean: no financial-knowledge features.
    experiments["clean"] = (
        False, False, False
    )

    # KG-only includes trained KGT vectors.
    experiments["kg_only"] = (
        True, False, False
    )

    experiments["sec_only"] = (
        False, True, False
    )

    experiments["kg_sec"] = (
        True, True, False
    )

    experiments["active"] = (
        True, True, True
    )

    for name, (
        use_kg,
        use_sec,
        use_policy
    ) in experiments.items():

        result = df[
            base_columns + feature_cols
        ].copy()

        if not use_kg:
            result[
                emb_cols + kg_features
            ] = 0.0

        if not use_sec:
            result[sec_features] = 0.0

        if use_policy:

            kg_allowed = result[
                "action"
            ].isin(["KG", "BOTH"])

            sec_allowed = result[
                "action"
            ].isin(["SEC", "BOTH"])

            result.loc[
                ~kg_allowed,
                emb_cols + kg_features
            ] = 0.0

            result.loc[
                ~sec_allowed,
                sec_features
            ] = 0.0

        result.to_csv(
            OUT / f"{name}.csv",
            index=False
        )

        print(
            f"[EXPORT] {name}: "
            f"{len(result)} rows"
        )

    return df, feature_cols


def main():

    print("=" * 76)
    print("AEGIS PORTFOLIO FEATURE FUSION")
    print("=" * 76)

    emb, emb_cols = read_embedding()

    emb = emb[
        (emb["date"] >= "2018-01-01")
        & (emb["date"] <= "2023-12-31")
    ].copy()

    kg = read_kg()
    signals = read_signals()
    chunks = read_chunks()

    print("KGT rows:", len(emb))
    print("KG edges:", len(kg))
    print("SEC signals:", len(signals))

    source_features = build_source_features(
        emb, kg, signals, chunks
    )

    master = load_policy_predictions(
        source_features
    )

    merged, feature_cols = export_ablations(
        master,
        emb,
        emb_cols
    )

    merged.to_csv(
        OUT / "master_features.csv",
        index=False
    )

    expected_rows = len(emb)

    assert len(merged) == expected_rows

    assert not merged.duplicated(
        ["date", "tic"]
    ).any()

    print("\n===== SPLITS =====")
    print(
        merged.groupby("split").size()
    )

    print("\n===== POLICY ACTIONS =====")
    print(
        pd.crosstab(
            merged["split"],
            merged["action"]
        ).to_string()
    )

    report = {
        "rows": len(merged),
        "kgt_dimensions": len(emb_cols),
        "feature_columns": feature_cols,
        "train_years": [2018, 2019, 2020],
        "validation_year": 2021,
        "test_years": [2022, 2023],
        "ablation_files": list([
            "clean.csv",
            "kg_only.csv",
            "sec_only.csv",
            "kg_sec.csv",
            "active.csv",
        ]),
        "limitations": [
            "Not yet a Portfolio RL training run.",
            "KGT train-period vectors are in-sample.",
            "V5 policy train-period actions are in-sample.",
            "Source features are as-of historical aggregates.",
            "SEC signal provenance is not independently audited.",
            "Historical feature availability uses date-level cutoff.",
            "Historical aggregates are not all retrieval-conditioned.",
            "Clean means no added knowledge features; "
            "not yet the original Clean FinRL observation tensor.",
            "No realized prices, returns, or risk rewards included.",
            "No adversarial portfolio defense experiment.",
        ],
    }

    (
        OUT / "validation_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nSaved:", OUT)
    print("AEGIS PORTFOLIO FUSION: PASS")


if __name__ == "__main__":
    main()
