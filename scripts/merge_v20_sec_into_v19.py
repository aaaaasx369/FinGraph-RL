#!/usr/bin/env python3
"""
V20 SEC + V19 Fusion Repair

- Uses downloaded SEC daily evidence
- Preserves V19 FinDKG and FinReflectKG features
- Rebuilds SEC train-only normalization
- Rebuilds SEC temporal-change features
- Checks both 2018-2023 panels
- Writes complete datasets and feature_sets.json
- Does not modify V18 or V19 data
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

V19 = ROOT / "outputs/v19"
V20 = ROOT / "outputs/v20"
OUT = V20 / "fusion"

TIERS = ["conservative", "inclusive"]

SEC = [
    "sec_risk_category_count",
    "sec_mention_count",
    "sec_PANDEMIC",
    "sec_SUPPLY_CHAIN",
    "sec_DEMAND",
    "sec_LITIGATION",
    "sec_REGULATION",
    "sec_CYBERSECURITY",
]


def load_sec():
    path = V20 / "sec_daily_features.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = [
        "date",
        "ticker",
        "kg_mask_sec",
        "sec_evidence_id",
        "sec_source_available_at",
        "sec_source_sha256",
        "sec_source_url",
    ] + SEC

    missing = [
        c for c in required if c not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing SEC columns: " + str(missing)
        )

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
    )

    df["sec_source_available_at"] = (
        pd.to_datetime(
            df["sec_source_available_at"],
            errors="coerce"
        ).dt.normalize()
    )

    df["kg_mask_sec"] = pd.to_numeric(
        df["kg_mask_sec"], errors="raise"
    ).astype(int)

    if not df["kg_mask_sec"].isin([0, 1]).all():
        raise ValueError("Invalid SEC mask")

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate SEC keys")

    present = df["kg_mask_sec"].eq(1)

    if not present.any():
        raise ValueError(
            "No SEC text evidence is available"
        )

    invalid = (
        present
        & (
            df["sec_source_available_at"].isna()
            | (
                df["sec_source_available_at"]
                >= df["date"]
            )
            | df["sec_source_sha256"].isna()
            | df["sec_evidence_id"].isna()
        )
    )

    if invalid.any():
        raise ValueError(
            "SEC Point-in-Time audit FAILED"
        )

    return df


def load_v19(tier):
    path = V19 / f"dataset_{tier}.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
    )

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"Duplicate V19 keys in {tier}"
        )

    return df


def normalize_sec(df):
    df = df.copy()

    present = df["kg_mask_sec"].eq(1)
    train = (
        present & df["split"].eq("train")
    )

    if not train.any():
        raise ValueError(
            "No SEC evidence in training period"
        )

    params = {}

    for col in SEC:
        raw = pd.to_numeric(
            df[col], errors="coerce"
        )

        if raw.loc[present].isna().any():
            raise ValueError(
                f"Missing observed SEC values: {col}"
            )

        observed = raw.loc[present].to_numpy(
            dtype=float
        )

        if (
            not np.isfinite(observed).all()
            or (observed < 0).any()
        ):
            raise ValueError(
                f"Invalid SEC values: {col}"
            )

        fitted = np.log1p(raw.loc[train])

        mean = float(fitted.mean())
        std = float(fitted.std(ddof=0))

        if not np.isfinite(std) or std <= 1e-12:
            std = 1.0

        transformed = (
            np.log1p(raw.where(present, 0.0))
            - mean
        ) / std

        transformed.loc[~present] = 0.0

        df[col] = transformed.astype(float)

        params[col] = {
            "mean": mean,
            "std": std,
            "training_rows": int(train.sum()),
        }

    return df, params


def temporal_sec_features(df):
    df = df.sort_values(
        ["ticker", "date"]
    ).copy()

    present = df["kg_mask_sec"].eq(1)

    previous_present = (
        df.groupby("ticker")["kg_mask_sec"]
        .shift(1)
        .eq(1)
    )

    both = present & previous_present

    previous = (
        df.groupby("ticker")[SEC].shift(1)
    )

    delta = (
        df["sec_risk_category_count"]
        - previous["sec_risk_category_count"]
    )

    df["dyn_sec_risk_category_count"] = (
        delta.where(both, 0.0)
        .fillna(0.0)
        .astype(float)
    )

    changed = (
        df[SEC].sub(previous)
        .abs()
        .gt(1e-12)
        .any(axis=1)
    )

    df["dyn_sec_changed"] = (
        both & changed
    ).astype(float)

    return df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


def build_one(tier, sec, feature_sets):
    print("\nBUILD:", tier, flush=True)

    df = load_v19(tier)

    # Drop only previous SEC features.
    old_sec = SEC + [
        "kg_mask_sec",
        "dyn_sec_risk_category_count",
        "dyn_sec_changed",
        "sec_evidence_id",
        "sec_source_available_at",
        "sec_source_sha256",
        "sec_source_url",
    ]

    df = df.drop(
        columns=[
            c for c in old_sec
            if c in df.columns
        ]
    )

    merged = df.merge(
        sec[
            [
                "date",
                "ticker",
                "kg_mask_sec",
                "sec_evidence_id",
                "sec_source_available_at",
                "sec_source_sha256",
                "sec_source_url",
            ] + SEC
        ],
        on=["date", "ticker"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    if not merged["_merge"].eq("both").all():
        missing = merged.loc[
            merged["_merge"] != "both",
            ["date", "ticker"]
        ].head()

        raise ValueError(
            f"SEC market panel mismatch:\n{missing}"
        )

    merged.drop(
        columns=["_merge"], inplace=True
    )

    merged, scaler = normalize_sec(merged)
    merged = temporal_sec_features(merged)

    for condition, cols in feature_sets.items():
        absent = [
            c for c in cols
            if c not in merged.columns
        ]

        if absent:
            raise ValueError(
                f"{tier}/{condition}: missing {absent}"
            )

        values = merged[cols].to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{tier}/{condition}: non-finite"
            )

    print(
        "rows =", len(merged),
        "SEC covered =", int(
            merged["kg_mask_sec"].sum()
        ),
        "SEC changes =", int(
            merged["dyn_sec_changed"].sum()
        ),
        flush=True
    )

    return merged, scaler


def main():
    feature_path = V19 / "feature_sets.json"

    if not feature_path.exists():
        raise FileNotFoundError(feature_path)

    feature_sets = json.loads(
        feature_path.read_text(encoding="utf-8")
    )

    sec = load_sec()

    print("\n=== V20 FUSION BUILD ===")
    print("SEC rows:", len(sec))
    print(
        "SEC observed:",
        int(sec["kg_mask_sec"].sum())
    )

    generated = {}
    scalers = {}

    # Build both tiers in memory before writing output.
    for tier in TIERS:
        data, scaler = build_one(
            tier, sec, feature_sets
        )
        generated[tier] = data
        scalers[tier] = scaler

    keys = ["date", "ticker", "split"]

    left = (
        generated["conservative"][keys]
        .sort_values(keys)
        .reset_index(drop=True)
    )

    right = (
        generated["inclusive"][keys]
        .sort_values(keys)
        .reset_index(drop=True)
    )

    if not left.equals(right):
        raise ValueError(
            "Conservative/Inclusive panels differ"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    for tier in TIERS:
        generated[tier].to_csv(
            OUT / f"dataset_{tier}.csv",
            index=False
        )

        (
            OUT / f"{tier}_sec_scaler.json"
        ).write_text(
            json.dumps(scalers[tier], indent=2),
            encoding="utf-8"
        )

    (
        OUT / "feature_sets.json"
    ).write_text(
        json.dumps(feature_sets, indent=2),
        encoding="utf-8"
    )

    report = {
        "status": "EXPLORATORY_FUSION_READY",
        "datasets": {
            tier: {
                "rows": len(df),
                "sec_observed": int(
                    df["kg_mask_sec"].sum()
                ),
                "sec_changes": int(
                    df["dyn_sec_changed"].sum()
                ),
            }
            for tier, df in generated.items()
        },
        "SEC_original_document_downloaded": True,
        "SEC_text_semantically_verified": False,
        "FinDKG_publication_verified": False,
        "FinReflectKG_pdf_mapping_verified": False,
        "strict_research_approved": False,
    }

    (
        OUT / "report.json"
    ).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\n=== V20 FUSION COMPLETE ===")
    print(json.dumps(report, indent=2))
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
