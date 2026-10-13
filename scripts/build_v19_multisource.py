#!/usr/bin/env python3
"""
V19 Temporal Multi-Source KG Feature Fusion.

Inputs:
 V15: historical SEC + FinDKG candidates
 V17: normalized Conservative/Inclusive FinReflectKG
 Original market: prices and technical indicator calculation

Outputs:
 dataset_<tier>.csv
 feature_sets.json
 temporal_audit.csv
 report.json

V19 is EXPLORATORY. It cannot certify public availability.
No RL training is performed here.
"""

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

SEC_COLS = [
    "sec_risk_category_count",
    "sec_mention_count",
    "sec_PANDEMIC",
    "sec_SUPPLY_CHAIN",
    "sec_DEMAND",
    "sec_LITIGATION",
    "sec_REGULATION",
    "sec_CYBERSECURITY",
]

FIN_COLS = [
    "fin_relation_count",
    "fin_produce_count",
    "fin_operate_in_count",
    "fin_distinct_object_count",
]

REF_COLS = [
    "frkg_triple_count",
    "frkg_unique_relation_count",
    "frkg_unique_target_count",
    "frkg_competition",
    "frkg_dependency",
    "frkg_disclosure",
    "frkg_exposure",
    "frkg_impact",
    "frkg_investment",
    "frkg_negative",
    "frkg_operations",
    "frkg_other",
    "frkg_partnership",
    "frkg_production",
    "frkg_supply",
]

SOURCE_MASKS = [
    "kg_mask_sec",
    "kg_mask_findkg",
    "frkg_fresh",
    "frkg_stale",
    "frkg_missing",
    "frkg_age_scaled",
]

DYNAMIC_BASE = [
    "sec_risk_category_count",
    "fin_relation_count",
    "fin_produce_count",
    "fin_operate_in_count",
    "frkg_triple_count",
    "frkg_competition",
    "frkg_dependency",
    "frkg_negative",
]

DYNAMIC_COLS = [
    "dyn_" + x for x in DYNAMIC_BASE
] + [
    "dyn_sec_changed",
    "dyn_findkg_changed",
    "dyn_frkg_filing_changed",
]

TIERS = ["conservative", "inclusive"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def import_v18():
    path = ROOT / "scripts/run_v18_frkg_ablation.py"
    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        "v19_v18_helpers", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_dates(df):
    df = df.copy()

    # V15 uses "tic", while V17 and market use "ticker".
    if "ticker" not in df.columns and "tic" in df.columns:
        df = df.rename(columns={"tic": "ticker"})

    if "ticker" not in df.columns:
        raise ValueError(
            "Missing ticker column. Available columns: "
            + str(df.columns.tolist())
        )

    if "date" not in df.columns:
        raise ValueError("Missing date column")

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.normalize()

    df["ticker"] = (
        df["ticker"].astype(str).str.upper()
    )

    return df


def require_unique(df, name):
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{name}: duplicate date/ticker"
        )


def require_same_keys(left, right, name):
    a = (
        left[["date", "ticker"]]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    b = (
        right[["date", "ticker"]]
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
    if not a.equals(b):
        raise ValueError(
            f"Date/ticker mismatch: {name}"
        )


def fit_transform_group(df, cols, mask_name):
    """
    Train-only scaler for nonnegative raw counts.
    Missing raw counts are imputed AFTER scaling.

    Only rows with source evidence and train split
    participate in parameter estimation.
    """
    df = df.copy()

    present = df[mask_name].eq(1)
    fit = present & df["split"].eq("train")

    if not fit.any():
        raise ValueError(
            f"No training evidence for {mask_name}"
        )

    params = {}

    for col in cols:
        x = pd.to_numeric(
            df[col], errors="coerce"
        )
        valid = x.loc[present]

        if valid.isna().any():
            raise ValueError(
                f"{col}: missing observed value"
            )

        if (
            (~np.isfinite(valid.to_numpy()))
            | (valid.to_numpy() < 0)
        ).any():
            raise ValueError(
                f"{col}: invalid source counts"
            )

        z = np.log1p(x.loc[fit])
        mu = float(z.mean())
        sigma = float(z.std(ddof=0))

        if not np.isfinite(sigma) or sigma < 1e-12:
            sigma = 1.0

        normalized = (
            np.log1p(x.where(present, 0.0))
            - mu
        ) / sigma

        normalized.loc[~present] = 0.0
        df[col] = normalized.astype(float)

        params[col] = {
            "mean": mu,
            "std": sigma,
            "train_fit_rows": int(fit.sum()),
        }

    return df, params


def build_dynamic(df):
    """
    Changes are differences between consecutive
    observed feature states, never evidence that
    a real-world relationship formed or ended.

    Difference is zero on first observed state,
    source gaps, and no-update days.
    """
    df = df.sort_values(
        ["ticker", "date"]
    ).copy()

    for col in DYNAMIC_BASE:
        df["dyn_" + col] = 0.0

    for col in [
        "dyn_sec_changed",
        "dyn_findkg_changed",
        "dyn_frkg_filing_changed",
    ]:
        df[col] = 0.0

    groups = [
        ("kg_mask_sec", SEC_COLS, "dyn_sec_changed"),
        ("kg_mask_findkg", FIN_COLS, "dyn_findkg_changed"),
        ("frkg_fresh", REF_COLS, "dyn_frkg_filing_changed"),
    ]

    for ticker, index in df.groupby(
        "ticker", sort=False
    ).groups.items():

        idx = list(index)
        g = df.loc[idx]

        for mask, cols, changed_col in groups:
            current_valid = g[mask].eq(1)
            previous_valid = current_valid.shift(
                1, fill_value=False
            )

            both = current_valid & previous_valid

            before = g[cols].shift(1)
            different = (
                (g[cols] - before)
                .abs()
                .gt(1e-12)
                .any(axis=1)
            )

            change = both & different

            df.loc[idx, changed_col] = (
                change.astype(float).to_numpy()
            )

            for col in cols:
                output_col = "dyn_" + col
                if output_col not in df:
                    continue

                delta = (
                    g[col] - before[col]
                ).where(both, 0.0).fillna(0.0)

                df.loc[idx, output_col] = (
                    delta.to_numpy(dtype=float)
                )

    # Numeric change indicators only; no assertion that
    # extraction count changes imply a true KG event.
    return df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


def build_feature_sets(tech_cols):
    base = list(tech_cols)

    sets = {
        "CLEAN": base,
        "SEC_ONLY": base + SEC_COLS + ["kg_mask_sec"],
        "FINDKG_ONLY": base + FIN_COLS + ["kg_mask_findkg"],
        "REFLECTKG_ONLY": base + REF_COLS + [
            "frkg_fresh", "frkg_stale",
            "frkg_missing", "frkg_age_scaled",
        ],
        "SEC_FINDKG": base + SEC_COLS + FIN_COLS + [
            "kg_mask_sec", "kg_mask_findkg",
        ],
        "SEC_REFLECTKG": base + SEC_COLS + REF_COLS + [
            "kg_mask_sec", "frkg_fresh",
            "frkg_stale", "frkg_missing",
            "frkg_age_scaled",
        ],
        "FULL_STATIC": (
            base + SEC_COLS + FIN_COLS + REF_COLS
            + SOURCE_MASKS
        ),
        "FULL_DYNAMIC": (
            base + SEC_COLS + FIN_COLS + REF_COLS
            + SOURCE_MASKS + DYNAMIC_COLS
        ),
    }

    for name, cols in sets.items():
        if len(cols) != len(set(cols)):
            raise ValueError(
                f"Duplicate feature column in {name}"
            )

    return sets


def main(args):
    v18 = import_v18()

    v15_path = Path(args.v15)
    v17_root = Path(args.v17_root)

    if not v15_path.exists():
        raise FileNotFoundError(
            f"V15 missing: {v15_path}. "
            "Run V15 on real market dates first."
        )

    market = v18.read_market(args.market)
    technical = v18.build_past_only_technical(market)
    technical = v18.assign_split(technical)

    # This scaler is fitted on 2018-2021 only.
    tech_scaler = v18.fit_technical_scaler(technical)
    technical = v18.transform_technical(
        technical, tech_scaler
    )
    technical = parse_dates(technical)
    require_unique(technical, "technical")

    v15 = pd.read_csv(v15_path)
    v15 = v15.loc[
        v15["condition"].eq("Full")
    ].copy()
    v15 = parse_dates(v15)

    require_unique(v15, "V15 Full")

    required_v15 = (
        SEC_COLS + FIN_COLS
        + ["kg_mask_sec", "kg_mask_findkg"]
    )

    missing = [
        c for c in required_v15
        if c not in v15.columns
    ]
    if missing:
        raise ValueError(
            f"V15 missing columns: {missing}"
        )

    # V15 supplies SEC + FinDKG; do not use the
    # unverified V15 ReflectKG snapshot as V17 data.
    v15_cols = [
        "date", "ticker", "source_snapshot_as_of",
        "evidence_ids",
        "kg_mask_sec", "kg_mask_findkg",
    ] + SEC_COLS + FIN_COLS

    v15 = v15[v15_cols].copy()

    for mask in ["kg_mask_sec", "kg_mask_findkg"]:
        v15[mask] = pd.to_numeric(
            v15[mask], errors="raise"
        )

        if not v15[mask].isin([0, 1]).all():
            raise ValueError(
                f"Invalid V15 mask: {mask}"
            )

    # Join each tier onto identical technical-market dates.
    results = {}
    reports = {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for tier in TIERS:
        source = (
            v17_root / f"{tier}_normalized_all.csv"
        )
        if not source.exists():
            raise FileNotFoundError(source)

        fr = pd.read_csv(source)
        fr = parse_dates(fr)
        require_unique(fr, f"V17 {tier}")

        expected = [
            "date", "ticker", "available_at",
            "source_file", "frkg_fresh",
            "frkg_stale", "frkg_missing",
            "frkg_age_scaled",
        ] + REF_COLS

        absent = [
            c for c in expected
            if c not in fr.columns
        ]
        if absent:
            raise ValueError(
                f"V17 {tier} missing: {absent}"
            )

        fr = fr[expected].copy()
        fr["available_at"] = pd.to_datetime(
            fr["available_at"],
            errors="coerce"
        )

        # The V17 date gate must remain valid.
        bad = (
            fr["frkg_fresh"].eq(1)
            & (
                fr["available_at"].isna()
                | (fr["available_at"] >= fr["date"])
            )
        )
        if bad.any():
            raise ValueError(
                f"V17 {tier} failed filing date gate"
            )

        merged = technical.merge(
            v15,
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        if not merged["_merge"].eq("both").all():
            raise ValueError(
                "V15 does not cover all V19 market dates"
            )
        merged = merged.drop(columns="_merge")

        merged = merged.merge(
            fr,
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        if not merged["_merge"].eq("both").all():
            raise ValueError(
                f"V17 {tier} missing V19 market dates"
            )
        merged = merged.drop(columns="_merge")

        for mask in [
            "frkg_fresh", "frkg_stale",
            "frkg_missing",
        ]:
            merged[mask] = pd.to_numeric(
                merged[mask], errors="raise"
            )

        if not (
            merged[[
                "frkg_fresh", "frkg_stale",
                "frkg_missing"
            ]].sum(axis=1).eq(1)
        ).all():
            raise ValueError(
                f"{tier}: invalid FRKG masks"
            )

        merged, sec_scaler = fit_transform_group(
            merged, SEC_COLS, "kg_mask_sec"
        )
        merged, fin_scaler = fit_transform_group(
            merged, FIN_COLS, "kg_mask_findkg"
        )

        merged = build_dynamic(merged)

        feature_sets = build_feature_sets(
            v18.TECH_FEATURES
        )

        for condition, columns in feature_sets.items():
            arr = merged[columns].to_numpy(
                dtype=float
            )

            if not np.isfinite(arr).all():
                raise ValueError(
                    f"{tier}/{condition}: "
                    "non-finite model inputs"
                )

        # Conditions are feature lists, NOT separate
        # populations or market date ranges.
        merged = merged.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        v18.validate_balanced(
            merged, TICKERS, f"V19 {tier}"
        )

        results[tier] = merged

        output = out / f"dataset_{tier}.csv"
        merged.to_csv(output, index=False)

        scaler_dir = out / "scalers"
        scaler_dir.mkdir(exist_ok=True)

        (
            scaler_dir / f"{tier}_sec.json"
        ).write_text(
            json.dumps(sec_scaler, indent=2),
            encoding="utf-8",
        )
        (
            scaler_dir / f"{tier}_findkg.json"
        ).write_text(
            json.dumps(fin_scaler, indent=2),
            encoding="utf-8",
        )

        reports[tier] = {
            "rows": len(merged),
            "SEC_observed": int(
                merged["kg_mask_sec"].sum()
            ),
            "FinDKG_observed": int(
                merged["kg_mask_findkg"].sum()
            ),
            "ReflectKG_fresh": int(
                merged["frkg_fresh"].sum()
            ),
            "dynamic_frkg_changed": int(
                merged["dyn_frkg_filing_changed"].sum()
            ),
            "dynamic_findkg_changed": int(
                merged["dyn_findkg_changed"].sum()
            ),
            "dynamic_sec_changed": int(
                merged["dyn_sec_changed"].sum()
            ),
        }

    require_same_keys(
        results["conservative"],
        results["inclusive"],
        "Conservative vs Inclusive",
    )

    feature_sets = build_feature_sets(
        v18.TECH_FEATURES
    )
    (
        out / "feature_sets.json"
    ).write_text(
        json.dumps(
            feature_sets, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8",
    )

    (
        out / "technical_scaler.json"
    ).write_text(
        json.dumps(tech_scaler, indent=2),
        encoding="utf-8",
    )

    audit_rows = []

    for tier, df in results.items():
        for source, mask in [
            ("SEC", "kg_mask_sec"),
            ("FinDKG", "kg_mask_findkg"),
            ("ReflectKG", "frkg_fresh"),
        ]:
            for split, part in df.groupby("split"):
                audit_rows.append({
                    "tier": tier,
                    "split": split,
                    "source": source,
                    "rows": len(part),
                    "observed_rows": int(
                        part[mask].sum()
                    ),
                    "missing_rows": int(
                        len(part) - part[mask].sum()
                    ),
                })

    pd.DataFrame(audit_rows).to_csv(
        out / "temporal_audit.csv",
        index=False
    )

    report = {
        "version": "V19",
        "mode": "EXPLORATORY",
        "train": "2018-2021",
        "validation": "2022",
        "test": "2023",
        "feature_sets": list(feature_sets),
        "tiers": reports,
        "inputs_sha256": {
            "market": sha256(args.market),
            "v15": sha256(v15_path),
            "v17_conservative": sha256(
                v17_root /
                "conservative_normalized_all.csv"
            ),
            "v17_inclusive": sha256(
                v17_root /
                "inclusive_normalized_all.csv"
            ),
        },
        "PIT_publication_verified": False,
        "strict_formal_finrl_approved": False,
        "limitations": [
            "SEC historical coverage may be sparse",
            "FinDKG observation dates are not verified "
            "public release dates",
            "FinReflectKG filing-map dates are not "
            "independently EDGAR verified",
            "Dynamic features are observed count changes, "
            "not verified formation/termination events",
            "No LLM retrieval is performed by this script",
            "No performance claim is made",
        ],
    }

    (out / "report.json").write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V19 MULTI-SOURCE RESULTS ===")
    for tier, info in reports.items():
        print("\nTIER:", tier)
        for k, v in info.items():
            print(" ", k, "=", v)

    print("\nFeature sets:")
    for name, cols in feature_sets.items():
        print(" ", name, len(cols))

    print("\nSaved:", out)
    print("PIT external verification: NOT COMPLETE")
    print("Formal research approval: BLOCKED")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--market",
        default="kg/data/market_5stocks_2017_2026.csv"
    )
    p.add_argument(
        "--v15",
        default=(
            "outputs/v15/portfolio_real/"
            "features_long.csv"
        )
    )
    p.add_argument(
        "--v17-root",
        default="outputs/v17"
    )
    p.add_argument(
        "--out-dir",
        default="outputs/v19"
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
