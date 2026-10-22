#!/usr/bin/env python3
"""
V17 — Staleness-aware, train-only normalized FinReflectKG.

Features:
- Strict filing date < trading date
- Max filing age (default: 455 days)
- Train-only log1p + z-score
- Separate scaler for each extraction tier
- Missing feature values imputed AFTER normalization
- Explicit observed/stale/missing masks
- Shared dates, five tickers, identical data splits
- Additional common-fresh-panel datasets for fair comparison
- Fail-closed temporal and train/test leakage checks
- No external provenance claims
- No model training

Dependencies: pandas, numpy.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
TIERS = ("conservative", "inclusive")

FEATURE_EXCLUDE = {
    "frkg_observed",
    "frkg_external_availability_verified",
    "frkg_formal_finrl_approved",
}

MASK_COLS = [
    "frkg_observed",
    "frkg_fresh",
    "frkg_stale",
    "frkg_missing",
    "frkg_age_days",
]

def load_data(path, max_age):
    df = pd.read_csv(path)

    required = {
        "date", "ticker", "available_at",
        "source_file", "frkg_observed",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.loc[df["ticker"].isin(TICKERS)].copy()

    for col in ("date", "available_at"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if df["date"].isna().any():
        raise ValueError("Invalid trading date")

    df["date"] = df["date"].dt.normalize()
    df["available_at"] = df["available_at"].dt.normalize()

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker rows")

    if not df["frkg_observed"].isin([0, 1]).all():
        raise ValueError("Invalid observed mask")

    has_filing = df["source_file"].notna()

    # A source document date is mandatory whenever it is
    # claimed to be observed.
    bad = (
        (df["frkg_observed"] == 1)
        & (
            (~has_filing)
            | df["available_at"].isna()
        )
    )
    if bad.any():
        raise ValueError("Observed filing without valid provenance")

    delta = (
        df["date"] - df["available_at"]
    ).dt.days

    # For any filing present, availability must precede
    # the trading date.
    if (has_filing & (delta <= 0)).any():
        raise ValueError(
            "LEAKAGE: same-day or future filing"
        )

    # Fail closed on strange negative or inconsistent dates.
    if (
        has_filing
        & df["available_at"].isna()
    ).any():
        raise ValueError("Missing filing date")

    df["frkg_age_days"] = delta.astype("Float64")

    observed = df["frkg_observed"] == 1
    df["frkg_fresh"] = (
        observed & delta.between(1, max_age)
    ).astype(int)

    df["frkg_stale"] = (
        observed & (delta > max_age)
    ).astype(int)

    df["frkg_missing"] = (
        ~observed
    ).astype(int)

    # Prevent any inconsistency among masks.
    if not (
        df[
            ["frkg_fresh", "frkg_stale", "frkg_missing"]
        ].sum(axis=1) == 1
    ).all():
        raise AssertionError("Masks are not mutually exclusive")

    features = sorted(
        col for col in df.columns
        if col.startswith("frkg_")
        and col not in FEATURE_EXCLUDE
        and col not in MASK_COLS
        and pd.api.types.is_numeric_dtype(df[col])
    )

    if not features:
        raise ValueError("No numerical FRKG features")

    # Confirm observed, fresh counts are finite and >= 0.
    fresh = df["frkg_fresh"] == 1
    values = df.loc[fresh, features]
    if values.isna().any().any():
        raise ValueError("Fresh filing has null KG feature")

    arr = values.to_numpy(dtype=float)
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError("Invalid KG feature counts")

    # A stale filing has no usable feature information.
    df.loc[~fresh, features] = np.nan

    return df, features


def assign_splits(df):
    d = df["date"]
    year = d.dt.year

    df = df.loc[year.between(2018, 2023)].copy()

    year = df["date"].dt.year
    df["split"] = np.select(
        [
            year.between(2018, 2021),
            year == 2022,
            year == 2023,
        ],
        ["train", "val", "test"],
        default="excluded",
    )

    if (df["split"] == "excluded").any():
        raise AssertionError("Unexpected split")

    return df


def check_panel(df, label):
    dates = sorted(df["date"].unique())
    expected = len(dates) * len(TICKERS)

    if len(df) != expected:
        raise ValueError(
            f"{label} unbalanced panel: "
            f"{len(df)} != {expected}"
        )

    for d, g in df.groupby("date"):
        if set(g["ticker"]) != set(TICKERS):
            raise ValueError(
                f"{label}: missing ticker at {d}"
            )

    if df.duplicated(["date", "ticker"]).any():
        raise ValueError(
            f"{label}: duplicate market key"
        )


def train_scaler(df, features):
    # Only fresh training observations can contribute
    # to the scaler statistics.
    mask = (
        (df["split"] == "train")
        & (df["frkg_fresh"] == 1)
    )

    data = df.loc[mask, features]

    if data.empty:
        raise ValueError(
            "No fresh training observations for scaler"
        )

    logged = np.log1p(data.astype(float))
    mean = logged.mean(axis=0)

    # ddof=0, avoid division by zero.
    std = logged.std(axis=0, ddof=0)
    constant_cols = std[
        std.isna() | (std <= 1e-12)
    ].index.tolist()

    std = std.fillna(1.0).mask(
        std <= 1e-12, 1.0
    )

    params = {
        "fit_split": "train",
        "fit_start": df.loc[
            mask, "date"
        ].min().date().isoformat(),
        "fit_end": df.loc[
            mask, "date"
        ].max().date().isoformat(),
        "n_fit_rows": int(mask.sum()),
        "method": "log1p_then_standardize",
        "ddof": 0,
        "constant_features": constant_cols,
        "features": {
            col: {
                "mean": float(mean[col]),
                "std": float(std[col]),
            }
            for col in features
        },
    }
    return params


def transform(df, features, params, max_age=455):
    result = df.copy()

    mean = pd.Series({
        k: v["mean"]
        for k, v in params["features"].items()
    })
    std = pd.Series({
        k: v["std"]
        for k, v in params["features"].items()
    })

    # Convert feature counts into normalized values.
    normalized = (
        np.log1p(result[features].astype(float))
        - mean
    ) / std

    fresh = result["frkg_fresh"] == 1

    # Missing/stale sources are neutral IMPUTATIONS
    # after standardization, not factual zero counts.
    normalized.loc[~fresh, :] = 0.0

    # Neutralize any non-finite values only for missing.
    # Invalid available feature values are fatal.
    if not np.isfinite(
        normalized.loc[fresh].to_numpy(dtype=float)
    ).all():
        raise ValueError("Non-finite normalized KG features")

    result.loc[:, features] = normalized

    # Retain age information without an unbounded
    # stale duration dominating the state.
    result["frkg_age_days_clipped"] = (
        result["frkg_age_days"]
        .clip(lower=0, upper=max_age)
        .fillna(0)
        .astype(float)
    )

    result["frkg_age_scaled"] = (
        result["frkg_age_days_clipped"] / float(max_age)
    )

    # Missing/stale rows are not given a numeric age
    # signal that might be confused with fresh data.
    result.loc[
        result["frkg_fresh"] == 0,
        "frkg_age_scaled"
    ] = 0.0

    # Training should receive explicit masks.
    result["frkg_valid_for_model"] = (
        result["frkg_fresh"].astype(int)
    )

    return result


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            part = f.read(1024 * 1024)
            if not part:
                break
            h.update(part)
    return h.hexdigest()


def main(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = {}
    features_by_tier = {}

    for tier in TIERS:
        path = (
            Path(args.v16_root)
            / tier
            / "daily_filing_aligned.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)

        df, features = load_data(path, args.max_age)
        df = assign_splits(df)
        df = df.sort_values(
            ["date", "ticker"]
        ).reset_index(drop=True)

        check_panel(df, tier)

        data[tier] = df
        features_by_tier[tier] = features

    if (
        features_by_tier["conservative"]
        != features_by_tier["inclusive"]
    ):
        raise ValueError(
            "Conservative/Inclusive feature schema mismatch"
        )

    keys = ["date", "ticker"]
    a = data["conservative"]
    b = data["inclusive"]

    if not a[keys].equals(b[keys]):
        raise ValueError(
            "Conservative/Inclusive date/ticker mismatch"
        )

    if not a["split"].equals(b["split"]):
        raise ValueError(
            "Conservative/Inclusive split mismatch"
        )

    # Shared source filing information should be the same
    # across tiers on overlapping observed dates.
    both_observed = (
        (a["frkg_observed"] == 1)
        & (b["frkg_observed"] == 1)
    )
    for col in ("available_at", "source_file"):
        if not a.loc[both_observed, col].reset_index(
            drop=True
        ).equals(
            b.loc[both_observed, col].reset_index(
                drop=True
            )
        ):
            raise ValueError(
                "Tier provenance mismatch: " + col
            )

    # Fair comparison subset: both tiers have fresh
    # evidence for ALL FIVE stocks on a given date.
    common_row_fresh = (
        (a["frkg_fresh"] == 1)
        & (b["frkg_fresh"] == 1)
    )

    common_frame = a.loc[
        common_row_fresh, ["date", "ticker"]
    ]

    per_date = (
        common_frame.groupby("date")["ticker"]
        .nunique()
    )

    common_dates = set(
        per_date[per_date == len(TICKERS)].index
    )

    report = {
        "experiment": "V17",
        "max_age_days": args.max_age,
        "split_ranges": {
            "train": "2018-01-01 to 2021-12-31",
            "val": "2022-01-01 to 2022-12-31",
            "test": "2023-01-01 to 2023-12-31",
        },
        "common_fresh_panel_days": len(common_dates),
        "public_availability_externally_verified": False,
        "formal_finrl_approved": False,
        "tiers": {},
    }

    coverage_rows = []

    for tier in TIERS:
        df = data[tier]
        features = features_by_tier[tier]

        scaler = train_scaler(df, features)

        if pd.Timestamp(
            scaler["fit_end"]
        ).year > 2021:
            raise AssertionError(
                "SCALER LEAKAGE: fit includes test data"
            )

        transformed = transform(
            df, features, scaler, args.max_age
        )

        transformed["frkg_external_availability_verified"] = 0
        transformed["frkg_formal_finrl_approved"] = 0

        # Entire 2018–2023 panel, with masks.
        output_all = out / (
            tier + "_normalized_all.csv"
        )
        transformed.to_csv(
            output_all, index=False
        )

        # Fair, commonly fresh data (5-stock daily panel).
        common = transformed.loc[
            transformed["date"].isin(common_dates)
        ].copy()

        check_panel(
            common, tier + "_common_fresh"
        )

        common.to_csv(
            out / (
                tier + "_normalized_common_fresh.csv"
            ),
            index=False
        )

        for split in ("train", "val", "test"):
            subset = transformed.loc[
                transformed["split"] == split
            ].copy()

            subset.to_csv(
                out / f"{tier}_{split}.csv",
                index=False
            )

            valid = int(
                subset["frkg_fresh"].sum()
            )
            stale = int(
                subset["frkg_stale"].sum()
            )
            missing = int(
                subset["frkg_missing"].sum()
            )

            coverage_rows.append({
                "tier": tier,
                "split": split,
                "rows": len(subset),
                "fresh": valid,
                "stale": stale,
                "missing": missing,
                "fresh_ratio": (
                    valid / len(subset)
                    if len(subset) else 0
                ),
            })

        scaler_path = out / (
            tier + "_train_scaler_parameters.json"
        )
        scaler_path.write_text(
            json.dumps(
                scaler, indent=2,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )

        report["tiers"][tier] = {
            "rows": len(transformed),
            "fresh": int(
                transformed["frkg_fresh"].sum()
            ),
            "stale": int(
                transformed["frkg_stale"].sum()
            ),
            "missing": int(
                transformed["frkg_missing"].sum()
            ),
            "train_fit_rows": scaler["n_fit_rows"],
            "train_fit_end": scaler["fit_end"],
            "common_fresh_rows": len(common),
            "scaler_sha256": hash_file(
                scaler_path
            ),
        }

    pd.DataFrame(coverage_rows).to_csv(
        out / "coverage_comparison.csv",
        index=False
    )

    audit = {
        "temporal_check": "PASS",
        "panel_alignment": "PASS",
        "tier_feature_schema": "PASS",
        "train_only_scalers": "PASS",
        "staleness_masks": "PASS",
        "external_publication_provenance": "NOT_VERIFIED",
        "formal_backtest_approval": "BLOCKED",
        "notes": [
            "Filing map available_at not independently verified",
            "Missing KG counts replaced after normalization",
            "Mask is mandatory when features are used",
            "Separate per-tier train scalers were fitted",
            "Common-fresh subset reduces but does not "
            "eliminate selection bias",
            "No FinRL agent was trained",
        ]
    }

    (out / "leakage_audit.json").write_text(
        json.dumps(
            audit, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )
    (out / "report.json").write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V17 RESULTS ===")
    print("Common fresh dates:",
          len(common_dates))

    for tier, stats in report["tiers"].items():
        print(
            tier,
            "fresh=", stats["fresh"],
            "stale=", stats["stale"],
            "missing=", stats["missing"],
            "scaler_train_rows=",
            stats["train_fit_rows"],
        )

    print("Saved:", out)
    print("PIT public availability: NOT VERIFIED")
    print("Formal FinRL approval: BLOCKED")


def self_test():
    sample = pd.DataFrame({
        "date": [
            "2021-01-05",
            "2021-12-01",
            "2022-01-05",
            "2023-01-05"
        ],
        "ticker": ["NVDA"] * 4,
        "available_at": [
            "2021-01-04",
            "2021-01-04",
            "2021-01-04",
            "2021-01-04"
        ],
        "source_file": ["NVDA_10k.pdf"] * 4,
        "frkg_observed": [1, 1, 1, 1],
        "frkg_triple_count": [1, 5, 1000, 10000],
    })
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as temp:
        path = Path(temp) / "sample.csv"
        sample.to_csv(path, index=False)

        d, cols = load_data(path, 455)
        d = assign_splits(d)
        scaler = train_scaler(d, cols)

        assert scaler["n_fit_rows"] == 2
        assert scaler["fit_end"] == "2021-12-01"

        transformed = transform(
            d, cols, scaler
        )

        assert transformed[
            "frkg_fresh"
        ].tolist() == [1, 1, 1, 0]

        assert transformed.iloc[3][
            "frkg_triple_count"
        ] == 0.0

        # Changing test data must not change train stats.
        changed = d.copy()
        changed.loc[
            changed["split"] == "test",
            "frkg_triple_count"
        ] = 9999999
        scaler2 = train_scaler(changed, cols)

        assert (
            scaler2["features"]
            == scaler["features"]
        )

    print("SELF-TEST PASSED")
    print("Staleness: PASS")
    print("Train-only scaler: PASS")
    print("Future test perturbation: PASS")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--v16-root", default="outputs/v16"
    )
    p.add_argument(
        "--out-dir", default="outputs/v17"
    )
    p.add_argument(
        "--max-age", type=int, default=455
    )
    p.add_argument(
        "--self-test", action="store_true"
    )
    args = p.parse_args()

    if args.max_age <= 0:
        p.error("--max-age must be positive")

    if args.self_test:
        self_test()
    else:
        main(args)
