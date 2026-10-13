#!/usr/bin/env python3
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_pit_2022_2023"
SOURCE = ROOT / "outputs/v20/fusion"
DEST = BASE / "finrl_inputs"

TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}
TIERS = ["conservative", "inclusive"]

ARMS = {
    "FULL15": BASE / "full15/original_daily_embeddings.csv",
    "FOUR4": BASE / "four4/original_daily_embeddings.csv",
}

KG_FEATURES = [f"dkg_emb_{i:02d}" for i in range(11)]

SPLITS = {
    "train": ("2018-01-01", "2020-12-31"),
    "val": ("2021-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2023-12-31"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assign_splits(dates):
    result = pd.Series("", index=dates.index, dtype=str)

    for name, (start, end) in SPLITS.items():
        result.loc[dates.between(start, end)] = name

    if (result == "").any():
        raise RuntimeError("Dates outside experiment period")

    return result


schema_path = SOURCE / "feature_sets.json"
schema = json.loads(schema_path.read_text(encoding="utf-8"))

static = list(schema["FULL_STATIC"])
dynamic_full = list(schema["FULL_DYNAMIC"])
clean = list(schema["CLEAN"])

if dynamic_full[:len(static)] != static:
    raise RuntimeError("Unexpected FULL_DYNAMIC schema")

slots = dynamic_full[len(static):]

if (
    len(clean) != 9
    or len(static) != 42
    or len(slots) != 11
):
    raise RuntimeError("Unexpected feature dimensions")

print("Dynamic feature slots:")
for slot, feature in zip(slots, KG_FEATURES):
    print(f"  {slot} <- {feature}")

DEST.mkdir(parents=True, exist_ok=True)

for arm, embedding_path in ARMS.items():
    if not embedding_path.is_file():
        raise FileNotFoundError(embedding_path)

    emb = pd.read_csv(embedding_path)

    required = {"date", "tic"} | set(KG_FEATURES)
    if not required.issubset(emb.columns):
        raise RuntimeError(f"{arm}: missing embedding columns")

    emb["date"] = pd.to_datetime(
        emb["date"], errors="raise"
    ).dt.normalize()

    emb["tic"] = emb["tic"].astype(str).str.upper()

    if set(emb["tic"]) != TICKERS:
        raise RuntimeError(f"{arm}: unexpected tickers")

    if emb.duplicated(["date", "tic"]).any():
        raise RuntimeError(f"{arm}: duplicate KG keys")

    if not np.isfinite(
        emb[KG_FEATURES].to_numpy(dtype=float)
    ).all():
        raise RuntimeError(f"{arm}: nonfinite KG data")

    outdir = DEST / arm
    outdir.mkdir(parents=True, exist_ok=True)

    for tier in TIERS:
        source_path = SOURCE / f"dataset_{tier}.csv"
        target_path = outdir / f"dataset_{tier}.csv"

        original = pd.read_csv(
            source_path, low_memory=False
        )

        if not {
            "date", "ticker", "split"
        }.issubset(original.columns):
            raise RuntimeError("Missing market data keys")

        frame = original.copy()

        frame["date"] = pd.to_datetime(
            frame["date"], errors="raise"
        ).dt.normalize()

        frame["ticker"] = (
            frame["ticker"].astype(str).str.upper()
        )

        if frame.duplicated(["date", "ticker"]).any():
            raise RuntimeError("Duplicate market panel keys")

        if set(frame["ticker"]) != TICKERS:
            raise RuntimeError("Unexpected market tickers")

        if not set(slots).issubset(frame.columns):
            raise RuntimeError("Dynamic slots missing in market data")

        # Reassign the actual splits, not merely the config.
        frame["split"] = assign_splits(frame["date"])

        signal = emb[
            ["date", "tic"] + KG_FEATURES
        ].rename(columns={"tic": "ticker"})

        merged = frame.merge(
            signal,
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
            sort=False,
        )

        if len(merged) != len(frame):
            raise RuntimeError("Merge changed market row count")

        if merged[KG_FEATURES].isna().any().any():
            raise RuntimeError(
                f"{arm}/{tier}: missing KG observations"
            )

        for slot, feature in zip(slots, KG_FEATURES):
            merged[slot] = merged[feature].to_numpy(
                dtype=float
            )

        # Preserve the complete market input schema.
        output = merged[list(original.columns)].copy()

        # Store dates in an unambiguous format.
        output["date"] = pd.to_datetime(
            output["date"]
        ).dt.strftime("%Y-%m-%d")

        # Strictly preserve every field except
        # split labels and the 11 KG feature slots.
        allowed_changes = set(slots) | {"split", "date"}

        for col in original.columns:
            if col not in allowed_changes:
                if not original[col].equals(output[col]):
                    raise RuntimeError(
                        f"Unexpected changed column: {col}"
                    )

        expected_counts = {
            "train": 3780,
            "val": 1260,
            "test": 2505,
        }

        actual_counts = (
            output["split"].value_counts().to_dict()
        )

        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Unexpected split counts: {actual_counts}"
            )

        if not output[
            output["split"].eq("test")
        ]["date"].between(
            "2022-01-01", "2023-12-31"
        ).all():
            raise RuntimeError("Invalid test dates")

        if target_path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing input: "
                f"{target_path}"
            )

        output.to_csv(target_path, index=False)

        print(
            f"READY {arm}/{tier}: "
            f"{len(output)} rows"
        )
        print("  splits:", actual_counts)
        print("  file:", target_path)

    target_schema = outdir / "feature_sets.json"

    if target_schema.exists():
        raise RuntimeError(
            f"Refusing to overwrite {target_schema}"
        )

    target_schema.write_text(
        schema_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    meta = {
        "arm": arm,
        "embeddings": str(embedding_path),
        "embedding_sha256": sha256(embedding_path),
        "training_period": "2018-2020",
        "validation_period": "2021",
        "test_period": "2022-2023",
        "splits": SPLITS,
        "kg_feature_columns": KG_FEATURES,
        "market_dynamic_slots": slots,
        "placebo_definition": "11 zero-valued features",
        "strict_pit_certified": False,
    }

    (outdir / "build_manifest.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

print("\n=== BOTH FINRL INPUT ARMS BUILT ===")
print("Original V20 market data: unchanged")
print("Original KG experiments: unchanged")
print("DRL training: not started")
