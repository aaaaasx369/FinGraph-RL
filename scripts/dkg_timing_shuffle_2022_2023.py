#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_pit_2022_2023"
OUT = BASE / "diagnostics"
INPUTS = BASE / "finrl_inputs"

MODELS = ["FULL15", "FOUR4"]
TIERS = ["conservative", "inclusive"]
TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
SPLITS = {
    "train": ("2018-01-01", "2020-12-31"),
    "val": ("2021-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2023-12-31"),
}

KG = [f"dkg_emb_{i:02d}" for i in range(11)]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stop(msg):
    raise RuntimeError(msg)


def read_embedding(model):
    folder = "full15" if model == "FULL15" else "four4"

    daily_path = BASE / folder / "original_daily_embeddings.csv"
    weekly_path = BASE / folder / "weekly_embeddings_64d.csv"

    daily = pd.read_csv(daily_path)
    weekly = pd.read_csv(weekly_path)

    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    weekly["week_date"] = pd.to_datetime(
        weekly["week_date"]
    ).dt.normalize()

    for df in [daily, weekly]:
        if set(df["tic"]) != set(TICKERS):
            stop(f"{model}: wrong ticker universe")

    if daily.duplicated(["date", "tic"]).any():
        stop(f"{model}: duplicated daily key")

    if weekly.duplicated(["week_date", "tic"]).any():
        stop(f"{model}: duplicated weekly key")

    if not np.isfinite(
        daily[KG].to_numpy(dtype=float)
    ).all():
        stop(f"{model}: nonfinite embeddings")

    return daily, weekly, daily_path, weekly_path


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    changes = []

    for model in MODELS:
        daily, weekly, daily_path, weekly_path = read_embedding(model)

        for ticker in TICKERS:
            d = (
                daily[daily["tic"].eq(ticker)]
                .sort_values("date")
                .copy()
            )
            w = (
                weekly[weekly["tic"].eq(ticker)]
                .sort_values("week_date")
                .copy()
            )

            # Count actual daily changes in the 11D signal.
            vectors = d[KG].to_numpy(dtype=float)
            changed = np.ones(len(d), dtype=bool)

            changed[1:] = np.any(
                ~np.isclose(
                    vectors[1:],
                    vectors[:-1],
                    rtol=1e-8,
                    atol=1e-10,
                ),
                axis=1,
            )

            change_dates = d.loc[changed, "date"]

            # These are snapshot labels, not independently
            # verified publication timestamps.
            wdates = w["week_date"].to_numpy(
                dtype="datetime64[ns]"
            )

            for date in change_dates:
                prior = np.searchsorted(
                    wdates, np.datetime64(date),
                    side="left",
                ) - 1

                if prior >= 0:
                    label = pd.Timestamp(wdates[prior])
                else:
                    label = pd.NaT

                changes.append({
                    "model": model,
                    "ticker": ticker,
                    "daily_change_date": date,
                    "latest_week_label_strictly_before_date": label,
                    "prior_week_label_exists": pd.notna(label),
                })

            for year in [2018, 2019, 2020, 2021, 2022, 2023]:
                year_df = d[d["date"].dt.year.eq(year)]
                unique = len(year_df[KG].drop_duplicates())

                reports.append({
                    "model": model,
                    "ticker": ticker,
                    "year": year,
                    "daily_rows": len(year_df),
                    "unique_11d_vectors": unique,
                    "changed_during_year": unique > 1,
                })

        print(
            model,
            "daily SHA256:", digest(daily_path),
            "weekly SHA256:", digest(weekly_path),
        )

    report = pd.DataFrame(reports)
    change_df = pd.DataFrame(changes)

    report.to_csv(
        OUT / "kg_timing_yearly_summary.csv",
        index=False,
    )
    change_df.to_csv(
        OUT / "kg_daily_change_dates.csv",
        index=False,
    )

    print("\n=== 2022 AND 2023 EMBEDDING AUDIT ===")

    print(
        report[
            report["year"].isin([2022, 2023])
        ].to_string(index=False)
    )

    notes = {
        "status": "PRELIMINARY_ONLY",
        "checks": [
            "Daily KG changes detected",
            "Weekly snapshot labels inspected",
            "2022 and 2023 uniqueness measured",
        ],
        "not_verified": [
            "Original news publication timestamp",
            "Relation availability at trading decision time",
            "Same-day versus next-day execution",
            "Full checkpoint train-period lookahead",
            "Historical public availability of FinDKG",
        ],
        "strict_pit_certified": False,
    }

    (OUT / "timing_audit_limitations.json").write_text(
        json.dumps(notes, indent=2),
        encoding="utf-8",
    )

    print("\nSTRICT PIT: NOT CERTIFIED")


def shuffle_model(model):
    source_dir = INPUTS / model
    destination = INPUTS / f"{model}_SHUFFLED"

    if destination.exists():
        stop(
            f"Destination already exists: {destination}\n"
            "Refusing to overwrite. Keep completed experiments."
        )

    schema_path = source_dir / "feature_sets.json"
    schema = json.loads(schema_path.read_text())

    static = schema["FULL_STATIC"]
    full = schema["FULL_DYNAMIC"]

    if full[:len(static)] != static:
        stop(f"{model}: schema mismatch")

    slots = full[len(static):]

    if len(slots) != 11:
        stop(f"{model}: expected 11 KG slots")

    destination.mkdir(parents=True)

    for tier in TIERS:
        src = source_dir / f"dataset_{tier}.csv"
        df = pd.read_csv(src, low_memory=False)

        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.normalize()

        if df.duplicated(["date", "ticker"]).any():
            stop(f"{model}/{tier}: duplicate date/ticker")

        if set(df["ticker"]) != set(TICKERS):
            stop(f"{model}/{tier}: wrong ticker universe")

        shuffled = df.copy()

        original_keys = df[
            ["date", "ticker", "split"]
        ].copy()

        for split, (start, end) in SPLITS.items():
            sub = df[df["split"].eq(split)].copy()

            if sub.empty:
                stop(f"{model}/{tier}: empty {split}")

            if not sub["date"].between(start, end).all():
                stop(f"{model}/{tier}: invalid {split} dates")

            dates = sorted(sub["date"].unique())

            # Deterministic shared date permutation across
            # all tickers, preserving each day's KG cross-section.
            seed_input = f"{model}|{split}|2026".encode()
            seed = int.from_bytes(
                hashlib.sha256(seed_input).digest()[:8],
                "little",
            )
            rng = np.random.default_rng(seed)

            permutation = rng.permutation(len(dates))

            if len(dates) > 1 and np.array_equal(
                permutation, np.arange(len(dates))
            ):
                permutation = np.roll(permutation, 1)

            destination_dates = pd.DatetimeIndex(dates)
            source_dates = destination_dates[permutation]

            for ticker in TICKERS:
                original_ticker = (
                    df[
                        df["ticker"].eq(ticker)
                        & df["split"].eq(split)
                    ]
                    .sort_values("date")
                )

                target_ticker = (
                    shuffled[
                        shuffled["ticker"].eq(ticker)
                        & shuffled["split"].eq(split)
                    ]
                    .sort_values("date")
                )

                if (
                    len(original_ticker) != len(dates)
                    or len(target_ticker) != len(dates)
                ):
                    stop("Incomplete five-stock market panel")

                source_matrix = (
                    original_ticker.set_index("date")
                    .loc[source_dates, slots]
                    .to_numpy(dtype=float)
                )

                shuffled.loc[
                    target_ticker.index, slots
                ] = source_matrix

        if not shuffled[
            ["date", "ticker", "split"]
        ].equals(original_keys):
            stop("Market keys were modified")

        untouched = [
            c for c in df.columns if c not in slots
        ]

        if not shuffled[untouched].equals(df[untouched]):
            stop("Non-KG columns were modified")

        for split in SPLITS:
            for ticker in TICKERS:
                mask = (
                    df["split"].eq(split)
                    & df["ticker"].eq(ticker)
                )

                a = np.sort(
                    df.loc[mask, slots].to_numpy(
                        dtype=float
                    ),
                    axis=0,
                )
                b = np.sort(
                    shuffled.loc[mask, slots].to_numpy(
                        dtype=float
                    ),
                    axis=0,
                )

                if not np.allclose(a, b):
                    stop("Shuffle altered feature distribution")

        shuffled["date"] = shuffled["date"].dt.strftime(
            "%Y-%m-%d"
        )

        dst = destination / f"dataset_{tier}.csv"
        shuffled.to_csv(dst, index=False)
        print("CREATED:", dst)

    (destination / "feature_sets.json").write_text(
        schema_path.read_text(),
        encoding="utf-8",
    )

    manifest = {
        "model": model,
        "control": "SHUFFLED_KG",
        "shuffle_seed_rule": "SHA256(model|split|2026)",
        "permutation": "shared daily permutation across tickers",
        "shuffle_scope": "independent train/val/test splits",
        "train_dates": SPLITS["train"],
        "validation_dates": SPLITS["val"],
        "test_dates": SPLITS["test"],
        "original_inputs_modified": False,
        "strict_pit_certified": False,
        "warning": (
            "Test-date shuffle can use later KG states "
            "on earlier dates. Research negative control only."
        ),
    }

    (destination / "shuffle_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["audit", "shuffle"],
    )
    args = parser.parse_args()

    if args.command == "audit":
        audit()
    else:
        for model in MODELS:
            shuffle_model(model)

        print("\nSHUFFLED-KG INPUTS READY")
        print("Existing FULL15/FOUR4 inputs unchanged")


if __name__ == "__main__":
    main()
