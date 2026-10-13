#!/usr/bin/env python3
"""
V36 - SEC-RAG Temporal KG -> FinRL-X Ablation

Commands:
    inspect
    prepare
    audit
    check
    train
    report

Data:
    V20  market / SEC fusion
    V35  SEC-grounded temporal KG features
    V21  original FinRL A2C/PPO/DDPG runner

Experiments:
    CLEAN
    KG3
    KG11
    PLACEBO

IMPORTANT:
    V35 is AI-silver, competition-only.
    Strict point-in-time certification is pending.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

V20 = ROOT / "outputs/v20/fusion"
V35 = ROOT / "outputs/v35"
OUT = ROOT / "outputs/v36"

RUNNER = ROOT / "scripts/run_v21_dynamic_ablation.py"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

KG3 = [
    "kg_known_competitors",
    "kg_competitor_mentions",
    "kg_new_edges_90d",
]

KG11 = [
    "kg_known_competitors",
    "kg_competitor_mentions",
    "kg_recent_mentions_30d",
    "kg_recent_mentions_90d",
    "kg_new_edges_30d",
    "kg_new_edges_90d",
    "kg_days_since_last_disclosure",
    "kg_days_since_first_disclosure",
    "kg_portfolio_competitors",
    "kg_external_competitors",
    "kg_competitor_remention_ratio",
]

TRAIN_START = "2018-01-01"
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"
VAL_END = "2022-12-31"
TEST_START = "2023-01-01"
TEST_END = "2023-12-31"


def fail(message):
    raise RuntimeError("V36 STOP: " + str(message))


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, ensure_ascii=False, indent=2,
            default=str
        ),
        encoding="utf-8",
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path):
    if not path.is_file():
        fail(f"Missing file: {path}")
    return pd.read_csv(path, low_memory=False)


def identify_keys(df):
    date_candidates = [
        "date", "Date", "datetime", "timestamp"
    ]

    ticker_candidates = [
        "tic", "ticker", "symbol", "Ticker"
    ]

    date = next(
        (c for c in date_candidates if c in df.columns),
        None
    )

    ticker = next(
        (c for c in ticker_candidates if c in df.columns),
        None
    )

    if date is None or ticker is None:
        fail(
            "Cannot identify date and ticker columns: "
            + str(df.columns.tolist())
        )

    return date, ticker


def validate_panel(df, label):
    df = df.copy()
    date_col, ticker_col = identify_keys(df)

    df["_v36_date"] = pd.to_datetime(
        df[date_col], errors="raise"
    ).dt.normalize()

    df["_v36_tic"] = (
        df[ticker_col].astype(str).str.upper()
    )

    if df["_v36_date"].isna().any():
        fail(label + ": missing dates")

    if set(df["_v36_tic"]) != set(TICKERS):
        fail(label + ": unexpected ticker universe")

    if df.duplicated(
        ["_v36_date", "_v36_tic"]
    ).any():
        fail(label + ": duplicate ticker-day")

    counts = df.groupby(
        "_v36_date"
    )["_v36_tic"].nunique()

    if not counts.eq(5).all():
        fail(label + ": not all dates have five stocks")

    return df


def walk_feature_lists(obj, path="root"):
    result = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            result.extend(
                walk_feature_lists(
                    value,
                    path + "." + str(key)
                )
            )

    elif isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            result.append((path, obj))
        else:
            for i, value in enumerate(obj):
                result.extend(
                    walk_feature_lists(
                        value,
                        path + f"[{i}]"
                    )
                )

    return result


def discover_dynamic_slots(config, available, explicit):
    if explicit:
        slots = [
            x.strip()
            for x in explicit.split(",")
            if x.strip()
        ]

        if len(slots) != 11:
            fail("--slots must contain exactly 11 columns")

        if len(set(slots)) != 11:
            fail("--slots contains duplicate columns")

        missing = set(slots) - available
        if missing:
            fail("Slots missing from V20: " + str(missing))

        return slots

    arrays = [
        (name, values)
        for name, values in walk_feature_lists(config)
        if (
            len(values) == len(set(values))
            and set(values).issubset(available)
        )
    ]

    dynamic = [
        (name, values)
        for name, values in arrays
        if len(values) == 53
    ]

    static = [
        (name, values)
        for name, values in arrays
        if len(values) == 42
    ]

    possibilities = []

    for d_name, d_cols in dynamic:
        for s_name, s_cols in static:
            if not set(s_cols).issubset(d_cols):
                continue

            delta = [
                c for c in d_cols
                if c not in set(s_cols)
            ]

            if len(delta) == 11:
                possibilities.append(
                    (d_name, s_name, tuple(delta))
                )

    unique = list(dict.fromkeys(
        entry[2] for entry in possibilities
    ))

    if len(unique) != 1:
        print("\n=== V21 FEATURE CONFIG ===")
        for name, values in arrays:
            print(
                name,
                "length=", len(values),
                "columns=", values
            )

        fail(
            "Cannot identify a unique 11-column dynamic "
            "feature delta. Inspect feature_sets.json and "
            "rerun with --slots col1,col2,...,col11"
        )

    print("\nDiscovered V21 dynamic slots:")
    for i, name in enumerate(unique[0], 1):
        print(f"{i:2d}. {name}")

    return list(unique[0])


def load_inputs(args):
    required = [
        RUNNER,
        V20 / "feature_sets.json",
        V20 / "dataset_conservative.csv",
        V20 / "dataset_inclusive.csv",
        V35 / "daily_kg_features.csv",
        V35 / "evidence_ledger.csv",
        V35 / "report.json",
    ]

    for path in required:
        if not path.is_file():
            fail("Missing required input: " + str(path))

    v35_report = json.loads(
        (V35 / "report.json").read_text(
            encoding="utf-8"
        )
    )

    if v35_report.get("strict_pit_approved") is True:
        print("V35 reports PIT approval; independently verify.")

    if v35_report.get("human_verified") is True:
        fail("Unexpected claim: V35 silver is human verified")

    if not set(KG11).issubset(
        set(v35_report.get("feature_columns", []))
    ):
        fail("V35 feature manifest mismatch")

    config = json.loads(
        (V20 / "feature_sets.json").read_text(
            encoding="utf-8"
        )
    )

    market = {}

    for tier in ["conservative", "inclusive"]:
        market[tier] = validate_panel(
            load_csv(
                V20 / f"dataset_{tier}.csv"
            ),
            tier,
        )

    common_columns = (
        set(market["conservative"].columns)
        & set(market["inclusive"].columns)
    )

    slots = discover_dynamic_slots(
        config, common_columns, args.slots
    )

    # The slots are exclusive to the dynamic input.
    # They must not overlap existing 9 base or 42 static columns.
    # The config delta identification above is the primary check.

    kg = validate_panel(
        load_csv(V35 / "daily_kg_features.csv"),
        "V35 KG",
    )

    missing_kg = set(KG11) - set(kg.columns)
    if missing_kg:
        fail("Missing KG columns: " + str(missing_kg))

    for name in KG11:
        kg[name] = pd.to_numeric(
            kg[name], errors="raise"
        )

    if not np.isfinite(
        kg[KG11].to_numpy(dtype=float)
    ).all():
        fail("KG features contain NaN or infinity")

    if (kg[KG11] < 0).any().any():
        fail("KG features contain negative values")

    source_keys = set(
        zip(kg["_v36_date"], kg["_v36_tic"])
    )

    for tier, df in market.items():
        expected = set(
            zip(df["_v36_date"], df["_v36_tic"])
        )

        if expected != source_keys:
            fail(
                f"{tier}: KG and market ticker-date "
                "keys do not match exactly"
            )

    if not kg[KG11].ne(0).any().any():
        fail("KG contains no nonzero signal")

    ledger = load_csv(
        V35 / "evidence_ledger.csv"
    )

    if ledger.empty:
        fail("Empty V35 evidence ledger")

    if not set(
        ledger["relation"].astype(str)
    ).issubset({
        "COMPETES_WITH",
        "SUPPLIES_TO",
        "CUSTOMER_OF",
        "SAME_SECTOR",
    }):
        fail("Unknown relation types in ledger")

    return market, kg, slots, v35_report


def calculate_scaler(kg):
    mask = kg["_v36_date"].between(
        TRAIN_START,
        TRAIN_END,
    )

    if not mask.any():
        fail("No KG rows in training interval")

    train = kg.loc[mask, KG11].copy()

    # Scaling is fitted ONLY from training dates.
    # Log scaling reduces cumulative count magnitude.
    # Use one shared scaler across all five stocks.
    log_train = np.log1p(train)

    mean = log_train.mean()
    std = log_train.std(ddof=0)

    std = std.replace(0, 1.0).fillna(1.0)

    return mean, std


def transform_kg(kg, mean, std, clip):
    out = kg[[
        "_v36_date", "_v36_tic"
    ]].copy()

    log_values = np.log1p(kg[KG11])

    scaled = (
        (log_values - mean) / std
    ).clip(-clip, clip)

    # Missing/no-evidence cases are represented as zero,
    # not as a misleading negative standardized value.
    # A separate coverage flag must be used in research
    # interpretation; here the existing V21 slots are fixed.
    scaled = scaled.where(
        kg[KG11].ne(0),
        0.0,
    )

    out[KG11] = scaled

    if not np.isfinite(
        out[KG11].to_numpy(dtype=float)
    ).all():
        fail("Invalid normalized KG features")

    return out


def build_placebo(scaled, seed):
    """
    Placebo: jointly permute whole daily 5-stock KG vectors
    within TRAIN / VAL / TEST separately.

    Keeps cross-sectional structure and empirical values.
    Destroys exact date alignment with SEC events.

    This is a negative control, NOT a deployable PIT signal.
    """

    rng = np.random.default_rng(seed)
    placebo = scaled.copy()

    periods = [
        (TRAIN_START, TRAIN_END),
        (VAL_START, VAL_END),
        (TEST_START, TEST_END),
    ]

    for start, end in periods:
        mask = placebo["_v36_date"].between(start, end)

        dates = sorted(
            placebo.loc[mask, "_v36_date"].unique()
        )

        if not dates:
            continue

        permutation = rng.permutation(len(dates))

        for ticker in TICKERS:
            idx = placebo.index[
                mask & placebo["_v36_tic"].eq(ticker)
            ].to_numpy()

            sub = placebo.loc[
                idx, ["_v36_date"] + KG11
            ].sort_values("_v36_date")

            if len(sub) != len(dates):
                fail("Placebo date panel incomplete")

            values = sub[KG11].to_numpy()

            placebo.loc[
                sub.index, KG11
            ] = values[permutation]

    return placebo


def inspect(args):
    market, kg, slots, report = load_inputs(args)

    print("\n=== V36 INSPECT ===")
    print("V20 conservative:", len(market["conservative"]))
    print("V20 inclusive:", len(market["inclusive"]))
    print("V35 KG:", len(kg))
    print("Dynamic slots:", len(slots))

    print("\nNonzero fraction:")
    print(
        kg[KG11].ne(0).mean().round(4).to_string()
    )

    print("\nOriginal V35 report:")
    print(json.dumps(
        report, indent=2, ensure_ascii=False
    ))

    print("\n=== STATUS ===")
    print("SEC-RAG temporal KG bridge: AVAILABLE")
    print("Four-relation extraction: INCOMPLETE")
    print("Human ground truth: NOT AVAILABLE")
    print("Strict PIT: NOT APPROVED")
    print("Runner schema: INSPECTION COMPLETE")


def prepare(args):
    market, kg, slots, v35_report = load_inputs(args)

    OUT.mkdir(parents=True, exist_ok=True)

    mean, std = calculate_scaler(kg)

    scaled = transform_kg(
        kg, mean, std, args.clip
    )

    placebo = build_placebo(
        scaled, args.seed
    )

    configs = {
        "KG3": (
            scaled,
            KG3 + [None] * 8,
        ),
        "KG11": (
            scaled,
            KG11,
        ),
        "PLACEBO": (
            placebo,
            KG11,
        ),
    }

    manifest = {
        "source_v20": str(V20),
        "source_v35": str(V35),
        "v20_conservative_sha256": digest(
            V20 / "dataset_conservative.csv"
        ),
        "v20_inclusive_sha256": digest(
            V20 / "dataset_inclusive.csv"
        ),
        "v35_daily_sha256": digest(
            V35 / "daily_kg_features.csv"
        ),
        "v21_feature_config_sha256": digest(
            V20 / "feature_sets.json"
        ),
        "dynamic_slots": slots,
        "kg3_features": KG3,
        "kg11_features": KG11,
        "experiments": list(configs),
        "train_period": [TRAIN_START, TRAIN_END],
        "validation_period": [VAL_START, VAL_END],
        "test_period": [TEST_START, TEST_END],
        "normalization": {
            "method": "log1p_train_zscore_clip",
            "clip": args.clip,
            "mean": mean.to_dict(),
            "std": std.to_dict(),
            "zero_evidence_representation": 0,
        },
        "placebo": {
            "method": "shared_date_permutation_within_split",
            "seed": args.seed,
            "not_for_deployment": True,
        },
        "kg_relation_types": (
            v35_report.get("relation_types", [])
        ),
        "human_verified": False,
        "strict_pit_approved": False,
    }

    for experiment, (signal, selected_features) in configs.items():
        target_dir = OUT / "inputs" / experiment
        target_dir.mkdir(
            parents=True, exist_ok=True
        )

        mapping = dict(zip(
            slots, selected_features
        ))

        for tier, original in market.items():
            original_columns = [
                c for c in original.columns
                if not c.startswith("_v36_")
            ]

            merged = original.merge(
                signal,
                on=["_v36_date", "_v36_tic"],
                how="left",
                validate="one_to_one",
            )

            if len(merged) != len(original):
                fail(
                    f"{experiment}/{tier}: row count changed"
                )

            if merged[KG11].isna().any().any():
                fail(
                    f"{experiment}/{tier}: missing KG values"
                )

            for slot, feature in mapping.items():
                if feature is None:
                    merged[slot] = 0.0
                else:
                    merged[slot] = merged[
                        feature
                    ].astype(float)

            # Never allow appended KG columns to silently
            # alter the V21 feature schema.
            result = merged[original_columns].copy()

            if len(result.columns) != len(original_columns):
                fail("Changed V21 input schema")

            if result[slots].isna().any().any():
                fail("Missing mapped KG slot values")

            result.to_csv(
                target_dir / f"dataset_{tier}.csv",
                index=False,
            )

        shutil.copy2(
            V20 / "feature_sets.json",
            target_dir / "feature_sets.json",
        )

        print(
            f"PREPARED {experiment}: "
            f"{target_dir}",
            flush=True,
        )

    write_json(
        OUT / "manifest.json",
        manifest,
    )

    print("\n=== V36 PREPARE PASS ===")
    print("KG3: 3 KG features + 8 zero slots")
    print("KG11: 11 KG features")
    print("PLACEBO: shuffled 11 KG features")
    print("Original V20: NOT MODIFIED")
    print("Original V21: NOT MODIFIED")


def audit(args):
    manifest_path = OUT / "manifest.json"

    if not manifest_path.exists():
        fail("Run prepare first")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest["v35_daily_sha256"] != digest(
        V35 / "daily_kg_features.csv"
    ):
        fail("V35 source changed after prepare")

    if manifest["v21_feature_config_sha256"] != digest(
        V20 / "feature_sets.json"
    ):
        fail("V21 feature configuration changed")

    slots = manifest["dynamic_slots"]

    originals = {
        tier: load_csv(
            V20 / f"dataset_{tier}.csv"
        )
        for tier in ["conservative", "inclusive"]
    }

    for experiment in ["KG3", "KG11", "PLACEBO"]:
        for tier, original in originals.items():
            path = (
                OUT / "inputs" / experiment
                / f"dataset_{tier}.csv"
            )

            produced = load_csv(path)

            if list(original.columns) != list(produced.columns):
                fail(
                    f"{experiment}/{tier}: schema mismatch"
                )

            if len(original) != len(produced):
                fail(
                    f"{experiment}/{tier}: row mismatch"
                )

            date_col, tic_col = identify_keys(original)

            if not original[
                [date_col, tic_col]
            ].equals(
                produced[[date_col, tic_col]]
            ):
                fail(
                    f"{experiment}/{tier}: ticker/date order changed"
                )

            unchanged = [
                c for c in original.columns
                if c not in slots
            ]

            if not original[unchanged].equals(
                produced[unchanged]
            ):
                fail(
                    f"{experiment}/{tier}: "
                    "non-KG market/SEC features changed"
                )

            values = produced[slots].to_numpy(dtype=float)

            if not np.isfinite(values).all():
                fail(
                    f"{experiment}/{tier}: nonfinite KG"
                )

            if experiment == "KG3":
                if not np.allclose(values[:, 3:], 0):
                    fail("KG3 extra slots are not zero")

            print(
                f"PASS {experiment}/{tier}: "
                f"{len(produced)} rows"
            )

    print("\n=== V36 AUDIT PASS ===")
    print("Original market and SEC features preserved")
    print("Input schema preserved")
    print("Five-stock ticker-date alignment preserved")
    print("All KG slots finite")
    print("Data lineage recorded")
    print("Strict PIT: NOT APPROVED")
    print("Human verified: NO")


def command_for(args, experiment, check_only=False):
    command = [
        PYTHON,
        str(RUNNER),
        "--input-dir",
        str(OUT / "inputs" / experiment),
        "--tiers",
        "conservative",
        "inclusive",
        "--agents",
        "A2C",
        "PPO",
        "DDPG",
        "--seeds",
        args.seeds,
        "--timesteps",
        str(args.timesteps),
        "--output-dir",
        str(OUT / "runs" / experiment),
    ]

    if check_only:
        command.append("--check-only")

    return command


def run(args, check_only):
    if not (
        OUT / "manifest.json"
    ).exists():
        fail("Run prepare first")

    # Do not start experiments with unchecked inputs.
    audit(args)

    for experiment in ["KG3", "KG11", "PLACEBO"]:
        command = command_for(
            args, experiment, check_only
        )

        print("\n" + "=" * 70)
        print(
            "EXPERIMENT:",
            experiment,
            "CHECK" if check_only else "TRAIN",
            flush=True,
        )
        print(" ".join(command), flush=True)

        if args.dry_run:
            continue

        log_dir = OUT / "logs"
        log_dir.mkdir(
            parents=True, exist_ok=True
        )

        mode = (
            "check" if check_only else "train"
        )

        log_path = (
            log_dir / f"{mode}_{experiment}.log"
        )

        with log_path.open(
            "w", encoding="utf-8"
        ) as logfile:
            proc = subprocess.run(
                command,
                cwd=ROOT,
                stdout=logfile,
                stderr=subprocess.STDOUT,
            )

        print("Exit code:", proc.returncode)
        print("Log:", log_path)

        if proc.returncode != 0:
            print("\n=== ERROR LOG TAIL ===")
            lines = log_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()

            print("\n".join(lines[-70:]))

            fail(
                f"{experiment} runner failed. "
                "See log above."
            )

    print(
        "\n=== V36 CHECK PASS ==="
        if check_only
        else "\n=== V36 TRAIN COMMANDS COMPLETED ==="
    )


def report(args):
    manifest_path = OUT / "manifest.json"

    if not manifest_path.exists():
        fail("Run prepare first")

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    print("\n=== V36 EXPERIMENT MANIFEST ===")
    print(json.dumps(
        manifest, indent=2, ensure_ascii=False
    ))

    print("\n=== RUN FILES ===")
    for experiment in ["KG3", "KG11", "PLACEBO"]:
        folder = OUT / "runs" / experiment

        if not folder.exists():
            print(experiment, ": not run")
            continue

        files = [
            str(p.relative_to(ROOT))
            for p in folder.rglob("*")
            if p.is_file()
        ]

        print(
            "\n", experiment,
            "files=", len(files)
        )

        for name in files[:15]:
            print(" ", name)

    print("\n=== INTERPRETATION ===")
    print(
        "Use V21 CLEAN from identical source market inputs "
        "as the common baseline."
    )
    print(
        "KG3: FULL_DYNAMIC in the KG3 runner output."
    )
    print(
        "KG11: FULL_DYNAMIC in the KG11 runner output."
    )
    print(
        "PLACEBO: FULL_DYNAMIC in the PLACEBO runner output."
    )
    print(
        "No claim of superiority until matched test "
        "metrics and random seeds are evaluated."
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "command",
        choices=[
            "inspect",
            "prepare",
            "audit",
            "check",
            "train",
            "report",
        ],
    )

    p.add_argument(
        "--slots",
        default="",
    )

    p.add_argument(
        "--seeds",
        default="0",
    )

    p.add_argument(
        "--timesteps",
        type=int,
        default=2000,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--clip",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = p.parse_args()

    if args.timesteps < 1:
        fail("--timesteps must be positive")

    if args.clip <= 0:
        fail("--clip must be positive")

    if args.command == "inspect":
        inspect(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "audit":
        audit(args)
    elif args.command == "check":
        run(args, check_only=True)
    elif args.command == "train":
        run(args, check_only=False)
    elif args.command == "report":
        report(args)


if __name__ == "__main__":
    main()
