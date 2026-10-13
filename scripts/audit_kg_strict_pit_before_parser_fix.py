#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = {
    "FULL_15": ROOT / "outputs/finrl_dkg_original_full",
    "FOUR_4": ROOT / "outputs/finrl_dkg_four_native",
}

OUT = ROOT / "outputs/strict_pit_audit"
OUT.mkdir(parents=True, exist_ok=True)

ISSUES = []
RESULTS = []


def record(model, check, status, detail):
    row = {
        "model": model,
        "check": check,
        "status": status,
        "detail": str(detail),
    }
    RESULTS.append(row)
    print(
        f"[{status}] {model} | "
        f"{check}: {detail}",
        flush=True,
    )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def inspect_manifest(name, base):
    path = base / "original_embedding_manifest.json"

    if not path.exists():
        record(name, "manifest", "UNKNOWN", "File missing")
        return

    manifest = json.loads(path.read_text())

    record(
        name, "manifest",
        "FOUND",
        f"checkpoint={manifest.get('checkpoint_path')}",
    )

    stored_sha = manifest.get("checkpoint_sha256")
    cp = manifest.get("checkpoint_path")

    if cp and Path(cp).is_file() and stored_sha:
        actual = digest(Path(cp))
        record(
            name, "checkpoint_sha256",
            "PASS" if actual == stored_sha else "FAIL",
            "Manifest matches checkpoint"
            if actual == stored_sha
            else "Checkpoint hash differs from manifest",
        )
    else:
        record(
            name, "checkpoint_sha256",
            "UNKNOWN",
            "Cannot verify checkpoint file/hash",
        )

    record(
        name, "manifest_strict_pit",
        "NOT_CERTIFIED",
        f"strict_pit_certified="
        f"{manifest.get('strict_pit_certified')}; "
        f"training_period_embedded_with_final_weights="
        f"{manifest.get('training_period_embedded_with_final_weights')}",
    )


def inspect_graph(name, base):
    folder = (
        base / "original_run/FinDKG_dataset/FinDKG-full"
    )
    time_path = folder / "time2id.txt"

    if not time_path.exists():
        record(name, "graph_dates", "UNKNOWN", "No time2id.txt")
        return

    dates = pd.read_csv(time_path)

    if not {"TimeID", "DATE_WK"}.issubset(dates.columns):
        record(
            name, "graph_dates", "FAIL",
            "Unexpected time2id.txt schema",
        )
        return

    dates["TimeID"] = pd.to_numeric(
        dates["TimeID"], errors="raise"
    ).astype(int)
    dates["DATE_WK"] = pd.to_datetime(
        dates["DATE_WK"], errors="raise"
    )

    if dates["TimeID"].duplicated().any():
        record(
            name, "graph_dates", "FAIL",
            "Duplicate time IDs",
        )
        return

    lookup = dates.set_index("TimeID")["DATE_WK"]

    all_frames = []

    # Original DKG event rows:
    # subject_id relation_id object_id time_id
    for split, filename in [
        ("train", "train.txt"),
        ("valid", "valid.txt"),
        ("test", "test.txt"),
    ]:
        path = folder / filename

        if not path.exists():
            record(
                name, f"graph_{split}",
                "UNKNOWN", "File missing",
            )
            continue

        rows = []
        bad = 0

        with path.open() as f:
            for line in f:
                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 4:
                    bad += 1
                    continue

                try:
                    rows.append(
                        [int(x) for x in parts]
                    )
                except ValueError:
                    bad += 1

        if not rows:
            record(
                name, f"graph_{split}",
                "FAIL", f"No readable events; invalid={bad}",
            )
            continue

        frame = pd.DataFrame(
            rows, columns=[
                "subject", "relation",
                "object", "time_id",
            ],
        )

        frame["date"] = frame["time_id"].map(lookup)

        if frame["date"].isna().any():
            record(
                name, f"graph_{split}",
                "FAIL", "Unmapped time IDs",
            )
            continue

        lo = frame["date"].min()
        hi = frame["date"].max()

        boundaries = {
            "train": (
                pd.Timestamp("2018-01-01"),
                pd.Timestamp("2021-12-31"),
            ),
            "valid": (
                pd.Timestamp("2022-01-01"),
                pd.Timestamp("2022-12-31"),
            ),
            "test": (
                pd.Timestamp("2023-01-01"),
                pd.Timestamp("2023-12-31"),
            ),
        }

        expected_lo, expected_hi = boundaries[split]
        good = (
            lo >= expected_lo
            and hi <= expected_hi
            and bad == 0
        )

        record(
            name,
            f"graph_{split}",
            "PASS" if good else "FAIL",
            f"events={len(frame)}, "
            f"weeks={frame['time_id'].nunique()}, "
            f"range={lo.date()}..{hi.date()}, "
            f"invalid_lines={bad}",
        )

        frame["split"] = split
        all_frames.append(frame)

    if all_frames:
        combined = pd.concat(
            all_frames, ignore_index=True
        )

        per_week = (
            combined.groupby(["split", "date"])
            .size().reset_index(name="events")
        )

        per_week.to_csv(
            OUT / f"{name.lower()}_events_per_week.csv",
            index=False,
        )


def inspect_embeddings(name, base):
    weekly_path = base / "weekly_embeddings_64d.csv"
    daily_path = base / "original_daily_embeddings.csv"
    pca_path = base / "train_only_pca.npz"

    if not weekly_path.exists() or not daily_path.exists():
        record(
            name, "embeddings", "UNKNOWN",
            "Weekly or daily CSV missing",
        )
        return

    weekly = pd.read_csv(weekly_path)
    daily = pd.read_csv(daily_path)

    weekly["week_date"] = pd.to_datetime(
        weekly["week_date"], errors="raise"
    )
    daily["date"] = pd.to_datetime(
        daily["date"], errors="raise"
    )

    features64 = [
        f"emb_{i:02d}" for i in range(64)
    ]
    features11 = [
        f"dkg_emb_{i:02d}" for i in range(11)
    ]

    required_week = {
        "week_date", "tic", *features64
    }
    required_daily = {
        "date", "tic", *features11
    }

    if not required_week.issubset(weekly.columns):
        record(
            name, "weekly_schema", "FAIL",
            "Missing 64D feature columns",
        )
        return

    if not required_daily.issubset(daily.columns):
        record(
            name, "daily_schema", "FAIL",
            "Missing 11D feature columns",
        )
        return

    tickers = {
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
    }

    for label, df, date_col, keys, columns in [
        (
            "weekly", weekly, "week_date",
            ["week_date", "tic"], features64,
        ),
        (
            "daily", daily, "date",
            ["date", "tic"], features11,
        ),
    ]:
        finite = np.isfinite(
            df[columns].to_numpy(dtype=float)
        ).all()

        unique = not df.duplicated(keys).any()
        correct = set(df["tic"]) == tickers

        record(
            name, f"{label}_integrity",
            "PASS" if finite and unique and correct else "FAIL",
            f"rows={len(df)}, "
            f"first={df[date_col].min().date()}, "
            f"last={df[date_col].max().date()}, "
            f"finite={finite}, "
            f"unique={unique}, "
            f"tickers_correct={correct}",
        )

    # Check whether each stock's features actually
    # change across the 2023 trading interval.
    test = daily[
        daily["date"].dt.year == 2023
    ].copy()

    variations = {}

    for ticker in sorted(tickers):
        rows = test[test["tic"] == ticker]
        variations[ticker] = (
            len(rows[features11].drop_duplicates())
            if len(rows)
            else 0
        )

    record(
        name, "2023_distinct_daily_vectors",
        "OBSERVATION",
        variations,
    )

    # Validate stored PCA transformation against
    # the weekly 64D input, including train-only mean
    # and standard deviation.
    if not pca_path.exists():
        record(
            name, "pca_artifact",
            "UNKNOWN", "PCA file missing",
        )
        return

    with np.load(pca_path) as pca:
        keys = set(pca.files)

        required = {
            "mean", "scale", "components", "center"
        }

        if not required.issubset(keys):
            record(
                name, "pca_artifact", "FAIL",
                f"Missing keys: {required - keys}",
            )
            return

        train = weekly[
            weekly["week_date"] < "2022-01-01"
        ][features64].to_numpy(dtype=float)

        mean = pca["mean"]
        scale = pca["scale"]

        expected_mean = train.mean(axis=0)
        expected_std = train.std(axis=0)
        expected_std[expected_std < 1e-8] = 1.0

        good = (
            np.allclose(
                mean, expected_mean,
                rtol=1e-6, atol=1e-7,
            )
            and np.allclose(
                scale, expected_std,
                rtol=1e-6, atol=1e-7,
            )
            and pca["components"].shape == (11, 64)
        )

        record(
            name, "pca_train_only_statistics",
            "PASS" if good else "FAIL",
            "Stored PCA preprocessing matches "
            "2018-2021 weekly features"
            if good
            else "Stored PCA preprocessing differs "
                 "from 2018-2021 weekly statistics",
        )


def inspect_exporter(name):
    filename = (
        "export_original_kgtransformer.py"
        if name == "FULL_15"
        else "export_four_native_kgtransformer.py"
    )
    path = ROOT / "scripts" / filename

    if not path.exists():
        record(
            name, "exporter_source", "UNKNOWN",
            f"Missing {filename}",
        )
        return

    source = path.read_text()

    indicators = {
        "prior_state_comment": (
            "State BEFORE this week's edges" in source
        ),
        "node_time_reset": (
            "node_latest_event_time.zero_" in source
        ),
        "model_strict_load": (
            "strict=True" in source
        ),
        "train_only_pca_filter": (
            'weekly["week_date"] < "2022-01-01"'
            in source
        ),
    }

    record(
        name, "exporter_code_indicators",
        "CODE_REVIEW_NEEDED",
        indicators,
    )

    # Source-code indicators do not prove
    # the actual execution-time information boundary.


for name, base in CONFIGS.items():
    print(f"\n{'=' * 65}")
    print("AUDITING:", name)
    print("=" * 65)

    inspect_manifest(name, base)
    inspect_graph(name, base)
    inspect_embeddings(name, base)
    inspect_exporter(name)

    record(
        name, "final_strict_pit_approval",
        "NOT_CERTIFIED",
        "Requires walk-forward/OOF model training, "
        "event publication-time verification, "
        "and FinRL execution-timing review",
    )


result = pd.DataFrame(RESULTS)

csv_path = OUT / "pit_audit_results.csv"
json_path = OUT / "pit_audit_results.json"

result.to_csv(csv_path, index=False)
json_path.write_text(
    json.dumps(
        RESULTS, indent=2,
        ensure_ascii=False, default=str
    ),
    encoding="utf-8",
)

print("\n=== STRICT PIT AUDIT SUMMARY ===")
print(
    result[
        ["model", "check", "status", "detail"]
    ].to_string(index=False)
)

print("\nCSV:", csv_path)
print("JSON:", json_path)
print("\nSTRICT PIT: NOT CERTIFIED")
