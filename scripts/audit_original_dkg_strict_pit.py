#!/usr/bin/env python3
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original_full"
OUT = BASE / "strict_pit"
OUT.mkdir(parents=True, exist_ok=True)

FUSION = ROOT / "outputs/v20/fusion"
V35 = ROOT / "outputs/v35/evidence_ledger.csv"
RUNNER = ROOT / "scripts/run_v21_dynamic_ablation.py"

WEEKLY = BASE / "weekly_embeddings_64d.csv"
DAILY = BASE / "original_daily_embeddings.csv"
PCA = BASE / "train_only_pca.npz"
MANIFEST = BASE / "original_embedding_manifest.json"

TICKERS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}

F = [f"dkg_emb_{i:02d}" for i in range(11)]
E = [f"emb_{i:02d}" for i in range(64)]

results = []


def record(section, status, detail, severity="INFO"):
    item = {
        "section": section,
        "status": status,
        "severity": severity,
        "detail": str(detail),
    }
    results.append(item)
    print(f"[{status}] {section}: {detail}")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024*1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def check_sources():
    required = {
        "weekly embeddings": WEEKLY,
        "daily embeddings": DAILY,
        "PCA": PCA,
        "manifest": MANIFEST,
        "market": FUSION / "dataset_conservative.csv",
        "SEC ledger": V35,
        "FinRL runner": RUNNER,
    }

    missing = []

    for name, path in required.items():
        if not path.is_file():
            record(
                "File: " + name,
                "FAIL",
                path,
                "BLOCKER",
            )
            missing.append(name)
        else:
            record(
                "File: " + name,
                "PASS",
                sha256(path)[:16],
            )

    return not missing


def check_embeddings():
    weekly = pd.read_csv(WEEKLY)
    daily = pd.read_csv(DAILY)

    weekly["week_date"] = pd.to_datetime(
        weekly["week_date"],
        errors="coerce",
    )
    daily["date"] = pd.to_datetime(
        daily["date"],
        errors="coerce",
    )

    if not set(E).issubset(weekly.columns):
        record(
            "Weekly 64 dimensions",
            "FAIL", "Missing columns", "BLOCKER"
        )
        return

    if not set(F).issubset(daily.columns):
        record(
            "Daily 11 dimensions",
            "FAIL", "Missing columns", "BLOCKER"
        )
        return

    invalid = (
        weekly["week_date"].isna().any()
        or daily["date"].isna().any()
        or not np.isfinite(
            weekly[E].to_numpy(float)
        ).all()
        or not np.isfinite(
            daily[F].to_numpy(float)
        ).all()
    )

    record(
        "Embedding finite/date",
        "FAIL" if invalid else "PASS",
        f"weekly={len(weekly)} daily={len(daily)}",
        "BLOCKER" if invalid else "INFO",
    )

    duplicate = (
        weekly.duplicated(
            ["time_id", "tic"]
        ).any()
        or daily.duplicated(
            ["date", "tic"]
        ).any()
    )

    record(
        "Unique ticker-time",
        "FAIL" if duplicate else "PASS",
        f"duplicate={bool(duplicate)}",
        "BLOCKER" if duplicate else "INFO",
    )

    market = pd.read_csv(
        FUSION / "dataset_conservative.csv",
        low_memory=False,
    )

    date_col = "date" if "date" in market else "Date"
    tic_col = "tic" if "tic" in market else "ticker"

    if date_col not in market or tic_col not in market:
        record(
            "Market keys",
            "FAIL",
            f"Columns: {list(market.columns)}",
            "BLOCKER",
        )
        return

    expected = market[
        [date_col, tic_col]
    ].copy()

    expected.columns = ["date", "tic"]
    expected["date"] = pd.to_datetime(
        expected["date"]
    )

    left = expected.sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    right = daily[
        ["date", "tic"]
    ].sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

    aligned = left.equals(right)

    record(
        "Market / KG day alignment",
        "PASS" if aligned else "FAIL",
        f"aligned={aligned}",
        "BLOCKER" if not aligned else "INFO",
    )

    # Recalculate daily as-of from weekly snapshots.
    # It must match the exporter's exact date convention.
    compared = 0
    maximum_error = 0.0
    missing = 0
    timing_fail = 0

    for ticker in sorted(TICKERS):
        w = weekly[
            weekly["tic"].eq(ticker)
        ].sort_values("week_date").copy()

        d = daily[
            daily["tic"].eq(ticker)
        ].sort_values("date").copy()

        if w.empty or d.empty:
            record(
                "Ticker coverage",
                "FAIL", ticker, "BLOCKER"
            )
            continue

        # Need the stored train-only PCA to recompute features.
        with np.load(PCA) as transform:
            mean = transform["mean"]
            scale = transform["scale"]
            components = transform["components"]
            center = transform["center"]

        source = w[E].to_numpy(float)
        transformed = (
            ((source - mean) / scale) - center
        ) @ components.T

        source11 = w[["week_date"]].copy()

        for j, feature in enumerate(F):
            source11[feature] = transformed[:, j]

        merged = pd.merge_asof(
            d[["date"] + F],
            source11.sort_values("week_date"),
            left_on="date",
            right_on="week_date",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("_daily", "_expected"),
        )

        for _, r in merged.iterrows():
            if pd.isna(r["week_date"]):
                target = np.zeros(11)
                missing += 1
            else:
                target = np.array([
                    r[f"{f}_expected"]
                    for f in F
                ], dtype=float)
                if not (r["week_date"] < r["date"]):
                    timing_fail += 1

            actual = np.array([
                r[f"{f}_daily"] for f in F
            ], dtype=float)

            err = np.max(np.abs(actual - target))
            maximum_error = max(
                maximum_error, float(err)
            )
            compared += 1

    matched = (
        compared == len(daily)
        and maximum_error <= 1e-6
        and timing_fail == 0
    )

    record(
        "Daily reconstruction from weekly",
        "PASS" if matched else "FAIL",
        (
            f"rows={compared}; max_error={maximum_error:.3g}; "
            f"no_prior_snapshot={missing}; "
            f"timing_violations={timing_fail}"
        ),
        "BLOCKER" if not matched else "INFO",
    )

    # The source's event date is not automatically
    # an independently verified publication timestamp.
    record(
        "KG actual public release time",
        "NOT CERTIFIED",
        "FinDKG weekly event dates are not verified "
        "publication timestamps for each event.",
        "BLOCKER",
    )


def check_pca_and_checkpoint():
    meta = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    with np.load(PCA) as values:
        shapes = {
            key: tuple(values[key].shape)
            for key in values.files
        }

    valid_shapes = (
        shapes.get("mean") == (64,)
        and shapes.get("scale") == (64,)
        and shapes.get("components") == (11, 64)
        and shapes.get("center") == (64,)
    )

    record(
        "PCA artifact dimensions",
        "PASS" if valid_shapes else "FAIL",
        str(shapes),
        "BLOCKER" if not valid_shapes else "INFO",
    )

    record(
        "PCA fitting provenance",
        "DOCUMENTED",
        (
            "Manifest fit period: "
            + str(meta.get("pca_fit_period"))
            + ". Dates are declared, not independently "
            "proven by transform artifact alone."
        ),
    )

    ckpt = Path(
        meta.get("checkpoint_path", "")
    )

    if ckpt.exists():
        actual_hash = sha256(ckpt)
        declared_hash = meta.get("checkpoint_sha256")
        ok = actual_hash == declared_hash
        record(
            "Original checkpoint SHA256",
            "PASS" if ok else "FAIL",
            f"hash_match={ok}",
            "BLOCKER" if not ok else "INFO",
        )
    else:
        record(
            "Original checkpoint SHA256",
            "FAIL", "checkpoint missing", "BLOCKER"
        )

    record(
        "Original KG model selection",
        "DISCLOSED",
        "2022 validation was used to choose "
        "best checkpoint; 2023 is the test period.",
    )

    record(
        "Historical train embedding leakage",
        "NOT CERTIFIED",
        "2018-2021 training features were replayed "
        "using a final checkpoint selected after "
        "observing later data, including validation. "
        "Strict historical walk-forward not established.",
        "BLOCKER",
    )


def check_sec():
    sec = pd.read_csv(
        V35, low_memory=False
    )

    print("V35 rows:", len(sec))

    flags = [
        "human_verified",
        "strict_pit_approved",
        "scope_verified",
    ]

    def positive(s):
        return (
            s.astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "yes"])
        )

    for flag in flags:
        if flag in sec.columns:
            count = int(positive(sec[flag]).sum())
            record(
                "SEC " + flag,
                "DOCUMENTED",
                f"{count}/{len(sec)} flagged true",
            )

    if {
        "filing_date", "available_from"
    }.issubset(sec.columns):
        filing = pd.to_datetime(
            sec["filing_date"],
            errors="coerce",
            utc=True,
        )
        available = pd.to_datetime(
            sec["available_from"],
            errors="coerce",
            utc=True,
        )

        invalid = (
            (available < filing)
            | filing.isna()
            | available.isna()
        )

        record(
            "SEC date-order sanity",
            "FAIL" if invalid.any() else "PASS",
            f"invalid_rows={int(invalid.sum())}",
            "BLOCKER" if invalid.any() else "INFO",
        )

    record(
        "SEC acceptance timestamp",
        "NOT CERTIFIED",
        "V35 filing_date and available_from are "
        "not enough to prove actual SEC EDGAR "
        "acceptance/public dissemination timestamp.",
        "BLOCKER",
    )


def check_trade_timing():
    if not RUNNER.exists():
        record(
            "FinRL timing source",
            "FAIL", "runner missing", "BLOCKER"
        )
        return

    content = RUNNER.read_text(
        encoding="utf-8", errors="replace"
    )

    try:
        ast.parse(content)
        record(
            "FinRL runner Python syntax",
            "PASS", "AST parse succeeded"
        )
    except SyntaxError as e:
        record(
            "FinRL runner Python syntax",
            "FAIL", str(e), "BLOCKER"
        )

    lines = content.splitlines()

    keywords = [
        "StockTradingEnv",
        "step(",
        "rebalance",
        "execution",
        "trade",
        "date",
        "price",
        "shift(",
    ]

    hits = []

    for i, line in enumerate(lines, start=1):
        if any(
            keyword.lower() in line.lower()
            for keyword in keywords
        ):
            hits.append({
                "line_number": i,
                "source": line.strip()[:240],
            })

    pd.DataFrame(
        hits,
        columns=["line_number", "source"],
    ).to_csv(
        OUT / "finrl_timing_source_hits.csv",
        index=False,
    )

    record(
        "FinRL action/execution time",
        "NOT CERTIFIED",
        "Saved source-line evidence for manual review. "
        "Need to trace observation date, action date, "
        "execution price and portfolio update.",
        "BLOCKER",
    )


def check_four_relations():
    path = (
        BASE
        / "four_relation_kg/manifest.json"
    )

    if not path.exists():
        record(
            "Four-relation graph",
            "NOT CERTIFIED",
            "Run four-relation builder first",
            "BLOCKER",
        )
        return

    meta = json.loads(
        path.read_text(encoding="utf-8")
    )

    ready = bool(
        meta.get("training_ready", False)
    )

    record(
        "Four relation coverage",
        "PASS" if ready else "NOT CERTIFIED",
        f"training_ready={ready}; "
        f"verified_events={meta.get('events')}",
        "BLOCKER" if not ready else "INFO",
    )

    record(
        "Four-relation KGTransformer model",
        "NOT CERTIFIED",
        "Four-relation KG has not been independently "
        "trained and integrated into FinRL-X.",
        "BLOCKER",
    )


def main():
    available = check_sources()

    if available:
        check_embeddings()
        check_pca_and_checkpoint()
        check_sec()
        check_trade_timing()
        check_four_relations()

    df = pd.DataFrame(results)

    path = OUT / "strict_pit_report.csv"
    df.to_csv(path, index=False)

    blockers = df[
        df["severity"].eq("BLOCKER")
    ]

    approved = (
        bool(len(df))
        and blockers.empty
    )

    result = {
        "strict_pit_approved": approved,
        "checks": len(df),
        "blockers": len(blockers),
        "status": (
            "APPROVED" if approved
            else "NOT CERTIFIED"
        ),
    }

    (OUT / "strict_pit_result.json").write_text(
        json.dumps(result, indent=2)
    )

    print("\n=== STRICT PIT FINAL ===")
    print(json.dumps(result, indent=2))
    print("Full report:", path)


if __name__ == "__main__":
    main()
