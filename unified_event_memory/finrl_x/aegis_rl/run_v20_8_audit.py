#!/usr/bin/env python3

import sys
import json
import hashlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import run_v16_dual_rag_finrl as v16

from unified_event_memory.finrl_x.aegis_rl import (
    run_aegis_evidence_portfolio as source
)

OUT = Path("results/aegis_rl_v1/v20_blind")
TEST_FILE = OUT / "test_2022_2023_SAFE.npz"
TRAIN_FILE = Path(
    "results/aegis_rl_v1/v18_1/ppo_full/"
    "joint_feature_tensors.npz"
)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

report = {
    "version": "V20.8",
    "test_performance_accessed": False,
    "checks": {},
    "warnings": [],
    "status": "NOT_CERTIFIED",
}


def record(name, passed, detail=None):
    report["checks"][name] = {
        "passed": bool(passed),
        "detail": detail,
    }
    print(
        "[PASS]" if passed else "[FAIL]",
        name,
        detail if detail is not None else "",
        flush=True
    )


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            h.update(chunk)
    return h.hexdigest()


def save_report():
    path = OUT / "v20_8_audit_report.json"
    path.write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )
    print("\nReport:", path)


def main():
    print("\n========== V20.8 TEST STRUCTURE ==========")

    if not TEST_FILE.exists():
        raise RuntimeError(
            f"Test staging file not found: {TEST_FILE}"
        )

    with np.load(
        TEST_FILE, allow_pickle=False
    ) as z:
        names = list(z.files)
        X = np.asarray(
            z["features"], dtype=np.float32
        )
        dates = pd.DatetimeIndex(
            pd.to_datetime(z["dates"].astype(str))
        )
        tickers = list(z["tickers"].astype(str))
        prices = np.asarray(z["prices"])

    record(
        "test_shape",
        X.shape == (501, 5, 328),
        list(X.shape)
    )

    record(
        "ticker_order",
        tickers == TICKERS,
        tickers
    )

    record(
        "date_range",
        (
            dates.min() >= pd.Timestamp("2022-01-01")
            and dates.max() <= pd.Timestamp("2023-12-31")
            and dates.is_unique
            and dates.is_monotonic_increasing
        ),
        [
            str(dates.min().date()),
            str(dates.max().date())
        ]
    )

    record(
        "finite_features",
        bool(np.isfinite(X).all())
    )

    record(
        "valid_prices",
        bool(
            prices.shape == (len(dates), 5)
            and np.isfinite(prices).all()
            and (prices > 0).all()
        )
    )

    record(
        "independent_joint_zero",
        bool(np.array_equal(
            X[:, :, -9:],
            np.zeros_like(X[:, :, -9:])
        ))
    )

    print("\n========== ORIGINAL MARKET CONTRACT ==========")

    schema, extra = source.load_schema()
    original = source.load_market(
        "conservative", schema, extra
    )

    market_cols = schema["FULL_DYNAMIC"]

    original["date"] = pd.to_datetime(
        original["date"]
    ).dt.normalize()

    history_dates = pd.DatetimeIndex(
        sorted(
            original.loc[
                original["date"] < "2022-01-01",
                "date"
            ].unique()
        )
    )

    panel = original.set_index(["date", "ticker"])

    original_market = np.empty(
        (len(history_dates), 5, 53),
        dtype=np.float32
    )

    for j, ticker in enumerate(TICKERS):
        keys = pd.MultiIndex.from_arrays(
            [
                history_dates,
                np.repeat(ticker, len(history_dates))
            ],
            names=["date", "ticker"]
        )

        original_market[:, j] = panel.reindex(
            keys
        )[market_cols].to_numpy(np.float32)

    expected_market = np.concatenate(
        [
            original_market,
            original_market
            - original_market.mean(
                axis=1, keepdims=True
            )
        ],
        axis=2
    ).astype(np.float32)

    record(
        "original_market_106D",
        (
            expected_market.shape
            == (1008, 5, 106)
            and np.isfinite(expected_market).all()
        ),
        list(expected_market.shape)
    )

    print("\n========== LOADER PARITY ==========")

    try:
        old_argv = sys.argv
        try:
            sys.argv = ["run_v16_dual_rag_finrl.py"]
            args = v16.parse_args()
        finally:
            sys.argv = old_argv

        args.algo = "A2C"

        train, val, historical_dates, contexts, n_train = (
            v16.v8b.load_data(args)
        )

        historical_dates = (
            v16.base.normalize_dates(historical_dates)
        )

        market_flat = np.concatenate(
            [train["market"], val["market"]],
            axis=0
        )

        trained_market, trained_kgt = (
            v16.base.validate_market_layout(
                market_flat
            )
        )

        same_dates = historical_dates.equals(
            history_dates
        )

        same_shape = (
            trained_market.shape
            == expected_market.shape
        )

        if same_dates and same_shape:
            max_diff = float(
                np.max(
                    np.abs(
                        trained_market
                        - expected_market
                    )
                )
            )
        else:
            max_diff = None

        record(
            "market_training_parity",
            (
                same_dates
                and same_shape
                and max_diff == 0.0
            ),
            {
                "date_match": same_dates,
                "shape_match": same_shape,
                "max_absolute_difference": max_diff,
            }
        )

    except Exception as exc:
        record(
            "market_training_parity",
            False,
            {
                "error": repr(exc),
                "traceback": traceback.format_exc()[-2000:]
            }
        )

    print("\n========== HISTORICAL TENSOR INVENTORY ==========")

    if TRAIN_FILE.exists():
        import zipfile
        import numpy.lib.format as fmt

        inventory = {}

        with zipfile.ZipFile(TRAIN_FILE) as archive:
            for name in archive.namelist():
                if not name.endswith(".npy"):
                    continue

                with archive.open(name) as stream:
                    version = fmt.read_magic(stream)

                    if version == (1, 0):
                        shape, fortran, dtype = (
                            fmt.read_array_header_1_0(stream)
                        )
                    elif version in ((2, 0), (3, 0)):
                        shape, fortran, dtype = (
                            fmt.read_array_header_2_0(stream)
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported NPY version: {version}"
                        )

                inventory[name[:-4]] = {
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "object_array": bool(dtype.hasobject),
                }

        report["historical_tensor_inventory"] = inventory

        print(
            json.dumps(
                inventory,
                indent=2,
                ensure_ascii=False
            )
        )

        record(
            "historical_tensor_available",
            True,
            str(TRAIN_FILE)
        )
    else:
        record(
            "historical_tensor_available",
            False,
            str(TRAIN_FILE)
        )

    print("\n========== SEC AUDIT ==========")

    sec_audit = (
        OUT / "v20_7_sec_retrieval_audit.csv"
    )

    if sec_audit.exists():
        sec = pd.read_csv(
            sec_audit, low_memory=False
        )

        print("SEC audit rows:", len(sec))
        print("SEC columns:", sec.columns.tolist())
        print(sec.head(3).to_string(index=False))

        report["sec_audit"] = {
            "rows": len(sec),
            "columns": sec.columns.tolist(),
            "sha256": sha256(sec_audit),
        }

        day_col = next(
            (
                c for c in (
                    "date", "decision_date",
                    "trading_date"
                )
                if c in sec.columns
            ),
            None
        )

        available_col = next(
            (
                c for c in (
                    "available_at",
                    "available_from",
                    "filing_available_at"
                )
                if c in sec.columns
            ),
            None
        )

        if day_col and available_col:
            decision = pd.to_datetime(
                sec[day_col], utc=True,
                errors="coerce"
            )

            available = pd.to_datetime(
                sec[available_col], utc=True,
                errors="coerce"
            )

            ok = bool(
                decision.notna().all()
                and available.notna().all()
                and (available < decision).all()
            )

            record(
                "sec_audit_timestamp_gate",
                ok,
                {
                    "decision_column": day_col,
                    "available_column": available_col,
                    "rows": len(sec)
                }
            )
        else:
            record(
                "sec_audit_timestamp_gate",
                False,
                "Timestamp columns not directly verifiable"
            )

    else:
        record(
            "sec_audit_timestamp_gate",
            False,
            "SEC audit missing"
        )

    print("\n========== KG AUDIT ==========")

    kg_audit = OUT / "kg_retrieval_audit.csv"

    if kg_audit.exists():
        kg = pd.read_csv(
            kg_audit, low_memory=False
        )

        print("KG audit rows:", len(kg))
        print("KG columns:", kg.columns.tolist())
        print(kg.head(3).to_string(index=False))

        report["kg_audit"] = {
            "rows": len(kg),
            "columns": kg.columns.tolist(),
            "sha256": sha256(kg_audit),
        }

        if {
            "date",
            "latest_observed_date"
        }.issubset(kg.columns):

            day = pd.to_datetime(
                kg["date"], errors="coerce"
            )
            observed = pd.to_datetime(
                kg["latest_observed_date"],
                errors="coerce"
            )

            ok = bool(
                day.notna().all()
                and observed.notna().all()
                and (observed < day).all()
            )

            record(
                "kg_observation_date_gate",
                ok,
                {
                    "rows": len(kg),
                    "note": (
                        "Observation-date check only; "
                        "public availability not certified."
                    )
                }
            )
        else:
            record(
                "kg_observation_date_gate",
                False,
                "Required KG audit columns unavailable"
            )
    else:
        record(
            "kg_observation_date_gate",
            False,
            "KG audit missing"
        )

    print("\n========== PIT CERTIFICATION STATUS ==========")

    report["warnings"] = [
        "Market source column-level PIT lineage not independently verified.",
        "Same-day Market feature availability versus execution timing unresolved.",
        "SEC audit timestamps need source-provenance validation.",
        "KG first-observed date is not necessarily public release date.",
        "V18.1 full SEC/KG numerical parity has not yet been established.",
        "A matching shape or nonzero ratio is not sufficient certification.",
    ]

    # Intentionally not certifying from structural checks alone.
    report["status"] = "NOT_CERTIFIED"

    save_report()

    print("\n========== V20.8 SUMMARY ==========")

    for name, info in report["checks"].items():
        print(
            name,
            "PASS" if info["passed"] else "FAIL"
        )

    print("\nTEST FEATURE STATUS: UNCERTIFIED")
    print("2022-2023 PPO PERFORMANCE: NOT ACCESSED")


if __name__ == "__main__":
    main()
