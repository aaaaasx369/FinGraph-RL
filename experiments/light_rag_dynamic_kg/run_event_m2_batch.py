#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PYTHON = ROOT / ".venv-drl/bin/python"

OFFICIAL = ROOT / (
    "experiments/light_rag_dynamic_kg/"
    "run_official_lightrag.py"
)

EVENTS = ROOT / (
    "results/light_rag_dynamic_kg/"
    "event_driven_v3/event_jobs.csv"
)

BASE = ROOT / "results/light_rag_dynamic_kg"

OUT = BASE / "event_driven_m2_v1"

TICKERS = [
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
]


def free_gib():
    return (
        shutil.disk_usage(ROOT).free
        / (1024 ** 3)
    )


def existing_output(ticker, date):
    return (
        BASE / f"lightrag_{ticker}_{date}.json"
    )


def choose_jobs():
    df = pd.read_csv(EVENTS)

    required = {
        "ticker",
        "date",
        "status",
        "new_kg_events",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing candidate columns: {missing}"
        )

    df = df.loc[
        df["status"].eq("CANDIDATE")
    ].copy()

    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    )

    df["new_kg_events"] = pd.to_numeric(
        df["new_kg_events"],
        errors="coerce",
    ).fillna(0)

    # Larger event batches first.
    df = df.sort_values(
        ["new_kg_events", "date"],
        ascending=[False, True],
    )

    selected = []

    for ticker in TICKERS:
        company = df.loc[
            df["ticker"].eq(ticker)
        ].copy()

        # Prefer 4 different months in training.
        train = company.loc[
            company["date"].dt.year.between(
                2018, 2020
            )
        ].copy()

        used_months = set()
        train_picks = []

        for row in train.itertuples(index=False):
            month = row.date.strftime("%Y-%m")

            if month in used_months:
                continue

            used_months.add(month)
            train_picks.append(row)

            if len(train_picks) == 4:
                break

        if len(train_picks) < 4:
            used_dates = {
                r.date for r in train_picks
            }

            for row in train.itertuples(index=False):
                if row.date in used_dates:
                    continue

                train_picks.append(row)
                used_dates.add(row.date)

                if len(train_picks) == 4:
                    break

        for row in train_picks:
            selected.append({
                "ticker": ticker,
                "date": row.date.strftime(
                    "%Y-%m-%d"
                ),
                "split": "TRAIN",
                "new_kg_events": int(
                    row.new_kg_events
                ),
            })

        # One validation event per company.
        val = company.loc[
            company["date"].dt.year.eq(2021)
        ]

        if len(val):
            row = next(
                val.itertuples(index=False)
            )

            selected.append({
                "ticker": ticker,
                "date": row.date.strftime(
                    "%Y-%m-%d"
                ),
                "split": "VAL",
                "new_kg_events": int(
                    row.new_kg_events
                ),
            })

    return selected


def is_completed(path, ticker, date):
    if not path.exists():
        return False

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        # Never overwrite an unknown/broken output.
        return False

    return (
        data.get("ticker") == ticker
        and data.get("decision_date") == date
        and isinstance(data.get("answer"), str)
        and bool(data["answer"].strip())
    )


def append_status(record):
    path = OUT / "batch_status.jsonl"

    with path.open(
        "a", encoding="utf-8"
    ) as f:
        f.write(
            json.dumps(
                record, ensure_ascii=False
            ) + "\n"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=20,
    )

    parser.add_argument(
        "--kg-k",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    logs = OUT / "logs"
    logs.mkdir(exist_ok=True)

    if not PYTHON.exists():
        raise FileNotFoundError(PYTHON)

    if not OFFICIAL.exists():
        raise FileNotFoundError(OFFICIAL)

    if not EVENTS.exists():
        raise FileNotFoundError(EVENTS)

    jobs_path = OUT / "selected_jobs.csv"

    if jobs_path.exists():
        jobs = pd.read_csv(
            jobs_path
        ).to_dict("records")

        print(
            "[RESUME] Reusing previous job list",
            flush=True,
        )
    else:
        jobs = choose_jobs()

        pd.DataFrame(jobs).to_csv(
            jobs_path,
            index=False,
        )

    print("=" * 78, flush=True)
    print("EVENT-DRIVEN M2 SNAPSHOT BATCH", flush=True)
    print("=" * 78, flush=True)
    print("Jobs:", len(jobs), flush=True)
    print("Free GiB:", round(free_gib(), 2), flush=True)

    completed = 0
    cached = 0
    failed = 0

    for i, job in enumerate(jobs, 1):
        ticker = str(job["ticker"])
        date = str(job["date"])
        split = str(job["split"])

        tag = f"{ticker}_{date}"
        output = existing_output(ticker, date)

        print(
            f"\n[{i}/{len(jobs)}] "
            f"{tag} {split}",
            flush=True,
        )

        if output.exists():
            if is_completed(output, ticker, date):
                print(
                    "[CACHED] Existing valid JSON",
                    flush=True,
                )

                cached += 1

                append_status({
                    "ticker": ticker,
                    "date": date,
                    "split": split,
                    "status": "CACHED",
                })

                continue

            # Do not overwrite malformed or unknown output.
            print(
                "[SKIP] Output exists but is not "
                "validated. Manual inspection needed.",
                flush=True,
            )

            failed += 1

            append_status({
                "ticker": ticker,
                "date": date,
                "split": split,
                "status": "EXISTING_UNVERIFIED",
            })

            continue

        free = free_gib()

        if free < args.min_free_gib:
            print(
                f"[STOP] Free space {free:.2f} GiB "
                f"< {args.min_free_gib:.2f} GiB",
                flush=True,
            )
            break

        command = [
            str(PYTHON),
            "-u",
            str(OFFICIAL),
            "--ticker", ticker,
            "--date", date,
            "--kg-k", str(args.kg_k),
            "--sec-k", "0",
        ]

        print(
            "[BUILD]",
            " ".join(command),
            flush=True,
        )

        start = time.monotonic()

        logfile = logs / f"{tag}.log"

        with logfile.open(
            "w", encoding="utf-8"
        ) as f:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )

        elapsed = time.monotonic() - start

        ok = (
            result.returncode == 0
            and is_completed(
                output, ticker, date
            )
        )

        status = (
            "COMPLETE" if ok else "FAILED"
        )

        if ok:
            completed += 1
        else:
            failed += 1

        print(
            f"[RESULT] {tag} "
            f"{status} "
            f"elapsed={elapsed:.1f}s "
            f"returncode={result.returncode}",
            flush=True,
        )

        append_status({
            "ticker": ticker,
            "date": date,
            "split": split,
            "status": status,
            "returncode": result.returncode,
            "elapsed_seconds": round(
                elapsed, 2
            ),
            "log": str(logfile),
        })

    print("\n" + "=" * 78, flush=True)
    print("BATCH FINISHED", flush=True)
    print("New complete:", completed, flush=True)
    print("Cached:", cached, flush=True)
    print("Failed or unverified:", failed, flush=True)
    print(
        "Free GiB:",
        round(free_gib(), 2),
        flush=True,
    )
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
