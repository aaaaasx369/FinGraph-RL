#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "results/light_rag_dynamic_kg"

PLAN = BASE / "event_driven_v3/event_jobs.csv"

OUT = BASE / "event_driven_full_v1"

RUNNER = ROOT / (
    "experiments/light_rag_dynamic_kg/"
    "run_official_lightrag.py"
)

PYTHON = ROOT / ".venv-drl/bin/python"

TICKERS = {
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
}


def free_gib():
    return shutil.disk_usage(ROOT).free / (1024 ** 3)


def result_path(ticker, date):
    return BASE / f"lightrag_{ticker}_{date}.json"


def result_valid(path, ticker, date):
    if not path.is_file():
        return False

    try:
        obj = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False

    return (
        obj.get("ticker") == ticker
        and obj.get("decision_date") == date
        and isinstance(obj.get("answer"), str)
        and bool(obj["answer"].strip())
    )


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(value, ensure_ascii=False)
            + "\n"
        )


def load_jobs():
    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)

    df = pd.read_csv(PLAN, dtype=str)

    required = {
        "ticker", "date", "status"
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    df = df.loc[
        df["status"] == "CANDIDATE"
    ].copy()

    if len(df) != 483:
        raise RuntimeError(
            f"Expected 483 candidates, got {len(df)}"
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    ).dt.strftime("%Y-%m-%d")

    df = df[["ticker", "date"]].copy()

    if not set(df["ticker"]).issubset(TICKERS):
        raise RuntimeError("Unexpected ticker")

    if df.duplicated(["ticker", "date"]).any():
        raise RuntimeError("Duplicate candidate jobs")

    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    return df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--kg-k",
        type=int,
        default=80,
    )

    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    logdir = OUT / "logs"
    logdir.mkdir(exist_ok=True)

    if not PYTHON.is_file():
        raise FileNotFoundError(PYTHON)

    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)

    jobs = load_jobs()

    job_path = OUT / "all_483_jobs.csv"

    if job_path.exists():
        existing = pd.read_csv(
            job_path, dtype=str
        )

        if not existing.equals(jobs):
            raise RuntimeError(
                "Existing job list differs. "
                "Refusing to change batch identity."
            )
    else:
        jobs.to_csv(job_path, index=False)

    status_file = OUT / "status.jsonl"

    cached = 0
    completed = 0
    failed = 0
    blocked = 0
    stopped_low_disk = False

    print("=" * 78, flush=True)
    print("FULL EVENT-DRIVEN LIGHTRAG", flush=True)
    print("PLANNED SNAPSHOTS: 483", flush=True)
    print("KG-K:", args.kg_k, flush=True)
    print("SEC-K: 0", flush=True)
    print(
        "START:",
        datetime.now().isoformat(),
        flush=True,
    )
    print(
        "FREE DISK:",
        f"{free_gib():.2f} GiB",
        flush=True,
    )
    print("=" * 78, flush=True)

    for idx, job in enumerate(
        jobs.itertuples(index=False), 1
    ):
        ticker = job.ticker
        date = job.date
        tag = f"{ticker}_{date}"

        output = result_path(ticker, date)

        print(
            f"\n[{idx}/483] {tag}",
            flush=True,
        )

        if result_valid(output, ticker, date):
            cached += 1

            print("[CACHED]", tag, flush=True)

            append_jsonl(status_file, {
                "ticker": ticker,
                "date": date,
                "status": "CACHED",
            })

            continue

        if output.exists():
            blocked += 1

            print(
                "[BLOCKED] Existing output is "
                "not validated; refusing overwrite:",
                output,
                flush=True,
            )

            append_jsonl(status_file, {
                "ticker": ticker,
                "date": date,
                "status": "EXISTING_UNVERIFIED",
            })

            continue

        available = free_gib()

        if available < args.min_free_gib:
            print(
                f"[STOP_LOW_DISK] {available:.2f} GiB",
                flush=True,
            )
            stopped_low_disk = True
            break

        command = [
            str(PYTHON),
            "-u",
            str(RUNNER),
            "--ticker", ticker,
            "--date", date,
            "--kg-k", str(args.kg_k),
            "--sec-k", "0",
        ]

        logfile = logdir / f"{tag}.log"

        print("[BUILD]", tag, flush=True)

        start = time.monotonic()

        try:
            with logfile.open(
                "a", encoding="utf-8"
            ) as log:
                log.write(
                    "\n===== ATTEMPT START =====\n"
                )
                log.flush()

                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )

            rc = result.returncode

        except Exception as exc:
            rc = -1

            print(
                "[EXCEPTION]",
                repr(exc),
                flush=True,
            )

        seconds = round(
            time.monotonic() - start, 1
        )

        success = (
            rc == 0
            and result_valid(
                output, ticker, date
            )
        )

        status = (
            "COMPLETE" if success else "FAILED"
        )

        if success:
            completed += 1
        else:
            failed += 1

        print(
            f"[RESULT] {tag} {status} "
            f"seconds={seconds} rc={rc}",
            flush=True,
        )

        append_jsonl(status_file, {
            "ticker": ticker,
            "date": date,
            "status": status,
            "seconds": seconds,
            "returncode": rc,
            "log": str(logfile),
        })

        if idx % 20 == 0:
            print(
                "[PROGRESS]",
                f"checked={idx}/483",
                f"new={completed}",
                f"cached={cached}",
                f"failed={failed}",
                f"blocked={blocked}",
                f"free_gib={free_gib():.2f}",
                flush=True,
            )

    print("\n" + "=" * 78, flush=True)
    print("FULL BATCH FINISHED", flush=True)
    print("New complete:", completed, flush=True)
    print("Cached:", cached, flush=True)
    print("Failed:", failed, flush=True)
    print("Blocked:", blocked, flush=True)
    print(
        "Low disk stop:",
        stopped_low_disk,
        flush=True,
    )
    print(
        "Free GiB:",
        round(free_gib(), 2),
        flush=True,
    )

    # Audit all 483 candidates rather than only this run.
    valid_count = sum(
        result_valid(
            result_path(row.ticker, row.date),
            row.ticker,
            row.date,
        )
        for row in jobs.itertuples(index=False)
    )

    print(
        "TOTAL VALID EVENT-DRIVEN SNAPSHOTS:",
        f"{valid_count}/483",
        flush=True,
    )

    print("=" * 78, flush=True)

    if valid_count != 483:
        sys.exit(2)


if __name__ == "__main__":
    main()
