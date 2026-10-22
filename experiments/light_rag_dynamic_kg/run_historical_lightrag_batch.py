#!/usr/bin/env python3
import argparse
import calendar
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/light_rag_dynamic_kg"
OUT = BASE / "full_pipeline"
OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

OFFICIAL = ROOT / (
    "experiments/light_rag_dynamic_kg/"
    "run_official_lightrag.py"
)
ABLATION = ROOT / (
    "experiments/light_rag_dynamic_kg/"
    "run_abc_ablation.py"
)


def get_flags(script):
    p = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if p.returncode != 0:
        raise RuntimeError(
            f"Cannot inspect {script.name} CLI:\n"
            + p.stderr[-3000:]
        )
    return p.stdout


def select_flag(help_text, names, required=True):
    import re

    for name in names:
        if re.search(
            r"(?<![\w-])" + re.escape(name) + r"(?![\w-])",
            help_text,
        ):
            return name

    if required:
        raise RuntimeError(
            f"None of these CLI flags exists: {names}\n"
            f"Existing flags:\n{help_text[:4000]}"
        )
    return None


def build_command(script, ticker, date):
    help_text = get_flags(script)

    ticker_flag = select_flag(
        help_text, ["--ticker", "--company"]
    )
    date_flag = select_flag(
        help_text,
        ["--date", "--decision-date", "--as-of-date"],
    )

    cmd = [
        sys.executable, "-u", str(script),
        ticker_flag, ticker,
        date_flag, date,
    ]

    kg_flag = select_flag(
        help_text, ["--kg-k", "--kg-top-k"],
        required=False,
    )
    sec_flag = select_flag(
        help_text, ["--sec-k", "--sec-top-k"],
        required=False,
    )

    if kg_flag:
        cmd += [kg_flag, "80"]
    if sec_flag:
        cmd += [sec_flag, "5"]

    return cmd


def month_ends(start, end):
    begin = pd.Timestamp(start)
    finish = pd.Timestamp(end)

    periods = pd.period_range(
        begin, finish, freq="M"
    )

    dates = []

    for period in periods:
        y = period.year
        m = period.month
        last = calendar.monthrange(y, m)[1]

        day = pd.Timestamp(y, m, last)

        if day > finish:
            day = finish

        # Weekday approximation only.
        # Downstream alignment MUST use actual exchange dates.
        while day.weekday() >= 5:
            day -= pd.Timedelta(days=1)

        if day >= begin:
            dates.append(day.strftime("%Y-%m-%d"))

    return sorted(set(dates))


def run_and_log(cmd, logfile):
    with logfile.open("w", encoding="utf-8") as f:
        p = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--tickers", nargs="+", default=TICKERS)
    ap.add_argument("--max-jobs", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    opt = ap.parse_args()

    for script in (OFFICIAL, ABLATION):
        if not script.exists():
            raise FileNotFoundError(script)

    dates = month_ends(opt.start, opt.end)
    jobs = [(t, d) for d in dates for t in opt.tickers]

    if opt.max_jobs > 0:
        jobs = jobs[:opt.max_jobs]

    manifest = OUT / "historical_jobs.jsonl"

    print("[DATES]", len(dates))
    print("[TICKERS]", opt.tickers)
    print("[JOBS]", len(jobs))
    print("[FIRST]", jobs[0] if jobs else None)
    print("[LAST]", jobs[-1] if jobs else None)

    # Check command interfaces before launching long jobs.
    build_command(OFFICIAL, opt.tickers[0], dates[0])
    build_command(ABLATION, opt.tickers[0], dates[0])

    if opt.dry_run:
        for ticker, date in jobs[:15]:
            print(
                "[INDEX]", build_command(
                    OFFICIAL, ticker, date
                )
            )
            print(
                "[ABLATE]", build_command(
                    ABLATION, ticker, date
                )
            )
        return

    for i, (ticker, date) in enumerate(jobs, 1):
        tag = f"{ticker}_{date}"

        expected = [
            BASE / "ablation_v1" /
            f"{tag}_{group}.json"
            for group in (
                "A_KG_RAG",
                "B_SEC_RAG",
                "C_KG_SEC_RAG",
            )
        ]

        if all(p.exists() for p in expected):
            print(f"[{i}/{len(jobs)}] CACHED {tag}")
            continue

        print(f"[{i}/{len(jobs)}] BUILD {tag}", flush=True)

        index_cmd = build_command(
            OFFICIAL, ticker, date
        )
        rc = run_and_log(
            index_cmd, OUT / f"{tag}_index.log"
        )

        if rc != 0:
            status = "INDEX_FAILED"
        else:
            ablation_cmd = build_command(
                ABLATION, ticker, date
            )
            rc2 = run_and_log(
                ablation_cmd,
                OUT / f"{tag}_ablation.log",
            )
            status = (
                "COMPLETE"
                if rc2 == 0 and all(
                    p.exists() for p in expected
                )
                else "ABLATION_FAILED"
            )

        with manifest.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "ticker": ticker,
                    "date": date,
                    "status": status,
                }) + "\n"
            )

        print("[RESULT]", tag, status, flush=True)

        if status != "COMPLETE" and not opt.retry_failed:
            raise RuntimeError(
                f"{tag}: {status}. "
                f"Check {OUT}/{tag}_*.log"
            )


if __name__ == "__main__":
    main()
