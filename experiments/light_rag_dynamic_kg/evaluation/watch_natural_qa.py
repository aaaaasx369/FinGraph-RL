#!/usr/bin/env python3
"""
Continuous Snapshot -> Natural Language QA Watcher.

- Discover completed historical LightRAG snapshots.
- Wait for index/manifest to settle.
- Execute one Natural QA process at a time.
- Reuse run_natural_qa.py.
- Preserve results and resume completed work.
- Retry failures after a cooldown.
- Never rebuild or insert the KG.

Note:
index_ready.json is a completion marker, not a formal
integrity certificate. Evaluation failures are logged.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

SNAPSHOTS = (
    ROOT / "results/light_rag_dynamic_kg/snapshots"
)

OUTPUT = (
    ROOT / "results/light_rag_dynamic_kg/natural_qa_v1"
)

LOG_DIR = ROOT / "logs/kg_qa_evaluation/continuous"

STATE_FILE = LOG_DIR / "watcher_state.json"

RUNNER = (
    ROOT /
    "experiments/light_rag_dynamic_kg/evaluation/run_natural_qa.py"
)

PYTHON = ROOT / ".venv-drl/bin/python"


def now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def read_json(path, default=None):
    if not path.exists():
        return default

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return default


def save_json_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    os.replace(temp, path)


def get_snapshots(settle_seconds, tickers):
    """
    Find indexed snapshots whose essential metadata
    have stopped changing.

    This is a conservative readiness filter, not a
    guarantee of vector-index integrity.
    """

    if not SNAPSHOTS.exists():
        return []

    eligible = []
    current_time = time.time()

    for folder in sorted(SNAPSHOTS.iterdir()):

        if not folder.is_dir():
            continue

        parts = folder.name.split("_", 2)

        if len(parts) != 3:
            continue

        ticker, date, fingerprint = parts

        if ticker not in tickers:
            continue

        marker = folder / "index_ready.json"
        manifest = folder / "source_manifest.json"

        if not marker.is_file():
            continue

        if not manifest.is_file():
            continue

        info = read_json(marker)
        sources = read_json(manifest)

        if not isinstance(info, dict):
            continue

        if not isinstance(sources, list):
            continue

        # The original runner writes this after indexing.
        if info.get("fingerprint") != fingerprint:
            continue

        newest_metadata = max(
            marker.stat().st_mtime,
            manifest.stat().st_mtime,
        )

        if (
            current_time - newest_metadata
            < settle_seconds
        ):
            continue

        eligible.append({
            "path": folder,
            "ticker": ticker,
            "date": date,
            "fingerprint": fingerprint,
        })

    return eligible


def finished_in_output(snapshot, args):
    """
    Detect a completed evaluation even if the watcher
    state was lost.

    Config includes the exact snapshot path and round
    settings, so results for another snapshot cannot
    be mistaken for this one.
    """

    if not OUTPUT.exists():
        return False

    for folder in OUTPUT.iterdir():

        if not folder.is_dir():
            continue

        config = read_json(
            folder / "config.json"
        )

        if not isinstance(config, dict):
            continue

        if config.get("snapshot") != str(
            snapshot["path"]
        ):
            continue

        if config.get("rounds") != args.rounds:
            continue

        if (
            config.get("questions_per_round")
            != args.n
        ):
            continue

        if (
            config.get("dev_per_round")
            != args.dev
        ):
            continue

        if config.get("seed") != args.seed:
            continue

        metrics = folder / "metrics.json"

        if not metrics.is_file():
            continue

        data = read_json(metrics)

        if not isinstance(data, dict):
            continue

        # Every expected mode must be present.
        if not all(
            mode in data
            for mode in ("local", "global", "hybrid")
        ):
            continue

        expected_test = (
            args.rounds * (args.n - args.dev)
        )

        if all(
            data[mode].get("test_attempted")
            == expected_test
            and
            data[mode].get("test_successful")
            == expected_test
            for mode in ("local", "global", "hybrid")
        ):
            return True

    return False


def execute(snapshot, args):
    """
    Run exactly one snapshot QA job synchronously.
    """

    name = snapshot["path"].name

    log_path = LOG_DIR / f"{name}.log"

    cmd = [
        str(PYTHON),
        "-u",
        str(RUNNER),

        "--ticker",
        snapshot["ticker"],

        "--date",
        snapshot["date"],

        "--fingerprint",
        snapshot["fingerprint"],

        "--rounds",
        str(args.rounds),

        "--n",
        str(args.n),

        "--dev",
        str(args.dev),

        "--seed",
        str(args.seed),
    ]

    print(
        f"[{now()}] START {name}",
        flush=True,
    )

    print(
        "[COMMAND]",
        " ".join(cmd),
        flush=True,
    )

    with log_path.open(
        "a", encoding="utf-8"
    ) as log:

        log.write(
            f"\n\n========== START {now()} ==========\n"
        )

        log.flush()

        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            check=False,
        )

        log.write(
            f"\n========== EXIT "
            f"{result.returncode} "
            f"{now()} ==========\n"
        )

    print(
        f"[{now()}] END {name} "
        f"exit={result.returncode}",
        flush=True,
    )

    return result.returncode


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rounds", type=int, default=1
    )

    parser.add_argument(
        "--n", type=int, default=25
    )

    parser.add_argument(
        "--dev", type=int, default=2
    )

    parser.add_argument(
        "--seed", type=int, default=20261021
    )

    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between directory scans",
    )

    parser.add_argument(
        "--settle-minutes", type=int, default=15,
        help="Minimum age of completion metadata",
    )

    parser.add_argument(
        "--retry-minutes", type=int, default=60
    )

    parser.add_argument(
        "--tickers",
        default="AAPL,AMZN,GOOGL,MSFT,NVDA"
    )

    parser.add_argument(
        "--once", action="store_true"
    )

    args = parser.parse_args()

    if (
        args.rounds < 1
        or args.n < 1
        or not 0 <= args.dev < args.n
        or args.interval < 1
        or args.settle_minutes < 0
        or args.retry_minutes < 1
    ):
        parser.error("Invalid settings")

    if not RUNNER.is_file():
        raise FileNotFoundError(
            f"Natural QA runner missing: {RUNNER}"
        )

    if not PYTHON.is_file():
        raise FileNotFoundError(
            f"Python environment missing: {PYTHON}"
        )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tickers = {
        x.strip().upper()
        for x in args.tickers.split(",")
        if x.strip()
    }

    import fcntl

    lock_file = LOG_DIR / "watcher.lock"

    with lock_file.open("w") as lock:

        try:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )

        except BlockingIOError:
            print(
                "Another Natural QA watcher is running.",
                flush=True,
            )
            return

        print(
            "[WATCHER STARTED]",
            now(),
            flush=True,
        )

        print(
            "[SCAN INTERVAL]",
            args.interval,
            "seconds",
            flush=True,
        )

        print(
            "[ROUNDS PER SNAPSHOT]",
            args.rounds,
            flush=True,
        )

        while True:

            state = read_json(
                STATE_FILE, {}
            ) or {}

            snapshots = get_snapshots(
                args.settle_minutes * 60,
                tickers,
            )

            print(
                f"[{now()}] Ready snapshots: "
                f"{len(snapshots)}",
                flush=True,
            )

            pending = []

            for snapshot in snapshots:

                name = snapshot["path"].name

                config_key = (
                    f"{name}:"
                    f"r{args.rounds}:"
                    f"n{args.n}:"
                    f"d{args.dev}:"
                    f"s{args.seed}"
                )

                if finished_in_output(
                    snapshot, args
                ):
                    if (
                        state.get(config_key, {})
                        .get("status") != "done"
                    ):
                        state[config_key] = {
                            "status": "done",
                            "updated_at": now(),
                        }

                        save_json_atomic(
                            STATE_FILE, state
                        )

                    continue

                previous = state.get(
                    config_key, {}
                )

                retry_after = previous.get(
                    "retry_after", 0
                )

                if time.time() < retry_after:
                    continue

                pending.append(
                    (snapshot, config_key)
                )

            print(
                f"[{now()}] Pending: {len(pending)}",
                flush=True,
            )

            for snapshot, key in pending:

                state[key] = {
                    "status": "running",
                    "updated_at": now(),
                }

                save_json_atomic(
                    STATE_FILE, state
                )

                exit_code = execute(
                    snapshot, args
                )

                if (
                    exit_code == 0
                    and finished_in_output(
                        snapshot, args
                    )
                ):

                    state[key] = {
                        "status": "done",
                        "updated_at": now(),
                    }

                else:

                    state[key] = {
                        "status": "failed",
                        "updated_at": now(),
                        "exit_code": exit_code,
                        "retry_after": (
                            time.time()
                            + args.retry_minutes * 60
                        ),
                    }

                save_json_atomic(
                    STATE_FILE, state
                )

            if args.once:
                print(
                    "[ONE SCAN COMPLETE]",
                    flush=True,
                )
                break

            time.sleep(args.interval)


if __name__ == "__main__":
    main()
