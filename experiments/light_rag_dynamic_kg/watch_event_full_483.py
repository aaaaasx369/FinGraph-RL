#!/usr/bin/env python3

import argparse
import json
import os
import signal
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / (
    "results/light_rag_dynamic_kg/"
    "event_driven_full_v1"
)

LOGS = BASE / "logs"
AUDIT = BASE / "watchdog_actions.jsonl"

CLK = os.sysconf("SC_CLK_TCK")

TICKERS = {
    "AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"
}


def now():
    return datetime.now().astimezone().isoformat()


def report(message):
    print(
        f"[{now()}] {message}",
        flush=True,
    )


def proc_info(pid):
    """Read a process without requiring psutil."""
    try:
        p = Path(f"/proc/{pid}")

        cmd = (
            (p / "cmdline")
            .read_bytes()
            .split(b"\x00")
        )

        argv = [
            x.decode("utf-8", errors="replace")
            for x in cmd if x
        ]

        stat = (p / "stat").read_text()

        # Split after the final closing parenthesis;
        # Linux process names may contain spaces.
        rest = stat[stat.rfind(")") + 2:].split()

        ppid = int(rest[1])
        start_ticks = int(rest[19])

        uptime = float(
            Path("/proc/uptime")
            .read_text()
            .split()[0]
        )

        age = max(
            0.0,
            uptime - start_ticks / CLK,
        )

        return {
            "pid": pid,
            "ppid": ppid,
            "argv": argv,
            "age": age,
            "start_ticks": start_ticks,
        }

    except (
        FileNotFoundError,
        ProcessLookupError,
        PermissionError,
        ValueError,
        IndexError,
        OSError,
    ):
        return None


def find_children(parent_pid):
    children = []

    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue

        p = proc_info(int(entry.name))

        if p is None:
            continue

        if p["ppid"] != parent_pid:
            continue

        argv = p["argv"]

        joined = " ".join(argv)

        if "run_official_lightrag.py" not in joined:
            continue

        # Only processes with explicit ticker and date.
        try:
            ticker = argv[argv.index("--ticker") + 1]
            date = argv[argv.index("--date") + 1]
        except (ValueError, IndexError):
            continue

        if ticker not in TICKERS:
            continue

        try:
            time.strptime(date, "%Y-%m-%d")
        except ValueError:
            continue

        p["ticker"] = ticker
        p["date"] = date
        p["tag"] = f"{ticker}_{date}"

        children.append(p)

    return children


def still_same_process(original, parent_pid):
    """Protect against PID reuse or a new batch."""
    current = proc_info(original["pid"])

    if current is None:
        return False

    return (
        current["ppid"] == parent_pid
        and current["start_ticks"]
            == original["start_ticks"]
        and current["argv"] == original["argv"]
    )


def audit(action, child, **kwargs):
    record = {
        "timestamp": now(),
        "action": action,
        "pid": child["pid"],
        "ppid": child["ppid"],
        "tag": child["tag"],
        "process_age_seconds": round(
            child["age"], 1
        ),
        **kwargs,
    }

    with AUDIT.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                record, ensure_ascii=False
            ) + "\n"
        )


def terminate(child, parent_pid, reason, grace):
    report(
        f"[TIMEOUT] {child['tag']} "
        f"PID={child['pid']} "
        f"age={child['age']/60:.1f}m "
        f"reason={reason}"
    )

    if not still_same_process(child, parent_pid):
        report("[SKIP] Process changed or exited")
        return

    audit("SIGTERM", child, reason=reason)

    try:
        os.kill(child["pid"], signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.monotonic() + grace

    while time.monotonic() < deadline:
        if not still_same_process(child, parent_pid):
            report(
                f"[EXITED] {child['tag']} after SIGTERM"
            )
            return

        time.sleep(2)

    if not still_same_process(child, parent_pid):
        return

    report(
        f"[ESCALATE] {child['tag']} "
        "did not exit after SIGTERM"
    )

    audit("SIGKILL", child, reason=reason)

    try:
        os.kill(child["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--parent-pid",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--max-job-minutes",
        type=float,
        default=40,
    )

    ap.add_argument(
        "--stale-minutes",
        type=float,
        default=12,
    )

    ap.add_argument(
        "--min-age-minutes",
        type=float,
        default=20,
    )

    ap.add_argument(
        "--interval",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--grace-seconds",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    LOGS.mkdir(parents=True, exist_ok=True)

    parent = proc_info(args.parent_pid)

    if parent is None:
        raise RuntimeError("Parent process not found")

    if not any(
        "run_event_full_483.py" in x
        for x in parent["argv"]
    ):
        raise RuntimeError(
            "Parent is not run_event_full_483.py"
        )

    original_parent_start = parent["start_ticks"]

    report("EVENT FULL 483 WATCHDOG STARTED")
    report(f"Parent PID: {args.parent_pid}")
    report(
        f"Hard timeout: {args.max_job_minutes} min"
    )
    report(
        f"Stale timeout: {args.stale_minutes} min"
    )

    acted = set()
    last_heartbeat = 0

    while True:
        parent = proc_info(args.parent_pid)

        if (
            parent is None
            or parent["start_ticks"]
                != original_parent_start
        ):
            report(
                "[DONE] Original Full Batch has exited"
            )
            break

        children = find_children(args.parent_pid)

        for child in children:
            key = (
                child["pid"],
                child["start_ticks"],
            )

            log_path = (
                LOGS / f"{child['tag']}.log"
            )

            if log_path.exists():
                idle_seconds = max(
                    0.0,
                    time.time()
                    - log_path.stat().st_mtime,
                )
            else:
                idle_seconds = child["age"]

            age_minutes = child["age"] / 60
            idle_minutes = idle_seconds / 60

            hard_timeout = (
                age_minutes
                >= args.max_job_minutes
            )

            stale_timeout = (
                age_minutes
                >= args.min_age_minutes
                and idle_minutes
                >= args.stale_minutes
            )

            if key in acted:
                continue

            if hard_timeout or stale_timeout:
                acted.add(key)

                reason = (
                    "HARD_TIMEOUT"
                    if hard_timeout
                    else "STALE_LOG_TIMEOUT"
                )

                terminate(
                    child,
                    args.parent_pid,
                    reason,
                    args.grace_seconds,
                )

        if time.monotonic() - last_heartbeat > 300:
            tags = [
                f"{c['tag']}({c['age']/60:.1f}m)"
                for c in children
            ]

            report(
                "[HEARTBEAT] "
                + (", ".join(tags) if tags else "idle")
            )

            last_heartbeat = time.monotonic()

        time.sleep(args.interval)

    report("WATCHDOG FINISHED")


if __name__ == "__main__":
    main()
