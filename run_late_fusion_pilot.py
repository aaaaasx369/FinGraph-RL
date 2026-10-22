#!/usr/bin/env python3
"""Run a resumable 5-stock x 4-date LightRAG + FinDKG + Qwen pilot.

Run from the FinRL-Trading root. Standard library only; existing helper scripts
need their usual LightRAG/Ollama dependencies. No changes to FinRL training.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL")
DEFAULT_DATES = ("2021-05-11", "2021-05-18", "2021-05-25", "2021-06-01")
FEATURES = ("date", "ticker", "impact", "risk", "confidence",
            "graph_change", "event_intensity")
ERROR_MARKERS = (
    "failed to extract document", "worker execution timeout",
    "failed to finalize", "error embedding pending vector",
    "traceback (most recent call last)",
)


def check_prerequisites(root):
    required = (
        "sec_section_selector.py", "late_fusion_v4.py",
        "findkg_retrieval_v2.py", "generate_financial_signals.py",
        "rag/data/processed/sec/sec_chunks.jsonl",
        "kg/data/findkg/findkg_full_edges_with_dates.csv",
    )
    missing = [x for x in required if not (root / x).exists()]
    if missing:
        raise SystemExit("Missing required files:\n" + "\n".join(missing))


def run(cmd, log_path, cwd):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f,
                              stderr=subprocess.STDOUT, check=False)
    message = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode:
        raise RuntimeError(f"exit={proc.returncode}; inspect {log_path}")
    markers = [m for m in ERROR_MARKERS if m in message.lower()]
    if markers:
        raise RuntimeError(f"Logged failure {markers}; inspect {log_path}")
    return message


def validate_result(signal_path, fusion_path, ticker, day):
    sig = json.loads(signal_path.read_text(encoding="utf-8"))
    fusion = json.loads(fusion_path.read_text(encoding="utf-8"))
    if sig.get("date") != day or sig.get("ticker") != ticker:
        raise ValueError("Signal date/ticker mismatch")
    if fusion.get("as_of") != day or fusion.get("ticker") != ticker:
        raise ValueError("Fusion date/ticker mismatch")
    valid = {x["evidence_id"] for x in fusion.get("text_evidence", []) +
             fusion.get("graph_evidence", [])}
    if not set(sig.get("evidence_ids", [])) <= valid:
        raise ValueError("Unknown cited evidence")
    if not fusion.get("text_evidence"):
        raise ValueError("No LightRAG text evidence: reject feature row")
    for k in FEATURES[2:]:
        val = sig.get(k)
        if val is not None and (not isinstance(val, (int, float)) or isinstance(val, bool)):
            raise ValueError(f"Invalid feature {k}: {val}")
    return sig, fusion


def aggregate(root, jobs, outdir):
    records, statuses = [], []
    for ticker, day in jobs:
        stem = f"{ticker}_{day}"
        loc = outdir / stem
        record_path = loc / "signals.json"
        fusion_path = loc / "fusion.json"
        status_path = loc / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {
            "ticker": ticker, "date": day, "status": "not_started"}
        statuses.append(status)
        if status.get("status") != "complete":
            continue
        try:
            sig, fusion = validate_result(record_path, fusion_path, ticker, day)
            records.append({
                **{k: sig.get(k) for k in FEATURES},
                "sec_count": len(fusion.get("text_evidence", [])),
                "graph_1hop": fusion.get("counts", {}).get("graph_1hop", 0),
                "graph_2hop": fusion.get("counts", {}).get("graph_2hop", 0),
                "evidence_count": len(sig.get("evidence_ids", [])),
                "model": sig.get("model"),
                "research_only": True,
                "availability_audited": False,
            })
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
            statuses[-1] = {**status, "status": "invalid", "error": str(exc)}
    cols = (*FEATURES, "sec_count", "graph_1hop", "graph_2hop",
            "evidence_count", "model", "research_only", "availability_audited")
    with (outdir / "pilot_features.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(records)
    (outdir / "pilot_status.json").write_text(
        json.dumps(statuses, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed {len(records)}/{len(jobs)} validated rows. "
          f"CSV: {outdir / 'pilot_features.csv'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dates", nargs="+", default=list(DEFAULT_DATES),
                    help="Four or more point-in-time decision dates, YYYY-MM-DD")
    ap.add_argument("--tickers", nargs="+", default=list(TICKERS), choices=TICKERS)
    ap.add_argument("--output-dir", default="results/late_fusion_pilot")
    ap.add_argument("--index-root", default="rag/data/lightrag_pilot")
    ap.add_argument("--host", default="http://140.113.87.164:11434")
    ap.add_argument("--llm", default="qwen2.5:3b")
    ap.add_argument("--embed", default="nomic-embed-text:latest")
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--max-docs", type=int, default=1)
    ap.add_argument("--chunks-per-doc", type=int, default=2)
    ap.add_argument("--max-chars-per-doc", type=int, default=3000)
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--graph-k", type=int, default=5)
    ap.add_argument("--resume", action="store_true", help="Skip fully validated jobs")
    ap.add_argument("--dry-run", action="store_true", help="List the planned jobs; no models called")
    args = ap.parse_args()
    root = Path.cwd()
    check_prerequisites(root)
    for d in args.dates:
        date.fromisoformat(d)
    jobs = [(t, d) for d in args.dates for t in args.tickers]
    outdir = (root / args.output_dir).resolve()
    if args.dry_run:
        for t, d in jobs:
            print(d, t)
        print(f"{len(jobs)} jobs; max-docs={args.max_docs}; pilot only.")
        return
    outdir.mkdir(parents=True, exist_ok=True)
    os.environ["OLLAMA_HOST"] = args.host
    # V4 itself must still have both LightRAG model functions routed to this
    # host (as patched in the preceding workflow).
    for i, (ticker, day) in enumerate(jobs, 1):
        loc = outdir / f"{ticker}_{day}"
        loc.mkdir(exist_ok=True)
        status_path = loc / "status.json"
        fusion_path = loc / "fusion.json"
        signal_path = loc / "signals.json"
        if args.resume and signal_path.exists() and fusion_path.exists():
            try:
                validate_result(signal_path, fusion_path, ticker, day)
                print(f"[{i}/{len(jobs)}] SKIP {ticker} {day}", flush=True)
                status_path.write_text(json.dumps(
                    {"ticker": ticker, "date": day, "status": "complete"}, indent=2))
                continue
            except (ValueError, KeyError, OSError, json.JSONDecodeError):
                pass
        status = {"ticker": ticker, "date": day, "status": "running"}
        status_path.write_text(json.dumps(status, indent=2))
        sec = loc / "sections.jsonl"
        index_args = [
            "--ticker", ticker, "--as-of", day, "--sec", str(sec),
            "--llm", args.llm, "--embed", args.embed,
            "--embed-dim", str(args.embed_dim), "--max-docs", str(args.max_docs),
            "--max-chars-per-doc", str(args.max_chars_per_doc),
            "--num-ctx", str(args.num_ctx), "--index-root", args.index_root
        ]
        try:
            print(f"[{i}/{len(jobs)}] {ticker} {day}: section selection", flush=True)
            run([sys.executable, "sec_section_selector.py",
                 "--ticker", ticker, "--as-of", day,
                 "--max-docs", str(args.max_docs),
                 "--chunks-per-doc", str(args.chunks_per_doc),
                 "--output", str(sec)], loc / "01_selector.log", root)
            if not sec.exists() or not sec.read_text(encoding="utf-8").strip():
                raise ValueError("Section selector returned no chunks")
            print(f"[{i}/{len(jobs)}] {ticker} {day}: LightRAG indexing", flush=True)
            run([sys.executable, "late_fusion_v4.py", "index", *index_args],
                loc / "02_index.log", root)
            print(f"[{i}/{len(jobs)}] {ticker} {day}: fusion retrieval", flush=True)
            run([sys.executable, "late_fusion_v4.py", "query", *index_args,
                 "--top-k", str(args.top_k), "--graph-k", str(args.graph_k),
                 "--out", str(fusion_path)], loc / "03_query.log", root)
            print(f"[{i}/{len(jobs)}] {ticker} {day}: Qwen signals", flush=True)
            run([sys.executable, "generate_financial_signals.py",
                 "--input", str(fusion_path), "--host", args.host,
                 "--model", args.llm, "--out-json", str(signal_path),
                 "--out-csv", str(loc / "signals.csv")],
                loc / "04_signals.log", root)
            validate_result(signal_path, fusion_path, ticker, day)
            status["status"] = "complete"
        except Exception as exc:
            status.update(status="failed", error=str(exc))
            print(f"FAILED {ticker} {day}: {exc}", file=sys.stderr, flush=True)
        finally:
            status_path.write_text(json.dumps(status, indent=2))
            aggregate(root, jobs, outdir)
    print("Research-only pilot. FinDKG public availability not audited; "
          "do not merge into trading backtests yet.")


if __name__ == "__main__":
    main()
