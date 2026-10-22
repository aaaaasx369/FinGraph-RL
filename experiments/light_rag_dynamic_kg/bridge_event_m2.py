#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/light_rag_dynamic_kg"
BATCH = BASE / "event_driven_m2_v1"
OUT = BASE / "event_driven_dual_channel_v1"
CONVERTED = OUT / "converted_ablation"

KG_FILE = ROOT / (
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)

TICKERS = {"AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-sec-kg", action="store_true")
    args = ap.parse_args()

    jobs_file = BATCH / "selected_jobs.csv"
    if not jobs_file.exists():
        raise FileNotFoundError(jobs_file)

    jobs = pd.read_csv(jobs_file, dtype=str)
    if not {"ticker", "date"}.issubset(jobs.columns):
        raise RuntimeError("Invalid selected_jobs.csv")

    kg = pd.read_csv(KG_FILE, dtype=str, low_memory=False)
    kg = kg.set_index("event_id", drop=False)

    CONVERTED.mkdir(parents=True, exist_ok=True)

    report = []

    for job in jobs.itertuples(index=False):
        ticker = str(job.ticker)
        date = str(job.date)

        if ticker not in TICKERS:
            raise RuntimeError(f"Unknown ticker: {ticker}")

        tag = f"{ticker}_{date}"
        output = BASE / f"lightrag_{tag}.json"

        status = {
            "ticker": ticker,
            "date": date,
            "status": "SKIPPED",
            "raw_manifest": 0,
            "accepted_kg": 0,
            "reason": "",
        }

        if not output.exists():
            status["reason"] = "NO_OUTPUT"
            report.append(status)
            continue

        try:
            data = json.loads(output.read_text(encoding="utf-8"))
        except Exception:
            status["reason"] = "INVALID_JSON"
            report.append(status)
            continue

        if (
            data.get("ticker") != ticker
            or data.get("decision_date") != date
            or not str(data.get("answer", "")).strip()
        ):
            status["reason"] = "INVALID_RESULT"
            report.append(status)
            continue

        fp = str(data.get("snapshot_fingerprint", ""))
        if not fp:
            status["reason"] = "MISSING_FINGERPRINT"
            report.append(status)
            continue

        workspace = BASE / "snapshots" / f"{tag}_{fp}"
        manifest_path = workspace / "source_manifest.json"
        ready_path = workspace / "index_ready.json"

        if not manifest_path.exists() or not ready_path.exists():
            status["reason"] = "INCOMPLETE_WORKSPACE"
            report.append(status)
            continue

        try:
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            ready = json.loads(
                ready_path.read_text(encoding="utf-8")
            )
        except Exception:
            status["reason"] = "INVALID_WORKSPACE"
            report.append(status)
            continue

        if (
            not isinstance(manifest, list)
            or ready.get("fingerprint") != fp
        ):
            status["reason"] = "WORKSPACE_MISMATCH"
            report.append(status)
            continue

        status["raw_manifest"] = len(manifest)
        evidence = []
        invalid = []

        for item in manifest:
            if not isinstance(item, dict):
                invalid.append("NON_DICT")
                continue

            event_id = str(item.get("original_event_id", ""))
            source_id = str(item.get("source_id", ""))

            # This is a KG-only experiment.
            if not source_id.startswith("KG_"):
                invalid.append(f"NON_KG:{source_id}")
                continue

            if event_id not in kg.index:
                invalid.append(f"UNKNOWN_EVENT:{event_id}")
                continue

            row = kg.loc[event_id]
            if isinstance(row, pd.DataFrame):
                invalid.append(f"DUPLICATE_ID:{event_id}")
                continue

            source = str(row["source"])
            if not args.include_sec_kg and source != "FINDKG":
                continue

            timestamp = pd.to_datetime(
                row["available_at"],
                errors="coerce",
                utc=True,
            )

            manifest_timestamp = pd.to_datetime(
                item.get("available_at"),
                errors="coerce",
                utc=True,
            )

            cutoff = pd.Timestamp(date).tz_localize("UTC")

            if (
                pd.isna(timestamp)
                or pd.isna(manifest_timestamp)
                or timestamp != manifest_timestamp
                or timestamp >= cutoff
            ):
                invalid.append(f"DATE_MISMATCH:{event_id}")
                continue

            company = f"COMPANY:{ticker}"
            if (
                str(row["head"]) != company
                and str(row["tail"]) != company
            ):
                invalid.append(f"NOT_FOCAL:{event_id}")
                continue

            evidence.append({
                "id": event_id,
                "type": "KG",
                "relation": str(row["relation"]),
                "available_at": timestamp.isoformat(),
                "head": str(row["head"]),
                "tail": str(row["tail"]),
                "source": source,
            })

        # A malformed source record must not be silently certified.
        if invalid:
            status["reason"] = (
                "INVALID_MANIFEST:" + ",".join(invalid[:5])
            )
            report.append(status)
            continue

        if not evidence:
            status["reason"] = "NO_ELIGIBLE_EVIDENCE"
            report.append(status)
            continue

        # Numerics stay missing. An LLM citation is not
        # quote-level financial grounding.
        converted = {
            "ticker": ticker,
            "date": date,
            "group": "A_KG_RAG",
            "evidence": evidence,
            "generated": {
                "summary": str(data["answer"]).strip(),
            },
            "validation": {
                "quote_checks_passed": False,
                "impact": None,
                "risk": None,
                "confidence": None,
            },
            "provenance": {
                "original_result": str(output),
                "source_manifest": str(manifest_path),
                "snapshot_fingerprint": fp,
                "financial_grounding_certified": False,
                "historical_pit_certified": False,
            },
        }

        target = CONVERTED / f"{tag}_A_KG_RAG.json"

        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            if old != converted:
                raise RuntimeError(
                    f"Existing converted file differs: {target}"
                )
            status["status"] = "CACHED"
        else:
            target.write_text(
                json.dumps(
                    converted,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            status["status"] = "CONVERTED"

        status["accepted_kg"] = len(evidence)
        report.append(status)

    report_df = pd.DataFrame(report)
    report_path = OUT / "bridge_report.csv"
    report_df.to_csv(report_path, index=False)

    print("\n========== EVENT-DRIVEN BRIDGE ==========")
    print(report_df.to_string(index=False))
    print("\nSTATUS COUNTS")
    print(report_df["status"].value_counts().to_string())
    print("\n[SAVED]", report_path)
    print("[CONVERTED]", CONVERTED)
    print("Original source files unchanged.")


if __name__ == "__main__":
    main()
