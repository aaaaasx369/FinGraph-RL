#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/light_rag_dynamic_kg"
SOURCE = BASE / "ablation_v1"
VERIFIED = BASE / "verified_v1"
OUT = BASE / "diagnostics"
OUT.mkdir(parents=True, exist_ok=True)

GROUPS = {
    "A_KG_RAG": "KG",
    "B_SEC_RAG": "SEC",
    "C_KG_SEC_RAG": "JOINT",
}

rows = []

for path in sorted(SOURCE.glob("*.json")):
    try:
        r = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue

    group = r.get("group")
    if group not in GROUPS:
        continue

    generated = r.get("generated")
    validation = r.get("validation") or {}

    if not isinstance(generated, dict):
        generated = {}

    vpath = VERIFIED / path.name
    judged = None

    if vpath.exists():
        try:
            judged = json.loads(vpath.read_text(encoding="utf-8"))
        except Exception:
            pass

    masks = (
        judged.get("accepted_masks", {})
        if isinstance(judged, dict) else {}
    )

    row = {
        "ticker": r.get("ticker"),
        "date": r.get("date"),
        "group": group,
        "kg_count": int(r.get("kg_count") or 0),
        "sec_count": int(r.get("sec_count") or 0),
        "retrieval_success": bool(
            (r.get("kg_count") or 0)
            + (r.get("sec_count") or 0)
        ),
        "generation_success": bool(generated),
        "accepted_claims": len(
            validation.get("accepted") or []
        ),
        "quote_check": bool(
            validation.get("quote_checks_passed")
        ),
        "grounding_status": validation.get("status"),
        "raw_impact": validation.get("impact"),
        "raw_risk": validation.get("risk"),
        "raw_confidence": validation.get("confidence"),
        "numeric_present": any(
            pd.notna(validation.get(k))
            for k in ("impact", "risk", "confidence")
        ),
        "judge_status": (
            judged.get("status")
            if isinstance(judged, dict) else "NOT_JUDGED"
        ),
        "accepted_numeric": any(
            masks.get(k) == 1
            for k in ("impact", "risk", "confidence")
        ),
        "json_path": str(path),
    }

    rows.append(row)

if not rows:
    raise SystemExit(
        "No A/B/C output JSON found in " + str(SOURCE)
    )

df = pd.DataFrame(rows)

df["year"] = pd.to_datetime(
    df["date"], errors="coerce"
).dt.year

df = df.sort_values(
    ["date", "ticker", "group"]
)

df.to_csv(
    OUT / "all_rag_outputs.csv",
    index=False,
)

summary = df.groupby(
    ["year", "group"], dropna=False
).agg(
    outputs=("date", "size"),
    unique_dates=("date", "nunique"),
    tickers=("ticker", "nunique"),
    retrieval_rate=("retrieval_success", "mean"),
    generation_rate=("generation_success", "mean"),
    numeric_rate=("numeric_present", "mean"),
    accepted_rate=("accepted_numeric", "mean"),
    avg_kg=("kg_count", "mean"),
    avg_sec=("sec_count", "mean"),
).reset_index()

summary.to_csv(
    OUT / "rag_coverage_summary.csv",
    index=False,
)

print("\n========== RAG COVERAGE DIAGNOSTICS ==========")
print(summary.to_string(index=False))

print("\n========== REJECTION DETAILS ==========")
print(
    df.groupby(
        ["group", "judge_status"],
        dropna=False
    ).size().to_string()
)

print("\n========== FILE COUNTS ==========")
print("A/B/C outputs:", len(df))
print("Unique dates:", df.date.nunique())
print("Tickers:", df.ticker.nunique())

print("\n[SAVED]", OUT / "all_rag_outputs.csv")
print("[SAVED]", OUT / "rag_coverage_summary.csv")
