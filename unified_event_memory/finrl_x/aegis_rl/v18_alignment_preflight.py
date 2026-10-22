#!/usr/bin/env python3

import ast
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path("results/aegis_rl_v1")
SRC = Path("unified_event_memory/finrl_x/aegis_rl")
OUT = ROOT / "v18_threeway"
OUT.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "original": ROOT / "v17_joint_rag_ppo_smoke/joint_rag_cache.jsonl",
    "b7": ROOT / "v17b7_alignscore_20/audit.jsonl",
    "b8": ROOT / "v17b8_event_outcome_20/audit.jsonl",
}

def load(path):
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def key(row):
    return (str(row["date"]), str(row["ticker"]))

def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

data = {name: load(path) for name, path in SOURCES.items()}
indices = {}

print("=" * 72)
print("V18 MATCHED RAG ALIGNMENT PREFLIGHT")
print("=" * 72)

for name, rows in data.items():
    index = {}

    for row in rows:
        k = key(row)
        if k in index:
            raise RuntimeError(f"Duplicate {name} key: {k}")
        index[k] = row

    indices[name] = index

    print(
        f"{name:12s}",
        "rows=", len(rows),
        "unique=", len(index)
    )

original_keys = set(indices["original"])
b7_keys = set(indices["b7"])
b8_keys = set(indices["b8"])

print("\n========== INTERSECTIONS ==========")
print("Original / B7:", len(original_keys & b7_keys))
print("Original / B8:", len(original_keys & b8_keys))
print("B7 / B8:", len(b7_keys & b8_keys))
print(
    "All three:",
    len(original_keys & b7_keys & b8_keys)
)

print("\n========== ORIGINAL REFRESH DATES ==========")

refresh_dates = sorted({d for d, _ in original_keys})

for date in refresh_dates:
    print(date)

print("\n========== REQUIRED MATCHED CASES ==========")

target = []

for date, ticker in sorted(original_keys):
    o = indices["original"][(date, ticker)]
    b7 = indices["b7"].get((date, ticker))
    b8 = indices["b8"].get((date, ticker))

    result = o.get("result")
    result = result if isinstance(result, dict) else {}

    original_usable = int(
        o.get("status") == "GENERATED"
        and bool(result.get("sec_evidence_ids"))
    )

    b7_verified = int(
        b7 is not None
        and bool(b7.get("has_verified_sec_signal"))
    )

    b8_classified = int(
        b8 is not None
        and bool(b8.get("b8_classified_claims", 0))
    )

    row = {
        "date": date,
        "ticker": ticker,
        "original_status": o.get("status", "MISSING"),
        "original_usable": original_usable,
        "sec_evidence_count": len(o.get("sec_evidence") or []),
        "kg_evidence_count": len(o.get("kg_evidence") or []),
        "b7_present": int(b7 is not None),
        "b7_verified": b7_verified,
        "b8_present": int(b8 is not None),
        "b8_classified": b8_classified,
        "requires_b7_generation": int(b7 is None),
        "requires_b8_generation": int(b8 is None),
    }
    target.append(row)

    print(
        date,
        ticker,
        "original=", row["original_status"],
        "SEC=", row["sec_evidence_count"],
        "KG=", row["kg_evidence_count"],
        "B7=", row["b7_present"],
        "B8=", row["b8_present"]
    )

columns = list(target[0]) if target else []

write_csv(
    OUT / "matched_refresh_targets.csv",
    target,
    columns
)

print("\n========== B7/B8 SOURCE INTERFACES ==========")

for filename in [
    "run_v17b7_paper_grounded.py",
    "run_v17b8_event_outcome.py",
]:
    path = SRC / filename
    print("\nFILE:", path)

    if not path.exists():
        print("NOT FOUND")
        continue

    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print("Syntax error:", e)
        continue

    print("Functions:")

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            print(" ", node.name)

    print("\nCLI argument definitions:")

    for i, line in enumerate(source.splitlines(), 1):
        if (
            "add_argument(" in line
            or "parse_args(" in line
            or "parse_known_args(" in line
            or "argparse.ArgumentParser" in line
        ):
            print(f"{i:4d}: {line.strip()[:200]}")

print("\n========== CACHE / DATA INTEGRITY ==========")

hashes = {}

for name, path in SOURCES.items():
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes[name] = digest
    print(name, digest)

summary = {
    "original_cases": len(original_keys),
    "b7_cases": len(b7_keys),
    "b8_cases": len(b8_keys),
    "matched_all": len(original_keys & b7_keys & b8_keys),
    "target_refresh_dates": refresh_dates,
    "original_usable_cases": sum(
        x["original_usable"] for x in target
    ),
    "missing_b7_cases": sum(
        x["requires_b7_generation"] for x in target
    ),
    "missing_b8_cases": sum(
        x["requires_b8_generation"] for x in target
    ),
    "source_sha256": hashes,
    "ppo_training_started": False,
}

(OUT / "alignment_preflight.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("\n========== SUMMARY ==========")

for k, v in summary.items():
    if k != "source_sha256":
        print(k, "=", v)

print("\nOUTPUT:")
print(OUT / "matched_refresh_targets.csv")
print(OUT / "alignment_preflight.json")

if summary["matched_all"] == 0:
    print("\nSTATUS: BLOCKED - MATCHED GROUNDING REQUIRED")
else:
    print("\nSTATUS: PARTIALLY MATCHED - CHECK USABLE SIGNALS")

print("\nPREFLIGHT COMPLETE")
