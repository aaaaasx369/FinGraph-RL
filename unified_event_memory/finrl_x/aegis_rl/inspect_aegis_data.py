#!/usr/bin/env python3

import csv
import json
from pathlib import Path
from collections import Counter

ROOT = Path.cwd()
OUT = ROOT / "results/aegis_rl_v1/schema_audit.json"

TARGETS = [
    "kg/data/edges_v4_full.csv",
    "kg/data/edges_v3_full.csv",
    "kg/data/edges_v2_full.csv",
    "kg/data/finreflectkg_5stocks_2018_2023.parquet",
    "kg/data/company_embedding_events.csv",
    "kg/data/company_embedding_events_with_vectors.csv",
    "kg/output/finreflectkg_dynamic_filing_map.csv",
    "kg/output/finreflectkg_dynamic_transe_32d.csv",
    "kg/output/sec_rag_plus_finreflectkg.csv",
    "kg/output/finreflect_reimplementation/"
    "temporal_kg_features_full.csv",
]

DATE_WORDS = [
    "date", "time", "filing", "accepted",
    "published", "observed", "available",
    "start", "end", "timestamp"
]

def safe_value(value, limit=85):
    s = str(value).replace("\n", " ").replace("\r", " ")
    return s[:limit] + ("..." if len(s) > limit else "")

def inspect_csv(path):
    with path.open(
        encoding="utf-8-sig", newline="", errors="replace"
    ) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = []
        for _, row in zip(range(3), reader):
            rows.append({
                str(k): safe_value(v)
                for k, v in row.items()
            })

    date_cols = [
        c for c in columns
        if any(w in c.lower() for w in DATE_WORDS)
    ]

    return {
        "columns": columns,
        "date_columns": date_cols,
        "sample": rows
    }

def inspect_jsonl(path):
    rows = []
    keys = Counter()

    with path.open(
        encoding="utf-8", errors="replace"
    ) as f:
        for _, line in zip(range(20), f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(obj, dict):
                continue

            keys.update(obj.keys())

            if len(rows) < 2:
                rows.append({
                    str(k): safe_value(v)
                    for k, v in obj.items()
                })

    columns = list(keys)
    return {
        "columns": columns,
        "date_columns": [
            c for c in columns
            if any(w in c.lower() for w in DATE_WORDS)
        ],
        "sample": rows
    }

def inspect_parquet(path):
    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        columns = pf.schema_arrow.names

        return {
            "columns": columns,
            "date_columns": [
                c for c in columns
                if any(w in c.lower() for w in DATE_WORDS)
            ],
            "num_rows": pf.metadata.num_rows,
            "sample": []
        }
    except ImportError:
        return {
            "error": "pyarrow missing; install pyarrow",
        }
    except Exception as exc:
        return {"error": str(exc)}

def inspect(path):
    if path.suffix.lower() == ".csv":
        return inspect_csv(path)
    if path.suffix.lower() == ".jsonl":
        return inspect_jsonl(path)
    if path.suffix.lower() == ".parquet":
        return inspect_parquet(path)
    return {"error": "Unsupported format"}

def main():
    results = {}

    for rel in TARGETS:
        path = ROOT / rel
        if not path.exists():
            results[rel] = {"exists": False}
            continue

        print("\n" + "=" * 76)
        print(rel)
        print("=" * 76)

        result = inspect(path)
        result["exists"] = True
        result["size_mb"] = round(
            path.stat().st_size / (1024 ** 2), 2
        )

        results[rel] = result

        print("Columns:", result.get("columns"))
        print("Time columns:", result.get("date_columns"))
        print("Rows:", result.get("num_rows", "not counted"))

        for row in result.get("sample", []):
            print("Sample:", json.dumps(
                row, ensure_ascii=False
            ))

        if "error" in result:
            print("Error:", result["error"])

    print("\n" + "=" * 76)
    print("SEARCH SEC / EMBEDDING DATA")
    print("=" * 76)

    found = []

    roots = [
        ROOT / "kg",
        ROOT / "unified_event_memory",
    ]

    for folder in roots:
        if not folder.exists():
            continue

        for path in folder.rglob("*"):
            if not path.is_file():
                continue

            name = path.name.lower()
            if (
                "sec_chunks" in name or
                "daily_kgt" in name or
                "dynamic_embedding" in name or
                "unified" in name and
                path.suffix.lower() in {
                    ".jsonl", ".csv", ".parquet", ".npz"
                }
            ):
                found.append(str(
                    path.relative_to(ROOT)
                ))

    for path in sorted(set(found))[:100]:
        print(path)

    results["discovered_files"] = sorted(set(found))

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        json.dump(
            results, f,
            ensure_ascii=False,
            indent=2
        )

    print("\nSaved:", OUT)

if __name__ == "__main__":
    main()
