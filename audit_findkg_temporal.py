import csv
import json
import re
import hashlib
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import date, timedelta

DEFAULT_EDGES = (
    "kg/data/findkg/findkg_full_edges_with_dates.csv"
)

SOURCE_DIRS = [
    Path("external/FinDKG-FinRL-Train2020"),
    Path("external/FinDKG"),
    Path("external/FinDKG-FinRL"),
    Path("kg/scripts"),
]

TIME_WORDS = re.compile(
    r"time_id|timestamp|time2id|publication|published|"
    r"release.date|event.date|train.test|split|"
    r"look.ahead|leakage|weekly|date.mapping",
    re.I
)

def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def scan_source_files():
    findings = []
    for root in SOURCE_DIRS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if ".git" in path.parts:
                continue
            if path.suffix.lower() not in {
                ".py", ".md", ".txt", ".json", ".yaml", ".yml"
            }:
                continue
            if path.stat().st_size > 2_000_000:
                continue

            try:
                lines = path.read_text(
                    encoding="utf-8",
                    errors="replace"
                ).splitlines()
            except OSError:
                continue

            matches = []
            for i, line in enumerate(lines, 1):
                if TIME_WORDS.search(line):
                    matches.append({
                        "line": i,
                        "text": line.strip()[:240]
                    })

            if matches:
                findings.append({
                    "file": str(path),
                    "matches": matches[:25],
                    "total_matches": len(matches)
                })

    return findings

def audit_edges(path):
    id_to_dates = defaultdict(set)
    date_to_ids = defaultdict(set)
    date_counts = Counter()
    split_counts = Counter()
    split_ranges = defaultdict(list)
    split_edges = defaultdict(set)
    invalid_dates = []
    duplicate_count = 0
    seen = set()
    rows = 0
    cutoff_mismatches = []

    with path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        reader = csv.DictReader(f)

        required = {
            "subject_id", "relation_id",
            "object_id", "time_id", "date"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        for row in reader:
            rows += 1
            try:
                d = date.fromisoformat(row["date"])
                tid = str(row["time_id"])
            except (ValueError, TypeError):
                invalid_dates.append(rows)
                continue

            id_to_dates[tid].add(d.isoformat())
            date_to_ids[d.isoformat()].add(tid)
            date_counts[d.isoformat()] += 1

            split = row.get("finrl_split") or "unknown"
            split_counts[split] += 1
            split_ranges[split].append(d)

            edge = (
                row["subject_id"],
                row["relation_id"],
                row["object_id"],
                d.isoformat()
            )

            if edge in seen:
                duplicate_count += 1
            seen.add(edge)

            split_edges[split].add(edge)

            if split == "train" and d >= date(2021, 1, 1):
                cutoff_mismatches.append(
                    {"row": rows, "date": d.isoformat()}
                )

    time_conflicts = {
        k: sorted(v)
        for k, v in id_to_dates.items()
        if len(v) > 1
    }

    reverse_conflicts = {
        k: sorted(v)
        for k, v in date_to_ids.items()
        if len(v) > 1
    }

    sorted_dates = sorted(
        date.fromisoformat(d)
        for d in date_counts
    )

    gaps = []
    for a, b in zip(sorted_dates, sorted_dates[1:]):
        delta = (b - a).days
        if delta != 7:
            gaps.append({
                "from": a.isoformat(),
                "to": b.isoformat(),
                "gap_days": delta
            })

    overlaps = {}
    names = sorted(split_edges)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlaps[f"{a}__{b}"] = len(
                split_edges[a] & split_edges[b]
            )

    return {
        "total_rows": rows,
        "unique_timestamped_edges": len(seen),
        "duplicate_edge_rows": duplicate_count,
        "snapshot_count": len(sorted_dates),
        "first_date": (
            sorted_dates[0].isoformat()
            if sorted_dates else None
        ),
        "last_date": (
            sorted_dates[-1].isoformat()
            if sorted_dates else None
        ),
        "date_weekdays": dict(Counter(
            d.strftime("%A") for d in sorted_dates
        )),
        "time_ids": len(id_to_dates),
        "time_id_to_multiple_dates": time_conflicts,
        "date_to_multiple_time_ids": reverse_conflicts,
        "non_weekly_gaps": gaps[:50],
        "non_weekly_gap_count": len(gaps),
        "split_counts": dict(split_counts),
        "split_date_ranges": {
            name: {
                "first": min(ds).isoformat(),
                "last": max(ds).isoformat()
            }
            for name, ds in split_ranges.items()
        },
        "same_timestamped_edges_across_splits": overlaps,
        "train_rows_2021_or_later": len(cutoff_mismatches),
        "train_rows_2021_or_later_examples":
            cutoff_mismatches[:10],
        "invalid_date_rows": invalid_dates[:20]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=DEFAULT_EDGES)
    ap.add_argument(
        "--out",
        default="results/late_fusion_pilot/"
                "findkg_temporal_audit.json"
    )
    args = ap.parse_args()

    source = Path(args.edges)
    if not source.exists():
        raise SystemExit(f"Missing: {source}")

    print("Auditing FinDKG temporal metadata...")
    statistics = audit_edges(source)

    print("Scanning source code and documentation...")
    findings = scan_source_files()

    report = {
        "source": str(source),
        "source_sha256": file_hash(source),
        "statistics": statistics,
        "source_file_findings": findings,
        "temporal_checks": {
            "time_id_date_consistency": (
                not statistics["time_id_to_multiple_dates"]
                and not statistics["date_to_multiple_time_ids"]
            ),
            "weekly_interval_consistency": (
                statistics["non_weekly_gap_count"] == 0
            ),
            "public_release_time_verified": False,
            "edge_level_available_at_verified": False,
            "source_generation_leakage_audited": False,
            "availability_audited": False,
            "usable_for_leakage_free_backtest": False
        },
        "interpretation": [
            "Consistent time_id/date mapping does not prove "
            "historical public availability.",
            "The full dataset and original split can have "
            "different temporal semantics.",
            "A time-based split does not prove KG embeddings "
            "were trained without future data.",
            "Historical trading evaluation requires verified "
            "available_at <= decision timestamp.",
            "Manual review of code and source provenance "
            "is required before declaring leakage-free data."
        ]
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n========== SUMMARY ==========")
    for k, v in statistics.items():
        if k in {
            "time_id_to_multiple_dates",
            "date_to_multiple_time_ids",
            "non_weekly_gaps",
            "train_rows_2021_or_later_examples",
            "invalid_date_rows"
        }:
            print(k, ":", len(v))
        else:
            print(k, ":", v)

    print("\n========== RELEVANT SOURCE FILES ==========")
    for item in findings[:30]:
        print(
            item["file"],
            "matches =", item["total_matches"]
        )

    print("\n========== AUDIT RESULT ==========")
    print("Public availability verified: NO")
    print("Leakage-free backtest permitted: NO")
    print("Report:", out)

if __name__ == "__main__":
    main()
