import csv
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import date, timedelta

EDGES = "kg/data/findkg/findkg_full_edges_with_dates.csv"
MAP = "kg/data/findkg/portfolio_entity_map.csv"
ENTITY = "external/FinDKG-FinRL-Train2020/data/FinDKG/entity2id.txt"
PILOT = "results/late_fusion_pilot/pilot_features.csv"

DEFAULT_DATES = [
    "2021-05-11",
    "2021-05-18",
    "2021-05-25",
    "2021-06-01",
]

def read_mapping(path):
    result = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                result[parts[0].upper()] = parts[1]
    return result

def read_entities(path):
    result = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                result[parts[1]] = parts[0]
    return result

def load_edges(path):
    by_date = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "subject_id", "relation_id",
            "object_id", "date"
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("FinDKG CSV missing columns")

        for row in reader:
            try:
                d = date.fromisoformat(row["date"])
            except (ValueError, TypeError):
                continue
            by_date[d].append(row)

    return by_date

def edge_key(e):
    # Preserve direction and relation type.
    return (
        e["subject_id"],
        e["relation_id"],
        e["object_id"]
    )

def company_edges(rows, entity_id):
    return {
        edge_key(e) for e in rows
        if entity_id in (
            e["subject_id"], e["object_id"]
        )
    }

def jaccard_change(a, b):
    if not a and not b:
        return None
    return 1.0 - len(a & b) / len(a | b)

def get_windows(by_date, query_date, weeks=4):
    eligible = sorted(
        d for d in by_date if d <= query_date
    )

    current_dates = eligible[-weeks:]
    previous_dates = eligible[-2*weeks:-weeks]

    return current_dates, previous_dates

def union_edges(by_date, dates, entity_id):
    result = set()
    for d in dates:
        result |= company_edges(
            by_date[d], entity_id
        )
    return result

def latest_company_snapshot(by_date, query_date, entity_id):
    # Find the latest snapshot containing this company.
    # This diagnoses sparse-week coverage.
    for d in sorted(by_date, reverse=True):
        if d > query_date:
            continue
        edges = company_edges(
            by_date[d], entity_id
        )
        if edges:
            return d, edges
    return None, set()

def audit_one(
    ticker, entity_id, query_date,
    by_date, weeks, entities
):
    current_dates, previous_dates = get_windows(
        by_date, query_date, weeks
    )

    current = union_edges(
        by_date, current_dates, entity_id
    )
    previous = union_edges(
        by_date, previous_dates, entity_id
    )

    recent_date, recent_edges = latest_company_snapshot(
        by_date, query_date, entity_id
    )

    current_count = len(current)
    previous_count = len(previous)

    shared = len(current & previous)
    union_count = len(current | previous)

    rolling_change = jaccard_change(
        current, previous
    )

    latest_age = (
        (query_date - recent_date).days
        if recent_date else None
    )

    current_coverage = sum(
        bool(company_edges(by_date[d], entity_id))
        for d in current_dates
    )

    previous_coverage = sum(
        bool(company_edges(by_date[d], entity_id))
        for d in previous_dates
    )

    enough_windows = (
        len(current_dates) == weeks
        and len(previous_dates) == weeks
    )

    # Distinguish missing vs sparse vs valid-but-unaudited.
    if not enough_windows:
        status = "insufficient_history"
        rolling_change = None
    elif not current:
        status = "no_recent_graph_evidence"
        rolling_change = None
    elif not previous:
        status = "no_previous_graph_evidence"
        rolling_change = None
    else:
        status = "computed_unverified"

    # The dataset has a graph/event date, NOT verified public
    # availability. No time-shift can prove public availability.
    return {
        "date": query_date.isoformat(),
        "ticker": ticker,
        "entity_id": entity_id,
        "entity_name": entities.get(
            entity_id, entity_id
        ),
        "status": status,
        "latest_entity_snapshot": (
            recent_date.isoformat()
            if recent_date else None
        ),
        "days_since_latest_entity_edge": latest_age,
        "latest_entity_edges": len(recent_edges),
        "current_window_dates": [
            d.isoformat() for d in current_dates
        ],
        "previous_window_dates": [
            d.isoformat() for d in previous_dates
        ],
        "current_coverage_weeks": current_coverage,
        "previous_coverage_weeks": previous_coverage,
        "current_unique_edges": current_count,
        "previous_unique_edges": previous_count,
        "shared_edges": shared,
        "union_edges": union_count,
        "rolling_graph_change": rolling_change,

        # Explicitly report that no availability audit passed.
        "availability_audited": False,
        "availability_status": (
            "unverified_public_availability"
        ),
        "usable_for_leakage_free_backtest": False,
    }

def diagnose_nvda(by_date, entities, mapping, last_date):
    print("\n========== NVDA DIAGNOSTIC ==========")

    canonical = mapping.get("NVDA")
    print("Portfolio NVDA ID:", canonical)
    print("Mapped entity:", entities.get(canonical))

    candidates = [
        (eid, name)
        for eid, name in entities.items()
        if "nvidia" in name.lower()
    ]

    print("NVIDIA candidate entities:", candidates[:20])

    for eid, name in candidates:
        hits = []
        for d in sorted(by_date):
            if d > last_date:
                continue
            matched = company_edges(
                by_date[d], eid
            )
            if matched:
                hits.append((d, len(matched)))

        print(f"\nEntity {eid}: {name}")
        print("Historical snapshots with edges:", len(hits))
        print(
            "Last 8 observed snapshots:",
            [(d.isoformat(), n) for d, n in hits[-8:]]
        )

    print(
        "\nNOTE: Do not automatically merge NVIDIA aliases. "
        "Entity 221 and 664 may represent distinct KG entities."
    )

def load_pilot_dates(path):
    if not Path(path).exists():
        return None

    dates = set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            dates.add(date.fromisoformat(r["date"]))

    return sorted(dates)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", default=EDGES)
    ap.add_argument("--mapping", default=MAP)
    ap.add_argument("--entity-file", default=ENTITY)
    ap.add_argument("--pilot", default=PILOT)
    ap.add_argument("--weeks", type=int, default=4)
    ap.add_argument(
        "--out",
        default="results/late_fusion_pilot/"
                "findkg_audit_v2.csv"
    )
    args = ap.parse_args()

    if args.weeks < 1:
        ap.error("--weeks must be positive")

    mapping = read_mapping(args.mapping)
    entities = read_entities(args.entity_file)
    by_date = load_edges(args.edges)

    dates = load_pilot_dates(args.pilot)
    if dates is None:
        dates = [
            date.fromisoformat(d)
            for d in DEFAULT_DATES
        ]

    print("Loaded FinDKG snapshots:", len(by_date))
    print("Dates:", [d.isoformat() for d in dates])

    rows = []
    for d in dates:
        for ticker in [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"
        ]:
            eid = mapping[ticker]

            row = audit_one(
                ticker, eid, d,
                by_date, args.weeks, entities
            )
            rows.append(row)

            gc = row["rolling_graph_change"]
            value = (
                f"{gc:.4f}" if gc is not None else "NA"
            )

            print(
                f"{d} {ticker:5s} "
                f"old={row['previous_unique_edges']:4d} "
                f"new={row['current_unique_edges']:4d} "
                f"shared={row['shared_edges']:4d} "
                f"rolling_change={value} "
                f"status={row['status']}"
            )

    output = Path(args.out)
    output.parent.mkdir(
        parents=True, exist_ok=True
    )

    with output.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(
            rows, indent=2, ensure_ascii=False
        ),
        encoding="utf-8"
    )

    diagnose_nvda(
        by_date, entities, mapping, max(dates)
    )

    print("\nSaved:", output)
    print("Saved:", json_output)
    print(
        "\nWARNING: Availability remains UNVERIFIED. "
        "Do not use these features in historical "
        "backtests until publication/release-time "
        "semantics are independently confirmed."
    )

if __name__ == "__main__":
    main()
