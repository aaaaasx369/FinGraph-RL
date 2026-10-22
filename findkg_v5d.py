import csv
import json
import argparse
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

EDGE_FILE = "kg/data/findkg/findkg_full_edges_with_dates.csv"
ENTITY_FILE = (
    "external/FinDKG-FinRL-Train2020/"
    "data/FinDKG/entity2id.txt"
)
REL_FILE = (
    "external/FinDKG-FinRL-Train2020/"
    "data/FinDKG/relation2id.txt"
)
PILOT_FILE = (
    "results/late_fusion_pilot/pilot_features.csv"
)

ALIASES = {
    "AAPL": ["9"],
    "MSFT": ["30"],
    "NVDA": ["221", "664"],
    "AMZN": ["20"],
    "GOOGL": ["11"]
}

def load_mapping(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                mapping[parts[1]] = parts[0]
    return mapping

def load_graph(path):
    graphs = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = date.fromisoformat(row["date"])
            graphs[d].append(row)
    return graphs

def normalize_entity(eid, aliases):
    if eid in aliases:
        return "TARGET_COMPANY"
    return eid

def edge_key(e, aliases):
    # Preserve edge direction and relation type.
    return (
        normalize_entity(e["subject_id"], aliases),
        e["relation_id"],
        normalize_entity(e["object_id"], aliases)
    )

def is_related(e, aliases):
    return (
        e["subject_id"] in aliases
        or e["object_id"] in aliases
    )

def get_window_dates(graphs, cutoff, weeks):
    dates = sorted(d for d in graphs if d <= cutoff)
    current = dates[-weeks:]
    previous = dates[-2 * weeks:-weeks]
    return current, previous

def collect_rows(graphs, dates, aliases):
    result = []
    for d in dates:
        for e in graphs[d]:
            if is_related(e, aliases):
                result.append(e)
    return result

def unique_edges(rows, aliases):
    return {edge_key(e, aliases) for e in rows}

def edge_turnover(a, b):
    if not a or not b:
        return None
    return 1.0 - len(a & b) / len(a | b)

def relation_distribution(rows, aliases, weighted):
    if weighted:
        relations = [
            e["relation_id"] for e in rows
        ]
    else:
        relations = [
            e[1] for e in unique_edges(rows, aliases)
        ]

    return Counter(relations)

def tv_distance(a, b):
    if not a or not b:
        return None

    total_a = sum(a.values())
    total_b = sum(b.values())

    relations = set(a) | set(b)

    return 0.5 * sum(
        abs(
            a[r] / total_a -
            b[r] / total_b
        )
        for r in relations
    )

def readable_edge(e, entities, relations):
    return {
        "subject_id": e["subject_id"],
        "subject": entities.get(
            e["subject_id"], e["subject_id"]
        ),
        "relation_id": e["relation_id"],
        "relation": relations.get(
            e["relation_id"], e["relation_id"]
        ),
        "object_id": e["object_id"],
        "object": entities.get(
            e["object_id"], e["object_id"]
        ),
        "date": e["date"]
    }

def calculate(
    ticker, decision_date, graphs,
    entities, relations, weeks, lag_days,
    top_k=10
):
    aliases = set(ALIASES[ticker])
    decision = date.fromisoformat(decision_date)

    # Sensitivity assumption ONLY.
    # This does not verify actual publication time.
    cutoff = decision - timedelta(days=lag_days)

    current_dates, previous_dates = get_window_dates(
        graphs, cutoff, weeks
    )

    current = collect_rows(
        graphs, current_dates, aliases
    )
    previous = collect_rows(
        graphs, previous_dates, aliases
    )

    current_set = unique_edges(current, aliases)
    previous_set = unique_edges(previous, aliases)

    enough_history = (
        len(current_dates) == weeks
        and len(previous_dates) == weeks
    )

    has_both = bool(current_set) and bool(previous_set)

    valid = enough_history and has_both

    if not enough_history:
        status = "insufficient_history"
    elif not current_set:
        status = "no_recent_edges"
    elif not previous_set:
        status = "no_previous_edges"
    else:
        status = "computed_unverified"

    if valid:
        turnover = edge_turnover(
            previous_set, current_set
        )

        relation_unique = tv_distance(
            relation_distribution(
                previous, aliases, False
            ),
            relation_distribution(
                current, aliases, False
            )
        )

        relation_weighted = tv_distance(
            relation_distribution(
                previous, aliases, True
            ),
            relation_distribution(
                current, aliases, True
            )
        )
    else:
        turnover = None
        relation_unique = None
        relation_weighted = None

    current_coverage = sum(
        bool(collect_rows(graphs, [d], aliases))
        for d in current_dates
    )

    previous_coverage = sum(
        bool(collect_rows(graphs, [d], aliases))
        for d in previous_dates
    )

    # Coverage uses the requested number of snapshots.
    coverage = current_coverage / weeks

    last_observed = max(
        (
            d for d in graphs
            if d <= cutoff
            and any(
                is_related(e, aliases)
                for e in graphs[d]
            )
        ),
        default=None
    )

    # Deduplicate identical relations in the evidence output.
    evidence = {}
    for e in current:
        k = edge_key(e, aliases)
        existing = evidence.get(k)
        if (
            existing is None
            or e["date"] > existing["date"]
        ):
            evidence[k] = e

    selected = sorted(
        evidence.values(),
        key=lambda e: (
            e["date"],
            e["relation_id"],
            e["subject_id"],
            e["object_id"]
        ),
        reverse=True
    )[:top_k]

    row = {
        "date": decision_date,
        "ticker": ticker,
        "lag_days_assumed": lag_days,
        "graph_cutoff": cutoff.isoformat(),
        "status": status,

        "edge_turnover": turnover,
        "relation_change_unique": relation_unique,
        "relation_change_weighted": relation_weighted,

        "graph_coverage": coverage,
        "current_observed_weeks": current_coverage,
        "previous_observed_weeks": previous_coverage,

        "current_unique_edges": len(current_set),
        "previous_unique_edges": len(previous_set),
        "shared_edges": len(
            current_set & previous_set
        ),

        "current_raw_rows": len(current),
        "previous_raw_rows": len(previous),

        "latest_observed_date": (
            last_observed.isoformat()
            if last_observed else None
        ),
        "days_since_last_edge": (
            (decision - last_observed).days
            if last_observed else None
        ),

        "entity_ids_used": "|".join(
            sorted(aliases, key=int)
        ),

        "availability_audited": False,
        "usable_for_leakage_free_backtest": False
    }

    detail = {
        "date": decision_date,
        "ticker": ticker,
        "aliases": sorted(aliases),
        "lag_days_assumed": lag_days,
        "current_window_dates": [
            d.isoformat() for d in current_dates
        ],
        "previous_window_dates": [
            d.isoformat() for d in previous_dates
        ],
        "features": row,
        "graph_evidence": [
            readable_edge(e, entities, relations)
            for e in selected
        ],
        "warnings": [
            "Alias merging is provisional.",
            "Some FinDKG relations may be noisy.",
            "Snapshot date is not verified publication time.",
            "Lag is a sensitivity assumption, not proof "
            "of historical availability.",
            "Do not use for leakage-free trading backtests."
        ]
    }

    return row, detail

def get_pilot_jobs(path):
    jobs = []
    seen = set()

    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["date"], row["ticker"])
            if key not in seen:
                seen.add(key)
                jobs.append(key)

    return jobs

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--edges", default=EDGE_FILE
    )
    parser.add_argument(
        "--pilot", default=PILOT_FILE
    )
    parser.add_argument(
        "--weeks", type=int, default=4
    )
    parser.add_argument(
        "--lags", nargs="+", type=int,
        default=[0, 7, 14]
    )
    parser.add_argument(
        "--top-k", type=int, default=10
    )
    parser.add_argument(
        "--out-dir",
        default="results/late_fusion_pilot/v5d"
    )

    args = parser.parse_args()

    if args.weeks < 1 or args.top_k < 1:
        parser.error("weeks and top-k must be positive")
    if any(x < 0 for x in args.lags):
        parser.error("lags cannot be negative")

    graphs = load_graph(args.edges)
    entities = load_mapping(ENTITY_FILE)
    relations = load_mapping(REL_FILE)

    jobs = get_pilot_jobs(args.pilot)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True, exist_ok=True
    )

    all_rows = []
    all_details = []

    for lag in args.lags:
        for day, ticker in jobs:

            row, detail = calculate(
                ticker=ticker,
                decision_date=day,
                graphs=graphs,
                entities=entities,
                relations=relations,
                weeks=args.weeks,
                lag_days=lag,
                top_k=args.top_k
            )

            all_rows.append(row)
            all_details.append(detail)

            print(
                day, ticker,
                f"lag={lag}",
                f"edges={row['current_unique_edges']}",
                f"coverage={row['graph_coverage']:.2f}",
                f"turnover={row['edge_turnover']}",
                f"relation={row['relation_change_unique']}",
                f"status={row['status']}"
            )

    csv_path = out_dir / "findkg_v5d_features.csv"
    json_path = out_dir / "findkg_v5d_evidence.json"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(all_rows[0])
        )
        writer.writeheader()
        writer.writerows(all_rows)

    json_path.write_text(
        json.dumps(
            all_details,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print(
        "\nIMPORTANT: All outputs remain "
        "availability_audited=False."
    )

if __name__ == "__main__":
    main()
