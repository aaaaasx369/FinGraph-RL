#!/usr/bin/env python3
"""
V15 Historical Multi-Company Feature Store

Inputs
------
--market       Market CSV containing date and tic/ticker
--graph        Historical FinDKG dated edges CSV
--mapping-dir  FinDKG entity/relation mapping directory
--v11-root     Existing V11 clean_evidence.json directories

Outputs
-------
features_long.csv
coverage.csv
temporal_audit.csv
report.json
finrl_full_exploratory.csv (optional)
finrl_<condition>.csv (optional)

Important
---------
1. Uses observations STRICTLY BEFORE each trading date.
2. FinDKG observation dates are NOT verified publication dates.
3. SEC/ReflectKG coverage only comes from actual existing snapshots.
4. No synthetic historical SEC filings or ReflectKG relations.
5. Zero-with-mask is an explicit exploratory modeling strategy.
6. Formal financial backtest approval is always False.
7. No model inference, training, or internet requests.
"""

import argparse
import csv
import json
import math
import re
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

TICKER_IDS = {
    "AAPL": 9,
    "MSFT": 30,
    "NVDA": 221,
    "AMZN": 20,
    "GOOGL": 11,
}

SEC_PATTERNS = {
    "PANDEMIC": r"\bCOVID[-\s]?19\b|\bpandemic\b",
    "SUPPLY_CHAIN": (
        r"\bsupply[\s-]?chain\b|"
        r"\bcomponent shortages?\b"
    ),
    "DEMAND": (
        r"\bcustomer spending\b|"
        r"\brecession\b|"
        r"\bweak(?:er)? demand\b"
    ),
    "LITIGATION": (
        r"\blegal proceedings\b|"
        r"\blitigation\b|\blawsuits?\b"
    ),
    "REGULATION": (
        r"\bexport controls?\b|"
        r"\bexport restrictions?\b|"
        r"\bgovernment regulations?\b|"
        r"\bregulatory requirements?\b"
    ),
    "CYBERSECURITY": (
        r"\bcybersecurity\b|"
        r"\bcyberattacks?\b|"
        r"\bdata breaches?\b"
    ),
}

SEC_COLUMNS = (
    ["sec_risk_category_count", "sec_mention_count"]
    + ["sec_" + x for x in SEC_PATTERNS]
)

FIN_COLUMNS = [
    "fin_relation_count",
    "fin_produce_count",
    "fin_operate_in_count",
    "fin_distinct_object_count",
]

REF_COLUMNS = [
    "ref_relation_count",
    "ref_supplier_relation_count",
    "ref_customer_relation_count",
    "ref_competitor_relation_count",
    "ref_same_sector_relation_count",
    "ref_observation_event_count",
    "ref_first_observed_count",
    "ref_reasserted_count",
]

FEATURE_COLUMNS = SEC_COLUMNS + FIN_COLUMNS + REF_COLUMNS

CONDITIONS = {
    "Clean": (),
    "SEC-only": ("SEC",),
    "FinDKG-only": ("FinDKG",),
    "ReflectKG-only": ("ReflectKG",),
    "SEC+FinDKG": ("SEC", "FinDKG"),
    "SEC+ReflectKG": ("SEC", "ReflectKG"),
    "Full": ("SEC", "FinDKG", "ReflectKG"),
}

GROUP_COLS = {
    "SEC": SEC_COLUMNS,
    "FinDKG": FIN_COLUMNS,
    "ReflectKG": REF_COLUMNS,
}

def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def save_csv(path, rows, cols):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=cols, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def day(value):
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None

def point_in_time(value, trade_date):
    d = day(value)
    return d is not None and d < trade_date

def mapping(path):
    result = {}
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            cells = line.rstrip("\n").split("\t")
            if len(cells) < 2:
                continue
            try:
                key = int(cells[1])
            except ValueError:
                continue
            result[key] = cells[0]
    if not result:
        raise ValueError("Empty mapping " + str(path))
    return result

def read_market(path):
    records = read_csv(path)
    if not records:
        raise ValueError("Empty market data")

    if "date" not in records[0]:
        raise ValueError("Market CSV requires date")

    ticker_key = (
        "tic" if "tic" in records[0] else "ticker"
    )
    if ticker_key not in records[0]:
        raise ValueError("Market CSV requires tic or ticker")

    seen = set()
    selected = []

    for record in records:
        ticker = str(record[ticker_key]).upper()
        if ticker not in TICKER_IDS:
            continue

        d = day(record["date"])
        if d is None:
            raise ValueError("Invalid market date")

        key = (d.isoformat(), ticker)
        if key in seen:
            raise ValueError("Duplicate market row " + str(key))

        seen.add(key)
        selected.append((d, ticker, record))

    selected.sort(key=lambda x: (x[0], x[1]))
    if not selected:
        raise ValueError("No matching portfolio tickers")

    return selected

def index_findkg(path):
    """
    Index ONLY direct focal-company edges.
    Avoid loading an enormous 2-hop graph for every date.
    """
    reverse_ids = {
        idx: ticker for ticker, idx in TICKER_IDS.items()
    }

    indexed = defaultdict(list)
    stats = Counter()

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "subject_id", "relation_id",
            "object_id", "date"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"FinDKG missing columns: {missing}"
            )

        for record in reader:
            stats["total_graph_rows"] += 1

            try:
                s = int(record["subject_id"])
                o = int(record["object_id"])
                r = int(record["relation_id"])
            except (TypeError, ValueError):
                stats["invalid_ids"] += 1
                continue

            observed = day(record["date"])
            if observed is None:
                stats["invalid_dates"] += 1
                continue

            involved = set()
            if s in reverse_ids:
                involved.add(reverse_ids[s])
            if o in reverse_ids:
                involved.add(reverse_ids[o])

            for ticker in involved:
                indexed[ticker].append({
                    "date": observed,
                    "s": s,
                    "r": r,
                    "o": o,
                })
                stats["focal_observation_rows"] += 1

    for ticker in TICKER_IDS:
        indexed[ticker].sort(
            key=lambda e: (
                e["date"], e["s"], e["r"], e["o"]
            )
        )

    return indexed, stats

def fin_features(
    ticker, trade_date, indexed,
    entities, relations, lookback_days
):
    history = indexed.get(ticker, [])
    focal_id = TICKER_IDS[ticker]
    cutoff = trade_date.toordinal()

    newest = {}
    rejected = Counter()

    # The focal index is typically small enough to scan.
    # Strictly exclude current and future dataset dates.
    for e in history:
        if e["date"] >= trade_date:
            break

        age = cutoff - e["date"].toordinal()
        if age > lookback_days:
            continue

        relation = relations.get(e["r"])
        if relation not in ("Produce", "Operate_In"):
            rejected["ambiguous_relation"] += 1
            continue

        # Only accept descriptive statements
        # where the focal company is the subject.
        if e["s"] != focal_id:
            rejected["wrong_direction"] += 1
            continue

        if e["s"] not in entities or e["o"] not in entities:
            rejected["unmapped_entity"] += 1
            continue

        triple = (e["s"], e["r"], e["o"])
        previous = newest.get(triple)

        if previous is None or e["date"] > previous["date"]:
            newest[triple] = e

    if not newest:
        return None, [], rejected

    events = list(newest.values())
    relation_counts = Counter(
        relations[e["r"]] for e in events
    )

    ids = [
        f"FIN:{e['s']}:{e['r']}:{e['o']}:{e['date']}"
        for e in events
    ]

    result = {
        "fin_relation_count": len(events),
        "fin_produce_count": relation_counts["Produce"],
        "fin_operate_in_count": relation_counts["Operate_In"],
        "fin_distinct_object_count": len({
            e["o"] for e in events
        }),
    }
    return result, ids, rejected

def discover_v11(root):
    """
    Existing V11 snapshots are indexed by ticker and as_of.
    A generated snapshot is never backdated.
    """
    snapshots = defaultdict(list)
    root = Path(root)

    if not root.exists():
        return snapshots

    for path in root.rglob("clean_evidence.json"):
        try:
            data = json.loads(path.read_text(
                encoding="utf-8"
            ))
            ticker = str(data["ticker"]).upper()
            d = day(data["as_of"])
            if ticker not in TICKER_IDS or d is None:
                continue
            snapshots[ticker].append((d, data, str(path)))
        except (ValueError, KeyError, TypeError):
            continue

    for ticker in snapshots:
        snapshots[ticker].sort(key=lambda x: x[0])

    return snapshots

def select_snapshot(snapshots, ticker, trade_date, max_age):
    eligible = [
        x for x in snapshots.get(ticker, [])
        if x[0] < trade_date
        and (trade_date - x[0]).days <= max_age
    ]
    if not eligible:
        return None
    return eligible[-1]

def sec_features(snapshot, trade_date):
    if snapshot is None:
        return None, []

    _, data, _ = snapshot
    counts = Counter()
    ids = set()

    for item in data.get("sec_evidence", []):
        if not point_in_time(
            item.get("published_at"), trade_date
        ):
            continue

        text = str(item.get("text", ""))
        for category, pattern in SEC_PATTERNS.items():
            count = len(re.findall(pattern, text, re.I))
            if count:
                counts[category] += count
                ids.add(str(item["evidence_id"]))

    if not ids:
        return None, []

    result = {
        "sec_risk_category_count": sum(
            counts[x] > 0 for x in SEC_PATTERNS
        ),
        "sec_mention_count": sum(counts.values()),
    }
    for category in SEC_PATTERNS:
        result["sec_" + category] = int(
            counts[category] > 0
        )

    return result, sorted(ids)

def reflect_features(snapshot, trade_date):
    if snapshot is None:
        return None, []

    _, data, _ = snapshot
    counts = Counter()
    ids = set()

    relation_names = {
        "SUPPLIES_TO", "CUSTOMER_OF",
        "COMPETES_WITH", "SAME_SECTOR"
    }

    for item in data.get("reflectkg_snapshot", []):
        if not point_in_time(
            item.get("last_observed_at"), trade_date
        ):
            continue

        relation = item.get("relation")
        if relation not in relation_names:
            continue

        counts[relation] += 1
        ids.add(str(item["evidence_id"]))

    event_count = 0
    for item in data.get("reflectkg_changes", []):
        if not point_in_time(
            item.get("published_at"), trade_date
        ):
            continue

        event_count += 1
        event_type = str(item.get("event_type", ""))
        counts["EVENT_" + event_type] += 1
        ids.add(str(item["evidence_id"]))

    if not ids:
        return None, []

    result = {
        "ref_relation_count": sum(
            counts[x] for x in relation_names
        ),
        "ref_supplier_relation_count":
            counts["SUPPLIES_TO"],
        "ref_customer_relation_count":
            counts["CUSTOMER_OF"],
        "ref_competitor_relation_count":
            counts["COMPETES_WITH"],
        "ref_same_sector_relation_count":
            counts["SAME_SECTOR"],
        "ref_observation_event_count": event_count,
        "ref_first_observed_count":
            counts["EVENT_FIRST_OBSERVED"],
        "ref_reasserted_count":
            counts["EVENT_REASSERTED"],
    }
    return result, sorted(ids)

def build(args):
    market = read_market(args.market)
    indexed, graph_stats = index_findkg(args.graph)

    mapping_dir = Path(args.mapping_dir)
    entities = mapping(mapping_dir / "entity2id.txt")
    relations = mapping(mapping_dir / "relation2id.txt")
    snapshots = discover_v11(args.v11_root)

    features = []
    coverage = []
    audit = []
    stats = Counter()

    cache = {}
    for trade_date, ticker, market_record in market:
        cache_key = (trade_date, ticker)

        if cache_key not in cache:
            snapshot = select_snapshot(
                snapshots, ticker, trade_date,
                args.snapshot_max_age
            )

            sec, sec_ids = sec_features(
                snapshot, trade_date
            )
            fin, fin_ids, fin_reject = fin_features(
                ticker, trade_date, indexed,
                entities, relations,
                args.findkg_lookback
            )
            ref, ref_ids = reflect_features(
                snapshot, trade_date
            )

            cache[cache_key] = {
                "snapshot": snapshot,
                "SEC": (sec, sec_ids),
                "FinDKG": (fin, fin_ids),
                "ReflectKG": (ref, ref_ids),
            }

            for reason, count in fin_reject.items():
                stats["findkg_" + reason] += count

        record = cache[cache_key]

        for condition, included in CONDITIONS.items():
            row = {
                "date": trade_date.isoformat(),
                "tic": ticker,
                "condition": condition,
                "mode": "EXPLORATORY",
                "source_snapshot_as_of": (
                    record["snapshot"][0].isoformat()
                    if record["snapshot"] else ""
                ),
                "kg_mask_sec": 0,
                "kg_mask_findkg": 0,
                "kg_mask_reflectkg": 0,
                "public_availability_verified": 0,
                "provenance_verified": 0,
                "usable_for_formal_finrl": 0,
                "quality_status": "",
                "evidence_ids": "",
            }

            for f in FEATURE_COLUMNS:
                row[f] = None

            ids = []
            missing = []

            for group in included:
                values, group_ids = record[group]

                if values is None:
                    missing.append(group)
                    continue

                for key, value in values.items():
                    row[key] = value

                key = {
                    "SEC": "kg_mask_sec",
                    "FinDKG": "kg_mask_findkg",
                    "ReflectKG": "kg_mask_reflectkg",
                }[group]
                row[key] = 1
                ids.extend(
                    f"{group}:{x}" for x in group_ids
                )

            row["evidence_ids"] = "|".join(ids)

            if condition == "Clean":
                status = "CLEAN_MARKET_BASELINE"
            elif missing:
                status = "EXPLORATORY_PARTIAL_COVERAGE"
            else:
                status = "EXPLORATORY_ALL_SOURCES_PRESENT"

            row["quality_status"] = status
            stats[status] += 1
            features.append(row)

            coverage.append({
                "date": row["date"],
                "tic": ticker,
                "condition": condition,
                "has_sec": row["kg_mask_sec"],
                "has_findkg": row["kg_mask_findkg"],
                "has_reflectkg": row["kg_mask_reflectkg"],
                "missing_sources": "|".join(missing),
                "status": status,
            })

            if missing:
                audit.append({
                    "date": row["date"],
                    "tic": ticker,
                    "condition": condition,
                    "issue": "MISSING_HISTORICAL_SOURCE",
                    "detail": "|".join(missing),
                })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = [
        "date", "tic", "condition", "mode",
        "source_snapshot_as_of",
        "kg_mask_sec", "kg_mask_findkg",
        "kg_mask_reflectkg",
        "public_availability_verified",
        "provenance_verified",
        "usable_for_formal_finrl",
        "quality_status", "evidence_ids",
    ]

    save_csv(
        out / "features_long.csv",
        features,
        metadata + FEATURE_COLUMNS
    )

    save_csv(
        out / "coverage.csv",
        coverage,
        [
            "date", "tic", "condition",
            "has_sec", "has_findkg",
            "has_reflectkg",
            "missing_sources", "status",
        ]
    )

    save_csv(
        out / "temporal_audit.csv",
        audit,
        ["date", "tic", "condition", "issue", "detail"]
    )

    report = {
        "mode": "EXPLORATORY_ONLY",
        "market_rows": len(market),
        "feature_rows": len(features),
        "tickers": sorted({x[1] for x in market}),
        "trading_dates": len({x[0] for x in market}),
        "discovered_v11_snapshots": {
            t: len(snapshots.get(t, []))
            for t in TICKER_IDS
        },
        "status_counts": dict(stats),
        "findkg_graph_stats": dict(graph_stats),
        "formal_backtest_approved": False,
        "limitations": [
            "FinDKG dates are observation dates, "
            "not independently verified publication dates.",
            "Existing V11 snapshots are not re-created "
            "for dates earlier than their as_of.",
            "SEC and ReflectKG sources may be missing.",
            "Feature zero indicates an observed count, "
            "not an estimated financial risk probability.",
            "This script does not train FinRL agents.",
        ],
    }
    save_json(out / "report.json", report)

    if args.export_finrl:
        export_finrl(
            market, features, out,
            args.missing_policy
        )

    print("\n=== V15 SUMMARY ===")
    print("Market rows:", len(market))
    print("Feature rows:", len(features))
    print("Dates:", report["trading_dates"])
    print("Tickers:", report["tickers"])
    print("V11 snapshots:",
          report["discovered_v11_snapshots"])
    print("Statuses:", dict(stats))
    print("Saved:", out)

def export_finrl(market, features, out, policy):
    """
    Export seven exploratory market+feature datasets.
    Zero-with-mask is a modeling option, never a claim
    that absent evidence equals zero risk.
    """
    market_map = {
        (d.isoformat(), t): record
        for d, t, record in market
    }

    for condition in CONDITIONS:
        rows = []

        for f in features:
            if f["condition"] != condition:
                continue

            key = (f["date"], f["tic"])
            m = dict(market_map[key])
            m["date"] = f["date"]
            m["tic"] = f["tic"]

            m["kg_condition"] = condition
            m["kg_mask_sec"] = f["kg_mask_sec"]
            m["kg_mask_findkg"] = f["kg_mask_findkg"]
            m["kg_mask_reflectkg"] = f["kg_mask_reflectkg"]

            m["kg_quality_status"] = f["quality_status"]

            # Preserve all market columns, append features.
            for field in FEATURE_COLUMNS:
                value = f.get(field)

                if condition == "Clean":
                    m[field] = 0.0
                elif value is None:
                    if policy == "zero-with-mask":
                        m[field] = 0.0
                    else:
                        m[field] = ""
                else:
                    m[field] = float(value)

            rows.append(m)

        if not rows:
            continue

        cols = list(rows[0])
        safe_name = (
            condition.lower()
            .replace("+", "_plus_")
            .replace("-", "_")
        )

        save_csv(
            out / f"finrl_{safe_name}.csv",
            rows, cols
        )

def self_test():
    assert day("2021-06-01") == date(2021, 6, 1)
    assert point_in_time(
        "2021-05-26", date(2021, 6, 1)
    )
    assert not point_in_time(
        "2021-06-01", date(2021, 6, 1)
    )
    assert not point_in_time(
        "2021-06-02", date(2021, 6, 1)
    )

    sample = {
        "ticker": "NVDA",
        "sec_evidence": [{
            "evidence_id": "SEC_001",
            "published_at": "2021-05-26",
            "text": (
                "COVID-19 could disrupt our supply chain "
                "and customer spending due to travel restrictions."
            ),
        }]
    }

    snapshot = (
        date(2021, 6, 1), sample, "test"
    )

    result, ids = sec_features(
        snapshot, date(2021, 6, 2)
    )
    assert result["sec_PANDEMIC"] == 1
    assert result["sec_SUPPLY_CHAIN"] == 1
    assert result["sec_DEMAND"] == 1
    assert result["sec_REGULATION"] == 0
    assert ids == ["SEC_001"]

    assert sec_features(
        snapshot, date(2021, 5, 1)
    )[0] is None

    print("SELF-TEST PASSED")
    print("Historical date rule: PASS")
    print("SEC categories: PASS")
    print("Travel restriction false positive: PASS")

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--market")
    p.add_argument(
        "--graph",
        default=(
            "kg/data/findkg/"
            "findkg_full_edges_with_dates.csv"
        )
    )
    p.add_argument(
        "--mapping-dir",
        default=(
            "external/FinDKG-FinRL-Train2020/"
            "data/FinDKG"
        )
    )
    p.add_argument(
        "--v11-root", default="outputs/v11"
    )
    p.add_argument(
        "--findkg-lookback", type=int,
        default=365
    )
    p.add_argument(
        "--snapshot-max-age", type=int,
        default=90
    )
    p.add_argument(
        "--out-dir", default="outputs/v15/portfolio"
    )
    p.add_argument(
        "--export-finrl", action="store_true"
    )
    p.add_argument(
        "--missing-policy",
        choices=["keep-null", "zero-with-mask"],
        default="keep-null"
    )
    p.add_argument(
        "--self-test", action="store_true"
    )

    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.market:
        p.error("--market is required")

    if args.findkg_lookback <= 0:
        p.error("--findkg-lookback must be positive")

    if args.snapshot_max_age < 0:
        p.error("--snapshot-max-age must be >= 0")

    build(args)

if __name__ == "__main__":
    main()
