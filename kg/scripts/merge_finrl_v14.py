#!/usr/bin/env python3
"""Merge V14 aligned features into an existing FinRL market CSV.

Research adapter only; does not change FinRL agents or train.
Exploratory execution requires explicit opt-in.
"""
import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

FEATURES = [
    "sec_risk_category_count",
    "sec_mention_count",
    "sec_PANDEMIC",
    "sec_SUPPLY_CHAIN",
    "sec_DEMAND",
    "sec_LITIGATION",
    "sec_REGULATION",
    "sec_CYBERSECURITY",
    "fin_relation_count",
    "fin_produce_count",
    "fin_operate_in_count",
    "fin_distinct_object_count",
    "ref_relation_count",
    "ref_supplier_relation_count",
    "ref_customer_relation_count",
    "ref_competitor_relation_count",
    "ref_same_sector_relation_count",
    "ref_observation_event_count",
    "ref_first_observed_count",
    "ref_reasserted_count",
]

PREFIXES = {
    "SEC": "sec_",
    "FinDKG": "fin_",
    "ReflectKG": "ref_",
}

def read_csv(p):
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def normalize_date(s):
    from datetime import date
    return date.fromisoformat(str(s)[:10]).isoformat()

def write_csv(p, rows, columns):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=columns, extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--market", required=True)
    p.add_argument("--aligned", required=True)
    p.add_argument("--condition", default="Full")
    p.add_argument("--out", required=True)
    p.add_argument(
        "--allow-exploratory", action="store_true"
    )
    p.add_argument(
        "--missing-policy",
        choices=["drop", "zero-with-mask"],
        default="drop"
    )
    args = p.parse_args()

    if not args.allow_exploratory:
        raise SystemExit(
            "BLOCKED: No formally verified feature dataset. "
            "Use --allow-exploratory for research diagnostics."
        )

    market = read_csv(args.market)
    aligned = read_csv(args.aligned)

    if not market:
        raise ValueError("Empty market CSV")

    if "date" not in market[0]:
        raise ValueError("Market must contain date")

    ticker_col = (
        "tic" if "tic" in market[0] else "ticker"
    )
    if ticker_col not in market[0]:
        raise ValueError("Market must contain tic or ticker")

    if any(f in market[0] for f in FEATURES):
        raise ValueError(
            "Market already contains KG feature names"
        )

    selected = {}
    for a in aligned:
        if a["condition"] != args.condition:
            continue

        key = (
            normalize_date(a["date"]),
            a["tic"].upper(),
        )
        if key in selected:
            raise ValueError(
                "Duplicate aligned date/ticker: " + str(key)
            )
        selected[key] = a

    if not selected:
        raise ValueError(
            "No aligned data for condition " + args.condition
        )

    records = []
    stats = Counter()

    for row in market:
        key = (
            normalize_date(row["date"]),
            str(row[ticker_col]).upper(),
        )
        match = selected.get(key)
        if match is None:
            stats["no_alignment"] += 1
            continue

        new = dict(row)
        new["date"] = key[0]
        new["tic"] = key[1]
        new["kg_snapshot_available"] = int(
            match["snapshot_available"] == "1"
        )
        new["kg_strict_verified"] = int(
            match["strict_verified"] == "1"
        )
        new["kg_quality_status"] = match["quality_status"]
        new["kg_snapshot_as_of"] = match["snapshot_as_of"]

        # A feature is observed only if its source is both
        # included and available in the selected condition.
        observed = {
            "SEC": match["sec_observed"] == "1",
            "FinDKG": match["findkg_observed"] == "1",
            "ReflectKG": match["reflectkg_observed"] == "1",
        }

        for group in PREFIXES:
            new["kg_mask_" + group.lower()] = int(
                observed[group]
            )

        missing = False
        for feature in FEATURES:
            raw = match.get(feature, "")
            if raw in ("", None):
                missing = True
                if args.missing_policy == "zero-with-mask":
                    new[feature] = 0.0
                else:
                    new[feature] = ""
            else:
                val = float(raw)
                if not math.isfinite(val):
                    raise ValueError(
                        "Non-finite feature: " + feature
                    )
                new[feature] = val

        if missing and args.missing_policy == "drop":
            stats["missing_dropped"] += 1
            continue

        # Explicitly require an available snapshot even for
        # the exploratory zero-imputation option.
        # Clean market baseline should instead use market
        # CSV directly without this adapter.
        if not new["kg_snapshot_available"]:
            stats["no_snapshot_dropped"] += 1
            continue

        records.append(new)
        stats["written"] += 1

    columns = list(market[0])
    for c in (
        "tic", "kg_snapshot_available",
        "kg_strict_verified", "kg_quality_status",
        "kg_snapshot_as_of",
        "kg_mask_sec", "kg_mask_findkg",
        "kg_mask_reflectkg",
    ) + tuple(FEATURES):
        if c not in columns:
            columns.append(c)

    if records:
        write_csv(args.out, records, columns)
    else:
        print("No eligible records; output CSV not written.")

    report = {
        "condition": args.condition,
        "mode": "EXPLORATORY_ONLY",
        "missing_policy": args.missing_policy,
        "rows": dict(stats),
        "usable_for_formal_backtest": False,
        "warning": (
            "Zero-with-mask is an explicit modeling "
            "choice, not evidence that absent risks are zero. "
            "Dropping missing data can create selection bias."
        ),
    }
    report_path = args.out + ".report.json"
    Path(report_path).parent.mkdir(
        parents=True, exist_ok=True
    )
    Path(report_path).write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Output:", args.out)


if __name__ == "__main__":
    main()
