#!/usr/bin/env python3
"""
V14 — Temporal Multi-Company KG-RAG -> FinRL Dataset

Inputs:
  V13 features.csv with one row per ticker/as_of/condition
  Market CSV with date + tic/ticker
  Optional manually reviewed provenance manifest CSV
  Optional V13 temporal_quality.json

Outputs:
  aligned_features.csv
  finrl_feature_columns.json
  coverage_report.json
  temporal_audit.csv

Exploratory:
  Dataset-observation alignment only.
  NEVER claimed lookahead-free.

Strict:
  Requires human-reviewed evidence provenance and independently
  established public availability dates for every contributing ID.
  Does not fabricate provenance or auto-approve unverified sources.

Does not train agents or invent missing historical snapshots.
Python standard library only.
"""

import argparse
import csv
import json
import math
import hashlib
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

GROUPS = {
    "SEC": [
        "sec_risk_category_count",
        "sec_mention_count",
        "sec_PANDEMIC",
        "sec_SUPPLY_CHAIN",
        "sec_DEMAND",
        "sec_LITIGATION",
        "sec_REGULATION",
        "sec_CYBERSECURITY",
    ],
    "FinDKG": [
        "fin_relation_count",
        "fin_produce_count",
        "fin_operate_in_count",
        "fin_distinct_object_count",
    ],
    "ReflectKG": [
        "ref_relation_count",
        "ref_supplier_relation_count",
        "ref_customer_relation_count",
        "ref_competitor_relation_count",
        "ref_same_sector_relation_count",
        "ref_observation_event_count",
        "ref_first_observed_count",
        "ref_reasserted_count",
    ],
}

CONDITIONS = {
    "Clean": [],
    "SEC-only": ["SEC"],
    "FinDKG-only": ["FinDKG"],
    "ReflectKG-only": ["ReflectKG"],
    "SEC+FinDKG": ["SEC", "FinDKG"],
    "SEC+ReflectKG": ["SEC", "ReflectKG"],
    "Full": ["SEC", "FinDKG", "ReflectKG"],
}

BASE_COLUMNS = [
    "date", "tic", "condition",
    "snapshot_as_of", "snapshot_age_days",
    "snapshot_available",
    "sec_observed", "findkg_observed", "reflectkg_observed",
    "strict_verified",
    "usable_for_formal_finrl",
    "quality_status",
    "evidence_ids",
]

FEATURE_COLUMNS = [
    x for group in GROUPS.values() for x in group
]

AUDIT_COLUMNS = [
    "date", "tic", "condition", "snapshot_as_of",
    "evidence_id", "reason", "detail",
]

MANIFEST_COLUMNS = [
    "ticker",
    "evidence_id",
    "source",
    "public_available_date",
    "source_url_or_document",
    "reviewer",
    "review_status",
    "content_sha256",
]


def load_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_date(value):
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None


def split_ids(value):
    return sorted({
        s.strip() for s in str(value or "").split("|")
        if s.strip()
    })


def as_number(value):
    if value in ("", None):
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def as_bool(value):
    return str(value).strip().lower() in (
        "1", "true", "yes"
    )


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_snapshots(path):
    rows = load_csv(path)
    result = defaultdict(list)
    unique = set()

    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        condition = row.get("condition")
        asof = parse_date(row.get("as_of"))

        if ticker not in TICKERS:
            continue

        if condition not in CONDITIONS:
            raise ValueError("Unknown condition: " + str(condition))

        if asof is None:
            raise ValueError("Invalid snapshot as_of")

        key = (ticker, asof.isoformat(), condition)
        if key in unique:
            raise ValueError("Duplicate snapshot: " + str(key))
        unique.add(key)

        # Original V13 explicitly disallows future/same-day
        # observations within each snapshot.
        if not as_bool(row.get("historical_date_screen_pass")):
            raise ValueError(
                "Snapshot failed historical date screen: "
                + str(key)
            )

        for group in CONDITIONS[condition]:
            for feature in GROUPS[group]:
                if as_number(row.get(feature)) is None:
                    raise ValueError(
                        f"Missing feature {feature} for {key}"
                    )

        result[(ticker, condition)].append(
            (asof, row)
        )

    for key in result:
        result[key].sort(key=lambda pair: pair[0])

    return result


def read_market(path):
    rows = load_csv(path)
    if not rows:
        raise ValueError("Market CSV is empty")

    results = []
    seen = set()
    for row in rows:
        ticker = str(
            row.get("tic") or row.get("ticker") or ""
        ).upper()
        if ticker not in TICKERS:
            continue
        d = parse_date(row.get("date"))
        if d is None:
            raise ValueError(
                "Market CSV must contain ISO date values"
            )
        key = (d.isoformat(), ticker)
        if key in seen:
            raise ValueError(
                "Duplicate market date/ticker: " + str(key)
            )
        seen.add(key)

        results.append({
            "date": d,
            "ticker": ticker,
        })

    if not results:
        raise ValueError("No portfolio tickers in market CSV")

    results.sort(key=lambda x: (
        x["date"], x["ticker"]
    ))
    return results


def read_manifest(path):
    if not path:
        return {}

    rows = load_csv(path)
    manifest = {}

    for row in rows:
        ticker = str(row.get("ticker", "")).upper()
        eid = str(row.get("evidence_id", "")).strip()

        if not ticker or not eid:
            raise ValueError(
                "Provenance manifest requires ticker and evidence_id"
            )

        key = (ticker, eid)
        if key in manifest:
            raise ValueError(
                "Duplicate provenance manifest key: " + str(key)
            )
        manifest[key] = row

    return manifest


def select_snapshot(series, trade_day, max_age):
    """
    A snapshot can be consumed only on a LATER trading date.
    A snapshot dated 2021-06-01 cannot be consumed at the
    2021-06-01 market decision time.
    """
    best = None

    for observed, row in series:
        if observed >= trade_day:
            break

        age = (trade_day - observed).days
        if age <= max_age:
            best = (observed, row, age)

    return best


def validate_provenance(
    ticker, ids, snapshot_day, trade_day, manifest
):
    """
    Fail-closed manifest check.

    Reviewer's documentation must be inspected separately.
    A CSV row cannot independently prove truthful provenance.
    """
    issues = []

    if not ids:
        return ["NO_EVIDENCE"]

    for eid in ids:
        m = manifest.get((ticker, eid))

        if not m:
            issues.append(f"{eid}:MISSING_MANIFEST")
            continue

        available = parse_date(
            m.get("public_available_date")
        )
        status = str(
            m.get("review_status", "")
        ).strip().upper()
        reviewer = str(m.get("reviewer", "")).strip()
        uri = str(
            m.get("source_url_or_document", "")
        ).strip()
        digest = str(
            m.get("content_sha256", "")
        ).strip()

        if available is None:
            issues.append(f"{eid}:UNKNOWN_PUBLIC_DATE")
        elif available >= snapshot_day:
            issues.append(
                f"{eid}:NOT_AVAILABLE_BEFORE_SNAPSHOT"
            )
        elif available >= trade_day:
            issues.append(
                f"{eid}:NOT_AVAILABLE_BEFORE_TRADE"
            )

        if status != "VERIFIED":
            issues.append(f"{eid}:NOT_REVIEWED")

        if not reviewer or not uri:
            issues.append(f"{eid}:PROVENANCE_INCOMPLETE")

        # SHA256 must refer to actual reviewed source content.
        if len(digest) != 64:
            issues.append(f"{eid}:SOURCE_HASH_MISSING")
        else:
            try:
                int(digest, 16)
            except ValueError:
                issues.append(f"{eid}:INVALID_SOURCE_HASH")

    return issues


def build(args):
    snapshots = read_snapshots(args.features)
    market = read_market(args.market)
    manifest = read_manifest(args.provenance_manifest)

    aligned = []
    audits = []
    statuses = Counter()

    for market_row in market:
        trade_day = market_row["date"]
        ticker = market_row["ticker"]

        for condition, included in CONDITIONS.items():
            series = snapshots.get(
                (ticker, condition), []
            )
            chosen = select_snapshot(
                series, trade_day, args.max_age_days
            )

            output = {
                "date": trade_day.isoformat(),
                "tic": ticker,
                "condition": condition,
                "snapshot_as_of": "",
                "snapshot_age_days": "",
                "snapshot_available": 0,
                "sec_observed": 0,
                "findkg_observed": 0,
                "reflectkg_observed": 0,
                "strict_verified": 0,
                "usable_for_formal_finrl": 0,
                "quality_status": "NO_ELIGIBLE_SNAPSHOT",
                "evidence_ids": "",
            }
            for f in FEATURE_COLUMNS:
                output[f] = None

            if condition == "Clean":
                output["snapshot_available"] = 0
                output["strict_verified"] = 1
                output["quality_status"] = "CLEAN_MARKET_BASELINE"
                output["usable_for_formal_finrl"] = 0
                aligned.append(output)
                statuses["CLEAN_MARKET_BASELINE"] += 1
                continue

            if chosen is None:
                aligned.append(output)
                statuses[output["quality_status"]] += 1
                continue

            snapshot_day, source, age = chosen
            ids = split_ids(source.get("evidence_ids"))

            output["snapshot_as_of"] = (
                snapshot_day.isoformat()
            )
            output["snapshot_age_days"] = age
            output["snapshot_available"] = 1
            output["evidence_ids"] = "|".join(ids)

            missing_groups = []

            for group in included:
                prefix = {
                    "SEC": "sec",
                    "FinDKG": "findkg",
                    "ReflectKG": "reflectkg",
                }[group]

                # Source coverage is separate from feature
                # extraction. Empty sources are never
                # automatically converted into zero evidence.
                present = as_bool(
                    source.get(prefix + "_source_present")
                )

                if not present:
                    missing_groups.append(group)
                    continue

                output[prefix + "_observed"] = 1

                for feature in GROUPS[group]:
                    output[feature] = as_number(
                        source.get(feature)
                    )

            issues = []

            if missing_groups:
                issues.extend(
                    f"MISSING_SOURCE:{g}"
                    for g in missing_groups
                )

            if condition == "Clean":
                # Clean uses no KG/RAG evidence, but still
                # requires a real market record.
                verified = True
            elif args.mode == "strict":
                if not issues:
                    issues.extend(
                        validate_provenance(
                            ticker, ids, snapshot_day,
                            trade_day, manifest
                        )
                    )
                verified = not issues
            else:
                verified = False

            if args.mode == "strict":
                if issues:
                    status = "STRICT_BLOCKED"
                elif condition == "Clean":
                    status = "CLEAN_MARKET_BASELINE"
                else:
                    status = "STRICT_MANIFEST_SCREEN_PASS"
            else:
                status = (
                    "EXPLORATORY_MISSING_SOURCE"
                    if issues
                    else "EXPLORATORY_UNVERIFIED"
                )

            output["strict_verified"] = int(
                verified and args.mode == "strict"
            )

            # A manifest screen is not independent source
            # verification. Approval remains disabled.
            output["usable_for_formal_finrl"] = 0
            output["quality_status"] = status

            for issue in issues:
                audits.append({
                    "date": trade_day.isoformat(),
                    "tic": ticker,
                    "condition": condition,
                    "snapshot_as_of": (
                        snapshot_day.isoformat()
                    ),
                    "evidence_id": "",
                    "reason": issue,
                    "detail": "",
                })

            aligned.append(output)
            statuses[status] += 1

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_csv(
        out / "aligned_features.csv",
        aligned,
        BASE_COLUMNS + FEATURE_COLUMNS,
    )
    write_csv(
        out / "temporal_audit.csv",
        audits, AUDIT_COLUMNS
    )

    columns = {
        "all_features": FEATURE_COLUMNS,
        "groups": GROUPS,
        "conditions": CONDITIONS,
        "market_key": ["date", "tic"],
        "availability_mask": [
            "sec_observed",
            "findkg_observed",
            "reflectkg_observed",
        ],
        "null_policy": (
            "Unavailable or excluded features are blank. "
            "No zero imputation was performed."
        ),
    }
    save_json(
        out / "finrl_feature_columns.json", columns
    )

    report = {
        "mode": args.mode,
        "market_rows": len(market),
        "aligned_rows": len(aligned),
        "conditions": len(CONDITIONS),
        "unique_tickers": sorted({
            r["ticker"] for r in market
        }),
        "unique_dates": len({
            r["date"] for r in market
        }),
        "available_snapshot_rows": sum(
            r["snapshot_available"] == 1
            for r in aligned
        ),
        "strict_manifest_pass_rows": sum(
            r["quality_status"] ==
            "STRICT_MANIFEST_SCREEN_PASS"
            for r in aligned
        ),
        "status_counts": dict(statuses),
        "feature_input_sha256": sha256_file(
            args.features
        ),
        "market_input_sha256": sha256_file(
            args.market
        ),
        "maximum_snapshot_age_days": (
            args.max_age_days
        ),
        "formal_finrl_approved": False,
        "limitations": [
            "Alignment does not create new historical evidence.",
            "Exploratory mode is not proof of no lookahead bias.",
            "Manifest entries require independent verification.",
            "Strict screen pass does not guarantee factual truth.",
            "Missing values have not been imputed.",
            "FinRL training and performance tests not run.",
        ],
    }
    save_json(
        out / "coverage_report.json", report
    )

    print("\n=== V14 RESULTS ===")
    print("Mode:", args.mode)
    print("Market rows:", len(market))
    print("Aligned rows:", len(aligned))
    print("Tickers:", report["unique_tickers"])
    print("Trading dates:", report["unique_dates"])
    print("Snapshot-covered rows:",
          report["available_snapshot_rows"])
    print("Statuses:", dict(statuses))
    print("Saved:", out)


def self_test():
    assert parse_date("2021-06-01") == date(2021, 6, 1)
    assert as_number("") is None
    assert as_number("0") == 0
    assert as_number("nan") is None
    assert split_ids("FIN_001|FIN_002|FIN_001") == [
        "FIN_001", "FIN_002"
    ]

    samples = [
        (date(2021, 6, 1), {"x": 1}),
        (date(2021, 6, 8), {"x": 2}),
    ]
    assert select_snapshot(
        samples, date(2021, 6, 1), 90
    ) is None

    x = select_snapshot(
        samples, date(2021, 6, 2), 90
    )
    assert x[1]["x"] == 1

    y = select_snapshot(
        samples, date(2021, 6, 10), 90
    )
    assert y[1]["x"] == 2

    assert select_snapshot(
        samples, date(2022, 6, 1), 90
    ) is None

    assert validate_provenance(
        "NVDA", ["SEC_001"],
        date(2021, 6, 1), date(2021, 6, 2), {}
    )

    print("SELF-TEST PASSED")
    print("Future snapshots excluded.")
    print("Same-day snapshots excluded.")
    print("Expired snapshots excluded.")
    print("Missing provenance blocked.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features",
        default="outputs/v13/nvda_20210601/features.csv"
    )
    p.add_argument("--market")
    p.add_argument(
        "--mode", choices=["exploratory", "strict"],
        default="exploratory"
    )
    p.add_argument("--provenance-manifest")
    p.add_argument(
        "--max-age-days", type=int, default=90
    )
    p.add_argument(
        "--out-dir", default="outputs/v14"
    )
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.market:
        p.error("--market is required")

    if args.max_age_days < 0:
        p.error("--max-age-days must be >= 0")

    if args.mode == "strict" and not args.provenance_manifest:
        p.error(
            "Strict mode requires --provenance-manifest"
        )

    build(args)


if __name__ == "__main__":
    main()
