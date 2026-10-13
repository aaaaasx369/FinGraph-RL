#!/usr/bin/env python3
"""
V13: Deterministic Temporal Multi-Source Feature Builder.

Reads V11 clean_evidence.json files.
Builds 6 source-ablation feature vectors per (ticker, as_of).

- No LLM-generated scores
- No automatic missing-value imputation
- No future/same-day observations
- No unverified KG edges upgraded to facts
- No automatic approval for point-in-time FinRL backtesting

Outputs:
  features.csv
  evidence_audit.csv
  temporal_quality.json
  feature_schema.json

Only Python standard library is required.
"""
import argparse
import csv
import glob
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path

CONDITIONS = {
    "Clean": (),
    "SEC-only": ("SEC",),
    "FinDKG-only": ("FinDKG",),
    "ReflectKG-only": ("ReflectKG",),
    "SEC+FinDKG": ("SEC", "FinDKG"),
    "SEC+ReflectKG": ("SEC", "ReflectKG"),
    "Full": ("SEC", "FinDKG", "ReflectKG"),
}

# Conservative lexicon: mentions, NOT probability/severity.
SEC_RULES = {
    "PANDEMIC": (
        r"\bCOVID[-\s]?19\b|\bpandemic\b"
    ),
    "SUPPLY_CHAIN": (
        r"\bsupply[\s-]?chain\b|"
        r"\bcomponent shortages?\b|"
        r"\bmanufacturing capacity\b"
    ),
    "DEMAND": (
        r"\bcustomer spending\b|"
        r"\bweak(?:er)? demand\b|"
        r"\brecession\b|"
        r"\bdemand uncertainty\b"
    ),
    "LITIGATION": (
        r"\blegal proceedings\b|"
        r"\blitigation\b|"
        r"\blawsuits?\b"
    ),
    "REGULATION": (
        r"\bexport controls?\b|"
        r"\bexport restrictions?\b|"
        r"\bgovernment regulations?\b|"
        r"\bregulatory requirements?\b|"
        r"\btrade sanctions?\b"
    ),
    "CYBERSECURITY": (
        r"\bcyberattacks?\b|"
        r"\bcybersecurity\b|"
        r"\bdata breaches?\b"
    ),
}

FIN_RELATIONS = ("Produce", "Operate_In")

FOCAL_IDS = {
    "AAPL": 9,
    "GOOGL": 11,
    "AMZN": 20,
    "MSFT": 30,
    "NVDA": 221,
}

REF_RELATIONS = (
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
)

FEATURE_GROUPS = {
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

META_FIELDS = [
    "ticker", "as_of", "condition",
    "sec_available", "findkg_available",
    "reflectkg_available",
    "sec_source_present",
    "findkg_source_present",
    "reflectkg_source_present",
    "historical_date_screen_pass",
    "public_availability_verified",
    "provenance_verified",
    "usable_for_formal_finrl",
    "evidence_ids",
    "source_file",
]

AUDIT_FIELDS = [
    "ticker", "as_of", "source",
    "evidence_id", "date", "date_status",
    "classification", "relation",
    "subject", "object",
    "excerpt", "note",
]

def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))

def write_json(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

def write_csv(p, rows, fields):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

def parse_date(v):
    if not v:
        return None
    try:
        return date.fromisoformat(str(v)[:10])
    except (ValueError, TypeError):
        return None

def eligible_date(v, as_of):
    d = parse_date(v)
    cutoff = parse_date(as_of)
    return d is not None and cutoff is not None and d < cutoff

def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def normalize_entity(value):
    return clean_text(value).casefold().rstrip(".")

def sec_features(items, ticker, as_of, audit):
    counts = Counter()
    eligible = []
    rejected = 0

    for e in items:
        eid = str(e.get("evidence_id", ""))
        d = e.get("published_at")
        txt = clean_text(e.get("text", ""))

        if not eligible_date(d, as_of):
            rejected += 1
            audit.append({
                "ticker": ticker, "as_of": as_of,
                "source": "SEC", "evidence_id": eid,
                "date": d, "date_status": "REJECTED",
                "classification": "",
                "relation": "", "subject": "",
                "object": "", "excerpt": "",
                "note": "Missing, future, or same-day date",
            })
            continue

        eligible.append(eid)

        for category, pattern in SEC_RULES.items():
            for match in re.finditer(pattern, txt, re.I):
                counts[category] += 1
                excerpt = txt[
                    max(0, match.start() - 100):
                    min(len(txt), match.end() + 180)
                ]
                audit.append({
                    "ticker": ticker,
                    "as_of": as_of,
                    "source": "SEC",
                    "evidence_id": eid,
                    "date": d,
                    "date_status": "DATASET_DATE_BEFORE_AS_OF",
                    "classification": category,
                    "relation": "", "subject": ticker,
                    "object": "",
                    "excerpt": excerpt,
                    "note": (
                        "Keyword mention only; not a proven "
                        "loss, severity, or realized event"
                    )
                })

    result = {
        "sec_risk_category_count": sum(
            counts[k] > 0 for k in SEC_RULES
        ),
        "sec_mention_count": sum(counts.values()),
    }
    for category in SEC_RULES:
        result["sec_" + category] = int(counts[category] > 0)

    return result, eligible, rejected

def findkg_features(items, ticker, as_of, audit):
    accepted = []
    objects = set()
    counts = Counter()
    rejected = 0

    for e in items:
        eid = str(e.get("evidence_id", ""))
        r = e.get("record") or {}
        observed = e.get("observed_at")
        rel = r.get("relation")
        s = r.get("subject", "")
        o = r.get("object", "")

        ok = (
            eligible_date(observed, as_of)
            and bool(r.get("subject_id") is not None)
            and bool(r.get("object_id") is not None)
            and rel in FIN_RELATIONS
            and r.get("direction") == (
                "subject -> relation -> object"
            )
            and FOCAL_IDS.get(ticker) in (
                int(r["subject_id"]),
                int(r["object_id"])
            )
        )

        if not ok:
            rejected += 1

        audit.append({
            "ticker": ticker, "as_of": as_of,
            "source": "FinDKG",
            "evidence_id": eid,
            "date": observed,
            "date_status": (
                "DATASET_DATE_BEFORE_AS_OF"
                if ok else "REJECTED"
            ),
            "classification": (
                "DESCRIPTIVE_KG_OBSERVATION" if ok else ""
            ),
            "relation": rel, "subject": s, "object": o,
            "excerpt": "",
            "note": (
                "Directed, unverified dataset relation"
                if ok else "Date or relation failed screen"
            ),
        })

        if not ok:
            continue

        accepted.append(eid)
        counts[rel] += 1
        objects.add(normalize_entity(o))

    return {
        "fin_relation_count": len(accepted),
        "fin_produce_count": counts["Produce"],
        "fin_operate_in_count": counts["Operate_In"],
        "fin_distinct_object_count": len(objects),
    }, accepted, rejected

def reflect_features(snapshots, events, ticker, as_of, audit):
    counts = Counter()
    ids = []
    rejected = 0

    # Counts measure observed dataset edges, not active
    # real-world business relationships.
    for e in snapshots:
        eid = str(e.get("evidence_id", ""))
        observed = e.get("last_observed_at")
        rel = e.get("relation")
        ok = (
            eligible_date(observed, as_of)
            and rel in REF_RELATIONS
        )
        if not ok:
            rejected += 1
        else:
            counts[rel] += 1
            ids.append(eid)

        audit.append({
            "ticker": ticker, "as_of": as_of,
            "source": "ReflectKG",
            "evidence_id": eid,
            "date": observed,
            "date_status": (
                "DATASET_DATE_BEFORE_AS_OF"
                if ok else "REJECTED"
            ),
            "classification": (
                "HISTORICAL_RELATION_OBSERVATION" if ok else ""
            ),
            "relation": rel,
            "subject": e.get("subject", ""),
            "object": e.get("object", ""),
            "excerpt": clean_text(e.get("evidence", ""))[:350],
            "note": (
                "Not proof of current business relationship"
            ),
        })

    n_events = 0
    for e in events:
        eid = str(e.get("evidence_id", ""))
        observed = e.get("published_at")
        event_type = str(e.get("event_type", ""))
        ok = eligible_date(observed, as_of)
        if not ok:
            rejected += 1
        else:
            n_events += 1
            ids.append(eid)
            counts["EVENT_" + event_type] += 1

        audit.append({
            "ticker": ticker, "as_of": as_of,
            "source": "ReflectKG",
            "evidence_id": eid,
            "date": observed,
            "date_status": (
                "DATASET_DATE_BEFORE_AS_OF"
                if ok else "REJECTED"
            ),
            "classification": "OBSERVATION_EVENT",
            "relation": e.get("relation", ""),
            "subject": e.get("subject", ""),
            "object": e.get("object", ""),
            "excerpt": clean_text(e.get("evidence", ""))[:350],
            "note": (
                f"{event_type}: observation, "
                "not actual relationship formation"
            ),
        })

    return {
        "ref_relation_count": sum(
            counts[r] for r in REF_RELATIONS
        ),
        "ref_supplier_relation_count": counts["SUPPLIES_TO"],
        "ref_customer_relation_count": counts["CUSTOMER_OF"],
        "ref_competitor_relation_count": counts["COMPETES_WITH"],
        "ref_same_sector_relation_count": counts["SAME_SECTOR"],
        "ref_observation_event_count": n_events,
        "ref_first_observed_count": counts["EVENT_FIRST_OBSERVED"],
        "ref_reasserted_count": counts["EVENT_REASSERTED"],
    }, ids, rejected

def build_one(path):
    data = read_json(path)
    ticker = data["ticker"]
    as_of = data["as_of"]
    if parse_date(as_of) is None:
        raise ValueError(f"Invalid as_of in {path}")

    audit = []
    s, sec_ids, sr = sec_features(
        data.get("sec_evidence", []), ticker, as_of, audit
    )
    f, fin_ids, fr = findkg_features(
        data.get("findkg_evidence", []), ticker, as_of, audit
    )
    r, ref_ids, rr = reflect_features(
        data.get("reflectkg_snapshot", []),
        data.get("reflectkg_changes", []),
        ticker, as_of, audit
    )

    groups = {"SEC": s, "FinDKG": f, "ReflectKG": r}
    evid_ids = {
        "SEC": sec_ids,
        "FinDKG": fin_ids,
        "ReflectKG": ref_ids,
    }

    # 'Source present' describes the submitted dataset,
    # not evidence truth or market information availability.
    source_present = {
        "SEC": bool(data.get("sec_evidence")),
        "FinDKG": bool(data.get("findkg_evidence")),
        "ReflectKG": bool(
            data.get("reflectkg_snapshot")
            or data.get("reflectkg_changes")
        ),
    }

    rejected = sr + fr + rr
    rows = []

    for condition, included in CONDITIONS.items():
        row = {
            "ticker": ticker,
            "as_of": as_of,
            "condition": condition,
            "sec_available": int("SEC" in included),
            "findkg_available": int("FinDKG" in included),
            "reflectkg_available": int("ReflectKG" in included),
            "sec_source_present": int(source_present["SEC"]),
            "findkg_source_present": int(
                source_present["FinDKG"]
            ),
            "reflectkg_source_present": int(
                source_present["ReflectKG"]
            ),
            "historical_date_screen_pass": int(rejected == 0),
            "public_availability_verified": 0,
            "provenance_verified": 0,
            "usable_for_formal_finrl": 0,
            "evidence_ids": "|".join(
                x
                for group in included
                for x in evid_ids[group]
            ),
            "source_file": str(path),
        }

        for group, keys in FEATURE_GROUPS.items():
            for key in keys:
                # Empty CSV cell means feature unavailable
                # because the source is excluded.
                # A numeric zero means the available dataset
                # contains no matching observation.
                row[key] = (
                    groups[group][key]
                    if group in included else None
                )

        rows.append(row)

    return rows, audit, {
        "ticker": ticker,
        "as_of": as_of,
        "input_file": str(path),
        "input_sha256": hashlib.sha256(
            Path(path).read_bytes()
        ).hexdigest(),
        "selected_ids": {
            k: len(v) for k, v in evid_ids.items()
        },
        "rejected_records": rejected,
        "sec_categories": {
            k: v for k, v in s.items()
            if k.startswith("sec_")
        },
        "public_availability_verified": False,
        "provenance_verified": False,
    }

def self_test():
    a = []
    s, ids, rejected = sec_features([{
        "evidence_id": "SEC_001",
        "published_at": "2021-05-26",
        "text": (
            "COVID-19 may affect our supply chain "
            "and customer spending due to restrictions on travel. "
            "See legal proceedings."
        ),
    }], "NVDA", "2021-06-01", a)

    assert ids == ["SEC_001"]
    assert rejected == 0
    assert s["sec_PANDEMIC"] == 1
    assert s["sec_SUPPLY_CHAIN"] == 1
    assert s["sec_DEMAND"] == 1
    assert s["sec_LITIGATION"] == 1
    assert s["sec_REGULATION"] == 0

    s2, ids2, rejected2 = sec_features([{
        "evidence_id": "SEC_FUTURE",
        "published_at": "2021-06-02",
        "text": "COVID-19"
    }], "NVDA", "2021-06-01", [])
    assert not ids2 and rejected2 == 1
    assert s2["sec_PANDEMIC"] == 0

    f, ids3, _ = findkg_features([{
        "evidence_id": "FIN_001",
        "observed_at": "2021-04-25",
        "record": {
            "subject_id": 221, "subject": "Nvidia Corp.",
            "relation": "Produce",
            "direction": "subject -> relation -> object",
            "object_id": 123,
            "object": "Computer Chips"
        }
    }], "NVDA", "2021-06-01", [])
    assert f["fin_relation_count"] == 1
    assert ids3 == ["FIN_001"]

    print("SELF-TEST PASSED")
    print("Travel restrictions do not trigger REGULATION.")
    print("Future observations excluded.")
    print("Focal FinDKG descriptive relation counted.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs", nargs="+",
        default=[
            "outputs/v11/nvda_20210601/clean_evidence.json"
        ]
    )
    ap.add_argument(
        "--out-dir", default="outputs/v13/nvda_20210601"
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    paths = sorted(set(
        p
        for pattern in args.inputs
        for p in glob.glob(pattern, recursive=True)
    ))
    if not paths:
        raise SystemExit("No input files matched")

    rows = []
    audit = []
    reports = []
    seen = set()

    for path in paths:
        xrows, xaudit, report = build_one(path)
        key = (report["ticker"], report["as_of"])

        if key in seen:
            raise ValueError(
                f"Duplicate ticker/date {key}: {path}"
            )
        seen.add(key)

        rows.extend(xrows)
        audit.extend(xaudit)
        reports.append(report)

    out = Path(args.out_dir)
    columns = (
        META_FIELDS
        + [k for keys in FEATURE_GROUPS.values() for k in keys]
    )

    write_csv(out / "features.csv", rows, columns)
    write_csv(out / "evidence_audit.csv", audit, AUDIT_FIELDS)

    schema = {
        "schema_version": "v13",
        "conditions": CONDITIONS,
        "feature_groups": FEATURE_GROUPS,
        "null_policy": (
            "Excluded source = blank. "
            "Included source with zero observed matches = 0. "
            "Missing source file coverage must be checked separately."
        ),
        "date_policy": (
            "Only observations strictly before as_of. "
            "This does not establish public availability."
        ),
        "value_semantics": (
            "Counts of historical dataset observations, "
            "not risk severity, probabilities or investment returns."
        ),
    }

    write_json(out / "feature_schema.json", schema)
    write_json(out / "temporal_quality.json", {
        "snapshots": len(reports),
        "conditions": len(rows),
        "reports": reports,
        "formal_finrl_approval": False,
        "blockers": [
            "Historical public availability unverified",
            "KG source facts not independently verified",
            "SEC keyword categories may require context review",
            "No cross-date leakage audit against market dataset",
            "No model training or out-of-sample backtest",
        ]
    })

    print("\n=== V13 RESULT ===")
    print("Snapshots:", len(reports))
    print("Condition rows:", len(rows))
    print("Audit records:", len(audit))
    print("Output:", out)

    for r in rows:
        print(
            f"{r['ticker']:5s} {r['condition']:18s} "
            f"SEC={str(r['sec_risk_category_count']):>4s} "
            f"FinDKG={str(r['fin_relation_count']):>4s} "
            f"ReflectKG={str(r['ref_relation_count']):>4s}"
        )

if __name__ == "__main__":
    main()
