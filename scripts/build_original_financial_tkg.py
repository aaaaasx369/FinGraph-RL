#!/usr/bin/env python3
"""
Historical four-relation KG for original Financial-TKG.

Inputs:
  outputs/v35/evidence_ledger.csv

Optional additional grounded sources:
  outputs/finrl_dkg_original/sources/
      historical_relations.csv
      historical_sectors.csv

historical_relations.csv:
  subject,relation,object,available_from,source_id,
  source_url,evidence_quote

historical_sectors.csv:
  company,sector_code,available_from,source_id,source_url

Rules:
  No fabrication.
  No look-ahead backfill.
  SAME_SECTOR requires point-in-time sector evidence.
  CUSTOMER_OF inverse requires SUPPLIES_TO source evidence.
  No future event in earlier training partition.
"""

import argparse
import json
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_original"
SOURCES = BASE / "sources"
DATA = BASE / "dataset" / "SupplyChain"

V35 = ROOT / "outputs/v35/evidence_ledger.csv"

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

TICKERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}

TRAIN_START = pd.Timestamp("2018-01-01")
VAL_START = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2024-01-01")


def stop(message):
    raise RuntimeError("DATA GATE FAILED: " + str(message))


def filehash(path):
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def date_column(df, name):
    if name not in df:
        stop(f"Missing date column: {name}")

    result = pd.to_datetime(
        df[name], errors="coerce"
    )

    if result.isna().any():
        stop(f"Invalid dates in {name}")

    return result.dt.normalize()


def load_v35():
    if not V35.exists():
        stop(f"Missing {V35}")

    raw = pd.read_csv(V35, low_memory=False)

    needed = {
        "subject", "relation",
        "object", "available_from",
    }

    if not needed.issubset(raw):
        stop(
            f"V35 schema missing: {needed-set(raw.columns)}"
        )

    if raw.empty:
        stop("V35 has no evidence")

    data = raw.copy()
    data["available_from"] = date_column(
        data, "available_from"
    )

    records = []

    for index, r in data.iterrows():
        if str(r["relation"]) not in RELATIONS:
            stop("Invalid V35 relation")

        subject = str(r["subject"]).strip()
        obj = str(r["object"]).strip()

        if not subject or not obj or subject == obj:
            continue

        eid = (
            str(r["evidence_id"])
            if "evidence_id" in data
            else f"v35-row-{index}"
        )

        records.append({
            "subject": subject,
            "relation": str(r["relation"]),
            "object": obj,
            "available_from": r["available_from"],
            "source_id": "V35:" + eid,
            "source_url": "",
            "evidence_quote": (
                str(r["evidence_quote"])
                if "evidence_quote" in data
                else ""
            ),
            "provenance": "V35_SEC_SILVER",
            "derived": False,
        })

    return records


def load_external_relations():
    path = SOURCES / "historical_relations.csv"

    if not path.exists():
        return []

    df = pd.read_csv(path).fillna("")

    required = {
        "subject", "relation", "object",
        "available_from", "source_id",
        "source_url", "evidence_quote",
    }

    if not required.issubset(df):
        stop(
            "External relation file missing columns: "
            + str(required - set(df.columns))
        )

    df["available_from"] = date_column(
        df, "available_from"
    )

    records = []

    for r in df.to_dict("records"):
        relation = str(r["relation"]).strip()

        if relation not in RELATIONS:
            stop("Unknown external relation " + relation)

        if not all([
            str(r["subject"]).strip(),
            str(r["object"]).strip(),
            str(r["source_id"]).strip(),
            str(r["source_url"]).strip(),
            str(r["evidence_quote"]).strip(),
        ]):
            stop(
                "External event lacks provenance: "
                + str(r)
            )

        if r["subject"] == r["object"]:
            stop("Self relation")

        records.append({
            "subject": str(r["subject"]).strip(),
            "relation": relation,
            "object": str(r["object"]).strip(),
            "available_from": r["available_from"],
            "source_id": str(r["source_id"]),
            "source_url": str(r["source_url"]),
            "evidence_quote": str(r["evidence_quote"]),
            "provenance": "EXTERNAL_HISTORICAL",
            "derived": False,
        })

    return records


def sector_relations():
    path = SOURCES / "historical_sectors.csv"

    if not path.exists():
        return []

    df = pd.read_csv(
        path, dtype={"sector_code": str}
    ).fillna("")

    required = {
        "company", "sector_code",
        "available_from", "source_id",
        "source_url",
    }

    if not required.issubset(df):
        stop(
            "Sector file missing columns: "
            + str(required - set(df.columns))
        )

    df["available_from"] = date_column(
        df, "available_from"
    )

    if not df[
        ["company", "sector_code", "source_id", "source_url"]
    ].astype(str).apply(
        lambda col: col.str.strip().ne("")
    ).all().all():
        stop("Sector rows lack historical provenance")

    if df.duplicated(
        ["company", "available_from"]
    ).any():
        stop(
            "Multiple sector values for one company-date; "
            "resolve this before using the data."
        )

    # These are historical classification observations,
    # NOT retroactively valid back to 2018.
    # Derive an edge only once both classifications
    # have been observed and are current as of that date.
    dates = sorted(df["available_from"].unique())
    edges = []

    for day in dates:
        current = (
            df[df["available_from"] <= day]
            .sort_values("available_from")
            .drop_duplicates("company", keep="last")
        )

        groups = current.groupby("sector_code")

        for sector, group in groups:
            members = group.to_dict("records")

            for a in members:
                for b in members:
                    if a["company"] >= b["company"]:
                        continue

                    # This relation is first knowable
                    # when BOTH classifications exist.
                    activation = max(
                        a["available_from"],
                        b["available_from"],
                    )

                    if activation != day:
                        continue

                    edge_evidence = (
                        f"Historical sector {sector}; "
                        f"{a['company']}: {a['source_id']}; "
                        f"{b['company']}: {b['source_id']}"
                    )

                    for head, tail in [
                        (a["company"], b["company"]),
                        (b["company"], a["company"]),
                    ]:
                        edges.append({
                            "subject": head,
                            "relation": "SAME_SECTOR",
                            "object": tail,
                            "available_from": day,
                            "source_id": (
                                "SECTOR:"
                                + str(a["source_id"])
                                + ":"
                                + str(b["source_id"])
                            ),
                            "source_url": (
                                str(a["source_url"])
                                + " | "
                                + str(b["source_url"])
                            ),
                            "evidence_quote": edge_evidence,
                            "provenance": (
                                "HISTORICAL_SECTOR_DERIVED"
                            ),
                            "derived": True,
                        })

    return edges


def add_inverse_customer(events):
    result = list(events)

    for e in events:
        if e["relation"] != "SUPPLIES_TO":
            continue

        inverse = dict(e)
        inverse["subject"] = e["object"]
        inverse["object"] = e["subject"]
        inverse["relation"] = "CUSTOMER_OF"
        inverse["provenance"] = (
            "DERIVED_INVERSE_OF_SUPPLIES_TO"
        )
        inverse["derived"] = True

        result.append(inverse)

    return result


def main(args):
    DATA.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)

    records = load_v35()
    records.extend(load_external_relations())
    records.extend(sector_relations())
    records = add_inverse_customer(records)

    df = pd.DataFrame(records)

    if df.empty:
        stop("No graph events")

    df["available_from"] = date_column(
        df, "available_from"
    )

    if (
        (df["available_from"] < TRAIN_START)
        | (df["available_from"] >= TEST_END)
    ).any():
        df = df[
            (df["available_from"] >= TRAIN_START)
            & (df["available_from"] < TEST_END)
        ].copy()

    if df.empty:
        stop("No 2018-2023 events")

    df = df.drop_duplicates(
        [
            "subject", "relation", "object",
            "available_from", "source_id"
        ]
    )

    coverage = df.groupby(
        "relation"
    ).agg(
        mentions=("source_id", "size"),
        unique_edges=("subject", "nunique"),
        event_dates=("available_from", "nunique"),
    )

    present = set(df["relation"])
    missing = set(RELATIONS) - present

    print("\n=== FOUR-RELATION COVERAGE ===")
    print(coverage.to_string())
    print("Missing:", sorted(missing))

    df.sort_values(
        ["available_from", "subject", "relation", "object"],
        inplace=True,
    )

    df.to_csv(
        BASE / "all_grounded_events.csv",
        index=False,
    )

    audit = {
        "total_events": len(df),
        "all_four_present": len(missing) == 0,
        "missing_relations": sorted(missing),
        "relations": {
            name: int((df["relation"] == name).sum())
            for name in RELATIONS
        },
        "v35_sha256": filehash(V35),
        "strict_pit_certified": False,
        "human_gold": False,
    }

    (BASE / "coverage_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    if missing and not args.allow_incomplete:
        stop(
            "Four grounded relation types required: "
            + ", ".join(sorted(missing))
            + ". Coverage audit has been saved. "
            "Add verifiable historical data; "
            "do not manufacture edges."
        )

    entities = sorted(
        set(df["subject"])
        | set(df["object"])
        | set(TICKERS.values())
    )

    entity_ids = {
        e: i for i, e in enumerate(entities)
    }

    relation_ids = {
        r: i for i, r in enumerate(RELATIONS)
    }

    df["head"] = df["subject"].map(entity_ids)
    df["rel"] = df["relation"].map(relation_ids)
    df["tail"] = df["object"].map(entity_ids)

    # Calendar-day integer time index.
    df["time"] = (
        df["available_from"]
        - pd.Timestamp("1970-01-01")
    ).dt.days.astype("int64")

    splits = {
        "train": df[
            df["available_from"] < VAL_START
        ],
        "valid": df[
            (df["available_from"] >= VAL_START)
            & (df["available_from"] < TEST_START)
        ],
        "test": df[
            df["available_from"] >= TEST_START
        ],
    }

    for name, part in splits.items():
        print(
            name,
            "events=", len(part),
            "time_steps=", part["time"].nunique()
        )

        output = part[
            ["head", "rel", "tail", "time"]
        ].copy()

        output["_"] = 1

        output.to_csv(
            DATA / f"{name}.txt",
            sep="\t",
            header=False,
            index=False,
        )

    with (DATA / "stat.txt").open("w") as f:
        f.write(
            f"{len(entities)}\t{len(RELATIONS)}\t0\n"
        )

    pd.DataFrame([
        [e, i, "company", 0]
        for e, i in entity_ids.items()
    ]).to_csv(
        DATA / "entity2id.txt",
        sep="\t",
        header=False,
        index=False,
    )

    pd.DataFrame([
        {"entity": name, "id": idx}
        for name, idx in entity_ids.items()
    ]).to_csv(
        BASE / "entity_mapping.csv",
        index=False,
    )

    df.to_csv(
        BASE / "encoded_events.csv",
        index=False,
    )

    if any(len(p) == 0 for p in splits.values()):
        stop(
            "Original loader requires nonempty "
            "train/valid/test; one split is empty."
        )

    # Useful for original DGL chronological loader:
    # at least two observed event timestamps.
    if df["time"].nunique() < 2:
        stop("Insufficient temporal variation")

    audit["train_events"] = len(splits["train"])
    audit["valid_events"] = len(splits["valid"])
    audit["test_events"] = len(splits["test"])
    audit["entities"] = len(entities)

    (BASE / "coverage_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )

    print("\n=== ORIGINAL KG DATASET CREATED ===")
    print("Dataset:", DATA)
    print("Entities:", len(entities))
    print("Relations:", len(RELATIONS))
    print("Events:", len(df))
    print("Four-relation coverage:", not bool(missing))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Development inspection only; not four-relation ready",
    )
    main(parser.parse_args())
