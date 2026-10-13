#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/finrl_dkg_original_full/four_relation_kg"
V35 = ROOT / "outputs/v35/evidence_ledger.csv"

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "Nvidia Corp.",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
]

# Official five-stock identifiers in the original FinDKG-full.
ORIGINAL_IDS = {
    "AAPL": 9,
    "MSFT": 30,
    "NVDA": 221,
    "AMZN": 20,
    "GOOGL": 11,
}

COLUMNS = [
    "head_ticker",
    "relation",
    "tail_ticker",
    "published_at",
    "available_from",
    "valid_until",
    "source_url",
    "accession",
    "source_sha256",
    "evidence_quote",
    "review_status",
    "source_type",
]

OUT.mkdir(parents=True, exist_ok=True)


def normalize_ticker(value):
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def initialize():
    candidate_path = OUT / "v35_competition_candidates.csv"

    if not V35.exists():
        print("V35 ledger not found:", V35)
        return

    df = pd.read_csv(V35, low_memory=False)

    required = {
        "ticker", "target_ticker", "relation",
        "filing_date", "available_from",
    }
    if not required.issubset(df.columns):
        raise RuntimeError(
            "V35 missing columns: "
            + str(sorted(required - set(df.columns)))
        )

    df = df[
        df["relation"].astype(str)
        .str.upper()
        .eq("COMPETES_WITH")
    ].copy()

    candidates = []

    for _, r in df.iterrows():
        head = normalize_ticker(r["ticker"])
        tail = normalize_ticker(r["target_ticker"])

        if head not in TICKERS or tail not in TICKERS:
            continue
        if head == tail:
            continue

        candidates.append({
            "head_ticker": head,
            "relation": "COMPETES_WITH",
            "tail_ticker": tail,
            "filing_date": r.get("filing_date", ""),
            "available_from_proxy": r.get(
                "available_from", ""
            ),
            "accession": r.get("accession", ""),
            "source_sha256": r.get(
                "source_sha256", ""
            ),
            "evidence_quote": r.get(
                "evidence_quote", ""
            ),
            "human_verified_v35": r.get(
                "human_verified", False
            ),
            "strict_pit_approved_v35": r.get(
                "strict_pit_approved", False
            ),
            "review_status": "NEEDS_REVIEW",
        })

    pd.DataFrame(
        candidates,
        columns=[
            "head_ticker", "relation",
            "tail_ticker", "filing_date",
            "available_from_proxy",
            "accession", "source_sha256",
            "evidence_quote",
            "human_verified_v35",
            "strict_pit_approved_v35",
            "review_status",
        ],
    ).to_csv(candidate_path, index=False)

    verified_path = OUT / "verified_relations.csv"

    if not verified_path.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(
            verified_path, index=False
        )

    print("V35 competition candidates:", len(candidates))
    print("Candidates:", candidate_path)
    print("Verified relation input:", verified_path)
    print(
        "Supplier/customer/sector relations must be "
        "supported by their own historical evidence."
    )


def build():
    path = OUT / "verified_relations.csv"

    if not path.exists():
        raise RuntimeError(
            "Run initialize first."
        )

    df = pd.read_csv(path, dtype=str).fillna("")

    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Missing columns: {sorted(missing)}"
        )

    rejections = []
    accepted = []

    for index, row in df.iterrows():
        r = row.to_dict()
        head = normalize_ticker(r["head_ticker"])
        tail = normalize_ticker(r["tail_ticker"])
        relation = str(r["relation"]).strip().upper()

        reasons = []

        if head not in TICKERS:
            reasons.append("unknown_head")
        if tail not in TICKERS:
            reasons.append("unknown_tail")
        if head == tail:
            reasons.append("self_edge")
        if relation not in RELATIONS:
            reasons.append("unknown_relation")

        if str(r["review_status"]).upper() != "VERIFIED":
            reasons.append("not_verified")

        source_url = str(r["source_url"]).strip()
        accession = str(r["accession"]).strip()
        sha = str(r["source_sha256"]).strip()

        if not (
            source_url.startswith("https://")
            or (accession and sha)
        ):
            reasons.append("missing_traceable_source")

        if len(str(r["evidence_quote"]).strip()) < 15:
            reasons.append("missing_evidence")

        published = pd.to_datetime(
            r["published_at"],
            utc=True,
            errors="coerce",
        )
        available = pd.to_datetime(
            r["available_from"],
            utc=True,
            errors="coerce",
        )
        until = pd.to_datetime(
            r["valid_until"],
            utc=True,
            errors="coerce",
        ) if r["valid_until"] else pd.NaT

        if pd.isna(published):
            reasons.append("missing_published_timestamp")

        if pd.isna(available):
            reasons.append("missing_available_timestamp")

        if (
            pd.notna(published)
            and pd.notna(available)
            and available < published
        ):
            reasons.append("available_before_publication")

        if (
            pd.notna(until)
            and pd.notna(available)
            and until <= available
        ):
            reasons.append("invalid_valid_until")

        if reasons:
            r["reject_reason"] = ";".join(reasons)
            rejections.append(r)
            continue

        r["head_ticker"] = head
        r["tail_ticker"] = tail
        r["relation"] = relation
        r["published_at"] = published.isoformat()
        r["available_from"] = available.isoformat()
        r["valid_until"] = (
            until.isoformat()
            if pd.notna(until) else ""
        )

        accepted.append(r)

    good = pd.DataFrame(accepted, columns=COLUMNS)
    bad = pd.DataFrame(rejections)

    good.to_csv(
        OUT / "accepted_verified_relations.csv",
        index=False,
    )
    bad.to_csv(
        OUT / "rejected_relations.csv",
        index=False,
    )

    print("Accepted verified evidence:", len(good))
    print("Rejected evidence:", len(bad))

    coverage = []

    for relation in RELATIONS:
        n = int(
            good["relation"].eq(relation).sum()
        ) if len(good) else 0
        coverage.append({
            "relation": relation,
            "verified_evidence_rows": n,
            "covered": n > 0,
        })

    coverage_df = pd.DataFrame(coverage)

    coverage_df.to_csv(
        OUT / "four_relation_coverage.csv",
        index=False,
    )

    print("\n=== FOUR RELATIONS COVERAGE ===")
    print(coverage_df.to_string(index=False))

    # Preserve evidence time. Do not move relations backward
    # to a convenient arbitrary date.
    events = []

    for _, r in good.iterrows():
        events.append({
            "head": r["head_ticker"],
            "relation": r["relation"],
            "tail": r["tail_ticker"],
            "event_timestamp": r["available_from"],
            "valid_until": r["valid_until"],
            "source_url": r["source_url"],
            "accession": r["accession"],
            "source_sha256": r["source_sha256"],
        })

    event_df = pd.DataFrame(
        events,
        columns=[
            "head", "relation", "tail",
            "event_timestamp", "valid_until",
            "source_url", "accession", "source_sha256",
        ],
    )

    event_df.to_csv(
        OUT / "four_relation_events.csv",
        index=False,
    )

    # Fixed IDs for ALL four relations, even when
    # some categories currently lack evidence.
    entity_map = {
        ticker: i
        for i, ticker in enumerate(TICKERS)
    }
    relation_map = {
        relation: i
        for i, relation in enumerate(RELATIONS)
    }

    (OUT / "entity2id.txt").write_text(
        "".join(
            f"{ticker}\t{i}\n"
            for ticker, i in entity_map.items()
        )
    )
    (OUT / "relation2id.txt").write_text(
        "".join(
            f"{relation}\t{i}\n"
            for relation, i in relation_map.items()
        )
    )

    # This graph is evidence-time based.
    # Keep event time rather than using
    # FinDKG-full's unrelated weekly time IDs.
    if len(event_df):
        event_df["timestamp"] = pd.to_datetime(
            event_df["event_timestamp"],
            utc=True,
        )

        event_df = event_df.sort_values(
            "timestamp"
        ).copy()

        unique_times = sorted(
            event_df["timestamp"].unique()
        )

        time_ids = {
            t: i for i, t in enumerate(unique_times)
        }

        event_df["time_id"] = (
            event_df["timestamp"].map(time_ids)
        )

        (OUT / "time2id.txt").write_text(
            "".join(
                f"{pd.Timestamp(t).isoformat()}\t{i}\n"
                for t, i in time_ids.items()
            )
        )

        for split, condition in [
            (
                "train",
                event_df["timestamp"].dt.year.le(2021),
            ),
            (
                "valid",
                event_df["timestamp"].dt.year.eq(2022),
            ),
            (
                "test",
                event_df["timestamp"].dt.year.eq(2023),
            ),
        ]:
            selected = event_df.loc[condition]

            with (
                OUT / f"{split}.txt"
            ).open("w") as f:
                for _, r in selected.iterrows():
                    f.write(
                        f"{entity_map[r['head']]}\t"
                        f"{relation_map[r['relation']]}\t"
                        f"{entity_map[r['tail']]}\t"
                        f"{int(r['time_id'])}\t0\n"
                    )
    else:
        (OUT / "time2id.txt").write_text("")
        for split in ["train", "valid", "test"]:
            (OUT / f"{split}.txt").write_text("")

    # Do not incorrectly present this as a trained model.
    manifest = {
        "graph": "FourRelationEvidenceKG",
        "entity_count": len(TICKERS),
        "relation_count": len(RELATIONS),
        "events": len(event_df),
        "relation_types": RELATIONS,
        "original_fintkg_checkpoint_compatible": False,
        "original_checkpoint_retrained": False,
        "strict_pit_approved": False,
        "training_ready": bool(
            len(event_df)
            and event_df["timestamp"].dt.year.le(
                2021
            ).any()
            and coverage_df["covered"].all()
        ),
        "notes": (
            "Four-relation graph uses its own "
            "entity and relation vocabulary. "
            "It cannot reuse the original "
            "15-relation checkpoint directly."
        ),
    }

    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\n=== FOUR-RELATION KG BUILT ===")
    print("Graph events:", len(event_df))
    print("Training ready:", manifest["training_ready"])
    print(
        "Requires new KGTransformer training:",
        True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["initialize", "build"],
    )
    args = parser.parse_args()

    if args.command == "initialize":
        initialize()
    else:
        build()


if __name__ == "__main__":
    main()
