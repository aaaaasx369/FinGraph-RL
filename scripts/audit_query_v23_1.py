#!/usr/bin/env python3
"""V23.1 as-of relation query and structural audit."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tier", default="conservative",
        choices=["conservative", "inclusive"]
    )
    p.add_argument(
        "--ticker", default="NVDA"
    )
    p.add_argument(
        "--as-of", default="2021-06-01"
    )
    p.add_argument(
        "--verified-text-only",
        action="store_true"
    )
    p.add_argument(
        "--top-k", type=int, default=15
    )
    args = p.parse_args()

    path = (
        ROOT / "outputs/v23_1" / args.tier
        / "normalized_events.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    df["event_date"] = pd.to_datetime(
        df["event_date"], errors="raise"
    ).dt.normalize()

    cutoff = pd.Timestamp(
        args.as_of
    ).normalize()

    assert df["event_id"].is_unique, (
        "Duplicate original event IDs"
    )

    eligible = df[
        df["typed_company_relation"]
    ]

    invalid_types = eligible[
        ~eligible["source_type"].eq("COMPANY")
        | ~eligible["target_type"].eq("COMPANY")
    ]

    assert invalid_types.empty, (
        "Invalid company-to-company entity types"
    )

    invalid_relations = eligible[
        ~eligible["relation_normalized"].isin([
            "SUPPLIES_TO",
            "CUSTOMER_OF",
            "COMPETES_WITH",
            "SAME_SECTOR",
            "PARTNERS_WITH",
        ])
    ]

    assert invalid_relations.empty, (
        "Invalid company relation type"
    )

    # Query strictly prior to selected date.
    g = df[
        df["ticker"].eq(args.ticker)
        & df["event_date"].lt(cutoff)
        & df["typed_company_relation"]
    ].copy()

    if args.verified_text_only:
        # This flag means text-support candidate only,
        # NOT independently verified ground truth.
        g = g[
            g["quality_gate_pass"]
        ].copy()

    g = (
        g.sort_values(
            ["event_date", "event_id"]
        )
        .drop_duplicates(
            "semantic_edge_id",
            keep="last"
        )
    )

    # Candidate-first ranking, then more recent evidence.
    g["support_rank"] = (
        g["evidence_status"]
        .eq("TEXT_SUPPORT_CANDIDATE")
        .astype(int)
    )

    g = g.sort_values(
        ["support_rank", "event_date"],
        ascending=[False, False]
    )

    print("\n=== V23.1 QUALITY AUDIT ===")
    print("Tier:", args.tier)
    print("Total events:", len(df))
    print("Company relation events:", len(eligible))
    print(
        "Text-supported candidate events:",
        int(df["quality_gate_pass"].sum())
    )
    print("Typed relation gate: PASS")
    print("Strict PIT approved: NO")

    print("\n=== HISTORICAL COMPANY RELATIONS ===")
    print("Ticker:", args.ticker)
    print("As-of:", args.as_of)
    print("Matching semantic edges:", len(g))

    for _, row in g.head(args.top_k).iterrows():
        print(
            f"{row['source_company']}"
            f" -- {row['relation_normalized']} --> "
            f"{row['target_company']}"
        )
        print(
            "Source:", row["source_file"],
            "| Date:", row["event_date"].date(),
            "| Evidence:", row["evidence_status"]
        )

    if g.empty:
        print(
            "ABSTAIN: No eligible historical "
            "company-to-company relations."
        )

if __name__ == "__main__":
    main()
