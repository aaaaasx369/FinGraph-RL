from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--companies",
        default="kg/data/company_master_with_sic.csv",
    )

    parser.add_argument(
        "--text-edges",
        default="kg/data/edges_v4_full.csv",
    )

    parser.add_argument(
        "--sector-output",
        default="kg/data/same_sector_edges.csv",
    )

    parser.add_argument(
        "--merged-output",
        default="kg/data/edges_kg_v1.csv",
    )

    args = parser.parse_args()

    # ========================================================
    # Load company master
    # ========================================================

    companies = pd.read_csv(
        args.companies,
        dtype={
            "cik": str,
            "sic": str,
        },
    )

    for col in [
        "ticker",
        "tickers",
        "company_name",
        "cik",
        "sic",
        "sic_description",
    ]:
        if col not in companies.columns:
            companies[col] = ""

        companies[col] = (
            companies[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    # Only valid SIC
    companies = companies[
        companies["sic"].ne("")
        &
        companies["sic"].str.lower().ne("nan")
    ].copy()

    print(
        f"Companies with SIC: {len(companies)}"
    )

    # ========================================================
    # Build SAME_SECTOR edges
    # ========================================================

    rows = []

    for sic, group in companies.groupby("sic"):

        records = group.to_dict("records")

        # Need at least 2 companies in the sector
        if len(records) < 2:
            continue

        for i in range(len(records)):
            for j in range(len(records)):

                if i == j:
                    continue

                a = records[i]
                b = records[j]

                source = clean_str(a["ticker"])
                target = clean_str(b["ticker"])

                # fall back to company name if no ticker
                if not source:
                    source = clean_str(
                        a["company_name"]
                    )

                if not target:
                    target = clean_str(
                        b["company_name"]
                    )

                if not source or not target:
                    continue

                if source == target:
                    continue

                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "relation": "SAME_SECTOR",

                        "source_name":
                            clean_str(
                                a["company_name"]
                            ),

                        "target_name":
                            clean_str(
                                b["company_name"]
                            ),

                        "source_cik":
                            clean_str(
                                a["cik"]
                            ),

                        "target_cik":
                            clean_str(
                                b["cik"]
                            ),

                        # SAME_SECTOR is structural,
                        # not tied to one filing.
                        "filing_ticker": "",
                        "filing_date": "",

                        "doc_id": "",
                        "chunk_id": "",
                        "form": "",
                        "url": "",

                        "evidence": (
                            f"SEC SIC exact match: "
                            f"{sic} - "
                            f"{clean_str(a['sic_description'])}"
                        ),

                        "confidence": 1.0,
                        "explicit": True,

                        "extraction_method":
                            "SEC_SIC_EXACT_MATCH",

                        "validation_status":
                            "ACCEPT",

                        "sic_source": sic,
                        "sic_target": sic,
                    }
                )

    sector_df = pd.DataFrame(rows)

    # Deduplicate
    if len(sector_df):
        sector_df = (
            sector_df
            .drop_duplicates(
                subset=[
                    "source",
                    "target",
                    "relation",
                    "sic_source",
                ]
            )
            .reset_index(drop=True)
        )

    sector_output = Path(
        args.sector_output
    )

    sector_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sector_df.to_csv(
        sector_output,
        index=False,
    )

    print(
        f"SAME_SECTOR edges: "
        f"{len(sector_df)}"
    )

    # ========================================================
    # Merge with existing text relations
    # ========================================================

    text_df = pd.read_csv(
        args.text_edges
    )

    # Make sure schemas align
    all_columns = sorted(
        set(text_df.columns)
        |
        set(sector_df.columns)
    )

    for col in all_columns:

        if col not in text_df.columns:
            text_df[col] = ""

        if col not in sector_df.columns:
            sector_df[col] = ""

    text_df = text_df[
        all_columns
    ]

    sector_df = sector_df[
        all_columns
    ]

    merged = pd.concat(
        [
            text_df,
            sector_df,
        ],
        ignore_index=True,
    )

    merged_output = Path(
        args.merged_output
    )

    merged.to_csv(
        merged_output,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINANCIAL KG V1")
    print("=" * 70)

    print(
        f"Total edges: {len(merged)}"
    )

    print()
    print("RELATION COUNTS")

    print(
        merged["relation"]
        .value_counts()
        .to_string()
    )

    print()
    print("UNIQUE PAIRS")

    for relation in [
        "COMPETES_WITH",
        "SUPPLIES_TO",
        "CUSTOMER_OF",
        "SAME_SECTOR",
    ]:

        x = merged[
            merged["relation"]
            == relation
        ]

        n = len(
            x[
                [
                    "source",
                    "target",
                ]
            ]
            .drop_duplicates()
        )

        print(
            f"{relation:15s}: {n}"
        )

    print()
    print(
        f"Sector edges -> "
        f"{sector_output}"
    )

    print(
        f"Merged KG    -> "
        f"{merged_output}"
    )


if __name__ == "__main__":
    main()
