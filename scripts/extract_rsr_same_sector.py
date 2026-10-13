#!/usr/bin/env python3
import json
import tarfile
from pathlib import Path
from itertools import combinations

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RSR = ROOT / "references/Temporal_Relational_Stock_Ranking"
OUT = (
    ROOT
    / "outputs/finrl_dkg_original_full/four_relation_kg_v2"
)

OUT.mkdir(parents=True, exist_ok=True)

STOCKS = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"}

archives = [
    RSR / "data/relation.tar.gz",
    RSR / "relation.tar.gz",
]

archive = next(
    (p for p in archives if p.is_file()),
    None,
)

if archive is None:
    raise RuntimeError("RSR relation.tar.gz not found")

rows = []

with tarfile.open(archive, "r:gz") as tar:
    candidates = [
        m for m in tar.getmembers()
        if m.isfile()
        and m.name.endswith("_industry_ticker.json")
        and "sector_industry" in m.name
    ]

    print("RSR industry files:", len(candidates))

    for member in candidates:
        file_obj = tar.extractfile(member)

        if file_obj is None:
            continue

        mapping = json.load(file_obj)

        if not isinstance(mapping, dict):
            continue

        for industry, tickers in mapping.items():
            if not isinstance(tickers, list):
                continue

            names = sorted(set(
                str(t).strip().upper()
                for t in tickers
                if str(t).strip()
            ))

            if len(names) < 2:
                continue

            # Limit to relationships involving
            # one of the five tradable stocks.
            relevant = [
                x for x in names
                if x in STOCKS
            ]

            if not relevant:
                continue

            for head in relevant:
                for tail in names:
                    if head == tail:
                        continue

                    rows.append({
                        "head": head,
                        "relation": "SAME_SECTOR",
                        "tail": tail,
                        "sector_name": industry,
                        "valid_from": "",
                        "source": "RSR_SECTOR_INDUSTRY",
                        "source_file": member.name,
                        "status": "UNDATED_CANDIDATE",
                    })

df = pd.DataFrame(rows, columns=[
    "head",
    "relation",
    "tail",
    "sector_name",
    "valid_from",
    "source",
    "source_file",
    "status",
])

df = df.drop_duplicates([
    "head",
    "relation",
    "tail",
    "sector_name",
])

path = OUT / "same_sector_historical_candidates.csv"

df.to_csv(path, index=False)

print("\n=== RSR SAME_SECTOR EXTRACTION ===")
print("Candidate edges:", len(df))

if len(df):
    print(
        df.groupby(
            ["head", "sector_name"]
        ).size().to_string()
    )

print("\nSAVED:", path)

print(
    "\nPIT STATUS: NOT CERTIFIED. "
    "RSR historical sector classifications "
    "lack verified effective-from dates."
)
