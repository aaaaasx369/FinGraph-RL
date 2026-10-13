#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

BASE = ROOT / "references/Financial-TKG/FinDKG_dataset"

DATASETS = [
    "FinDKG",
    "FinDKG-full",
    "SupplyChain",
]

ALIASES = {
    "AAPL": [
        "Apple Inc", "Apple", "Apple Incorporated",
    ],
    "MSFT": [
        "Microsoft", "Microsoft Corporation",
    ],
    "NVDA": [
        "NVIDIA", "Nvidia Corporation",
    ],
    "AMZN": [
        "Amazon", "Amazon.com",
    ],
    "GOOGL": [
        "Alphabet", "Google", "Google LLC",
    ],
}


def load_table(path):
    if not path.exists():
        return None

    return pd.read_csv(
        path, sep="\t", header=None,
        dtype=str, on_bad_lines="skip",
    ).fillna("")


def print_relations(folder):
    path = folder / "relation2id.txt"

    print("\n=== RELATION DICTIONARY ===")

    rel = load_table(path)

    if rel is None:
        print("No relation2id.txt")
        return None

    print(rel.to_string(index=False, header=False))
    return rel


def print_entities(folder):
    path = folder / "entity2id.txt"

    print("\n=== FIVE-STOCK ENTITY MATCHES ===")

    entities = load_table(path)

    if entities is None:
        print("No entity2id.txt")
        return None

    for ticker, names in ALIASES.items():
        matches = []

        for _, row in entities.iterrows():
            entity = str(row.iloc[0])

            if any(
                re.search(
                    r"(?<![A-Za-z])"
                    + re.escape(alias)
                    + r"(?![A-Za-z])",
                    entity,
                    re.I,
                )
                for alias in names
            ):
                matches.append(
                    row.tolist()
                )

        print("\n", ticker)

        for match in matches[:20]:
            print(" ", match)

        print("Total:", len(matches))

    return entities


def print_times(folder):
    path = folder / "time2id.txt"

    print("\n=== TIME MAPPING ===")

    if not path.exists():
        print("No time2id.txt")
        return

    try:
        df = pd.read_csv(path)

        print("Columns:", df.columns.tolist())
        print("First:")
        print(df.head(5).to_string(index=False))
        print("Last:")
        print(df.tail(5).to_string(index=False))
    except Exception as exc:
        print("Error:", exc)


def print_events(folder, rel):
    print("\n=== EVENT STATISTICS ===")

    for name in [
        "train.txt",
        "valid.txt",
        "test.txt",
    ]:
        path = folder / name

        if not path.exists():
            print(name, "MISSING")
            continue

        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
        )

        if df.shape[1] < 4:
            print(name, "INVALID FORMAT")
            continue

        print("\n", name)
        print("Events:", len(df))
        print("Distinct heads:", df[0].nunique())
        print("Distinct tails:", df[2].nunique())
        print("Relations:", df[1].nunique())
        print("Time range:", df[3].min(), "to", df[3].max())
        print("Unique time steps:", df[3].nunique())

        print("\nRelation frequency:")
        print(
            df[1].value_counts()
            .sort_index()
            .to_string()
        )


for dataset in DATASETS:
    folder = BASE / dataset

    print("\n" + "=" * 70)
    print("DATASET:", dataset)
    print("=" * 70)

    if not folder.exists():
        print("NOT FOUND")
        continue

    relations = print_relations(folder)
    entities = print_entities(folder)
    print_times(folder)
    print_events(folder, relations)

print("\n=== ORIGINAL FINDKG DATA AUDIT FINISHED ===")
