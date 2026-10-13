#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "kg/data/finreflectkg_5stocks_2018_2023.parquet"
)

OUT_DIR = (
    ROOT
    / "kg/output/finreflectkg_schema"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


df = pd.read_parquet(INPUT).copy()


required = [
    "entity_type",
    "relationship",
    "target_type",
]

missing = [
    x for x in required
    if x not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


for col in required:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
    )


# ============================================================
# 1. Full relation signature counts
# ============================================================

sig = (
    df.groupby(
        [
            "entity_type",
            "relationship",
            "target_type",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .sort_values(
        [
            "relationship",
            "count",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


# ============================================================
# 2. Conditional frequency within each relation
# ============================================================

relation_total = (
    sig.groupby(
        "relationship"
    )["count"]
    .transform("sum")
)

sig[
    "pct_within_relation"
] = (
    sig["count"]
    / relation_total
)


# ============================================================
# 3. Conditional frequency for
#    (entity_type, relationship)
# ============================================================

source_relation_total = (
    sig.groupby(
        [
            "entity_type",
            "relationship",
        ]
    )["count"]
    .transform("sum")
)

sig[
    "pct_within_source_relation"
] = (
    sig["count"]
    / source_relation_total
)


# ============================================================
# Save CSV
# ============================================================

csv_path = (
    OUT_DIR
    / "relation_signatures.csv"
)

sig.to_csv(
    csv_path,
    index=False,
)


# ============================================================
# 4. Entity schema
# ============================================================

entity_types = sorted(
    set(
        df["entity_type"]
    )
    |
    set(
        df["target_type"]
    )
)

relations = sorted(
    df["relationship"]
    .unique()
    .tolist()
)


schema = {
    "entity_types":
        entity_types,

    "relations":
        relations,
}


schema_path = (
    OUT_DIR
    / "derived_schema.json"
)

with open(
    schema_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        schema,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 5. Build signature JSON
# ============================================================

signature_json = {}


for relation, group in (
    sig.groupby(
        "relationship"
    )
):

    records = []

    for _, row in (
        group
        .sort_values(
            "count",
            ascending=False,
        )
        .iterrows()
    ):

        records.append(
            {
                "subject_type":
                    row[
                        "entity_type"
                    ],

                "object_type":
                    row[
                        "target_type"
                    ],

                "count":
                    int(
                        row[
                            "count"
                        ]
                    ),

                "pct_within_relation":
                    float(
                        row[
                            "pct_within_relation"
                        ]
                    ),

                "pct_within_source_relation":
                    float(
                        row[
                            "pct_within_source_relation"
                        ]
                    ),
            }
        )

    signature_json[
        relation
    ] = records


json_path = (
    OUT_DIR
    / "relation_signatures.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        signature_json,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# 6. Print useful summary
# ============================================================

print("=" * 110)
print("FINREFLECTKG DERIVED RELATION SIGNATURES")
print("=" * 110)

print(
    "Input rows:",
    len(df),
)

print(
    "Entity types:",
    len(entity_types),
)

print(
    "Relations:",
    len(relations),
)

print(
    "Unique signatures:",
    len(sig),
)


print("\nENTITY TYPES")
print(
    ", ".join(
        entity_types
    )
)


print("\nTOP 50 SIGNATURES")
print(
    sig
    .sort_values(
        "count",
        ascending=False,
    )
    .head(50)
    .to_string(
        index=False,
    )
)


print("\nTOP SIGNATURE PER RELATION")

top_per_rel = (
    sig
    .sort_values(
        "count",
        ascending=False,
    )
    .groupby(
        "relationship",
        as_index=False,
    )
    .first()
    .sort_values(
        "count",
        ascending=False,
    )
)

print(
    top_per_rel[
        [
            "relationship",
            "entity_type",
            "target_type",
            "count",
            "pct_within_relation",
        ]
    ]
    .to_string(
        index=False,
    )
)


print("\nSaved:")
print(csv_path)
print(schema_path)
print(json_path)
