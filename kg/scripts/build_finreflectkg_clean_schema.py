#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "kg/output/finreflectkg_schema/relation_signatures.csv"
)

OUT_DIR = (
    ROOT
    / "kg/output/finreflectkg_schema"
)


# ============================================================
# Config
# ============================================================

MIN_RELATION_COUNT = 20
MIN_SIGNATURE_COUNT = 3
MAX_SIGNATURES_PER_RELATION = 5


df = pd.read_csv(INPUT)


# ============================================================
# Relation totals
# ============================================================

relation_totals = (
    df.groupby("relationship")["count"]
    .sum()
    .sort_values(ascending=False)
)


valid_relations = set(
    relation_totals[
        relation_totals >= MIN_RELATION_COUNT
    ].index
)


# ============================================================
# Filter signatures
# ============================================================

clean = df[
    df["relationship"].isin(
        valid_relations
    )
    &
    (
        df["count"]
        >= MIN_SIGNATURE_COUNT
    )
].copy()


clean = (
    clean
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
    .groupby(
        "relationship",
        group_keys=False,
    )
    .head(
        MAX_SIGNATURES_PER_RELATION
    )
)


# ============================================================
# Keep only entity types appearing in accepted signatures
# ============================================================

entity_types = sorted(
    set(
        clean["entity_type"]
    )
    |
    set(
        clean["target_type"]
    )
)

relations = sorted(
    clean["relationship"]
    .unique()
)


# ============================================================
# Schema JSON
# ============================================================

schema = {
    "entity_types":
        entity_types,

    "relations":
        relations,
}


schema_path = (
    OUT_DIR
    / "clean_schema.json"
)

with open(
    schema_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        schema,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Signature JSON
# ============================================================

signatures = {}


for relation, g in clean.groupby(
    "relationship"
):

    signatures[
        relation
    ] = [
        {
            "subject_type":
                r["entity_type"],

            "object_type":
                r["target_type"],

            "count":
                int(r["count"]),

            "pct_within_relation":
                float(
                    r[
                        "pct_within_relation"
                    ]
                ),
        }

        for _, r in g.iterrows()
    ]


signature_path = (
    OUT_DIR
    / "clean_relation_signatures.json"
)

with open(
    signature_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        signatures,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# CSV
# ============================================================

clean_path = (
    OUT_DIR
    / "clean_relation_signatures.csv"
)

clean.to_csv(
    clean_path,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 100)
print("CLEAN FINREFLECTKG EMPIRICAL SCHEMA")
print("=" * 100)

print(
    "Relations before:",
    df["relationship"].nunique(),
)

print(
    "Relations after:",
    len(relations),
)

print(
    "Entity types after:",
    len(entity_types),
)

print(
    "Signatures after:",
    len(clean),
)


print("\nTOP RELATIONS")

print(
    relation_totals[
        relation_totals.index.isin(
            relations
        )
    ]
    .head(50)
    .to_string()
)


print("\nCLEAN SIGNATURES")

print(
    clean[
        [
            "entity_type",
            "relationship",
            "target_type",
            "count",
            "pct_within_relation",
        ]
    ]
    .head(100)
    .to_string(index=False)
)


print("\nSaved:")
print(schema_path)
print(signature_path)
print(clean_path)
