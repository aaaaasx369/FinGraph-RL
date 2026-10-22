#!/usr/bin/env python3

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "kg/output/finreflectkg_schema/relation_signatures.csv"
)

OUTPUT = (
    ROOT
    / "kg/configs/finreflect_final_schema.json"
)


# ============================================================
# Final relation policy
# ============================================================

KEEP_RELATIONS = {
    "DISCLOSES",
    "NEGATIVELY_IMPACTS",
    "SUBJECT_TO",
    "IMPACTED_BY",
    "INCREASE",
    "FACE",
    "INTRODUCES",
    "PRODUCE",
    "HAS_STAKE_IN",
    "OPERATES_IN",
    "POSITIVELY_IMPACTS",
    "DECREASE",
    "GUIDES_ON",
    "COMPLIES_WITH",
    "PARTNERS_WITH",
    "COMPETES_WITH",
    "IMPACT",
    "OFFER",
    "ISSUE",
    "SUPPLY",
    "INVOLVED_IN",
    "SERVES_AS",
    "PROVIDE",
    "RECOGNIZES",
    "HEDGE",
    "HOLDS_POSITION",
    "MEMBER_OF",
    "CONTRIBUTES_TO",
    "ESTIMATE",
    "DEVELOPS",

    # important multi-signature relations
    "DEPENDS_ON",
    "INVESTS_IN",
    "ANNOUNCES",
}

DROP_RELATIONS = {
    "US",
    "SERF",
}


# Minimum observations for a specific
# subject_type -> relation -> object_type signature
MIN_SIGNATURE_COUNT = 5


df = pd.read_csv(INPUT).copy()

df["relationship"] = (
    df["relationship"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["entity_type"] = (
    df["entity_type"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["target_type"] = (
    df["target_type"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# Relation filtering
# ============================================================

df = df[
    df["relationship"].isin(
        KEEP_RELATIONS
    )
]

df = df[
    ~df["relationship"].isin(
        DROP_RELATIONS
    )
]


# ============================================================
# Signature filtering
# ============================================================

df = df[
    df["count"] >= MIN_SIGNATURE_COUNT
].copy()


# ============================================================
# Sort strongest signatures first
# ============================================================

df = df.sort_values(
    [
        "relationship",
        "count",
    ],
    ascending=[
        True,
        False,
    ],
)


# ============================================================
# Entity types
# ============================================================

entity_types = sorted(
    set(df["entity_type"])
    |
    set(df["target_type"])
)


# ============================================================
# Build relation dictionary
# ============================================================

relations = {}

for relation, g in df.groupby(
    "relationship"
):

    items = []

    for _, r in g.iterrows():

        items.append(
            {
                "subject_type":
                    r["entity_type"],

                "object_type":
                    r["target_type"],

                "count":
                    int(r["count"]),

                "pct_within_relation":
                    round(
                        float(
                            r[
                                "pct_within_relation"
                            ]
                        ),
                        6,
                    ),
            }
        )

    relations[relation] = items


# ============================================================
# Final schema
# ============================================================

schema = {
    "entity_types":
        entity_types,

    "relations":
        relations,

    "metadata": {
        "source":
            "Derived from released FinReflectKG data",

        "min_signature_count":
            MIN_SIGNATURE_COUNT,

        "note":
            (
                "Empirical schema derived from frequent "
                "FinReflectKG type-relation-type signatures. "
                "Not claimed to be the exact original "
                "FinReflectKG ontology."
            ),
    },
}


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT,
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
# Summary
# ============================================================

print("=" * 100)
print("FINAL FINREFLECT EMPIRICAL SCHEMA")
print("=" * 100)

print(
    "Entity types:",
    len(entity_types),
)

print(
    "Relations:",
    len(relations),
)

print(
    "Signatures:",
    len(df),
)


print("\nRELATIONS")

for relation in sorted(relations):

    print(
        f"\n{relation}"
    )

    for x in relations[relation]:

        print(
            "  "
            f"{x['subject_type']} "
            f"--{relation}--> "
            f"{x['object_type']} "
            f"[n={x['count']}, "
            f"share={x['pct_within_relation']:.3f}]"
        )


print("\nSaved:")
print(OUTPUT)
