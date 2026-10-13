#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "kg/output/finreflect_reimplementation/"
    "temporal_kg_features_full.csv"
)

OUTPUT = (
    ROOT
    / "kg/output/finreflect_reimplementation/"
    "reflectkg_compact_features.csv"
)


KEEP = [
    # structure
    "kg_edge_count",
    "kg_unique_relation_count",
    "kg_unique_object_count",
    "kg_unique_subject_count",
    "kg_confidence_mean",

    # important relations
    "kg_rel_DISCLOSES",
    "kg_rel_SUBJECT_TO",
    "kg_rel_OPERATES_IN",
    "kg_rel_HAS_STAKE_IN",
    "kg_rel_PRODUCE",
    "kg_rel_INVOLVED_IN",
    "kg_rel_HEDGE",
    "kg_rel_COMPETES_WITH",
    "kg_rel_IMPACTED_BY",
    "kg_rel_DEPENDS_ON",
    "kg_rel_INVESTS_IN",
    "kg_rel_PARTNERS_WITH",
    "kg_rel_NEGATIVELY_IMPACTS",

    # grouped semantic features
    "kg_negative_count",
    "kg_positive_count",
    "kg_dependency_count",
    "kg_regulatory_count",
    "kg_competition_count",
    "kg_disclosure_count",

    # ratios
    "kg_negative_ratio",
    "kg_positive_ratio",
    "kg_regulatory_ratio",
]


df = pd.read_csv(INPUT)

required = [
    "published_at",
    "ticker",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing keys: {missing}"
    )


# Some relation columns may legitimately not exist.
# Create them as zero instead of crashing.
for c in KEEP:
    if c not in df.columns:
        print(
            "Missing ReflectKG feature -> fill 0:",
            c,
        )
        df[c] = 0.0


out = df[
    [
        "published_at",
        "ticker",
        *KEEP,
    ]
].copy()


out["published_at"] = pd.to_datetime(
    out["published_at"]
)

out["ticker"] = (
    out["ticker"]
    .astype(str)
    .str.upper()
)


for c in KEEP:
    out[c] = pd.to_numeric(
        out[c],
        errors="coerce",
    ).fillna(0.0)


out = (
    out
    .sort_values(
        [
            "ticker",
            "published_at",
        ]
    )
    .drop_duplicates(
        [
            "ticker",
            "published_at",
        ],
        keep="last",
    )
    .reset_index(
        drop=True
    )
)


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUTPUT,
    index=False,
)


print("=" * 90)
print("REFLECTKG COMPACT FEATURES")
print("=" * 90)

print("rows:", len(out))
print("feature dims:", len(KEEP))

print(
    "tickers:",
    sorted(
        out["ticker"].unique()
    ),
)

print(
    "date range:",
    out["published_at"].min(),
    "->",
    out["published_at"].max(),
)

print("\nfeatures:")
for x in KEEP:
    print(" ", x)

print("\nSaved:")
print(OUTPUT)
