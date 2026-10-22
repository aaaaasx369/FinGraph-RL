#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = (
    ROOT
    / "kg/output/finreflect_reimplementation/single_pass.jsonl"
)

DEFAULT_OUTPUT = (
    ROOT
    / "kg/output/finreflect_reimplementation/temporal_kg_features.csv"
)

INPUT = Path(
    os.getenv(
        "KG_EDGE_FILE",
        str(DEFAULT_INPUT),
    )
)

OUTPUT = Path(
    os.getenv(
        "KG_FEATURE_FILE",
        str(DEFAULT_OUTPUT),
    )
)


# ============================================================
# Helpers
# ============================================================

def clean_relation_name(x: str) -> str:
    x = str(x).strip().upper()
    x = re.sub(
        r"[^A-Z0-9]+",
        "_",
        x,
    )
    return x.strip("_")


def load_edges(path: Path) -> pd.DataFrame:

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            rows.append(x)

    if not rows:
        raise ValueError(
            f"No edges found in {path}"
        )

    df = pd.DataFrame(rows)

    required = [
        "ticker",
        "published_at",
        "relation",
        "subject",
        "object",
    ]

    missing = [
        x
        for x in required
        if x not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ============================================================
# Load
# ============================================================

df = load_edges(INPUT)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["published_at"] = pd.to_datetime(
    df["published_at"],
    errors="coerce",
)

df = df[
    df["published_at"].notna()
].copy()

df["relation"] = (
    df["relation"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df["subject_type"] = (
    df.get(
        "subject_type",
        "",
    )
    .astype(str)
    .str.upper()
    .str.strip()
)

df["object_type"] = (
    df.get(
        "object_type",
        "",
    )
    .astype(str)
    .str.upper()
    .str.strip()
)

df["confidence"] = pd.to_numeric(
    df.get(
        "confidence",
        1.0,
    ),
    errors="coerce",
).fillna(1.0)


# ============================================================
# 1. Relation-count features
# ============================================================

relation_counts = (
    df.groupby(
        [
            "published_at",
            "ticker",
            "relation",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

relation_counts.columns = [
    "kg_rel_"
    + clean_relation_name(x)
    for x in relation_counts.columns
]

relation_counts = (
    relation_counts
    .reset_index()
)


# ============================================================
# 2. Generic graph statistics
# ============================================================

base = (
    df.groupby(
        [
            "published_at",
            "ticker",
        ]
    )
    .agg(
        kg_edge_count=(
            "relation",
            "size",
        ),

        kg_unique_relation_count=(
            "relation",
            "nunique",
        ),

        kg_unique_object_count=(
            "object",
            "nunique",
        ),

        kg_unique_subject_count=(
            "subject",
            "nunique",
        ),

        kg_confidence_mean=(
            "confidence",
            "mean",
        ),

        kg_confidence_min=(
            "confidence",
            "min",
        ),
    )
    .reset_index()
)


# ============================================================
# 3. Financial-semantic feature groups
# ============================================================

NEGATIVE_RELATIONS = {
    "NEGATIVELY_IMPACTS",
    "IMPACTED_BY",
    "FACE",
}

POSITIVE_RELATIONS = {
    "POSITIVELY_IMPACTS",
    "PARTNERS_WITH",
    "INVESTS_IN",
}

DEPENDENCY_RELATIONS = {
    "DEPENDS_ON",
    "SUPPLY",
}

REGULATORY_RELATIONS = {
    "SUBJECT_TO",
    "COMPLIES_WITH",
    "REGULATES",
}

COMPETITION_RELATIONS = {
    "COMPETES_WITH",
}

DISCLOSURE_RELATIONS = {
    "DISCLOSES",
    "RECOGNIZES",
    "GUIDES_ON",
}


def group_count(
    frame: pd.DataFrame,
    relations: set[str],
    name: str,
) -> pd.DataFrame:

    x = (
        frame[
            frame["relation"].isin(
                relations
            )
        ]
        .groupby(
            [
                "published_at",
                "ticker",
            ]
        )
        .size()
        .rename(name)
        .reset_index()
    )

    return x


groups = [
    (
        NEGATIVE_RELATIONS,
        "kg_negative_count",
    ),
    (
        POSITIVE_RELATIONS,
        "kg_positive_count",
    ),
    (
        DEPENDENCY_RELATIONS,
        "kg_dependency_count",
    ),
    (
        REGULATORY_RELATIONS,
        "kg_regulatory_count",
    ),
    (
        COMPETITION_RELATIONS,
        "kg_competition_count",
    ),
    (
        DISCLOSURE_RELATIONS,
        "kg_disclosure_count",
    ),
]


features = base.merge(
    relation_counts,
    on=[
        "published_at",
        "ticker",
    ],
    how="left",
)


for relations, name in groups:

    x = group_count(
        df,
        relations,
        name,
    )

    features = features.merge(
        x,
        on=[
            "published_at",
            "ticker",
        ],
        how="left",
    )


count_cols = [
    c
    for c in features.columns
    if c.startswith(
        "kg_"
    )
    and c not in {
        "kg_confidence_mean",
        "kg_confidence_min",
    }
]

features[count_cols] = (
    features[count_cols]
    .fillna(0)
)


# ============================================================
# 4. Ratios
# ============================================================

denom = (
    features["kg_edge_count"]
    .clip(lower=1)
)

features[
    "kg_negative_ratio"
] = (
    features["kg_negative_count"]
    / denom
)

features[
    "kg_positive_ratio"
] = (
    features["kg_positive_count"]
    / denom
)

features[
    "kg_regulatory_ratio"
] = (
    features["kg_regulatory_count"]
    / denom
)


# ============================================================
# 5. Sort / save
# ============================================================

features = (
    features
    .sort_values(
        [
            "published_at",
            "ticker",
        ]
    )
    .reset_index(
        drop=True
    )
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

features.to_csv(
    OUTPUT,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print("=" * 110)
print("TEMPORAL KG FEATURES")
print("=" * 110)

print("Input:")
print(INPUT)

print(
    "\nEdges:",
    len(df),
)

print(
    "Ticker-date rows:",
    len(features),
)

print(
    "Tickers:",
    sorted(
        features[
            "ticker"
        ].unique()
    ),
)

print(
    "Date range:",
    features[
        "published_at"
    ].min(),
    "->",
    features[
        "published_at"
    ].max(),
)

print(
    "\nFeature columns:",
    len(
        [
            c
            for c in features.columns
            if c.startswith("kg_")
        ]
    ),
)

print("\nFEATURE LIST")

for c in features.columns:
    if c.startswith("kg_"):
        print(" ", c)


print("\nSAMPLE")

print(
    features
    .head(10)
    .to_string(
        index=False
    )
)


print("\nSaved:")
print(OUTPUT)
