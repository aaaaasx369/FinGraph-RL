from __future__ import annotations

from pathlib import Path

import pandas as pd


REFLECTKG_FEATURES = [
    "kg_edge_count",
    "kg_unique_relation_count",
    "kg_unique_object_count",
    "kg_unique_subject_count",
    "kg_confidence_mean",

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

    "kg_negative_count",
    "kg_positive_count",
    "kg_dependency_count",
    "kg_regulatory_count",
    "kg_competition_count",
    "kg_disclosure_count",

    "kg_negative_ratio",
    "kg_positive_ratio",
    "kg_regulatory_ratio",
]


def load_reflectkg(
    path: str | Path,
) -> pd.DataFrame:

    df = pd.read_csv(path)

    df["published_at"] = pd.to_datetime(
        df["published_at"]
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
    )

    return df


def attach_reflectkg_strict(
    market: pd.DataFrame,
    kg: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "tic",
) -> pd.DataFrame:

    left = market.copy()

    left[date_col] = pd.to_datetime(
        left[date_col]
    )

    left[ticker_col] = (
        left[ticker_col]
        .astype(str)
        .str.upper()
    )

    pieces = []

    for ticker, m in left.groupby(
        ticker_col,
        sort=False,
    ):

        m = (
            m.sort_values(date_col)
            .copy()
        )

        k = (
            kg[
                kg["ticker"] == ticker
            ]
            .sort_values("published_at")
            .copy()
        )

        if k.empty:

            for c in REFLECTKG_FEATURES:
                m[c] = 0.0

            m[
                "reflectkg_source_date"
            ] = pd.NaT

            pieces.append(m)

            continue


        k = k[
            [
                "published_at",
                *REFLECTKG_FEATURES,
            ]
        ].copy()


        merged = pd.merge_asof(
            m,
            k,
            left_on=date_col,
            right_on="published_at",
            direction="backward",

            # CRITICAL anti-leakage rule:
            # same filing date cannot be used.
            allow_exact_matches=False,
        )


        merged = merged.rename(
            columns={
                "published_at":
                    "reflectkg_source_date"
            }
        )


        for c in REFLECTKG_FEATURES:
            merged[c] = (
                pd.to_numeric(
                    merged[c],
                    errors="coerce",
                )
                .fillna(0.0)
            )


        leakage = merged[
            merged[
                "reflectkg_source_date"
            ].notna()
            &
            (
                merged[
                    "reflectkg_source_date"
                ]
                >=
                merged[
                    date_col
                ]
            )
        ]

        if len(leakage) != 0:
            raise RuntimeError(
                "ReflectKG temporal leakage detected."
            )


        pieces.append(
            merged
        )


    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    return (
        out
        .sort_values(
            [
                date_col,
                ticker_col,
            ]
        )
        .reset_index(drop=True)
    )
