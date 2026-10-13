from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(
    "rag/data/features/"
    "sec_rag_features_business_days_multiquery_2018_2026.csv"
)

WEEKLY = Path(
    "kg/data/findkg/"
    "findkg_weekly_dynamic_embeddings_2018_2022.csv"
)

OUT = Path(
    "kg/data/findkg/"
    "findkg_finrl_daily_2018_2022.csv"
)

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]


def main():

    print("=" * 72)
    print("LOAD BASE FEATURES")
    print("=" * 72)

    base = pd.read_csv(BASE)

    base["date"] = pd.to_datetime(
        base["date"],
        errors="coerce",
    )

    base["ticker"] = (
        base["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    base = base[
        base["ticker"].isin(TICKERS)
    ].copy()

    print("base rows:", len(base))
    print(
        "base date:",
        base["date"].min(),
        "->",
        base["date"].max(),
    )

    print()
    print("=" * 72)
    print("LOAD WEEKLY FINDKG EMBEDDINGS")
    print("=" * 72)

    weekly = pd.read_csv(WEEKLY)

    weekly["date"] = pd.to_datetime(
        weekly["date"],
        errors="coerce",
    )

    weekly["ticker"] = (
        weekly["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    emb_cols = sorted(
        c
        for c in weekly.columns
        if c.startswith("findkg_emb_")
    )

    if len(emb_cols) != 200:
        raise RuntimeError(
            f"Expected 200 dimensions, got {len(emb_cols)}"
        )

    print("weekly rows:", len(weekly))
    print("dims:", len(emb_cols))
    print(
        "weekly date:",
        weekly["date"].min(),
        "->",
        weekly["date"].max(),
    )

    # ========================================================
    # Keep FinDKG experiment period only
    # ========================================================

    base = base[
        (base["date"] >= "2018-01-01")
        &
        (base["date"] <= "2022-12-31")
    ].copy()

    pieces = []

    # ========================================================
    # Strict backward as-of merge by ticker
    #
    # FinDKG snapshots are weekly (mostly Sunday).
    # A trading day only receives the latest already-existing
    # graph state.
    # ========================================================

    for ticker in TICKERS:

        left = (
            base[
                base["ticker"] == ticker
            ]
            .sort_values("date")
            .copy()
        )

        right = (
            weekly[
                weekly["ticker"] == ticker
            ][
                [
                    "date",
                    "ticker",
                    "time_id",
                    "split",
                    "entity_id",
                ]
                + emb_cols
            ]
            .sort_values("date")
            .copy()
        )

        right = right.rename(
            columns={
                "date": "findkg_available_date",
                "split": "findkg_weekly_split",
            }
        )

        merged = pd.merge_asof(
            left,
            right,
            left_on="date",
            right_on="findkg_available_date",
            by="ticker",
            direction="backward",
            allow_exact_matches=False,
        )

        pieces.append(
            merged
        )

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    out = out.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
    )

    # ========================================================
    # No future filling.
    # Drop trading days before first available KG state.
    # ========================================================

    missing_rows = (
        out[
            emb_cols
        ]
        .isna()
        .any(axis=1)
    )

    print()
    print(
        "rows before first available KG state:",
        int(missing_rows.sum()),
    )

    out = out[
        ~missing_rows
    ].copy()

    # ========================================================
    # Experiment split
    # ========================================================

    out["split"] = "other"

    out.loc[
        (
            out["date"] >= "2018-01-01"
        )
        &
        (
            out["date"] <= "2021-12-31"
        ),
        "split"
    ] = "train"

    out.loc[
        (
            out["date"] >= "2022-01-01"
        )
        &
        (
            out["date"] <= "2022-06-30"
        ),
        "split"
    ] = "val"

    out.loc[
        (
            out["date"] >= "2022-07-01"
        )
        &
        (
            out["date"] <= "2022-12-31"
        ),
        "split"
    ] = "test"

    out = out[
        out["split"] != "other"
    ].copy()

    # ========================================================
    # Temporal QA
    # ========================================================

    out["findkg_age_days"] = (
        out["date"]
        -
        out["findkg_available_date"]
    ).dt.days

    leakage = out[
        out["findkg_available_date"]
        >=
        out["date"]
    ]

    if len(leakage) != 0:
        raise RuntimeError(
            f"Temporal leakage found: {len(leakage)} rows"
        )

    # ========================================================
    # Embedding QA
    # ========================================================

    X = out[
        emb_cols
    ].to_numpy(
        dtype=np.float32
    )

    out["findkg_embedding_norm"] = (
        np.linalg.norm(
            X,
            axis=1,
        )
    )

    print()
    print("=" * 72)
    print("DAILY FINDKG QA")
    print("=" * 72)

    print(
        "rows:",
        len(out)
    )

    print(
        "dims:",
        len(emb_cols)
    )

    print(
        "date:",
        out["date"].min(),
        "->",
        out["date"].max(),
    )

    print()
    print("split:")
    print(
        out.groupby("split").size()
    )

    print()
    print("ticker:")
    print(
        out.groupby("ticker").size()
    )

    print()
    print(
        "missing embedding values:",
        int(
            out[
                emb_cols
            ]
            .isna()
            .sum()
            .sum()
        )
    )

    print(
        "leakage rows:",
        len(leakage)
    )

    print()
    print("KG age days:")
    print(
        out[
            "findkg_age_days"
        ].describe()
    )

    print()
    print(
        "duplicate date/ticker:",
        int(
            out.duplicated(
                [
                    "date",
                    "ticker",
                ]
            ).sum()
        )
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        OUT,
        index=False,
    )

    print()
    print("OUTPUT:", OUT)


if __name__ == "__main__":
    main()
