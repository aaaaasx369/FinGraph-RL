from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


PORTFOLIO = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]


# ============================================================
# Helpers
# ============================================================

def load_company_embeddings(path: str):
    data = np.load(
        path,
        allow_pickle=True,
    )

    tickers = data[
        "tickers"
    ].astype(str)

    dates = data[
        "dates"
    ].astype(str)

    embeddings = data[
        "embeddings"
    ].astype(np.float32)

    return (
        tickers,
        dates,
        embeddings,
    )


def load_kg_embeddings(path: str):
    data = np.load(
        path,
        allow_pickle=True,
    )

    nodes = data[
        "nodes"
    ].astype(str)

    embeddings = data[
        "embeddings"
    ].astype(np.float32)

    return {
        node: embeddings[i]
        for i, node in enumerate(nodes)
    }


def detect_column(
    df: pd.DataFrame,
    candidates: list[str],
    label: str,
) -> str:

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find {label}. "
        f"Tried: {candidates}"
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build FinRL-ready company text embedding "
            "+ KG embedding features."
        )
    )

    parser.add_argument(
        "--company-embeddings",
        default=(
            "kg/data/"
            "company_embeddings_events.npz"
        ),
    )

    parser.add_argument(
        "--kg-embeddings",
        default=(
            "kg/data/"
            "kg_structural_embeddings.npz"
        ),
    )

    parser.add_argument(
        "--base-features",
        default=(
            "rag/data/features/"
            "sec_rag_features_business_days.csv"
        ),
    )

    parser.add_argument(
        "--text-dim",
        type=int,
        default=16,
    )

    # ========================================================
    # NEW SPLIT
    # Train: 2018-2023
    # Val:   2024
    # Test:  2025
    # ========================================================

    parser.add_argument(
        "--train-start",
        default="2018-01-01",
    )

    parser.add_argument(
        "--train-end",
        default="2023-12-31",
    )

    parser.add_argument(
        "--val-start",
        default="2024-01-01",
    )

    parser.add_argument(
        "--val-end",
        default="2024-12-31",
    )

    parser.add_argument(
        "--test-start",
        default="2025-01-01",
    )

    parser.add_argument(
        "--test-end",
        default="2025-12-31",
    )

    parser.add_argument(
        "--output",
        default=(
            "kg/data/"
            "finrl_text_kg_features_"
            "train2018_2023_"
            "val2024_"
            "test2025.csv"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Parse split dates
    # ========================================================

    train_start = pd.Timestamp(
        args.train_start
    )

    train_end = pd.Timestamp(
        args.train_end
    )

    val_start = pd.Timestamp(
        args.val_start
    )

    val_end = pd.Timestamp(
        args.val_end
    )

    test_start = pd.Timestamp(
        args.test_start
    )

    test_end = pd.Timestamp(
        args.test_end
    )

    print("=" * 70)
    print("EXPERIMENT SPLIT")
    print("=" * 70)

    print(
        f"Train : "
        f"{train_start.date()} "
        f"-> {train_end.date()}"
    )

    print(
        f"Val   : "
        f"{val_start.date()} "
        f"-> {val_end.date()}"
    )

    print(
        f"Test  : "
        f"{test_start.date()} "
        f"-> {test_end.date()}"
    )

    # ========================================================
    # Load base SEC-RAG business-day features
    # ========================================================

    print()
    print(
        "Loading base business-day features..."
    )

    base = pd.read_csv(
        args.base_features
    )

    ticker_col = detect_column(
        base,
        [
            "ticker",
            "tic",
            "symbol",
        ],
        "ticker column",
    )

    date_col = detect_column(
        base,
        [
            "date",
            "datetime",
            "timestamp",
        ],
        "date column",
    )

    base["ticker"] = (
        base[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    base["date"] = pd.to_datetime(
        base[date_col],
        errors="coerce",
    )

    base = base[
        base["date"].notna()
    ].copy()

    base = base[
        base["ticker"].isin(
            PORTFOLIO
        )
    ].copy()

    # Only keep the full experimental horizon.
    base = base[
        (base["date"] >= train_start)
        &
        (base["date"] <= test_end)
    ].copy()

    print(
        f"Base rows: {len(base)}"
    )

    print(
        "Base date range:",
        base["date"].min(),
        "->",
        base["date"].max(),
    )

    # ========================================================
    # Load company SEC text embeddings
    # ========================================================

    print()
    print(
        "Loading company embeddings..."
    )

    (
        tickers,
        dates,
        embeddings,
    ) = load_company_embeddings(
        args.company_embeddings
    )

    print(
        "Company embedding shape:",
        embeddings.shape,
    )

    event = pd.DataFrame(
        {
            "ticker": tickers,
            "date": pd.to_datetime(
                dates,
                errors="coerce",
            ),
            "embedding_index":
                np.arange(
                    len(tickers)
                ),
        }
    )

    event = event[
        event["date"].notna()
    ].copy()

    event["ticker"] = (
        event["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    event = event[
        event["ticker"].isin(
            PORTFOLIO
        )
    ].copy()

    # ========================================================
    # CRITICAL:
    # PCA FIT ONLY ON TRAIN 2018-2023
    # ========================================================

    train_event_mask = (
        (event["date"] >= train_start)
        &
        (event["date"] <= train_end)
    )

    train_idx = event.loc[
        train_event_mask,
        "embedding_index",
    ].to_numpy()

    print()
    print("=" * 70)
    print("PCA FIT")
    print("=" * 70)

    print(
        "Training embedding events:",
        len(train_idx),
    )

    if len(train_idx) == 0:
        raise ValueError(
            "No company embedding events "
            "found in 2018-2023 training period."
        )

    # PCA dimension cannot exceed:
    # - requested dimension
    # - number of training observations
    # - original embedding dimension
    max_dim = min(
        args.text_dim,
        len(train_idx),
        embeddings.shape[1],
    )

    if max_dim < 1:
        raise ValueError(
            "Invalid PCA dimension."
        )

    print(
        "Original dimension:",
        embeddings.shape[1],
    )

    print(
        "Requested PCA dimension:",
        args.text_dim,
    )

    print(
        "Actual PCA dimension:",
        max_dim,
    )

    pca = PCA(
        n_components=max_dim,
        random_state=42,
    )

    # --------------------------------------------------------
    # ONLY training period enters fit()
    # --------------------------------------------------------

    pca.fit(
        embeddings[
            train_idx
        ]
    )

    # --------------------------------------------------------
    # After fit:
    # transform ALL periods using frozen PCA
    # --------------------------------------------------------

    reduced = pca.transform(
        embeddings
    ).astype(np.float32)

    explained = float(
        pca.explained_variance_ratio_
        .sum()
    )

    print(
        f"Explained variance: "
        f"{explained:.6f}"
    )

    # ========================================================
    # Build reduced embedding event table
    # ========================================================

    text_cols = [
        f"text_emb_{i:02d}"
        for i in range(
            reduced.shape[1]
        )
    ]

    event_features = pd.DataFrame(
        reduced,
        columns=text_cols,
    )

    event_features.insert(
        0,
        "date",
        pd.to_datetime(
            dates,
            errors="coerce",
        ),
    )

    event_features.insert(
        0,
        "ticker",
        tickers,
    )

    event_features["ticker"] = (
        event_features["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    event_features = event_features[
        event_features["date"].notna()
    ].copy()

    event_features = event_features[
        event_features["ticker"].isin(
            PORTFOLIO
        )
    ].copy()

    event_features = (
        event_features
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
    )

    # ========================================================
    # Forward-fill latest AVAILABLE SEC embedding
    #
    # For each business day:
    # use only the latest SEC embedding whose publication date
    # is <= current trading date.
    #
    # This prevents future leakage.
    # ========================================================

    output_parts = []

    for ticker in PORTFOLIO:

        daily = (
            base[
                base["ticker"]
                == ticker
            ]
            .sort_values("date")
            .copy()
        )

        events = (
            event_features[
                event_features["ticker"]
                == ticker
            ]
            .drop(
                columns=["ticker"]
            )
            .sort_values("date")
            .copy()
        )

        if len(daily) == 0:
            print(
                f"WARNING: no base rows "
                f"for {ticker}"
            )
            continue

        if len(events) == 0:

            print(
                f"WARNING: no SEC "
                f"embedding events for {ticker}"
            )

            for col in text_cols:
                daily[col] = 0.0

        else:

            daily = pd.merge_asof(
                daily,
                events,
                on="date",
                direction="backward",
                allow_exact_matches=True,
            )

            daily[text_cols] = (
                daily[text_cols]
                .fillna(0.0)
            )

        output_parts.append(
            daily
        )

    if not output_parts:
        raise ValueError(
            "No portfolio rows generated."
        )

    result = pd.concat(
        output_parts,
        ignore_index=True,
    )

    # ========================================================
    # Load static KG structural embedding
    # ========================================================

    print()
    print(
        "Loading KG structural embeddings..."
    )

    kg = load_kg_embeddings(
        args.kg_embeddings
    )

    if not kg:
        raise ValueError(
            "KG embedding file is empty."
        )

    kg_dim = len(
        next(iter(kg.values()))
    )

    kg_cols = [
        f"kg_emb_{i:02d}"
        for i in range(kg_dim)
    ]

    print(
        "KG dimension:",
        kg_dim,
    )

    for col in kg_cols:
        result[col] = 0.0

    for ticker in PORTFOLIO:

        if ticker not in kg:

            print(
                f"WARNING: {ticker} "
                f"not present in KG."
            )

            continue

        mask = (
            result["ticker"]
            == ticker
        )

        result.loc[
            mask,
            kg_cols,
        ] = kg[ticker]

    # ========================================================
    # Add explicit split labels
    # ========================================================

    def assign_split(date):

        if (
            train_start
            <= date
            <= train_end
        ):
            return "train"

        if (
            val_start
            <= date
            <= val_end
        ):
            return "val"

        if (
            test_start
            <= date
            <= test_end
        ):
            return "test"

        return "outside"

    result["split"] = (
        result["date"]
        .apply(assign_split)
    )

    # Remove dates outside requested experiment
    result = result[
        result["split"]
        != "outside"
    ].copy()

    result = (
        result
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # QA
    # ========================================================

    print()
    print("=" * 70)
    print("SPLIT COUNTS")
    print("=" * 70)

    print(
        result["split"]
        .value_counts()
        .to_string()
    )

    print()
    print("DATE RANGE BY SPLIT")

    print(
        result.groupby("split")["date"]
        .agg(
            [
                "min",
                "max",
                "count",
            ]
        )
        .to_string()
    )

    print()
    print("=" * 70)
    print("NONZERO TEXT EMBEDDING")
    print("=" * 70)

    for ticker, g in result.groupby(
        "ticker"
    ):

        nonzero = (
            g[text_cols]
            .abs()
            .sum(axis=1)
            > 0
        ).sum()

        print(
            f"{ticker:5s}: "
            f"{nonzero}/{len(g)}"
        )

    print()
    print("=" * 70)
    print("NONZERO KG EMBEDDING")
    print("=" * 70)

    for ticker, g in result.groupby(
        "ticker"
    ):

        nonzero = (
            g[kg_cols]
            .abs()
            .sum(axis=1)
            > 0
        ).sum()

        print(
            f"{ticker:5s}: "
            f"{nonzero}/{len(g)}"
        )

    # ========================================================
    # Save
    # ========================================================

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINRL EMBEDDING FEATURES")
    print("=" * 70)

    print(
        "Rows:",
        len(result),
    )

    print(
        "Text embedding dimensions:",
        len(text_cols),
    )

    print(
        "KG embedding dimensions:",
        len(kg_cols),
    )

    print(
        "Total new embedding features:",
        len(text_cols)
        +
        len(kg_cols),
    )

    print()
    print(
        f"Saved -> {output}"
    )

    # ========================================================
    # Small preview
    # ========================================================

    preview_cols = (
        [
            "date",
            "ticker",
            "split",
        ]
        +
        text_cols[:3]
        +
        kg_cols[:3]
    )

    print()
    print("PREVIEW")

    print(
        result[
            preview_cols
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
