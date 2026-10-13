from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# Helpers
# ============================================================

def normalize_ticker_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "ticker" not in df.columns:
        if "tic" in df.columns:
            df = df.rename(columns={"tic": "ticker"})
        else:
            raise ValueError(
                "No ticker/tic column found. "
                f"Columns: {df.columns.tolist()}"
            )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return df


def detect_date_column(
    df: pd.DataFrame,
    candidates: List[str],
    label: str,
) -> str:

    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find {label} date column. "
        f"Candidates={candidates}, "
        f"columns={df.columns.tolist()}"
    )


def detect_embedding_columns(
    df: pd.DataFrame,
) -> List[str]:

    # --------------------------------------------------------
    # Preferred explicit names
    # --------------------------------------------------------

    patterns = [
        r"^emb_\d+$",
        r"^embedding_\d+$",
        r"^text_emb_\d+$",
        r"^company_emb_\d+$",
        r"^dim_\d+$",
        r"^e\d+$",
    ]

    cols = []

    for c in df.columns:
        for p in patterns:
            if re.match(p, str(c)):
                cols.append(c)
                break

    if cols:
        return sorted(
            cols,
            key=lambda x: int(
                re.findall(r"\d+", x)[-1]
            )
        )

    # --------------------------------------------------------
    # Fall back:
    # numeric columns excluding obvious metadata/scores
    # --------------------------------------------------------

    excluded = {
        "ticker",
        "tic",
        "date",
        "available_date",
        "event_date",
        "published_at",
        "filing_date",
        "year",
        "impact",
        "risk",
        "confidence",
        "retrieval_score_mean",
        "retrieval_score_max",
        "source_chunk_count",
        "source_doc_count",
        "query_count",
        "per_query_k",
        "final_k",
        "retrieved_chunk_count",
        "sec_signal_age_days",
    }

    numeric = [
        c
        for c in df.columns
        if c not in excluded
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    if len(numeric) < 8:
        raise ValueError(
            "Could not confidently detect embedding columns.\n"
            f"Numeric candidates: {numeric}\n"
            f"All columns: {df.columns.tolist()}"
        )

    print(
        "WARNING: embedding columns detected "
        "using numeric fallback:"
    )
    print(numeric)

    return numeric


def assign_split(
    dates: pd.Series,
) -> pd.Series:

    years = dates.dt.year

    split = pd.Series(
        "other",
        index=dates.index,
        dtype="object",
    )

    split[
        (years >= 2018)
        &
        (years <= 2023)
    ] = "train"

    split[
        years == 2024
    ] = "val"

    split[
        years == 2025
    ] = "test"

    split[
        years == 2026
    ] = "holdout"

    return split


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--business-days",
        default=(
            "rag/data/features/"
            "sec_rag_features_business_days_"
            "multiquery_2018_2026.csv"
        ),
    )

    parser.add_argument(
        "--company-events",
        default=(
            "kg/data/"
            "company_embedding_events.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "kg/data/"
            "m1_finrl_features_2018_2026.csv"
        ),
    )

    parser.add_argument(
        "--pca-model-output",
        default=(
            "kg/data/models/"
            "company_text_pca_2018_2023.joblib"
        ),
    )

    parser.add_argument(
        "--scaler-output",
        default=(
            "kg/data/models/"
            "company_text_scaler_2018_2023.joblib"
        ),
    )

    parser.add_argument(
        "--pca-components",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--train-start",
        default="2018-01-01",
    )

    parser.add_argument(
        "--train-end",
        default="2023-12-31",
    )

    args = parser.parse_args()

    # ========================================================
    # Paths
    # ========================================================

    business_path = Path(
        args.business_days
    )

    company_path = Path(
        args.company_events
    )

    output_path = Path(
        args.output
    )

    pca_path = Path(
        args.pca_model_output
    )

    scaler_path = Path(
        args.scaler_output
    )

    if not business_path.exists():
        raise FileNotFoundError(
            business_path
        )

    if not company_path.exists():
        raise FileNotFoundError(
            company_path
        )

    # ========================================================
    # Load business-day SEC-RAG backbone
    # ========================================================

    print("=" * 72)
    print("LOAD BUSINESS-DAY FEATURES")
    print("=" * 72)

    daily = pd.read_csv(
        business_path
    )

    daily = normalize_ticker_column(
        daily
    )

    daily_date_col = detect_date_column(
        daily,
        [
            "date",
            "trade_date",
            "datetime",
        ],
        "business-day",
    )

    if daily_date_col != "date":
        daily = daily.rename(
            columns={
                daily_date_col:
                    "date"
            }
        )

    daily["date"] = pd.to_datetime(
        daily["date"],
        errors="coerce",
    )

    daily = daily.dropna(
        subset=[
            "ticker",
            "date",
        ]
    ).copy()

    daily = (
        daily
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print(
        "rows:",
        len(daily),
    )

    print(
        "range:",
        daily["date"].min(),
        "->",
        daily["date"].max(),
    )

    print()

    print(
        daily.groupby(
            "ticker"
        ).size()
    )

    # ========================================================
    # Load company embedding events
    # ========================================================

    print()
    print("=" * 72)
    print("LOAD COMPANY EMBEDDING EVENTS")
    print("=" * 72)

    company = pd.read_csv(
        company_path
    )

    company = normalize_ticker_column(
        company
    )

    company_date_col = detect_date_column(
        company,
        [
            "available_date",
            "event_date",
            "published_at",
            "filing_date",
            "date",
        ],
        "company embedding event",
    )

    if (
        company_date_col
        !=
        "available_date"
    ):
        company = company.rename(
            columns={
                company_date_col:
                    "available_date"
            }
        )

    company[
        "available_date"
    ] = pd.to_datetime(
        company[
            "available_date"
        ],
        errors="coerce",
    )

    company = company.dropna(
        subset=[
            "ticker",
            "available_date",
        ]
    ).copy()

    embedding_cols = (
        detect_embedding_columns(
            company
        )
    )

    print(
        "company event rows:",
        len(company),
    )

    print(
        "embedding dimensions:",
        len(
            embedding_cols
        ),
    )

    print(
        "range:",
        company[
            "available_date"
        ].min(),
        "->",
        company[
            "available_date"
        ].max(),
    )

    print()

    print(
        company.groupby(
            "ticker"
        ).size()
    )

    # ========================================================
    # Check duplicate event rows
    # ========================================================

    duplicate_mask = (
        company.duplicated(
            subset=[
                "ticker",
                "available_date",
            ],
            keep=False,
        )
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    print()
    print(
        "duplicate ticker/date rows:",
        duplicate_count,
    )

    if duplicate_count > 0:

        # Average duplicated event embeddings.
        metadata_cols = [
            "ticker",
            "available_date",
        ]

        company = (
            company[
                metadata_cols
                +
                embedding_cols
            ]
            .groupby(
                metadata_cols,
                as_index=False,
            )[
                embedding_cols
            ]
            .mean()
        )

        print(
            "after averaging duplicates:",
            len(company),
        )

    # ========================================================
    # PCA FIT DATA
    #
    # IMPORTANT:
    # only 2018-2023 company embedding events
    # ========================================================

    print()
    print("=" * 72)
    print("FIT TRAIN-ONLY PCA")
    print("=" * 72)

    train_start = pd.Timestamp(
        args.train_start
    )

    train_end = pd.Timestamp(
        args.train_end
    )

    company_train = company[
        (
            company[
                "available_date"
            ]
            >= train_start
        )
        &
        (
            company[
                "available_date"
            ]
            <= train_end
        )
    ].copy()

    if len(company_train) == 0:
        raise ValueError(
            "No company embedding events "
            "inside train period."
        )

    X_train = (
        company_train[
            embedding_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # Replace nonfinite values
    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    print(
        "PCA fit rows:",
        X_train.shape[0],
    )

    print(
        "raw embedding dims:",
        X_train.shape[1],
    )

    # ========================================================
    # Standardize using TRAIN ONLY
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    # ========================================================
    # PCA using TRAIN ONLY
    # ========================================================

    max_components = min(
        args.pca_components,
        X_train_scaled.shape[0],
        X_train_scaled.shape[1],
    )

    if (
        max_components
        <
        args.pca_components
    ):

        print(
            "WARNING: requested PCA dims",
            args.pca_components,
            "but only",
            max_components,
            "are possible."
        )

    pca = PCA(
        n_components=
            max_components,
        random_state=42,
    )

    pca.fit(
        X_train_scaled
    )

    print(
        "PCA components:",
        pca.n_components_,
    )

    print(
        "Explained variance ratio sum:",
        float(
            pca
            .explained_variance_ratio_
            .sum()
        ),
    )

    # Save fitted train-only transformers
    pca_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scaler_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        pca,
        pca_path,
    )

    joblib.dump(
        scaler,
        scaler_path,
    )

    print(
        "Saved PCA ->",
        pca_path,
    )

    print(
        "Saved scaler ->",
        scaler_path,
    )

    # ========================================================
    # Transform ALL company embedding events
    #
    # No refit after 2023.
    # ========================================================

    X_all = (
        company[
            embedding_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    X_all = np.nan_to_num(
        X_all,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_all_scaled = (
        scaler.transform(
            X_all
        )
    )

    X_pca = (
        pca.transform(
            X_all_scaled
        )
        .astype(
            np.float32
        )
    )

    pca_cols = [
        f"text_emb_{i:02d}"
        for i in range(
            X_pca.shape[1]
        )
    ]

    company_pca = (
        company[
            [
                "ticker",
                "available_date",
            ]
        ]
        .copy()
    )

    for i, col in enumerate(
        pca_cols
    ):

        company_pca[
            col
        ] = X_pca[
            :,
            i
        ]

    # ========================================================
    # AS-OF JOIN TO BUSINESS DAYS
    #
    # latest available SEC/company embedding
    # <= current business date
    #
    # NO FUTURE FILL.
    # ========================================================

    print()
    print("=" * 72)
    print("AS-OF JOIN COMPANY TEXT EMBEDDINGS")
    print("=" * 72)

    pieces = []

    for ticker in sorted(
        daily[
            "ticker"
        ].unique()
    ):

        d = (
            daily[
                daily[
                    "ticker"
                ] == ticker
            ]
            .copy()
            .sort_values(
                "date"
            )
            .reset_index(
                drop=True
            )
        )

        e = (
            company_pca[
                company_pca[
                    "ticker"
                ] == ticker
            ]
            .copy()
            .sort_values(
                "available_date"
            )
            .reset_index(
                drop=True
            )
        )

        if len(e) == 0:

            for c in pca_cols:
                d[c] = 0.0

            d[
                "text_embedding_available_date"
            ] = pd.NaT

            pieces.append(
                d
            )

            continue

        e = e.drop(
            columns=[
                "ticker"
            ]
        )

        merged = pd.merge_asof(
            d,
            e,
            left_on=
                "date",
            right_on=
                "available_date",
            direction=
                "backward",
            allow_exact_matches=
                True,
        )

        merged = merged.rename(
            columns={
                "available_date":
                    "text_embedding_available_date"
            }
        )

        # Before first event:
        # zero vector, not future-filled.
        for c in pca_cols:

            merged[c] = (
                merged[c]
                .fillna(0.0)
                .astype(
                    np.float32
                )
            )

        pieces.append(
            merged
        )

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    out = (
        out
        .sort_values(
            [
                "date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # Temporal leakage check
    # ========================================================

    has_embedding = out[
        "text_embedding_available_date"
    ].notna()

    leakage = out[
        has_embedding
        &
        (
            out[
                "text_embedding_available_date"
            ]
            >
            out[
                "date"
            ]
        )
    ]

    if len(leakage) > 0:

        print(
            leakage[
                [
                    "ticker",
                    "date",
                    "text_embedding_available_date",
                ]
            ].head(
                20
            )
        )

        raise AssertionError(
            "TEXT EMBEDDING TEMPORAL "
            f"LEAKAGE: {len(leakage)} rows"
        )

    # ========================================================
    # Split
    # ========================================================

    out[
        "split"
    ] = assign_split(
        out[
            "date"
        ]
    )

    # ========================================================
    # Text embedding age
    # ========================================================

    out[
        "text_embedding_age_days"
    ] = (
        out[
            "date"
        ]
        -
        out[
            "text_embedding_available_date"
        ]
    ).dt.days

    # ========================================================
    # QA: zero vector rows
    # ========================================================

    emb_matrix = (
        out[
            pca_cols
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    row_norm = np.linalg.norm(
        emb_matrix,
        axis=1,
    )

    out[
        "text_embedding_nonzero"
    ] = (
        row_norm > 1e-12
    ).astype(int)

    # ========================================================
    # Save
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.to_csv(
        output_path,
        index=False,
    )

    # ========================================================
    # Final QA
    # ========================================================

    print()
    print("=" * 72)
    print("M1 FEATURE BUILD COMPLETE")
    print("=" * 72)

    print(
        "ROWS:",
        len(out),
    )

    print(
        "DATE RANGE:",
        out[
            "date"
        ].min(),
        "->",
        out[
            "date"
        ].max(),
    )

    print()

    print("SPLITS")

    print(
        out.groupby(
            "split"
        ).size()
    )

    print()

    print("ROWS BY TICKER")

    print(
        out.groupby(
            "ticker"
        ).size()
    )

    print()

    print(
        "TEXT PCA DIMS:",
        len(
            pca_cols
        ),
    )

    print(
        "PCA EXPLAINED VARIANCE:",
        float(
            pca
            .explained_variance_ratio_
            .sum()
        ),
    )

    print()

    print(
        "TEMPORAL LEAKAGE ROWS:",
        len(
            leakage
        ),
    )

    print()

    print(
        "NONZERO TEXT EMBEDDING "
        "ROWS BY TICKER"
    )

    print(
        out.groupby(
            "ticker"
        )[
            "text_embedding_nonzero"
        ].agg(
            [
                "sum",
                "count",
            ]
        )
    )

    print()

    print(
        "FIRST/LAST TEXT SIGNAL "
        "BY TICKER"
    )

    real = out[
        out[
            "text_embedding_available_date"
        ].notna()
    ]

    print(
        real.groupby(
            "ticker"
        )[
            "text_embedding_available_date"
        ].agg(
            [
                "min",
                "max",
            ]
        )
    )

    print()

    sec_cols = [
        c
        for c in [
            "impact",
            "risk",
            "confidence",
        ]
        if c in out.columns
    ]

    if sec_cols:

        print(
            "SEC-RAG SCORE RANGE"
        )

        print(
            out[
                sec_cols
            ].agg(
                [
                    "min",
                    "max",
                ]
            )
        )

        print()

    print(
        "OUTPUT:",
        output_path,
    )

    print(
        "PCA MODEL:",
        pca_path,
    )

    print(
        "SCALER:",
        scaler_path,
    )


if __name__ == "__main__":
    main()
