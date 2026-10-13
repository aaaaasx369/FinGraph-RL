from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


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

def detect_column(df, candidates, name):
    for c in candidates:
        if c in df.columns:
            return c

    raise ValueError(
        f"Could not find {name}. Tried: {candidates}"
    )


def load_sec_metadata(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            x = json.loads(line)

            rows.append({
                "embedding_index": i,
                "ticker": str(
                    x.get("ticker", "")
                ).upper().strip(),
                "date": (
                    x.get("published_at")
                    or x.get("filing_date")
                    or x.get("date")
                ),
                "form": x.get("form", ""),
                "chunk_id": x.get("chunk_id", ""),
                "doc_id": x.get("doc_id", ""),
            })

    df = pd.DataFrame(rows)

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df = df[
        df["date"].notna()
    ].copy()

    return df


def build_daily_sec_features(
    calendar_df,
    metadata_df,
    embeddings,
):
    """
    Build simple leakage-safe daily SEC features from available filings.

    For each ticker/date:
    - latest_sec_age_days
    - sec_chunks_today
    - sec_chunks_30d
    - sec_activity_30d
    - latest SEC embedding norm
    """

    output = []

    for ticker in PORTFOLIO:

        daily = (
            calendar_df[
                calendar_df["ticker"] == ticker
            ]
            .sort_values("date")
            .copy()
        )

        events = (
            metadata_df[
                metadata_df["ticker"] == ticker
            ]
            .sort_values("date")
            .copy()
        )

        if len(daily) == 0:
            continue

        if len(events) == 0:
            daily["latest_sec_date"] = pd.NaT
            daily["latest_sec_age_days"] = np.nan
            daily["sec_chunks_today"] = 0
            daily["sec_chunks_30d"] = 0
            daily["sec_activity_30d"] = 0.0
            daily["latest_sec_embedding_norm"] = 0.0

            output.append(daily)
            continue

        # ----------------------------------------------------
        # Daily chunk counts
        # ----------------------------------------------------

        chunk_counts = (
            events.groupby("date")
            .size()
            .rename("sec_chunks_today")
            .reset_index()
        )

        daily = daily.merge(
            chunk_counts,
            on="date",
            how="left",
        )

        daily["sec_chunks_today"] = (
            daily["sec_chunks_today"]
            .fillna(0)
            .astype(int)
        )

        # ----------------------------------------------------
        # Rolling 30-day filing/chunk activity
        # ----------------------------------------------------

        temp = (
            daily[
                ["date", "sec_chunks_today"]
            ]
            .set_index("date")
            .sort_index()
        )

        rolling = (
            temp["sec_chunks_today"]
            .rolling("30D")
            .sum()
        )

        daily["sec_chunks_30d"] = (
            rolling.values
        )

        # Squashed activity feature
        daily["sec_activity_30d"] = np.log1p(
            daily["sec_chunks_30d"]
        )

        # ----------------------------------------------------
        # Latest filing/chunk date available as of that day
        # ----------------------------------------------------

        event_dates = (
            events[
                ["date"]
            ]
            .drop_duplicates()
            .sort_values("date")
            .rename(
                columns={
                    "date": "latest_sec_date"
                }
            )
        )

        latest = pd.merge_asof(
            daily[
                ["date"]
            ].sort_values("date"),
            event_dates,
            left_on="date",
            right_on="latest_sec_date",
            direction="backward",
            allow_exact_matches=True,
        )

        daily["latest_sec_date"] = (
            latest["latest_sec_date"].values
        )

        daily["latest_sec_age_days"] = (
            daily["date"]
            - daily["latest_sec_date"]
        ).dt.days

        # ----------------------------------------------------
        # Latest available SEC embedding norm
        # ----------------------------------------------------

        event_embedding_rows = []

        for event_date, g in events.groupby("date"):

            idx = g["embedding_index"].to_numpy()

            x = embeddings[idx]

            # mean embedding for that event date
            z = x.mean(axis=0)

            event_embedding_rows.append({
                "date": event_date,
                "latest_sec_embedding_norm":
                    float(np.linalg.norm(z)),
            })

        event_emb = pd.DataFrame(
            event_embedding_rows
        ).sort_values("date")

        merged_norm = pd.merge_asof(
            daily[
                ["date"]
            ].sort_values("date"),
            event_emb,
            on="date",
            direction="backward",
            allow_exact_matches=True,
        )

        daily[
            "latest_sec_embedding_norm"
        ] = (
            merged_norm[
                "latest_sec_embedding_norm"
            ]
            .fillna(0.0)
            .values
        )

        output.append(daily)

    return pd.concat(
        output,
        ignore_index=True,
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--market-data",
        required=True,
        help=(
            "CSV containing at least date + ticker/tic/symbol "
            "for business-day calendar."
        ),
    )

    parser.add_argument(
        "--sec-metadata",
        default=(
            "rag/data/index/"
            "sec_metadata.jsonl"
        ),
    )

    parser.add_argument(
        "--sec-embeddings",
        default=(
            "rag/data/index/"
            "sec_embeddings.npy"
        ),
    )

    parser.add_argument(
        "--start-date",
        default="2018-01-01",
    )

    parser.add_argument(
        "--end-date",
        default="2026-09-29",
    )

    parser.add_argument(
        "--output",
        default=(
            "rag/data/features/"
            "sec_rag_features_business_days_2018_2026.csv"
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Load market calendar
    # ========================================================

    market = pd.read_csv(
        args.market_data
    )

    ticker_col = detect_column(
        market,
        [
            "ticker",
            "tic",
            "symbol",
        ],
        "ticker column",
    )

    date_col = detect_column(
        market,
        [
            "date",
            "datetime",
            "timestamp",
        ],
        "date column",
    )

    market["ticker"] = (
        market[ticker_col]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    market["date"] = pd.to_datetime(
        market[date_col],
        errors="coerce",
    )

    market = market[
        market["date"].notna()
    ].copy()

    market = market[
        market["ticker"].isin(
            PORTFOLIO
        )
    ].copy()

    market = market[
        (market["date"] >= args.start_date)
        &
        (market["date"] <= args.end_date)
    ].copy()

    calendar = (
        market[
            ["date", "ticker"]
        ]
        .drop_duplicates()
        .sort_values(
            ["ticker", "date"]
        )
    )

    print("=" * 70)
    print("MARKET CALENDAR")
    print("=" * 70)

    print(
        "Rows:",
        len(calendar),
    )

    print(
        "Range:",
        calendar["date"].min(),
        "->",
        calendar["date"].max(),
    )

    print()
    print(
        calendar.groupby(
            "ticker"
        )["date"]
        .agg(
            [
                "min",
                "max",
                "count",
            ]
        )
    )

    # ========================================================
    # Load SEC index
    # ========================================================

    metadata = load_sec_metadata(
        args.sec_metadata
    )

    embeddings = np.load(
        args.sec_embeddings
    )

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"SEC metadata/embedding mismatch: "
            f"{len(metadata)} vs {embeddings.shape[0]}"
        )

    print()
    print("=" * 70)
    print("SEC INDEX")
    print("=" * 70)

    print(
        "Metadata:",
        len(metadata),
    )

    print(
        "Embeddings:",
        embeddings.shape,
    )

    print(
        "Range:",
        metadata["date"].min(),
        "->",
        metadata["date"].max(),
    )

    # ========================================================
    # Build daily SEC features
    # ========================================================

    result = build_daily_sec_features(
        calendar,
        metadata,
        embeddings,
    )

    # ========================================================
    # Split labels
    # ========================================================

    def split_fn(d):

        if (
            pd.Timestamp("2018-01-01")
            <= d
            <= pd.Timestamp("2023-12-31")
        ):
            return "train"

        if (
            pd.Timestamp("2024-01-01")
            <= d
            <= pd.Timestamp("2024-12-31")
        ):
            return "val"

        if (
            pd.Timestamp("2025-01-01")
            <= d
            <= pd.Timestamp("2025-12-31")
        ):
            return "test"

        if (
            pd.Timestamp("2026-01-01")
            <= d
            <= pd.Timestamp("2026-12-31")
        ):
            return "holdout"

        return "outside"

    result["split"] = (
        result["date"]
        .apply(split_fn)
    )

    result = result[
        result["split"] != "outside"
    ].copy()

    result = result.sort_values(
        [
            "date",
            "ticker",
        ]
    ).reset_index(
        drop=True
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
    print("SEC BUSINESS-DAY FEATURES")
    print("=" * 70)

    print(
        "Rows:",
        len(result),
    )

    print(
        "Range:",
        result["date"].min(),
        "->",
        result["date"].max(),
    )

    print()
    print("SPLITS")

    print(
        result.groupby(
            "split"
        )["date"]
        .agg(
            [
                "min",
                "max",
                "count",
            ]
        )
    )

    print()
    print("BY TICKER")

    print(
        result.groupby(
            "ticker"
        )["date"]
        .agg(
            [
                "min",
                "max",
                "count",
            ]
        )
    )

    print()
    print(
        f"Saved -> {output}"
    )


if __name__ == "__main__":
    main()
