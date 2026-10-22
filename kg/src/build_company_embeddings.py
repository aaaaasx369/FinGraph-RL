from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings",
        default="rag/data/index/sec_embeddings.npy",
    )

    parser.add_argument(
        "--metadata",
        default="rag/data/index/sec_metadata.jsonl",
    )

    parser.add_argument(
        "--output",
        default="kg/data/company_embeddings_events.npz",
    )

    parser.add_argument(
        "--csv-output",
        default="kg/data/company_embedding_events.csv",
    )

    args = parser.parse_args()

    print("Loading SEC embeddings...")
    emb = np.load(args.embeddings)

    print("Embedding shape:", emb.shape)

    metadata = []

    with open(args.metadata, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))

    print("Metadata rows:", len(metadata))

    if len(metadata) != len(emb):
        raise ValueError(
            f"Embedding/metadata mismatch: "
            f"{len(emb)} vs {len(metadata)}"
        )

    rows = []

    for i, m in enumerate(metadata):
        ticker = str(
            m.get("ticker", "")
        ).upper().strip()

        date = (
            m.get("published_at")
            or m.get("filing_date")
            or m.get("date")
            or ""
        )

        form = str(
            m.get("form", "")
        ).strip()

        if not ticker or not date:
            continue

        rows.append(
            {
                "embedding_index": i,
                "ticker": ticker,
                "date": str(date)[:10],
                "form": form,
            }
        )

    meta = pd.DataFrame(rows)

    print()
    print("Ticker counts:")
    print(
        meta["ticker"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Aggregate all chunks available on same ticker/date
    # ========================================================

    event_dates = []
    event_tickers = []
    event_embeddings = []
    event_counts = []

    grouped = meta.groupby(
        ["ticker", "date"],
        sort=True,
    )

    for (ticker, date), group in grouped:

        idx = group["embedding_index"].to_numpy()

        x = emb[idx]

        # normalize chunks first
        x = l2_normalize(x)

        # mean pool
        company_emb = x.mean(axis=0)

        # normalize aggregated embedding
        company_emb = l2_normalize(
            company_emb.reshape(1, -1)
        )[0]

        event_tickers.append(ticker)
        event_dates.append(date)
        event_embeddings.append(company_emb)
        event_counts.append(len(idx))

    event_embeddings = np.asarray(
        event_embeddings,
        dtype=np.float32,
    )

    event_tickers = np.asarray(event_tickers)
    event_dates = np.asarray(event_dates)
    event_counts = np.asarray(event_counts)

    output = Path(args.output)
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,
        tickers=event_tickers,
        dates=event_dates,
        embeddings=event_embeddings,
        chunk_counts=event_counts,
    )

    summary = pd.DataFrame(
        {
            "ticker": event_tickers,
            "date": event_dates,
            "chunk_count": event_counts,
        }
    )

    summary.to_csv(
        args.csv_output,
        index=False,
    )

    print()
    print("=" * 70)
    print("COMPANY EMBEDDINGS")
    print("=" * 70)

    print(
        "Events:",
        len(event_embeddings),
    )

    print(
        "Dimension:",
        event_embeddings.shape[1],
    )

    print()
    print(
        summary.groupby("ticker")
        .size()
        .to_string()
    )

    print()
    print("Saved:")
    print(output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
