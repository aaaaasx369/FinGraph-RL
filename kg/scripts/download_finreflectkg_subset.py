from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("kg/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_TICKERS = {
    "aapl",
    "msft",
    "nvda",
    "amzn",
    "googl",
}

# 先配合你目前 FinRL-X 時間範圍
START_YEAR = 2018
END_YEAR = 2023

print("Loading FinReflectKG...")

ds = load_dataset(
    "domyn/FinReflectKG",
    split="train",
    streaming=True
)

rows = []

for i, row in enumerate(ds):

    ticker = str(row.get("ticker", "")).lower()
    year = row.get("year", None)

    if ticker not in TARGET_TICKERS:
        continue

    if year is None:
        continue

    try:
        year = int(year)
    except Exception:
        continue

    if not (START_YEAR <= year <= END_YEAR):
        continue

    rows.append({
        "triplet_id": row.get("triplet_id"),
        "entity": row.get("entity"),
        "entity_type": row.get("entity_type"),
        "relationship": row.get("relationship"),
        "target": row.get("target"),
        "target_type": row.get("target_type"),
        "ticker": ticker.upper(),
        "year": year,
        "start_date": row.get("start_date"),
        "end_date": row.get("end_date"),
        "extraction_type": row.get("extraction_type"),
        "source_file": row.get("source_file"),
        "page_id": row.get("page_id"),
        "chunk_id": row.get("chunk_id"),
    })

    if len(rows) % 10000 == 0:
        print(f"Collected {len(rows):,} rows")

df = pd.DataFrame(rows)

print("\n===== SUMMARY =====")
print(df["ticker"].value_counts())
print(df["year"].value_counts().sort_index())
print("Total:", len(df))

out = OUT_DIR / "finreflectkg_5stocks_2018_2023.parquet"
df.to_parquet(out, index=False)

print(f"\nSaved to: {out}")
