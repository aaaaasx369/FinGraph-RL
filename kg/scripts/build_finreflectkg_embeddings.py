import pandas as pd
import numpy as np
from pathlib import Path
import re

INPUT = Path("kg/data/finreflectkg_5stocks_2018_2023.parquet")
OUTPUT = Path("kg/output/finreflectkg_relation_embeddings.csv")
RELATION_LIST_OUTPUT = Path("kg/output/finreflectkg_top_relations.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

TOP_K = 50

print("Loading:", INPUT)

df = pd.read_parquet(INPUT)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# -----------------------------
# 1. Clean basic fields
# -----------------------------
df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

df["relationship"] = (
    df["relationship"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.strip()
)

df["entity"] = df["entity"].fillna("").astype(str).str.strip()
df["target"] = df["target"].fillna("").astype(str).str.strip()

# -----------------------------
# 2. Make safe feature names
# -----------------------------
def safe_name(x):
    x = str(x).lower().strip()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


# -----------------------------
# 3. Find top relations
# -----------------------------
relation_counts = df["relationship"].value_counts()

top_relations = relation_counts.head(TOP_K).index.tolist()

print("\n===== TOP RELATIONS =====")

for i, rel in enumerate(top_relations, 1):
    print(f"{i:02d}. {rel}: {relation_counts[rel]:,}")

pd.DataFrame({
    "relationship": top_relations,
    "count": [relation_counts[x] for x in top_relations]
}).to_csv(RELATION_LIST_OUTPUT, index=False)


# -----------------------------
# 4. Build ticker-year embedding
# -----------------------------
rows = []

for (ticker, year), g in df.groupby(["ticker", "year"]):

    relation_freq = g["relationship"].value_counts()

    total = len(g)

    row = {
        "ticker": ticker,
        "year": int(year),

        # graph size / diversity features
        "kg_total_triples": total,
        "kg_unique_entities": g["entity"].nunique(),
        "kg_unique_targets": g["target"].nunique(),
        "kg_unique_relations": g["relationship"].nunique(),
    }

    # normalized relation frequency
    for rel in top_relations:

        feature_name = f"kg_rel_{safe_name(rel)}"

        count = relation_freq.get(rel, 0)

        row[feature_name] = (
            float(count) / float(total)
            if total > 0
            else 0.0
        )

    rows.append(row)


emb = pd.DataFrame(rows)

emb = emb.sort_values(
    ["ticker", "year"]
).reset_index(drop=True)


# -----------------------------
# 5. log transform raw count features
# -----------------------------
for col in [
    "kg_total_triples",
    "kg_unique_entities",
    "kg_unique_targets",
    "kg_unique_relations",
]:
    emb[f"{col}_log1p"] = np.log1p(emb[col])


# -----------------------------
# 6. Basic validation
# -----------------------------
print("\n===== EMBEDDING SUMMARY =====")

print("Shape:", emb.shape)

print("\nTicker/year coverage:")
print(
    emb.groupby("ticker")["year"]
    .agg(["min", "max", "count"])
)

print("\nMissing values:")
print(emb.isna().sum().sum())

print("\nFirst rows:")
print(emb.head())


# -----------------------------
# 7. Save
# -----------------------------
emb.to_csv(OUTPUT, index=False)

print("\nSaved embedding:")
print(OUTPUT)

print("\nSaved relation list:")
print(RELATION_LIST_OUTPUT)

print("\nDone.")
