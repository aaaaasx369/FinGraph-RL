import pandas as pd
import numpy as np
from pathlib import Path

KG = Path(
    "outputs/unified_temporal_kg_v3/"
    "unified_temporal_edges.csv"
)

EMB = Path(
    "outputs/aegis_kgt_daily_2018_2023/"
    "daily_dynamic_kgt.csv"
)

OUT = Path(
    "results/aegis_rl_v1/"
    "event_embedding_alignment"
)
OUT.mkdir(parents=True, exist_ok=True)

kg = pd.read_csv(KG)
emb = pd.read_csv(EMB)

sec = kg[
    kg["relation"] == "HAS_SEC_SIGNAL"
].copy()

sec["ticker"] = sec["head"].str.replace(
    "COMPANY:", "", regex=False
)
sec["event_date"] = pd.to_datetime(
    sec["available_at"]
).dt.normalize()

emb["date"] = pd.to_datetime(emb["date"])

cols = [
    c for c in emb.columns
    if c.startswith("kg_") and c[3:].isdigit()
]

records = []

for ticker, group in emb.groupby("tic"):

    group = group.sort_values("date").copy()
    vectors = group[cols].to_numpy(dtype=float)

    changes = np.zeros(len(group))

    changes[1:] = np.linalg.norm(
        np.diff(vectors, axis=0),
        axis=1
    )

    group["embedding_change"] = changes

    events = sec[sec["ticker"] == ticker]

    for event in events.itertuples(index=False):

        date = event.event_date

        # First trading observation STRICTLY AFTER
        # the event's date.
        future = group[group["date"] > date]

        if future.empty:
            continue

        row = future.iloc[0]

        records.append({
            "ticker": ticker,
            "event_date": date,
            "response_date": row["date"],
            "embedding_change": row["embedding_change"],
            "event_id": event.event_id,
        })

result = pd.DataFrame(records)

if result.empty:
    raise RuntimeError("No aligned SEC events")

result.to_csv(
    OUT / "event_alignment.csv",
    index=False
)

summary = result.groupby("ticker").agg(
    events=("event_id", "count"),
    mean_response=("embedding_change", "mean"),
    median_response=("embedding_change", "median"),
    zero_response_rate=(
        "embedding_change",
        lambda x: float((x < 1e-6).mean())
    ),
)

summary.to_csv(OUT / "summary.csv")

print("\n===== EVENT-EMBEDDING ALIGNMENT =====")
print(summary.round(6).to_string())

print("\nSaved:", OUT.resolve())
