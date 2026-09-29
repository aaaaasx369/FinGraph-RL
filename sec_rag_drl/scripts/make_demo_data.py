#!/usr/bin/env python3
"""
Creates synthetic files so you can verify the whole RL pipeline BEFORE
pointing it at real data. This is not research data.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
dates = pd.bdate_range("2018-01-02", "2025-10-24")

rows = []
for j, t in enumerate(tickers):
    mu = 0.00025 + j * 0.00003
    sigma = 0.014 + j * 0.001
    rets = rng.normal(mu, sigma, len(dates))
    close = 100 * np.cumprod(1 + rets)
    for d, c in zip(dates, close):
        rows.append({"date": d.date(), "ticker": t, "close": c})

pd.DataFrame(rows).to_csv(DATA / "market.csv", index=False)

rag_rows = []
for t in tickers:
    for d in pd.date_range("2018-02-15", "2025-08-15", freq="90D"):
        impact = float(np.clip(rng.normal(0, 0.4), -1, 1))
        risk = float(np.clip(rng.beta(2, 4), 0, 1))
        confidence = float(np.clip(rng.beta(5, 2), 0, 1))
        rag_rows.append({
            "ticker": t,
            "available_date": d.date(),
            "filing_type": "10-Q",
            "impact": impact,
            "risk": risk,
            "confidence": confidence,
            "rationale": "Synthetic demo only.",
        })

pd.DataFrame(rag_rows).to_csv(DATA / "rag_signals.csv", index=False)
print("Created:")
print(DATA / "market.csv")
print(DATA / "rag_signals.csv")
