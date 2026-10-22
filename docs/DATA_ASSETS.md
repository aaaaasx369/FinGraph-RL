# Research Data Assets

This document describes selected tracked datasets used by FinGraph-RL.

## 1. finrl_trading.7z

Path: `data/finrl_trading.7z`

- Size: approximately 42.48 MiB.
- Format: 7z archive.
- Archive signature was verified.
- Internal contents and archive integrity have not yet been verified.
- No exact filename references were found in the tracked source-code search.
- Retained to avoid losing potentially important historical research data.

## 2. fundamental_data_full.csv

Path: `data/fundamental_data_full.csv`

- Size: approximately 22.98 MiB.
- Contains financial fundamental observations.
- Observed columns include ticker, datadate, EPS, BPS, roe,
  filing_date, accepted_date, tradedate, and trade_price.
- No exact filename references were found in the tracked source-code search.
- Relationship to the SQLite fundamental_data table is not yet verified.
- Retained until provenance and reproducibility are established.

## 3. sp500_historical_constituents.csv

Path: `data/sp500_historical_constituents.csv`

- Size: approximately 5.25 MiB.
- Referenced by src/data/backfill_historical_sp500.py.
- Referenced by src/data/data_fetcher.py.
- Documented in ML_STOCK_SELECTION.md.
- Required path should be preserved unless consumers are updated.

## Data Management Policy

- Do not remove datasets solely because filename searches return zero matches.
- Confirm whether an asset is required for historical reproducibility.
- Verify provenance and alternatives before removing tracked data.
- Do not rewrite Git history without a separate migration plan.
- Keep local backups outside the repository.
- Avoid committing credentials, private datasets, or newly generated
  large artifacts without review.

## Outstanding Verification

- Inspect the contents and integrity of finrl_trading.7z.
- Compare fundamental_data_full.csv with the SQLite database,
  if the database is available.
- Confirm licensing and redistribution permissions for datasets.
