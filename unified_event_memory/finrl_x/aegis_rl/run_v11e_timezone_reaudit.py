#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np

source = Path(
    "results/aegis_rl_v1/v11d_sec_metadata/"
    "sec_evidence_provenance.csv"
)

out = Path(
    "results/aegis_rl_v1/v11e_timezone_reaudit"
)
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source, low_memory=False)

required = [
    "ticker",
    "matched_accession",
    "source_available_from",
    "sec_acceptance_raw",
    "source_filing_date",
    "sec_filing_date",
    "source_form",
    "sec_form",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

# SEC submission timestamps in the cached catalog
# are represented as UTC timestamps ending in Z.
accepted_utc = pd.to_datetime(
    df["sec_acceptance_raw"],
    utc=True,
    errors="coerce",
)

accepted_et = accepted_utc.dt.tz_convert(
    "America/New_York"
)

available = pd.to_datetime(
    df["source_available_from"],
    errors="coerce",
).dt.normalize()

accepted_et_date = (
    accepted_et.dt.tz_localize(None)
    .dt.normalize()
)

# Match the original project's conservative
# next-CALENDAR-day availability convention.
expected = (
    accepted_et_date
    + pd.Timedelta(days=1)
)

df["acceptance_utc"] = accepted_utc.astype(str)
df["acceptance_et"] = accepted_et.astype(str)
df["acceptance_et_date"] = accepted_et_date
df["expected_available_et_plus_1"] = expected

df["availability_delta_days"] = (
    available - expected
).dt.days

original_filing = pd.to_datetime(
    df["source_filing_date"],
    errors="coerce",
).dt.normalize()

official_filing = pd.to_datetime(
    df["sec_filing_date"],
    errors="coerce",
).dt.normalize()

form_equal = (
    df["source_form"].fillna("").astype(str)
    == df["sec_form"].fillna("").astype(str)
)

date_equal = (
    original_filing == official_filing
)

metadata_conflict = (
    (~form_equal)
    | (~date_equal)
)

df["reaudit_status"] = np.select(
    [
        accepted_utc.isna() | available.isna(),
        metadata_conflict,
        available < expected,
        available > expected,
    ],
    [
        "MISSING_TIMESTAMP",
        "METADATA_CONFLICT",
        "EARLY_REVIEW",
        "DELAYED",
    ],
    default="CONSISTENT",
)

df.to_csv(
    out / "evidence_timezone_reaudit.csv",
    index=False,
)

filings = (
    df.sort_values(
        ["ticker", "matched_accession"],
        kind="stable",
    )
    .drop_duplicates(
        ["ticker", "matched_accession"]
    )
    .copy()
)

filings.to_csv(
    out / "filing_timezone_reaudit.csv",
    index=False,
)

evidence_summary = pd.crosstab(
    df["ticker"],
    df["reaudit_status"],
)

filing_summary = pd.crosstab(
    filings["ticker"],
    filings["reaudit_status"],
)

evidence_summary.to_csv(
    out / "evidence_status_summary.csv"
)

filing_summary.to_csv(
    out / "filing_status_summary.csv"
)

print("\n========== EVIDENCE LEVEL ==========")
print(evidence_summary.to_string())

print("\n========== FILING LEVEL ==========")
print(filing_summary.to_string())

print("\n========== OVERALL ==========")
print(df["reaudit_status"].value_counts().to_string())

print("\n========== DATE DELTA ==========")
print(
    df["availability_delta_days"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

print("\n========== REMAINING EARLY CASES ==========")

early = filings[
    filings["reaudit_status"] == "EARLY_REVIEW"
]

columns = [
    "ticker",
    "matched_accession",
    "source_filing_date",
    "acceptance_et",
    "source_available_from",
    "expected_available_et_plus_1",
    "availability_delta_days",
]

print(
    early[columns].head(30).to_string(index=False)
    if len(early)
    else "No early availability cases found."
)

print("\nV11-E TIMEZONE REAUDIT: PASS")
print("Official excerpt content verification: NOT DONE")
print("Trading-session alignment verification: NOT DONE")
