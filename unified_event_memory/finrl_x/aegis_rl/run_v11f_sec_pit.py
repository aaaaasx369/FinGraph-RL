#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V11-F
SEC Official Acceptance -> NYSE Trading Calendar -> PIT RAG

Inputs:
  outputs/v31/evidence_alignment.parquet
  results/aegis_rl_v1/v11d_sec_metadata/
      sec_evidence_provenance.csv

Outputs:
  sec_pit_evidence.parquet
  sec_pit_retrieval.parquet
  sec_pit_quarantine.csv
  sec_pit_filing_audit.csv
  sec_pit_summary.json

Method:
  1. Parse official SEC acceptance timestamps in UTC.
  2. Convert to America/New_York for reporting.
  3. Locate earliest NYSE session close >= acceptance.
  4. Apply configurable additional session lag.
  5. Verify accession/form/filing metadata consistency.
  6. Retain original alignment availability separately.
  7. Export a retrieval-compatible PIT parquet.

Default lag_sessions=1:
  At least one complete additional trading session
  after the first eligible NYSE closing boundary.

IMPORTANT:
  Acceptance metadata matched, excerpt content NOT verified.
  This cannot certify original text provenance.
  The execution timing of FinRL-X must be audited separately.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


def log(*x):
    print(*x, flush=True)


def normalize_day(values):
    return pd.to_datetime(
        values, utc=True, errors="coerce"
    ).dt.tz_convert(None).dt.normalize()


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--alignment",
        type=Path,
        default=Path(
            "outputs/v31/evidence_alignment.parquet"
        ),
    )

    p.add_argument(
        "--metadata",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v11d_sec_metadata/"
            "sec_evidence_provenance.csv"
        ),
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v11f_sec_pit"
        ),
    )

    p.add_argument(
        "--lag-sessions",
        type=int,
        default=1,
    )

    p.add_argument(
        "--calendar-start",
        default="2015-01-01",
    )

    p.add_argument(
        "--calendar-end",
        default="2028-12-31",
    )

    args = p.parse_args()

    if args.lag_sessions < 0:
        raise ValueError(
            "--lag-sessions must be nonnegative"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("===== LOAD ORIGINAL SEC ALIGNMENT =====")

    raw = pd.read_parquet(
        args.alignment
    ).reset_index(drop=True)

    raw["source_row"] = np.arange(
        len(raw),
        dtype=np.int64,
    )

    log("[ALIGNMENT ROWS]", len(raw))

    log("===== LOAD VERIFIED SEC METADATA CATALOG MATCHES =====")

    meta = pd.read_csv(
        args.metadata,
        low_memory=False,
        dtype={
            "matched_accession": "string",
            "source_accession": "string",
            "evidence_id": "string",
        },
    )

    required = [
        "source_row",
        "ticker",
        "matched_accession",
        "match_method",
        "sec_acceptance_raw",
        "sec_filing_date",
        "source_filing_date",
        "sec_form",
        "source_form",
        "status",
    ]

    missing = [
        c for c in required
        if c not in meta.columns
    ]

    if missing:
        raise RuntimeError(
            f"Metadata missing columns: {missing}"
        )

    if (
        meta["source_row"].duplicated().any()
        or len(meta) != len(raw)
    ):
        raise RuntimeError(
            "Source row count or identity mismatch. "
            "Recreate V11-D metadata from this alignment."
        )

    # Verify exact row alignment with source before joining.
    checker = raw[
        ["source_row", "ticker", "current_accession"]
    ].merge(
        meta[
            ["source_row", "ticker", "matched_accession"]
        ],
        on="source_row",
        suffixes=("_source", "_metadata"),
        validate="one_to_one",
    )

    if not (
        checker["ticker_source"].astype(str)
        == checker["ticker_metadata"].astype(str)
    ).all():
        raise RuntimeError(
            "Ticker mismatch between source and metadata."
        )

    def normalize_accession(s):
        return (
            s.astype("string")
            .fillna("")
            .str.replace(
                r"[^0-9]",
                "",
                regex=True,
            )
        )

    acc_source = normalize_accession(
        checker["current_accession"]
    )

    acc_official = normalize_accession(
        checker["matched_accession"]
    )

    if not (
        acc_source == acc_official
    ).all():
        raise RuntimeError(
            "Accession mismatch between original "
            "SEC evidence and V11-D catalog."
        )

    # Keep the original Alignment columns intact.
    merge_cols = [
        "source_row",
        "matched_accession",
        "match_method",
        "sec_acceptance_raw",
        "sec_filing_date",
        "source_filing_date",
        "sec_form",
        "source_form",
        "status",
        "reason",
    ]

    merge_cols = [
        c for c in merge_cols
        if c in meta.columns
    ]

    merged = raw.merge(
        meta[merge_cols].rename(
            columns={
                "match_method": "sec_catalog_match_method",
                "status": "v11d_status",
                "reason": "v11d_reason",
            }
        ),
        on="source_row",
        how="left",
        validate="one_to_one",
    )

    log("===== PARSE SEC ACCEPTANCE TIMES =====")

    accepted_utc = pd.to_datetime(
        merged["sec_acceptance_raw"],
        utc=True,
        errors="coerce",
    )

    accepted_et = accepted_utc.dt.tz_convert(
        "America/New_York"
    )

    merged["sec_acceptance_utc"] = (
        accepted_utc.astype("string")
    )

    merged["sec_acceptance_et"] = (
        accepted_et.astype("string")
    )

    merged["sec_acceptance_et_date"] = (
        accepted_et.dt.tz_localize(None)
        .dt.normalize()
    )

    merged["alignment_available_from"] = (
        merged["available_from"]
    )

    log("===== BUILD NYSE SESSION CALENDAR =====")

    nyse = mcal.get_calendar("NYSE")

    schedule = nyse.schedule(
        start_date=args.calendar_start,
        end_date=args.calendar_end,
    )

    if schedule.empty:
        raise RuntimeError(
            "NYSE schedule is empty"
        )

    # Times are timezone-aware UTC timestamps.
    close_times = pd.to_datetime(
        schedule["market_close"],
        utc=True,
    )

    close_ns = close_times.astype(
        "int64"
    ).to_numpy()

    sessions = pd.DatetimeIndex(
        schedule.index
    ).normalize()

    log(
        "[NYSE SESSIONS]",
        len(sessions),
    )

    log("===== ASSIGN PIT SESSION =====")

    accepted_valid = accepted_utc.notna().to_numpy()

    first_index = np.full(
        len(merged),
        -1,
        dtype=np.int64,
    )

    if accepted_valid.any():
        accepted_ns = (
            accepted_utc[accepted_valid]
            .astype("int64")
            .to_numpy()
        )

        # Earliest session whose close is not earlier
        # than SEC acceptance.
        first_index[accepted_valid] = (
            np.searchsorted(
                close_ns,
                accepted_ns,
                side="left",
            )
        )

    decision_index = np.where(
        first_index >= 0,
        first_index + args.lag_sessions,
        -1,
    )

    valid_session = (
        (decision_index >= 0)
        & (decision_index < len(sessions))
    )

    pit_dates = np.full(
        len(merged),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    first_close_dates = np.full(
        len(merged),
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    first_valid = (
        (first_index >= 0)
        & (first_index < len(sessions))
    )

    if first_valid.any():
        first_close_dates[first_valid] = (
            sessions[
                first_index[first_valid]
            ].to_numpy(dtype="datetime64[ns]")
        )

    if valid_session.any():
        pit_dates[valid_session] = (
            sessions[
                decision_index[valid_session]
            ].to_numpy(dtype="datetime64[ns]")
        )

    merged["first_eligible_session"] = (
        pd.to_datetime(first_close_dates)
    )

    merged["pit_usable_from"] = (
        pd.to_datetime(pit_dates)
    )

    original_date = normalize_day(
        merged["alignment_available_from"]
    )

    merged["original_vs_pit_days"] = (
        original_date
        - merged["pit_usable_from"]
    ).dt.days

    log("===== CHECK FILING METADATA =====")

    official_filing = normalize_day(
        merged["sec_filing_date"]
    )

    source_filing = normalize_day(
        merged["source_filing_date"]
    )

    correct_filing_date = (
        official_filing.notna()
        & source_filing.notna()
        & (
            official_filing
            == source_filing
        )
    )

    correct_form = (
        merged["sec_form"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        ==
        merged["source_form"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    accession_match = (
        merged["sec_catalog_match_method"]
        .fillna("")
        .eq("ACCESSION")
        & merged["matched_accession"].notna()
    )

    valid_metadata = (
        valid_session
        & accepted_valid
        & correct_filing_date.to_numpy()
        & correct_form.to_numpy()
        & accession_match.to_numpy()
    )

    # An old V11-D date conflict by itself is NOT
    # grounds for quarantine, because its UTC-day
    # test has been superseded by this calendar audit.
    merged["pit_metadata_status"] = np.where(
        valid_metadata,
        "METADATA_MATCHED",
        "QUARANTINE",
    )

    def why_failed(i):
        reasons = []

        if not accepted_valid[i]:
            reasons.append("missing_acceptance")

        if not valid_session[i]:
            reasons.append("no_calendar_session")

        if not bool(correct_filing_date.iloc[i]):
            reasons.append("filing_date_mismatch")

        if not bool(correct_form.iloc[i]):
            reasons.append("form_mismatch")

        if not bool(accession_match.iloc[i]):
            reasons.append("accession_mismatch")

        return "|".join(reasons)

    merged["pit_failure_reason"] = [
        "" if valid_metadata[i] else why_failed(i)
        for i in range(len(merged))
    ]

    # Date difference is not automatically a failure:
    # original alignment and PIT use different rules.
    merged["original_available_earlier_than_pit"] = (
        original_date
        < merged["pit_usable_from"]
    )

    merged["excerpt_content_verified"] = False

    merged["pit_rule"] = (
        "NYSE_FIRST_CLOSE_AFTER_ACCEPTANCE"
        f"_PLUS_{args.lag_sessions}_SESSIONS"
    )

    log("===== EXPORT PIT EVIDENCE =====")

    evidence = merged.loc[
        valid_metadata
    ].copy()

    quarantine = merged.loc[
        ~valid_metadata
    ].copy()

    evidence.to_parquet(
        args.out / "sec_pit_evidence.parquet",
        index=False,
    )

    quarantine.to_csv(
        args.out / "sec_pit_quarantine.csv",
        index=False,
    )

    # IMPORTANT:
    # Existing V8/V9 Retrieval uses available_from.
    # Replace only in an exported copy, preserving
    # the original under alignment_available_from.
    retrieval = evidence.copy()

    retrieval["available_from"] = (
        retrieval["pit_usable_from"]
        .dt.strftime("%Y-%m-%d")
    )

    retrieval.to_parquet(
        args.out / "sec_pit_retrieval.parquet",
        index=False,
    )

    log("===== FILING-LEVEL AUDIT =====")

    filing = (
        merged.sort_values(
            ["ticker", "matched_accession", "source_row"],
            kind="stable",
        )
        .groupby(
            ["ticker", "matched_accession"],
            dropna=False,
        )
        .agg(
            evidence_rows=("source_row", "size"),
            first_eligible_session=(
                "first_eligible_session", "min"
            ),
            pit_usable_from=("pit_usable_from", "min"),
            original_available_min=(
                "alignment_available_from", "min"
            ),
            original_available_max=(
                "alignment_available_from", "max"
            ),
            metadata_matched=(
                "pit_metadata_status",
                lambda x: int(
                    (x == "METADATA_MATCHED").sum()
                ),
            ),
            quarantined=(
                "pit_metadata_status",
                lambda x: int(
                    (x == "QUARANTINE").sum()
                ),
            ),
        )
        .reset_index()
    )

    filing.to_csv(
        args.out / "sec_pit_filing_audit.csv",
        index=False,
    )

    log("===== SUMMARY =====")

    delta_counts = (
        merged["original_vs_pit_days"]
        .value_counts(dropna=False)
        .sort_index()
    )

    summary = {
        "experiment": "FinRL-X V11-F",
        "original_alignment_rows": int(len(raw)),
        "metadata_matched_rows": int(len(evidence)),
        "quarantine_rows": int(len(quarantine)),
        "unique_matched_filings": int(
            evidence["matched_accession"].nunique()
        ),
        "lag_sessions": args.lag_sessions,
        "calendar": "NYSE",
        "calendar_start": args.calendar_start,
        "calendar_end": args.calendar_end,
        "original_available_earlier_than_pit": int(
            merged[
                "original_available_earlier_than_pit"
            ].sum()
        ),
        "date_delta_distribution": {
            str(k): int(v)
            for k, v in delta_counts.items()
        },
        "official_filing_metadata_matched": True,
        "excerpt_content_verified": False,
        "portfolio_execution_timing_verified": False,
        "validation_or_test_performance_evaluated": False,
        "notes": [
            "Original alignment is unchanged.",
            "Only the output retrieval copy replaces available_from.",
            "Official SEC acceptance determines session availability.",
            "NYSE holidays and early closes are included by calendar.",
            "Conservative additional session lag is configurable.",
            "SEC excerpt-to-filing content matching remains necessary.",
            "FinRL-X daily market data execution assumptions remain to be audited.",
        ],
    }

    (
        args.out / "sec_pit_summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    log(
        "[MATCHED]",
        len(evidence),
        "/",
        len(merged)
    )
    log(
        "[QUARANTINE]",
        len(quarantine)
    )
    log(
        "[ORIGINAL EARLIER THAN PIT]",
        summary["original_available_earlier_than_pit"]
    )
    log("[DATE DELTA]")
    log(delta_counts.to_string())

    log()
    log("V11-F: PASS")
    log("EXCERPT CONTENT VERIFICATION: PENDING")
    log("TRADING EXECUTION ALIGNMENT: PENDING")


if __name__ == "__main__":
    main()
