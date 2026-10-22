#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V11-D
SEC EDGAR Filing Metadata Reconstruction

Inputs:
  outputs/v31/evidence_alignment.parquet
  SEC official submissions API:
    https://data.sec.gov/submissions/CIK##########.json
    historical filing index files referenced there

Outputs:
  sec_filing_catalog.csv
  sec_evidence_provenance.csv
  sec_verified_events.parquet
  sec_corroborated_events.parquet
  sec_unresolved_events.csv
  sec_filing_summary.csv
  provenance_report.json

No trading, no training, no future return labels.

VERIFIED_ACCESSION only if:
  - original evidence has valid accession;
  - accession matches the right company in SEC catalog;
  - available_from is not earlier than conservative
    acceptance-derived usable date;
  - provided filing date/form do not conflict;
  - SEC record has acceptanceDateTime.

It does NOT prove excerpt content belongs to the filing.
"""

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


CIKS = {
    "AAPL": 320193,
    "AMZN": 1018724,
    "GOOGL": 1652044,
    "MSFT": 789019,
    "NVDA": 1045810,
}

DATE_START = pd.Timestamp("2016-01-01")
DATE_END = pd.Timestamp("2024-12-31")

FIELDS = [
    "accessionNumber",
    "filingDate",
    "reportDate",
    "acceptanceDateTime",
    "form",
    "primaryDocument",
]


def log(*x):
    print(*x, flush=True)


def find_col(frame, names):
    lookup = {
        str(c).lower(): c
        for c in frame.columns
    }
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def date_parse(values):
    return pd.to_datetime(
        values, utc=True, errors="coerce"
    ).dt.tz_convert(None).dt.normalize()


def accession_clean(value):
    if pd.isna(value):
        return ""
    value = str(value).strip()
    if not value or value.lower() in ("nan", "none", "null"):
        return ""
    digits = re.sub(r"[^0-9]", "", value)
    return (
        f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
        if len(digits) == 18
        else ""
    )


def normalize_form(value):
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    return text if text not in ("NAN", "NONE") else ""


def read_json(url, cache_file, agent, refresh, delay, retries):
    if cache_file.exists() and not refresh:
        return json.loads(
            cache_file.read_text(encoding="utf-8")
        )

    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": agent,
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                },
            )

            with urllib.request.urlopen(
                request, timeout=45
            ) as response:
                content = response.read()

            data = json.loads(content.decode("utf-8"))

            cache_file.parent.mkdir(
                parents=True, exist_ok=True
            )
            cache_file.write_text(
                json.dumps(data),
                encoding="utf-8",
            )

            time.sleep(delay)
            return data

        except (
            urllib.error.URLError,
            TimeoutError,
            json.JSONDecodeError,
        ) as exc:
            error = exc
            log(
                f"[RETRY {attempt+1}/{retries}]",
                url,
                repr(exc),
            )
            time.sleep(min(3 * (attempt + 1), 15))

    raise RuntimeError(
        f"Unable to retrieve SEC metadata: {url}\n{error}\n"
        "Check internet access and SEC User-Agent."
    )


def records_to_frame(records):
    if not isinstance(records, dict):
        return pd.DataFrame(columns=FIELDS)

    lengths = [
        len(v) for v in records.values()
        if isinstance(v, list)
    ]

    if not lengths:
        return pd.DataFrame(columns=FIELDS)

    n = max(lengths)
    data = {}

    for field in FIELDS:
        v = records.get(field, [])
        if not isinstance(v, list):
            v = []
        data[field] = (v + [None] * n)[:n]

    return pd.DataFrame(data)


def sec_catalog(ticker, cik, args):
    base = "https://data.sec.gov/submissions/"
    prefix = f"CIK{cik:010d}"

    top = read_json(
        base + prefix + ".json",
        args.cache / (prefix + ".json"),
        args.user_agent,
        args.refresh,
        args.delay,
        args.retries,
    )

    filings = top.get("filings", {})

    frames = [
        records_to_frame(filings.get("recent", {}))
    ]

    archived = filings.get("files", [])

    for item in archived:
        name = item.get("name", "")
        if not name:
            continue

        archive = read_json(
            base + name,
            args.cache / name,
            args.user_agent,
            args.refresh,
            args.delay,
            args.retries,
        )

        frames.append(records_to_frame(archive))

    result = pd.concat(
        frames, ignore_index=True
    )

    result["ticker"] = ticker
    result["cik"] = cik

    result["accession"] = (
        result["accessionNumber"]
        .map(accession_clean)
    )

    result["filing_date"] = date_parse(
        result["filingDate"]
    )

    result["report_date"] = date_parse(
        result["reportDate"]
    )

    # EDGAR submissions acceptanceDateTime is
    # normally an ET local datetime string.
    # Retain the exact original value, and
    # use its calendar date for a conservative
    # next-day availability check.
    result["acceptance_raw"] = (
        result["acceptanceDateTime"]
        .fillna("").astype(str)
    )

    result["acceptance_date"] = date_parse(
        result["acceptanceDateTime"]
    )

    result["form_normalized"] = (
        result["form"].map(normalize_form)
    )

    result = result[
        result["accession"].ne("")
    ].drop_duplicates(
        ["ticker", "accession"]
    ).copy()

    log(
        f"[SEC] {ticker}: {len(result)} filing records; "
        f"acceptance present "
        f"{result['acceptance_date'].notna().sum()}"
    )

    return result


def load_alignment(path):
    raw = pd.read_parquet(path).reset_index(drop=True)

    ticker = find_col(
        raw, ["ticker", "company"]
    )
    available = find_col(
        raw, ["available_from"]
    )
    filing = find_col(
        raw, [
            "current_filing_date",
            "filing_date",
            "filed_at",
        ],
    )
    form = find_col(
        raw, [
            "current_form",
            "filing_type",
            "form_type",
            "form",
        ],
    )
    accession = find_col(
        raw, [
            "current_accession_number",
            "current_accession",
            "accession_number",
            "accession",
            "accession_no",
        ],
    )
    identifier = find_col(
        raw, [
            "evidence_id",
            "source_id",
            "pair_id",
        ],
    )
    excerpt = find_col(
        raw, ["current_excerpt"]
    )

    if ticker is None or available is None:
        raise RuntimeError(
            "Alignment requires ticker and available_from. "
            f"Found: {raw.columns.tolist()}"
        )

    def get(name):
        if name is None:
            return pd.Series(
                [None] * len(raw),
                index=raw.index,
            )
        return raw[name]

    result = pd.DataFrame({
        "source_row": np.arange(len(raw)),
        "ticker": get(ticker).astype(str).str.upper(),
        "available_from": date_parse(get(available)),
        "filing_date": date_parse(get(filing)),
        "form": get(form).map(normalize_form),
        "accession": get(accession).map(accession_clean),
        "evidence_id": get(identifier).astype("string"),
        "excerpt": get(excerpt).fillna("").astype(str),
    })

    result = result[
        result["ticker"].isin(CIKS)
    ].copy()

    return result, {
        "columns": list(raw.columns),
        "accession_column": accession,
        "filing_column": filing,
        "form_column": form,
        "available_column": available,
        "evidence_id_column": identifier,
        "excerpt_column": excerpt,
    }


def match_evidence(alignment, catalog):
    lookup = {}

    for ticker, group in catalog.groupby("ticker"):
        lookup[ticker] = {
            "accession": {
                accession: row
                for accession, row in group.set_index(
                    "accession",
                    drop=False,
                ).iterrows()
            },
            "filing_date": {
                d: part
                for d, part in group.groupby(
                    "filing_date",
                    dropna=True,
                )
            },
        }

    rows = []

    for record in alignment.itertuples(index=False):
        ticker = record.ticker
        source_acc = record.accession
        original_date = record.filing_date
        original_form = record.form
        recorded_available = record.available_from

        candidates = []
        match_method = ""
        failure = ""

        mapping = lookup[ticker]

        if source_acc:
            found = mapping["accession"].get(source_acc)

            if found is not None:
                candidates = [found]
                match_method = "ACCESSION"
            else:
                failure = "accession_not_found_in_sec_catalog"

        elif pd.notna(original_date):
            part = mapping["filing_date"].get(
                original_date
            )

            if part is not None:
                if original_form:
                    part = part[
                        part["form_normalized"]
                        == original_form
                    ]

                candidates = [
                    row
                    for _, row in part.iterrows()
                ]

            match_method = "DATE_FORM"

            if not candidates:
                failure = "filing_date_or_form_unmatched"

        else:
            failure = "missing_accession_and_filing_date"

        official = (
            candidates[0]
            if len(candidates) == 1 else None
        )

        if len(candidates) > 1:
            failure = "multiple_candidate_filings"

        sec_acc = (
            str(official["accession"])
            if official is not None else ""
        )

        official_date = (
            official["filing_date"]
            if official is not None else pd.NaT
        )

        accepted_date = (
            official["acceptance_date"]
            if official is not None else pd.NaT
        )

        official_form = (
            str(official["form_normalized"])
            if official is not None else ""
        )

        accepted_raw = (
            str(official["acceptance_raw"])
            if official is not None else ""
        )

        # Next calendar day after SEC acceptance.
        # This is deliberately conservative for
        # date-only backtests, NOT intraday execution.
        conservative = (
            accepted_date + pd.Timedelta(days=1)
            if pd.notna(accepted_date)
            else pd.NaT
        )

        conflicts = []

        if official is not None:
            if (
                pd.notna(original_date)
                and original_date != official_date
            ):
                conflicts.append("filing_date_conflict")

            if (
                original_form
                and original_form != official_form
            ):
                conflicts.append("form_conflict")

            if (
                pd.notna(recorded_available)
                and pd.notna(conservative)
                and recorded_available < conservative
            ):
                conflicts.append(
                    "available_before_safe_acceptance_date"
                )

        if conflicts:
            status = "CONFLICT"
            failure = "|".join(conflicts)

        elif official is None:
            status = "UNRESOLVED"

        elif pd.isna(accepted_date):
            status = "UNRESOLVED"
            failure = "sec_acceptance_time_missing"

        elif pd.isna(recorded_available):
            status = "UNRESOLVED"
            failure = "alignment_available_from_missing"

        elif match_method == "ACCESSION":
            status = "VERIFIED_ACCESSION"

        else:
            status = "CORROBORATED_FILING"

        # A conservative corrected date is useful
        # but never silently overwrites the original.
        safe_date = (
            max(recorded_available, conservative)
            if (
                pd.notna(recorded_available)
                and pd.notna(conservative)
            )
            else pd.NaT
        )

        rows.append({
            "source_row": int(record.source_row),
            "ticker": ticker,
            "cik": CIKS[ticker],
            "evidence_id": record.evidence_id,
            "source_accession": source_acc,
            "matched_accession": sec_acc,
            "match_method": match_method,
            "match_candidate_count": len(candidates),
            "source_filing_date": original_date,
            "sec_filing_date": official_date,
            "source_form": original_form,
            "sec_form": official_form,
            "source_available_from": recorded_available,
            "sec_acceptance_raw": accepted_raw,
            "sec_acceptance_date": accepted_date,
            "conservative_available_from": conservative,
            "safe_available_from": safe_date,
            "status": status,
            "reason": failure,
            "excerpt": record.excerpt,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--documents",
        type=Path,
        default=Path(
            "outputs/v31/evidence_alignment.parquet"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/v11d_sec_metadata"
        ),
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path(
            "results/aegis_rl_v1/"
            "v11d_sec_metadata/sec_api_cache"
        ),
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help="Descriptive application/contact string containing your email",
    )
    parser.add_argument(
        "--delay", type=float, default=0.25
    )
    parser.add_argument(
        "--retries", type=int, default=3
    )
    parser.add_argument(
        "--refresh", action="store_true"
    )
    args = parser.parse_args()

    if (
        "@" not in args.user_agent
        or "example.com" in args.user_agent.lower()
    ):
        raise ValueError(
            "Supply your actual contact email in --user-agent"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    args.cache.mkdir(parents=True, exist_ok=True)

    log("===== LOAD SEC ALIGNMENT =====")
    alignment, schema = load_alignment(args.documents)

    log(
        "[ALIGNMENT ROWS]",
        len(alignment),
    )
    log("[ALIGNMENT SCHEMA]", schema)

    log("===== RECONSTRUCT SEC FILING CATALOG =====")

    catalog_frames = []

    for ticker, cik in CIKS.items():
        catalog_frames.append(
            sec_catalog(ticker, cik, args)
        )

    catalog = pd.concat(
        catalog_frames, ignore_index=True
    )

    catalog.to_csv(
        args.out / "sec_filing_catalog.csv",
        index=False,
    )

    log(
        "[OFFICIAL FILINGS]",
        len(catalog)
    )

    log("===== MATCH EVIDENCE TO FILINGS =====")

    result = match_evidence(
        alignment,
        catalog,
    )

    result.to_csv(
        args.out / "sec_evidence_provenance.csv",
        index=False,
    )

    # This is a high-confidence metadata candidate
    # set, not proof that excerpt text is in filing.
    verified = result[
        result["status"] == "VERIFIED_ACCESSION"
    ].copy()

    corroborated = result[
        result["status"] == "CORROBORATED_FILING"
    ].copy()

    unresolved = result[
        result["status"].isin(
            ["UNRESOLVED", "CONFLICT"]
        )
    ].copy()

    verified.to_parquet(
        args.out / "sec_verified_events.parquet",
        index=False,
    )

    corroborated.to_parquet(
        args.out / "sec_corroborated_events.parquet",
        index=False,
    )

    unresolved.to_csv(
        args.out / "sec_unresolved_events.csv",
        index=False,
    )

    summary_table = (
        result.groupby(
            ["ticker", "status"]
        )
        .size()
        .rename("rows")
        .reset_index()
    )

    summary_table.to_csv(
        args.out / "sec_filing_summary.csv",
        index=False,
    )

    counts = {
        str(k): int(v)
        for k, v in result["status"].value_counts().items()
    }

    report = {
        "experiment": "FinRL-X V11-D",
        "alignment_rows": len(alignment),
        "official_filing_records": len(catalog),
        "status_counts": counts,
        "schema": schema,
        "external_metadata_source": (
            "https://data.sec.gov/submissions/"
        ),
        "public_filing_metadata_checked": True,
        "excerpt_text_verified_in_official_document": False,
        "strict_pit_document_provenance_complete": False,
        "historical_return_labels_used": False,
        "important_notes": [
            "Date-only conservative acceptance+1 rule",
            "Matching date/form does not prove excerpt provenance",
            "VERIFIED_ACCESSION requires source accession",
            "2018-2020 and other filings may be present",
            "Check EDGAR content before asserting verified evidence",
        ],
    }

    (
        args.out / "provenance_report.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    log()
    log("=" * 75)
    log("V11-D SEC METADATA RECONSTRUCTION")
    log("=" * 75)
    log(summary_table.to_string(index=False))
    log()
    log("STATUS COUNTS:", counts)
    log("VERIFIED ACCESSION:", len(verified))
    log("CORROBORATED FILING:", len(corroborated))
    log("UNRESOLVED/CONFLICT:", len(unresolved))
    log("OUTPUT:", args.out)
    log("EXCERPT CONTENT VERIFICATION: NOT DONE")
    log("V11-D: PASS")


if __name__ == "__main__":
    main()
