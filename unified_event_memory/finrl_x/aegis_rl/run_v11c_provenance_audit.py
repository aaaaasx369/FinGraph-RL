#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FinRL-X V11-C: SEC + Unified Temporal KG Provenance Audit

Read-only inputs:
  SEC evidence_alignment.parquet
  Unified Temporal KG CSV
  V11-B clean_event_memory.csv (optional)
  relation2id.json (optional)

Outputs:
  sec_provenance_rows.csv
  sec_date_distribution.csv
  sec_boundary_examples.csv
  kg_relation_audit.csv
  kg_temporal_audit.csv
  relation_mapping_candidates.csv
  event_quality_candidates.csv
  audit_summary.json
  audit_report.txt

No model training.
No price labels.
No 2021 validation/test evaluation.
No assumption of verified public-release timestamps.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]
START = pd.Timestamp("2018-01-01")
END = pd.Timestamp("2020-12-31")


def log(*parts):
    print(*parts, flush=True)


def col(df, names):
    lookup = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lookup:
            return lookup[name.lower()]
    return None


def series_or_empty(df, names):
    name = col(df, names)
    if name is None:
        return pd.Series(
            pd.NA, index=df.index, dtype="object"
        )
    return df[name]


def day_parse(s):
    # UTC-normalize consistently for internal comparison.
    return pd.to_datetime(
        s, utc=True, errors="coerce"
    ).dt.tz_convert(None).dt.normalize()


def company_match(value, ticker):
    return bool(re.search(
        rf"(?<![A-Z0-9]){re.escape(ticker)}(?![A-Z0-9])",
        str(value).upper()
    ))


def date_string(series):
    return series.dt.strftime("%Y-%m-%d")


def summarize_flag(df, field):
    return {
        str(k): int(v)
        for k, v in df[field].value_counts(
            dropna=False
        ).items()
    }


def sec_audit(path, out):
    log("===== SEC SOURCE AUDIT =====")

    df = pd.read_parquet(path).reset_index(drop=True)

    required = {
        "ticker": col(df, ["ticker", "company"]),
        "available": col(df, ["available_from"]),
        "excerpt": col(df, ["current_excerpt"]),
    }

    if any(v is None for v in required.values()):
        raise RuntimeError(
            f"SEC missing essential columns: {required}\n"
            f"Available: {list(df.columns)}"
        )

    log("[SEC COLUMNS]", list(df.columns))

    date_fields = {
        "available_from": ["available_from"],
        "filing_date": [
            "current_filing_date", "filing_date",
            "filed_at", "filed_date"
        ],
        "acceptance_time": [
            "acceptance_datetime",
            "accepted_at", "acceptance_time",
            "sec_accepted_at"
        ],
        "report_period": [
            "period_of_report", "report_date",
            "period_end", "current_report_date",
            "period_ending"
        ],
        "prior_filing_date": [
            "previous_filing_date", "prior_filing_date"
        ],
    }

    audit = pd.DataFrame(index=df.index)

    audit["ticker"] = (
        df[required["ticker"]]
        .fillna("")
        .astype(str)
        .str.upper()
    )

    audit["source_row"] = np.arange(len(df))

    audit["evidence_id"] = (
        series_or_empty(
            df, ["evidence_id", "source_id", "pair_id"]
        ).astype("string")
    )

    audit["topic"] = (
        series_or_empty(
            df, ["topic", "section", "item"]
        ).fillna("UNKNOWN").astype(str)
    )

    audit["excerpt"] = (
        df[required["excerpt"]]
        .fillna("").astype(str)
    )

    audit["excerpt_length"] = (
        audit["excerpt"].str.len()
    )

    raw_availability = df[required["available"]]

    # Keep original timestamps unmodified in a separate
    # field, since converting to date loses intraday detail.
    audit["available_from_raw"] = (
        raw_availability.astype("string")
    )

    for target, aliases in date_fields.items():
        source = col(df, aliases)
        audit[target] = day_parse(
            series_or_empty(df, aliases)
        )
        audit[target + "_column"] = (
            source if source else ""
        )

    audit["has_acceptance_timestamp"] = (
        audit["acceptance_time"].notna()
    )

    audit["lag_from_filing_days"] = (
        audit["available_from"]
        - audit["filing_date"]
    ).dt.days

    audit["lag_from_acceptance_days"] = (
        audit["available_from"]
        - audit["acceptance_time"]
    ).dt.days

    audit["available_before_filing"] = (
        audit["filing_date"].notna()
        & (
            audit["available_from"]
            < audit["filing_date"]
        )
    )

    audit["available_before_acceptance"] = (
        audit["acceptance_time"].notna()
        & (
            audit["available_from"]
            < audit["acceptance_time"]
        )
    )

    audit["at_dataset_start"] = (
        audit["available_from"]
        <= pd.Timestamp("2018-01-02")
    )

    audit["very_old_filing_at_start"] = (
        audit["at_dataset_start"]
        & audit["filing_date"].notna()
        & (
            audit["filing_date"]
            < pd.Timestamp("2017-07-01")
        )
    )

    audit["long_filing_lag"] = (
        audit["lag_from_filing_days"] > 180
    )

    audit["filing_lag_unknown"] = (
        audit["filing_date"].isna()
    )

    audit["future_dated"] = (
        audit["available_from"]
        > pd.Timestamp("2020-12-31")
    )

    audit["missing_availability"] = (
        audit["available_from"].isna()
    )

    audit["empty_excerpt"] = (
        audit["excerpt_length"] < 25
    )

    hard_fail = (
        audit["missing_availability"]
        | audit["available_before_filing"]
        | audit["available_before_acceptance"]
        | audit["empty_excerpt"]
    )

    review = (
        audit["at_dataset_start"]
        | audit["very_old_filing_at_start"]
        | audit["long_filing_lag"]
        | audit["filing_lag_unknown"]
        | (~audit["has_acceptance_timestamp"])
    )

    audit["quality_status"] = np.select(
        [hard_fail, review],
        ["QUARANTINE", "REVIEW"],
        default="PROVISIONAL",
    )

    def reasons(row):
        flags = [
            "missing_availability",
            "available_before_filing",
            "available_before_acceptance",
            "empty_excerpt",
            "at_dataset_start",
            "very_old_filing_at_start",
            "long_filing_lag",
            "filing_lag_unknown",
        ]
        result = [f for f in flags if row[f]]
        if not row["has_acceptance_timestamp"]:
            result.append("acceptance_timestamp_unverified")
        return "|".join(result)

    audit["quality_reasons"] = audit.apply(
        reasons, axis=1
    )

    # Keep raw rows for later source-level audits.
    audit.to_csv(
        out / "sec_provenance_rows.csv",
        index=False,
    )

    grouped = (
        audit.groupby(
            ["ticker", "available_from"],
            dropna=False
        )
        .agg(
            rows=("source_row", "size"),
            unique_evidence=("evidence_id", "nunique"),
            distinct_topics=("topic", "nunique"),
            filings=("filing_date", "nunique"),
            min_filing=("filing_date", "min"),
            max_filing=("filing_date", "max"),
            quarantined=(
                "quality_status",
                lambda s: int((s == "QUARANTINE").sum())
            ),
            review=(
                "quality_status",
                lambda s: int((s == "REVIEW").sum())
            ),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    grouped.to_csv(
        out / "sec_date_distribution.csv",
        index=False,
    )

    boundary = audit[
        audit["at_dataset_start"]
        | audit["very_old_filing_at_start"]
        | audit["available_before_filing"]
    ].copy()

    boundary = boundary.sort_values(
        ["ticker", "available_from", "filing_date"],
        kind="stable",
    )

    boundary["excerpt_sample"] = (
        boundary["excerpt"].str[:350]
    )

    boundary.drop(
        columns=["excerpt"]
    ).to_csv(
        out / "sec_boundary_examples.csv",
        index=False,
    )

    summary = {
        "rows": len(audit),
        "quality_status": summarize_flag(
            audit, "quality_status"
        ),
        "at_dataset_start": int(
            audit["at_dataset_start"].sum()
        ),
        "very_old_filing_at_start": int(
            audit["very_old_filing_at_start"].sum()
        ),
        "available_before_filing": int(
            audit["available_before_filing"].sum()
        ),
        "available_before_acceptance": int(
            audit["available_before_acceptance"].sum()
        ),
        "long_filing_lag": int(
            audit["long_filing_lag"].sum()
        ),
        "missing_filing_dates": int(
            audit["filing_lag_unknown"].sum()
        ),
        "has_acceptance_timestamp": int(
            audit["has_acceptance_timestamp"].sum()
        ),
        "raw_date_columns": {
            k: col(df, aliases)
            for k, aliases in date_fields.items()
        },
        "top_available_dates": [
            {
                "ticker": str(r.ticker),
                "date": str(r.available_from),
                "rows": int(r.rows),
                "distinct_topics": int(r.distinct_topics),
            }
            for r in grouped.head(15).itertuples()
        ],
    }

    log("[SEC]", summary)

    return audit, summary


def relation_map_search(root, optional_path, out):
    log("===== RELATION METADATA DISCOVERY =====")

    paths = []

    if optional_path:
        paths.append(Path(optional_path))

    # Restricted local discovery, no repository-wide
    # expensive recursive crawl.
    for base in [
        root / "outputs",
        root / "unified_event_memory",
        root / "kg" / "output",
    ]:
        if base.exists():
            paths.extend(
                list(base.glob("*/relation2id.json"))
            )
            paths.extend(
                list(base.glob("*/*/relation2id.json"))
            )
            paths.extend(
                list(base.glob("*/*/*/relation2id.json"))
            )

    paths = sorted(
        set(paths),
        key=lambda p: str(p)
    )

    records = []

    for path in paths:
        if not path.is_file():
            continue

        try:
            obj = json.loads(
                path.read_text(encoding="utf-8")
            )

            if isinstance(obj, dict):
                keys = list(obj.keys())
                values = list(obj.values())
            elif isinstance(obj, list):
                keys = list(map(str, obj))
                values = list(range(len(obj)))
            else:
                continue

            numeric = sum(
                bool(re.search(
                    r"FINDKG_REL[:_]\d+$",
                    str(k)
                ))
                for k in keys
            )

            records.append({
                "path": str(path),
                "n_relations": len(keys),
                "n_numbered_findkg": numeric,
                "relation_names": json.dumps(
                    keys, ensure_ascii=False
                ),
                "mapping_values": json.dumps(
                    values, ensure_ascii=False,
                    default=str
                ),
                "semantic_definition_verified": False,
            })
        except Exception as exc:
            records.append({
                "path": str(path),
                "error": repr(exc),
                "semantic_definition_verified": False,
            })

    pd.DataFrame(
        records,
        columns=[
            "path", "n_relations", "n_numbered_findkg",
            "relation_names", "mapping_values",
            "semantic_definition_verified", "error"
        ],
    ).to_csv(
        out / "relation_mapping_candidates.csv",
        index=False,
    )

    return records


def kg_audit(path, out):
    log("===== TEMPORAL KG PROVENANCE AUDIT =====")

    raw = pd.read_csv(
        path, low_memory=False
    ).reset_index(drop=True)

    head = col(raw, [
        "head", "subject", "src", "source_entity",
        "head_entity", "subject_id", "src_entity"
    ])

    tail = col(raw, [
        "tail", "object", "dst", "target_entity",
        "tail_entity", "object_id", "dst_entity"
    ])

    rel = col(raw, [
        "relation", "predicate", "rel",
        "relation_type", "edge_type"
    ])

    time = col(raw, [
        "available_at", "available_from",
        "event_date", "timestamp", "date", "time"
    ])

    if any(v is None for v in [head, tail, rel, time]):
        raise RuntimeError(
            "KG schema unresolved. "
            f"Available columns: {list(raw.columns)}"
        )

    kg = pd.DataFrame({
        "head": raw[head].fillna("").astype(str),
        "tail": raw[tail].fillna("").astype(str),
        "relation": raw[rel].fillna("").astype(str),
        "event_date": day_parse(raw[time]),
    })

    kg["source_row"] = np.arange(len(kg))

    kg["numbered_relation"] = (
        kg["relation"].str.contains(
            r"FINDKG_REL[:_]\d+$",
            regex=True, na=False
        )
    )

    kg["known_label"] = kg["relation"].isin([
        "COMPETES_WITH",
        "SUPPLIES_TO",
        "CUSTOMER_OF",
        "HAS_SEC_SIGNAL",
    ])

    kg["missing_date"] = (
        kg["event_date"].isna()
    )

    kg["before_2018"] = (
        kg["event_date"] < START
    )

    kg["after_2020"] = (
        kg["event_date"] > END
    )

    kg["within_2018_2020"] = (
        (kg["event_date"] >= START)
        & (kg["event_date"] <= END)
    )

    kg = kg.sort_values(
        ["event_date", "head", "relation", "tail"],
        kind="stable",
    )

    keys = ["head", "relation", "tail"]

    kg["first_observed_date"] = (
        kg.groupby(keys, sort=False)[
            "event_date"
        ].transform("min")
    )

    kg["is_first_observation"] = (
        kg["event_date"]
        == kg["first_observed_date"]
    )

    kg["observations_per_triple"] = (
        kg.groupby(keys, sort=False)[
            "source_row"
        ].transform("size")
    )

    kg["tied_first_rows"] = (
        kg.groupby(
            keys + ["event_date"],
            sort=False
        )["source_row"].transform("size")
    )

    kg["direct_company_link"] = False

    for ticker in TICKERS:
        kg["direct_company_link"] |= (
            kg["head"].map(
                lambda s: company_match(s, ticker)
            )
            | kg["tail"].map(
                lambda s: company_match(s, ticker)
            )
        )

    # Without external release metadata, even a
    # readable edge is not a verified PIT event.
    kg["quality_status"] = np.where(
        kg["missing_date"],
        "QUARANTINE",
        "REVIEW",
    )

    kg["quality_reasons"] = np.where(
        kg["missing_date"],
        "missing_event_date",
        "event_time_not_verified_public_release",
    )

    kg.loc[
        kg["numbered_relation"],
        "quality_reasons"
    ] += "|relation_semantics_unknown"

    kg.loc[
        ~kg["is_first_observation"],
        "quality_reasons"
    ] += "|repeat_snapshot_or_triple"

    rel_stats = (
        kg.groupby("relation", dropna=False)
        .agg(
            rows=("source_row", "size"),
            unique_heads=("head", "nunique"),
            unique_tails=("tail", "nunique"),
            first_date=("event_date", "min"),
            last_date=("event_date", "max"),
            distinct_triples=(
                "first_observed_date", "size"
            ),
            first_observed_rows=(
                "is_first_observation", "sum"
            ),
            company_linked_rows=(
                "direct_company_link", "sum"
            ),
            numbered_relation=(
                "numbered_relation", "max"
            ),
            known_label=("known_label", "max"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )

    # Correct distinct triple count for each relation.
    distinct = (
        kg[["relation", "head", "tail"]]
        .drop_duplicates()
        .groupby("relation")
        .size()
        .to_dict()
    )
    rel_stats["distinct_triples"] = (
        rel_stats["relation"].map(distinct)
    )

    rel_stats.to_csv(
        out / "kg_relation_audit.csv",
        index=False,
    )

    timeline = (
        kg.assign(
            month=kg["event_date"].dt.to_period("M")
            .astype("string")
        )
        .groupby(["month", "relation"], dropna=False)
        .agg(
            rows=("source_row", "size"),
            first_observed_rows=(
                "is_first_observation", "sum"
            ),
            company_linked_rows=(
                "direct_company_link", "sum"
            ),
        )
        .reset_index()
    )

    timeline.to_csv(
        out / "kg_temporal_audit.csv",
        index=False,
    )

    samples = kg[
        kg["direct_company_link"]
        & kg["within_2018_2020"]
    ].head(150).copy()

    samples.to_csv(
        out / "kg_company_edge_samples.csv",
        index=False,
    )

    summary = {
        "raw_edges": int(len(kg)),
        "relations": int(kg["relation"].nunique()),
        "distinct_triples": int(
            kg[keys].drop_duplicates().shape[0]
        ),
        "numbered_relation_rows": int(
            kg["numbered_relation"].sum()
        ),
        "company_linked_rows": int(
            kg["direct_company_link"].sum()
        ),
        "first_observed_rows": int(
            kg["is_first_observation"].sum()
        ),
        "missing_date_rows": int(
            kg["missing_date"].sum()
        ),
        "date_column": time,
        "status_counts": summarize_flag(
            kg, "quality_status"
        ),
    }

    log("[KG]", summary)

    return kg, summary


def event_quality(v11b_path, sec, out):
    log("===== V11-B EVENT QUALITY JOIN =====")

    if not Path(v11b_path).is_file():
        log(
            "[WARNING] V11-B event memory not found:",
            v11b_path
        )
        return {"available": False}

    events = pd.read_csv(
        v11b_path, low_memory=False
    )

    needed = [
        "event_id", "source", "ticker",
        "source_time", "source_id"
    ]

    missing = [
        x for x in needed if x not in events.columns
    ]

    if missing:
        raise RuntimeError(
            f"V11-B event memory missing {missing}"
        )

    events["source_time"] = day_parse(
        events["source_time"]
    )

    events["quality_status"] = "REVIEW"

    events["quality_reason"] = np.where(
        events["source"] == "KG",
        "kg_event_time_not_proven_public_release",
        "sec_original_provenance_unverified",
    )

    events["source_row_match_count"] = 0

    # SEC matching is done through retained source ID,
    # ticker and date, not row-order assumptions.
    sec_keys = sec[
        [
            "ticker", "available_from",
            "evidence_id", "quality_status",
            "quality_reasons"
        ]
    ].copy()

    sec_keys["evidence_id"] = (
        sec_keys["evidence_id"].astype(str)
    )

    sec_keys = (
        sec_keys.groupby(
            ["ticker", "available_from", "evidence_id"],
            dropna=False
        )
        .agg(
            sec_status=(
                "quality_status",
                lambda x: (
                    "QUARANTINE"
                    if "QUARANTINE" in set(x)
                    else "REVIEW"
                    if "REVIEW" in set(x)
                    else "PROVISIONAL"
                )
            ),
            sec_reasons=(
                "quality_reasons",
                lambda x: "|".join(
                    sorted(set(
                        str(s) for s in x
                        if str(s) not in ("", "nan")
                    ))
                )
            ),
            matched_rows=("quality_status", "size"),
        )
        .reset_index()
    )

    # Use a simple merge and preserve events even when
    # source-ID mapping does not match.
    events["source_id"] = (
        events["source_id"].astype(str)
    )

    result = events.merge(
        sec_keys,
        left_on=[
            "ticker", "source_time", "source_id"
        ],
        right_on=[
            "ticker", "available_from", "evidence_id"
        ],
        how="left",
        validate="many_to_one",
    )

    is_sec = result["source"] == "SEC"
    matched = is_sec & result["sec_status"].notna()

    result.loc[
        matched, "quality_status"
    ] = result.loc[matched, "sec_status"]

    result.loc[
        matched, "quality_reason"
    ] = result.loc[matched, "sec_reasons"]

    result.loc[
        is_sec & ~matched, "quality_reason"
    ] = "sec_source_id_or_date_unresolved"

    result.loc[
        matched, "source_row_match_count"
    ] = result.loc[matched, "matched_rows"].astype(int)

    # Never promote KG rows to verified simply because
    # relation label is readable.
    # All PROVISIONAL records require external check.
    result["eligible_for_final_pit_claim"] = False

    # These records may be considered for a later
    # sensitivity analysis, NOT certified PIT tests.
    result["candidate_for_reviewed_replay"] = (
        result["quality_status"] == "PROVISIONAL"
    )

    result = result.drop(
        columns=[
            "available_from",
            "evidence_id",
            "sec_status",
            "sec_reasons",
            "matched_rows",
        ],
        errors="ignore",
    )

    result.to_csv(
        out / "event_quality_candidates.csv",
        index=False,
    )

    report = {
        "available": True,
        "events": len(result),
        "sec_event_match_count": int(matched.sum()),
        "sec_event_unmatched": int(
            (is_sec & ~matched).sum()
        ),
        "status_counts": summarize_flag(
            result, "quality_status"
        ),
        "candidate_for_reviewed_replay": int(
            result["candidate_for_reviewed_replay"].sum()
        ),
        "verified_pit_events": 0,
    }

    log("[EVENT QUALITY]", report)
    return report


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root", type=Path, default=Path(".")
    )
    p.add_argument(
        "--documents",
        default="outputs/v31/evidence_alignment.parquet"
    )
    p.add_argument(
        "--temporal-kg",
        default=(
            "outputs/unified_temporal_kg_v3/"
            "unified_temporal_edges.csv"
        )
    )
    p.add_argument(
        "--v11b-memory",
        default=(
            "results/aegis_rl_v1/"
            "v11b_event_semantics/clean_event_memory.csv"
        )
    )
    p.add_argument(
        "--relation-map", default=""
    )
    p.add_argument(
        "--out", type=Path,
        default=Path(
            "results/aegis_rl_v1/v11c_provenance"
        )
    )

    args = p.parse_args()

    root = args.root.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    sec, sec_summary = sec_audit(
        root / args.documents, args.out
    )

    relation_candidates = relation_map_search(
        root, args.relation_map, args.out
    )

    _, kg_summary = kg_audit(
        root / args.temporal_kg, args.out
    )

    event_summary = event_quality(
        root / args.v11b_memory, sec, args.out
    )

    summary = {
        "audit": "FinRL-X V11-C",
        "scope": "SEC/KG data provenance",
        "sec": sec_summary,
        "kg": kg_summary,
        "relation_mapping_candidates": len(
            relation_candidates
        ),
        "v11b_event_quality": event_summary,
        "public_release_verified": False,
        "outcome_labels_used": False,
        "validation_2021_used": False,
        "test_2022_2023_used": False,
        "interpretation": (
            "PROVISIONAL means no immediate internal "
            "date contradiction was detected; it does "
            "not establish point-in-time public availability."
        ),
    }

    (args.out / "audit_summary.json").write_text(
        json.dumps(
            summary, indent=2, ensure_ascii=False,
            default=str
        ),
        encoding="utf-8",
    )

    lines = [
        "FinRL-X V11-C PROVENANCE AUDIT",
        "=" * 65,
        "",
        "SEC:",
        json.dumps(
            sec_summary, indent=2, ensure_ascii=False
        ),
        "",
        "TEMPORAL KG:",
        json.dumps(
            kg_summary, indent=2, ensure_ascii=False
        ),
        "",
        "EVENT QUALITY:",
        json.dumps(
            event_summary, indent=2, ensure_ascii=False
        ),
        "",
        "RELATION MAPPING CANDIDATES:",
        str(len(relation_candidates)),
        "",
        "CRITICAL:",
        "No rows are externally verified as PIT-safe.",
        "No investment performance was evaluated.",
    ]

    (args.out / "audit_report.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    log()
    log("=" * 65)
    log("V11-C PROVENANCE AUDIT FINISHED")
    log("=" * 65)

    log("[SEC QUALITY]", sec_summary["quality_status"])
    log("[KG RELATIONS]", kg_summary["relations"])
    log(
        "[RELATION MAPPING CANDIDATES]",
        len(relation_candidates)
    )
    log("[EVENT QUALITY]", event_summary)
    log("[OUTPUT]", args.out)
    log("PIT VERIFIED: NO (requires original sources)")


if __name__ == "__main__":
    main()
