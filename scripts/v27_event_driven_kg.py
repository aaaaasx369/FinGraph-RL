#!/usr/bin/env python3
"""
V27: SEC Filing Event-Driven + Time-Aware Concept KG

Fixed portfolio universe:
    AAPL / MSFT / NVDA / AMZN / GOOGL

Inputs:
    outputs/v26/filing_features.csv
    outputs/v26/filing_evidence.parquet
    outputs/v26/source_audit.csv
    kg/data/market_5stocks_2017_2026.csv
    outputs/v20/fusion/dataset_{tier}.csv
    outputs/v20/fusion/feature_sets.json

Commands:
    build   : filing change events + daily as-of KG panel
    bridge  : V21-compatible 11-dimensional dynamic arm
    audit   : provenance, temporal and five-stock checks
    all     : build + audit + bridge

Important:
    - No invented filing events.
    - Do not equate NOT_REMENTIONED with termination.
    - Strictly earlier filing date for each trading day.
    - SEC local hash check only certifies cache consistency.
    - No assertion of full PIT/semantic verification.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v27"
V26 = ROOT / "outputs/v26"
V20 = ROOT / "outputs/v20/fusion"

STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
TIERS = ["conservative", "inclusive"]

TOPICS = [
    "supply_chain",
    "manufacturing",
    "cloud",
    "competition",
    "regulation",
    "litigation",
    "finance",
    "research",
]

FORMS = ["10-K", "10-Q"]

# Fixed ex ante. Do not optimize on 2023 test results.
HALF_LIFE_DAYS = 180.0

def log_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False,
                   default=str),
        encoding="utf-8",
    )

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()

def dt(series):
    return pd.to_datetime(series, errors="raise").dt.normalize()

def load_inputs():
    files = [
        V26 / "filing_features.csv",
        V26 / "filing_evidence.parquet",
        V26 / "source_audit.csv",
    ]
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(p)

    filings = pd.read_csv(
        files[0], dtype={"accession": str}
    )
    evidence = pd.read_parquet(files[1])
    audit = pd.read_csv(
        files[2], dtype={"accession": str}
    )

    required = [
        "ticker", "accession", "form",
        "filing_date", "available_from",
    ] + ["sec_" + t for t in TOPICS]

    missing = [c for c in required if c not in filings]
    if missing:
        raise ValueError("V26 missing columns: " + str(missing))

    filings["ticker"] = filings["ticker"].astype(str).str.upper()
    filings["form"] = filings["form"].astype(str).str.upper()
    filings["filing_date"] = dt(filings["filing_date"])
    filings["available_from"] = dt(filings["available_from"])

    filings = filings[
        filings["ticker"].isin(STOCKS)
        & filings["form"].isin(FORMS)
    ].copy()

    if filings.duplicated(["ticker", "accession"]).any():
        raise ValueError("Duplicate ticker/accession")

    if not (filings["available_from"] >
            filings["filing_date"]).all():
        raise ValueError("Unsafe filing availability date")

    if len(filings) != 140:
        raise ValueError(
            f"Expected 140 SEC filings, got {len(filings)}"
        )

    audit["ticker"] = audit["ticker"].astype(str).str.upper()
    if audit.duplicated(["ticker", "accession"]).any():
        raise ValueError("Duplicate SEC cache audit keys")

    if not {"included", "sha256_matches_manifest"}.issubset(
        audit.columns
    ):
        raise ValueError("V26 SEC audit columns missing")

    # Only verified local cache entries may enter the graph.
    good = audit[
        audit["included"].astype(str).str.lower().isin(
            ["true", "1"]
        )
        & audit["sha256_matches_manifest"]
        .astype(str).str.lower().isin(["true", "1"])
    ][["ticker", "accession"]].drop_duplicates()

    checked = filings.merge(
        good, on=["ticker", "accession"],
        how="inner", validate="one_to_one"
    )
    if len(checked) != len(filings):
        raise ValueError("Unverified/missing SEC cached filing")

    if "event_id" not in evidence or evidence["event_id"].duplicated().any():
        raise ValueError("Invalid V26 evidence IDs")

    # Five-company market panel.
    market = pd.read_csv(
        ROOT / "kg/data/market_5stocks_2017_2026.csv"
    )
    if "ticker" not in market and "tic" in market:
        market = market.rename(columns={"tic": "ticker"})
    market["date"] = dt(market["date"])
    market["ticker"] = market["ticker"].astype(str).str.upper()
    market = market[
        market["ticker"].isin(STOCKS)
        & market["date"].between(
            "2018-01-01", "2023-12-31"
        )
    ][["date", "ticker"]].drop_duplicates()

    if set(market["ticker"]) != set(STOCKS):
        raise ValueError("Wrong market universe")
    if len(market) != 7545:
        raise ValueError(f"Expected 7545 stock-days, got {len(market)}")
    if not (market.groupby("ticker").size() == 1509).all():
        raise ValueError("Unequal market history")

    # All count values must be nonnegative, finite numbers.
    for t in TOPICS:
        col = "sec_" + t
        filings[col] = pd.to_numeric(
            filings[col], errors="raise"
        ).astype(float)
        if (
            not np.isfinite(filings[col]).all()
            or (filings[col] < 0).any()
        ):
            raise ValueError("Invalid SEC concept count: " + col)

    return (
        filings.sort_values(
            ["ticker", "available_from", "accession"]
        ).reset_index(drop=True),
        evidence,
        market.sort_values(
            ["ticker", "date"]
        ).reset_index(drop=True),
    )


def build_events(filings):
    rows = []

    for ticker in STOCKS:
        for form in FORMS:
            sub = filings[
                filings["ticker"].eq(ticker)
                & filings["form"].eq(form)
            ].sort_values(
                ["available_from", "accession"]
            )

            previous = None

            for _, f in sub.iterrows():
                for topic in TOPICS:
                    current = float(f["sec_" + topic])
                    old = (
                        float(previous["sec_" + topic])
                        if previous is not None else None
                    )

                    if old is None:
                        status = "FIRST_OBSERVATION"
                    elif old <= 0 and current > 0:
                        status = "NEW_MENTION"
                    elif old > 0 and current > 0:
                        status = "REPEATED_MENTION"
                    elif old > 0 and current <= 0:
                        status = "NOT_REMENTIONED"
                    else:
                        status = "STILL_NOT_MENTIONED"

                    rows.append({
                        "ticker": ticker,
                        "accession": str(f["accession"]),
                        "form": form,
                        "filing_date": f["filing_date"],
                        "available_from": f["available_from"],
                        "concept": topic,
                        "event_type": status,
                        "current_mentions": current,
                        "previous_same_form_mentions": old,
                        "mention_count_delta": (
                            current - old
                            if old is not None else np.nan
                        ),
                        "has_prior_same_form": old is not None,
                        "is_mention_event": current > 0,
                        "relationship_termination_inferred": False,
                        "relation_direction_verified": False,
                    })

                previous = f

    events = pd.DataFrame(rows)
    if len(events) != len(filings) * len(TOPICS):
        raise ValueError("Filing/concept coverage mismatch")
    if events.duplicated(
        ["ticker", "accession", "concept"]
    ).any():
        raise ValueError("Duplicate filing-concept events")
    return events


def build_daily(filings, market):
    """
    Maintain latest 10-K and 10-Q separately.

    Feature 'known' means presence in most recent filing
    of either form, not a legally verified active relation.
    'age' uses latest positive mention across known filings.
    """
    all_rows = []
    ln2 = math.log(2.0)

    for ticker in STOCKS:
        company = filings[
            filings["ticker"].eq(ticker)
        ].sort_values(
            ["available_from", "accession"]
        ).to_dict("records")

        days = market.loc[
            market["ticker"].eq(ticker), "date"
        ].sort_values().tolist()

        pointer = 0
        latest_by_form = {}
        last_positive = {}
        last_filing = None
        known_count = 0

        for day in days:
            new_filings = 0

            while (
                pointer < len(company)
                and company[pointer]["available_from"] <= day
            ):
                doc = company[pointer]
                if not doc["filing_date"] < day:
                    raise AssertionError(
                        "Same-day filing allowed into trade panel"
                    )

                latest_by_form[doc["form"]] = doc
                last_filing = doc
                known_count += 1
                new_filings += 1

                for topic in TOPICS:
                    if float(doc["sec_" + topic]) > 0:
                        last_positive[topic] = doc["available_from"]

                pointer += 1

            record = {
                "date": day,
                "ticker": ticker,
                "v27_has_filing": int(known_count > 0),
                "v27_known_filing_count": known_count,
                "v27_new_filings": new_filings,
                "v27_has_new_filing": int(new_filings > 0),
                "v27_latest_accession": (
                    str(last_filing["accession"])
                    if last_filing is not None else ""
                ),
                "v27_latest_filing_date": (
                    last_filing["filing_date"]
                    if last_filing is not None else pd.NaT
                ),
                "v27_days_since_filing": (
                    int((day - last_filing["available_from"]).days)
                    if last_filing is not None else -1
                ),
            }

            for topic in TOPICS:
                mentioned = any(
                    float(doc["sec_" + topic]) > 0
                    for doc in latest_by_form.values()
                )
                age = (
                    (day - last_positive[topic]).days
                    if topic in last_positive else -1
                )
                score = (
                    math.exp(-ln2 * age / HALF_LIFE_DAYS)
                    if age >= 0 else 0.0
                )
                record[f"v27_{topic}_known"] = int(mentioned)
                record[f"v27_{topic}_age_days"] = int(age)
                record[f"v27_{topic}_decay"] = (
                    score if mentioned else 0.0
                )

            all_rows.append(record)

    daily = pd.DataFrame(all_rows).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if len(daily) != 7545:
        raise ValueError("Incorrect daily row count")

    if daily.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker")

    # Verify actual underlying date, not just a name
    # like 'as_of', and exclude same-day data.
    dates = pd.to_datetime(daily["date"])
    last = pd.to_datetime(daily["v27_latest_filing_date"])
    has = last.notna()
    if not (last.loc[has] < dates.loc[has]).all():
        raise ValueError("Future or same-day filing exposure")

    for topic in TOPICS:
        x = daily[f"v27_{topic}_decay"].to_numpy(dtype=float)
        if (x < 0).any() or (x > 1).any() or not np.isfinite(x).all():
            raise ValueError("Invalid time decay")
    return daily


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    filings, evidence, market = load_inputs()

    events = build_events(filings)
    daily = build_daily(filings, market)

    events.to_parquet(
        OUT / "filing_change_events.parquet",
        index=False
    )
    events.to_csv(
        OUT / "filing_change_events.csv",
        index=False
    )
    daily.to_csv(
        OUT / "daily_temporal_features.csv",
        index=False
    )

    # Preserve original V26 evidence IDs and offsets
    # instead of silently inventing new evidence.
    evidence[[
        "event_id", "ticker", "accession",
        "concept", "matched_term", "offset",
        "source_sha256", "excerpt"
    ]].to_parquet(
        OUT / "source_evidence_index.parquet",
        index=False
    )

    report = {
        "universe": STOCKS,
        "SEC_source_filings": int(len(filings)),
        "filing_concept_comparisons": int(len(events)),
        "event_type_counts": {
            str(k): int(v)
            for k, v in events["event_type"].value_counts().items()
        },
        "stock_days": int(len(daily)),
        "filings_entering_test_window": int(
            daily.loc[
                daily["date"].between(
                    pd.Timestamp("2023-01-01"),
                    pd.Timestamp("2023-12-31")
                ),
                "v27_new_filings"
            ].sum()
        ),
        "update_stock_days": int(
            daily["v27_has_new_filing"].sum()
        ),
        "update_stock_days_by_ticker": {
            t: int(
                daily.loc[
                    daily["ticker"].eq(t),
                    "v27_has_new_filing"
                ].sum()
            )
            for t in STOCKS
        },
        "time_half_life_days": HALF_LIFE_DAYS,
        "concept_mention_is_not_relationship_truth": True,
        "strict_pit_approved": False,
        "finrl_execution_audit_complete": False,
    }
    log_json(OUT / "build_report.json", report)

    print("\n=== V27 EVENT-DRIVEN KG BUILD ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", OUT)


def bridge():
    path = OUT / "daily_temporal_features.csv"
    if not path.is_file():
        raise FileNotFoundError("Run build first")

    graph = pd.read_csv(path)
    graph["date"] = dt(graph["date"])
    graph["ticker"] = graph["ticker"].astype(str).str.upper()

    schema = json.loads(
        (V20 / "feature_sets.json").read_text()
    )

    full_static = schema["FULL_STATIC"]
    full_dynamic = schema["FULL_DYNAMIC"]
    dynamic_columns = [
        c for c in full_dynamic if c not in full_static
    ]

    if len(dynamic_columns) != 11:
        raise ValueError(
            "Expected V21 11 dynamic columns"
        )

    # Eight time-aware concept strengths
    # + filing recency + new filing + evidence mask.
    features = [
        f"v27_{t}_decay" for t in TOPICS
    ] + [
        "v27_days_since_filing",
        "v27_has_new_filing",
        "v27_has_filing",
    ]
    if len(features) != 11:
        raise AssertionError("Expected 11 features")

    dest = OUT / "finrl"
    dest.mkdir(parents=True, exist_ok=True)
    log_json(dest / "feature_sets.json", schema)

    bridge_report = {}

    for tier in TIERS:
        base_path = V20 / f"dataset_{tier}.csv"
        if not base_path.is_file():
            raise FileNotFoundError(base_path)

        df = pd.read_csv(base_path)
        df["date"] = dt(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper()

        if df.duplicated(["date", "ticker"]).any():
            raise ValueError("Duplicate V20 keys")
        if set(df["ticker"]) != set(STOCKS):
            raise ValueError("Unexpected portfolio universe")

        df = df.drop(columns=dynamic_columns)
        merged = df.merge(
            graph[["date", "ticker"] + features],
            on=["date", "ticker"], how="left",
            validate="one_to_one", indicator=True
        )
        if len(merged) != 7545 or not merged["_merge"].eq("both").all():
            raise ValueError("V27 graph panel missing rows")
        merged.drop(columns=["_merge"], inplace=True)

        train = merged["split"].eq("train")
        if not train.any():
            raise ValueError("No training split")

        scalers = {}

        for raw, model_col in zip(features, dynamic_columns):
            series = pd.to_numeric(
                merged[raw], errors="raise"
            ).astype(float)

            if raw == "v27_days_since_filing":
                # -1 means unknown; reserve 0 raw
                # and use the explicit filing mask.
                series = np.log1p(series.clip(lower=0))

            if not np.isfinite(series).all():
                raise ValueError(f"Nonfinite {raw}")

            mean = float(series[train].mean())
            std = float(series[train].std(ddof=0))
            if not np.isfinite(std) or std < 1e-12:
                std = 1.0

            merged[model_col] = (series - mean) / std

            scalers[model_col] = {
                "v27_source": raw,
                "train_mean": mean,
                "train_std": std,
            }

        for cond, cols in schema.items():
            missing = [c for c in cols if c not in merged]
            if missing:
                raise ValueError(f"{cond} missing: {missing}")
            if not np.isfinite(
                merged[cols].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"{cond} nonfinite features")

        merged.sort_values(
            ["date", "ticker"], inplace=True
        )
        merged.to_csv(
            dest / f"dataset_{tier}.csv",
            index=False
        )
        log_json(dest / f"scaler_{tier}.json", scalers)

        bridge_report[tier] = {
            "rows": len(merged),
            "tickers": STOCKS,
            "full_static_dimensions": len(full_static),
            "full_dynamic_dimensions": len(full_dynamic),
            "replacement_graph_features": features,
            "strict_pit_approved": False,
        }

    log_json(dest / "report.json", bridge_report)
    print("\n=== V27 FINRL BRIDGE ===")
    print(json.dumps(bridge_report, indent=2))
    print("Saved:", dest)


def audit():
    events_path = OUT / "filing_change_events.parquet"
    daily_path = OUT / "daily_temporal_features.csv"
    if not events_path.is_file() or not daily_path.is_file():
        raise FileNotFoundError("Run build first")

    events = pd.read_parquet(events_path)
    daily = pd.read_csv(daily_path)
    daily["date"] = dt(daily["date"])
    events["filing_date"] = dt(events["filing_date"])
    events["available_from"] = dt(events["available_from"])

    assert set(daily["ticker"]) == set(STOCKS)
    assert len(daily) == 7545
    assert not daily.duplicated(["date", "ticker"]).any()
    assert (events["filing_date"] <
            events["available_from"]).all()
    assert not events["relationship_termination_inferred"].any()
    assert not events["relation_direction_verified"].any()

    # Rebuild expected daily rows from actual filing dates
    # to detect accidental temporal changes to saved CSV.
    filings, _, market = load_inputs()
    recomputed = build_daily(filings, market)
    original = daily.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    compare_columns = [
        "v27_new_filings",
        "v27_has_filing",
        "v27_known_filing_count",
    ] + [
        f"v27_{t}_decay" for t in TOPICS
    ]

    for col in compare_columns:
        a = pd.to_numeric(original[col], errors="raise")
        b = pd.to_numeric(recomputed[col], errors="raise")
        if not np.allclose(
            a.to_numpy(dtype=float),
            b.to_numpy(dtype=float),
            atol=1e-9, rtol=1e-9
        ):
            raise ValueError(
                f"Temporal replay mismatch: {col}"
            )

    print("\n=== V27 AUDIT ===")
    print("Five-company universe: PASS")
    print("SEC evidence provenance: V26 cache-hash checked")
    print("Strict earlier-than-trade-date: PASS")
    print("Replayed daily state: PASS")
    print("No automatic relation termination: PASS")
    print("Strict PIT certification: NOT COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["build", "bridge", "audit", "all"]
    )
    args = parser.parse_args()

    if args.command in ["build", "all"]:
        build()
    if args.command in ["audit", "all"]:
        audit()
    if args.command in ["bridge", "all"]:
        bridge()

    if args.command == "all":
        print("\n=== V27 PIPELINE COMPLETE ===")
        print("Portfolio:", STOCKS)
        print("Strict PIT: NOT COMPLETE")


if __name__ == "__main__":
    main()
