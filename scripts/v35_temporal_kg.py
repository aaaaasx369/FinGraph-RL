#!/usr/bin/env python3
"""
V35: SEC-grounded Silver Temporal Knowledge Graph.

Commands:
  python scripts/v35_temporal_kg.py build
  python scripts/v35_temporal_kg.py audit
  python scripts/v35_temporal_kg.py report

Conservative timing:
  SEC evidence is eligible only when trading_date > available_from.

IMPORTANT:
  This is a conservative timing safeguard, NOT formal PIT certification.
  AI silver edges are NOT human-verified ground truth.

Outputs:
  evidence_ledger.csv
  temporal_edges.csv
  daily_kg_features.csv
  v20_kg_bridge.csv
  graph_events.jsonl
  report.json
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v35"
EVIDENCE_PATH = ROOT / "outputs/v34/relation_evidence.csv"
MARKET_PATH = ROOT / "outputs/v20/fusion/dataset_conservative.csv"

ISSUERS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.",
}
CANONICAL_TO_TICKER = {v: k for k, v in ISSUERS.items()}
TICKERS = list(ISSUERS)

FEATURES = [
    "kg_known_competitors",
    "kg_competitor_mentions",
    "kg_recent_mentions_30d",
    "kg_recent_mentions_90d",
    "kg_new_edges_30d",
    "kg_new_edges_90d",
    "kg_days_since_last_disclosure",
    "kg_days_since_first_disclosure",
    "kg_portfolio_competitors",
    "kg_external_competitors",
    "kg_competitor_remention_ratio",
]

def sha(*parts):
    return hashlib.sha256(
        "\x1f".join(map(str, parts)).encode("utf-8")
    ).hexdigest()[:24]

def fail(message):
    raise RuntimeError(message)

def as_date(s):
    return pd.Timestamp(s).normalize()

def locate_silver(run):
    p = ROOT / "outputs/v341/runs" / run / "silver_relations.csv"
    if not p.is_file():
        fail("Silver relations not found: " + str(p))
    return p

def detect_market_columns(df):
    date_candidates = ["date", "Date", "datetime", "timestamp"]
    ticker_candidates = ["tic", "ticker", "symbol", "Ticker"]

    dates = [x for x in date_candidates if x in df.columns]
    tickers = [x for x in ticker_candidates if x in df.columns]

    if not dates or not tickers:
        fail(
            "Cannot identify V20 date/ticker columns. Got: "
            + ", ".join(df.columns)
        )

    return dates[0], tickers[0]

def load_market():
    if not MARKET_PATH.is_file():
        fail("Missing V20 dataset: " + str(MARKET_PATH))

    df = pd.read_csv(MARKET_PATH)
    dc, tc = detect_market_columns(df)

    df["_date"] = pd.to_datetime(
        df[dc], errors="raise"
    ).dt.normalize()
    df["_ticker"] = df[tc].astype(str).str.upper()

    if df["_date"].isna().any():
        fail("Invalid V20 market dates")

    unexpected = set(df["_ticker"]) - set(TICKERS)
    if unexpected:
        fail("Unexpected trading tickers: " + str(unexpected))

    if df.duplicated(["_date", "_ticker"]).any():
        fail("Duplicate market ticker-day rows")

    dates = sorted(df["_date"].unique())

    expected = pd.MultiIndex.from_product(
        [dates, TICKERS], names=["_date", "_ticker"]
    )
    actual = pd.MultiIndex.from_frame(
        df[["_date", "_ticker"]]
    )

    if len(df) != len(expected) or not actual.is_unique:
        fail("V20 does not contain exactly five assets per date")

    if set(actual) != set(expected):
        fail("Missing some date-ticker combinations")

    return df, dates, dc, tc

def source_evidence_lookup():
    src = pd.read_csv(
        EVIDENCE_PATH, dtype={"accession": str}
    ).fillna("")

    if src["evidence_id"].duplicated().any():
        fail("Duplicate V34 evidence IDs")

    return src.set_index("evidence_id")

def make_ledger(silver, source):
    required = {
        "candidate_id", "evidence_id",
        "ticker", "subject", "relation", "object",
        "filing_date", "available_from",
        "accession", "source_sha256",
        "offset_start", "offset_end",
        "evidence_quote", "accepted_silver"
    }

    missing = required - set(silver.columns)
    if missing:
        fail("Missing silver columns: " + str(sorted(missing)))

    if silver["candidate_id"].duplicated().any():
        fail("Duplicate silver candidate IDs")

    result = []

    for row in silver.to_dict("records"):
        ticker = str(row["ticker"])
        eid = str(row["evidence_id"])

        if ticker not in ISSUERS:
            fail("Unsupported SEC issuer: " + ticker)
        if row["subject"] != ISSUERS[ticker]:
            fail("SEC issuer mismatch: " + eid)
        if row["relation"] != "COMPETES_WITH":
            fail("V35 currently supports COMPETES_WITH only")
        if eid not in source.index:
            fail("Missing source evidence: " + eid)

        src = source.loc[eid]
        quote = str(row["evidence_quote"])
        sentence = str(src["sentence"])

        if not isinstance(row["accepted_silver"], (bool, int)):
            if str(row["accepted_silver"]).lower() != "true":
                fail("Non-accepted relation in silver file")

        if str(row["accepted_silver"]).lower() not in ("true", "1"):
            fail("Non-accepted silver candidate: " + eid)

        if len(quote.strip()) < 12 or quote not in sentence:
            fail("Invalid exact SEC quote: " + eid)

        checks = {
            "ticker": ticker,
            "accession": str(row["accession"]),
            "source_sha256": str(row["source_sha256"]),
            "offset_start": int(row["offset_start"]),
            "offset_end": int(row["offset_end"]),
            "filing_date": str(row["filing_date"])[:10],
            "available_from": str(row["available_from"])[:10],
        }

        for key, expected in checks.items():
            value = src[key]
            if key in ("offset_start", "offset_end"):
                value = int(value)
            else:
                value = str(value)[:10] if key.endswith("date") or key == "available_from" else str(value)
            if value != expected:
                fail(f"Source mismatch {eid} field={key}")

        filing_date = as_date(row["filing_date"])
        available = as_date(row["available_from"])

        if available <= filing_date:
            fail("Invalid filing / availability chronology")

        target = str(row["object"]).strip()
        if not target or target == ISSUERS[ticker]:
            fail("Invalid edge target")

        relation_id = sha(
            ISSUERS[ticker], "COMPETES_WITH", target
        )

        result.append({
            "candidate_id": str(row["candidate_id"]),
            "evidence_id": eid,
            "relation_id": relation_id,
            "ticker": ticker,
            "subject": ISSUERS[ticker],
            "relation": "COMPETES_WITH",
            "object": target,
            "target_ticker": CANONICAL_TO_TICKER.get(target, ""),
            "filing_date": filing_date.date().isoformat(),
            "available_from": available.date().isoformat(),
            "accession": str(row["accession"]),
            "source_sha256": str(row["source_sha256"]),
            "offset_start": int(row["offset_start"]),
            "offset_end": int(row["offset_end"]),
            "sentence": sentence,
            "evidence_quote": quote,
            "market_scope": (
                ""
                if pd.isna(row.get("market_scope"))
                else str(row.get("market_scope", ""))
            ),
            "model": str(row.get("model", "")),
            "label_source": "AI_SILVER",
            "human_verified": False,
            "strict_pit_approved": False,
            "scope_verified": False,
        })

    ledger = pd.DataFrame(result)

    if ledger.empty:
        fail("No accepted relations in selected V34.1 run")

    return ledger.sort_values(
        ["available_from", "ticker", "relation_id", "candidate_id"]
    ).reset_index(drop=True)

def create_temporal_edges(ledger):
    rows = []

    for (ticker, relation_id), g in ledger.groupby(
        ["ticker", "relation_id"], sort=True
    ):
        g = g.sort_values(
            ["available_from", "candidate_id"]
        )

        rows.append({
            "relation_id": relation_id,
            "ticker": ticker,
            "subject": g.iloc[0]["subject"],
            "relation": g.iloc[0]["relation"],
            "object": g.iloc[0]["object"],
            "target_ticker": g.iloc[0]["target_ticker"],
            "first_disclosed": g.iloc[0]["available_from"],
            "last_disclosed": g.iloc[-1]["available_from"],
            "evidence_mentions": len(g),
            "distinct_filings": g["accession"].nunique(),
            "evidence_ids": json.dumps(
                g["evidence_id"].drop_duplicates().tolist()
            ),
            "accessions": json.dumps(
                g["accession"].drop_duplicates().tolist()
            ),
            "human_verified": False,
            "strict_pit_approved": False,
        })

    return pd.DataFrame(rows)

def build_daily(ledger, market_dates):
    evidence = ledger.copy()
    evidence["_available"] = pd.to_datetime(
        evidence["available_from"]
    )

    result = []
    milestones = {}

    for ticker in TICKERS:
        # For each stock, include relations disclosed either
        # by that company OR about that company by another
        # SEC issuer, while keeping separate known provenance.
        own = evidence[evidence["ticker"] == ticker].copy()

        for date in market_dates:
            date = as_date(date)

            # CRITICAL: strictly later trading DATE only.
            # Never include same-day filing information.
            past = own[own["_available"] < date]

            if past.empty:
                stats = dict.fromkeys(FEATURES, 0.0)
                stats["kg_days_since_last_disclosure"] = 0.0
                stats["kg_days_since_first_disclosure"] = 0.0
            else:
                first_by_relation = past.groupby(
                    "relation_id"
                )["_available"].min()

                recent30 = past[
                    past["_available"] >= date - pd.Timedelta(days=30)
                ]
                recent90 = past[
                    past["_available"] >= date - pd.Timedelta(days=90)
                ]

                new30 = (
                    first_by_relation >= date - pd.Timedelta(days=30)
                ).sum()
                new90 = (
                    first_by_relation >= date - pd.Timedelta(days=90)
                ).sum()

                relation_targets = past.drop_duplicates(
                    "relation_id"
                )

                portfolio = (
                    relation_targets["target_ticker"].astype(str) != ""
                ).sum()

                total = len(relation_targets)
                mentions = len(past)

                stats = {
                    "kg_known_competitors": float(total),
                    "kg_competitor_mentions": float(mentions),
                    "kg_recent_mentions_30d": float(len(recent30)),
                    "kg_recent_mentions_90d": float(len(recent90)),
                    "kg_new_edges_30d": float(new30),
                    "kg_new_edges_90d": float(new90),
                    "kg_days_since_last_disclosure": float(
                        (date - past["_available"].max()).days
                    ),
                    "kg_days_since_first_disclosure": float(
                        (date - past["_available"].min()).days
                    ),
                    "kg_portfolio_competitors": float(portfolio),
                    "kg_external_competitors": float(total - portfolio),
                    "kg_competitor_remention_ratio": float(
                        (mentions - total) / max(mentions, 1)
                    ),
                }

            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "tic": ticker,
                **stats,
                "kg_data_status": "AI_SILVER_COMPETITOR_ONLY",
                "kg_strict_pit_approved": False,
            })

    return pd.DataFrame(result).sort_values(
        ["date", "tic"]
    ).reset_index(drop=True)

def write_events(ledger):
    output = OUT / "graph_events.jsonl"
    with output.open("w", encoding="utf-8") as f:
        for row in ledger.to_dict("records"):
            event = {
                "event_type": "RELATION_DISCLOSURE",
                "relation_id": row["relation_id"],
                "evidence_id": row["evidence_id"],
                "available_from": row["available_from"],
                "subject": row["subject"],
                "relation": row["relation"],
                "object": row["object"],
                "accession": row["accession"],
                "source_sha256": row["source_sha256"],
                "evidence_quote": row["evidence_quote"],
                "human_verified": False,
                "strict_pit_approved": False,
            }
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

def build(args):
    OUT.mkdir(parents=True, exist_ok=True)

    silver_file = locate_silver(args.run_name)

    silver = pd.read_csv(
        silver_file,
        dtype={"accession": str}
    )
    source = source_evidence_lookup()

    ledger = make_ledger(silver, source)
    temporal = create_temporal_edges(ledger)

    market, dates, dc, tc = load_market()
    daily = build_daily(ledger, dates)

    # Preserve original V20 columns while attaching V35 features.
    bridge = market.merge(
        daily,
        left_on=["_date", "_ticker"],
        right_on=[
            pd.to_datetime(daily["date"]),
            daily["tic"]
        ],
        how="left",
        validate="one_to_one",
        suffixes=("", "_v35"),
    )

    if len(bridge) != len(market):
        fail("Market bridge changed the number of rows")

    if bridge[FEATURES].isna().any().any():
        fail("Missing daily KG values after market merge")

    # Drop only temporary helpers and duplicate join keys.
    bridge = bridge.drop(
        columns=["_date", "_ticker"],
        errors="ignore"
    )

    ledger.to_csv(OUT / "evidence_ledger.csv", index=False)
    temporal.to_csv(OUT / "temporal_edges.csv", index=False)
    daily.to_csv(OUT / "daily_kg_features.csv", index=False)
    bridge.to_csv(OUT / "v20_kg_bridge.csv", index=False)
    write_events(ledger)

    summary = {
        "silver_run": args.run_name,
        "evidence_mentions": len(ledger),
        "unique_directed_edges": len(temporal),
        "daily_feature_rows": len(daily),
        "market_bridge_rows": len(bridge),
        "portfolio_tickers": TICKERS,
        "feature_columns": FEATURES,
        "relation_types": ["COMPETES_WITH"],
        "temporal_policy": "trading_date > available_from",
        "human_verified": False,
        "strict_pit_approved": False,
        "note": (
            "Filing date + 1 calendar day is a provisional "
            "availability proxy, not audited SEC acceptance time."
        ),
    }

    (OUT / "report.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    print("\n=== V35 BUILD COMPLETE ===")
    print(json.dumps(summary, indent=2))

def audit():
    ledger = pd.read_csv(OUT / "evidence_ledger.csv")
    edges = pd.read_csv(OUT / "temporal_edges.csv")
    daily = pd.read_csv(OUT / "daily_kg_features.csv")
    bridge = pd.read_csv(OUT / "v20_kg_bridge.csv")

    market, dates, _, _ = load_market()

    if len(daily) != len(market):
        fail("Daily panel row count mismatch")
    if len(bridge) != len(market):
        fail("Bridge row count mismatch")
    if daily.duplicated(["date", "tic"]).any():
        fail("Duplicate daily KG rows")
    if (daily[FEATURES] < 0).any().any():
        fail("Negative KG feature value")

    # Recompute independently from evidence at sampled
    # and boundary dates to detect accidental future use.
    ledger["_available"] = pd.to_datetime(
        ledger["available_from"]
    )
    daily["_date"] = pd.to_datetime(daily["date"])

    for ticker in TICKERS:
        part = ledger[ledger["ticker"] == ticker]
        panel = daily[daily["tic"] == ticker]

        for row in panel.itertuples(index=False):
            date = row._date if hasattr(row, "_date") else pd.Timestamp(row.date)
            known = part[part["_available"] < date]

            actual_mentions = int(row.kg_competitor_mentions)
            actual_edges = int(row.kg_known_competitors)

            if actual_mentions != len(known):
                fail(f"Temporal mentions leak: {ticker} {date}")

            if actual_edges != known["relation_id"].nunique():
                fail(f"Temporal edges leak: {ticker} {date}")

    if not set(ledger["ticker"]).issubset(ISSUERS):
        fail("Unexpected SEC issuer")
    if not set(daily["tic"]).issubset(ISSUERS):
        fail("Unexpected trading ticker")

    if ledger["human_verified"].astype(str).str.lower().eq("true").any():
        fail("Silver mislabeled as human verified")

    if daily["kg_strict_pit_approved"].astype(str).str.lower().eq("true").any():
        fail("Uncertified PIT incorrectly approved")

    print("\n=== V35 AUDIT PASS ===")
    print("SEC silver evidence:", len(ledger))
    print("Unique directed graph edges:", len(edges))
    print("Daily portfolio rows:", len(daily))
    print("V20 merged rows:", len(bridge))
    print("Future-evidence temporal test: PASS")
    print("Five tradeable tickers: PASS")
    print("Human verified: NO")
    print("Strict PIT certified: NO")

def report():
    p = OUT / "report.json"
    if not p.exists():
        fail("Run build first")

    print(p.read_text(encoding="utf-8"))

    edges = pd.read_csv(OUT / "temporal_edges.csv")
    print("\n=== TEMPORAL EDGES ===")
    print(
        edges[[
            "subject", "relation", "object",
            "first_disclosed", "last_disclosed",
            "evidence_mentions"
        ]].to_string(index=False)
    )

    daily = pd.read_csv(OUT / "daily_kg_features.csv")
    print("\n=== LATEST DAILY KG FEATURES ===")
    latest = daily["date"].max()
    print("Date:", latest)
    print(
        daily.loc[
            daily["date"] == latest,
            ["tic",
             "kg_known_competitors",
             "kg_competitor_mentions",
             "kg_recent_mentions_90d",
             "kg_portfolio_competitors",
             "kg_external_competitors"]
        ].to_string(index=False)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command", choices=["build", "audit", "report"]
    )
    ap.add_argument(
        "--run-name", default="pilot_27b"
    )
    args = ap.parse_args()

    if args.command == "build":
        build(args)
    elif args.command == "audit":
        audit()
    else:
        report()

if __name__ == "__main__":
    main()
