#!/usr/bin/env python3
"""
V25 Evidence + Provenance + PIT Audit

Research universe:
    AAPL MSFT NVDA AMZN GOOGL

Uses:
    V23.1 normalized relation events
    V24 fixed-five graph observations
    V20 official SEC manifest and cached documents
    V22 candidate PDF/accession map

Produces:
    Concept-level SEC lexical evidence
    Historical KG event frequency audit
    PDF identity candidate audit
    Execution environment review
    Quality-gated historical feature panel

DOES NOT:
    Invent new KG observations
    Claim text co-mention proves a relationship
    Mark matching filing dates as PDF identity verified
    Claim strict PIT approval
    Modify V18/V21/V24 datasets
"""

import argparse
import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import timedelta
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v25"

STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
TIERS = ["conservative", "inclusive"]

CONCEPT_TERMS = {
    "supply_chain": [
        "supply chain", "supply disruption",
        "supplier", "foundry", "procurement",
        "manufacturing", "wafer", "fabrication"
    ],
    "cloud": [
        "cloud", "azure", "data center",
        "cloud infrastructure"
    ],
    "competition": [
        "competition", "competitor",
        "competitive", "market share"
    ],
    "regulation": [
        "regulation", "regulatory",
        "litigation", "antitrust",
        "export control", "sanctions"
    ],
    "finance": [
        "revenue", "cash flow",
        "accounts receivable",
        "profit", "operating income"
    ],
}

class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def parse_html(data):
    raw = data.decode("utf-8", errors="replace")
    parser = HTMLText()
    try:
        parser.feed(raw)
        return " ".join(
            " ".join(parser.parts).split()
        )
    except Exception:
        return " ".join(
            re.sub(r"<[^>]+>", " ", raw).split()
        )


def norm(s):
    if pd.isna(s):
        return ""
    return " ".join(
        re.sub(
            r"[^a-z0-9]+", " ",
            str(s).casefold()
        ).split()
    )


def clean_date(s):
    return pd.to_datetime(
        s, errors="coerce"
    ).dt.normalize()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sec_file_path(row):
    name = (
        f"{row['ticker']}_{row['accession']}_"
        + re.sub(
            r"[^A-Za-z0-9_.-]", "_",
            str(row["primary_document"])
        )
    )
    return (
        ROOT
        / "outputs/v20/sec_cache/documents"
        / name
    )


def read_manifest():
    path = (
        ROOT
        / "outputs/v20/sec_filings_manifest.csv"
    )
    m = pd.read_csv(path, dtype={"accession": str})

    needed = [
        "ticker", "accession", "form",
        "filing_date", "primary_document",
        "source_sha256", "text_verified"
    ]
    missing = [c for c in needed if c not in m]
    if missing:
        raise ValueError(
            f"SEC manifest missing: {missing}"
        )

    m["ticker"] = m["ticker"].str.upper()
    m["filing_date"] = clean_date(
        m["filing_date"]
    )
    m = m[
        m["ticker"].isin(STOCKS)
    ].copy()

    if m.duplicated(
        ["ticker", "accession"]
    ).any():
        raise ValueError(
            "Duplicate SEC accession entries"
        )

    return m


def build_sec_index(manifest):
    """
    Verify SEC cache hashes, build lexical text windows.

    The local SEC document can be verified as matching
    the stored V20 SHA256. This does not verify that
    a separately sourced PDF is the same filing.
    """
    index = {}
    audit = []

    for _, r in manifest.iterrows():
        key = (
            str(r["ticker"]),
            str(r["accession"])
        )
        path = sec_file_path(r)
        result = {
            "ticker": key[0],
            "accession": key[1],
            "form": str(r["form"]),
            "filing_date": r["filing_date"],
            "cache_found": path.is_file(),
            "hash_match": False,
            "text_indexed": False,
        }

        if not path.is_file():
            audit.append(result)
            continue

        actual = sha_file(path)
        expected = str(r["source_sha256"])

        result["hash_match"] = actual == expected

        if actual != expected:
            audit.append(result)
            continue

        if str(r["text_verified"]).lower() not in {
            "1", "true", "1.0"
        }:
            audit.append(result)
            continue

        text = parse_html(path.read_bytes())
        windows = []

        for offset in range(0, len(text), 1200):
            part = text[offset:offset + 1600]
            if len(part) >= 100:
                windows.append((offset, part))

        index[key] = {
            "ticker": key[0],
            "accession": key[1],
            "filing_date": r["filing_date"],
            "form": str(r["form"]),
            "windows": windows,
            "document_path": str(path),
            "sha256": actual,
        }

        result["text_indexed"] = bool(windows)
        audit.append(result)

    return index, pd.DataFrame(audit)


def concept_evidence(events, sec_index):
    """
    Inspect the source filing only, not a newer filing.

    NO_MATCH: no concept phrase
    CONCEPT_MENTION: phrase found in source SEC filing
    RELATION_COOCCURRENCE: concept and a compatible
        relation cue in one local text window

    None of these labels verifies the extracted triple.
    """
    out_rows = []
    cache = {}

    for r in events.itertuples(index=False):
        event_id = str(r.event_id)
        key = (
            str(r.ticker),
            str(r.accession)
        )

        result = {
            "event_id": event_id,
            "ticker": r.ticker,
            "event_date": r.event_date,
            "source_file": r.source_file,
            "accession_candidate": r.accession,
            "relation": r.relation_normalized,
            "target": str(r.target),
            "target_type": r.target_graph_type,
            "evidence_status_v25": "NOT_CHECKED",
            "matched_excerpt": "",
            "offset": -1,
            "document_sha256": "",
            "factual_claim_verified": False,
            "pdf_identity_verified": False,
        }

        if not (
            bool(r.concept_edge_usable)
            or bool(r.company_edge_usable)
        ):
            result["evidence_status_v25"] = "EXCLUDED"
            out_rows.append(result)
            continue

        if key not in sec_index:
            result["evidence_status_v25"] = "NO_SEC_TEXT"
            out_rows.append(result)
            continue

        doc = sec_index[key]
        relation = str(r.relation_normalized)
        target = norm(r.target)

        if not target or len(target) < 4:
            result["evidence_status_v25"] = "VAGUE_TARGET"
            out_rows.append(result)
            continue

        # Exact phrase matching is deliberately
        # conservative. Do not substitute fuzzy
        # guessing for factual verification.
        cache_key = (key, target, relation)

        if cache_key in cache:
            status, snippet, offset = cache[cache_key]
        else:
            status = "NO_MATCH"
            snippet = ""
            offset = -1

            relation_cues = {
                "DEPENDS_ON": [
                    "depend", "rely", "reliance"
                ],
                "EXPOSED_TO": [
                    "risk", "exposure", "disruption",
                    "litigation", "adverse"
                ],
                "DISCLOSES": [
                    "report", "disclos", "revenue",
                    "risk", "operating"
                ],
                "MANUFACTURED_BY": [
                    "manufactur", "foundry",
                    "fabricat"
                ],
                "PARTNERS_WITH": [
                    "partner", "collaborat"
                ],
                "COMPETES_WITH": [
                    "compet", "rival"
                ],
                "CUSTOMER_OF": [
                    "customer", "purchase", "procure"
                ],
                "SUPPLIES_TO": [
                    "suppl", "seller", "provide"
                ],
            }.get(relation, [])

            for pos, piece in doc["windows"]:
                body = norm(piece)

                if not re.search(
                    r"(?<![a-z0-9])"
                    + re.escape(target)
                    + r"(?![a-z0-9])",
                    body
                ):
                    continue

                current = "CONCEPT_MENTION"

                if any(
                    cue in body
                    for cue in relation_cues
                ):
                    current = "RELATION_COOCCURRENCE"

                if status == "NO_MATCH" or (
                    current == "RELATION_COOCCURRENCE"
                ):
                    status = current
                    snippet = piece[:700]
                    offset = pos

                if status == "RELATION_COOCCURRENCE":
                    break

            cache[cache_key] = (
                status, snippet, offset
            )

        result["evidence_status_v25"] = status
        result["matched_excerpt"] = snippet
        result["offset"] = offset
        result["document_sha256"] = doc["sha256"]

        out_rows.append(result)

    return pd.DataFrame(out_rows)


def source_identity_audit():
    """
    Date-matched accession is ONLY a candidate.
    Optionally locate local PDF bytes and report hashes.

    PDF vs HTML cannot be identity-verified merely
    by SHA comparison (different file formats).
    A verified source association needs explicit
    accession provenance or supervised review.
    """
    rows = []

    for tier in TIERS:
        path = (
            ROOT / f"outputs/v22/{tier}"
            / "filing_link_audit.csv"
        )
        if not path.exists():
            continue

        links = pd.read_csv(
            path, dtype={"accession": str}
        )

        for _, r in links.iterrows():
            filename = str(r["source_file"])
            candidates = []

            for folder in [
                ROOT / "kg/data",
                ROOT / "kg/output",
                ROOT / "kg",
                ROOT / "data",
                ROOT / "outputs/v16",
            ]:
                p = folder / filename
                if p.is_file():
                    candidates.append(p)

            unique = sorted(set(candidates))

            rows.append({
                "tier": tier,
                "ticker": r["ticker"],
                "source_file": filename,
                "accession_candidate": r.get(
                    "accession", ""
                ),
                "link_status": r["link_status"],
                "pdf_found": bool(unique),
                "pdf_path": (
                    str(unique[0]) if unique else ""
                ),
                "pdf_sha256": (
                    sha_file(unique[0])
                    if unique else ""
                ),
                "pdf_identity_verified": False,
                "review_status": (
                    "LOCAL_PDF_FOUND_NEEDS_IDENTITY_REVIEW"
                    if unique
                    else "PDF_NOT_FOUND_AT_CHECKED_PATHS"
                ),
            })

    return pd.DataFrame(rows)


def temporal_audit(tier, observations):
    """
    Validate every produced market-day feature's
    source observations are strictly older than
    the date being predicted/traded.
    """
    market = pd.read_csv(
        ROOT / "kg/data/market_5stocks_2017_2026.csv"
    )
    if "ticker" not in market and "tic" in market:
        market = market.rename(
            columns={"tic": "ticker"}
        )

    market["ticker"] = (
        market["ticker"].str.upper()
    )
    market["date"] = clean_date(
        market["date"]
    )
    market = market[
        market["ticker"].isin(STOCKS)
        & market["date"].between(
            "2018-01-01", "2023-12-31"
        )
    ][["date", "ticker"]].drop_duplicates()

    if len(market) != 7545:
        raise ValueError(
            "Unexpected market panel size"
        )

    obs = observations[
        observations["usable_for_graph_features"]
    ].copy()

    obs["event_date"] = clean_date(
        obs["event_date"]
    )

    result = []

    for ticker in STOCKS:
        subset = (
            obs[obs["ticker"].eq(ticker)]
            .sort_values(["event_date", "event_id"])
        )

        days = sorted(
            market.loc[
                market["ticker"].eq(ticker),
                "date"
            ]
        )

        times = list(
            subset["event_date"]
        )
        ptr = 0
        count = 0
        latest = pd.NaT
        previous_count = 0

        for day in days:
            while (
                ptr < len(times)
                and times[ptr] < day
            ):
                latest = times[ptr]
                count += 1
                ptr += 1

            changed = count != previous_count
            previous_count = count

            if pd.notna(latest) and not latest < day:
                raise ValueError(
                    f"Temporal leakage {ticker}/{day}"
                )

            age = (
                (day - latest).days
                if pd.notna(latest)
                else -1
            )

            result.append({
                "date": day,
                "ticker": ticker,
                "observations_known": count,
                "new_observations_since_last_day": int(changed),
                "last_kg_public_date": (
                    latest if pd.notna(latest)
                    else pd.NaT
                ),
                "days_since_last_kg_update": age,
                "has_past_kg_evidence": int(count > 0),
            })

    panel = pd.DataFrame(result).sort_values(
        ["date", "ticker"]
    )

    if len(panel) != 7545:
        raise ValueError(
            "Temporal panel incomplete"
        )

    return panel


def execution_review():
    """
    Static inspection, not certification.

    Identify environment implementation and
    references to close, market_close,
    action/reward/step.
    """
    filepath = (
        ROOT / "scripts/run_sixway_train2020.py"
    )

    if not filepath.is_file():
        return {
            "status": "SOURCE_NOT_FOUND",
            "strict_pit_approved": False,
        }

    content = filepath.read_text(
        encoding="utf-8"
    )
    lines = content.splitlines()

    hits = []

    terms = [
        "class PortfolioEnv",
        "def step(",
        "def _get_observation",
        "market_close",
        "next_day",
        "transaction_cost",
        "reward",
        "self.day",
        "self.date",
        "self.idx",
        "self.index",
    ]

    for i, line in enumerate(lines):
        if any(term in line for term in terms):
            hits.append({
                "line": i + 1,
                "source": line.strip()[:230],
            })

    report = {
        "file": str(filepath),
        "sha256": sha_file(filepath),
        "candidate_lines": hits[:180],
        "needs_manual_review": [
            "Observation at t uses only information known before trade execution",
            "Orders at t execute at t or t+1 consistently",
            "Features do not include same-day closing prices before close",
            "Reward uses returns earned after the action",
            "Transaction costs use executed portfolio changes",
            "Reset and train/val/test boundaries avoid future state",
        ],
        "strict_pit_approved": False,
        "status": "STATIC_REVIEW_ONLY",
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tiers", nargs="+",
        choices=TIERS,
        default=TIERS
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest()
    sec_index, sec_audit = build_sec_index(manifest)
    sec_audit.to_csv(
        OUT / "sec_cache_audit.csv",
        index=False
    )

    pdf_audit = source_identity_audit()
    pdf_audit.to_csv(
        OUT / "source_identity_audit.csv",
        index=False
    )

    environment_review = execution_review()
    (OUT / "execution_timing_review.json").write_text(
        json.dumps(
            environment_review, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    summary = {
        "research_companies": STOCKS,
        "sec_documents_indexed": len(sec_index),
        "pdf_identity_verified": False,
        "trading_timing_verified": False,
        "strict_pit_approved": False,
        "tiers": {},
    }

    for tier in args.tiers:
        print(f"\n=== V25 {tier.upper()} ===")

        src = (
            ROOT / "outputs/v24_fixed5"
            / tier / "observations.parquet"
        )

        if not src.exists():
            raise FileNotFoundError(src)

        obs = pd.read_parquet(src)
        obs["event_date"] = clean_date(
            obs["event_date"]
        )

        if not set(obs["ticker"]).issubset(set(STOCKS)):
            raise ValueError(
                "Unexpected research company"
            )

        out = OUT / tier
        out.mkdir(parents=True, exist_ok=True)

        matches = concept_evidence(
            obs, sec_index
        )
        if matches["event_id"].duplicated().any():
            raise ValueError(
                "Duplicate evidence results"
            )

        matches.to_parquet(
            out / "evidence_results.parquet",
            index=False
        )

        matches[
            [
                "event_id",
                "ticker",
                "event_date",
                "source_file",
                "accession_candidate",
                "relation",
                "target",
                "evidence_status_v25",
                "offset",
            ]
        ].to_csv(
            out / "evidence_summary.csv",
            index=False
        )

        panel = temporal_audit(
            tier, obs
        )
        panel.to_csv(
            out / "temporal_audit_panel.csv",
            index=False
        )

        status_counts = {
            str(k): int(v)
            for k, v in matches[
                "evidence_status_v25"
            ].value_counts().items()
        }

        change_count = int(
            panel["new_observations_since_last_day"]
            .sum()
        )

        report = {
            "events": len(obs),
            "evidence_status_counts": status_counts,
            "changed_stock_days": change_count,
            "stock_days": len(panel),
            "strict_pit_approved": False,
        }

        summary["tiers"][tier] = report

        print(json.dumps(
            report, indent=2
        ))

    (OUT / "summary.json").write_text(
        json.dumps(
            summary, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V25 COMPLETE ===")
    print("Research companies:", STOCKS)
    print("SEC documents indexed:", len(sec_index))
    print("Strict PIT approval: NOT COMPLETE")
    print("Output:", OUT)


if __name__ == "__main__":
    main()
