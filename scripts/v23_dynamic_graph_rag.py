#!/usr/bin/env python3
"""
V23 Relation-Aware Temporal Graph-RAG.

Commands:
  build  -> normalize V22 relation events
  query  -> historical KG + SEC retrieval
  panel  -> daily temporal graph feature panel
  audit  -> validate provenance and time gates

Research level:
  EXPLORATORY / CANDIDATE-PROVENANCE

No API key. No LLM-generated claims. No rewriting old runs.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

ALIASES = {
    "AAPL": [
        "apple", "apple inc", "apple inc.",
        "apple computer"
    ],
    "MSFT": [
        "microsoft", "microsoft corporation",
        "microsoft corp"
    ],
    "NVDA": [
        "nvidia", "nvidia corporation",
        "nvidia corp"
    ],
    "AMZN": [
        "amazon", "amazon.com",
        "amazon.com inc", "amazon com inc"
    ],
    "GOOGL": [
        "alphabet", "alphabet inc",
        "google", "google llc", "google inc"
    ],
    "TSMC": [
        "tsmc",
        "taiwan semiconductor manufacturing",
        "taiwan semiconductor manufacturing company",
        "taiwan semiconductor manufacturing company limited"
    ],
    "SSNLF": [
        "samsung electronics",
        "samsung electronics co ltd",
        "samsung electronics co. ltd."
    ],
    "INTC": [
        "intel", "intel corporation", "intel corp"
    ],
    "AMD": [
        "advanced micro devices",
        "advanced micro devices inc", "amd"
    ],
    "QCOM": [
        "qualcomm", "qualcomm incorporated"
    ],
    "AVGO": [
        "broadcom", "broadcom inc"
    ],
    "META": [
        "meta platforms", "facebook inc",
        "facebook"
    ],
}

RELATION_PATTERNS = [
    ("SUPPLIES_TO", [
        r"\bsupplies?_to\b",
        r"\bprovides?_to\b",
        r"\bships?_to\b",
    ]),
    ("CUSTOMER_OF", [
        r"\bcustomer_of\b",
        r"\bpurchases?_from\b",
        r"\bprocures?_from\b",
        r"\bbuy(s|ing)?_from\b",
        r"\bdepends?_on\b",
    ]),
    ("COMPETES_WITH", [
        r"\bcompetes?_with\b",
        r"\bcompetitor\b",
        r"\bcompetition\b",
        r"\brival\b",
    ]),
    ("SAME_SECTOR", [
        r"\bsame_sector\b",
        r"\bsame_industry\b",
    ]),
    ("PARTNERS_WITH", [
        r"\bpartners?_with\b",
        r"\bcollaborates?_with\b",
        r"\balliance\b",
    ]),
    ("OPERATES_IN", [
        r"\boperates?_in\b",
        r"\blocated?_in\b",
    ]),
    ("DISCLOSES", [
        r"\bdiscloses?\b",
        r"\breports?\b",
    ]),
    ("EXPOSED_TO", [
        r"\bimpacted?_by\b",
        r"\bfaces?\b",
        r"\bexposed?_to\b",
    ]),
]

RELATIONS = [
    "SUPPLIES_TO", "CUSTOMER_OF",
    "COMPETES_WITH", "SAME_SECTOR",
    "PARTNERS_WITH", "OPERATES_IN",
    "DISCLOSES", "EXPOSED_TO", "OTHER",
]

TOPIC_PATTERNS = {
    "supply": re.compile(
        r"supplier|supply|manufactur|foundr|wafer|"
        r"fabricat|component|procure|vendor|"
        r"customer|dependency", re.I
    ),
    "competition": re.compile(
        r"competi|rival|market.share|sector|industry",
        re.I
    ),
    "regulation": re.compile(
        r"regulat|export.control|sanction|"
        r"litigation|lawsuit|antitrust", re.I
    ),
    "partnership": re.compile(
        r"partner|collaborat|alliance|joint.venture",
        re.I
    ),
}

STOP = {
    "the", "and", "for", "with", "from", "what",
    "which", "when", "does", "has", "have", "been",
    "this", "that", "into", "about", "company",
    "companies", "corporation", "business",
    "related", "information",
}

def norm(s):
    if pd.isna(s):
        return ""
    s = str(s).casefold()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())

ALIAS_TO_ID = {
    norm(alias): key
    for key, aliases in ALIASES.items()
    for alias in aliases + [key]
}

def entity_id(value):
    x = norm(value)
    if x in ALIAS_TO_ID:
        return "COMPANY:" + ALIAS_TO_ID[x], "COMPANY"

    if not x:
        return "", "EMPTY"

    return "CONCEPT:" + hashlib.sha1(
        x.encode()
    ).hexdigest()[:14], "CONCEPT"

def relation_type(value):
    v = norm(value).replace(" ", "_")
    for label, patterns in RELATION_PATTERNS:
        if any(re.search(p, v) for p in patterns):
            return label
    return "OTHER"

def checksum(*values):
    return hashlib.sha256(
        "\x1f".join(map(str, values)).encode()
    ).hexdigest()[:24]

def tokens(s):
    return [
        w for w in re.findall(
            r"[a-z][a-z0-9]{2,}", str(s).casefold()
        )
        if w not in STOP
    ]

def lexical_score(question, text):
    q = set(tokens(question))
    terms = Counter(tokens(text))
    if not q:
        return 0.0
    return sum(
        math.log1p(terms[x])
        for x in q if x in terms
    ) / math.sqrt(1 + len(terms) / 30)

def topic(question):
    q = str(question)
    hits = [
        name for name, pat in TOPIC_PATTERNS.items()
        if pat.search(q)
    ]
    return hits

def topic_bonus(qtopics, relation, description):
    bonus = 0.
    if "supply" in qtopics:
        if relation in {
            "SUPPLIES_TO", "CUSTOMER_OF"
        }:
            bonus += 2.
        if TOPIC_PATTERNS["supply"].search(description):
            bonus += 1.
    if "competition" in qtopics and relation in {
        "COMPETES_WITH", "SAME_SECTOR"
    }:
        bonus += 2.
    if "regulation" in qtopics and relation in {
        "EXPOSED_TO", "DISCLOSES"
    }:
        bonus += 0.6
    if "partnership" in qtopics and relation == "PARTNERS_WITH":
        bonus += 2.
    return bonus

def paths(args):
    base = ROOT / "outputs/v23" / args.tier
    base.mkdir(parents=True, exist_ok=True)
    return base

def source_v22(tier):
    return (
        ROOT / "outputs/v22" / tier
        / "relation_events.parquet"
    )

def build(args):
    src = source_v22(args.tier)
    if not src.exists():
        raise FileNotFoundError(
            f"Build V22 first: {src}"
        )

    df = pd.read_parquet(src)
    needed = [
        "ticker", "entity", "relationship", "target",
        "event_date", "event_id", "relation_key",
        "event_type", "source_file", "link_status",
    ]
    missing = [c for c in needed if c not in df]
    if missing:
        raise ValueError(f"V22 missing: {missing}")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["event_date"] = pd.to_datetime(
        df["event_date"], errors="raise"
    ).dt.normalize()

    df["source_node"] = [
        entity_id(x)[0] for x in df["entity"]
    ]
    df["target_node"] = [
        entity_id(x)[0] for x in df["target"]
    ]
    df["source_kind"] = [
        entity_id(x)[1] for x in df["entity"]
    ]
    df["target_kind"] = [
        entity_id(x)[1] for x in df["target"]
    ]

    df["relation_normalized"] = [
        relation_type(x) for x in df["relationship"]
    ]

    # Require evidence provenance in graph events.
    df = df[
        df["ticker"].isin(TICKERS)
        & df["source_node"].ne("")
        & df["target_node"].ne("")
        & df["event_date"].notna()
        & df["link_status"].eq("CANDIDATE_MATCH")
    ].copy()

    df["semantic_edge_id"] = [
        checksum(a, b, c, d)
        for a, b, c, d in zip(
            df["ticker"],
            df["source_node"],
            df["relation_normalized"],
            df["target_node"]
        )
    ]

    # The original extraction may contain broad or
    # misleading relations. Keep them, but label them.
    df["typed_company_relation"] = (
        df["source_kind"].eq("COMPANY")
        & df["target_kind"].eq("COMPANY")
        & df["relation_normalized"].isin([
            "SUPPLIES_TO", "CUSTOMER_OF",
            "COMPETES_WITH", "SAME_SECTOR",
            "PARTNERS_WITH",
        ])
    )

    df["source_verified"] = False
    df["pdf_identity_verified"] = False
    df["historical_evidence_level"] = (
        "EDGAR_DATE_CANDIDATE"
    )

    # Event history is keyed to normalized semantics,
    # not solely to the original raw relation spelling.
    df = df.sort_values(
        ["semantic_edge_id", "event_date", "event_id"]
    ).reset_index(drop=True)

    df["semantic_observation_number"] = (
        df.groupby("semantic_edge_id").cumcount() + 1
    )
    df["semantic_event_type"] = np.where(
        df["semantic_observation_number"].eq(1),
        "FIRST_OBSERVED",
        "REOBSERVED",
    )

    out = paths(args)
    dest = out / "normalized_events.parquet"
    df.to_parquet(dest, index=False)

    stats = {
        "tier": args.tier,
        "event_rows": int(len(df)),
        "distinct_semantic_edges": int(
            df["semantic_edge_id"].nunique()
        ),
        "company_to_company_events": int(
            df["typed_company_relation"].sum()
        ),
        "relation_distribution": {
            str(k): int(v)
            for k, v in df[
                "relation_normalized"
            ].value_counts().items()
        },
        "candidate_only": True,
        "strict_pit_verified": False,
        "entity_resolution": "RULE_BASED_V23",
        "relation_resolution": "RULE_BASED_V23",
    }

    (out / "build_report.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(stats, indent=2))
    print("SAVED:", dest)

def load_events(args, cutoff=None):
    path = paths(args) / "normalized_events.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Run build first: {path}"
        )
    df = pd.read_parquet(path)
    df["event_date"] = pd.to_datetime(
        df["event_date"], errors="raise"
    ).dt.normalize()
    if cutoff is not None:
        # Exclude the query/trade date itself.
        df = df[df["event_date"] < cutoff].copy()
    return df

def snapshot(events, ticker, cutoff):
    g = events[
        events["ticker"].eq(ticker)
        & events["event_date"].lt(cutoff)
    ].copy()

    if g.empty:
        return g

    g = g.sort_values(
        ["event_date", "event_id"]
    )

    latest = g.drop_duplicates(
        "semantic_edge_id", keep="last"
    ).copy()

    counts = g.groupby(
        "semantic_edge_id"
    ).size()
    latest["observation_count"] = (
        latest["semantic_edge_id"].map(counts)
    )

    return latest

class ExtractText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

def document_text(raw):
    parser = ExtractText()
    parser.feed(raw.decode("utf-8", errors="replace"))
    return " ".join(" ".join(parser.parts).split())

def sec_retrieval(args, cutoff, qtopics):
    manifest_path = (
        ROOT / "outputs/v20/sec_filings_manifest.csv"
    )
    cache = ROOT / "outputs/v20/sec_cache/documents"

    manifest = pd.read_csv(manifest_path)
    manifest["filing_date"] = pd.to_datetime(
        manifest["filing_date"], errors="coerce"
    ).dt.normalize()

    filings = manifest[
        manifest["ticker"].eq(args.ticker)
        & manifest["filing_date"].lt(cutoff)
        & manifest["text_verified"].eq(1)
    ]

    candidates = []

    for _, item in filings.iterrows():
        name = (
            f"{item['ticker']}_{item['accession']}_"
            + re.sub(
                r"[^A-Za-z0-9_.-]", "_",
                str(item["primary_document"])
            )
        )
        filepath = cache / "documents" / name
        if not filepath.exists():
            filepath = cache / name
        if not filepath.is_file():
            continue

        raw = filepath.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        expected = str(item["source_sha256"])

        if actual != expected:
            raise RuntimeError(
                f"SEC SHA256 mismatch: {filepath}"
            )

        full = document_text(raw)
        size, stride = 1200, 1000

        for pos in range(0, len(full), stride):
            excerpt = full[pos:pos + size]
            if len(excerpt) < 100:
                break

            relevance = lexical_score(
                args.question, excerpt
            )

            if qtopics and any(
                TOPIC_PATTERNS[k].search(excerpt)
                for k in qtopics
            ):
                relevance += 0.3

            if relevance <= 0:
                continue

            candidates.append({
                "type": "SEC_EVIDENCE",
                "score": relevance,
                "filing_date": str(
                    item["filing_date"].date()
                ),
                "accession": str(item["accession"]),
                "document_url": str(item["document_url"]),
                "source_sha256": actual,
                "character_offset": pos,
                "excerpt": excerpt[:750],
                "source_hash_verified": True,
                "semantic_claim_verified": False,
            })

    candidates.sort(
        key=lambda x: x["score"], reverse=True
    )
    return candidates[:args.top_k]

def graph_retrieval(args, cutoff, qtopics):
    events = load_events(args, cutoff)
    graph = snapshot(events, args.ticker, cutoff)

    if graph.empty:
        return []

    anchor = "COMPANY:" + args.ticker
    results = []

    for _, e in graph.iterrows():
        desc = (
            f"{e['entity']} {e['relationship']} "
            f"{e['target']} {e['relation_normalized']}"
        )

        relevance = lexical_score(
            args.question, desc
        )
        relevance += topic_bonus(
            qtopics, e["relation_normalized"], desc
        )

        touches_anchor = (
            e["source_node"] == anchor
            or e["target_node"] == anchor
        )
        if touches_anchor:
            relevance += 0.3

        if e["typed_company_relation"]:
            relevance += 0.75

        if relevance <= 0:
            continue

        results.append({
            "type": "KG_EDGE",
            "score": float(relevance),
            "entity": str(e["entity"]),
            "relation_original": str(e["relationship"]),
            "relation_normalized": str(
                e["relation_normalized"]
            ),
            "target": str(e["target"]),
            "source_node": str(e["source_node"]),
            "target_node": str(e["target_node"]),
            "typed_company_relation": bool(
                e["typed_company_relation"]
            ),
            "last_observed": str(
                e["event_date"].date()
            ),
            "observation_count": int(
                e["observation_count"]
            ),
            "source_file": str(e["source_file"]),
            "accession_candidate": str(
                e.get("accession", "")
            ),
            "event_id": str(e["event_id"]),
            "source_level": (
                "EDGAR_DATE_CANDIDATE"
            ),
        })

    return sorted(
        results, key=lambda x: x["score"],
        reverse=True
    )[:args.top_k]

def query(args):
    cutoff = pd.Timestamp(args.as_of).normalize()
    if pd.isna(cutoff):
        raise ValueError("Invalid date")

    topics = topic(args.question)
    graph = graph_retrieval(args, cutoff, topics)
    sec = sec_retrieval(args, cutoff, topics)

    output = {
        "ticker": args.ticker,
        "as_of": args.as_of,
        "query": args.question,
        "inferred_topics": topics,
        "KG_results": graph,
        "SEC_results": sec,
        "answer_mode": "EVIDENCE_ONLY",
        "generated_narrative": False,
        "abstain": not bool(graph or sec),
        "provenance_warning": (
            "KG PDF identity and semantic relation "
            "claims remain unverified candidates."
        ),
    }

    dest = paths(args) / "queries"
    dest.mkdir(parents=True, exist_ok=True)

    filename = (
        args.ticker + "_" + args.as_of
        + "_" + checksum(args.question)[:8]
        + ".json"
    )
    path = dest / filename
    path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== V23 GRAPH-RAG RESULTS ===")
    print("Ticker:", args.ticker)
    print("As of:", args.as_of)
    print("Topics:", topics)
    print("KG matches:", len(graph))
    print("SEC matches:", len(sec))

    for i, r in enumerate(graph[:5], 1):
        print(
            f"KG {i}:",
            r["entity"], "--",
            r["relation_normalized"], "-->",
            r["target"],
            "| score", round(r["score"], 3)
        )

    for i, r in enumerate(sec[:3], 1):
        print(
            f"SEC {i}:",
            r["filing_date"],
            r["accession"],
            "| score", round(r["score"], 3)
        )
        print(" ", r["excerpt"][:230])

    if output["abstain"]:
        print("ABSTAIN: no historical matching evidence")

    print("Saved:", path)

def panel(args):
    events = load_events(args)

    market_path = ROOT / (
        "kg/data/market_5stocks_2017_2026.csv"
    )
    market = pd.read_csv(market_path)

    if "ticker" not in market:
        market = market.rename(
            columns={"tic": "ticker"}
        )

    market["date"] = pd.to_datetime(
        market["date"], errors="raise"
    ).dt.normalize()
    market["ticker"] = (
        market["ticker"].astype(str).str.upper()
    )

    market = market[
        market["ticker"].isin(TICKERS)
        & market["date"].between(
            "2018-01-01", "2023-12-31"
        )
    ][["date", "ticker"]].drop_duplicates()

    if market.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate market keys")

    out_rows = []

    for ticker in TICKERS:
        rows = (
            events[events["ticker"].eq(ticker)]
            .sort_values(["event_date", "event_id"])
            .to_dict("records")
        )

        dates = sorted(
            market.loc[
                market["ticker"].eq(ticker),
                "date"
            ].tolist()
        )

        pointer = 0
        seen = {}
        observed_count = Counter()

        for day in dates:
            new_today = 0
            repeat_today = 0

            # Strictly earlier than the trade date.
            while (
                pointer < len(rows)
                and rows[pointer]["event_date"] < day
            ):
                event = rows[pointer]
                key = event["semantic_edge_id"]

                if key not in seen:
                    new_today += 1
                else:
                    repeat_today += 1

                seen[key] = event
                observed_count[key] += 1
                pointer += 1

            values = {
                "date": day,
                "ticker": ticker,
                "tkg_edge_count": len(seen),
                "tkg_reobserved_edges": sum(
                    n > 1 for n in observed_count.values()
                ),
                "tkg_new_since_previous_market_day": new_today,
                "tkg_reobserved_since_previous_market_day": repeat_today,
                "tkg_company_edges": sum(
                    bool(e["typed_company_relation"])
                    for e in seen.values()
                ),
            }

            for rel in RELATIONS:
                values["tkg_" + rel.lower()] = sum(
                    e["relation_normalized"] == rel
                    for e in seen.values()
                )

            # Missing state is explicit; no imputed fact.
            values["tkg_has_evidence"] = int(
                len(seen) > 0
            )
            out_rows.append(values)

    df = pd.DataFrame(out_rows)
    df = df.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if len(df) != 7545:
        raise ValueError(
            f"Unexpected panel rows: {len(df)}"
        )

    dest = paths(args) / "daily_graph_features.csv"
    df.to_csv(dest, index=False)

    print("\n=== V23 DAILY GRAPH PANEL ===")
    print("Rows:", len(df))
    print(
        "Max graph edges:",
        int(df["tkg_edge_count"].max())
    )
    print(
        "Evidence rows:",
        int(df["tkg_has_evidence"].sum())
    )
    print("Saved:", dest)

def audit(args):
    events = load_events(args)

    problems = []
    if events["event_id"].duplicated().any():
        problems.append("Duplicate event IDs")
    if events["semantic_edge_id"].isna().any():
        problems.append("Null semantic edge")
    if events["event_date"].isna().any():
        problems.append("Null event date")
    if events["source_verified"].astype(bool).any():
        problems.append(
            "Unexpected strict source verification"
        )
    if not events["link_status"].eq(
        "CANDIDATE_MATCH"
    ).all():
        problems.append(
            "Unexpected filing-link status"
        )

    print("\n=== V23 AUDIT ===")
    print("Events:", len(events))
    print(
        "Semantic edges:",
        events["semantic_edge_id"].nunique()
    )
    print(
        "Company-to-company events:",
        int(events["typed_company_relation"].sum())
    )
    print("Problems:", problems)
    print("Strict PIT approved: NO")

    if problems:
        raise RuntimeError(
            "V23 AUDIT FAILED: " + str(problems)
        )

    print("V23 STRUCTURAL AUDIT PASSED")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(
        dest="command", required=True
    )

    for name in [
        "build", "query", "panel", "audit"
    ]:
        sp = sub.add_parser(name)
        sp.add_argument(
            "--tier",
            choices=["conservative", "inclusive"],
            default="conservative"
        )

        if name == "query":
            sp.add_argument(
                "--ticker", required=True,
                choices=TICKERS
            )
            sp.add_argument(
                "--as-of", required=True
            )
            sp.add_argument(
                "--question", required=True
            )
            sp.add_argument(
                "--top-k", type=int, default=5
            )

    args = parser.parse_args()

    if args.command == "build":
        build(args)
    elif args.command == "query":
        query(args)
    elif args.command == "panel":
        panel(args)
    elif args.command == "audit":
        audit(args)


if __name__ == "__main__":
    main()
