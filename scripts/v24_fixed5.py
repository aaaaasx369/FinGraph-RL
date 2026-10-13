#!/usr/bin/env python3
"""
V24 FIXED-FIVE HETEROGENEOUS DYNAMIC KG-RAG

Research universe:
    AAPL, MSFT, NVDA, AMZN, GOOGL ONLY

External firms:
    Evidence nodes only.
    Never become portfolio assets.
    Never enter the five-stock market panel.

Commands:
    build
    query
    panel
    bridge
    audit
    all

Required inputs:
    outputs/v23_1/{tier}/normalized_events.parquet
    outputs/v20/sec_filings_manifest.csv
    outputs/v20/sec_cache/documents/
    outputs/v20/fusion/dataset_{tier}.csv
    outputs/v20/fusion/feature_sets.json
    kg/data/market_5stocks_2017_2026.csv

Output:
    outputs/v24_fixed5/

Methodological limitations:
    KG PDF-to-accession identity is unverified.
    SEC lexical evidence is not factual verification.
    Graph vectors are deterministic hashing, not a trained GNN.
    Strict PIT approval is NOT COMPLETE.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------
# ONLY FIVE RESEARCH COMPANIES
# --------------------------------------------------

STOCKS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

STOCK_SET = set(STOCKS)

ALIASES = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "NVDA": "NVIDIA",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet / Google",
}

EMBED_DIM = 32
OUTPUT_ROOT = ROOT / "outputs/v24_fixed5"

TIERS = ["conservative", "inclusive"]

RELATIONS = [
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
    "PARTNERS_WITH",
    "DEPENDS_ON",
    "MANUFACTURED_BY",
    "ACQUIRED",
    "OPERATES_IN",
    "DISCLOSES",
    "EXPOSED_TO",
    "OTHER",
    "AMBIGUOUS",
]

COMPANY_RELATIONS = {
    "SUPPLIES_TO",
    "CUSTOMER_OF",
    "COMPETES_WITH",
    "SAME_SECTOR",
    "PARTNERS_WITH",
}


# --------------------------------------------------
# GENERAL HELPERS
# --------------------------------------------------

def digest(*parts):
    text = "\x1f".join(map(str, parts))
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def make_id(*parts):
    return digest(*parts)[:24]


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).casefold()
    value = value.replace("&", " and ")

    return " ".join(
        re.sub(
            r"[^a-z0-9]+",
            " ",
            value
        ).split()
    )


def dates(series):
    return pd.to_datetime(
        series,
        errors="raise"
    ).dt.normalize()


def output_dir(tier):
    path = OUTPUT_ROOT / tier
    path.mkdir(
        parents=True,
        exist_ok=True
    )
    return path


def require_columns(df, columns, label):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{label} missing columns: {missing}"
        )


def normalize_ticker(frame):
    frame = frame.copy()

    if (
        "ticker" not in frame.columns
        and "tic" in frame.columns
    ):
        frame = frame.rename(
            columns={"tic": "ticker"}
        )

    require_columns(
        frame,
        ["ticker"],
        "Ticker input"
    )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    return frame


def read_market():
    path = ROOT / (
        "kg/data/"
        "market_5stocks_2017_2026.csv"
    )

    market = normalize_ticker(
        pd.read_csv(path)
    )

    require_columns(
        market,
        ["date", "ticker"],
        "Market data"
    )

    market["date"] = dates(
        market["date"]
    )

    market = market[
        market["ticker"].isin(STOCK_SET)
        & market["date"].between(
            "2018-01-01",
            "2023-12-31"
        )
    ][["date", "ticker"]].copy()

    market = market.drop_duplicates()

    if len(market) != 7545:
        raise ValueError(
            "Expected 7545 stock-day rows, "
            f"got {len(market)}"
        )

    expected_counts = {
        ticker: 1509
        for ticker in STOCKS
    }

    observed_counts = (
        market["ticker"]
        .value_counts()
        .to_dict()
    )

    if observed_counts != expected_counts:
        raise ValueError(
            "Unexpected five-stock coverage: "
            f"{observed_counts}"
        )

    return market.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)


# --------------------------------------------------
# ENTITY TYPES
# --------------------------------------------------

def graph_entity(original_id, entity_type):
    """
    Five portfolio companies become COMPANY nodes.

    All other detected firms remain EXTERNAL_ENTITY.
    Concepts remain CONCEPT.

    An external firm never becomes a sixth research
    company.
    """

    node = str(original_id)

    if entity_type == "COMPANY":
        if node.startswith("COMPANY:"):
            company = node.split(":", 1)[1]

            if company in STOCK_SET:
                return node, "COMPANY"

            return (
                "EXTERNAL:" + company,
                "EXTERNAL_ENTITY"
            )

        return (
            "EXTERNAL:" + make_id(node),
            "EXTERNAL_ENTITY"
        )

    if entity_type == "CONCEPT":
        return node, "CONCEPT"

    return (
        "AMBIGUOUS:" + make_id(node),
        "AMBIGUOUS_ENTITY"
    )


def load_source_events(tier):
    path = (
        ROOT
        / "outputs/v23_1"
        / tier
        / "normalized_events.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing V23.1 data: {path}"
        )

    df = normalize_ticker(
        pd.read_parquet(path)
    )

    needed = [
        "ticker",
        "entity",
        "target",
        "source_node",
        "target_node",
        "source_type",
        "target_type",
        "relation_normalized",
        "relationship",
        "event_id",
        "event_date",
        "semantic_edge_id",
        "source_file",
        "accession",
        "link_status",
        "evidence_status",
        "quality_gate_pass",
        "typed_company_relation",
    ]

    require_columns(
        df, needed, "V23.1 events"
    )

    df = df[
        df["ticker"].isin(STOCK_SET)
    ].copy()

    df["event_date"] = dates(
        df["event_date"]
    )

    if df["event_id"].duplicated().any():
        raise ValueError(
            "Duplicate V23.1 event IDs"
        )

    return df


# --------------------------------------------------
# BUILD HETEROGENEOUS GRAPH
# --------------------------------------------------

def build(tier):
    df = load_source_events(tier)

    source_data = [
        graph_entity(a, b)
        for a, b in zip(
            df["source_node"],
            df["source_type"]
        )
    ]

    target_data = [
        graph_entity(a, b)
        for a, b in zip(
            df["target_node"],
            df["target_type"]
        )
    ]

    df["source_graph_id"] = [
        x[0] for x in source_data
    ]
    df["source_graph_type"] = [
        x[1] for x in source_data
    ]

    df["target_graph_id"] = [
        x[0] for x in target_data
    ]
    df["target_graph_type"] = [
        x[1] for x in target_data
    ]

    # Main study subject must be one of five
    # portfolio companies, not an external source.
    df["main_company"] = (
        "COMPANY:" + df["ticker"]
    )

    df["source_candidate"] = (
        df["link_status"].eq(
            "CANDIDATE_MATCH"
        )
    )

    # External-firm edges are allowed only as
    # evidence about one of the five study companies.
    df["company_edge_usable"] = (
        df["typed_company_relation"].astype(bool)
        & df["quality_gate_pass"].astype(bool)
        & df["source_candidate"]
        & df["relation_normalized"].isin(
            COMPANY_RELATIONS
        )
        & (
            df["source_graph_id"].eq(
                df["main_company"]
            )
            | df["target_graph_id"].eq(
                df["main_company"]
            )
        )
    )

    # Preserve concept observations, even when they
    # are not independently verified as facts.
    df["concept_edge_usable"] = (
        df["source_graph_type"].eq("COMPANY")
        & df["target_graph_type"].eq("CONCEPT")
        & df["source_graph_id"].eq(
            df["main_company"]
        )
        & df["source_candidate"]
    )

    df["usable_for_graph_features"] = (
        df["company_edge_usable"]
        | df["concept_edge_usable"]
    )

    # Candidate means candidate.
    # Never silently promote to VERIFIED.
    df["pdf_identity_verified"] = False
    df["strict_pit_verified"] = False

    df["semantic_graph_edge_id"] = [
        make_id(
            ticker, source, relation, target
        )
        for ticker, source, relation, target in zip(
            df["ticker"],
            df["source_graph_id"],
            df["relation_normalized"],
            df["target_graph_id"],
        )
    ]

    df["document_node"] = [
        "DOCUMENT:" + make_id(
            ticker,
            accession,
            source_file,
        )
        for ticker, accession, source_file in zip(
            df["ticker"],
            df["accession"].fillna(""),
            df["source_file"],
        )
    ]

    df["event_node"] = (
        "EVENT:"
        + df["event_id"].astype(str)
    )

    nodes = {}

    def add_node(node_id, node_type, label):
        if not node_id:
            return

        nodes[node_id] = {
            "node_id": node_id,
            "node_type": node_type,
            "label": str(label),
        }

    # Explicitly register exactly five main
    # company nodes, even if no event exists.
    for ticker in STOCKS:
        add_node(
            "COMPANY:" + ticker,
            "COMPANY",
            ALIASES[ticker]
        )

    graph_edges = []

    for r in df.itertuples(index=False):

        add_node(
            r.source_graph_id,
            r.source_graph_type,
            r.entity
        )

        add_node(
            r.target_graph_id,
            r.target_graph_type,
            r.target
        )

        add_node(
            r.event_node,
            "EVENT",
            r.event_id
        )

        add_node(
            r.document_node,
            "DOCUMENT",
            r.source_file
        )

        # Reified graph representation:
        # subject -> event -> object
        #                    -> document
        links = [
            (
                r.source_graph_id,
                r.event_node,
                "SUBJECT_OF"
            ),
            (
                r.event_node,
                r.target_graph_id,
                str(r.relation_normalized)
            ),
            (
                r.event_node,
                r.document_node,
                "OBSERVED_IN"
            ),
        ]

        for left, right, relation in links:
            graph_edges.append({
                "graph_edge_id": make_id(
                    r.event_id,
                    left, right, relation
                ),
                "ticker": r.ticker,
                "source_id": left,
                "target_id": right,
                "edge_type": relation,
                "event_date": r.event_date,
                "event_id": r.event_id,
                "usable_for_graph_features": bool(
                    r.usable_for_graph_features
                ),
                "strict_pit_verified": False,
            })

    nodes_df = pd.DataFrame(
        list(nodes.values())
    )

    edges_df = pd.DataFrame(
        graph_edges
    )

    out = output_dir(tier)

    df.to_parquet(
        out / "observations.parquet",
        index=False
    )

    nodes_df.to_parquet(
        out / "nodes.parquet",
        index=False
    )

    edges_df.to_parquet(
        out / "edges.parquet",
        index=False
    )

    company_nodes = nodes_df[
        nodes_df["node_type"].eq(
            "COMPANY"
        )
    ]

    observed_company_ids = set(
        company_nodes["node_id"]
    )

    required_company_ids = {
        "COMPANY:" + ticker
        for ticker in STOCKS
    }

    if observed_company_ids != required_company_ids:
        raise RuntimeError(
            "Five-company universe violated"
        )

    report = {
        "tier": tier,
        "portfolio_companies": STOCKS,
        "main_company_nodes": int(
            len(company_nodes)
        ),
        "total_nodes": int(
            len(nodes_df)
        ),
        "total_graph_edges": int(
            len(edges_df)
        ),
        "raw_observations": int(
            len(df)
        ),
        "usable_company_relation_observations": int(
            df["company_edge_usable"].sum()
        ),
        "usable_concept_observations": int(
            df["concept_edge_usable"].sum()
        ),
        "external_nodes": int(
            nodes_df["node_type"].eq(
                "EXTERNAL_ENTITY"
            ).sum()
        ),
        "strict_pit_approved": False,
        "semantic_fact_verified": False,
    }

    (out / "build_report.json").write_text(
        json.dumps(
            report, indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V24 FIXED FIVE BUILD ===")
    print(json.dumps(
        report,
        indent=2,
        ensure_ascii=False
    ))


# --------------------------------------------------
# SEC DOCUMENT RETRIEVAL
# --------------------------------------------------

class HTMLText(HTMLParser):
    def __init__(self):
        super().__init__(
            convert_charrefs=True
        )
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(
                0, self.skip - 1
            )

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_to_text(data):
    parser = HTMLText()

    parser.feed(
        data.decode(
            "utf-8",
            errors="replace"
        )
    )

    return " ".join(
        " ".join(parser.parts).split()
    )


STOPWORDS = {
    "the", "and", "for",
    "with", "from", "which",
    "what", "when", "has",
    "have", "this", "that",
    "were", "into", "about",
}


def tokens(value):
    return [
        x for x in re.findall(
            r"[a-z][a-z0-9]{2,}",
            str(value).casefold()
        )
        if x not in STOPWORDS
    ]


def lexical_score(query, evidence):
    query_terms = set(
        tokens(query)
    )

    evidence_counts = Counter(
        tokens(evidence)
    )

    if not query_terms:
        return 0.0

    overlap = sum(
        math.log1p(
            evidence_counts[term]
        )
        for term in query_terms
        if term in evidence_counts
    )

    return float(
        overlap / math.sqrt(
            1 + len(evidence_counts) / 50
        )
    )


def sec_retrieve(
    ticker, cutoff, question, top_k
):
    manifest_path = ROOT / (
        "outputs/v20/"
        "sec_filings_manifest.csv"
    )

    manifest = normalize_ticker(
        pd.read_csv(
            manifest_path
        )
    )

    manifest["filing_date"] = dates(
        manifest["filing_date"]
    )

    manifest = manifest[
        manifest["ticker"].eq(ticker)
        & manifest[
            "filing_date"
        ].lt(cutoff)
        & manifest[
            "text_verified"
        ].eq(1)
    ]

    cache = ROOT / (
        "outputs/v20/"
        "sec_cache/documents"
    )

    results = []

    for _, r in manifest.iterrows():

        filename = (
            str(r["ticker"])
            + "_"
            + str(r["accession"])
            + "_"
            + re.sub(
                r"[^A-Za-z0-9_.-]",
                "_",
                str(r["primary_document"])
            )
        )

        path = cache / filename

        if not path.is_file():
            continue

        raw = path.read_bytes()

        actual_sha = hashlib.sha256(
            raw
        ).hexdigest()

        if actual_sha != str(
            r["source_sha256"]
        ):
            raise RuntimeError(
                "SEC SHA256 mismatch: "
                + str(path)
            )

        content = html_to_text(
            raw
        )

        for offset in range(
            0, len(content), 1200
        ):
            excerpt = content[
                offset:offset + 1600
            ]

            if len(excerpt) < 100:
                break

            relevance = lexical_score(
                question, excerpt
            )

            if relevance <= 0:
                continue

            results.append({
                "score": relevance,
                "ticker": ticker,
                "filing_date": str(
                    r["filing_date"].date()
                ),
                "accession": str(
                    r["accession"]
                ),
                "document_url": str(
                    r.get(
                        "document_url", ""
                    )
                ),
                "document_sha256": actual_sha,
                "offset": int(offset),
                "excerpt": excerpt[:900],
                "file_hash_verified": True,
                "relation_claim_verified": False,
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# --------------------------------------------------
# HISTORICAL GRAPH SNAPSHOT
# --------------------------------------------------

def historical_snapshot(
    tier, ticker, cutoff, usable_only=True
):
    path = (
        output_dir(tier)
        / "observations.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Run build first"
        )

    df = pd.read_parquet(path)

    df["event_date"] = dates(
        df["event_date"]
    )

    mask = (
        df["ticker"].eq(ticker)
        & df["event_date"].lt(cutoff)
    )

    if usable_only:
        mask &= df[
            "usable_for_graph_features"
        ].astype(bool)

    df = df[mask].copy()

    df = (
        df.sort_values(
            ["event_date", "event_id"]
        )
        .drop_duplicates(
            "semantic_graph_edge_id",
            keep="last"
        )
    )

    return df


def query(
    tier, ticker, as_of,
    question, top_k
):
    if ticker not in STOCK_SET:
        raise ValueError(
            "Only five tickers are allowed"
        )

    cutoff = pd.Timestamp(
        as_of
    ).normalize()

    historical = historical_snapshot(
        tier, ticker, cutoff
    )

    kg = []

    for _, r in historical.iterrows():

        description = (
            str(r["entity"])
            + " "
            + str(r["relation_normalized"])
            + " "
            + str(r["target"])
            + " "
            + str(r["relationship"])
        )

        relevance = lexical_score(
            question,
            description
        )

        if relevance <= 0:
            continue

        if r["company_edge_usable"]:
            relevance += 0.5

        kg.append({
            "score": relevance,
            "ticker": ticker,
            "source": str(
                r["entity"]
            ),
            "relation": str(
                r["relation_normalized"]
            ),
            "target": str(
                r["target"]
            ),
            "source_type": str(
                r["source_graph_type"]
            ),
            "target_type": str(
                r["target_graph_type"]
            ),
            "event_date": str(
                r["event_date"].date()
            ),
            "source_file": str(
                r["source_file"]
            ),
            "accession_candidate": str(
                r["accession"]
            ),
            "evidence_status": str(
                r["evidence_status"]
            ),
            "claim_verified": False,
        })

    kg.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    kg = kg[:top_k]

    sec = sec_retrieve(
        ticker,
        cutoff,
        question,
        top_k
    )

    result = {
        "ticker": ticker,
        "as_of": as_of,
        "question": question,
        "kg": kg,
        "sec": sec,
        "strict_pit_approved": False,
        "answer_generated": False,
        "abstain": not bool(kg or sec),
    }

    out = (
        output_dir(tier)
        / "queries"
    )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    dest = out / (
        ticker + "_"
        + as_of + "_"
        + make_id(question)[:8]
        + ".json"
    )

    dest.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n=== V24 FIXED FIVE QUERY ===")
    print("Ticker:", ticker)
    print("As-of:", as_of)
    print("KG evidence:", len(kg))
    print("SEC evidence:", len(sec))

    for i, r in enumerate(
        kg[:10], 1
    ):
        print(
            f"KG {i}:",
            r["source"],
            "--",
            r["relation"],
            "-->",
            r["target"],
            "|",
            r["evidence_status"]
        )

    for i, r in enumerate(
        sec[:3], 1
    ):
        print(
            f"SEC {i}:",
            r["filing_date"],
            r["accession"]
        )
        print(
            r["excerpt"][:250]
        )

    if result["abstain"]:
        print(
            "ABSTAIN: no matching "
            "historical evidence"
        )

    print("Saved:", dest)


# --------------------------------------------------
# TEMPORAL GRAPH REPRESENTATION
# --------------------------------------------------

def signed_hash_embedding(
    feature_items,
    dimensions=EMBED_DIM
):
    vector = np.zeros(
        dimensions,
        dtype=np.float64
    )

    for feature, weight in feature_items:

        h = hashlib.sha256(
            str(feature).encode(
                "utf-8"
            )
        ).digest()

        index = int.from_bytes(
            h[:8],
            "big"
        ) % dimensions

        sign = (
            1.0
            if h[8] % 2 == 0
            else -1.0
        )

        vector[index] += (
            sign * weight
        )

    return vector


def panel(tier):
    path = (
        output_dir(tier)
        / "observations.parquet"
    )

    events = pd.read_parquet(
        path
    )

    events["event_date"] = dates(
        events["event_date"]
    )

    events = events[
        events[
            "usable_for_graph_features"
        ].astype(bool)
    ].copy()

    market = read_market()

    records = []

    for ticker in STOCKS:

        company_events = (
            events[
                events["ticker"].eq(
                    ticker
                )
            ]
            .sort_values(
                ["event_date", "event_id"]
            )
            .to_dict("records")
        )

        company_dates = (
            market[
                market["ticker"].eq(
                    ticker
                )
            ]["date"]
            .sort_values()
            .tolist()
        )

        pointer = 0
        latest = {}

        previous = np.zeros(
            EMBED_DIM
        )

        previous_count = 0

        for day in company_dates:

            # Strict < trade day:
            # never use same-day KG events.
            while (
                pointer < len(
                    company_events
                )
                and company_events[
                    pointer
                ]["event_date"] < day
            ):
                event = company_events[
                    pointer
                ]

                latest[
                    event[
                        "semantic_graph_edge_id"
                    ]
                ] = event

                pointer += 1

            features = []

            company_count = 0
            concept_count = 0

            for edge in latest.values():

                if edge[
                    "company_edge_usable"
                ]:
                    weight = 1.0
                    company_count += 1

                elif edge[
                    "concept_edge_usable"
                ]:
                    # Exploratory weighting.
                    # NOT a calibrated confidence.
                    weight = 0.25
                    concept_count += 1

                else:
                    continue

                triple = (
                    edge["source_graph_id"]
                    + "|"
                    + edge[
                        "relation_normalized"
                    ]
                    + "|"
                    + edge["target_graph_id"]
                )

                features.append((
                    "TRIPLE:" + triple,
                    weight
                ))

                features.append((
                    "REL:"
                    + edge[
                        "relation_normalized"
                    ]
                    + "|"
                    + edge[
                        "target_graph_type"
                    ],
                    weight * 0.5
                ))

            vector = signed_hash_embedding(
                features
            )

            denominator = math.sqrt(
                max(1, len(latest))
            )

            vector = (
                vector / denominator
            )

            delta = (
                vector - previous
            )

            previous = vector.copy()

            current_count = len(latest)

            new_count = max(
                0,
                current_count - previous_count
            )

            previous_count = current_count

            item = {
                "date": day,
                "ticker": ticker,
                "v24_company_edges": company_count,
                "v24_concept_edges": concept_count,
                "v24_new_edges": new_count,
                "v24_has_evidence": int(
                    current_count > 0
                ),
            }

            for j in range(
                EMBED_DIM
            ):
                item[
                    f"v24_embed_{j:02d}"
                ] = float(vector[j])

                item[
                    f"v24_delta_{j:02d}"
                ] = float(delta[j])

            records.append(item)

    daily = pd.DataFrame(
        records
    ).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if len(daily) != 7545:
        raise RuntimeError(
            "Incorrect daily panel size"
        )

    if set(
        daily["ticker"]
    ) != STOCK_SET:
        raise RuntimeError(
            "Unexpected study ticker"
        )

    embed_cols = [
        c for c in daily.columns
        if c.startswith(
            ("v24_embed_", "v24_delta_")
        )
    ]

    if len(embed_cols) != 64:
        raise RuntimeError(
            "Expected 64 graph-vector columns"
        )

    if not np.isfinite(
        daily[
            embed_cols
        ].to_numpy(
            dtype=float
        )
    ).all():
        raise RuntimeError(
            "Invalid embedding numbers"
        )

    dest = (
        output_dir(tier)
        / "daily_embeddings.csv"
    )

    daily.to_csv(
        dest,
        index=False
    )

    print("\n=== V24 FIXED FIVE PANEL ===")
    print("Tier:", tier)
    print("Rows:", len(daily))
    print("Tickers:", STOCKS)
    print(
        "Embedding dimensions:",
        EMBED_DIM
    )
    print(
        "Evidence stock-days:",
        int(
            daily[
                "v24_has_evidence"
            ].sum()
        )
    )
    print("Saved:", dest)


# --------------------------------------------------
# FINRL BRIDGE
# --------------------------------------------------

def bridge():

    source = ROOT / (
        "outputs/v20/fusion"
    )

    destination = (
        OUTPUT_ROOT / "finrl"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    schema = json.loads(
        (
            source
            / "feature_sets.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    static = schema[
        "FULL_STATIC"
    ]

    full_dynamic = schema[
        "FULL_DYNAMIC"
    ]

    dynamic_columns = [
        col for col in full_dynamic
        if col not in static
    ]

    if len(dynamic_columns) != 11:
        raise ValueError(
            "Expected exactly 11 "
            "V21 dynamic columns"
        )

    graph_columns = [
        f"v24_delta_{j:02d}"
        for j in range(8)
    ] + [
        "v24_company_edges",
        "v24_concept_edges",
        "v24_has_evidence",
    ]

    reports = {}

    for tier in TIERS:

        old = pd.read_csv(
            source
            / f"dataset_{tier}.csv"
        )

        graph = pd.read_csv(
            output_dir(tier)
            / "daily_embeddings.csv"
        )

        old = normalize_ticker(
            old
        )

        graph = normalize_ticker(
            graph
        )

        for frame in [
            old, graph
        ]:
            frame["date"] = dates(
                frame["date"]
            )

            if frame.duplicated(
                ["date", "ticker"]
            ).any():
                raise ValueError(
                    "Duplicate stock-day key"
                )

        old = old[
            old["ticker"].isin(
                STOCK_SET
            )
        ].copy()

        require_columns(
            old,
            ["split"] + dynamic_columns,
            "V20 fusion"
        )

        # Prevent accidental reuse of old dyn_*.
        old = old.drop(
            columns=dynamic_columns
        )

        merged = old.merge(
            graph[
                ["date", "ticker"]
                + graph_columns
            ],
            how="left",
            on=["date", "ticker"],
            validate="one_to_one",
            indicator=True
        )

        if not merged[
            "_merge"
        ].eq("both").all():
            raise RuntimeError(
                "Graph features missing "
                f"for tier {tier}"
            )

        merged.drop(
            columns=["_merge"],
            inplace=True
        )

        if len(merged) != 7545:
            raise RuntimeError(
                "Wrong merged panel size"
            )

        # Do not use validation or test data
        # to fit any feature scaler.
        train_mask = (
            merged["split"].eq(
                "train"
            )
        )

        if not train_mask.any():
            raise RuntimeError(
                "No training observations"
            )

        scaler = {}

        for raw, model_column in zip(
            graph_columns,
            dynamic_columns
        ):

            values = pd.to_numeric(
                merged[raw],
                errors="raise"
            ).astype(float)

            if not np.isfinite(
                values
            ).all():
                raise ValueError(
                    f"Nonfinite {raw}"
                )

            if raw in (
                "v24_company_edges",
                "v24_concept_edges",
            ):
                values = np.log1p(
                    values
                )

            mean = float(
                values[
                    train_mask
                ].mean()
            )

            std = float(
                values[
                    train_mask
                ].std(
                    ddof=0
                )
            )

            if (
                not np.isfinite(std)
                or std <= 1e-12
            ):
                std = 1.0

            merged[
                model_column
            ] = (
                values - mean
            ) / std

            scaler[
                model_column
            ] = {
                "graph_feature": raw,
                "train_mean": mean,
                "train_std": std,
            }

        for name, cols in schema.items():

            require_columns(
                merged, cols, name
            )

            x = merged[
                cols
            ].to_numpy(
                dtype=float
            )

            if not np.isfinite(
                x
            ).all():
                raise RuntimeError(
                    "Nonfinite features "
                    f"in {tier}/{name}"
                )

        merged = merged.sort_values(
            ["date", "ticker"]
        )

        merged.to_csv(
            destination
            / f"dataset_{tier}.csv",
            index=False
        )

        (
            destination
            / f"scaler_{tier}.json"
        ).write_text(
            json.dumps(
                scaler,
                indent=2
            ),
            encoding="utf-8"
        )

        reports[tier] = {
            "rows": len(merged),
            "tickers": STOCKS,
            "full_static_dim": len(
                static
            ),
            "full_dynamic_dim": len(
                full_dynamic
            ),
            "new_graph_features": graph_columns,
            "strict_pit_approved": False,
        }

    (
        destination
        / "feature_sets.json"
    ).write_text(
        json.dumps(
            schema, indent=2
        ),
        encoding="utf-8"
    )

    (
        destination
        / "report.json"
    ).write_text(
        json.dumps(
            reports, indent=2
        ),
        encoding="utf-8"
    )

    print("\n=== V24 FIXED FIVE BRIDGE ===")
    print(json.dumps(
        reports,
        indent=2
    ))
    print("Saved:", destination)


# --------------------------------------------------
# STRUCTURAL / TEMPORAL AUDIT
# --------------------------------------------------

def audit(tier):

    folder = output_dir(
        tier
    )

    nodes = pd.read_parquet(
        folder / "nodes.parquet"
    )

    edges = pd.read_parquet(
        folder / "edges.parquet"
    )

    events = pd.read_parquet(
        folder / "observations.parquet"
    )

    daily = pd.read_csv(
        folder / "daily_embeddings.csv"
    )

    problems = []

    companies = nodes[
        nodes["node_type"].eq(
            "COMPANY"
        )
    ]

    required = {
        "COMPANY:" + ticker
        for ticker in STOCKS
    }

    if set(
        companies["node_id"]
    ) != required:
        problems.append(
            "Main company universe is not five"
        )

    if nodes[
        "node_id"
    ].duplicated().any():
        problems.append(
            "Duplicate graph nodes"
        )

    if edges[
        "graph_edge_id"
    ].duplicated().any():
        problems.append(
            "Duplicate graph edge IDs"
        )

    if events[
        "event_id"
    ].duplicated().any():
        problems.append(
            "Duplicate event IDs"
        )

    valid_nodes = set(
        nodes["node_id"]
    )

    if not set(
        edges["source_id"]
    ).issubset(valid_nodes):
        problems.append(
            "Missing source nodes"
        )

    if not set(
        edges["target_id"]
    ).issubset(valid_nodes):
        problems.append(
            "Missing target nodes"
        )

    if set(
        daily["ticker"]
    ) != STOCK_SET:
        problems.append(
            "Unexpected panel tickers"
        )

    if len(daily) != 7545:
        problems.append(
            "Incorrect daily rows"
        )

    if daily.duplicated(
        ["date", "ticker"]
    ).any():
        problems.append(
            "Duplicate panel key"
        )

    typed = events[
        events[
            "company_edge_usable"
        ].astype(bool)
    ]

    if not typed[
        "quality_gate_pass"
    ].astype(bool).all():
        problems.append(
            "Ungated company relation"
        )

    if (
        events[
            "strict_pit_verified"
        ].astype(bool).any()
    ):
        problems.append(
            "False PIT verification claim"
        )

    # Verify all graph input events precede
    # market dates using a representative test
    # across every study company.
    events["event_date"] = dates(
        events["event_date"]
    )

    for ticker in STOCKS:
        cutoff = pd.Timestamp(
            "2021-06-01"
        )

        historical = events[
            events["ticker"].eq(
                ticker
            )
            & events[
                "event_date"
            ].lt(cutoff)
        ]

        if (
            historical[
                "event_date"
            ] >= cutoff
        ).any():
            problems.append(
                "Future snapshot leakage"
            )

    print("\n=== V24 FIXED FIVE AUDIT ===")
    print("Tier:", tier)
    print(
        "Main companies:",
        companies[
            "node_id"
        ].tolist()
    )
    print("Nodes:", len(nodes))
    print("Graph edges:", len(edges))
    print(
        "Company relation observations:",
        int(
            events[
                "company_edge_usable"
            ].sum()
        )
    )
    print(
        "Concept observations:",
        int(
            events[
                "concept_edge_usable"
            ].sum()
        )
    )
    print("Daily rows:", len(daily))
    print("Problems:", problems)
    print(
        "Strict PIT approval: NOT COMPLETE"
    )

    if problems:
        raise RuntimeError(
            f"Audit failed: {problems}"
        )

    print(
        "V24 FIXED FIVE "
        "STRUCTURAL AUDIT PASSED"
    )


# --------------------------------------------------
# COMMAND LINE
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "V24 five-stock heterogeneous "
            "temporal KG-RAG"
        )
    )

    sub = parser.add_subparsers(
        dest="command",
        required=True
    )

    for command in [
        "build",
        "query",
        "panel",
        "bridge",
        "audit",
        "all",
    ]:
        p = sub.add_parser(
            command
        )

        if command in (
            "build", "panel", "audit",
        ):
            p.add_argument(
                "--tier",
                choices=TIERS,
                default="conservative"
            )

        if command == "query":
            p.add_argument(
                "--tier",
                choices=TIERS,
                default="conservative"
            )
            p.add_argument(
                "--ticker",
                choices=STOCKS,
                required=True
            )
            p.add_argument(
                "--as-of",
                required=True
            )
            p.add_argument(
                "--question",
                required=True
            )
            p.add_argument(
                "--top-k",
                type=int,
                default=10
            )

    args = parser.parse_args()

    if args.command == "build":
        build(
            args.tier
        )

    elif args.command == "query":
        query(
            tier=args.tier,
            ticker=args.ticker,
            as_of=args.as_of,
            question=args.question,
            top_k=args.top_k,
        )

    elif args.command == "panel":
        panel(
            args.tier
        )

    elif args.command == "bridge":
        bridge()

    elif args.command == "audit":
        audit(
            args.tier
        )

    elif args.command == "all":
        for tier in TIERS:
            build(tier)

        for tier in TIERS:
            panel(tier)

        for tier in TIERS:
            audit(tier)

        bridge()

        print(
            "\n=== V24 FIXED FIVE "
            "PIPELINE COMPLETE ==="
        )
        print(
            "Study tickers:",
            STOCKS
        )
        print(
            "Strict PIT approval: "
            "NOT COMPLETE"
        )


if __name__ == "__main__":
    main()
