from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BAD_MATCH_COMPANIES = {
    "public policy holding company, inc.",
    "quality industrial corp.",
}

WRONG_RELATION_CONTEXT = [
    r"\bboard of directors\b",
    r"\bserves on the board\b",
    r"\bserved on the board\b",
    r"\bdirector of\b",
    r"\bexecutive officer\b",
    r"\bappointed\b",
    r"\bjoined the board\b",
]


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class RelationEdge:
    source: str
    target: str
    relation: str

    source_name: str = ""
    target_name: str = ""

    source_cik: str = ""
    target_cik: str = ""

    filing_ticker: str = ""
    filing_date: str = ""

    doc_id: str = ""
    chunk_id: str = ""
    form: str = ""
    url: str = ""

    evidence: str = ""

    confidence: float = 0.0
    explicit: bool = True

    extraction_method: str = ""
    validation_status: str = "ACCEPT"

    sic_source: str = ""
    sic_target: str = ""


# ============================================================
# HELPERS
# ============================================================

def clean_text(x) -> str:
    if x is None:
        return ""
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_name(x) -> str:
    return clean_text(x).lower()


def sentence_split(text: str) -> List[str]:
    text = clean_text(text)

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


def wrong_relation_context(text: str) -> bool:
    return any(
        re.search(p, text, re.I)
        for p in WRONG_RELATION_CONTEXT
    )


# ============================================================
# LOAD DATA
# ============================================================

def load_chunks(path: str) -> List[Dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def load_portfolio_companies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"ticker", "name"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"companies.csv missing columns: {missing}"
        )

    for col in ["sic", "cik"]:
        if col not in df.columns:
            df[col] = ""

    df["ticker"] = (
        df["ticker"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df["name"] = (
        df["name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["sic"] = (
        df["sic"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["cik"] = (
        df["cik"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def load_external_companies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "cik",
        "company_name",
        "tickers",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"external company file missing columns: {missing}"
        )

    for col in required:
        df[col] = (
            df[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


# ============================================================
# ALIASES
# ============================================================

LEGAL_SUFFIX = re.compile(
    r"\b("
    r"incorporated|inc\.?|"
    r"corporation|corp\.?|"
    r"company|co\.?|"
    r"limited|ltd\.?|"
    r"plc|"
    r"holdings?"
    r")\b",
    re.I,
)

SAFE_SINGLE_WORD = {
    "apple",
    "microsoft",
    "nvidia",
    "amazon",
    "google",
    "oracle",
    "salesforce",
    "adobe",
    "cisco",
    "intel",
    "qualcomm",
    "broadcom",
    "netflix",
    "tesla",
    "paypal",
    "fabrinet",
}


def normalize_alias(name: str) -> str:
    name = clean_text(name).lower()
    name = name.replace("&", " and ")

    name = re.sub(
        r"[^a-z0-9 ]+",
        " ",
        name,
    )

    return re.sub(
        r"\s+",
        " ",
        name,
    ).strip()


def strip_legal_suffix(name: str) -> str:
    x = normalize_alias(name)

    x = LEGAL_SUFFIX.sub(
        " ",
        x,
    )

    return re.sub(
        r"\s+",
        " ",
        x,
    ).strip()


def make_aliases(
    company_name: str,
    tickers: str = "",
) -> List[str]:

    aliases = set()

    full = normalize_alias(company_name)

    if len(full) >= 4:
        aliases.add(full)

    simple = strip_legal_suffix(
        company_name
    )

    tokens = simple.split()

    if len(tokens) >= 2:
        aliases.add(simple)

    elif (
        len(tokens) == 1
        and tokens[0] in SAFE_SINGLE_WORD
    ):
        aliases.add(simple)

    # IMPORTANT:
    # no bare ticker matching

    return sorted(
        aliases,
        key=len,
        reverse=True,
    )


# ============================================================
# COMPANY UNIVERSE
# ============================================================

def build_company_universe(
    portfolio_df: pd.DataFrame,
    external_df: pd.DataFrame,
) -> Dict[str, Dict]:

    companies = {}

    for _, row in portfolio_df.iterrows():

        ticker = row["ticker"]

        companies[f"PORTFOLIO::{ticker}"] = {
            "ticker": ticker,
            "tickers": [ticker],
            "name": row["name"],
            "cik": row["cik"],
            "sic": row["sic"],
            "aliases": make_aliases(
                row["name"],
                ticker,
            ),
            "portfolio": True,
        }

    for _, row in external_df.iterrows():

        name = row["company_name"]

        if (
            normalize_name(name)
            in BAD_MATCH_COMPANIES
        ):
            continue

        ticker_list = [
            x.strip()
            for x in row["tickers"].split("|")
            if x.strip()
        ]

        ticker = (
            ticker_list[0]
            if ticker_list
            else ""
        )

        companies[f"CIK::{row['cik']}"] = {
            "ticker": ticker,
            "tickers": ticker_list,
            "name": name,
            "cik": row["cik"],
            "sic": "",
            "aliases": make_aliases(
                name,
                row["tickers"],
            ),
            "portfolio": False,
        }

    return companies


# ============================================================
# FIND COMPANY MENTIONS
# ============================================================

def find_company_mentions(
    sentence: str,
    universe: Dict[str, Dict],
    filing_ticker: str,
) -> List[Tuple[Dict, str, int, int]]:

    sentence_lower = sentence.lower()
    found = []

    for company in universe.values():

        ticker_candidates = set(
            company.get("tickers", [])
        )

        if company.get("ticker"):
            ticker_candidates.add(
                company["ticker"]
            )

        if filing_ticker in ticker_candidates:
            continue

        for alias in company["aliases"]:

            pattern = (
                r"(?<![a-z0-9])"
                + re.escape(alias)
                .replace(r"\ ", r"\s+")
                + r"(?![a-z0-9])"
            )

            match = re.search(
                pattern,
                sentence_lower,
                re.I,
            )

            if match:
                found.append(
                    (
                        company,
                        alias,
                        match.start(),
                        match.end(),
                    )
                )
                break

    return found


# ============================================================
# LOCAL CONTEXT
# ============================================================

def local_context(
    sentence: str,
    start: int,
    end: int,
    radius: int = 120,
) -> str:

    left = max(
        0,
        start - radius,
    )

    right = min(
        len(sentence),
        end + radius,
    )

    return clean_text(
        sentence[left:right]
    )


# ============================================================
# RELATION RULES
# ============================================================

def detect_competitor(
    sentence: str,
    company_start: int,
    company_end: int,
) -> Optional[float]:

    before = sentence[
        max(0, company_start - 220):
        company_start
    ].lower()

    after = sentence[
        company_end:
        min(len(sentence), company_end + 180)
    ].lower()

    local = local_context(
        sentence,
        company_start,
        company_end,
        radius=240,
    ).lower()

    # --------------------------------------------------------
    # Reject obvious distribution / ecosystem / availability
    # contexts. These companies may be customers, partners,
    # cloud providers, OEMs, etc., but not necessarily rivals.
    # --------------------------------------------------------

    reject_patterns = [
        r"\bavailable\s+in\b",
        r"\bavailable\s+from\b",
        r"\bservers?\s+from\b",
        r"\bcloud\s+service\s+providers?\b",
        r"\bcomputer\s+makers?\b",
        r"\bintegrated\s+into\b",
        r"\bpartners?\b",
        r"\bcustomers?\b",
    ]

    if any(
        re.search(p, local, re.I)
        for p in reject_patterns
    ):
        # Do not reject if explicit competition wording is
        # very near the company mention.
        explicit_competition = [
            r"\bcurrent\s+competitors?\s+(?:include|includes)\b",
            r"\bcompetitors?\s+(?:include|includes|included)\b",
            r"\bcompete(?:s|d)?\s+with\b",
            r"\bcompete(?:s|d)?\s+against\b",
            r"\bcompetition\s+(?:comes?|came)\s+from\b",
            r"\bcompetition\s+from\b",
            r"\bsource\s+of\s+competition\b",
        ]

        if not any(
            re.search(p, local, re.I)
            for p in explicit_competition
        ):
            return None

    # --------------------------------------------------------
    # High precision competitor rules
    # --------------------------------------------------------

    strong_patterns = [
        r"\bcurrent\s+competitors?\s+(?:include|includes)\b",
        r"\bcompetitors?\s+(?:include|includes|included)\b",
        r"\bcompete(?:s|d)?\s+with\b",
        r"\bcompete(?:s|d)?\s+against\b",
        r"\bcompetition\s+(?:comes?|came)\s+from\b",
        r"\bcompetition\s+from\b",
        r"\bsource\s+of\s+competition\b",
    ]

    if any(
        re.search(p, local, re.I)
        for p in strong_patterns
    ):
        return 0.98

    # --------------------------------------------------------
    # List form:
    # "Dynamics competes with ... providers such as Oracle..."
    #
    # Here "such as" is acceptable ONLY because an explicit
    # compete phrase occurs shortly before it.
    # --------------------------------------------------------

    if (
        re.search(
            r"\bcompete(?:s|d)?\s+with\b",
            before,
            re.I,
        )
        and
        re.search(
            r"\bsuch\s+as\s*$",
            before,
            re.I,
        )
    ):
        return 0.95

    # "offerings from Adobe" is only valid when the beginning
    # of the same local context explicitly says compete.
    if (
        re.search(
            r"\bcompete(?:s|d)?\s+against\b",
            local,
            re.I,
        )
        and
        re.search(
            r"\bofferings?\s+from\b",
            local,
            re.I,
        )
    ):
        return 0.95

    return None

def detect_supplier(
    sentence: str,
    alias: str,
    company_start: int,
    company_end: int,
) -> Optional[float]:

    before = sentence[
        max(0, company_start - 140):
        company_start
    ].lower()

    after = sentence[
        company_end:
        min(len(sentence), company_end + 140)
    ].lower()

    local = local_context(
        sentence,
        company_start,
        company_end,
        radius=160,
    ).lower()

    # Very strong:
    # "suppliers, such as TSMC"
    strong_before_patterns = [
        r"\bsuppliers?\s*,?\s*(?:including|include|such as)?\s*$",
        r"\bmanufacturers?\s*,?\s*(?:including|include|such as)?\s*$",
        r"\bfoundries?\s*,?\s*(?:including|include|such as)?\s*$",
        r"\bcontract manufacturers?\s*,?\s*(?:including|include|such as)?\s*$",
        r"\bodms?\s*,?\s*(?:including|include|such as)?\s*$",
        r"\bsource(?:d)?\s+from\s*$",
        r"\bpurchase(?:s|d)?\s+from\s*$",
        r"\bprocure(?:s|d)?\s+from\s*$",
        r"\bcomponents?\s+from\s*$",
        r"\brely(?:ies|ied)?\s+on\s*$",
        r"\bdepend(?:s|ed)?\s+on\s*$",
    ]

    if any(
        re.search(p, before, re.I)
        for p in strong_before_patterns
    ):
        return 0.95

    # X supplies us
    if re.search(
        r"\bsuppl(?:y|ies|ied)\b",
        after,
        re.I,
    ):
        return 0.95

    # Explicit sentence patterns
    explicit_local = [
        r"\butilize\s+suppliers?\b",
        r"\butilize\s+.*contract manufacturers?\b",
        r"\bmanufactured\s+by\b",
        r"\bmanufacturing\s+partner\b",
    ]

    if any(
        re.search(p, local, re.I)
        for p in explicit_local
    ):
        return 0.90

    return None


def detect_customer(
    sentence: str,
    alias: str,
    company_start: int,
    company_end: int,
) -> Optional[float]:

    before = sentence[
        max(0, company_start - 160):
        company_start
    ].lower()

    after = sentence[
        company_end:
        min(len(sentence), company_end + 160)
    ].lower()

    local = local_context(
        sentence,
        company_start,
        company_end,
        radius=180,
    ).lower()

    # --------------------------------------------------------
    # Strong explicit cases
    # --------------------------------------------------------

    # "Sales to Dell..."
    if re.search(
        r"\bsales\s+to\s*$",
        before,
        re.I,
    ):
        return 0.98

    # "revenue from Dell"
    if re.search(
        r"\brevenue\s+from\s*$",
        before,
        re.I,
    ):
        return 0.98

    # "Dell accounted for 11% of revenue"
    if re.search(
        r"\baccounted\s+for\b.*?\brevenue\b",
        after,
        re.I,
    ):
        return 0.98

    # "Dell represented 10% of revenue"
    if re.search(
        r"\brepresented\b.*?\brevenue\b",
        after,
        re.I,
    ):
        return 0.98

    # "Dell is a major customer"
    if re.search(
        r"\b(?:is|was|were|are)\s+(?:a\s+)?(?:major|largest)?\s*customer\b",
        after,
        re.I,
    ):
        return 0.98

    # "customers include Dell"
    if re.search(
        r"\bcustomers?\s+(?:include|includes|included|such as)\s*$",
        before,
        re.I,
    ):
        return 0.95

    # --------------------------------------------------------
    # Explicitly reject ambiguous constructions
    # --------------------------------------------------------

    ambiguous_patterns = [
        r"\bmany\s+of\s+which\b",
        r"\bcurrent\s+or\s+potential\s+partners?\s+and\s+customers?\b",
        r"\bpartners?\s+and\s+customers?\b",
    ]

    if any(
        re.search(p, local, re.I)
        for p in ambiguous_patterns
    ):
        return None

    # Contract manufacturer is NOT customer
    manufacturing_patterns = [
        r"\bcontract manufacturers?\b",
        r"\bmanufacturers?\b",
        r"\bodms?\b",
        r"\bsuppliers?\b",
    ]

    if any(
        re.search(p, local, re.I)
        for p in manufacturing_patterns
    ):
        return None

    return None


# ============================================================
# RELATION CLASSIFICATION PER ENTITY
# ============================================================

def classify_entity_relation(
    sentence: str,
    alias: str,
    start: int,
    end: int,
) -> Tuple[Optional[str], float]:

    if wrong_relation_context(
        sentence.lower()
    ):
        return None, 0.0

    # Competition first
    confidence = detect_competitor(
        sentence,
        start,
        end,
    )

    if confidence is not None:
        return (
            "COMPETES_WITH",
            confidence,
        )

    # Customer before supplier:
    # because "sales to X" is highly explicit.
    confidence = detect_customer(
        sentence,
        alias,
        start,
        end,
    )

    if confidence is not None:
        return (
            "CUSTOMER_OF",
            confidence,
        )

    confidence = detect_supplier(
        sentence,
        alias,
        start,
        end,
    )

    if confidence is not None:
        return (
            "SUPPLIES_TO",
            confidence,
        )

    return None, 0.0


# ============================================================
# EDGE DIRECTION
# ============================================================

def build_direction(
    relation: str,
    filing_ticker: str,
    filing_name: str,
    filing_cik: str,
    mentioned: Dict,
):

    mt = (
        mentioned.get("ticker")
        or mentioned.get("name")
        or ""
    )

    mn = mentioned.get(
        "name",
        "",
    )

    mcik = mentioned.get(
        "cik",
        "",
    )

    # filing company competes with mentioned company
    if relation == "COMPETES_WITH":
        return (
            filing_ticker,
            mt,
            filing_name,
            mn,
            filing_cik,
            mcik,
        )

    # mentioned company supplies filing company
    if relation == "SUPPLIES_TO":
        return (
            mt,
            filing_ticker,
            mn,
            filing_name,
            mcik,
            filing_cik,
        )

    # mentioned company is customer of filing company
    if relation == "CUSTOMER_OF":
        return (
            mt,
            filing_ticker,
            mn,
            filing_name,
            mcik,
            filing_cik,
        )

    raise ValueError(relation)


# ============================================================
# TEXT EDGE EXTRACTION
# ============================================================

def extract_text_edges(
    chunks: List[Dict],
    portfolio_df: pd.DataFrame,
    universe: Dict[str, Dict],
) -> List[RelationEdge]:

    portfolio_lookup = {
        row["ticker"]: {
            "name": row["name"],
            "cik": row["cik"],
        }
        for _, row in portfolio_df.iterrows()
    }

    edges = []

    for idx, row in enumerate(chunks):

        filing_ticker = clean_text(
            row.get("ticker", "")
        ).upper()

        if filing_ticker not in portfolio_lookup:
            continue

        filing_name = (
            portfolio_lookup[
                filing_ticker
            ]["name"]
        )

        filing_cik = (
            portfolio_lookup[
                filing_ticker
            ]["cik"]
        )

        text = clean_text(
            row.get("text", "")
        )

        if not text:
            continue

        for sentence in sentence_split(text):

            mentions = find_company_mentions(
                sentence,
                universe,
                filing_ticker,
            )

            if not mentions:
                continue

            for (
                company,
                alias,
                start,
                end,
            ) in mentions:

                relation, confidence = (
                    classify_entity_relation(
                        sentence,
                        alias,
                        start,
                        end,
                    )
                )

                if relation is None:
                    continue

                (
                    source,
                    target,
                    source_name,
                    target_name,
                    source_cik,
                    target_cik,
                ) = build_direction(
                    relation,
                    filing_ticker,
                    filing_name,
                    filing_cik,
                    company,
                )

                if (
                    clean_text(source).upper()
                    ==
                    clean_text(target).upper()
                ):
                    continue

                evidence = local_context(
                    sentence,
                    start,
                    end,
                    radius=220,
                )

                edges.append(
                    RelationEdge(
                        source=source,
                        target=target,
                        relation=relation,

                        source_name=source_name,
                        target_name=target_name,

                        source_cik=source_cik,
                        target_cik=target_cik,

                        filing_ticker=filing_ticker,
                        filing_date=clean_text(
                            row.get(
                                "published_at",
                                "",
                            )
                        ),

                        doc_id=clean_text(
                            row.get(
                                "doc_id",
                                "",
                            )
                        ),

                        chunk_id=clean_text(
                            row.get(
                                "chunk_id",
                                "",
                            )
                        ),

                        form=clean_text(
                            row.get(
                                "form",
                                "",
                            )
                        ),

                        url=clean_text(
                            row.get(
                                "url",
                                "",
                            )
                        ),

                        evidence=evidence,
                        confidence=confidence,
                        explicit=True,

                        extraction_method=(
                            "ENTITY_LOCAL_RULE_V3"
                        ),

                        validation_status="ACCEPT",
                    )
                )

        if (
            idx > 0
            and idx % 1000 == 0
        ):
            print(
                f"Processed {idx}/{len(chunks)} "
                f"chunks | raw_edges={len(edges)}"
            )

    return edges


# ============================================================
# SAME_SECTOR
# ============================================================

def build_same_sector_edges(
    portfolio_df: pd.DataFrame,
) -> List[RelationEdge]:

    edges = []

    df = portfolio_df.copy()

    df = df[
        df["sic"]
        .astype(str)
        .str.strip()
        .ne("")
    ]

    for sic, group in df.groupby("sic"):

        records = (
            group.to_dict("records")
        )

        for i in range(len(records)):
            for j in range(len(records)):

                if i == j:
                    continue

                a = records[i]
                b = records[j]

                edges.append(
                    RelationEdge(
                        source=a["ticker"],
                        target=b["ticker"],
                        relation="SAME_SECTOR",

                        source_name=a["name"],
                        target_name=b["name"],

                        source_cik=a["cik"],
                        target_cik=b["cik"],

                        confidence=1.0,
                        explicit=True,

                        extraction_method=(
                            "SIC_EXACT_MATCH"
                        ),

                        validation_status="ACCEPT",

                        sic_source=str(sic),
                        sic_target=str(sic),
                    )
                )

    return edges


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_edges(
    edges: List[RelationEdge],
) -> List[RelationEdge]:

    best = {}

    for edge in edges:

        key = (
            edge.source,
            edge.target,
            edge.relation,
            edge.filing_date,
        )

        if key not in best:
            best[key] = edge
            continue

        prev = best[key]

        if edge.confidence > prev.confidence:
            best[key] = edge

        elif (
            edge.confidence == prev.confidence
            and len(edge.evidence)
            > len(prev.evidence)
        ):
            best[key] = edge

    return list(
        best.values()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--chunks",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    parser.add_argument(
        "--companies",
        default="kg/data/companies.csv",
    )

    parser.add_argument(
        "--external-companies",
        default=(
            "kg/data/"
            "company_candidates_sec_full.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default="kg/data/edges_v3.csv",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    print("Loading portfolio companies...")

    portfolio_df = load_portfolio_companies(
        args.companies
    )

    print("Loading external companies...")

    external_df = load_external_companies(
        args.external_companies
    )

    print("Building company universe...")

    universe = build_company_universe(
        portfolio_df,
        external_df,
    )

    print(
        f"Company universe: {len(universe)}"
    )

    print("Loading SEC chunks...")

    chunks = load_chunks(
        args.chunks
    )

    if args.max_chunks is not None:
        chunks = chunks[
            :args.max_chunks
        ]

    print(
        f"Chunks loaded: {len(chunks)}"
    )

    print()
    print("Extracting relations...")

    text_edges = extract_text_edges(
        chunks,
        portfolio_df,
        universe,
    )

    print(
        f"Raw text edges: {len(text_edges)}"
    )

    sector_edges = build_same_sector_edges(
        portfolio_df
    )

    print(
        f"SAME_SECTOR edges: "
        f"{len(sector_edges)}"
    )

    all_edges = (
        text_edges
        +
        sector_edges
    )

    all_edges = deduplicate_edges(
        all_edges
    )

    df = pd.DataFrame(
        [
            asdict(x)
            for x in all_edges
        ]
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print("FINANCIAL KG V3")
    print("=" * 70)

    print(
        f"Final edges: {len(df)}"
    )

    if len(df):

        print()
        print("RELATION COUNTS")
        print(
            df["relation"]
            .value_counts()
            .to_string()
        )

        print()
        print("BY FILING COMPANY")

        print(
            pd.crosstab(
                df["filing_ticker"],
                df["relation"],
            )
        )

        print()
        print("FIRST 30 EDGES")

        print(
            df[
                [
                    "source",
                    "relation",
                    "target",
                    "filing_ticker",
                    "filing_date",
                    "confidence",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print()
    print(
        f"Saved -> {output}"
    )


if __name__ == "__main__":
    main()
