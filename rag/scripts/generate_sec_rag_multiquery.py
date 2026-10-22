from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer


# ============================================================
# Portfolio
# ============================================================

PORTFOLIO = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]


# ============================================================
# Query templates
#
# Same semantic templates for every company.
# Only {ticker} is substituted.
#
# This keeps retrieval methodology consistent across stocks.
# ============================================================

QUERY_TEMPLATES = {
    "financial_impact": (
        "What information in this SEC filing could have a positive "
        "or negative financial impact on {ticker}, including revenue, "
        "profitability, growth, demand, valuation, financial condition, "
        "or cash flow?"
    ),

    "risk": (
        "What material risks, uncertainties, liabilities, regulatory "
        "issues, legal risks, customer dependencies, supplier dependencies, "
        "supply constraints, or operational risks affect {ticker}?"
    ),

    "business_changes": (
        "What important changes in operations, competition, customers, "
        "suppliers, products, margins, costs, capital spending, market "
        "position, or strategy affect {ticker}?"
    ),

    "forward_looking": (
        "What forward-looking information, management discussion, trends, "
        "expectations, opportunities, or challenges could materially affect "
        "{ticker}'s future business or financial performance?"
    ),
}


# ============================================================
# Utility
# ============================================================

def clean_str(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, float) and np.isnan(x):
        return ""

    return str(x).strip()


def clamp(
    value: Any,
    low: float,
    high: float,
) -> float:

    try:
        value = float(value)

    except Exception:
        value = 0.0

    return max(
        low,
        min(
            high,
            value,
        ),
    )


def l2_normalize(
    x: np.ndarray,
) -> np.ndarray:

    norms = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return (
        x
        /
        np.maximum(
            norms,
            1e-12,
        )
    )


def parse_json_from_text(
    text: str,
) -> Dict[str, Any]:

    text = text.strip()

    # Remove markdown fences if model ignores instruction.
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        return json.loads(text)

    except Exception:
        pass

    # Fallback: extract first JSON object.
    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.S,
    )

    if match is None:
        raise ValueError(
            "No JSON object found in response:\n"
            + text[:1500]
        )

    return json.loads(
        match.group(0)
    )


# ============================================================
# Metadata
# ============================================================

def load_metadata(
    path: str,
) -> pd.DataFrame:

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for embedding_index, line in enumerate(f):

            line = line.strip()

            if not line:
                continue

            x = json.loads(line)

            available_date = (
                x.get("published_at")
                or x.get("filing_date")
                or x.get("date")
            )

            rows.append(
                {
                    "embedding_index":
                        embedding_index,

                    "ticker":
                        clean_str(
                            x.get("ticker")
                        ).upper(),

                    "available_date":
                        available_date,

                    "form":
                        clean_str(
                            x.get("form")
                        ),

                    "chunk_id":
                        clean_str(
                            x.get("chunk_id")
                        ),

                    "doc_id":
                        clean_str(
                            x.get("doc_id")
                        ),

                    "text":
                        clean_str(
                            x.get("text")
                        ),

                    "url":
                        clean_str(
                            x.get("url")
                        ),
                }
            )

    df = pd.DataFrame(rows)

    df["available_date"] = pd.to_datetime(
        df["available_date"],
        errors="coerce",
    )

    df = df[
        df["available_date"].notna()
    ].copy()

    return df


# ============================================================
# Build ticker-specific query vectors
#
# 4 fixed semantic templates × 5 tickers
# ============================================================

def build_query_vectors(
    encoder: SentenceTransformer,
) -> Dict[str, Dict[str, np.ndarray]]:

    result = {}

    for ticker in PORTFOLIO:

        query_names = []
        query_texts = []

        for query_name, template in QUERY_TEMPLATES.items():

            query_names.append(
                query_name
            )

            query_texts.append(
                template.format(
                    ticker=ticker
                )
            )

        vectors = encoder.encode(
            query_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(
            np.float32
        )

        result[ticker] = {}

        for i, query_name in enumerate(
            query_names
        ):

            result[ticker][
                query_name
            ] = vectors[i]

    return result


# ============================================================
# Multi-query retrieval
#
# For each event:
#
# Q1 -> Top-2
# Q2 -> Top-2
# Q3 -> Top-2
# Q4 -> Top-2
#
# Then:
#
# deduplicate by embedding_index
# keep best score for duplicate chunk
# final sort
# final Top-6
# ============================================================

def retrieve_multiquery(
    event_df: pd.DataFrame,
    all_embeddings: np.ndarray,
    query_vectors: Dict[str, np.ndarray],
    per_query_k: int,
    final_k: int,
) -> List[Dict[str, Any]]:

    if len(event_df) == 0:
        return []

    global_indices = (
        event_df[
            "embedding_index"
        ]
        .astype(int)
        .to_numpy()
    )

    event_embeddings = (
        all_embeddings[
            global_indices
        ]
        .astype(
            np.float32
        )
    )

    event_embeddings = l2_normalize(
        event_embeddings
    )

    # Key = global embedding index
    candidate_map: Dict[
        int,
        Dict[str, Any]
    ] = {}

    for query_name, query_vector in query_vectors.items():

        scores = (
            event_embeddings
            @ query_vector
        )

        ranking = np.argsort(
            scores
        )[::-1]

        top_n = min(
            per_query_k,
            len(ranking),
        )

        for query_rank in range(
            top_n
        ):

            local_idx = int(
                ranking[
                    query_rank
                ]
            )

            global_idx = int(
                global_indices[
                    local_idx
                ]
            )

            score = float(
                scores[
                    local_idx
                ]
            )

            row = event_df.iloc[
                local_idx
            ]

            if global_idx not in candidate_map:

                candidate_map[
                    global_idx
                ] = {
                    "embedding_index":
                        global_idx,

                    "chunk_id":
                        row[
                            "chunk_id"
                        ],

                    "doc_id":
                        row[
                            "doc_id"
                        ],

                    "form":
                        row[
                            "form"
                        ],

                    "url":
                        row[
                            "url"
                        ],

                    "text":
                        row[
                            "text"
                        ],

                    "best_score":
                        score,

                    "matched_queries":
                        [
                            query_name
                        ],

                    "query_scores":
                        {
                            query_name:
                                score
                        },
                }

            else:

                candidate = candidate_map[
                    global_idx
                ]

                candidate[
                    "matched_queries"
                ].append(
                    query_name
                )

                candidate[
                    "query_scores"
                ][
                    query_name
                ] = score

                candidate[
                    "best_score"
                ] = max(
                    candidate[
                        "best_score"
                    ],
                    score,
                )

    # --------------------------------------------------------
    # Deduplicate complete.
    #
    # Sort primarily by number of matched semantic queries,
    # then by best similarity.
    #
    # This gives preference to chunks relevant to multiple
    # semantic aspects without allowing one query to produce
    # all six candidates.
    # --------------------------------------------------------

    candidates = list(
        candidate_map.values()
    )

    for x in candidates:

        x[
            "query_coverage"
        ] = len(
            set(
                x[
                    "matched_queries"
                ]
            )
        )

    candidates.sort(
        key=lambda x: (
            x[
                "query_coverage"
            ],
            x[
                "best_score"
            ],
        ),
        reverse=True,
    )

    final = candidates[
        :min(
            final_k,
            len(candidates),
        )
    ]

    for rank, item in enumerate(
        final,
        1,
    ):

        item[
            "final_rank"
        ] = rank

        item[
            "matched_queries"
        ] = sorted(
            set(
                item[
                    "matched_queries"
                ]
            )
        )

    return final


# ============================================================
# Remote Qwen inference
# ============================================================

SYSTEM_PROMPT = """
You are a financial SEC filing analysis engine.

Analyze ONLY the supplied SEC filing evidence.

Do not use outside knowledge.

Return exactly one valid JSON object:

{
  "impact": number,
  "risk": number,
  "confidence": number,
  "rationale": string
}

Scoring definitions:

impact:
Range [-1, 1].

-1.0 = strongly negative expected financial effect
-0.5 = moderately negative
 0.0 = neutral, mixed, immaterial, or unclear
+0.5 = moderately positive
+1.0 = strongly positive expected financial effect

Consider effects on revenue, profitability, cash flow, growth,
demand, costs, market position, operations, and valuation.

risk:
Range [0, 1].

0.0 = little or no material risk disclosed
0.5 = moderate material risk
1.0 = extremely severe or highly material risk

Consider financial, legal, regulatory, operational, competitive,
customer, supplier, supply-chain, technology, and market risks.

confidence:
Range [0, 1].

Use high confidence only when the supplied SEC evidence directly
and clearly supports the impact and risk scores.

Use lower confidence when evidence is vague, generic, incomplete,
contradictory, or mixed.

rationale:
Give one concise explanation based only on the supplied evidence.

Do not infer facts not present in the evidence.
Do not use markdown.
Do not output anything outside the JSON object.
""".strip()


def call_remote_qwen(
    client: ollama.Client,
    model_name: str,
    ticker: str,
    event_date: str,
    forms: str,
    retrieved: List[Dict[str, Any]],
    retries: int = 3,
) -> Dict[str, Any]:

    evidence_parts = []

    for item in retrieved:

        queries = ", ".join(
            item[
                "matched_queries"
            ]
        )

        evidence_parts.append(
            (
                f"[Evidence {item['final_rank']}]\n"
                f"Form: {item['form']}\n"
                f"Retrieval score: "
                f"{item['best_score']:.4f}\n"
                f"Matched semantic queries: "
                f"{queries}\n"
                f"SEC text:\n"
                f"{item['text']}"
            )
        )

    evidence = "\n\n".join(
        evidence_parts
    )

    user_prompt = f"""
Ticker: {ticker}
SEC event date: {event_date}
Filing type(s): {forms}

Retrieved SEC evidence:

{evidence}

Evaluate the financial impact, material risk, and confidence.
Return JSON only.
""".strip()

    last_error = None

    for attempt in range(
        1,
        retries + 1,
    ):

        try:

            response = client.chat(
                model=model_name,
                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            SYSTEM_PROMPT,
                    },
                    {
                        "role":
                            "user",

                        "content":
                            user_prompt,
                    },
                ],
                options={
                    "temperature":
                        0,
                },
            )

            raw_answer = clean_str(
                response[
                    "message"
                ][
                    "content"
                ]
            )

            parsed = parse_json_from_text(
                raw_answer
            )

            return {
                "impact":
                    clamp(
                        parsed.get(
                            "impact",
                            0.0,
                        ),
                        -1,
                        1,
                    ),

                "risk":
                    clamp(
                        parsed.get(
                            "risk",
                            0.0,
                        ),
                        0,
                        1,
                    ),

                "confidence":
                    clamp(
                        parsed.get(
                            "confidence",
                            0.0,
                        ),
                        0,
                        1,
                    ),

                "rationale":
                    clean_str(
                        parsed.get(
                            "rationale",
                            ""
                        )
                    ),

                "raw_answer":
                    raw_answer,
            }

        except Exception as e:

            last_error = e

            print(
                f"    Qwen attempt "
                f"{attempt}/{retries} failed: "
                f"{type(e).__name__}: {e}"
            )

            if attempt < retries:
                time.sleep(
                    3 * attempt
                )

    raise RuntimeError(
        "Remote Qwen failed after "
        f"{retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    rows: List[Dict[str, Any]],
    output_path: Path,
):

    if not rows:
        return

    df = pd.DataFrame(
        rows
    )

    df[
        "available_date"
    ] = pd.to_datetime(
        df[
            "available_date"
        ],
        errors="coerce",
    )

    df = (
        df
        .sort_values(
            [
                "available_date",
                "ticker",
            ]
        )
        .drop_duplicates(
            subset=[
                "ticker",
                "available_date",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )


# ============================================================
# QA
# ============================================================

def print_final_qa(
    df: pd.DataFrame,
):

    print()
    print("=" * 70)
    print("FINAL QA")
    print("=" * 70)

    print(
        "TOTAL:",
        len(df),
    )

    print(
        "RANGE:",
        df[
            "available_date"
        ].min(),
        "->",
        df[
            "available_date"
        ].max(),
    )

    print()
    print("BY TICKER")

    print(
        df.groupby(
            "ticker"
        )[
            "available_date"
        ].agg(
            [
                "min",
                "max",
                "count",
            ]
        )
    )

    temp = df.copy()

    temp[
        "year"
    ] = pd.to_datetime(
        temp[
            "available_date"
        ]
    ).dt.year

    print()
    print("BY YEAR")

    print(
        temp.groupby(
            "year"
        ).size()
    )

    print()
    print("MISSING SCORES")

    print(
        df[
            [
                "impact",
                "risk",
                "confidence",
            ]
        ]
        .isna()
        .sum()
    )

    print()
    print("SCORE RANGE")

    print(
        df[
            [
                "impact",
                "risk",
                "confidence",
            ]
        ].agg(
            [
                "min",
                "max",
            ]
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metadata",
        default=(
            "rag/data/index/"
            "sec_metadata.jsonl"
        ),
    )

    parser.add_argument(
        "--embeddings",
        default=(
            "rag/data/index/"
            "sec_embeddings.npy"
        ),
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            "BAAI/"
            "bge-small-en-v1.5"
        ),
    )

    parser.add_argument(
        "--ollama-host",
        default=(
            "http://140.113.87.164:"
            "11434"
        ),
    )

    parser.add_argument(
        "--llm-model",
        default=(
            "qwen3.6:35b"
        ),
    )

    parser.add_argument(
        "--start-date",
        default=(
            "2018-01-01"
        ),
    )

    parser.add_argument(
        "--end-date",
        default=(
            "2026-09-14"
        ),
    )

    parser.add_argument(
        "--per-query-k",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--output",
        default=(
            "rag/data/features/"
            "sec_rag_events_"
            "2018_2026_multiquery.csv"
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SEC-RAG MULTI-QUERY PIPELINE")
    print("=" * 70)

    print(
        "Period:",
        args.start_date,
        "->",
        args.end_date,
    )

    print(
        "Embedding model:",
        args.embedding_model,
    )

    print(
        "LLM:",
        args.llm_model,
    )

    print(
        "Ollama:",
        args.ollama_host,
    )

    print(
        "Retrieval:",
        f"{len(QUERY_TEMPLATES)} queries",
        f"x Top-{args.per_query_k}",
        "-> dedup ->",
        f"Top-{args.final_k}",
    )

    # ========================================================
    # Load index
    # ========================================================

    metadata = load_metadata(
        args.metadata
    )

    embeddings = np.load(
        args.embeddings
    )

    if (
        len(metadata)
        != embeddings.shape[0]
    ):

        raise ValueError(
            "Metadata / embedding row mismatch: "
            f"{len(metadata)} vs "
            f"{embeddings.shape[0]}"
        )

    print()
    print(
        "Metadata rows:",
        len(metadata),
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Metadata range:",
        metadata[
            "available_date"
        ].min(),
        "->",
        metadata[
            "available_date"
        ].max(),
    )

    # ========================================================
    # Filter event universe
    # ========================================================

    start_date = pd.Timestamp(
        args.start_date
    )

    end_date = pd.Timestamp(
        args.end_date
    )

    metadata = metadata[
        (
            metadata[
                "available_date"
            ] >= start_date
        )
        &
        (
            metadata[
                "available_date"
            ] <= end_date
        )
        &
        (
            metadata[
                "ticker"
            ].isin(
                PORTFOLIO
            )
        )
    ].copy()

    events = (
        metadata[
            [
                "ticker",
                "available_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "available_date",
                "ticker",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print(
        "Unique events:",
        len(events),
    )

    print()

    print(
        events.groupby(
            "ticker"
        ).size()
    )

    # ========================================================
    # Load BGE locally
    # ========================================================

    print()
    print("=" * 70)
    print("BUILD MULTI-QUERY EMBEDDINGS")
    print("=" * 70)

    encoder = SentenceTransformer(
        args.embedding_model,
        device=(
            "cuda"
        ),
    )

    query_vectors = build_query_vectors(
        encoder
    )

    first_vector = (
        query_vectors[
            PORTFOLIO[0]
        ][
            next(
                iter(
                    QUERY_TEMPLATES
                )
            )
        ]
    )

    if (
        first_vector.shape[0]
        != embeddings.shape[1]
    ):

        raise ValueError(
            "Query embedding dimension "
            "does not match SEC index: "
            f"{first_vector.shape[0]} "
            f"vs "
            f"{embeddings.shape[1]}"
        )

    print(
        "Query vector dimension:",
        first_vector.shape[0],
    )

    print(
        "Queries per ticker:",
        len(
            QUERY_TEMPLATES
        ),
    )

    del encoder

    # ========================================================
    # Remote Ollama
    # ========================================================

    print()
    print("=" * 70)
    print("TEST REMOTE QWEN")
    print("=" * 70)

    client = ollama.Client(
        host=args.ollama_host
    )

    test_response = client.chat(
        model=args.llm_model,
        messages=[
            {
                "role":
                    "user",

                "content":
                    'Return exactly '
                    '{"status":"OK"}',
            }
        ],
        options={
            "temperature":
                0,
        },
    )

    print(
        "Remote test:",
        clean_str(
            test_response[
                "message"
            ][
                "content"
            ]
        )
    )

    # ========================================================
    # Resume checkpoint
    # ========================================================

    output_path = Path(
        args.output
    )

    generated_rows: List[
        Dict[str, Any]
    ] = []

    completed_keys = set()

    if output_path.exists():

        checkpoint = pd.read_csv(
            output_path
        )

        if len(checkpoint) > 0:

            checkpoint[
                "available_date"
            ] = pd.to_datetime(
                checkpoint[
                    "available_date"
                ],
                errors="coerce",
            )

            generated_rows = (
                checkpoint.to_dict(
                    "records"
                )
            )

            for _, row in checkpoint.iterrows():

                if pd.isna(
                    row[
                        "available_date"
                    ]
                ):
                    continue

                completed_keys.add(
                    (
                        clean_str(
                            row[
                                "ticker"
                            ]
                        ).upper(),

                        pd.Timestamp(
                            row[
                                "available_date"
                            ]
                        ).strftime(
                            "%Y-%m-%d"
                        ),
                    )
                )

    print()
    print(
        "Previously completed:",
        len(
            completed_keys
        ),
    )

    # ========================================================
    # Event loop
    # ========================================================

    total_events = len(
        events
    )

    for event_idx, event in events.iterrows():

        ticker = clean_str(
            event[
                "ticker"
            ]
        ).upper()

        event_date = pd.Timestamp(
            event[
                "available_date"
            ]
        )

        event_date_str = (
            event_date.strftime(
                "%Y-%m-%d"
            )
        )

        event_key = (
            ticker,
            event_date_str,
        )

        if event_key in completed_keys:

            print(
                f"[{event_idx + 1}/"
                f"{total_events}] "
                f"SKIP "
                f"{ticker} "
                f"{event_date_str}"
            )

            continue

        event_df = metadata[
            (
                metadata[
                    "ticker"
                ] == ticker
            )
            &
            (
                metadata[
                    "available_date"
                ] == event_date
            )
        ].copy()

        if len(event_df) == 0:

            print(
                f"[{event_idx + 1}/"
                f"{total_events}] "
                f"NO CHUNKS "
                f"{ticker} "
                f"{event_date_str}"
            )

            continue

        forms = "|".join(
            sorted(
                set(
                    event_df[
                        "form"
                    ]
                    .dropna()
                    .astype(str)
                )
            )
        )

        try:

            retrieved = retrieve_multiquery(
                event_df=
                    event_df,

                all_embeddings=
                    embeddings,

                query_vectors=
                    query_vectors[
                        ticker
                    ],

                per_query_k=
                    args.per_query_k,

                final_k=
                    args.final_k,
            )

            if len(retrieved) == 0:

                raise ValueError(
                    "Retrieval returned "
                    "zero chunks."
                )

            signal = call_remote_qwen(
                client=
                    client,

                model_name=
                    args.llm_model,

                ticker=
                    ticker,

                event_date=
                    event_date_str,

                forms=
                    forms,

                retrieved=
                    retrieved,
            )

            retrieval_scores = [
                x[
                    "best_score"
                ]
                for x in retrieved
            ]

            retrieved_ids = [
                x[
                    "chunk_id"
                ]
                for x in retrieved
            ]

            retrieved_queries = [
                {
                    "chunk_id":
                        x[
                            "chunk_id"
                        ],

                    "matched_queries":
                        x[
                            "matched_queries"
                        ],

                    "query_coverage":
                        x[
                            "query_coverage"
                        ],

                    "best_score":
                        x[
                            "best_score"
                        ],
                }
                for x in retrieved
            ]

            row = {
                "pipeline_version":
                    "sec_rag_multiquery_v1",

                "status":
                    "ok",

                "ticker":
                    ticker,

                "available_date":
                    event_date_str,

                "filing_type":
                    forms,

                "impact":
                    signal[
                        "impact"
                    ],

                "risk":
                    signal[
                        "risk"
                    ],

                "confidence":
                    signal[
                        "confidence"
                    ],

                "rationale":
                    signal[
                        "rationale"
                    ],

                "source_chunk_count":
                    len(
                        event_df
                    ),

                "source_doc_count":
                    event_df[
                        "doc_id"
                    ].nunique(),

                "query_count":
                    len(
                        QUERY_TEMPLATES
                    ),

                "per_query_k":
                    args.per_query_k,

                "final_k":
                    args.final_k,

                "retrieved_chunk_count":
                    len(
                        retrieved
                    ),

                "retrieval_score_mean":
                    float(
                        np.mean(
                            retrieval_scores
                        )
                    ),

                "retrieval_score_max":
                    float(
                        np.max(
                            retrieval_scores
                        )
                    ),

                "retrieved_chunk_ids":
                    json.dumps(
                        retrieved_ids,
                        ensure_ascii=False,
                    ),

                "retrieval_query_details":
                    json.dumps(
                        retrieved_queries,
                        ensure_ascii=False,
                    ),

                "embedding_model":
                    args.embedding_model,

                "llm_model":
                    args.llm_model,

                "llm_host":
                    args.ollama_host,

                "raw_model_output":
                    signal[
                        "raw_answer"
                    ],
            }

            generated_rows.append(
                row
            )

            completed_keys.add(
                event_key
            )

            save_checkpoint(
                generated_rows,
                output_path,
            )

            print(
                f"[{event_idx + 1}/"
                f"{total_events}] "
                f"OK "
                f"{ticker} "
                f"{event_date_str} "
                f"I={row['impact']:+.3f} "
                f"R={row['risk']:.3f} "
                f"C={row['confidence']:.3f} "
                f"chunks="
                f"{row['retrieved_chunk_count']}"
            )

        except Exception as e:

            print(
                f"[{event_idx + 1}/"
                f"{total_events}] "
                f"FAIL "
                f"{ticker} "
                f"{event_date_str}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # ========================================================
    # Final
    # ========================================================

    if output_path.exists():

        final_df = pd.read_csv(
            output_path
        )

        final_df[
            "available_date"
        ] = pd.to_datetime(
            final_df[
                "available_date"
            ],
            errors="coerce",
        )

        print_final_qa(
            final_df
        )

        print()
        print(
            "Saved ->",
            output_path,
        )

    else:

        print()
        print(
            "ERROR: No output "
            "file generated."
        )


if __name__ == "__main__":
    main()
