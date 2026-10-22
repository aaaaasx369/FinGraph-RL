from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer


PORTFOLIO = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]

QUERIES = [
    (
        "What information in this SEC filing could have a "
        "positive or negative financial impact on the company, "
        "its revenue, profitability, growth, or valuation?"
    ),
    (
        "What material risks, uncertainties, liabilities, "
        "competitive threats, supply constraints, regulatory "
        "issues, or operational risks are disclosed?"
    ),
    (
        "What important changes in business performance, "
        "customers, suppliers, products, competition, capital "
        "spending, margins, or strategy are disclosed?"
    ),
    (
        "What forward-looking information or management "
        "discussion could materially affect future company "
        "performance?"
    ),
]


def clean_str(x: Any) -> str:
    if x is None:
        return ""

    if isinstance(x, float) and np.isnan(x):
        return ""

    return str(x).strip()


def clamp(x: float, low: float, high: float) -> float:
    return max(
        low,
        min(
            high,
            float(x),
        ),
    )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(
        x,
        axis=-1,
        keepdims=True,
    )

    return x / np.maximum(
        norm,
        1e-12,
    )


def parse_json_from_text(
    text: str,
) -> Dict[str, Any]:

    text = text.strip()

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

    match = re.search(
        r"\{.*?\}",
        text,
        flags=re.S,
    )

    if not match:
        raise ValueError(
            "No JSON object found:\n"
            + text[:1000]
        )

    return json.loads(
        match.group(0)
    )


def load_metadata(
    path: str,
) -> pd.DataFrame:

    rows = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for i, line in enumerate(f):

            if not line.strip():
                continue

            x = json.loads(line)

            rows.append(
                {
                    "embedding_index": i,
                    "ticker": clean_str(
                        x.get("ticker")
                    ).upper(),
                    "available_date": (
                        x.get("published_at")
                        or x.get("filing_date")
                        or x.get("date")
                    ),
                    "form": clean_str(
                        x.get("form")
                    ),
                    "chunk_id": clean_str(
                        x.get("chunk_id")
                    ),
                    "doc_id": clean_str(
                        x.get("doc_id")
                    ),
                    "text": clean_str(
                        x.get("text")
                    ),
                    "url": clean_str(
                        x.get("url")
                    ),
                }
            )

    df = pd.DataFrame(rows)

    df["available_date"] = pd.to_datetime(
        df["available_date"],
        errors="coerce",
    )

    return df[
        df["available_date"].notna()
    ].copy()


def load_existing_baseline(
    path: str,
) -> pd.DataFrame:

    p = Path(path)

    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p)

    if "ticker" not in df.columns:
        if "tic" in df.columns:
            df = df.rename(
                columns={
                    "tic": "ticker"
                }
            )

    if "available_date" not in df.columns:

        for c in [
            "filing_date",
            "published_at",
            "date",
        ]:
            if c in df.columns:

                df = df.rename(
                    columns={
                        c: "available_date"
                    }
                )

                break

    if (
        "impact_score" in df.columns
        and
        "impact" not in df.columns
    ):
        df = df.rename(
            columns={
                "impact_score": "impact"
            }
        )

    if (
        "risk_score" in df.columns
        and
        "risk" not in df.columns
    ):
        df = df.rename(
            columns={
                "risk_score": "risk"
            }
        )

    if "ticker" in df.columns:

        df["ticker"] = (
            df["ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

    if "available_date" in df.columns:

        df["available_date"] = pd.to_datetime(
            df["available_date"],
            errors="coerce",
        )

    return df


def retrieve_top_chunks(
    event_df: pd.DataFrame,
    all_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:

    indices = (
        event_df["embedding_index"]
        .astype(int)
        .to_numpy()
    )

    x = (
        all_embeddings[
            indices
        ]
        .astype(
            np.float32
        )
    )

    x = l2_normalize(x)

    score_matrix = (
        query_embeddings
        @
        x.T
    )

    chunk_scores = (
        score_matrix.max(
            axis=0
        )
    )

    order = np.argsort(
        -chunk_scores
    )

    top = order[
        :min(
            top_k,
            len(order),
        )
    ]

    results = []

    for rank, local_idx in enumerate(
        top,
        1,
    ):

        row = event_df.iloc[
            int(local_idx)
        ]

        results.append(
            {
                "rank": rank,
                "score": float(
                    chunk_scores[
                        local_idx
                    ]
                ),
                "chunk_id": row[
                    "chunk_id"
                ],
                "doc_id": row[
                    "doc_id"
                ],
                "form": row[
                    "form"
                ],
                "url": row[
                    "url"
                ],
                "text": row[
                    "text"
                ],
            }
        )

    return results


def call_remote_qwen(
    client,
    model_name: str,
    ticker: str,
    date: str,
    forms: str,
    retrieved: List[
        Dict[str, Any]
    ],
) -> Dict[str, Any]:

    evidence_parts = []

    for x in retrieved:

        evidence_parts.append(
            (
                f"[Chunk {x['rank']} | "
                f"score={x['score']:.4f} | "
                f"form={x['form']}]\n"
                f"{x['text']}"
            )
        )

    evidence = "\n\n".join(
        evidence_parts
    )

    system_prompt = """
You are a financial SEC filing analysis engine.

Analyze ONLY the supplied SEC filing evidence.
Do not use outside knowledge.

Return exactly one JSON object:

{
  "impact": number,
  "risk": number,
  "confidence": number,
  "rationale": string
}

impact:
Range [-1, 1].
-1 = strongly negative expected financial effect.
0 = neutral, mixed, or unclear.
+1 = strongly positive expected financial effect.

risk:
Range [0, 1].
0 = little disclosed material risk.
1 = extremely severe/material disclosed risk.

confidence:
Range [0, 1].
Confidence that the supplied evidence clearly supports the scores.

rationale:
A concise explanation grounded only in the supplied SEC evidence.

Return valid JSON only.
No markdown.
No commentary outside JSON.
""".strip()

    user_prompt = f"""
Ticker: {ticker}
SEC public date: {date}
Filing type(s): {forms}

Retrieved SEC evidence:

{evidence}
""".strip()

    response = client.chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    answer = (
        response[
            "message"
        ][
            "content"
        ]
        .strip()
    )

    parsed = parse_json_from_text(
        answer
    )

    return {
        "impact": clamp(
            parsed.get(
                "impact",
                0.0,
            ),
            -1,
            1,
        ),

        "risk": clamp(
            parsed.get(
                "risk",
                0.0,
            ),
            0,
            1,
        ),

        "confidence": clamp(
            parsed.get(
                "confidence",
                0.0,
            ),
            0,
            1,
        ),

        "rationale": clean_str(
            parsed.get(
                "rationale",
                ""
            )
        ),

        "raw_answer": answer,
    }


def save_checkpoint(
    rows: List[
        Dict[str, Any]
    ],
    path: Path,
):

    if not rows:
        return

    df = pd.DataFrame(rows)

    df["available_date"] = pd.to_datetime(
        df["available_date"],
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
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )


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
        "--existing-events",
        default=(
            "rag/data/features/"
            "sec_rag_events.csv"
        ),
    )

    parser.add_argument(
        "--start-date",
        default="2024-01-01",
    )

    parser.add_argument(
        "--end-date",
        default="2026-09-14",
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
            "http://140.113.87.164:11434"
        ),
    )

    parser.add_argument(
        "--llm-model",
        default="qwen3.6:35b",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--new-events-output",
        default=(
            "rag/data/features/"
            "sec_rag_events_2024_2026.csv"
        ),
    )

    parser.add_argument(
        "--merged-output",
        default=(
            "rag/data/features/"
            "sec_rag_events_2018_2026.csv"
        ),
    )

    args = parser.parse_args()

    print("=" * 70)
    print("SEC-RAG 2024-2026")
    print(".166 RETRIEVAL -> .164 REMOTE QWEN")
    print("=" * 70)

    metadata = load_metadata(
        args.metadata
    )

    embeddings = np.load(
        args.embeddings
    )

    if (
        len(metadata)
        !=
        embeddings.shape[0]
    ):
        raise ValueError(
            f"Metadata/embedding mismatch: "
            f"{len(metadata)} vs "
            f"{embeddings.shape[0]}"
        )

    print(
        "Metadata:",
        len(metadata),
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "SEC range:",
        metadata[
            "available_date"
        ].min(),
        "->",
        metadata[
            "available_date"
        ].max(),
    )

    start = pd.Timestamp(
        args.start_date
    )

    end = pd.Timestamp(
        args.end_date
    )

    recent = metadata[
        (
            metadata[
                "available_date"
            ] >= start
        )
        &
        (
            metadata[
                "available_date"
            ] <= end
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
        recent[
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
        "Event dates:",
        len(events),
    )

    print()

    print(
        events.groupby(
            "ticker"
        ).size()
    )

    # ========================================================
    # BGE on local .166
    # ========================================================

    print()
    print("=" * 70)
    print("LOCAL BGE RETRIEVAL")
    print("=" * 70)

    encoder = SentenceTransformer(
        args.embedding_model,
        device="cuda:0",
    )

    query_embeddings = encoder.encode(
        QUERIES,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(
        np.float32
    )

    if (
        query_embeddings.shape[1]
        !=
        embeddings.shape[1]
    ):

        raise ValueError(
            "BGE embedding dimension mismatch: "
            f"{query_embeddings.shape[1]} "
            f"vs "
            f"{embeddings.shape[1]}"
        )

    print(
        "Query embeddings:",
        query_embeddings.shape,
    )

    del encoder

    # ========================================================
    # Remote .164 Ollama
    # ========================================================

    print()
    print("=" * 70)
    print("REMOTE QWEN")
    print("=" * 70)

    print(
        "Host:",
        args.ollama_host,
    )

    print(
        "Model:",
        args.llm_model,
    )

    client = ollama.Client(
        host=args.ollama_host
    )

    # Small connectivity test
    test_response = client.chat(
        model=args.llm_model,
        messages=[
            {
                "role": "user",
                "content": (
                    'Return exactly '
                    '{"status":"OK"}'
                ),
            }
        ],
        options={
            "temperature": 0,
        },
    )

    print(
        "Connection test:",
        test_response[
            "message"
        ][
            "content"
        ][:100]
    )

    # ========================================================
    # Resume checkpoint
    # ========================================================

    new_output = Path(
        args.new_events_output
    )

    generated_rows = []
    done = set()

    if new_output.exists():

        checkpoint = pd.read_csv(
            new_output
        )

        if len(checkpoint):

            checkpoint[
                "available_date"
            ] = pd.to_datetime(
                checkpoint[
                    "available_date"
                ],
                errors="coerce",
            )

            generated_rows = (
                checkpoint
                .to_dict(
                    "records"
                )
            )

            for _, r in checkpoint.iterrows():

                if pd.isna(
                    r[
                        "available_date"
                    ]
                ):
                    continue

                done.add(
                    (
                        clean_str(
                            r[
                                "ticker"
                            ]
                        ).upper(),

                        pd.Timestamp(
                            r[
                                "available_date"
                            ]
                        )
                        .strftime(
                            "%Y-%m-%d"
                        ),
                    )
                )

    print(
        "Checkpoint completed:",
        len(done),
    )

    # ========================================================
    # Generate signals
    # ========================================================

    total = len(events)

    for i, event in events.iterrows():

        ticker = event[
            "ticker"
        ]

        event_date = pd.Timestamp(
            event[
                "available_date"
            ]
        )

        date_str = event_date.strftime(
            "%Y-%m-%d"
        )

        key = (
            ticker,
            date_str,
        )

        if key in done:

            print(
                f"[{i+1}/{total}] "
                f"SKIP "
                f"{ticker} "
                f"{date_str}"
            )

            continue

        group = recent[
            (
                recent[
                    "ticker"
                ] == ticker
            )
            &
            (
                recent[
                    "available_date"
                ] == event_date
            )
        ].copy()

        forms = "|".join(
            sorted(
                set(
                    group[
                        "form"
                    ]
                    .dropna()
                    .astype(str)
                )
            )
        )

        try:

            retrieved = (
                retrieve_top_chunks(
                    event_df=group,
                    all_embeddings=embeddings,
                    query_embeddings=
                        query_embeddings,
                    top_k=args.top_k,
                )
            )

            signal = call_remote_qwen(
                client=client,
                model_name=
                    args.llm_model,
                ticker=ticker,
                date=date_str,
                forms=forms,
                retrieved=retrieved,
            )

            scores = [
                x["score"]
                for x in retrieved
            ]

            row = {
                "ticker":
                    ticker,

                "available_date":
                    date_str,

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

                "retrieval_score_mean":
                    float(
                        np.mean(
                            scores
                        )
                    )
                    if scores
                    else np.nan,

                "retrieval_score_max":
                    float(
                        np.max(
                            scores
                        )
                    )
                    if scores
                    else np.nan,

                "retrieved_chunks":
                    len(
                        retrieved
                    ),

                "source_chunk_count":
                    len(
                        group
                    ),

                "source_doc_count":
                    group[
                        "doc_id"
                    ].nunique(),

                "llm_model":
                    args.llm_model,

                "llm_host":
                    args.ollama_host,

                "embedding_model":
                    args.embedding_model,

                "raw_answer":
                    signal[
                        "raw_answer"
                    ],
            }

            generated_rows.append(
                row
            )

            done.add(
                key
            )

            save_checkpoint(
                generated_rows,
                new_output,
            )

            print(
                f"[{i+1}/{total}] "
                f"OK "
                f"{ticker} "
                f"{date_str} "
                f"I={row['impact']:+.3f} "
                f"R={row['risk']:.3f} "
                f"C={row['confidence']:.3f}"
            )

        except Exception as e:

            print(
                f"[{i+1}/{total}] "
                f"FAIL "
                f"{ticker} "
                f"{date_str}: "
                f"{type(e).__name__}: "
                f"{e}"
            )

    # ========================================================
    # Merge old 2018-2023 + new 2024-2026
    # ========================================================

    new_df = pd.DataFrame(
        generated_rows
    )

    if len(new_df):

        new_df[
            "available_date"
        ] = pd.to_datetime(
            new_df[
                "available_date"
            ],
            errors="coerce",
        )

    old_df = load_existing_baseline(
        args.existing_events
    )

    if len(old_df):

        old_df = old_df[
            old_df[
                "available_date"
            ] < start
        ].copy()

    all_columns = sorted(
        set(
            old_df.columns
            if len(old_df)
            else []
        )
        |
        set(
            new_df.columns
            if len(new_df)
            else []
        )
    )

    for col in all_columns:

        if (
            len(old_df)
            and
            col not in old_df.columns
        ):
            old_df[col] = np.nan

        if (
            len(new_df)
            and
            col not in new_df.columns
        ):
            new_df[col] = np.nan

    pieces = []

    if len(old_df):
        pieces.append(
            old_df[
                all_columns
            ]
        )

    if len(new_df):
        pieces.append(
            new_df[
                all_columns
            ]
        )

    if not pieces:
        raise ValueError(
            "No SEC-RAG events generated."
        )

    merged = pd.concat(
        pieces,
        ignore_index=True,
    )

    merged[
        "available_date"
    ] = pd.to_datetime(
        merged[
            "available_date"
        ],
        errors="coerce",
    )

    merged = (
        merged
        .dropna(
            subset=[
                "ticker",
                "available_date",
            ]
        )
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

    merged.to_csv(
        args.merged_output,
        index=False,
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        "Old events:",
        len(old_df),
    )

    print(
        "New events:",
        len(new_df),
    )

    print(
        "Merged events:",
        len(merged),
    )

    print(
        "Range:",
        merged[
            "available_date"
        ].min(),
        "->",
        merged[
            "available_date"
        ].max(),
    )

    print()

    print(
        merged.groupby(
            "ticker"
        )[
            "available_date"
        ]
        .agg(
            [
                "min",
                "max",
                "count",
            ]
        )
    )

    print()
    print(
        "Saved new ->",
        new_output,
    )

    print(
        "Saved all ->",
        args.merged_output,
    )


if __name__ == "__main__":
    main()
