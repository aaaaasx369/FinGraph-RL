import json
import argparse
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INDEX_DIR = (
    ROOT
    / "rag"
    / "data"
    / "index"
)

EMBEDDING_PATH = (
    INDEX_DIR
    / "sec_embeddings.npy"
)

METADATA_PATH = (
    INDEX_DIR
    / "sec_metadata.jsonl"
)


# ============================================================
# Model
# ============================================================

MODEL_NAME = "BAAI/bge-small-en-v1.5"

QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
)


# ============================================================
# Command-line arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="Search SEC RAG corpus"
)

parser.add_argument(
    "--query",
    required=True,
    help="Natural-language search query",
)

parser.add_argument(
    "--ticker",
    default=None,
    help="Optional ticker filter, e.g. NVDA",
)

parser.add_argument(
    "--as-of",
    default=None,
    help=(
        "Only use documents published on or before "
        "this date. Format: YYYY-MM-DD"
    ),
)

parser.add_argument(
    "--form",
    default=None,
    help="Optional filing form filter: 10-K, 10-Q, 8-K",
)

parser.add_argument(
    "--top-k",
    type=int,
    default=5,
)

args = parser.parse_args()


# ============================================================
# Check files
# ============================================================

if not EMBEDDING_PATH.exists():

    raise FileNotFoundError(
        f"Embedding file not found:\n{EMBEDDING_PATH}\n"
        "Run build_sec_embeddings.py first."
    )


if not METADATA_PATH.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n{METADATA_PATH}\n"
        "Run build_sec_embeddings.py first."
    )


# ============================================================
# Load embeddings
# ============================================================

print("=" * 70)
print("LOADING SEC VECTOR INDEX")
print("=" * 70)


embeddings = np.load(
    EMBEDDING_PATH
)


metadata = []

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        metadata.append(
            json.loads(line)
        )


if len(metadata) != len(embeddings):

    raise RuntimeError(
        "Metadata / embedding length mismatch.\n"
        f"Metadata: {len(metadata)}\n"
        f"Embeddings: {len(embeddings)}"
    )


print(
    "Total vectors:",
    len(embeddings),
)

print(
    "Vector dimension:",
    embeddings.shape[1],
)


# ============================================================
# Point-in-time + ticker + form filtering
# ============================================================

candidate_indices = []


for i, doc in enumerate(
    metadata
):

    # --------------------------------------------------------
    # Ticker filter
    # --------------------------------------------------------

    if args.ticker:

        if (
            doc["ticker"].upper()
            != args.ticker.upper()
        ):
            continue


    # --------------------------------------------------------
    # Point-in-time filter
    #
    # This is critical for financial backtesting.
    # Future documents must not be visible.
    # --------------------------------------------------------

    if args.as_of:

        if (
            doc["published_at"]
            > args.as_of
        ):
            continue


    # --------------------------------------------------------
    # Filing form filter
    # --------------------------------------------------------

    if args.form:

        if (
            doc["form"].upper()
            != args.form.upper()
        ):
            continue


    candidate_indices.append(
        i
    )


if not candidate_indices:

    raise RuntimeError(
        "No documents remain after filtering."
    )


print("\n" + "=" * 70)
print("SEC RETRIEVER")
print("=" * 70)

print(
    "Query:",
    args.query,
)

print(
    "Ticker:",
    args.ticker,
)

print(
    "As-of:",
    args.as_of,
)

print(
    "Form:",
    args.form,
)

print(
    "Candidate chunks:",
    len(candidate_indices),
)


# ============================================================
# Load BGE embedding model
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "Device:",
    device,
)


model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)


# ============================================================
# Build query embedding
#
# BGE recommends adding a retrieval instruction to queries.
#
# Documents do NOT need this prefix.
# Queries DO use this prefix.
# ============================================================

query_text = (
    QUERY_PREFIX
    + args.query
)


query_embedding = model.encode(
    [query_text],
    normalize_embeddings=True,
    convert_to_numpy=True,
)[0]


query_embedding = (
    query_embedding
    .astype(
        np.float32
    )
)


# ============================================================
# Candidate embeddings
# ============================================================

candidate_embeddings = embeddings[
    candidate_indices
]


# ============================================================
# Cosine similarity
#
# Both document vectors and query vector are normalized.
#
# Therefore:
#
# cosine_similarity(q, d)
# =
# q dot d
# ============================================================

scores = (
    candidate_embeddings
    @ query_embedding
)


# Highest score first
ranking = np.argsort(
    scores
)[::-1]


top_n = min(
    args.top_k,
    len(ranking),
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("TOP RESULTS")
print("=" * 70)


for rank in range(
    top_n
):

    local_idx = ranking[
        rank
    ]

    global_idx = candidate_indices[
        local_idx
    ]

    doc = metadata[
        global_idx
    ]

    score = float(
        scores[
            local_idx
        ]
    )


    print(
        f"\nRANK {rank + 1}"
    )

    print(
        f"Score : {score:.4f}"
    )

    print(
        "Ticker:",
        doc["ticker"],
    )

    print(
        "Date  :",
        doc["published_at"],
    )

    print(
        "Form  :",
        doc["form"],
    )

    print(
        "Title :",
        doc["title"],
    )

    print(
        "Chunk :",
        doc["chunk_index"],
    )

    print(
        "Chunk ID:",
        doc["chunk_id"],
    )

    print(
        "URL   :",
        doc["url"],
    )


    print("\nText:")

    preview = doc[
        "text"
    ][
        :1500
    ]

    print(
        preview
    )

    print(
        "\n"
        + "-"
        * 70
    )