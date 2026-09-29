import json
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "processed"
    / "sec"
    / "sec_chunks.jsonl"
)

INDEX_DIR = (
    ROOT
    / "rag"
    / "data"
    / "index"
)

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EMBEDDING_PATH = (
    INDEX_DIR
    / "sec_embeddings.npy"
)

METADATA_PATH = (
    INDEX_DIR
    / "sec_metadata.jsonl"
)


MODEL_NAME = "BAAI/bge-small-en-v1.5"

BATCH_SIZE = 32


print("=" * 70)
print("SEC EMBEDDING BUILD")
print("=" * 70)

print("Model:", MODEL_NAME)
print("CUDA:", torch.cuda.is_available())


# ============================================================
# Load chunks
# ============================================================

documents = []

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        if not line.strip():
            continue

        documents.append(
            json.loads(line)
        )


print(
    "Chunks:",
    len(documents),
)


texts = [
    doc["text"]
    for doc in documents
]


# ============================================================
# Load embedding model
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = SentenceTransformer(
    MODEL_NAME,
    device=device,
)


# ============================================================
# Embed
#
# normalize_embeddings=True means cosine similarity becomes
# simple dot product later.
# ============================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)


embeddings = embeddings.astype(
    np.float32
)


print(
    "\nEmbedding shape:",
    embeddings.shape,
)


# ============================================================
# Save vectors
# ============================================================

np.save(
    EMBEDDING_PATH,
    embeddings,
)


# ============================================================
# Save metadata in exact same row order
# ============================================================

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    for doc in documents:

        metadata = {
            "chunk_id": doc["chunk_id"],
            "doc_id": doc["doc_id"],
            "ticker": doc["ticker"],
            "published_at": doc["published_at"],
            "source_type": doc["source_type"],
            "source": doc["source"],
            "form": doc["form"],
            "title": doc["title"],
            "url": doc["url"],
            "chunk_index": doc["chunk_index"],
            "text": doc["text"],
        }

        f.write(
            json.dumps(
                metadata,
                ensure_ascii=False,
            )
            + "\n"
        )


print("\nSaved embeddings:")
print(EMBEDDING_PATH)

print("\nSaved metadata:")
print(METADATA_PATH)

print("\nDONE")