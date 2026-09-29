import json
import re
from pathlib import Path


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "rag"
    / "data"
    / "raw"
    / "sec"
    / "sec_documents.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "rag"
    / "data"
    / "processed"
    / "sec"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "sec_chunks.jsonl"
)


# ============================================================
# Chunk settings
#
# BGE-small has limited input length.
# 300 words is much safer than the old 900-word chunks.
# ============================================================

CHUNK_WORDS = 300
OVERLAP_WORDS = 60
MIN_CHUNK_WORDS = 60


# ============================================================
# Text cleaning
# ============================================================

def clean_text(text):

    if not text:
        return ""

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    # Remove obvious remaining XBRL / taxonomy artifacts
    text = re.sub(
        r"\b(?:us-gaap|dei|srt|ifrs-full):?[A-Za-z0-9_.\-]*\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\b(?:fasb|xbrl)\S*",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # XBRL duration-like artifacts
    text = re.sub(
        r"\bP\d+[YMD]\b",
        " ",
        text,
    )

    # Boolean artifacts
    text = re.sub(
        r"\b(?:true|false)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Collapse whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Chunking
# ============================================================

def chunk_text(text):

    words = text.split()

    if len(words) < MIN_CHUNK_WORDS:
        return []

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + CHUNK_WORDS,
            len(words),
        )

        chunk_words = words[start:end]

        if len(chunk_words) >= MIN_CHUNK_WORDS:

            chunk = " ".join(
                chunk_words
            )

            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        # 60-word overlap
        start = end - OVERLAP_WORDS

    return chunks


# ============================================================
# Load SEC filings
# ============================================================

documents = []

with open(
    INPUT_PATH,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        documents.append(
            json.loads(line)
        )


print("=" * 70)
print("SEC RAG PREPROCESSING")
print("=" * 70)

print(
    "Input filings:",
    len(documents),
)

print(
    "Chunk words:",
    CHUNK_WORDS,
)

print(
    "Overlap words:",
    OVERLAP_WORDS,
)


# ============================================================
# Process
# ============================================================

total_chunks = 0

ticker_counts = {}
form_counts = {}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as fout:

    for doc in documents:

        cleaned = clean_text(
            doc.get(
                "text",
                "",
            )
        )

        chunks = chunk_text(
            cleaned
        )

        ticker = doc["ticker"]

        form = doc.get(
            "form",
            "UNKNOWN",
        )

        ticker_counts[ticker] = (
            ticker_counts.get(
                ticker,
                0,
            )
            + len(chunks)
        )

        form_counts[form] = (
            form_counts.get(
                form,
                0,
            )
            + len(chunks)
        )


        for idx, chunk in enumerate(
            chunks
        ):

            chunk_id = (
                f"{doc['doc_id']}"
                f"_chunk_{idx:04d}"
            )

            record = {
                "chunk_id": chunk_id,

                "doc_id": doc[
                    "doc_id"
                ],

                "ticker": ticker,

                "published_at": doc[
                    "published_at"
                ],

                "source_type": doc[
                    "source_type"
                ],

                "source": doc[
                    "source"
                ],

                "form": form,

                "title": doc[
                    "title"
                ],

                "url": doc[
                    "url"
                ],

                "chunk_index": idx,

                "text": chunk,
            }

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            total_chunks += 1


# ============================================================
# Summary
# ============================================================

print(
    "\nTotal chunks:",
    total_chunks,
)

print(
    "\nChunks by ticker:"
)

for ticker in sorted(
    ticker_counts
):
    print(
        ticker,
        ticker_counts[ticker],
    )


print(
    "\nChunks by form:"
)

for form in sorted(
    form_counts
):
    print(
        form,
        form_counts[form],
    )


print(
    "\nSaved:"
)

print(
    OUTPUT_PATH
)

print(
    "\nDONE"
)