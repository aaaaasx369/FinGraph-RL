from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


CHUNK_WORDS = 300
OVERLAP_WORDS = 60
MIN_CHUNK_WORDS = 60


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def chunk_words(
    text: str,
    chunk_words: int,
    overlap_words: int,
    min_chunk_words: int,
) -> List[str]:

    words = text.split()

    if not words:
        return []

    chunks = []

    step = max(
        1,
        chunk_words - overlap_words,
    )

    start = 0

    while start < len(words):

        end = min(
            len(words),
            start + chunk_words,
        )

        piece = words[start:end]

        if len(piece) >= min_chunk_words:
            chunks.append(
                " ".join(piece)
            )

        if end >= len(words):
            break

        start += step

    return chunks


def load_documents(
    path: Path,
) -> List[Dict]:

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line_no, line in enumerate(
            f,
            1,
        ):

            if not line.strip():
                continue

            try:
                row = json.loads(line)
                rows.append(row)

            except Exception as e:
                print(
                    f"WARNING line {line_no}: "
                    f"{type(e).__name__}: {e}"
                )

    return rows


def build_chunks(
    docs: List[Dict],
) -> List[Dict]:

    output = []

    total_docs = len(docs)

    for doc_idx, doc in enumerate(
        docs,
        1,
    ):

        doc_id = str(
            doc.get(
                "doc_id",
                "",
            )
        ).strip()

        ticker = str(
            doc.get(
                "ticker",
                "",
            )
        ).strip().upper()

        published_at = str(
            doc.get(
                "published_at",
                "",
            )
        ).strip()

        source_type = str(
            doc.get(
                "source_type",
                "sec_filing",
            )
        ).strip()

        source = str(
            doc.get(
                "source",
                "SEC EDGAR",
            )
        ).strip()

        form = str(
            doc.get(
                "form",
                "",
            )
        ).strip()

        title = str(
            doc.get(
                "title",
                "",
            )
        ).strip()

        url = str(
            doc.get(
                "url",
                "",
            )
        ).strip()

        text = clean_text(
            doc.get(
                "text",
                "",
            )
        )

        if not doc_id:
            print(
                f"SKIP doc {doc_idx}: "
                f"missing doc_id"
            )
            continue

        if not ticker:
            print(
                f"SKIP {doc_id}: "
                f"missing ticker"
            )
            continue

        if not text:
            print(
                f"SKIP {doc_id}: "
                f"empty text"
            )
            continue

        pieces = chunk_words(
            text=text,
            chunk_words=CHUNK_WORDS,
            overlap_words=OVERLAP_WORDS,
            min_chunk_words=MIN_CHUNK_WORDS,
        )

        for chunk_index, chunk_text in enumerate(
            pieces
        ):

            chunk_id = (
                f"{doc_id}_chunk_"
                f"{chunk_index:04d}"
            )

            output.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "ticker": ticker,
                    "published_at": published_at,
                    "source_type": source_type,
                    "source": source,
                    "form": form,
                    "title": title,
                    "url": url,
                    "chunk_index": chunk_index,
                    "text": chunk_text,
                }
            )

        if (
            doc_idx % 50 == 0
            or doc_idx == total_docs
        ):
            print(
                f"Processed "
                f"{doc_idx}/{total_docs} docs "
                f"| chunks={len(output)}"
            )

    return output


def save_chunks(
    path: Path,
    chunks: List[Dict],
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in chunks:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temp_path.replace(
        path
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=(
            "rag/data/raw/sec/"
            "sec_documents.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "rag/data/processed/sec/"
            "sec_chunks.jsonl"
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    output_path = Path(
        args.output
    )

    print("=" * 70)
    print("SEC PROCESSOR")
    print("=" * 70)

    print(
        "Input:",
        input_path,
    )

    print(
        "Output:",
        output_path,
    )

    print(
        "CHUNK_WORDS:",
        CHUNK_WORDS,
    )

    print(
        "OVERLAP_WORDS:",
        OVERLAP_WORDS,
    )

    print(
        "MIN_CHUNK_WORDS:",
        MIN_CHUNK_WORDS,
    )

    docs = load_documents(
        input_path
    )

    print()
    print(
        "Documents loaded:",
        len(docs),
    )

    if not docs:
        raise ValueError(
            "No SEC documents found."
        )

    dates = sorted(
        {
            str(
                x.get(
                    "published_at",
                    "",
                )
            )
            for x in docs
            if x.get(
                "published_at"
            )
        }
    )

    if dates:
        print(
            "Raw date range:",
            dates[0],
            "->",
            dates[-1],
        )

    chunks = build_chunks(
        docs
    )

    save_chunks(
        output_path,
        chunks,
    )

    print()
    print("=" * 70)
    print("COMPLETE")
    print("=" * 70)

    print(
        "Documents:",
        len(docs),
    )

    print(
        "Chunks:",
        len(chunks),
    )

    print(
        "Saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
