from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List

import requests
from bs4 import BeautifulSoup


# ============================================================
# Portfolio
# ============================================================

COMPANIES = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
}

FORMS = {
    "10-K",
    "10-Q",
    "8-K",
    "10-K/A",
    "10-Q/A",
    "8-K/A",
}


# ============================================================
# SEC access
# ============================================================

USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "",
).strip()

if not USER_AGENT:
    raise RuntimeError(
        "SEC_USER_AGENT is not set.\n\n"
        "Example:\n"
        "export SEC_USER_AGENT='Your Name your_email@example.com'\n"
    )

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ~5 requests/sec, safely below SEC's 10/sec ceiling.
REQUEST_SLEEP = 0.20


def get_json(url: str) -> dict:
    r = SESSION.get(
        url,
        timeout=60,
    )
    r.raise_for_status()
    time.sleep(REQUEST_SLEEP)
    return r.json()


def get_text(url: str) -> str:
    r = SESSION.get(
        url,
        timeout=60,
    )
    r.raise_for_status()
    time.sleep(REQUEST_SLEEP)
    return r.text


# ============================================================
# Helpers
# ============================================================

def clean_text(html: str) -> str:
    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
        ]
    ):
        tag.decompose()

    text = soup.get_text(
        " ",
        strip=True,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def accession_no_dash(
    accession: str,
) -> str:
    return accession.replace(
        "-",
        "",
    )


def filing_url(
    cik: str,
    accession: str,
    primary_document: str,
) -> str:

    cik_int = str(
        int(cik)
    )

    accession_clean = (
        accession_no_dash(
            accession
        )
    )

    return (
        "https://www.sec.gov/Archives/"
        f"edgar/data/{cik_int}/"
        f"{accession_clean}/"
        f"{primary_document}"
    )


def make_doc_id(
    ticker: str,
    filing_date: str,
    form: str,
    accession: str,
) -> str:

    return (
        f"SEC_{ticker}_"
        f"{filing_date}_"
        f"{form}_"
        f"{accession_no_dash(accession)}"
    )


# ============================================================
# SEC submissions metadata
# ============================================================

def columnar_rows(obj: Dict) -> List[Dict]:

    if not obj:
        return []

    keys = list(obj.keys())

    if not keys:
        return []

    n = len(
        obj[keys[0]]
    )

    rows = []

    for i in range(n):

        row = {}

        for key in keys:

            values = obj.get(
                key,
                [],
            )

            row[key] = (
                values[i]
                if i < len(values)
                else ""
            )

        rows.append(row)

    return rows


def get_all_submission_rows(
    cik: str,
) -> List[Dict]:

    cik10 = str(cik).zfill(10)

    root_url = (
        "https://data.sec.gov/"
        f"submissions/CIK{cik10}.json"
    )

    root = get_json(
        root_url
    )

    rows = []

    recent = (
        root.get("filings", {})
        .get("recent", {})
    )

    rows.extend(
        columnar_rows(recent)
    )

    # SEC may move older submission metadata into extra JSON files.
    extra_files = (
        root.get("filings", {})
        .get("files", [])
    )

    for item in extra_files:

        name = item.get(
            "name",
            "",
        )

        if not name:
            continue

        url = (
            "https://data.sec.gov/"
            f"submissions/{name}"
        )

        payload = get_json(
            url
        )

        rows.extend(
            columnar_rows(payload)
        )

    return rows


# ============================================================
# Existing raw docs
# ============================================================

def load_existing(
    path: Path,
):

    docs = {}

    if not path.exists():
        return docs

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            x = json.loads(line)

            doc_id = str(
                x.get(
                    "doc_id",
                    "",
                )
            )

            if doc_id:
                docs[doc_id] = x

    return docs


# ============================================================
# Download one company
# ============================================================

def download_company(
    ticker: str,
    cik: str,
    start_date: str,
    end_date: str,
    existing_ids: set,
):

    rows = get_all_submission_rows(
        cik
    )

    selected = []

    for row in rows:

        form = str(
            row.get(
                "form",
                "",
            )
        ).strip()

        filing_date = str(
            row.get(
                "filingDate",
                "",
            )
        ).strip()

        accession = str(
            row.get(
                "accessionNumber",
                "",
            )
        ).strip()

        primary_document = str(
            row.get(
                "primaryDocument",
                "",
            )
        ).strip()

        if form not in FORMS:
            continue

        if not filing_date:
            continue

        if filing_date < start_date:
            continue

        if filing_date > end_date:
            continue

        if (
            not accession
            or not primary_document
        ):
            continue

        selected.append(row)

    selected.sort(
        key=lambda x: (
            str(
                x.get(
                    "filingDate",
                    "",
                )
            ),
            str(
                x.get(
                    "accessionNumber",
                    "",
                )
            ),
        )
    )

    print()
    print(
        f"{ticker}: "
        f"{len(selected)} filings "
        f"{start_date} -> {end_date}"
    )

    downloaded = []

    for i, row in enumerate(
        selected,
        1,
    ):

        filing_date = str(
            row["filingDate"]
        )

        form = str(
            row["form"]
        )

        accession = str(
            row["accessionNumber"]
        )

        primary_document = str(
            row["primaryDocument"]
        )

        doc_id = make_doc_id(
            ticker,
            filing_date,
            form,
            accession,
        )

        if doc_id in existing_ids:

            print(
                f"[{i:03d}/{len(selected):03d}] "
                f"SKIP {doc_id}"
            )

            continue

        url = filing_url(
            cik,
            accession,
            primary_document,
        )

        try:

            html = get_text(
                url
            )

            text = clean_text(
                html
            )

            if not text:

                print(
                    f"[{i:03d}/{len(selected):03d}] "
                    f"EMPTY {doc_id}"
                )

                continue

            document = {
                "doc_id": doc_id,
                "ticker": ticker,
                "published_at": filing_date,
                "source_type": "sec_filing",
                "source": "SEC EDGAR",
                "form": form,
                "title": (
                    f"{ticker} {form} "
                    f"{filing_date}"
                ),
                "url": url,
                "accession_number": accession,
                "primary_document": primary_document,
                "cik": str(cik).zfill(10),
                "text": text,
            }

            downloaded.append(
                document
            )

            existing_ids.add(
                doc_id
            )

            print(
                f"[{i:03d}/{len(selected):03d}] "
                f"OK   {doc_id} "
                f"chars={len(text):,}"
            )

        except Exception as e:

            print(
                f"[{i:03d}/{len(selected):03d}] "
                f"FAIL {doc_id}: "
                f"{type(e).__name__}: {e}"
            )

    return downloaded


# ============================================================
# Atomic save
# ============================================================

def save_documents(
    path: Path,
    docs: Dict[str, Dict],
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    ordered = sorted(
        docs.values(),
        key=lambda x: (
            str(
                x.get(
                    "published_at",
                    "",
                )
            ),
            str(
                x.get(
                    "ticker",
                    "",
                )
            ),
            str(
                x.get(
                    "doc_id",
                    "",
                )
            ),
        ),
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:

        for doc in ordered:

            f.write(
                json.dumps(
                    doc,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(
        path
    )


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start-date",
        default="2024-01-01",
    )

    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
    )

    parser.add_argument(
        "--output",
        default=(
            "rag/data/raw/sec/"
            "sec_documents.jsonl"
        ),
    )

    args = parser.parse_args()

    output = Path(
        args.output
    )

    existing = load_existing(
        output
    )

    print(
        "Existing documents:",
        len(existing),
    )

    print(
        "Update period:",
        args.start_date,
        "->",
        args.end_date,
    )

    existing_ids = set(
        existing.keys()
    )

    total_new = 0

    for ticker, cik in COMPANIES.items():

        new_docs = download_company(
            ticker=ticker,
            cik=cik,
            start_date=args.start_date,
            end_date=args.end_date,
            existing_ids=existing_ids,
        )

        for doc in new_docs:
            existing[
                doc["doc_id"]
            ] = doc

        total_new += len(
            new_docs
        )

        # checkpoint after every company
        save_documents(
            output,
            existing,
        )

    print()
    print("=" * 70)
    print("SEC INCREMENTAL DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        "New documents:",
        total_new,
    )

    print(
        "Total documents:",
        len(existing),
    )

    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()
