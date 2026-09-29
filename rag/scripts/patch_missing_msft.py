import os
import re
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]

CORPUS = (
    ROOT
    / "rag"
    / "data"
    / "raw"
    / "sec"
    / "sec_documents.jsonl"
)

USER_AGENT = os.getenv("SEC_USER_AGENT")

if not USER_AGENT:
    raise RuntimeError(
        'SEC_USER_AGENT is not set.\n'
        'Example:\n'
        'export SEC_USER_AGENT="YourName your@email.com"'
    )


HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}


session = requests.Session()
session.headers.update(HEADERS)


MISSING = [
    {
        "ticker": "MSFT",
        "date": "2023-04-25",
        "form": "10-Q",
        "accession": "0000950170-23-014423",
        "primary_doc": "msft-20230331.htm",
    },
    {
        "ticker": "MSFT",
        "date": "2023-07-27",
        "form": "10-K",
        "accession": "0000950170-23-035122",
        "primary_doc": "msft-20230630.htm",
    },
    {
        "ticker": "MSFT",
        "date": "2023-10-24",
        "form": "10-Q",
        "accession": "0000950170-23-054855",
        "primary_doc": "msft-20230930.htm",
    },
]


CIK = "789019"


# ============================================================
# Robust SEC HTML -> visible text
# ============================================================

def extract_visible_text(html):

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    # --------------------------------------------------------
    # 1. Remove script/style/noscript
    # Take snapshots before modifying the tree.
    # --------------------------------------------------------

    for tag in list(
        soup.find_all(
            ["script", "style", "noscript"]
        )
    ):
        try:
            tag.decompose()
        except Exception:
            pass

    # --------------------------------------------------------
    # 2. Remove Inline XBRL hidden/header sections
    # --------------------------------------------------------

    tags = list(
        soup.find_all()
    )

    for tag in tags:

        try:
            name = getattr(
                tag,
                "name",
                None,
            )

            if not name:
                continue

            name = str(name).lower()

            if name in {
                "ix:hidden",
                "ix:header",
            }:
                tag.decompose()

        except Exception:
            continue

    # --------------------------------------------------------
    # 3. Remove CSS-hidden elements safely
    # --------------------------------------------------------

    tags = list(
        soup.find_all()
    )

    for tag in tags:

        try:
            attrs = getattr(
                tag,
                "attrs",
                None,
            )

            if not isinstance(
                attrs,
                dict,
            ):
                continue

            style = attrs.get(
                "style",
                "",
            )

            if style is None:
                style = ""

            style = (
                str(style)
                .replace(" ", "")
                .lower()
            )

            hidden_attr = (
                "hidden" in attrs
            )

            if (
                "display:none" in style
                or
                "visibility:hidden" in style
                or
                hidden_attr
            ):
                tag.decompose()

        except Exception:
            continue

    # --------------------------------------------------------
    # IMPORTANT:
    # Do not remove <table>.
    # Real SEC content often lives in tables.
    # --------------------------------------------------------

    text = soup.get_text(
        " ",
        strip=True,
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Read existing corpus
# ============================================================

existing = []

with open(
    CORPUS,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        existing.append(
            json.loads(line)
        )


existing_ids = {
    x["doc_id"]
    for x in existing
}


print(
    "Existing documents:",
    len(existing),
)


added = []


# ============================================================
# Download only the 3 missing filings
# ============================================================

for item in MISSING:

    accession_clean = (
        item["accession"]
        .replace("-", "")
    )

    doc_id = (
        f"SEC_MSFT_"
        f"{item['date']}_"
        f"{item['form']}_"
        f"{accession_clean}"
    )

    if doc_id in existing_ids:

        print(
            "Already exists:",
            doc_id,
        )

        continue

    url = (
        "https://www.sec.gov/"
        "Archives/edgar/data/"
        f"{CIK}/"
        f"{accession_clean}/"
        f"{item['primary_doc']}"
    )

    print("\n" + "=" * 70)
    print(
        "Downloading:",
        item["date"],
        item["form"],
    )
    print(url)

    response = session.get(
        url,
        timeout=120,
    )

    response.raise_for_status()

    print(
        "HTTP:",
        response.status_code,
    )

    print(
        "Downloaded bytes:",
        len(response.content),
    )

    time.sleep(0.2)

    text = extract_visible_text(
        response.text
    )

    print(
        "Extracted characters:",
        len(text),
    )

    print(
        "Text preview:",
        text[:200],
    )

    if len(text) < 1000:

        raise RuntimeError(
            "Extracted filing is unexpectedly short:\n"
            f"{url}"
        )

    record = {
        "doc_id": doc_id,
        "ticker": "MSFT",
        "published_at": item["date"],
        "source_type": "sec_filing",
        "source": "SEC EDGAR",
        "form": item["form"],
        "title": (
            f"MSFT "
            f"{item['form']} "
            f"{item['date']}"
        ),
        "url": url,
        "text": text,
    }

    added.append(
        record
    )


# ============================================================
# Append
# ============================================================

with open(
    CORPUS,
    "a",
    encoding="utf-8",
) as f:

    for record in added:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


print("\n" + "=" * 70)

print(
    "Added:",
    len(added),
)

print(
    "New total:",
    len(existing) + len(added),
)

print("DONE")