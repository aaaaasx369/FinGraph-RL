import os
import re
import json
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[2]

OUT_DIR = ROOT / "rag" / "data" / "raw" / "sec"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUT_DIR / "sec_documents.jsonl"


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
}

START_YEAR = 2018
END_YEAR = 2023


USER_AGENT = os.getenv("SEC_USER_AGENT")

if not USER_AGENT:
    raise RuntimeError(
        'Please set SEC_USER_AGENT first.\n'
        'Example:\n'
        'export SEC_USER_AGENT="YourName your@email.com"'
    )


HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}

session = requests.Session()
session.headers.update(HEADERS)


# ============================================================
# SEC HTTP helpers
# ============================================================

def get_json(url):

    r = session.get(
        url,
        timeout=60,
    )

    r.raise_for_status()

    # Stay comfortably below SEC request-rate limit
    time.sleep(0.15)

    return r.json()


def get_text(url):

    r = session.get(
        url,
        timeout=60,
    )

    r.raise_for_status()

    time.sleep(0.15)

    return r.text


# ============================================================
# Better human-readable extraction
# ============================================================

def extract_visible_text(html):

    soup = BeautifulSoup(
        html,
        "lxml",
    )

    # --------------------------------------------------------
    # Remove things that are definitely not document content
    # --------------------------------------------------------

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
        ]
    ):
        tag.decompose()

    # --------------------------------------------------------
    # Inline XBRL header / hidden blocks
    #
    # SEC says ix:hidden content is not intended for browser
    # display. Remove it before get_text().
    # --------------------------------------------------------

    for tag in soup.find_all():
        name = (
            tag.name.lower()
            if tag.name
            else ""
        )

        if name in {
            "ix:hidden",
            "ix:header",
        }:
            tag.decompose()

    # --------------------------------------------------------
    # Remove elements explicitly hidden with CSS / HTML
    # --------------------------------------------------------

    for tag in soup.find_all(
        style=True
    ):

        style = (
            tag.get("style", "")
            .replace(" ", "")
            .lower()
        )

        if (
            "display:none" in style
            or "visibility:hidden" in style
        ):
            tag.decompose()

    for tag in soup.find_all(
        attrs={"hidden": True}
    ):
        tag.decompose()

    # IMPORTANT:
    # Do NOT delete tables.
    # SEC filings contain real financial/narrative content
    # inside tables.

    text = soup.get_text(
        " ",
        strip=True,
    )

    # --------------------------------------------------------
    # Basic cleanup
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# Get filing metadata
#
# Main SEC submissions file contains at least one year or
# 1,000 recent filings. Older filing history can be referenced
# through filings.files, so support both.
# ============================================================

def rows_from_submission_json(data):

    recent = data[
        "filings"
    ][
        "recent"
    ]

    rows = []

    n = len(
        recent["accessionNumber"]
    )

    for i in range(n):

        rows.append(
            {
                "accessionNumber":
                    recent["accessionNumber"][i],

                "form":
                    recent["form"][i],

                "filingDate":
                    recent["filingDate"][i],

                "primaryDocument":
                    recent["primaryDocument"][i],
            }
        )

    return rows


def get_all_filing_rows(cik):

    submission_url = (
        "https://data.sec.gov/"
        f"submissions/CIK{cik}.json"
    )

    main = get_json(
        submission_url
    )

    rows = rows_from_submission_json(
        main
    )

    # If SEC provides additional historical submission files,
    # load those as well.
    extra_files = (
        main
        .get("filings", {})
        .get("files", [])
    )

    for item in extra_files:

        name = item.get(
            "name"
        )

        if not name:
            continue

        extra_url = (
            "https://data.sec.gov/"
            "submissions/"
            f"{name}"
        )

        try:

            extra = get_json(
                extra_url
            )

        except Exception as e:

            print(
                "WARNING historical metadata failed:",
                extra_url,
                e,
            )

            continue

        # Historical files use the same compact arrays
        if "accessionNumber" in extra:

            n = len(
                extra["accessionNumber"]
            )

            for i in range(n):

                rows.append(
                    {
                        "accessionNumber":
                            extra["accessionNumber"][i],

                        "form":
                            extra["form"][i],

                        "filingDate":
                            extra["filingDate"][i],

                        "primaryDocument":
                            extra["primaryDocument"][i],
                    }
                )

    return rows


# ============================================================
# Main
# ============================================================

documents = []

seen_accessions = set()


for ticker, cik in COMPANIES.items():

    print(
        "\n" + "=" * 70
    )

    print(
        "Downloading:",
        ticker,
    )

    print(
        "=" * 70
    )

    filing_rows = get_all_filing_rows(
        cik
    )

    cik_no_zero = str(
        int(cik)
    )

    ticker_count = 0

    for row in filing_rows:

        accession = row[
            "accessionNumber"
        ]

        form = row[
            "form"
        ]

        filing_date = row[
            "filingDate"
        ]

        primary_doc = row[
            "primaryDocument"
        ]

        if accession in seen_accessions:
            continue

        try:
            year = int(
                filing_date[:4]
            )

        except Exception:
            continue

        if not (
            START_YEAR
            <= year
            <= END_YEAR
        ):
            continue

        if form not in FORMS:
            continue

        seen_accessions.add(
            accession
        )

        accession_clean = (
            accession.replace(
                "-",
                "",
            )
        )

        filing_url = (
            "https://www.sec.gov/"
            "Archives/edgar/data/"
            f"{cik_no_zero}/"
            f"{accession_clean}/"
            f"{primary_doc}"
        )

        print(
            ticker,
            filing_date,
            form,
        )

        try:

            html = get_text(
                filing_url
            )

            text = extract_visible_text(
                html
            )

        except Exception as e:

            print(
                "FAILED:",
                filing_url,
                e,
            )

            continue

        if len(text) < 500:

            print(
                "SKIP short filing:",
                len(text),
            )

            continue

        doc_id = (
            f"SEC_{ticker}_"
            f"{filing_date}_"
            f"{form}_"
            f"{accession_clean}"
        )

        documents.append(
            {
                "doc_id": doc_id,
                "ticker": ticker,
                "published_at": filing_date,
                "source_type": "sec_filing",
                "source": "SEC EDGAR",
                "form": form,
                "title": (
                    f"{ticker} "
                    f"{form} "
                    f"{filing_date}"
                ),
                "url": filing_url,
                "text": text,
            }
        )

        ticker_count += 1

    print(
        ticker,
        "documents:",
        ticker_count,
    )


# ============================================================
# Save JSONL
# ============================================================

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    for doc in documents:

        f.write(
            json.dumps(
                doc,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "\n" + "=" * 70
)

print(
    "SEC DOWNLOAD COMPLETE"
)

print(
    "=" * 70
)

print(
    "Total documents:",
    len(documents),
)

print(
    "Saved:",
    OUTPUT_PATH,
)