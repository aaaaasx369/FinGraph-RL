#!/usr/bin/env python3
"""
V20 SEC EDGAR PIT evidence builder.

Five companies, historical SEC 10-K/10-Q.
- Fetch official SEC submissions metadata and historical pages
- Record accession, acceptance timestamp, URL
- Optionally fetch original HTML/TXT filing documents
- Cache responses, SHA256 evidence
- Refuse unsupported dates and failed downloads
- Build daily as-of SEC features with explicit evidence IDs
- Strictly use filings accepted BEFORE the market date
- No forward backfill
- No FinDKG provenance certification
- Exploratory full text keyword features, NOT realized outcomes

Python stdlib only.
"""

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import date, datetime, timezone

CIKS = {
    "AAPL": 320193,
    "MSFT": 789019,
    "NVDA": 1045810,
    "AMZN": 1018724,
    "GOOGL": 1652044,
}

CATEGORIES = {
    "PANDEMIC": r"\bpandemic\b|\bcovid[-\s]?19\b",
    "SUPPLY_CHAIN":
        r"\bsupply[\s-]?chain\b|\bcomponent shortages?\b",
    "DEMAND":
        r"\bcustomer spending\b|\brecession\b|\bweak demand\b",
    "LITIGATION":
        r"\blitigation\b|\bmaterial litigation\b|\blegal proceedings\b",
    "REGULATION":
        r"\bexport controls?\b|\bexport restrictions?\b|"
        r"\bregulatory requirements?\b|\btrade sanctions?\b",
    "CYBERSECURITY":
        r"\bcybersecurity\b|\bcyberattacks?\b|\bdata breaches?\b",
}
COLS = ["sec_risk_category_count", "sec_mention_count"] + [
    "sec_" + x for x in CATEGORIES
]

FIELDS = [
    "ticker", "form", "accession", "filing_date",
    "acceptance_utc", "available_date", "primary_document",
    "document_url", "evidence_id", "source_sha256",
    "metadata_verified", "text_verified",
    "text_status", "sec_risk_category_count",
    "sec_mention_count",
] + ["sec_" + x for x in CATEGORIES]

def sha(b):
    return hashlib.sha256(b).hexdigest()

def dateparse(s):
    try:
        return date.fromisoformat(str(s)[:10])
    except Exception:
        return None

class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)

def clean_text(raw):
    source = raw.decode("utf-8", errors="replace")
    parser = TextParser()
    try:
        parser.feed(source)
        txt = " ".join(parser.parts)
    except Exception:
        txt = re.sub(r"<[^>]+>", " ", source)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt)

def features(text):
    counts = {
        k: len(re.findall(v, text, re.I))
        for k, v in CATEGORIES.items()
    }
    return {
        "sec_risk_category_count": sum(
            v > 0 for v in counts.values()
        ),
        "sec_mention_count": sum(counts.values()),
        **{"sec_" + k: int(v > 0)
           for k, v in counts.items()},
    }

def writecsv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore"
        )
        w.writeheader()
        w.writerows(rows)

def request_cached(url, path, ua, pause):
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()

    req = urllib.request.Request(
        url, headers={
            "User-Agent": ua,
            "Accept-Encoding": "identity",
            "Accept": "application/json,text/html,*/*",
        }
    )

    last_error = None
    for attempt in range(3):
        try:
            time.sleep(pause)
            with urllib.request.urlopen(req, timeout=45) as r:
                content = r.read()
            if not content:
                raise ValueError("Empty HTTP response")

            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(content)
            temp.replace(path)
            return content
        except Exception as e:
            last_error = e
            if isinstance(e, urllib.error.HTTPError):
                if e.code in (401, 403, 404):
                    break
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"Request failed: {url}: {last_error}")

def records(block):
    """
    SEC submission records are stored as parallel arrays.
    """
    forms = block.get("form", [])
    result = []
    for i in range(len(forms)):
        r = {}
        for key, values in block.items():
            if isinstance(values, list):
                r[key] = values[i] if i < len(values) else ""
        result.append(r)
    return result

def parse_accepted(value):
    """
    SEC acceptanceDateTime is generally Eastern local time
    in ISO-like notation; convert to UTC where possible.
    With date-only availability, a full-day lag is used.
    """
    s = str(value or "").strip()
    if not s:
        return ""

    try:
        from zoneinfo import ZoneInfo
        d = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=ZoneInfo("America/New_York"))
        return d.astimezone(timezone.utc).isoformat()
    except Exception:
        return ""

def load_company(ticker, args):
    cik = CIKS[ticker]
    base = "https://data.sec.gov/submissions/"
    root = Path(args.cache) / "submissions"

    name = f"CIK{cik:010d}.json"
    data = json.loads(
        request_cached(
            base + name, root / name, args.user_agent,
            args.pause
        )
    )

    all_rows = records(data.get("filings", {}).get("recent", {}))

    for item in data.get("filings", {}).get("files", []):
        first = dateparse(item.get("filingFrom"))
        last = dateparse(item.get("filingTo"))
        if last and last.year < args.start_year:
            continue
        if first and first.year > args.end_year:
            continue

        extra_name = item["name"]
        extra = json.loads(request_cached(
            base + extra_name,
            root / extra_name,
            args.user_agent,
            args.pause
        ))
        all_rows.extend(records(extra))

    selected = []
    used = set()

    for r in all_rows:
        form = str(r.get("form", "")).upper()
        if form not in args.forms:
            continue

        filed = dateparse(r.get("filingDate"))
        if not filed:
            continue
        if not args.start_year <= filed.year <= args.end_year:
            continue

        accession = r.get("accessionNumber", "")
        doc = r.get("primaryDocument", "")
        accepted = parse_accepted(r.get("acceptanceDateTime"))
        if not accession or not doc or not accepted:
            continue

        if accession in used:
            continue
        used.add(accession)

        url = (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{cik}/{accession.replace('-', '')}/{doc}"
        )

        # Conservative day-level rule:
        # Use only trading dates strictly AFTER filing date.
        selected.append({
            "ticker": ticker,
            "form": form,
            "accession": accession,
            "filing_date": filed.isoformat(),
            "acceptance_utc": accepted,
            "available_date": filed.isoformat(),
            "primary_document": doc,
            "document_url": url,
            "evidence_id": f"EDGAR:{ticker}:{accession}",
            "source_sha256": "",
            "metadata_verified": 1,
            "text_verified": 0,
            "text_status": "NOT_DOWNLOADED",
            **{k: "" for k in COLS},
        })

    return sorted(selected, key=lambda x: (
        x["available_date"], x["accession"]
    ))

def enrich_text(rows, args):
    for n, row in enumerate(rows, 1):
        accession = row["accession"]
        ticker = row["ticker"]
        cache_name = (
            f"{ticker}_{accession}_"
            + re.sub(
                r"[^A-Za-z0-9_.-]", "_",
                row["primary_document"]
            )
        )
        path = Path(args.cache) / "documents" / cache_name

        try:
            content = request_cached(
                row["document_url"], path,
                args.user_agent, args.pause
            )
            if len(content) > args.max_bytes:
                row["text_status"] = "TOO_LARGE"
                continue

            txt = clean_text(content)
            if len(txt) < 1000:
                row["text_status"] = "TOO_SHORT"
                continue

            row.update(features(txt))
            row["source_sha256"] = sha(content)
            row["text_verified"] = 1
            row["text_status"] = "OK"

        except Exception as e:
            row["text_status"] = (
                "DOWNLOAD_ERROR:" + str(e)[:100]
            )

        if n % 20 == 0:
            print(
                "Processed SEC documents:",
                n, "/", len(rows),
                flush=True
            )

def build_daily(rows, market_path, max_age):
    with open(market_path, encoding="utf-8-sig", newline="") as f:
        market = list(csv.DictReader(f))

    valid_by_ticker = defaultdict(list)
    for row in rows:
        if row["text_verified"] == 1:
            valid_by_ticker[row["ticker"]].append(row)

    for ticker in valid_by_ticker:
        valid_by_ticker[ticker].sort(
            key=lambda x: (
                x["available_date"], x["accession"]
            )
        )

    output = []
    counter = Counter()

    for mr in market:
        ticker = str(
            mr.get("ticker") or mr.get("tic") or ""
        ).upper()
        d = dateparse(mr.get("date"))
        if ticker not in CIKS or not d:
            continue
        if not (2018 <= d.year <= 2023):
            continue

        candidates = [
            r for r in valid_by_ticker[ticker]
            if dateparse(r["available_date"]) < d
            and (d - dateparse(r["available_date"])).days <= max_age
        ]

        selected = candidates[-1] if candidates else None

        out = {
            "date": d.isoformat(),
            "ticker": ticker,
            "kg_mask_sec": int(selected is not None),
            "sec_evidence_id": (
                selected["evidence_id"] if selected else ""
            ),
            "sec_source_available_at": (
                selected["available_date"] if selected else ""
            ),
            "sec_source_sha256": (
                selected["source_sha256"] if selected else ""
            ),
            "sec_source_url": (
                selected["document_url"] if selected else ""
            ),
        }

        for col in COLS:
            out[col] = selected[col] if selected else ""

        if selected:
            counter[ticker] += 1
        output.append(out)

    output.sort(key=lambda x: (x["date"], x["ticker"]))
    return output, counter

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--user-agent",
        default=os.environ.get("SEC_USER_AGENT", "")
    )
    p.add_argument(
        "--market",
        default="kg/data/market_5stocks_2017_2026.csv"
    )
    p.add_argument(
        "--start-year", type=int, default=2017
    )
    p.add_argument(
        "--end-year", type=int, default=2023
    )
    p.add_argument(
        "--forms", nargs="+",
        default=["10-K", "10-Q"]
    )
    p.add_argument(
        "--pause", type=float, default=0.3
    )
    p.add_argument(
        "--cache", default="outputs/v20/sec_cache"
    )
    p.add_argument(
        "--out-dir", default="outputs/v20"
    )
    p.add_argument(
        "--max-age", type=int, default=455
    )
    p.add_argument(
        "--max-bytes", type=int,
        default=30_000_000
    )
    p.add_argument(
        "--download-text", action="store_true"
    )
    p.add_argument(
        "--self-test", action="store_true"
    )
    args = p.parse_args()

    if args.self_test:
        sample = (
            "COVID-19 impacted the supply chain, "
            "customer spending and export controls."
        )
        x = features(sample)
        assert x["sec_PANDEMIC"] == 1
        assert x["sec_SUPPLY_CHAIN"] == 1
        assert x["sec_DEMAND"] == 1
        assert x["sec_REGULATION"] == 1
        assert x["sec_CYBERSECURITY"] == 0
        assert dateparse("2021-06-01") < dateparse("2021-06-02")
        print("SELF-TEST PASSED")
        return

    if not args.user_agent or "@" not in args.user_agent:
        p.error(
            "Set a descriptive SEC_USER_AGENT including "
            "your contact email."
        )

    if args.pause < 0.15:
        p.error("--pause must be >= 0.15 seconds")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = {}

    for ticker in CIKS:
        print("SEC metadata:", ticker, flush=True)
        try:
            got = load_company(ticker, args)
            rows.extend(got)
            print(" ", ticker, len(got), "filings")
        except Exception as e:
            failures[ticker] = str(e)
            print(" FAILED:", e)

    if not rows:
        raise RuntimeError(
            "No SEC filings retrieved. Check network, "
            "User-Agent, and API accessibility."
        )

    if args.download_text:
        enrich_text(rows, args)

    writecsv(out / "sec_filings_manifest.csv", rows, FIELDS)

    daily, covered = build_daily(
        rows, args.market, args.max_age
    )
    daily_fields = [
        "date", "ticker", "kg_mask_sec",
        "sec_evidence_id", "sec_source_available_at",
        "sec_source_sha256", "sec_source_url",
    ] + COLS

    writecsv(
        out / "sec_daily_features.csv",
        daily, daily_fields
    )

    report = {
        "filing_metadata_rows": len(rows),
        "downloaded_and_hashed_text_rows": sum(
            x["text_verified"] for x in rows
        ),
        "market_daily_rows": len(daily),
        "sec_observed_daily": sum(
            int(x["kg_mask_sec"]) for x in daily
        ),
        "coverage_by_ticker": dict(covered),
        "metadata_failures": failures,
        "source": "SEC EDGAR submissions API",
        "lookback_days": args.max_age,
        "strictly_prior_to_market_date": True,
        "sec_text_semantically_fact_checked": False,
        "findkg_historical_publication_verified": False,
        "formal_full_fusion_approved": False,
        "interpretation": (
            "Keyword categories indicate mentions in "
            "documents, not realized risk or price impact."
        ),
    }

    (out / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== V20 SEC RESULTS ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("Saved:", out)

if __name__ == "__main__":
    main()
