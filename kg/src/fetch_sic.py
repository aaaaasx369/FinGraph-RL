from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests


SEC_HEADERS = {
    # 換成你自己的聯絡 email 也可以
    "User-Agent": "FinRL-Financial-KG research@example.com",
    "Accept-Encoding": "gzip, deflate",
}


PORTFOLIO_CIK = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "NVDA": "0001045810",
    "AMZN": "0001018724",
    "GOOGL": "0001652044",
}


def clean_cik(x) -> str:
    s = str(x or "").strip()

    if not s or s.lower() == "nan":
        return ""

    try:
        s = str(int(float(s)))
    except Exception:
        s = s.lstrip("0")

    return s.zfill(10)


def fetch_submission(cik: str) -> dict:
    cik10 = clean_cik(cik)

    url = (
        "https://data.sec.gov/submissions/"
        f"CIK{cik10}.json"
    )

    r = requests.get(
        url,
        headers=SEC_HEADERS,
        timeout=30,
    )

    r.raise_for_status()

    return r.json()


def sic_from_submission(data: dict):
    sic = str(
        data.get("sic", "")
    ).strip()

    sic_description = str(
        data.get("sicDescription", "")
    ).strip()

    return sic, sic_description


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--portfolio",
        default="kg/data/companies.csv",
    )

    parser.add_argument(
        "--external",
        default="kg/data/company_candidates_sec_full.csv",
    )

    parser.add_argument(
        "--output",
        default="kg/data/company_master_with_sic.csv",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.12,
    )

    args = parser.parse_args()

    # ========================================================
    # Portfolio companies
    # ========================================================

    portfolio = pd.read_csv(
        args.portfolio
    )

    portfolio["ticker"] = (
        portfolio["ticker"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    portfolio_rows = []

    for _, row in portfolio.iterrows():

        ticker = row["ticker"]

        cik = clean_cik(
            row.get(
                "cik",
                PORTFOLIO_CIK.get(
                    ticker,
                    "",
                ),
            )
        )

        if not cik:
            cik = PORTFOLIO_CIK.get(
                ticker,
                "",
            )

        portfolio_rows.append(
            {
                "ticker": ticker,
                "tickers": ticker,
                "company_name": row["name"],
                "cik": cik,
                "portfolio_member": 1,
            }
        )

    # ========================================================
    # External companies
    # ========================================================

    external = pd.read_csv(
        args.external
    )

    external_rows = []

    for _, row in external.iterrows():

        external_rows.append(
            {
                "ticker": str(
                    row.get(
                        "tickers",
                        "",
                    )
                ).split("|")[0],

                "tickers": str(
                    row.get(
                        "tickers",
                        "",
                    )
                ),

                "company_name": str(
                    row.get(
                        "company_name",
                        "",
                    )
                ),

                "cik": clean_cik(
                    row.get(
                        "cik",
                        "",
                    )
                ),

                "portfolio_member": 0,
            }
        )

    df = pd.DataFrame(
        portfolio_rows
        +
        external_rows
    )

    # Deduplicate same legal entity by CIK
    df = (
        df[
            df["cik"].astype(str).str.len() > 0
        ]
        .drop_duplicates(
            subset=["cik"]
        )
        .reset_index(drop=True)
    )

    print(
        f"Unique companies: {len(df)}"
    )

    sic_values = []
    sic_descriptions = []
    statuses = []

    cache_path = Path(
        "kg/data/reference/sec_sic_cache.json"
    )

    cache_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if cache_path.exists():
        with open(
            cache_path,
            "r",
            encoding="utf-8",
        ) as f:
            cache = json.load(f)
    else:
        cache = {}

    for i, row in df.iterrows():

        cik = clean_cik(
            row["cik"]
        )

        name = row[
            "company_name"
        ]

        if cik in cache:

            info = cache[cik]

            sic = info.get(
                "sic",
                "",
            )

            desc = info.get(
                "sic_description",
                "",
            )

            status = "CACHE"

        else:

            try:

                data = fetch_submission(
                    cik
                )

                sic, desc = (
                    sic_from_submission(
                        data
                    )
                )

                status = (
                    "OK"
                    if sic
                    else "NO_SIC"
                )

                cache[cik] = {
                    "sic": sic,
                    "sic_description": desc,
                    "name": name,
                }

                with open(
                    cache_path,
                    "w",
                    encoding="utf-8",
                ) as f:

                    json.dump(
                        cache,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                time.sleep(
                    args.sleep
                )

            except Exception as e:

                sic = ""
                desc = ""

                status = (
                    "ERROR:"
                    + str(e)[:100]
                )

        sic_values.append(
            sic
        )

        sic_descriptions.append(
            desc
        )

        statuses.append(
            status
        )

        print(
            f"[{i+1}/{len(df)}] "
            f"{row['ticker']:8s} "
            f"CIK={cik} "
            f"SIC={sic or '-':4s} "
            f"{name}"
        )

    df["sic"] = sic_values

    df[
        "sic_description"
    ] = sic_descriptions

    df[
        "sic_fetch_status"
    ] = statuses

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print("SIC SUMMARY")
    print("=" * 70)

    print(
        "Companies:",
        len(df),
    )

    print(
        "With SIC:",
        df["sic"]
        .astype(str)
        .str.strip()
        .ne("")
        .sum(),
    )

    print()
    print(
        df[
            [
                "ticker",
                "company_name",
                "cik",
                "sic",
                "sic_description",
                "portfolio_member",
            ]
        ]
        .sort_values(
            ["sic", "ticker"]
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved -> {output}"
    )


if __name__ == "__main__":
    main()
