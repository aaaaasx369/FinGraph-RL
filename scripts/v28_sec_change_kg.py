#!/usr/bin/env python3
"""
V28 SEC Evidence-grounded Filing Change Detection.

Fixed universe:
 AAPL, MSFT, NVDA, AMZN, GOOGL

Commands:
 build  - source verification, SEC extraction,
          same-form changes, as-of daily panel
 bridge - V21-compatible FinRL datasets
 audit  - source, temporal, feature, split audits
 all    - build + bridge + audit

NO:
 paid API, GPU, new assets, invented events,
 automatic claim verification, strict PIT approval.

Dependencies: pandas numpy pyarrow.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v28"
V20 = ROOT / "outputs/v20/fusion"
MANIFEST = ROOT / "outputs/v20/sec_filings_manifest.csv"
CACHE = ROOT / "outputs/v20/sec_cache/documents"
MARKET = ROOT / "kg/data/market_5stocks_2017_2026.csv"

STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
TIERS = ["conservative", "inclusive"]
FORMS = {"10-K", "10-Q"}

TOPIC_REGEX = {
    "supply_chain": (
        r"\bsupply chain\b|"
        r"\bsuppliers?\b|"
        r"\bprocurement\b|"
        r"\bcomponent shortages?\b"
    ),
    "manufacturing": (
        r"\bmanufactur(?:e|es|ed|ing|er|ers)\b|"
        r"\bfoundr(?:y|ies)\b|"
        r"\bfabricat(?:e|ion|ing)\b|"
        r"\bsemiconductor wafers?\b"
    ),
    "cloud": (
        r"\bcloud services?\b|"
        r"\bcloud computing\b|"
        r"\bdata cent(?:er|re)s?\b|"
        r"\bazure\b"
    ),
    "competition": (
        r"\bcompetition\b|"
        r"\bcompetitors?\b|"
        r"\bcompetitive\b|"
        r"\bmarket share\b"
    ),
    "regulation": (
        r"\bregulat(?:ion|ions|ory)\b|"
        r"\bantitrust\b|"
        r"\bexport controls?\b|"
        r"\bsanctions?\b"
    ),
    "litigation": (
        r"\blitigation\b|"
        r"\blawsuits?\b|"
        r"\blegal proceedings?\b"
    ),
    "finance": (
        r"\bcash flows?\b|"
        r"\boperating income\b|"
        r"\baccounts receivable\b|"
        r"\brevenue\b"
    ),
    "research": (
        r"\bresearch and development\b|"
        r"\br&d\b|"
        r"\bproduct development\b"
    ),
}

TOPICS = list(TOPIC_REGEX)
PATTERNS = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in TOPIC_REGEX.items()
}

# Predefined analysis parameters. Not tuned on 2023.
WINDOW_RADIUS = 240
MAX_EXCERPTS_PER_TOPIC = 30
SHINGLE_N = 5
NOVELTY_THRESHOLD = 0.45
HALF_LIFE_DAYS = 180.0


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            obj, indent=2,
            ensure_ascii=False,
            default=str
        ),
        encoding="utf-8"
    )


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(*items):
    return sha_bytes(
        "\x1f".join(
            str(x) for x in items
        ).encode("utf-8")
    )[:24]


def parse_dates(s):
    return pd.to_datetime(
        s, errors="raise"
    ).dt.normalize()


def valid_bool(series):
    return (
        series.astype(str)
        .str.lower()
        .isin(["true", "1", "1.0"])
    )


def tokenize(text):
    return re.findall(
        r"[a-z][a-z0-9]+",
        str(text).lower()
    )


def shingle_set(text):
    words = tokenize(text)
    if len(words) < SHINGLE_N:
        return set(words)

    return {
        digest(*words[i:i + SHINGLE_N])
        for i in range(
            len(words) - SHINGLE_N + 1
        )
    }


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    return (
        len(a & b) / len(a | b)
    )


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(
            convert_charrefs=True
        )
        self.parts = []
        self.ignore = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignore += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.ignore = max(
                0, self.ignore - 1
            )

    def handle_data(self, data):
        if not self.ignore:
            self.parts.append(data)


def to_text(raw):
    source = raw.decode(
        "utf-8", errors="replace"
    )

    parser = TextExtractor()
    parser.feed(source)

    return " ".join(
        " ".join(parser.parts).split()
    )


def sec_cache_path(record):
    filename = (
        f"{record['ticker']}_"
        f"{record['accession']}_"
        + re.sub(
            r"[^A-Za-z0-9_.-]",
            "_",
            str(record["primary_document"])
        )
    )
    return CACHE / filename


def read_sec_manifest():
    frame = pd.read_csv(
        MANIFEST,
        dtype={"accession": str}
    )

    required = [
        "ticker", "form", "accession",
        "filing_date", "primary_document",
        "source_sha256", "text_verified"
    ]
    missing = [
        c for c in required
        if c not in frame
    ]
    if missing:
        raise ValueError(
            f"SEC manifest missing: {missing}"
        )

    frame["ticker"] = (
        frame["ticker"]
        .astype(str).str.upper()
    )
    frame["form"] = (
        frame["form"]
        .astype(str).str.upper()
    )
    frame["filing_date"] = parse_dates(
        frame["filing_date"]
    )

    frame = frame[
        frame["ticker"].isin(STOCKS)
        & frame["form"].isin(FORMS)
    ].copy()

    if frame.duplicated(
        ["ticker", "accession"]
    ).any():
        raise ValueError(
            "Duplicate ticker/accession"
        )

    if len(frame) != 140:
        raise ValueError(
            f"Expected 140 filings, got {len(frame)}"
        )

    return frame.sort_values(
        ["ticker", "filing_date", "accession"]
    ).reset_index(drop=True)


def read_market():
    df = pd.read_csv(MARKET)

    if (
        "ticker" not in df
        and "tic" in df
    ):
        df = df.rename(
            columns={"tic": "ticker"}
        )

    df["ticker"] = (
        df["ticker"]
        .astype(str).str.upper()
    )
    df["date"] = parse_dates(
        df["date"]
    )

    df = df[
        df["ticker"].isin(STOCKS)
        & df["date"].between(
            "2018-01-01",
            "2023-12-31"
        )
    ][["date", "ticker"]].drop_duplicates()

    if len(df) != 7545:
        raise ValueError(
            f"Expected 7545 market rows, got {len(df)}"
        )

    counts = df.groupby(
        "ticker"
    ).size().to_dict()

    if counts != {
        ticker: 1509
        for ticker in STOCKS
    }:
        raise ValueError(
            f"Unexpected market coverage: {counts}"
        )

    return df.sort_values(
        ["ticker", "date"]
    ).reset_index(drop=True)


def extract_one_filing(record):
    path = sec_cache_path(record)

    if not path.is_file():
        raise FileNotFoundError(path)

    raw = path.read_bytes()
    actual = sha_bytes(raw)
    expected = str(
        record["source_sha256"]
    )

    if actual != expected:
        raise ValueError(
            f"SEC SHA256 mismatch: {path}"
        )

    if not bool(
        valid_bool(
            pd.Series([record["text_verified"]])
        ).iloc[0]
    ):
        raise ValueError(
            f"SEC text not verified: {path}"
        )

    content = to_text(raw)
    nwords = len(tokenize(content))

    if nwords < 100:
        raise ValueError(
            f"Unexpectedly short SEC text: {path}"
        )

    filing = {
        "ticker": record["ticker"],
        "accession": str(
            record["accession"]
        ),
        "form": record["form"],
        "filing_date": record["filing_date"],
        "available_from": (
            record["filing_date"]
            + pd.Timedelta(days=1)
        ),
        "document_sha256": actual,
        "document_word_count": nwords,
        "strict_source_identity_verified": False,
        "cache_sha256_verified": True,
    }

    if "acceptance_utc" in record.index:
        filing["acceptance_utc"] = str(
            record["acceptance_utc"]
        )

    evidence = []
    signatures = {}

    for topic, pattern in PATTERNS.items():
        matches = list(
            pattern.finditer(content)
        )

        count = len(matches)
        rate = (
            count * 10000.0 / nwords
        )

        filing[
            "v28_" + topic + "_count"
        ] = count

        filing[
            "v28_" + topic + "_per10k"
        ] = rate

        # All matches contribute to rate.
        # At most 30 examples are retained for
        # novelty analysis and source inspection.
        if count <= MAX_EXCERPTS_PER_TOPIC:
            selected = matches
        else:
            indices = np.linspace(
                0,
                count - 1,
                MAX_EXCERPTS_PER_TOPIC,
                dtype=int
            )
            selected = [
                matches[int(i)]
                for i in sorted(set(indices))
            ]

        snippets = []
        hashes = set()

        for i, match in enumerate(selected):
            left = max(
                0,
                match.start() - WINDOW_RADIUS
            )
            right = min(
                len(content),
                match.end() + WINDOW_RADIUS
            )

            excerpt = content[left:right]
            signature = shingle_set(excerpt)

            # Avoid duplicated windows within
            # a single filing/topic.
            exact = digest(excerpt)
            if exact in hashes:
                continue
            hashes.add(exact)

            snippets.append(
                (excerpt, signature, left)
            )

            evidence.append({
                "evidence_id": digest(
                    record["ticker"],
                    record["accession"],
                    topic, match.start()
                ),
                "ticker": record["ticker"],
                "accession": str(
                    record["accession"]
                ),
                "form": record["form"],
                "filing_date": record["filing_date"],
                "available_from": filing[
                    "available_from"
                ],
                "topic": topic,
                "matched_term": match.group(0),
                "character_offset": match.start(),
                "excerpt": excerpt,
                "document_sha256": actual,
                "cache_sha256_verified": True,
                "relation_direction_verified": False,
                "economic_event_verified": False,
            })

        signatures[topic] = snippets

    return filing, evidence, signatures


def compute_change_events(filings, signature_cache):
    """
    Compare only the previous filing of the SAME FORM.

    Novelty measures wording differences in
    sampled snippets, not economic events.
    """
    output = []
    filings = filings.sort_values(
        ["ticker", "form", "filing_date", "accession"]
    )

    for ticker in STOCKS:
        for form in ["10-K", "10-Q"]:
            subset = filings[
                filings["ticker"].eq(ticker)
                & filings["form"].eq(form)
            ]

            previous = None

            for _, row in subset.iterrows():
                key = (
                    ticker,
                    str(row["accession"])
                )
                current_snippets = (
                    signature_cache[key]
                )

                for topic in TOPICS:
                    rate = float(
                        row[
                            "v28_" + topic + "_per10k"
                        ]
                    )

                    old_rate = (
                        float(
                            previous[
                                "v28_" + topic + "_per10k"
                            ]
                        )
                        if previous is not None
                        else np.nan
                    )

                    old_snippets = (
                        signature_cache[
                            (
                                ticker,
                                str(previous["accession"])
                            )
                        ][topic]
                        if previous is not None
                        else []
                    )

                    samples = (
                        current_snippets[topic]
                    )

                    novel_count = 0
                    novel_examples = []

                    if previous is not None:
                        old_sets = [
                            signature
                            for _, signature, _ in old_snippets
                        ]

                        for excerpt, signature, offset in samples:
                            best_similarity = max(
                                [
                                    jaccard(
                                        signature,
                                        old_signature
                                    )
                                    for old_signature in old_sets
                                ],
                                default=0.0
                            )

                            if (
                                1.0 - best_similarity
                                >= NOVELTY_THRESHOLD
                            ):
                                novel_count += 1
                                if len(novel_examples) < 3:
                                    novel_examples.append({
                                        "offset": offset,
                                        "excerpt": excerpt[:450],
                                        "novelty_score": (
                                            1.0
                                            - best_similarity
                                        ),
                                    })

                    if previous is None:
                        event_type = "FIRST_SAME_FORM"
                    elif old_rate == 0 and rate > 0:
                        event_type = "NEW_MENTION"
                    elif old_rate > 0 and rate == 0:
                        event_type = "NOT_REMENTIONED"
                    elif old_rate > 0 and rate > 0:
                        event_type = "REPEATED_MENTION"
                    else:
                        event_type = "STILL_NOT_MENTIONED"

                    output.append({
                        "ticker": ticker,
                        "form": form,
                        "accession": str(row["accession"]),
                        "filing_date": row["filing_date"],
                        "available_from": row["available_from"],
                        "topic": topic,
                        "event_type": event_type,
                        "rate_per10k": rate,
                        "previous_same_form_rate": old_rate,
                        "rate_delta": (
                            rate - old_rate
                            if previous is not None
                            else np.nan
                        ),
                        "has_comparable_prior": (
                            previous is not None
                        ),
                        "novel_excerpt_count": novel_count,
                        "sampled_excerpt_count": len(samples),
                        "novel_examples_json": json.dumps(
                            novel_examples,
                            ensure_ascii=False
                        ),
                        "economic_change_verified": False,
                        "relationship_terminated": False,
                    })

                previous = row

    result = pd.DataFrame(output)

    if result.duplicated(
        ["ticker", "accession", "topic"]
    ).any():
        raise ValueError(
            "Duplicate filing/topic comparison"
        )

    return result


def daily_panel(filings, changes, market):
    """
    As-of state using the most recent same-form
    filing for each form (10-K and 10-Q).

    10-K/10-Q are not averaged as if equivalent:
    raw rates are retained separately for audit.
    """
    lookup = {
        (r["ticker"], str(r["accession"])): r
        for _, r in changes.iterrows()
    }

    # Faster per-filing summaries.
    changes_by_filing = {}

    for key, subset in changes.groupby(
        ["ticker", "accession"]
    ):
        changes_by_filing[
            (key[0], str(key[1]))
        ] = {
            str(r["topic"]): r
            for _, r in subset.iterrows()
        }

    rows = []
    half_life = math.log(2.0) / HALF_LIFE_DAYS

    for ticker in STOCKS:
        docs = (
            filings[
                filings["ticker"].eq(ticker)
            ]
            .sort_values(
                ["available_from", "accession"]
            )
            .to_dict("records")
        )

        days = market.loc[
            market["ticker"].eq(ticker),
            "date"
        ].sort_values().tolist()

        pointer = 0
        latest = {}
        last_filing = None
        last_mention = {}
        pending_novelty = 0
        known = 0

        for day in days:
            new_filings = 0
            new_novel_excerpts = 0

            while (
                pointer < len(docs)
                and docs[pointer][
                    "available_from"
                ] <= day
            ):
                doc = docs[pointer]
                if not doc["filing_date"] < day:
                    raise RuntimeError(
                        "Same-day SEC leakage"
                    )

                latest[doc["form"]] = doc
                last_filing = doc
                known += 1
                new_filings += 1

                c = changes_by_filing[
                    (
                        ticker,
                        str(doc["accession"])
                    )
                ]

                for topic in TOPICS:
                    if (
                        doc[
                            "v28_" + topic + "_count"
                        ] > 0
                    ):
                        last_mention[topic] = (
                            doc["available_from"]
                        )

                    new_novel_excerpts += int(
                        c[topic][
                            "novel_excerpt_count"
                        ]
                    )

                pointer += 1

            item = {
                "date": day,
                "ticker": ticker,
                "v28_has_filing": int(known > 0),
                "v28_known_filings": known,
                "v28_new_filings": new_filings,
                "v28_novel_excerpts_today":
                    new_novel_excerpts,
                "v28_latest_accession": (
                    str(last_filing["accession"])
                    if last_filing else ""
                ),
                "v28_last_filing_date": (
                    last_filing["filing_date"]
                    if last_filing else pd.NaT
                ),
                "v28_filing_age_days": (
                    (day - last_filing[
                        "available_from"
                    ]).days
                    if last_filing else -1
                ),
            }

            for topic in TOPICS:
                annual = latest.get("10-K")
                quarterly = latest.get("10-Q")

                k_rate = (
                    float(
                        annual[
                            "v28_" + topic + "_per10k"
                        ]
                    )
                    if annual else 0.0
                )
                q_rate = (
                    float(
                        quarterly[
                            "v28_" + topic + "_per10k"
                        ]
                    )
                    if quarterly else 0.0
                )

                # Preserve separate form rates.
                item[
                    "v28_k_" + topic
                ] = k_rate

                item[
                    "v28_q_" + topic
                ] = q_rate

                # A simple equal-weight baseline.
                # The mask distinguishes missing form
                # from true zero mention.
                available_rates = [
                    x for x, doc in [
                        (k_rate, annual),
                        (q_rate, quarterly)
                    ]
                    if doc is not None
                ]

                combined = (
                    float(np.mean(available_rates))
                    if available_rates else 0.0
                )

                age = (
                    (day - last_mention[topic]).days
                    if topic in last_mention
                    else -1
                )

                decay = (
                    math.exp(-half_life * age)
                    if age >= 0 else 0.0
                )

                item[
                    "v28_" + topic + "_rate"
                ] = combined

                item[
                    "v28_" + topic + "_age"
                ] = age

                item[
                    "v28_" + topic + "_decay"
                ] = decay

                item[
                    "v28_" + topic + "_weighted"
                ] = combined * decay

            rows.append(item)

    daily = pd.DataFrame(rows).sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if len(daily) != 7545:
        raise ValueError(
            f"Wrong daily rows: {len(daily)}"
        )

    if daily.duplicated(
        ["date", "ticker"]
    ).any():
        raise ValueError(
            "Duplicate daily stock keys"
        )

    known_dates = pd.to_datetime(
        daily["v28_last_filing_date"]
    )
    trade_dates = pd.to_datetime(
        daily["date"]
    )

    valid = known_dates.notna()

    if not (
        known_dates[valid]
        < trade_dates[valid]
    ).all():
        raise ValueError(
            "As-of filing date leakage"
        )

    return daily


def build():
    OUT.mkdir(
        parents=True, exist_ok=True
    )

    manifest = read_sec_manifest()
    market = read_market()

    filing_rows = []
    evidence_rows = []
    signature_cache = {}
    audit_rows = []

    for _, record in manifest.iterrows():
        filing, evidence, signatures = (
            extract_one_filing(record)
        )

        filing_rows.append(filing)
        evidence_rows.extend(evidence)

        signature_cache[
            (
                filing["ticker"],
                str(filing["accession"])
            )
        ] = signatures

        audit_rows.append({
            "ticker": filing["ticker"],
            "accession": filing["accession"],
            "form": filing["form"],
            "file_hash_verified": True,
            "document_sha256": filing[
                "document_sha256"
            ],
            "word_count": filing[
                "document_word_count"
            ],
        })

    filings = pd.DataFrame(filing_rows)
    evidence = pd.DataFrame(evidence_rows)
    source_audit = pd.DataFrame(audit_rows)

    if len(filings) != 140:
        raise RuntimeError(
            "Missing SEC documents"
        )

    if evidence.empty:
        raise RuntimeError(
            "No SEC evidence extracted"
        )

    if evidence[
        "evidence_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate evidence ID"
        )

    changes = compute_change_events(
        filings,
        signature_cache
    )

    daily = daily_panel(
        filings, changes, market
    )

    filings.to_csv(
        OUT / "filing_features.csv",
        index=False
    )

    evidence.to_parquet(
        OUT / "filing_evidence.parquet",
        index=False
    )

    changes.to_csv(
        OUT / "filing_changes.csv",
        index=False
    )

    daily.to_csv(
        OUT / "daily_features.csv",
        index=False
    )

    source_audit.to_csv(
        OUT / "source_audit.csv",
        index=False
    )

    summary = {
        "research_stocks": STOCKS,
        "filings": int(
            len(filings)
        ),
        "cache_hash_verified": int(
            source_audit[
                "file_hash_verified"
            ].sum()
        ),
        "evidence_excerpts": int(
            len(evidence)
        ),
        "filing_topic_comparisons": int(
            len(changes)
        ),
        "event_types": {
            str(k): int(v)
            for k, v in changes[
                "event_type"
            ].value_counts().items()
        },
        "novel_excerpt_candidates": int(
            changes[
                "novel_excerpt_count"
            ].sum()
        ),
        "updated_stock_days": int(
            (daily["v28_new_filings"] > 0).sum()
        ),
        "days_with_novel_excerpt_candidates": int(
            (
                daily[
                    "v28_novel_excerpts_today"
                ] > 0
            ).sum()
        ),
        "temporal_panel_rows": int(
            len(daily)
        ),
        "strict_pit_approved": False,
        "novelty_is_economic_event": False,
        "method": (
            "Per-10k SEC mention rates, "
            "same-form filing comparisons, "
            "local excerpt shingle novelty, "
            "as-of daily carry-forward"
        ),
    }

    write_json(
        OUT / "build_report.json",
        summary
    )

    print("\n=== V28 BUILD COMPLETE ===")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False
    ))


def bridge():
    path = OUT / "daily_features.csv"
    if not path.is_file():
        raise FileNotFoundError(
            "Run build first"
        )

    daily = pd.read_csv(path)
    daily["date"] = parse_dates(
        daily["date"]
    )
    daily["ticker"] = (
        daily["ticker"].astype(str).str.upper()
    )

    schema = json.loads(
        (
            V20 / "feature_sets.json"
        ).read_text(encoding="utf-8")
    )

    static = schema["FULL_STATIC"]
    dynamic = schema["FULL_DYNAMIC"]

    dynamic_columns = [
        col for col in dynamic
        if col not in static
    ]

    if len(dynamic_columns) != 11:
        raise ValueError(
            "V21 compatibility requires "
            "11 dynamic columns"
        )

    # 8 continuous topic indicators, evidence
    # novelty, filing updates, availability.
    graph_features = [
        "v28_" + topic + "_weighted"
        for topic in TOPICS
    ] + [
        "v28_novel_excerpts_today",
        "v28_new_filings",
        "v28_has_filing",
    ]

    out = OUT / "finrl"
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    write_json(
        out / "feature_sets.json",
        schema
    )

    report = {}

    for tier in TIERS:
        source = V20 / (
            f"dataset_{tier}.csv"
        )
        if not source.is_file():
            raise FileNotFoundError(
                source
            )

        frame = pd.read_csv(source)
        frame["date"] = parse_dates(
            frame["date"]
        )
        frame["ticker"] = (
            frame["ticker"]
            .astype(str).str.upper()
        )

        if set(frame["ticker"]) != set(STOCKS):
            raise ValueError(
                "Unexpected portfolio universe"
            )

        if frame.duplicated(
            ["date", "ticker"]
        ).any():
            raise ValueError(
                "Duplicate market keys"
            )

        if len(frame) != 7545:
            raise ValueError(
                "Wrong FinRL panel size"
            )

        frame = frame.drop(
            columns=dynamic_columns,
            errors="raise"
        )

        merged = frame.merge(
            daily[
                ["date", "ticker"]
                + graph_features
            ],
            on=["date", "ticker"],
            how="left",
            validate="one_to_one",
            indicator=True
        )

        if not merged[
            "_merge"
        ].eq("both").all():
            raise ValueError(
                "Incomplete V28 graph merge"
            )

        merged.drop(
            columns=["_merge"],
            inplace=True
        )

        split = merged[
            "split"
        ].astype(str)

        if set(split) != {
            "train", "val", "test"
        }:
            raise ValueError(
                "Unexpected training splits"
            )

        train = split.eq("train")
        scalers = {}

        for raw, model_col in zip(
            graph_features,
            dynamic_columns
        ):
            series = pd.to_numeric(
                merged[raw],
                errors="raise"
            ).astype(float)

            if not np.isfinite(
                series
            ).all():
                raise ValueError(
                    f"Nonfinite feature: {raw}"
                )

            if raw in [
                "v28_novel_excerpts_today",
                "v28_new_filings"
            ]:
                series = np.log1p(
                    series
                )

            mu = float(
                series[train].mean()
            )
            sd = float(
                series[train].std(ddof=0)
            )

            if (
                not np.isfinite(sd)
                or sd <= 1e-12
            ):
                sd = 1.0

            merged[model_col] = (
                series - mu
            ) / sd

            scalers[model_col] = {
                "source": raw,
                "train_mean": mu,
                "train_std": sd,
            }

        for condition, columns in schema.items():
            missing = [
                c for c in columns
                if c not in merged
            ]
            if missing:
                raise ValueError(
                    f"{condition} missing {missing}"
                )

            if not np.isfinite(
                merged[
                    columns
                ].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"Invalid {condition} features"
                )

        merged = merged.sort_values(
            ["date", "ticker"]
        )

        merged.to_csv(
            out / f"dataset_{tier}.csv",
            index=False
        )

        write_json(
            out / f"scaler_{tier}.json",
            scalers
        )

        report[tier] = {
            "rows": len(merged),
            "stocks": STOCKS,
            "static_dimensions": len(static),
            "dynamic_dimensions": len(dynamic),
            "new_feature_sources": graph_features,
            "scaler_fit_on_train_only": True,
            "strict_pit_approved": False,
        }

    write_json(
        out / "report.json",
        report
    )

    print("\n=== V28 FINRL BRIDGE ===")
    print(json.dumps(
        report, indent=2
    ))


def audit():
    paths = [
        OUT / "filing_features.csv",
        OUT / "filing_evidence.parquet",
        OUT / "filing_changes.csv",
        OUT / "daily_features.csv",
        OUT / "source_audit.csv",
    ]

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing V28 output: {path}"
            )

    filings = pd.read_csv(
        paths[0],
        dtype={"accession": str}
    )
    evidence = pd.read_parquet(
        paths[1]
    )
    changes = pd.read_csv(
        paths[2],
        dtype={"accession": str}
    )
    daily = pd.read_csv(
        paths[3]
    )
    sources = pd.read_csv(
        paths[4]
    )

    assert len(filings) == 140
    assert len(sources) == 140
    assert sources["file_hash_verified"].all()

    assert set(
        filings["ticker"]
    ) == set(STOCKS)

    assert set(
        daily["ticker"]
    ) == set(STOCKS)

    assert len(daily) == 7545

    assert not daily.duplicated(
        ["date", "ticker"]
    ).any()

    assert not evidence[
        "evidence_id"
    ].duplicated().any()

    assert not changes.duplicated(
        ["ticker", "accession", "topic"]
    ).any()

    filing_dates = pd.to_datetime(
        filings["filing_date"]
    )
    available_dates = pd.to_datetime(
        filings["available_from"]
    )

    assert (
        available_dates > filing_dates
    ).all()

    known = pd.to_datetime(
        daily["v28_last_filing_date"]
    )
    dates = pd.to_datetime(
        daily["date"]
    )

    mask = known.notna()

    assert (
        known[mask] < dates[mask]
    ).all()

    assert not changes[
        "economic_change_verified"
    ].astype(bool).any()

    assert not changes[
        "relationship_terminated"
    ].astype(bool).any()

    for topic in TOPICS:
        column = (
            "v28_" + topic + "_weighted"
        )

        assert np.isfinite(
            daily[column].to_numpy(
                dtype=float
            )
        ).all()

    print("\n=== V28 AUDIT ===")
    print("Five stock universe: PASS")
    print("SEC cache integrity: PASS")
    print("Filing dates: PASS")
    print("Daily as-of date gate: PASS")
    print("No fabricated relation terminations: PASS")
    print("Finite historical features: PASS")
    print("Strict PIT approved: NO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "build", "bridge",
            "audit", "all"
        ]
    )
    args = parser.parse_args()

    if args.command in [
        "build", "all"
    ]:
        build()

    if args.command in [
        "bridge", "all"
    ]:
        bridge()

    if args.command in [
        "audit", "all"
    ]:
        audit()

    if args.command == "all":
        print("\n=== V28 COMPLETE ===")
        print(
            "Stocks:",
            ", ".join(STOCKS)
        )
        print(
            "Strict PIT: NOT COMPLETE"
        )


if __name__ == "__main__":
    main()
