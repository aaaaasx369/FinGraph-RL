#!/usr/bin/env python3
"""
V29: Historical evidence graph and controlled FinRL ablations.

Inputs:
  outputs/v28/filing_evidence.parquet
  outputs/v28/filing_features.csv
  outputs/v20/sec_filings_manifest.csv
  outputs/v20/sec_cache/documents/
  outputs/v20/fusion/dataset_{tier}.csv
  outputs/v20/fusion/feature_sets.json
  kg/data/market_5stocks_2017_2026.csv

Commands:
  build
  bridge
  audit
  all

Limitations:
  SEC text mention != verified economic relationship.
  Novel wording != verified economic event.
  No strict PIT certification.
  Relation arm disabled until human labels are verified.
"""

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v29"
SEC = ROOT / "outputs/v20/sec_cache/documents"
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
TIERS = ["conservative", "inclusive"]

TOPICS = [
    "supply_chain", "manufacturing", "cloud",
    "competition", "regulation", "litigation",
    "finance", "research",
]

ARMS = [
    "FILING_ONLY",
    "TOPIC_DENSITY",
    "EVIDENCE_CHANGE",
    "RELATION_GRAPH",
]

# Fixed before inspecting test-period performance.
NOVELTY_THRESHOLD = 0.45
BOILERPLATE_SIMILARITY = 0.90
MIN_CONTENT_WORDS = 12
HALF_LIFE = 180.0
SEED = 29


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2,
            ensure_ascii=False, default=str
        ),
        encoding="utf-8",
    )


def sha(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def item_id(*values):
    return sha("\x1f".join(map(str, values)))[:24]


def date_series(values):
    return pd.to_datetime(
        values, errors="raise"
    ).dt.normalize()


def canon(text, remove_numbers=False):
    text = str(text).lower()
    if remove_numbers:
        text = re.sub(
            r"\b[-+]?\d[\d,.%/-]*\b",
            " NUM ", text
        )
    return " ".join(
        re.findall(r"[a-z0-9]+", text)
    )


def word_list(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def shingles(text, length=5):
    words = word_list(text)
    if len(words) < length:
        return set(words)
    return {
        sha(" ".join(words[i:i + length]))[:16]
        for i in range(len(words) - length + 1)
    }


def similarity(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class SECText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip:
            self.parts.append(text)


def parse_sec(data):
    parser = SECText()
    parser.feed(data.decode(
        "utf-8", errors="replace"
    ))
    return " ".join(" ".join(parser.parts).split())


def cache_path(row):
    filename = (
        str(row["ticker"]) + "_"
        + str(row["accession"]) + "_"
        + re.sub(
            r"[^A-Za-z0-9_.-]", "_",
            str(row["primary_document"])
        )
    )
    return SEC / filename


def load_data():
    filings = pd.read_csv(
        ROOT / "outputs/v28/filing_features.csv",
        dtype={"accession": str}
    )
    evidence = pd.read_parquet(
        ROOT / "outputs/v28/filing_evidence.parquet"
    )
    manifest = pd.read_csv(
        ROOT / "outputs/v20/sec_filings_manifest.csv",
        dtype={"accession": str}
    )

    market = pd.read_csv(
        ROOT / "kg/data/market_5stocks_2017_2026.csv"
    )

    if "ticker" not in market and "tic" in market:
        market = market.rename(columns={"tic": "ticker"})

    for df in (filings, evidence, manifest, market):
        df["ticker"] = (
            df["ticker"].astype(str).str.upper()
        )

    filings["filing_date"] = date_series(
        filings["filing_date"]
    )
    filings["available_from"] = date_series(
        filings["available_from"]
    )
    evidence["available_from"] = date_series(
        evidence["available_from"]
    )
    market["date"] = date_series(
        market["date"]
    )

    market = market[
        market["ticker"].isin(STOCKS)
        & market["date"].between(
            "2018-01-01", "2023-12-31"
        )
    ][["date", "ticker"]].drop_duplicates()

    if len(market) != 7545:
        raise ValueError("Market panel is not 7545 rows")

    if set(market["ticker"]) != set(STOCKS):
        raise ValueError("Wrong investment universe")

    if len(filings) != 140:
        raise ValueError("Expected 140 V28 filings")

    if filings.duplicated(
        ["ticker", "accession"]
    ).any():
        raise ValueError("Duplicate filing identity")

    if evidence["evidence_id"].duplicated().any():
        raise ValueError("Duplicate V28 evidence identity")

    manifest = manifest[
        manifest["ticker"].isin(STOCKS)
        & manifest["form"].isin(["10-K", "10-Q"])
    ].copy()

    if manifest.duplicated(
        ["ticker", "accession"]
    ).any():
        raise ValueError("Duplicate manifest identity")

    joined = filings.merge(
        manifest[
            [
                "ticker", "accession",
                "primary_document",
                "source_sha256", "text_verified"
            ]
        ],
        on=["ticker", "accession"],
        validate="one_to_one",
        how="left",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Unmatched SEC filing")
    joined.drop(columns="_merge", inplace=True)

    # Validate every referenced cache document again.
    verified_text = {}
    for _, f in joined.iterrows():
        p = cache_path(f)
        if not p.is_file():
            raise FileNotFoundError(p)

        data = p.read_bytes()
        if sha(data) != str(f["source_sha256"]):
            raise ValueError("SEC hash mismatch: " + str(p))

        if str(f["text_verified"]).lower() not in (
            "1", "1.0", "true"
        ):
            raise ValueError("Unverified SEC text: " + str(p))

        verified_text[
            (f["ticker"], str(f["accession"]))
        ] = parse_sec(data)

    if not set(evidence["ticker"]).issubset(set(STOCKS)):
        raise ValueError("Evidence contains other portfolio ticker")

    # Source spans must exist in the exact archived document.
    source_ok = []
    for r in evidence.itertuples(index=False):
        key = (r.ticker, str(r.accession))
        document = verified_text.get(key, "")
        snippet = str(r.excerpt)
        ok = bool(snippet.strip()) and snippet in document
        source_ok.append(ok)

    evidence = evidence.copy()
    evidence["span_verified"] = source_ok

    if not all(source_ok):
        bad = evidence.loc[
            ~evidence["span_verified"],
            ["ticker", "accession", "evidence_id"]
        ]
        raise ValueError(
            "Some evidence snippets are not found in "
            "their hashed SEC documents:\n"
            + bad.head(10).to_string(index=False)
        )

    return joined, evidence, market


# -------------------------------------------------
# EVIDENCE CHANGE + BOILERPLATE
# -------------------------------------------------

def build_changes(filings, evidence):
    """
    Only compare with documents available earlier
    for the same company, form and topic.

    First document cannot claim novelty.
    Numeric-only edits are separately flagged.
    Boilerplate comparison never uses future docs.
    """
    filing_order = filings.sort_values(
        ["ticker", "form", "available_from", "accession"]
    )

    evidence_groups = {
        (ticker, accession, topic): group.to_dict("records")
        for (ticker, accession, topic), group
        in evidence.groupby(
            ["ticker", "accession", "topic"]
        )
    }

    rows = []
    history = defaultdict(list)

    for f in filing_order.to_dict("records"):
        ticker = f["ticker"]
        form = f["form"]
        accession = str(f["accession"])

        for topic in TOPICS:
            key = (ticker, form, topic)
            previous = history[key]

            entries = evidence_groups.get(
                (ticker, accession, topic), []
            )

            current_features = []

            for e in entries:
                original = str(e["excerpt"])
                normalized = canon(
                    original, remove_numbers=True
                )
                full = canon(original)

                word_count = len(word_list(original))
                source_ok = bool(e["span_verified"])

                if word_count < MIN_CONTENT_WORDS:
                    label = "TOO_SHORT"
                    sim = 0.0
                elif not previous:
                    label = "FIRST_BASELINE"
                    sim = 0.0
                else:
                    signature = shingles(normalized)
                    scores = [
                        similarity(signature, old["signature"])
                        for old in previous
                    ]
                    best_i = int(np.argmax(scores))
                    sim = float(scores[best_i])
                    best = previous[best_i]

                    if full == best["full"]:
                        label = "EXACT_REPEAT"
                    elif normalized == best["normalized"]:
                        label = "NUMERIC_OR_FORMAT_CHANGE"
                    elif sim >= BOILERPLATE_SIMILARITY:
                        label = "BOILERPLATE_SIMILAR"
                    elif (
                        1.0 - sim
                        >= NOVELTY_THRESHOLD
                    ):
                        label = "SUBSTANTIVE_TEXT_CANDIDATE"
                    else:
                        label = "MINOR_WORDING_CHANGE"

                if not source_ok:
                    label = "UNVERIFIED_SPAN"

                rows.append({
                    "evidence_id": e["evidence_id"],
                    "ticker": ticker,
                    "form": form,
                    "accession": accession,
                    "filing_date": f["filing_date"],
                    "available_from": f["available_from"],
                    "topic": topic,
                    "label": label,
                    "best_prior_similarity": sim,
                    "excerpt": original,
                    "matched_term": e["matched_term"],
                    "character_offset": e["character_offset"],
                    "document_sha256": e["document_sha256"],
                    "source_span_verified": source_ok,
                    "economic_event_verified": False,
                    "directed_relation_verified": False,
                })

                current_features.append({
                    "signature": shingles(normalized),
                    "normalized": normalized,
                    "full": full,
                })

            # Add this document only AFTER classifying
            # all its evidence, so same-filing snippets
            # cannot become their own past evidence.
            history[key].extend(current_features)

    result = pd.DataFrame(rows)

    if len(result) != len(evidence):
        raise ValueError("Evidence count mismatch")
    if result["evidence_id"].duplicated().any():
        raise ValueError("Duplicate classified evidence")
    return result


# -------------------------------------------------
# REVIEW SAMPLE (NEVER AUTO-APPROVE)
# -------------------------------------------------

def build_review_sample(changes):
    candidates = changes[
        changes["label"].isin(
            [
                "SUBSTANTIVE_TEXT_CANDIDATE",
                "NUMERIC_OR_FORMAT_CHANGE",
                "MINOR_WORDING_CHANGE",
                "BOILERPLATE_SIMILAR",
                "EXACT_REPEAT",
            ]
        )
    ].copy()

    # Keep year labels for later train/test audit.
    candidates["year"] = pd.to_datetime(
        candidates["filing_date"]
    ).dt.year

    sampled = []

    rng = np.random.default_rng(SEED)

    # Stratify by company, topic and detector label.
    # Maximum 2 examples per cell to keep workload
    # manageable. Does not label data automatically.
    for _, group in candidates.groupby(
        ["ticker", "topic", "label"]
    ):
        n = min(2, len(group))
        selection = rng.choice(
            group.index.to_numpy(),
            size=n, replace=False
        )
        sampled.append(
            candidates.loc[selection]
        )

    if sampled:
        output = pd.concat(
            sampled, ignore_index=True
        )
    else:
        output = candidates.iloc[:0].copy()

    output["human_event_label"] = ""
    output["human_relation_label"] = ""
    output["human_reviewer"] = ""
    output["review_notes"] = ""
    output["human_reviewed"] = False

    return output.sort_values(
        ["ticker", "topic", "filing_date"]
    ).reset_index(drop=True)


# -------------------------------------------------
# EVENT GRAPH (REIFIED, PROVENANCE PRESERVED)
# -------------------------------------------------

def create_graph(changes):
    nodes = {}
    edges = []

    def node(identifier, kind, label):
        nodes[identifier] = {
            "node_id": identifier,
            "node_type": kind,
            "label": str(label)
        }

    for ticker in STOCKS:
        node(
            "COMPANY:" + ticker,
            "COMPANY", ticker
        )

    for r in changes.itertuples(index=False):
        company = "COMPANY:" + r.ticker
        document = (
            "DOCUMENT:" + r.ticker + ":" + r.accession
        )
        topic = "CONCEPT:" + r.topic
        event = "EVIDENCE:" + str(r.evidence_id)

        node(document, "DOCUMENT", r.accession)
        node(topic, "CONCEPT", r.topic)
        node(event, "EVIDENCE_EVENT", r.evidence_id)

        for src, dst, typ in [
            (company, event, "HAS_EVIDENCE"),
            (event, document, "FROM_DOCUMENT"),
            (event, topic, "MENTIONS_CONCEPT"),
        ]:
            edges.append({
                "edge_id": item_id(
                    r.evidence_id, src, dst, typ
                ),
                "ticker": r.ticker,
                "source_id": src,
                "target_id": dst,
                "edge_type": typ,
                "available_from": r.available_from,
                "evidence_id": r.evidence_id,
                "change_label": r.label,
                "economic_relation_verified": False,
            })

    return (
        pd.DataFrame(list(nodes.values())),
        pd.DataFrame(edges)
    )


# -------------------------------------------------
# DAILY POINT-IN-TIME PANEL
# -------------------------------------------------

def create_daily(filings, changes, market):
    summaries = []

    labels = [
        "SUBSTANTIVE_TEXT_CANDIDATE",
        "NUMERIC_OR_FORMAT_CHANGE",
        "MINOR_WORDING_CHANGE",
        "BOILERPLATE_SIMILAR",
        "EXACT_REPEAT",
    ]

    counts = (
        changes.groupby(
            ["ticker", "accession", "label"]
        ).size().unstack(fill_value=0)
    )

    for f in filings.to_dict("records"):
        key = (f["ticker"], str(f["accession"]))
        row = {
            "ticker": f["ticker"],
            "accession": str(f["accession"]),
            "form": f["form"],
            "filing_date": f["filing_date"],
            "available_from": f["available_from"],
        }
        for label in labels:
            row[label] = (
                int(counts.at[key, label])
                if key in counts.index
                and label in counts.columns
                else 0
            )
        for topic in TOPICS:
            row["rate_" + topic] = float(
                f["v28_" + topic + "_per10k"]
            )
        summaries.append(row)

    summaries = pd.DataFrame(summaries)
    records = []

    for ticker in STOCKS:
        docs = (
            summaries[summaries["ticker"].eq(ticker)]
            .sort_values(
                ["available_from", "accession"]
            )
            .to_dict("records")
        )

        days = market.loc[
            market["ticker"].eq(ticker),
            "date"
        ].sort_values().tolist()

        ptr = 0
        latest_by_form = {}
        last_filing = None
        known = 0

        for day in days:
            updates = []
            while (
                ptr < len(docs)
                and docs[ptr]["available_from"] <= day
            ):
                f = docs[ptr]
                if not f["filing_date"] < day:
                    raise ValueError("Same-day filing leakage")
                latest_by_form[f["form"]] = f
                last_filing = f
                known += 1
                updates.append(f)
                ptr += 1

            age = (
                (day - last_filing["available_from"]).days
                if last_filing is not None else -1
            )

            state = {
                "date": day,
                "ticker": ticker,
                "has_filing": int(known > 0),
                "known_filings": known,
                "filings_today": len(updates),
                "filing_age": age,
                "last_accession": (
                    last_filing["accession"]
                    if last_filing else ""
                ),
                "last_filing_date": (
                    last_filing["filing_date"]
                    if last_filing else pd.NaT
                ),
                "latest_10k": int("10-K" in latest_by_form),
                "latest_10q": int("10-Q" in latest_by_form),
            }

            for label in labels:
                state["new_" + label] = sum(
                    int(doc[label]) for doc in updates
                )

            # Topic density uses separately available
            # 10-K and 10-Q rates, equal-weight baseline.
            for topic in TOPICS:
                present = [
                    float(doc["rate_" + topic])
                    for doc in latest_by_form.values()
                ]
                state["topic_" + topic] = (
                    float(np.mean(present))
                    if present else 0.0
                )

            # Not an event signal: recency information.
            state["filing_decay"] = (
                math.exp(
                    -math.log(2) * age / HALF_LIFE
                ) if age >= 0 else 0.0
            )

            records.append(state)

    daily = pd.DataFrame(records)
    daily = daily.sort_values(
        ["date", "ticker"]
    ).reset_index(drop=True)

    if len(daily) != 7545:
        raise ValueError("Unexpected daily rows")

    if daily.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate market keys")

    known_dates = pd.to_datetime(
        daily["last_filing_date"]
    )
    trade_dates = pd.to_datetime(daily["date"])
    good = known_dates.notna()

    if not (
        known_dates[good] < trade_dates[good]
    ).all():
        raise ValueError("Future information exposure")

    return daily


# -------------------------------------------------
# FINRL ABLATION
# -------------------------------------------------

def arm_features(arm):
    """
    All four arms write to identical 11 dynamic slots.
    Features absent from an arm are padded with zeros.
    """
    if arm == "FILING_ONLY":
        return [
            "filings_today",
            "filing_decay",
            "has_filing",
            "latest_10k",
            "latest_10q",
        ]

    if arm == "TOPIC_DENSITY":
        return [
            "topic_" + topic for topic in TOPICS
        ] + ["has_filing"]

    if arm == "EVIDENCE_CHANGE":
        return [
            "new_SUBSTANTIVE_TEXT_CANDIDATE",
            "new_NUMERIC_OR_FORMAT_CHANGE",
            "new_MINOR_WORDING_CHANGE",
            "new_BOILERPLATE_SIMILAR",
            "new_EXACT_REPEAT",
            "filing_decay",
            "has_filing",
        ]

    if arm == "RELATION_GRAPH":
        # No machine-only company/company claims.
        # Must wait for verified human labels.
        return []

    raise ValueError(arm)


def bridge():
    daily = pd.read_csv(OUT / "daily_features.csv")
    daily["date"] = date_series(daily["date"])

    schema = json.loads(
        (
            ROOT / "outputs/v20/fusion/feature_sets.json"
        ).read_text()
    )
    static = schema["FULL_STATIC"]
    dynamic = schema["FULL_DYNAMIC"]
    slots = [c for c in dynamic if c not in static]

    if len(slots) != 11:
        raise ValueError("Expected 11 dynamic slots")

    report = {}

    for arm in ARMS:
        if arm == "RELATION_GRAPH":
            report[arm] = {
                "status": "DISABLED_PENDING_HUMAN_VERIFICATION",
                "trained": False,
            }
            continue

        columns = arm_features(arm)
        output = OUT / "ablations" / arm
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "feature_sets.json", schema)

        for tier in TIERS:
            src = (
                ROOT / "outputs/v20/fusion"
                / f"dataset_{tier}.csv"
            )
            base = pd.read_csv(src)
            base["date"] = date_series(base["date"])
            base["ticker"] = (
                base["ticker"].astype(str).str.upper()
            )

            if set(base["ticker"]) != set(STOCKS):
                raise ValueError("Unexpected FinRL tickers")

            if base.duplicated(
                ["date", "ticker"]
            ).any():
                raise ValueError("Duplicate FinRL keys")

            base = base.drop(columns=slots)
            joined = base.merge(
                daily[
                    ["date", "ticker"] + columns
                ],
                on=["date", "ticker"],
                how="left",
                indicator=True,
                validate="one_to_one"
            )

            if (
                len(joined) != 7545
                or not joined["_merge"].eq("both").all()
            ):
                raise ValueError("Incomplete daily join")

            joined.drop(
                columns="_merge", inplace=True
            )

            train = joined["split"].eq("train")
            if not train.any():
                raise ValueError("No training split")

            scaler = {}

            for i, model_col in enumerate(slots):
                if i >= len(columns):
                    joined[model_col] = 0.0
                    scaler[model_col] = {
                        "source": "ZERO_PADDING"
                    }
                    continue

                raw = columns[i]
                series = pd.to_numeric(
                    joined[raw], errors="raise"
                ).astype(float)

                if raw.startswith("new_") or raw == "filings_today":
                    series = np.log1p(series)

                if not np.isfinite(series).all():
                    raise ValueError("Nonfinite dynamic feature")

                mean = float(series[train].mean())
                sd = float(series[train].std(ddof=0))
                if sd < 1e-12:
                    sd = 1.0

                joined[model_col] = (series - mean) / sd

                scaler[model_col] = {
                    "source": raw,
                    "train_mean": mean,
                    "train_std": sd,
                }

            for name, cols in schema.items():
                if not np.isfinite(
                    joined[cols].to_numpy(dtype=float)
                ).all():
                    raise ValueError(
                        f"Nonfinite features in {name}"
                    )

            joined.sort_values(
                ["date", "ticker"]
            ).to_csv(
                output / f"dataset_{tier}.csv",
                index=False
            )
            write_json(
                output / f"scaler_{tier}.json",
                scaler
            )

        report[arm] = {
            "status": "DATA_READY",
            "active_features": columns,
            "effective_dimensions": len(columns),
            "dynamic_slots": len(slots),
            "tiers": TIERS,
            "strict_pit_approved": False,
        }

    write_json(
        OUT / "ablation_report.json", report
    )
    print("\n=== V29 ABLATIONS ===")
    print(json.dumps(report, indent=2))


# -------------------------------------------------
# BUILD + AUDIT
# -------------------------------------------------

def build():
    OUT.mkdir(parents=True, exist_ok=True)
    filings, evidence, market = load_data()

    changes = build_changes(filings, evidence)
    nodes, edges = create_graph(changes)
    daily = create_daily(filings, changes, market)
    review = build_review_sample(changes)

    changes.to_parquet(
        OUT / "evidence_changes.parquet",
        index=False
    )
    changes.to_csv(
        OUT / "evidence_changes.csv",
        index=False
    )
    nodes.to_parquet(
        OUT / "graph_nodes.parquet",
        index=False
    )
    edges.to_parquet(
        OUT / "graph_edges.parquet",
        index=False
    )
    daily.to_csv(
        OUT / "daily_features.csv",
        index=False
    )
    review.to_csv(
        OUT / "human_review_template.csv",
        index=False
    )

    report = {
        "tickers": STOCKS,
        "source_filings": len(filings),
        "verified_evidence_spans": len(changes),
        "event_labels": changes["label"].value_counts().to_dict(),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "daily_stock_rows": len(daily),
        "review_sample_rows": len(review),
        "strict_pit_approved": False,
        "verified_economic_events": 0,
        "verified_company_relations": 0,
    }
    write_json(OUT / "build_report.json", report)

    print("\n=== V29 BUILD ===")
    print(json.dumps(report, indent=2))


def audit():
    changes = pd.read_parquet(
        OUT / "evidence_changes.parquet"
    )
    nodes = pd.read_parquet(
        OUT / "graph_nodes.parquet"
    )
    edges = pd.read_parquet(
        OUT / "graph_edges.parquet"
    )
    daily = pd.read_csv(
        OUT / "daily_features.csv"
    )

    main = nodes[
        nodes["node_type"].eq("COMPANY")
    ]
    assert set(main["node_id"]) == {
        "COMPANY:" + t for t in STOCKS
    }
    assert len(daily) == 7545
    assert set(daily["ticker"]) == set(STOCKS)
    assert not daily.duplicated(
        ["date", "ticker"]
    ).any()
    assert not changes["evidence_id"].duplicated().any()

    assert changes["source_span_verified"].all()
    assert not changes["economic_event_verified"].any()
    assert not changes["directed_relation_verified"].any()

    assert set(edges["source_id"]).issubset(
        set(nodes["node_id"])
    )
    assert set(edges["target_id"]).issubset(
        set(nodes["node_id"])
    )

    known = pd.to_datetime(
        daily["last_filing_date"]
    )
    dates = pd.to_datetime(
        daily["date"]
    )
    mask = known.notna()
    assert (known[mask] < dates[mask]).all()

    for arm in ARMS:
        if arm == "RELATION_GRAPH":
            continue
        for tier in TIERS:
            path = (
                OUT / "ablations" / arm
                / f"dataset_{tier}.csv"
            )
            df = pd.read_csv(path)
            assert len(df) == 7545
            assert set(df["ticker"]) == set(STOCKS)

    print("\n=== V29 AUDIT ===")
    print("Five-company universe: PASS")
    print("SEC excerpt provenance: PASS")
    print("Graph node/edge integrity: PASS")
    print("Historical date ordering: PASS")
    print("Three executable ablation arms: PASS")
    print("Unverified relation arm: DISABLED")
    print("Strict PIT certification: NOT COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["build", "bridge", "audit", "all"]
    )
    args = parser.parse_args()

    if args.command in ("build", "all"):
        build()
    if args.command in ("bridge", "all"):
        bridge()
    if args.command in ("audit", "all"):
        audit()

    if args.command == "all":
        print("\n=== V29 PIPELINE COMPLETE ===")
        print("Stocks:", STOCKS)
        print("Strict PIT: NOT COMPLETE")


if __name__ == "__main__":
    main()
