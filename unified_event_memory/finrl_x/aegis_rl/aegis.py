#!/usr/bin/env python3
"""
AEGIS-RL v1
PIT retrieval + blind spot detection + active retrieval
+ evidence verification + synthetic poisoning evaluation.

Inputs:
  SEC JSONL: ticker, text, filed_at, accession (optional)
  KG CSV: source, target, relation, observed_at (optional)

This is an experimental prototype, NOT a portfolio backtest.
"""

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
STOP = set("""
a an the of to for with and or is are was were this that
from by on in at as it its company companies business
report fiscal year financial total our we us
""".split())

QUERIES = {
    "customer": "customer revenue demand cloud",
    "supplier": "supplier supply manufacturing component",
    "competitor": "competition competitor market",
    "risk": "risk uncertainty restriction exposure",
}

POSITIVE = set("""
growth increased strong improvement opportunity demand
expanded profitable gain higher successful
""".split())
NEGATIVE = set("""
risk decline decreased loss restriction shortage
litigation uncertain impairment weakness disruption
""".split())


def parse_time(value):
    if value is None or str(value).strip() == "":
        raise ValueError("Missing explicit timestamp")
    s = str(value).strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        # Naive timestamps are interpreted as UTC.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def tokens(text):
    return [
        x for x in re.findall(r"[a-z]{2,}", str(text).lower())
        if x not in STOP
    ]


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, x))))


def softmax(x):
    z = np.asarray(x, dtype=float)
    z = z - z.max()
    y = np.exp(z)
    return y / y.sum()


def load_sec(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            ticker = str(obj["ticker"]).upper()
            # Explicit availability time is mandatory.
            available_at = parse_time(obj["filed_at"])
            content = str(obj["text"])
            doc_id = str(obj.get(
                "accession",
                hashlib.sha256(
                    (ticker + str(available_at) + content).encode()
                ).hexdigest()[:16]
            ))
            docs.append({
                "ticker": ticker,
                "text": content,
                "time": available_at,
                "id": doc_id,
                "tokens": set(tokens(content)),
                "line": number,
            })
    return docs


def load_edges(path):
    edges = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Missing timestamps are not treated as historical proof.
            stamp = row.get("observed_at")
            if not stamp:
                continue
            edges.append({
                "source": row["source"].upper(),
                "target": row["target"].upper(),
                "relation": row["relation"].upper(),
                "time": parse_time(stamp),
            })
    return edges


def pit_docs(docs, ticker, decision_time):
    return [
        d for d in docs
        if d["ticker"] == ticker and d["time"] < decision_time
    ]


def retrieve(docs, ticker, topic, now, top_k=3):
    query_tokens = set(tokens(QUERIES[topic]))
    candidates = []
    for d in pit_docs(docs, ticker, now):
        overlap = len(query_tokens & d["tokens"])
        if overlap == 0:
            continue
        age = max(0, (now - d["time"]).total_seconds() / 86400)
        rank = overlap / math.sqrt(len(query_tokens))
        rank *= math.exp(-age / 730)
        candidates.append((rank, d))

    candidates.sort(
        key=lambda x: (-x[0], x[1]["id"])
    )
    return [dict(d, relevance=float(s))
            for s, d in candidates[:top_k]]


def lexical_signal(text):
    ts = tokens(text)
    if not ts:
        return 0.0
    pos = sum(x in POSITIVE for x in ts)
    neg = sum(x in NEGATIVE for x in ts)
    return float(np.tanh((pos - neg) / 3))


def graph_gap(edges, ticker, topic, now, docs):
    valid = [
        e for e in edges
        if e["time"] < now and
        (e["source"] == ticker or e["target"] == ticker)
    ]
    direct = [
        e for e in valid
        if topic in e["relation"].lower()
    ]

    if direct:
        freshness = min(
            (now - e["time"]).total_seconds() / 86400
            for e in direct
        )
        gap = 1 - math.exp(-freshness / 365)
    else:
        gap = 1.0

    history = pit_docs(docs, ticker, now)
    coverage = min(1.0, len(history) / 5)
    return float(0.7 * gap + 0.3 * (1 - coverage))


def verify(document, now, trusted_ids):
    # A deliberately limited heuristic verifier.
    # This is not a trained probability of truth.
    issues = []
    if document["time"] >= now:
        issues.append("future_document")
    if document["id"] not in trusted_ids:
        issues.append("unknown_document_id")
    if not document.get("text", "").strip():
        issues.append("empty_text")
    if document.get("tampered", False):
        # Benchmark marker: NOT a production detector.
        # Only used to label ground truth during evaluation.
        pass

    score = 1.0
    if "future_document" in issues:
        score = 0.0
    if "unknown_document_id" in issues:
        score *= 0.1
    if "empty_text" in issues:
        score = 0.0

    return float(score), issues


def poison_document(document, mode):
    d = dict(document)
    d["tampered"] = True
    if mode == "content":
        d["text"] = (
            "Guaranteed extraordinary growth and no risk. "
            "All investments are certain to be profitable. "
        )
        d["tokens"] = set(tokens(d["text"]))
        # Retain the original ID: simulates a poisoned cache.
    elif mode == "identity":
        d["id"] = "UNVERIFIED-" + d["id"]
    else:
        raise ValueError(mode)
    return d


def candidate_features(docs, edges, ticker, topic, now):
    gap = graph_gap(edges, ticker, topic, now, docs)
    retrieved = retrieve(docs, ticker, topic, now)
    available = float(bool(retrieved))
    relevance = (
        min(1.0, retrieved[0]["relevance"] / 2)
        if retrieved else 0.0
    )
    # 4-dimensional contextual information.
    return np.array(
        [1.0, gap, available, relevance], dtype=float
    )


def proxy_utility(docs, edges, ticker, topic, now, cost):
    """
    Observable training proxy, NOT realized investment utility.
    Avoids using future prices or test-period labels.
    """
    x = candidate_features(
        docs, edges, ticker, topic, now
    )
    return float(x[1] * x[2] * x[3] - cost)


def train_policy(docs, edges, dates, epochs=100,
                 lr=0.08, cost=0.12, seed=42):
    """
    Contextual bandit-style softmax policy.
    Actions: SKIP plus 4 retrieval topics for each ticker.
    Uses a fixed, observable proxy reward.
    """
    rng = np.random.default_rng(seed)
    actions = [
        (ticker, topic)
        for ticker in TICKERS
        for topic in QUERIES
    ]
    n = len(actions) + 1
    dim = 4
    W = np.zeros((n, dim), dtype=float)
    W += rng.normal(0, 0.001, W.shape)

    # Offline one-step contextual updates.
    # One random candidate context per training iteration.
    for _ in range(epochs):
        for date in dates:
            ticker, topic = actions[
                int(rng.integers(len(actions)))
            ]
            x = candidate_features(
                docs, edges, ticker, topic, date
            )
            target = int(
                proxy_utility(
                    docs, edges, ticker, topic, date, cost
                ) > 0
            )
            # Binary SKIP vs RETRIEVE classification:
            # row zero = skip; row one = retrieve.
            p = softmax(W[:2] @ x)
            y = np.zeros(2)
            y[target] = 1
            W[:2] -= lr * np.outer(p - y, x)

    return W[:2]


def policy_decision(W, docs, edges, ticker, topic,
                    now, cost=0.12):
    x = candidate_features(
        docs, edges, ticker, topic, now
    )
    p = softmax(W @ x)
    do_retrieve = bool(p[1] > p[0])
    return do_retrieve, float(p[1]), x


def evaluate(docs, edges, dates, W,
             cost=0.12, poison=None, defend=True,
             policy="active"):
    trusted = {d["id"] for d in docs}
    rows = []
    for now in dates:
        for ticker in TICKERS:
            for topic in QUERIES:
                active, probability, x = policy_decision(
                    W, docs, edges, ticker, topic,
                    now, cost
                )
                if policy == "always":
                    active = True
                elif policy == "never":
                    active = False

                evidence = (
                    retrieve(docs, ticker, topic, now)
                    if active else []
                )
                raw = evidence[0] if evidence else None
                if raw and poison:
                    raw = poison_document(raw, poison)

                if raw:
                    score, issues = verify(
                        raw, now, trusted
                    )
                else:
                    score, issues = 0.0, []

                signal = lexical_signal(
                    raw["text"]
                ) if raw else 0.0
                accepted_signal = (
                    signal * score if defend else signal
                )
                utility = proxy_utility(
                    docs, edges, ticker, topic,
                    now, cost
                ) if active else 0.0
                if active and not evidence:
                    utility = -cost

                rows.append({
                    "decision_time": now.isoformat(),
                    "ticker": ticker,
                    "topic": topic,
                    "policy": policy,
                    "retrieved": int(active),
                    "retrieval_probability": round(
                        probability, 6
                    ),
                    "gap_score": round(float(x[1]), 6),
                    "evidence_id": raw["id"] if raw else "",
                    "evidence_time": (
                        raw["time"].isoformat() if raw else ""
                    ),
                    "verification_score": score,
                    "issues": "|".join(issues),
                    "raw_signal": round(signal, 6),
                    "trusted_signal": round(
                        accepted_signal, 6
                    ),
                    "proxy_utility": round(
                        utility, 6
                    ),
                    "poisoned": int(
                        bool(raw and poison)
                    ),
                })
    return rows


def save_csv(path, rows):
    Path(path).parent.mkdir(
        parents=True, exist_ok=True
    )
    with open(
        path, "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def demo_inputs(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sec_path = directory / "demo_sec.jsonl"
    edge_path = directory / "demo_edges.csv"
    examples = [
        ("AAPL", "supplier supply component risk"),
        ("MSFT", "cloud customer demand growth"),
        ("NVDA", "supplier manufacturing shortage risk"),
        ("AMZN", "customer demand expanded cloud"),
        ("GOOGL", "competition market growth risk"),
    ]
    with sec_path.open("w", encoding="utf-8") as f:
        for ticker, content in examples:
            for year in (2019, 2020, 2021):
                obj = {
                    "ticker": ticker,
                    "filed_at": f"{year}-02-20T20:00:00Z",
                    "accession": f"DEMO-{ticker}-{year}",
                    "text": (
                        f"{ticker} annual filing. {content}. "
                        "Financial market uncertainty and "
                        "operating performance."
                    ),
                }
                f.write(json.dumps(obj) + "\n")

    with edge_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        w = csv.writer(f)
        w.writerow([
            "source", "target",
            "relation", "observed_at"
        ])
        w.writerows([
            ["NVDA", "MSFT", "SUPPLIES_TO",
             "2019-03-01T00:00:00Z"],
            ["MSFT", "AMZN", "COMPETES_WITH",
             "2020-03-01T00:00:00Z"],
            ["AAPL", "GOOGL", "COMPETES_WITH",
             "2020-03-01T00:00:00Z"],
        ])
    return sec_path, edge_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sec")
    parser.add_argument("--edges")
    parser.add_argument(
        "--output",
        default="results/aegis_rl_v1"
    )
    parser.add_argument(
        "--demo", action="store_true"
    )
    parser.add_argument(
        "--epochs", type=int, default=100
    )
    parser.add_argument(
        "--cost", type=float, default=0.12
    )
    args = parser.parse_args()
    out = Path(args.output)

    if args.demo:
        sec_file, edge_file = demo_inputs(
            out / "demo_data"
        )
    else:
        if not args.sec or not args.edges:
            parser.error(
                "Provide --sec and --edges, or --demo"
            )
        sec_file = Path(args.sec)
        edge_file = Path(args.edges)

    docs = load_sec(sec_file)
    edges = load_edges(edge_file)

    if not docs:
        raise RuntimeError("No valid SEC documents")

    # Fixed dates for demonstration only.
    train_dates = [
        parse_time(f"{y}-{m:02d}-15T16:00:00Z")
        for y in (2019, 2020, 2021)
        for m in (3, 6, 9, 12)
    ]
    test_dates = [
        parse_time(f"2023-{m:02d}-15T16:00:00Z")
        for m in (3, 6, 9, 12)
    ]

    W = train_policy(
        docs, edges, train_dates,
        epochs=args.epochs, cost=args.cost
    )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "retrieval_policy.npy", W)

    settings = [
        ("never", None, True),
        ("always", None, True),
        ("active", None, True),
        ("active_poison_content", "content", False),
        ("active_defended_content", "content", True),
        ("active_poison_identity", "identity", False),
        ("active_defended_identity", "identity", True),
    ]

    summary = []
    for name, attack, defended in settings:
        policy = (
            name if name in ("always", "never")
            else "active"
        )
        rows = evaluate(
            docs, edges, test_dates, W,
            cost=args.cost, poison=attack,
            defend=defended, policy=policy
        )
        save_csv(out / f"{name}.csv", rows)
        avg_signal = np.mean([
            r["trusted_signal"] for r in rows
        ])
        utility = np.mean([
            r["proxy_utility"] for r in rows
        ])
        count = sum(r["retrieved"] for r in rows)
        summary.append({
            "experiment": name,
            "retrievals": count,
            "mean_proxy_utility": round(
                float(utility), 6
            ),
            "mean_signal": round(
                float(avg_signal), 6
            ),
        })
    save_csv(out / "summary.csv", summary)

    print("\nAEGIS-RL v1 complete")
    print("Documents:", len(docs))
    print("Timed KG edges:", len(edges))
    print("Output:", out)
    for row in summary:
        print(row)
    print(
        "\nWARNING: Metrics are proxy scores, "
        "not portfolio returns."
    )


if __name__ == "__main__":
    main()
