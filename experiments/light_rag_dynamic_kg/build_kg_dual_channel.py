#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SOURCE = (
    ROOT / "results/light_rag_dynamic_kg/ablation_v1"
)

OUT = (
    ROOT / "results/light_rag_dynamic_kg/kg_dual_channel_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

TICKERS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

RELATIONS = [
    "Relate_To",
    "Control",
    "Operate_In",
    "Impact",
    "Has",
    "Negative_Impact_On",
    "Raise",
    "Is_Member_Of",
    "Participates_In",
    "Announce",
    "Invests_In",
    "Produce",
    "Introduce",
    "Decrease",
    "Positive_Impact_On",
]

RELATION_LOOKUP = {
    str(i): name
    for i, name in enumerate(RELATIONS)
}

BASE_COLUMNS = [
    "impact",
    "risk",
    "confidence",
    "impact_valid",
    "risk_valid",
    "confidence_valid",
    "log_evidence_count",
    "log_relation_count",
    "relation_diversity",
    "retrieval_valid",
    "summary_present",
    "freshness",
]

RELATION_COLUMNS = [
    "relation_" + str(i).zfill(2)
    for i in range(15)
] + ["relation_other"]

FEATURE_COLUMNS = BASE_COLUMNS + RELATION_COLUMNS

assert len(FEATURE_COLUMNS) == 28


def safe_number(value, lo, hi):
    if value is None or isinstance(value, bool):
        return None

    try:
        x = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(x):
        return None

    return x if lo <= x <= hi else None


def relation_index(value):
    name = str(value).strip()

    if name.startswith("FINDKG_REL:"):
        suffix = name.split(":", 1)[1]

        if suffix.isdigit() and int(suffix) < 15:
            return int(suffix)

    if name in RELATIONS:
        return RELATIONS.index(name)

    return 15


def asof_ok(item, date):
    timestamp = pd.to_datetime(
        item.get("available_at"),
        utc=True,
        errors="coerce",
    )

    if pd.isna(timestamp):
        return False

    cutoff = pd.Timestamp(date).normalize()

    # Conservative pre-decision-day restriction.
    return timestamp.tz_localize(None) < cutoff


def read_records():
    records = []
    files = sorted(SOURCE.glob("*_A_KG_RAG.json"))

    for path in files:
        try:
            data = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (ValueError, OSError):
            # May be an incomplete concurrent write.
            continue

        ticker = data.get("ticker")
        date = data.get("date")

        if ticker not in TICKERS:
            continue

        try:
            dt = pd.Timestamp(date).normalize()
        except Exception:
            continue

        if not (
            pd.Timestamp("2018-01-01")
            <= dt
            <= pd.Timestamp("2021-12-31")
        ):
            continue

        if data.get("group") != "A_KG_RAG":
            continue

        evidence = data.get("evidence") or []

        if not isinstance(evidence, list):
            continue

        # Never treat source-tagged SEC evidence
        # as KG-only evidence.
        if any(
            not isinstance(e, dict)
            or e.get("type") != "KG"
            for e in evidence
        ):
            print("[SKIP SOURCE VIOLATION]", path)
            continue

        if not all(
            asof_ok(e, date) for e in evidence
        ):
            print("[SKIP PIT VIOLATION]", path)
            continue

        generated = data.get("generated") or {}
        validation = data.get("validation") or {}

        if not isinstance(generated, dict):
            generated = {}

        if not isinstance(validation, dict):
            validation = {}

        summary = generated.get("summary") or ""

        if not isinstance(summary, str):
            summary = ""

        summary = summary.strip()

        counts = np.zeros(16, dtype=np.float32)

        for item in evidence:
            idx = relation_index(
                item.get("relation", "")
            )
            counts[idx] += 1

        count = len(evidence)
        unique_relation_types = int(
            (counts > 0).sum()
        )

        # Raw candidate scores are NOT semantically
        # certified. Require quote-level checks before
        # exposing candidate score masks.
        citation_ok = bool(
            validation.get("quote_checks_passed")
        )

        ranges = [
            ("impact", -1, 1),
            ("risk", 0, 1),
            ("confidence", 0, 1),
        ]

        scores = []

        masks = []

        for key, lo, hi in ranges:
            value = safe_number(
                validation.get(key), lo, hi
            )

            valid = (
                value is not None
                and citation_ok
                and count > 0
            )

            scores.append(
                value if valid else 0.0
            )

            masks.append(
                1.0 if valid else 0.0
            )

        # Structural observations are measured from
        # returned evidence, not model judgement.
        # Count normalization is fixed, not fitted
        # using future test data.
        relation_features = (
            counts / max(count, 1)
        )

        vector = np.array(
            scores
            + masks
            + [
                np.log1p(count) / np.log1p(80),
                np.log1p(unique_relation_types)
                / np.log1p(16),
                unique_relation_types / 16.0,
                float(count > 0),
                float(bool(summary) and count > 0),
                1.0,
            ]
            + relation_features.tolist(),
            dtype=np.float32,
        )

        assert vector.shape == (28,)

        # Keep the original raw evidence identities.
        evidence_ids = [
            e["id"] for e in evidence
            if "id" in e
        ]

        records.append({
            "ticker": ticker,
            "date": str(dt.date()),
            "vector": vector,
            "summary": summary,
            "evidence_ids": evidence_ids,
            "evidence_count": count,
            "unique_relations": unique_relation_types,
            "citation_ok": citation_ok,
            "candidate_scores": int(sum(masks)),
            "semantic_verified": False,
            "source_path": str(path),
        })

    return records


def embed_summary(client, summary):
    response = client.embed(
        model="bge-m3:latest",
        input=summary,
    )

    vec = np.asarray(
        response.embeddings[0],
        dtype=np.float32,
    )

    if vec.shape != (1024,):
        raise RuntimeError(
            f"Unexpected BGE embedding dim: {vec.shape}"
        )

    if not np.isfinite(vec).all():
        raise RuntimeError("Nonfinite embedding")

    norm = np.linalg.norm(vec)

    if norm > 0:
        vec = vec / norm

    return vec


def make_embeddings(records, enabled):
    embeddings = np.zeros(
        (len(records), 1024),
        dtype=np.float32,
    )

    if not enabled:
        return embeddings

    import ollama

    client = ollama.Client(
        host="http://140.113.87.164:11434",
        timeout=1200,
    )

    cache = OUT / "embedding_cache"
    cache.mkdir(parents=True, exist_ok=True)

    for i, row in enumerate(records):
        summary = row["summary"]

        if (
            not summary
            or row["evidence_count"] == 0
        ):
            continue

        key = hashlib.sha256(
            (
                "bge-m3:latest|"
                + summary
            ).encode("utf-8")
        ).hexdigest()

        path = cache / f"{key}.npy"

        if path.exists():
            vec = np.load(path)
        else:
            vec = embed_summary(client, summary)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, vec)
            tmp.replace(path)

        if vec.shape != (1024,):
            raise RuntimeError(
                "Cached embedding shape mismatch"
            )

        embeddings[i] = vec

        if (i + 1) % 10 == 0:
            print(
                f"[EMBED] {i+1}/{len(records)}",
                flush=True,
            )

    return embeddings


def get_native_dates():
    from unified_event_memory.finrl_x.aegis_rl import (
        run_v16_dual_rag_finrl as v16
    )

    old = sys.argv

    try:
        sys.argv = ["run_v16_dual_rag_finrl.py"]
        args = v16.parse_args()
    finally:
        sys.argv = old

    train, val, dates, contexts, n_train = (
        v16.v8b.load_data(args)
    )

    dates = pd.DatetimeIndex(
        v16.base.normalize_dates(dates)
    ).normalize()

    if not (
        dates[0].year == 2018
        and dates[n_train].year == 2021
        and dates[-1].year == 2021
    ):
        raise RuntimeError(
            "Unexpected native train/validation split"
        )

    return dates, n_train


def align(records, embeddings, dates, max_age_days):
    n = len(dates)

    structured = np.zeros(
        (n, 5, 28),
        dtype=np.float32,
    )

    semantic = np.zeros(
        (n, 5, 1024),
        dtype=np.float32,
    )

    source_index = np.full(
        (n, 5),
        -1,
        dtype=np.int32,
    )

    # Separate each ticker's historical updates.
    by_ticker = {}

    for ticker in TICKERS:
        by_ticker[ticker] = sorted(
            [
                (pd.Timestamp(r["date"]), i)
                for i, r in enumerate(records)
                if r["ticker"] == ticker
            ]
        )

    for j, ticker in enumerate(TICKERS):
        updates = by_ticker[ticker]
        pos = 0
        latest_idx = None
        latest_day = None

        for t, day in enumerate(dates):
            # Prevent feature state carrying from
            # training episodes into validation.
            if (
                t > 0
                and dates[t-1].year <= 2020
                and day.year >= 2021
            ):
                latest_idx = None
                latest_day = None

            # A historical snapshot becomes usable
            # only AFTER its snapshot date.
            while (
                pos < len(updates)
                and updates[pos][0] < day
            ):
                latest_day, latest_idx = updates[pos]
                pos += 1

            if latest_idx is None:
                continue

            age = int((day - latest_day).days)

            if (
                age <= 0
                or age > max_age_days
            ):
                continue

            row = records[latest_idx]

            x = row["vector"].copy()

            # Freshness declines with staleness.
            x[11] = 1.0 - (
                age / max_age_days
            )

            structured[t, j] = x

            semantic[t, j] = (
                embeddings[latest_idx]
            )

            source_index[t, j] = latest_idx

    return structured, semantic, source_index


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embed-summary",
        action="store_true",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--align-finrl",
        action="store_true",
    )

    args = parser.parse_args()

    records = read_records()

    print("[KG-RAG SNAPSHOTS]", len(records))

    if not records:
        print("[NO DATA] Waiting for historical batch.")
        return

    embeddings = make_embeddings(
        records, args.embed_summary
    )

    rows = []
    audit = []

    for i, record in enumerate(records):
        row = {
            "ticker": record["ticker"],
            "date": record["date"],
        }

        row.update(
            dict(
                zip(
                    FEATURE_COLUMNS,
                    record["vector"].tolist(),
                )
            )
        )

        row["summary"] = record["summary"]
        row["embedding_available"] = bool(
            np.linalg.norm(embeddings[i]) > 0
        )

        rows.append(row)

        audit.append({
            "ticker": record["ticker"],
            "date": record["date"],
            "evidence_count":
                record["evidence_count"],
            "unique_relations":
                record["unique_relations"],
            "candidate_scores":
                record["candidate_scores"],
            "citation_ok":
                record["citation_ok"],
            "semantic_verified": False,
            "source_path":
                record["source_path"],
            "evidence_ids":
                json.dumps(
                    record["evidence_ids"]
                ),
        })

    pd.DataFrame(rows).to_csv(
        OUT / "kg_dual_channel_snapshots.csv",
        index=False,
    )

    pd.DataFrame(audit).to_csv(
        OUT / "kg_dual_channel_audit.csv",
        index=False,
    )

    np.savez_compressed(
        OUT / "kg_dual_channel_snapshot_arrays.npz",
        structured=np.stack(
            [r["vector"] for r in records]
        ),
        summary_embeddings=embeddings,
        tickers=np.array(
            [r["ticker"] for r in records]
        ),
        dates=np.array(
            [r["date"] for r in records]
        ),
    )

    print("[SNAPSHOT SHAPE]", (len(records), 28))
    print("[SEMANTIC SHAPE]", embeddings.shape)

    if args.align_finrl:
        dates, n_train = get_native_dates()

        structured, semantic, index = align(
            records,
            embeddings,
            dates,
            args.max_age_days,
        )

        np.savez_compressed(
            OUT / "kg_dual_channel_finrl_aligned.npz",
            structured=structured,
            summary_embeddings=semantic,
            source_index=index,
            dates=dates.strftime("%Y-%m-%d").to_numpy(),
            tickers=np.array(TICKERS),
            feature_names=np.array(FEATURE_COLUMNS),
            n_train=np.array(n_train),
        )

        retrieval_valid = (
            structured[:, :, 9] > 0
        )

        numeric_valid = (
            structured[:, :, 3:6].sum(axis=2) > 0
        )

        embedding_valid = (
            np.linalg.norm(
                semantic, axis=2
            ) > 0
        )

        print("\n========== FINRL ALIGNMENT ==========")
        print("[STRUCTURED]", structured.shape)
        print("[SEMANTIC]", semantic.shape)
        print("[TRAIN DAYS]", n_train)
        print("[VAL DAYS]", len(dates)-n_train)

        for name, mask in [
            ("retrieval", retrieval_valid),
            ("numeric candidate", numeric_valid),
            ("summary embedding", embedding_valid),
        ]:
            print(
                f"[{name.upper()} TRAIN COVERAGE]",
                f"{mask[:n_train].mean():.2%}",
            )

            print(
                f"[{name.upper()} VAL COVERAGE]",
                f"{mask[n_train:].mean():.2%}",
            )

    print("\n[SAVED]", OUT)


if __name__ == "__main__":
    main()
