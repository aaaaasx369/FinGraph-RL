#!/usr/bin/env python3
"""
Unified FinKG SEC Numeric As-of Features V1

Inputs:
    outputs/unified_temporal_kg_v3/
      - unified_temporal_edges.csv
      - sec_signal_nodes.csv
      - entity2id.json
      - relation2id.json

Outputs:
    outputs/unified_sec_fusion_v1/
      - sec_asof_features.csv
      - normalization.json
      - metadata.json

Policy:
    At date D, only SEC signals with event_date < D are used.
    Same-day information is excluded.
    Max age defaults to 180 calendar days.
    Missing signals are explicitly masked.

This is an engineering timestamp cutoff.
It does not certify that upstream event timestamps are PIT-safe.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
FEATURES = ["impact", "risk", "confidence"]


def find_column(df, candidates, label):
    lower = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    raise RuntimeError(
        f"Cannot resolve {label}.\n"
        f"Columns: {list(df.columns)}\n"
        f"Candidates: {candidates}"
    )


def optional_column(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    return None


def load_mapping(path):
    obj = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(obj, dict):
        return {str(k): int(v) for k, v in obj.items()}

    raise RuntimeError(f"Unexpected mapping format: {path}")


def parse_dates(series, name):
    result = pd.to_datetime(series, errors="coerce", utc=True)

    if result.isna().any():
        examples = series[result.isna()].head(5).tolist()
        raise RuntimeError(
            f"Unparseable {name}: {examples}"
        )

    return result.dt.tz_convert(None).dt.normalize()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kg-dir",
        default=str(
            ROOT / "outputs/unified_temporal_kg_v3"
        ),
    )

    parser.add_argument(
        "--out",
        default=str(
            ROOT / "outputs/unified_sec_fusion_v1"
        ),
    )

    parser.add_argument(
        "--start",
        default="2018-01-01",
    )

    parser.add_argument(
        "--end",
        default="2023-12-31",
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=180,
    )

    args = parser.parse_args()

    kg_dir = Path(args.kg_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    edges = pd.read_csv(
        kg_dir / "unified_temporal_edges.csv",
        low_memory=False,
    )

    signals = pd.read_csv(
        kg_dir / "sec_signal_nodes.csv",
        low_memory=False,
    )

    entity2id = load_mapping(
        kg_dir / "entity2id.json"
    )

    relation2id = load_mapping(
        kg_dir / "relation2id.json"
    )

    print("=" * 72)
    print("UNIFIED SEC AS-OF BUILDER")
    print("=" * 72)

    print("Edge columns:", list(edges.columns))
    print("Signal columns:", list(signals.columns))

    company_ids = {}

    for ticker in TICKERS:
        key = f"COMPANY:{ticker}"

        if key not in entity2id:
            raise RuntimeError(
                f"Company missing from entity2id: {key}"
            )

        company_ids[ticker] = entity2id[key]

    print("Portfolio company IDs:", company_ids)

    # --------------------------------------------------
    # 1. Detect graph schema
    # --------------------------------------------------

    src_col = find_column(
        edges,
        [
            "head",
            "head_id",
            "src",
            "src_id",
            "source",
            "source_id",
            "subject",
            "subject_id",
            "h",
        ],
        "edge source",
    )

    dst_col = find_column(
        edges,
        [
            "tail",
            "tail_id",
            "dst",
            "dst_id",
            "target",
            "target_id",
            "object",
            "object_id",
            "t",
        ],
        "edge target",
    )

    rel_col = find_column(
        edges,
        [
            "relation",
            "relation_id",
            "rel",
            "rel_id",
            "predicate",
            "rel_type",
        ],
        "edge relation",
    )

    time_col = find_column(
        edges,
        [
            "available_at",
            "event_date",
            "date",
            "timestamp",
            "time",
            "event_time",
            "valid_from",
        ],
        "edge timestamp",
    )

    node_col = find_column(
        signals,
        [
            "node_id",
            "entity_id",
            "signal_node_id",
            "event_node_id",
            "sec_signal_node_id",
            "id",
            "entity",
            "node",
        ],
        "signal node ID",
    )

    feature_cols = {}

    for feature in FEATURES:
        feature_cols[feature] = find_column(
            signals,
            [
                feature,
                f"sec_{feature}",
                f"{feature}_score",
            ],
            feature,
        )

    if "HAS_SEC_SIGNAL" not in relation2id:
        raise RuntimeError(
            "HAS_SEC_SIGNAL missing from relation2id"
        )

    signal_relation_id = relation2id[
        "HAS_SEC_SIGNAL"
    ]

    # --------------------------------------------------
    # 2. SEC signal graph edges
    # --------------------------------------------------

    rel_numeric = pd.to_numeric(
        edges[rel_col],
        errors="coerce",
    )

    is_sec_edge = (
        edges[rel_col].astype(str).eq(
            "HAS_SEC_SIGNAL"
        )
        |
        rel_numeric.eq(signal_relation_id)
    )

    sec_edges = edges.loc[
        is_sec_edge,
        [src_col, dst_col, time_col],
    ].copy()

    print("HAS_SEC_SIGNAL edges:", len(sec_edges))

    if sec_edges.empty:
        raise RuntimeError(
            "No HAS_SEC_SIGNAL edges found"
        )


    # Unified KG V3 can store entities as
    # canonical text identifiers (COMPANY:MSFT)
    # rather than integer IDs.
    #
    # Convert using entity2id.json, while also
    # accepting already-numeric global IDs.

    def resolve_entity_ids(values, label):
        mapping = {
            str(k): int(v)
            for k, v in entity2id.items()
        }

        valid_ids = set(mapping.values())

        def resolve(value):
            if pd.isna(value):
                raise RuntimeError(
                    f"{label}: missing entity"
                )

            key = str(value).strip()

            if key in mapping:
                return mapping[key]

            try:
                number = int(key)
            except ValueError:
                number = None

            if number is not None and number in valid_ids:
                return number

            raise RuntimeError(
                f"{label}: unknown entity {key!r}"
            )

        return values.map(resolve).astype("int64")

    sec_edges["_src"] = resolve_entity_ids(
        sec_edges[src_col],
        "SEC source",
    )

    sec_edges["_dst"] = resolve_entity_ids(
        sec_edges[dst_col],
        "SEC target",
    )

    sec_edges["_date"] = parse_dates(
        sec_edges[time_col],
        "SEC edge date",
    )

    signals = signals.copy()

    signals["_signal_node_id"] = resolve_entity_ids(
        signals[node_col],
        "SEC signal node",
    )

    if signals["_signal_node_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate signal node IDs detected"
        )

    signal_ids = set(
        signals["_signal_node_id"].tolist()
    )

    portfolio_id_to_ticker = {
        int(v): k for k, v in company_ids.items()
    }

    # Determine graph direction using the actual
    # portfolio company IDs and signal node IDs.
    forward = (
        sec_edges["_src"].isin(
            portfolio_id_to_ticker
        )
        &
        sec_edges["_dst"].isin(signal_ids)
    )

    reverse = (
        sec_edges["_dst"].isin(
            portfolio_id_to_ticker
        )
        &
        sec_edges["_src"].isin(signal_ids)
    )

    matched_forward = int(forward.sum())
    matched_reverse = int(reverse.sum())

    print("Forward matches:", matched_forward)
    print("Reverse matches:", matched_reverse)

    if matched_forward and matched_reverse:
        raise RuntimeError(
            "Mixed SEC signal edge directions. "
            "Resolve schema explicitly."
        )

    if matched_forward:
        selected = sec_edges.loc[forward].copy()
        selected["company_id"] = selected["_src"]
        selected["signal_id"] = selected["_dst"]

    elif matched_reverse:
        selected = sec_edges.loc[reverse].copy()
        selected["company_id"] = selected["_dst"]
        selected["signal_id"] = selected["_src"]

    else:
        raise RuntimeError(
            "No edges connect portfolio companies "
            "to SEC signal nodes. Check node IDs "
            "and graph schema."
        )

    selected["ticker"] = selected[
        "company_id"
    ].map(portfolio_id_to_ticker)

    signal_values = signals[
        ["_signal_node_id"]
        + list(feature_cols.values())
    ].copy()

    signal_values = signal_values.rename(
        columns={
            "_signal_node_id": "signal_id",
            **{
                feature_cols[k]: k
                for k in FEATURES
            },
        }
    )

    events = selected.merge(
        signal_values,
        how="left",
        on="signal_id",
        validate="many_to_one",
    )

    if events[FEATURES].isna().all(axis=1).any():
        raise RuntimeError(
            "At least one signal edge has no "
            "matching numeric signal values"
        )

    for f in FEATURES:
        events[f] = pd.to_numeric(
            events[f],
            errors="coerce",
        )

    if events[FEATURES].isna().any().any():
        raise RuntimeError(
            "Incomplete SEC numeric values detected. "
            "No automatic zero imputation is performed."
        )

    events = events.rename(
        columns={"_date": "event_date"}
    )

    events = events.sort_values(
        ["ticker", "event_date", "signal_id"]
    ).drop_duplicates(
        ["ticker", "event_date", "signal_id"]
    )

    print("Portfolio SEC signal events:", len(events))

    # --------------------------------------------------
    # 3. Training-only normalization
    # --------------------------------------------------

    train_cutoff = pd.Timestamp("2022-01-01")

    train_events = events[
        events["event_date"] < train_cutoff
    ]

    if train_events.empty:
        raise RuntimeError(
            "No pre-2022 SEC events available "
            "for fitting normalization"
        )

    stats = {}

    for f in FEATURES:
        mean = float(train_events[f].mean())
        std = float(
            train_events[f].std(ddof=0)
        )

        if not np.isfinite(mean):
            raise RuntimeError(
                f"Nonfinite training mean: {f}"
            )

        if not np.isfinite(std):
            raise RuntimeError(
                f"Nonfinite training std: {f}"
            )

        if std < 1e-8:
            std = 1.0

        stats[f] = {
            "mean": mean,
            "std": std,
        }

        events[f + "_norm"] = (
            (events[f] - mean) / std
        ).clip(-5.0, 5.0)

    # --------------------------------------------------
    # 4. Build business-day As-of features
    # --------------------------------------------------

    dates = pd.bdate_range(
        args.start,
        args.end,
    )

    # These are weekday proxy dates, NOT
    # verified NYSE trading sessions.

    records = []

    for ticker in TICKERS:

        company_events = (
            events[events["ticker"] == ticker]
            .sort_values(
                ["event_date", "signal_id"]
            )
        )

        # Multiple signals on the same date are
        # aggregated as that date's mean.
        daily = company_events.groupby(
            "event_date",
            as_index=False
        ).agg({
            **{
                f + "_norm": "mean"
                for f in FEATURES
            },
            **{
                f: "mean"
                for f in FEATURES
            },
            "signal_id": "count",
        })

        # Advance only through events STRICTLY
        # before the requested feature date.
        values = daily.to_dict("records")
        cursor = 0
        latest = None

        for day in dates:

            while (
                cursor < len(values)
                and pd.Timestamp(
                    values[cursor]["event_date"]
                ) < day
            ):
                latest = values[cursor]
                cursor += 1

            age_days = -1
            has_signal = 0

            if latest is not None:
                age_days = int(
                    (
                        day
                        - pd.Timestamp(
                            latest["event_date"]
                        )
                    ).days
                )

                if 0 < age_days <= args.max_age:
                    has_signal = 1

            record = {
                "date": day.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "company_id": company_ids[ticker],
                "sec_mask": has_signal,
                "sec_age_days": (
                    age_days if has_signal else -1
                ),
                "sec_event_date": (
                    pd.Timestamp(
                        latest["event_date"]
                    ).strftime("%Y-%m-%d")
                    if has_signal else ""
                ),
                "sec_signal_count": (
                    int(latest["signal_id"])
                    if has_signal else 0
                ),
            }

            for f in FEATURES:

                record[f] = (
                    float(latest[f])
                    if has_signal else 0.0
                )

                record[f + "_norm"] = (
                    float(latest[f + "_norm"])
                    if has_signal else 0.0
                )

            records.append(record)

    result = pd.DataFrame(records)

    # --------------------------------------------------
    # 5. Leakage and consistency checks
    # --------------------------------------------------

    observed = result[
        result["sec_mask"] == 1
    ].copy()

    if not observed.empty:

        event_dates = pd.to_datetime(
            observed["sec_event_date"]
        )

        feature_dates = pd.to_datetime(
            observed["date"]
        )

        assert bool(
            (event_dates < feature_dates).all()
        ), "Same-day/future SEC leakage detected"

        assert bool(
            (observed["sec_age_days"] > 0).all()
        )

        assert bool(
            (
                observed["sec_age_days"]
                <= args.max_age
            ).all()
        )

    missing = result[
        result["sec_mask"] == 0
    ]

    for f in FEATURES:
        assert bool(
            (missing[f + "_norm"] == 0.0).all()
        )

    assert not result.duplicated(
        ["date", "ticker"]
    ).any()

    # --------------------------------------------------
    # 6. Write outputs
    # --------------------------------------------------

    result_path = out / "sec_asof_features.csv"

    result.to_csv(
        result_path,
        index=False
    )

    (out / "normalization.json").write_text(
        json.dumps(
            {
                "fitted_on": (
                    "SEC events strictly before "
                    "2022-01-01"
                ),
                "stats": stats,
                "clip": [-5.0, 5.0],
            },
            indent=2,
        ),
        encoding="utf-8"
    )

    metadata = {
        "company_ids": company_ids,
        "entities": len(entity2id),
        "relations": len(relation2id),
        "sec_relation_id": signal_relation_id,
        "portfolio_signal_events": len(events),
        "date_start": args.start,
        "date_end": args.end,
        "max_age_days": args.max_age,
        "feature_rows": len(result),
        "masked_rows": int(
            (result["sec_mask"] == 0).sum()
        ),
        "available_rows": int(
            (result["sec_mask"] == 1).sum()
        ),
        "same_day_excluded": True,
        "calendar": "weekday_proxy_not_nyse",
        "strict_pit_certified": False,
        "note": (
            "SEC available-at timestamps must "
            "be independently audited."
        ),
    }

    (out / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8"
    )

    print()
    print("FEATURE ROWS:", len(result))
    print("AVAILABLE ROWS:", metadata["available_rows"])
    print("MASKED ROWS:", metadata["masked_rows"])
    print("OUTPUT:", result_path)
    print("SEC AS-OF FEATURE BUILD PASS")


if __name__ == "__main__":
    main()
