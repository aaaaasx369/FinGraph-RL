#!/usr/bin/env python3

"""
Unified KG V3 -> FinDKG-compatible DGL event batches.

Phase 3A:
- Read existing SQLite event store.
- Stream chronological event batches.
- Build compact DGL graphs.
- Preserve global entity/relation IDs.
- Supply event time required by EventTimeHelper.
- Keep persistent relations separate from event updates.
- Attach SEC numeric attributes to signal nodes.

This is a data loader, not a trained KGTransformer.
"""

import argparse
import csv
import json
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

import torch
import dgl


ROOT = Path(__file__).resolve().parents[1]

KG_DIR = ROOT / "outputs/unified_temporal_kg_v3"
DB_PATH = (
    ROOT
    / "outputs/unified_daily_asof_v1"
    / "temporal_kg.sqlite"
)

ORIGIN = date(2018, 1, 1)

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
]


def load_json(path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_day(value):
    return date.fromisoformat(str(value)[:10])


class UnifiedTemporalLoader:

    def __init__(self):

        if not DB_PATH.exists():
            raise FileNotFoundError(DB_PATH)

        self.entity2id = load_json(
            KG_DIR / "entity2id.json"
        )

        self.relation2id = load_json(
            KG_DIR / "relation2id.json"
        )

        self.num_nodes = len(self.entity2id)
        self.num_relations = len(self.relation2id)

        self.portfolio_ids = {
            ticker: self.entity2id[
                f"COMPANY:{ticker}"
            ]
            for ticker in TICKERS
        }

        self.con = sqlite3.connect(
            str(DB_PATH)
        )

        self.con.row_factory = sqlite3.Row

        self.signal_attrs = {}

        for row in self.con.execute(
            "SELECT node, payload FROM signal_attributes"
        ):
            self.signal_attrs[row["node"]] = (
                json.loads(row["payload"])
            )

    def close(self):
        self.con.close()

    def iter_event_days(self, start, end):

        """
        Yields one batch per date with actual events.

        Historical persistent state is NOT replayed as
        a new event.
        """

        query = """
            SELECT available_at, payload
            FROM events
            WHERE available_at >= ?
              AND available_at <= ?
            ORDER BY available_at, event_id
        """

        cursor = self.con.execute(
            query,
            (
                as_day(start).isoformat(),
                as_day(end).isoformat(),
            )
        )

        current_date = None
        batch = []

        for row in cursor:

            d = row["available_at"]

            if current_date is not None and d != current_date:
                yield current_date, batch
                batch = []

            current_date = d
            batch.append(json.loads(row["payload"]))

        if batch:
            yield current_date, batch

    def graph_from_events(self, d, events):

        if not events:
            raise ValueError("Empty event batch")

        current = as_day(d)

        day_id = (current - ORIGIN).days

        if day_id < 0:
            raise ValueError("Date precedes time origin")

        global_nodes = set()

        for e in events:

            global_nodes.add(int(e["head_id"]))
            global_nodes.add(int(e["tail_id"]))

            if as_day(e["available_at"]) != current:
                raise AssertionError(
                    "Event date does not match batch date"
                )

        global_nodes = sorted(global_nodes)

        local = {
            global_id: local_id
            for local_id, global_id
            in enumerate(global_nodes)
        }

        src = torch.tensor(
            [
                local[int(e["head_id"])]
                for e in events
            ],
            dtype=torch.int64
        )

        dst = torch.tensor(
            [
                local[int(e["tail_id"])]
                for e in events
            ],
            dtype=torch.int64
        )

        g = dgl.graph(
            (src, dst),
            num_nodes=len(global_nodes)
        )

        # Preserve global entity IDs.
        g.ndata[dgl.NID] = torch.tensor(
            global_nodes,
            dtype=torch.int64
        )

        # Placeholder: original entity-type mapping
        # must be resolved before full training.
        g.ndata["node_type"] = torch.zeros(
            len(global_nodes),
            dtype=torch.int64
        )

        g.edata["rel_type"] = torch.tensor(
            [
                int(e["relation_id"])
                for e in events
            ],
            dtype=torch.int64
        )

        # Required by original EventTimeHelper.
        # Time is measured in calendar days.
        g.edata["time"] = torch.full(
            (len(events),),
            float(day_id),
            dtype=torch.float32
        )

        g.edata["occurrence_count"] = torch.tensor(
            [
                int(e["occurrence_count"])
                for e in events
            ],
            dtype=torch.int64
        )

        source_codes = {
            "FINDKG": 0,
            "SEC_RELATION": 1,
            "SEC_RAG": 2,
        }

        g.edata["source_type"] = torch.tensor(
            [
                source_codes[e["source"]]
                for e in events
            ],
            dtype=torch.int64
        )

        # Event edges are distinct from state edges.
        g.edata["edge_role"] = torch.zeros(
            len(events),
            dtype=torch.int64
        )

        # SEC-RAG attributes at signal nodes.
        # Missing values have a separate validity mask.
        features = torch.zeros(
            (len(global_nodes), 3),
            dtype=torch.float32
        )

        feature_mask = torch.zeros(
            (len(global_nodes), 3),
            dtype=torch.bool
        )

        for e in events:

            if e["source"] != "SEC_RAG":
                continue

            attr = self.signal_attrs.get(
                e["tail"]
            )

            if attr is None:
                raise AssertionError(
                    "Missing SEC signal attributes: "
                    + e["tail"]
                )

            node_idx = local[
                int(e["tail_id"])
            ]

            for j, field in enumerate(
                ("impact", "risk", "confidence")
            ):

                raw = attr.get(field)

                if raw is None or str(raw).strip() == "":
                    continue

                try:
                    value = float(raw)
                except (ValueError, TypeError):
                    continue

                if not (
                    float("-inf") < value < float("inf")
                ):
                    continue

                features[node_idx, j] = value
                feature_mask[node_idx, j] = True

        g.ndata["sec_features"] = features
        g.ndata["sec_feature_mask"] = feature_mask

        g.num_relations = self.num_relations
        g.num_all_nodes = self.num_nodes

        self.validate_graph(g, events)

        return g

    def validate_graph(self, g, events):

        assert g.num_edges() == len(events)

        assert g.ndata[dgl.NID].unique().numel() == (
            g.num_nodes()
        )

        assert torch.all(
            g.ndata[dgl.NID] >= 0
        )

        assert torch.all(
            g.ndata[dgl.NID] < self.num_nodes
        )

        assert torch.all(
            g.edata["rel_type"] >= 0
        )

        assert torch.all(
            g.edata["rel_type"] < self.num_relations
        )

        assert torch.isfinite(
            g.edata["time"]
        ).all()

        assert torch.isfinite(
            g.ndata["sec_features"]
        ).all()

    def state_asof(self, target):

        """
        Returns latest observed company relationships.

        These are historical observations.
        Their current business validity is unverified.
        """

        target = as_day(target).isoformat()

        rows = self.con.execute(
            """
            SELECT
                head,
                relation,
                tail,
                MIN(available_at) first_date,
                MAX(available_at) last_date,
                COUNT(*) n

            FROM events

            WHERE source = 'SEC_RELATION'
              AND available_at <= ?

            GROUP BY head, relation, tail
            """,
            (target,)
        ).fetchall()

        return [
            dict(r)
            for r in rows
        ]


def run(args):

    loader = UnifiedTemporalLoader()

    try:

        print("=" * 70)
        print("UNIFIED KGTRANSFORMER EVENT LOADER")
        print("=" * 70)

        print("Entities:", loader.num_nodes)
        print("Relations:", loader.num_relations)

        print("\nPortfolio IDs:")
        print(loader.portfolio_ids)

        expected = {
            "AAPL": 0,
            "MSFT": 12,
            "NVDA": 14,
            "AMZN": 4,
            "GOOGL": 10,
        }

        assert loader.portfolio_ids == expected, (
            "Portfolio entity mapping changed"
        )

        days = 0
        event_count = 0

        max_nodes = 0
        max_edges = 0
        largest_day = None

        by_source = Counter()

        sample_graph = None

        for d, events in loader.iter_event_days(
            args.start,
            args.end
        ):

            g = loader.graph_from_events(
                d,
                events
            )

            days += 1
            event_count += len(events)

            for e in events:
                by_source[e["source"]] += 1

            if g.num_nodes() > max_nodes:
                max_nodes = g.num_nodes()

            if g.num_edges() > max_edges:
                max_edges = g.num_edges()
                largest_day = d

            if d == args.sample_date:
                sample_graph = g

        print("\nEvent replay schedule:")
        print("Start:", args.start)
        print("End:", args.end)
        print("Nonempty days:", days)
        print("Total events:", event_count)
        print("Source counts:", dict(by_source))

        print("\nLargest batches:")
        print("Max nodes:", max_nodes)
        print("Max edges:", max_edges)
        print("Max edge date:", largest_day)

        # Check entire 2018-2023 event range
        if (
            args.start == "2018-01-01"
            and args.end == "2023-12-31"
        ):

            assert days == 572, (
                f"Expected 572 days, got {days}"
            )

            assert event_count == 224403, (
                f"Expected 224403, got {event_count}"
            )

            print(
                "\n2018-2023 manifest consistency: PASS"
            )

        print("\nSample date:", args.sample_date)

        if sample_graph is None:
            print(
                "No event batch exists on sample date."
            )
        else:

            g = sample_graph

            print("Nodes:", g.num_nodes())
            print("Edges:", g.num_edges())
            print(
                "Time values:",
                torch.unique(
                    g.edata["time"]
                ).tolist()
            )

            print(
                "Relation IDs:",
                torch.unique(
                    g.edata["rel_type"]
                ).numel()
            )

            print(
                "SEC feature nodes:",
                int(
                    g.ndata["sec_feature_mask"]
                    .any(dim=1)
                    .sum()
                )
            )

            print(
                "Graph fields:",
                list(g.ndata.keys()),
                list(g.edata.keys())
            )

            if args.save_sample:

                path = Path(
                    args.save_sample
                )

                path.parent.mkdir(
                    parents=True,
                    exist_ok=True
                )

                dgl.save_graphs(
                    str(path),
                    [g]
                )

                print(
                    "Saved sample graph:", path
                )

        state = loader.state_asof(
            args.state_date
        )

        print("\nPersistent relationship state:")
        print("As-of:", args.state_date)
        print("Latest-observed keys:", len(state))

        if args.state_date == "2022-06-15":
            assert len(state) == 13

        print("\nEVENT LOADER PASS")
        print(
            "NOTE: Real KGT forward/backward has not run."
        )
        print(
            "NOTE: SEC features are attached, "
            "not yet consumed by KGT."
        )
        print(
            "NOTE: PIT and relation validity unverified."
        )

    finally:
        loader.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--start",
        default="2018-01-01"
    )

    parser.add_argument(
        "--end",
        default="2023-12-31"
    )

    parser.add_argument(
        "--sample-date",
        default="2018-03-11"
    )

    parser.add_argument(
        "--state-date",
        default="2022-06-15"
    )

    parser.add_argument(
        "--save-sample",
        default=(
            "outputs/unified_kgtransformer_loader_v1/"
            "sample_20180311.bin"
        )
    )

    run(parser.parse_args())
