#!/usr/bin/env python3

from pathlib import Path

root = Path(__file__).resolve().parents[1]

src = root / "unified_event_memory/smoke_original_kgt.py"
dst = root / "unified_event_memory/smoke_kgt_full_replay.py"

source = src.read_text(encoding="utf-8")

marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if source.count(marker) != 1:
    raise RuntimeError("Smoke test execution marker not found")

prefix = source.split(marker, 1)[0]

body = r'''
import csv
import json
from kgtransformer_event_loader import UnifiedTemporalLoader

OUT = ROOT / "outputs/unified_kgt_replay_v1"
OUT.mkdir(parents=True, exist_ok=True)

loader = UnifiedTemporalLoader()

portfolio_ids = loader.portfolio_ids

previous_entity = dynamic_entity
previous_relation = dynamic_relation

updater.eval()

summary = []
features = []

previous_day = None
total_events = 0
day_count = 0

try:
    with torch.no_grad():

        for day, events in loader.iter_event_days(
            "2018-01-01",
            "2023-12-31"
        ):

            if previous_day is not None:
                assert day > previous_day

            graph = loader.graph_from_events(day, events)

            # Observe only current event pairs.
            src_local, dst_local = graph.edges()
            ids = graph.ndata[dgl.NID]

            h = ids[src_local].long().to(device)
            t = ids[dst_local].long().to(device)

            before_forward = (
                node_latest_event_time[t, h, 0]
                .detach()
                .clone()
            )

            before_reverse = (
                node_latest_event_time[h, t, 1]
                .detach()
                .clone()
            )

            next_entity, next_relation = updater.forward(
                prior_G=graph,
                batch_G=graph,
                cumul_G=graph,
                static_entity_emb=static_entity,
                dynamic_entity_emb=previous_entity,
                dynamic_relation_emb=previous_relation,
                device=device,
            )

            after_forward = node_latest_event_time[t, h, 0]
            after_reverse = node_latest_event_time[h, t, 1]

            changed_time = int(
                (after_forward != before_forward).sum().item()
                +
                (after_reverse != before_reverse).sum().item()
            )

            # Check only the entity states updated this day.
            changed_nodes = graph.ndata[dgl.NID].long().cpu()

            updated_struct = next_entity.structural[
                changed_nodes
            ]

            updated_temp = next_entity.temporal[
                changed_nodes
            ]

            assert torch.isfinite(updated_struct).all()
            assert torch.isfinite(updated_temp).all()

            assert torch.isfinite(
                next_relation.structural
            ).all()

            assert torch.isfinite(
                next_relation.temporal
            ).all()

            day_count += 1
            total_events += len(events)

            summary.append({
                "date": day,
                "events": len(events),
                "nodes": graph.num_nodes(),
                "changed_time_pairs": changed_time,
            })

            # Five-ticker latest memory snapshot.
            # On days with no company-specific events,
            # its state is retained from previous updates.
            for ticker, eid in portfolio_ids.items():

                structural = (
                    next_entity.structural[eid, -1]
                    .detach()
                    .cpu()
                )

                temporal = (
                    next_entity.temporal[eid, -1]
                    .mean(dim=-1)
                    .detach()
                    .cpu()
                )

                vector = torch.cat(
                    [structural, temporal],
                    dim=0
                )

                assert vector.numel() == 2 * DIM
                assert torch.isfinite(vector).all()

                features.append({
                    "date": day,
                    "ticker": ticker,
                    "entity_id": eid,
                    "embedding": vector.tolist(),
                })

            previous_entity = next_entity
            previous_relation = next_relation
            previous_day = day

            if day_count == 1 or day_count % 25 == 0:
                print(
                    f"Day {day_count:3d} | "
                    f"{day} | "
                    f"edges={len(events):5d} | "
                    f"nodes={graph.num_nodes():4d} | "
                    f"total={total_events:,}",
                    flush=True
                )

    assert day_count == 572, (
        f"Expected 572 event days, got {day_count}"
    )

    assert total_events == 224403, (
        f"Expected 224403 events, got {total_events}"
    )

    with (OUT / "replay_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    # Store embeddings in structured JSONL,
    # not a very wide 400-column CSV.
    with (OUT / "portfolio_embeddings.jsonl").open(
        "w", encoding="utf-8"
    ) as f:
        for row in features:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    metadata = {
        "model": "original FinDKG KGT+RNN",
        "trained": False,
        "checkpoint_loaded": False,
        "event_days": day_count,
        "events": total_events,
        "entities": N,
        "relations": R,
        "portfolio_tickers": list(portfolio_ids),
        "embedding_dimensions": 2 * DIM,
        "embedding_description": (
            "Untrained structural hidden state "
            "concatenated with mean of two temporal "
            "directional hidden states"
        ),
        "snapshot_frequency": (
            "event days only; not trading days"
        ),
        "sec_numeric_features_used": False,
        "persistent_state_fusion_used": False,
        "strict_pit_certified": False,
    }

    (OUT / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("FULL KGT MEMORY REPLAY PASS")
    print("=" * 70)
    print("Event days:", day_count)
    print("Total events:", total_events)
    print("Portfolio snapshots:", len(features))
    print("Output:", OUT)

    if device.type == "cuda":
        print(
            "Peak CUDA allocated MiB:",
            round(
                torch.cuda.max_memory_allocated(device)
                / 1024**2,
                2
            )
        )
        print(
            "Peak CUDA reserved MiB:",
            round(
                torch.cuda.max_memory_reserved(device)
                / 1024**2,
                2
            )
        )

    print("WARNING: Embeddings are UNTRAINED.")
    print("Do not use these as validated FinRL features.")

finally:
    loader.close()
'''

output = prefix + "\n" + body

compile(output, str(dst), "exec")
dst.write_text(output, encoding="utf-8")

print("Generated:", dst)
