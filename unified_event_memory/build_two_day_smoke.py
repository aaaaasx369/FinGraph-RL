from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "unified_event_memory/smoke_original_kgt.py"
output = root / "unified_event_memory/smoke_kgt_two_days.py"

s = source.read_text(encoding="utf-8")

marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if s.count(marker) != 1:
    raise RuntimeError("Could not locate smoke test execution block")

prefix = s.split(marker, 1)[0]

# Import event loader before executing new test.
prefix += """
from kgtransformer_event_loader import UnifiedTemporalLoader

loader = UnifiedTemporalLoader()

try:
    batches = list(
        loader.iter_event_days("2018-03-11", "2018-03-25")
    )[:2]

    assert len(batches) == 2, "Need two nonempty event days"

    updater.eval()

    previous_entity = dynamic_entity
    previous_relation = dynamic_relation
    previous_day = None

    for step, (day, events) in enumerate(batches, 1):
        graph = loader.graph_from_events(day, events)

        before_time = node_latest_event_time.clone()

        with torch.no_grad():
            next_entity, next_relation = updater.forward(
                prior_G=graph,
                batch_G=graph,
                cumul_G=graph,
                static_entity_emb=static_entity,
                dynamic_entity_emb=previous_entity,
                dynamic_relation_emb=previous_relation,
                device=device,
            )

        changed_time = int(
            (node_latest_event_time != before_time).sum().item()
        )

        changed_entity = int(
            (
                next_entity.structural
                != previous_entity.structural
            ).any(dim=(1, 2)).sum().item()
        )

        print("=" * 65)
        print("REPLAY STEP:", step)
        print("Date:", day)
        print("Edges:", graph.num_edges())
        print("Nodes:", graph.num_nodes())
        print("Changed time-state entries:", changed_time)
        print("Changed entity states:", changed_entity)

        assert changed_time > 0
        assert changed_entity > 0

        for emb in (
            next_entity.structural,
            next_entity.temporal,
            next_relation.structural,
            next_relation.temporal,
        ):
            assert torch.isfinite(emb).all()

        if previous_day is not None:
            assert day > previous_day

        previous_day = day
        previous_entity = next_entity
        previous_relation = next_relation

    if device.type == "cuda":
        print(
            "Peak CUDA allocated MiB:",
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
        )

    print("=" * 65)
    print("TWO-DAY MEMORY REPLAY PASS")
    print("NOTE: Untrained parameters; diagnostic replay only.")
    print("NOTE: Persistent state and SEC features not fused.")

finally:
    loader.close()
"""

output.write_text(prefix, encoding="utf-8")
compile(prefix, str(output), "exec")

print("Generated:", output)
