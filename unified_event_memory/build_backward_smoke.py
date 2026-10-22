from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "unified_event_memory/smoke_original_kgt.py"
dst = root / "unified_event_memory/smoke_kgt_backward.py"

source = src.read_text(encoding="utf-8")
marker = 'print("\\nRunning REAL EmbeddingUpdater.forward()...")'

if source.count(marker) != 1:
    raise RuntimeError("Expected Forward marker not found")

prefix = source.split(marker, 1)[0]

body = r'''
print("\n" + "=" * 70)
print("ORIGINAL KGT+RNN BACKWARD SMOKE TEST")
print("=" * 70)

updater.train()

optimizer = torch.optim.AdamW(
    updater.parameters(),
    lr=1e-4
)
optimizer.zero_grad(set_to_none=True)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

try:
    # Real KGT+RNN forward, with gradients enabled.
    updated_entity, updated_relation = updater.forward(
        prior_G=g,
        batch_G=g,
        cumul_G=g,
        static_entity_emb=static_entity,
        dynamic_entity_emb=dynamic_entity,
        dynamic_relation_emb=dynamic_relation,
        device=device,
    )

    # Only gather nodes participating in this batch.
    ids = g.ndata[dgl.NID].long().cpu()

    structural = updated_entity.structural[ids]
    temporal = updated_entity.temporal[ids]

    # DIAGNOSTIC LOSS ONLY.
    # This is NOT FinDKG's original Edge Loss.
    loss = (
        structural.square().mean()
        + temporal.square().mean()
    )

    print("Diagnostic loss:", loss.item())

    assert torch.isfinite(loss)
    assert loss.requires_grad, "Loss is detached"

    loss.backward()

    total_params = 0
    grad_params = 0
    nonzero_grad_params = 0

    structural_grad = 0
    temporal_grad = 0

    for name, p in updater.named_parameters():
        total_params += 1

        if p.grad is None:
            continue

        grad_params += 1

        assert torch.isfinite(p.grad).all(), (
            f"Non-finite gradient: {name}"
        )

        if torch.count_nonzero(p.grad).item() > 0:
            nonzero_grad_params += 1

            if "graph_structural_conv" in name:
                structural_grad += 1

            if "graph_temporal_conv" in name:
                temporal_grad += 1

    print("Total parameter tensors:", total_params)
    print("Parameter tensors with gradients:", grad_params)
    print("Nonzero-gradient tensors:", nonzero_grad_params)
    print("Structural nonzero-gradient tensors:", structural_grad)
    print("Temporal nonzero-gradient tensors:", temporal_grad)

    assert structural_grad > 0, (
        "No structural pathway gradients"
    )
    assert temporal_grad > 0, (
        "No temporal pathway gradients"
    )

    optimizer.step()

    print("\nBACKWARD + OPTIMIZER STEP PASS")

    if device.type == "cuda":
        print(
            "Peak CUDA allocated MiB:",
            round(
                torch.cuda.max_memory_allocated(device)
                / 1024**2, 2
            )
        )
        print(
            "Peak CUDA reserved MiB:",
            round(
                torch.cuda.max_memory_reserved(device)
                / 1024**2, 2
            )
        )

    print("NOTE: Diagnostic loss, not original FinDKG training.")
    print("NOTE: No SEC feature fusion or persistent state fusion.")

except Exception:
    print("\nBACKWARD FAILED — FULL TRACEBACK")
    traceback.print_exc()
    raise
'''

output = prefix + "\n" + body
compile(output, str(dst), "exec")
dst.write_text(output, encoding="utf-8")

print("Generated:", dst)
