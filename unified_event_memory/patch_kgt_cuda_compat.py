#!/usr/bin/env python3

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ORIGINAL = (
    ROOT / "references/FinDKG/DKG/model/embedding.py"
)

COMPAT = (
    ROOT / "unified_event_memory/fintkg_compat/embedding.py"
)

SMOKE = (
    ROOT / "unified_event_memory/smoke_original_kgt.py"
)

if not ORIGINAL.exists():
    raise FileNotFoundError(ORIGINAL)

if not SMOKE.exists():
    raise FileNotFoundError(SMOKE)

source = ORIGINAL.read_text(encoding="utf-8")

print("=" * 70)
print("BUILD ISOLATED CUDA COMPATIBILITY MODULE")
print("=" * 70)

print("Original:", ORIGINAL)
print("Compatibility:", COMPAT)

# --------------------------------------------------
# 1. Dynamic entity embeddings are stored on CPU.
#    Global graph IDs must be moved to CPU BEFORE
#    indexing these tensors.
# --------------------------------------------------

global_idx = (
    r"batch_G\.ndata\[dgl\.NID\]"
    r"\[batch_node_indices\]\.long\(\)"
)

patterns = [
    (
        "structural dynamic CPU index",
        rf"(dynamic_entity_emb\.structural\[)"
        rf"({global_idx})(\])",
    ),
    (
        "temporal dynamic CPU index",
        rf"(dynamic_entity_emb\.temporal\[)"
        rf"({global_idx})(\])",
    ),
    (
        "updated structural CPU index",
        rf"(updated_structural\[)"
        rf"({global_idx})(\])",
    ),
    (
        "updated temporal CPU index",
        rf"(updated_temporal\[)"
        rf"({global_idx})(\])",
    ),
]

patch_counts = {}

for label, pattern in patterns:

    source, count = re.subn(
        pattern,
        lambda m: (
            m.group(1)
            + m.group(2)
            + ".cpu()"
            + m.group(3)
        ),
        source,
    )

    patch_counts[label] = count
    print(f"{label}: {count}")

# --------------------------------------------------
# 2. Dynamic relation embeddings are stored on CPU.
# --------------------------------------------------

relation_patterns = [
    (
        "dynamic relation CPU index",
        "dynamic_relation_emb[batch_G_uniq_rel]",
        "dynamic_relation_emb[batch_G_uniq_rel.cpu()]",
    ),
    (
        "updated structural relation CPU index",
        "updated_structural_relation_emb[batch_G_uniq_rel]",
        "updated_structural_relation_emb[batch_G_uniq_rel.cpu()]",
    ),
    (
        "updated temporal relation CPU index",
        "updated_temporal_relation_emb[batch_G_uniq_rel]",
        "updated_temporal_relation_emb[batch_G_uniq_rel.cpu()]",
    ),
]

for label, old, new in relation_patterns:

    count = source.count(old)

    if count:
        source = source.replace(old, new)

    patch_counts[label] = count
    print(f"{label}: {count}")

# --------------------------------------------------
# 3. Keep the time-state assignment on the same
#    device. The isolated smoke test will allocate
#    the pairwise time tensor on CUDA.
# --------------------------------------------------

source = source.replace(
    "= batch_G_time.cpu()",
    "= batch_G_time",
)

source = source.replace(
    "= batch_G_max_event_time.cpu()",
    "= batch_G_max_event_time",
)

# Ensure the two confirmed dynamic entity indexing
# operations were actually changed.
assert patch_counts[
    "structural dynamic CPU index"
] >= 1

assert patch_counts[
    "temporal dynamic CPU index"
] >= 1

# Ensure the confirmed RelationRNN index was fixed.
assert patch_counts[
    "dynamic relation CPU index"
] >= 1

compile(
    source,
    str(COMPAT),
    "exec"
)

COMPAT.write_text(
    source,
    encoding="utf-8"
)

# --------------------------------------------------
# 4. Point the smoke test at the isolated module.
# --------------------------------------------------

smoke_source = SMOKE.read_text(
    encoding="utf-8"
)

backup = SMOKE.with_suffix(
    ".py.bak_before_cuda_compat"
)

if not backup.exists():
    backup.write_text(
        smoke_source,
        encoding="utf-8"
    )

old_import = (
    "from DKG.model import "
    "EmbeddingUpdater, MultiAspectEmbedding"
)

new_import = """from DKG.model import MultiAspectEmbedding

import importlib.util

compat_path = (
    ROOT
    / "unified_event_memory"
    / "fintkg_compat"
    / "embedding.py"
)

spec = importlib.util.spec_from_file_location(
    "DKG.model.embedding_compat",
    str(compat_path)
)

compat_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = compat_module
spec.loader.exec_module(compat_module)

EmbeddingUpdater = compat_module.EmbeddingUpdater

print("Using isolated CUDA-compatible EmbeddingUpdater")
"""

if old_import in smoke_source:
    smoke_source = smoke_source.replace(
        old_import,
        new_import,
        1,
    )
elif "Using isolated CUDA-compatible EmbeddingUpdater" not in smoke_source:
    raise RuntimeError(
        "Cannot find expected EmbeddingUpdater import"
    )

old_time_device = '''    device="cpu",
)'''

new_time_device = '''    device=device,
)'''

if old_time_device in smoke_source:
    smoke_source = smoke_source.replace(
        old_time_device,
        new_time_device,
        1,
    )
elif new_time_device not in smoke_source:
    raise RuntimeError(
        "Cannot find expected pairwise time-state device"
    )

# Detect accidental repeated patching.
assert smoke_source.count(
    "EmbeddingUpdater = compat_module.EmbeddingUpdater"
) == 1

compile(
    smoke_source,
    str(SMOKE),
    "exec"
)

SMOKE.write_text(
    smoke_source,
    encoding="utf-8"
)

print()
print("CUDA COMPATIBILITY PATCH PASS")
print("Original FinDKG files: UNCHANGED")
print("Smoke backup:", backup)
print("Compatibility module:", COMPAT)
