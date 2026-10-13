#!/usr/bin/env python3
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/finrl_dkg_four_native"

PREVIOUS = (
    ROOT / "outputs/finrl_dkg_original_full/"
    "original_run"
)

WORK = BASE / "original_run"

DATA = BASE / "dataset/FinDKG-full"

if not PREVIOUS.exists():
    raise RuntimeError(
        f"Previous successful run not found: {PREVIOUS}"
    )

if WORK.exists():
    raise RuntimeError(
        f"Training work already exists: {WORK}\n"
        "Refusing to overwrite it."
    )

if not (DATA / "train.txt").exists():
    raise RuntimeError(
        "Build four-relation dataset first."
    )

WORK.mkdir(parents=True)

# Reuse the actual DKG implementation that was
# used for the successful 15-relation experiment.
old_dkg = PREVIOUS / "DKG"

if not old_dkg.is_dir():
    raise RuntimeError(
        "Previous DKG library not found."
    )

shutil.copytree(old_dkg, WORK / "DKG")

# Find the previous executable training script.
candidates = []

for p in PREVIOUS.glob("*.py"):
    text = p.read_text(errors="replace")

    if (
        "DynamicGraphModel" in text
        and "KGTransformer" in text
        and "load_temporal_knowledge_graph" in text
    ):
        candidates.append(p)

if len(candidates) != 1:
    raise RuntimeError(
        "Could not uniquely identify the previous "
        "trainer; found: "
        + str([p.name for p in candidates])
    )

trainer_src = candidates[0]
text = trainer_src.read_text()

# Require existing successful configuration.
config_patterns = [
    r'(?m)^graph_mode\s*=\s*["\']FinDKG-full["\']',
    r'(?m)^model_ver\s*=\s*["\']KGTransformer["\']',
]

for pattern in config_patterns:
    if not re.search(pattern, text):
        raise RuntimeError(
            "Previous trainer config differs. "
            "Manual inspection needed: "
            + str(trainer_src)
        )

# Dataset name stays FinDKG-full in this isolated
# directory; the new edges carry exactly four IDs.
text = re.sub(
    r'(?m)^data_root_path\s*=.*$',
    'data_root_path = "./FinDKG_dataset"',
    text,
    count=1,
)

(WORK / "run_four_native.py").write_text(text)

target = WORK / "FinDKG_dataset/FinDKG-full"
target.parent.mkdir(parents=True)

shutil.copytree(DATA, target)

# Keep all files in the isolated working folder.
for folder in [
    "result/FinDKG-full/DKG",
    "logs",
    "cache",
]:
    (WORK / folder).mkdir(
        parents=True, exist_ok=True
    )

settings = WORK / "DKG/settings.py"
settings_text = settings.read_text()

if '"FinDKG-full"' not in settings_text:
    settings_text += (
        '\nif "FinDKG-full" not in ALL_GRAPHS:\n'
        '    ALL_GRAPHS.append("FinDKG-full")\n'
    )

settings.write_text(settings_text)

# Keep the original DKG loader, which already
# worked for the old FinDKG-full run.
print("\n=== FOUR-RELATION TRAINER READY ===")
print("Previous trainer:", trainer_src)
print("New trainer:", WORK / "run_four_native.py")
print("Dataset:", target)
print("Original 15-relation run: UNCHANGED")
print("New model trained: NO")
