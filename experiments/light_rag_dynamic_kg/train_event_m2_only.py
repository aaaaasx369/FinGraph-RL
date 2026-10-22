#!/usr/bin/env python3

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg import (
    train_option_b_threeway as trainer
)

NEW_DUAL = ROOT / (
    "results/light_rag_dynamic_kg/"
    "event_driven_dual_channel_v1/"
    "kg_dual_channel_finrl_aligned.npz"
)

if not NEW_DUAL.is_file():
    raise FileNotFoundError(
        f"Event-Driven NPZ missing: {NEW_DUAL}"
    )

# Use the Event-Driven aligned data.
trainer.DUAL = NEW_DUAL

# Train M2 only. Preserve original M0/M1 models.
trainer.ARMS = ["M2"]

if __name__ == "__main__":
    trainer.main()
