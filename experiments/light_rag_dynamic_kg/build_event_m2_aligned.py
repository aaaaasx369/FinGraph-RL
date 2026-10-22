#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.light_rag_dynamic_kg import (
    build_kg_dual_channel as builder,
)

BASE = ROOT / "results/light_rag_dynamic_kg"
OUT = BASE / "event_driven_dual_channel_v1"

builder.SOURCE = OUT / "converted_ablation"
builder.OUT = OUT

if __name__ == "__main__":
    builder.OUT.mkdir(parents=True, exist_ok=True)

    sys.argv = [
        "build_event_m2_aligned.py",
        "--embed-summary",
        "--max-age-days", "35",
        "--align-finrl",
    ]

    builder.main()
