#!/usr/bin/env python3

import json
import subprocess
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/unified_daily_graph_v1"

entities = json.loads(
    (BASE / "entity2id.json").read_text()
)
relations = json.loads(
    (BASE / "relation2id.json").read_text()
)

N = len(entities)
R = len(relations)
D = 200

MiB = 1024 ** 2
GiB = 1024 ** 3

def show(label, size):
    print(f"{label:43s} {size / MiB:12.2f} MiB")

print("=" * 70)
print("1. GPU HARDWARE")
print("=" * 70)

subprocess.run([
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,"
    "memory.free,utilization.gpu",
    "--format=csv"
], check=False)

print("\n" + "=" * 70)
print("2. UNIFIED KG MEMORY ESTIMATES")
print("=" * 70)

print("Entities:", N)
print("Relations:", R)
print("Hidden dimension:", D)

pairwise = N * (N + 1) * 4

show("One dense pairwise timestamp state", pairwise)
show("Two directional timestamp states", 2 * pairwise)

show(
    "Static entity embeddings (2 aspects)",
    2 * N * D * 4
)

show(
    "Dynamic structural entity embeddings",
    N * D * 4
)

show(
    "Dynamic temporal entity embeddings (2 dirs)",
    2 * N * D * 4
)

show(
    "Dynamic relation embeddings (2 aspects/dirs)",
    4 * R * D * 4
)

print("\n" + "=" * 70)
print("3. CUDA MEMORY STATUS")
print("=" * 70)

if not torch.cuda.is_available():
    print("CUDA unavailable in this environment.")
else:
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        with torch.cuda.device(i):
            free, total = torch.cuda.mem_get_info()
            reserved = torch.cuda.memory_reserved(i)
            allocated = torch.cuda.memory_allocated(i)

        print(f"\nGPU {i}: {props.name}")
        print(f"Total VRAM:     {total / GiB:.2f} GiB")
        print(f"Free VRAM:      {free / GiB:.2f} GiB")
        print(f"Allocated:      {allocated / MiB:.2f} MiB")
        print(f"Reserved:       {reserved / MiB:.2f} MiB")

        estimated = 2 * pairwise

        print(
            "Dense time states / total VRAM: "
            f"{estimated / total * 100:.1f}%"
        )
        print(
            "Dense time states / free VRAM:  "
            f"{estimated / free * 100:.1f}%"
            if free > 0 else
            "No free VRAM"
        )

print("\n" + "=" * 70)
print("4. CUDA / PYTORCH")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

print("\nNOTE:")
print("This is a capacity estimate, not a training benchmark.")
print("No large GPU tensors were allocated.")
print("Run an actual model forward/backward profile next.")
