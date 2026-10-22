#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from unified_event_memory.finrl_x.aegis_rl import (
    run_v7b_structured_evidence as v7b,
)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--kgt",
        default="outputs/aegis_kgt_daily_2018_2023/daily_dynamic_kgt.csv",
    )
    p.add_argument(
        "--kg",
        default="outputs/unified_temporal_kg_v3/unified_temporal_edges.csv",
    )
    p.add_argument(
        "--sec",
        default="rag/data/processed/sec/sec_chunks.jsonl",
    )
    p.add_argument(
        "--sec-signals",
        default="outputs/unified_event_memory_v2/sec_numeric_signals.csv",
    )
    p.add_argument(
        "--model",
        default=(
            "results/aegis_rl_v1/active_rag_v5_smoke/"
            "runs/ACTIVE__A2C__seed42/model.zip"
        ),
    )
    p.add_argument("--ollama-host", default="http://140.113.87.164:11434")
    p.add_argument("--llm", default="qwen2.5:3b")
    p.add_argument("--device", default="cpu")
    p.add_argument("--embed", default="bge-m3:latest")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--transaction-cost", type=float, default=0.001)
    p.add_argument("--cost-scale", type=float, default=0.0)
    p.add_argument("--reward-scaling", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=0.0003)
    p.add_argument("--tier", default="conservative")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", default="train")
    p.add_argument("--algo", default="A2C")
    p.add_argument("--max-days", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/aegis_rl_v1/v8_phase2c"),
    )

    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source = Path(args.kgt)
    if not source.is_file():
        raise FileNotFoundError(source)

    print("===== KGT FILE INSPECTION =====", flush=True)

    df = pd.read_csv(source)
    print("KGT shape:", df.shape, flush=True)
    print("KGT columns:", df.columns.tolist()[:40], flush=True)

    # Explicitly protect identifiers, dates and index fields.
    protected = re.compile(
        r"(^|_)(date|time|timestamp|year|month|day|"
        r"ticker|symbol|company|entity|node|relation|"
        r"id|index|split|source)(_|$)",
        re.IGNORECASE,
    )

    numeric = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if not protected.search(str(c))
    ]

    if not numeric:
        raise RuntimeError(
            "KGT has no safely identifiable numeric feature columns. "
            "Ablation not performed. Inspect the KGT CSV schema first."
        )

    print("Ablated numeric columns:", len(numeric), flush=True)
    print("Examples:", numeric[:20], flush=True)

    modified = df.copy()
    modified[numeric] = 0.0

    print("===== ORIGINAL KGT LOAD =====", flush=True)
    original = v7b.load_split(args, "train")
    X = np.asarray(original["market"], dtype=np.float32)

    # Temporary file is deleted automatically.
    with tempfile.TemporaryDirectory(prefix="v8_kgt_audit_") as td:
        modified_path = Path(td) / source.name
        modified.to_csv(modified_path, index=False)

        alt = SimpleNamespace(**vars(args))
        alt.kgt = str(modified_path)

        print("===== ABLATED KGT LOAD =====", flush=True)
        ablated = v7b.load_split(alt, "train")
        Y = np.asarray(ablated["market"], dtype=np.float32)

    if X.shape != Y.shape:
        raise RuntimeError(
            f"KGT ablation changed observation shape: {X.shape} vs {Y.shape}"
        )

    if not np.array_equal(
        np.asarray(original["dates"]),
        np.asarray(ablated["dates"]),
    ):
        raise RuntimeError("KGT ablation changed trading dates")

    if not np.allclose(
        original["prices"],
        ablated["prices"],
        rtol=0,
        atol=0,
    ):
        raise RuntimeError("KGT ablation unexpectedly changed prices")

    delta = np.abs(X.astype(np.float64) - Y.astype(np.float64))

    changed_columns = np.flatnonzero(
        (delta > 1e-8).any(axis=0)
    )

    changed_rows = np.flatnonzero(
        (delta > 1e-8).any(axis=1)
    )

    report = {
        "kgt_file": str(source),
        "kg_file": str(args.kg),
        "kgt_rows": len(df),
        "kgt_columns": len(df.columns),
        "ablated_columns": numeric,
        "market_shape": list(X.shape),
        "changed_feature_columns": changed_columns.tolist(),
        "n_changed_features": len(changed_columns),
        "n_changed_days": len(changed_rows),
        "max_abs_difference": float(delta.max()),
        "mean_abs_difference": float(delta.mean()),
        "market_depends_on_kgt": bool(len(changed_columns) > 0),
        "note": (
            "Input sensitivity to the numeric KGT CSV columns; "
            "does not prove full temporal KG provenance or point-in-time safety."
        ),
    }

    path = out / "dynamic_kg_audit.json"
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n===== DYNAMIC KG AUDIT =====", flush=True)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    if not report["market_depends_on_kgt"]:
        raise RuntimeError(
            "FAIL: market features did not change under KGT ablation. "
            "Do not claim Dynamic KGT is used until investigated."
        )

    print("\nDYNAMIC KGT INPUT DEPENDENCE: PASS", flush=True)


if __name__ == "__main__":
    main()
