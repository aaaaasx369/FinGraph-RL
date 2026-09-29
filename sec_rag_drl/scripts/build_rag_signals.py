#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag import SECRAGSignalGenerator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--sec-csv", required=True,
                    help="CSV columns: ticker,filing_date,filing_type,(text or text_path)")
    ap.add_argument("--out", default=str(ROOT / "data" / "rag_signals.csv"))
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sec_path = Path(args.sec_csv).resolve()
    sec = pd.read_csv(sec_path)
    required = {"ticker", "filing_date"}
    missing = required - set(sec.columns)
    if missing:
        raise ValueError(f"SEC CSV missing {sorted(missing)}")
    if "text" not in sec.columns and "text_path" not in sec.columns:
        raise ValueError("SEC CSV needs either a text column or text_path column.")

    generator = SECRAGSignalGenerator(cfg["rag_generation"])
    out = generator.process_dataframe(sec, sec_path.parent)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} RAG signals -> {out_path}")


if __name__ == "__main__":
    main()
