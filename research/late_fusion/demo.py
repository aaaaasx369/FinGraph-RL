"""Demo: python -m research.late_fusion.demo [--text ... --graph ...]"""
import argparse
import json
import tempfile
from pathlib import Path
from .retrieval import retrieve_text_jsonl, retrieve_findkg_csv, fuse_evidence, render_llm_prompt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--as-of", default="2021-06-01")
    p.add_argument("--text", help="Dated SEC/news JSONL")
    p.add_argument("--graph", help="Normalized FinDKG temporal edges CSV")
    p.add_argument("--hops", type=int, default=1, choices=[1, 2])
    p.add_argument("--output", help="Write bundle JSON to this file")
    args = p.parse_args()
    if bool(args.text) != bool(args.graph):
        p.error("Provide both --text and --graph, or neither for the synthetic demo")
    if not args.text:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            text_path = root / "texts.jsonl"
            graph_path = root / "edges.csv"
            text_path.write_text(
                '{"id":"sec1","ticker":"AAPL","published_at":"2021-04-29","source":"SEC",'
                '"content":"AAPL stated its business faces supply chain risks."}\n'
                '{"id":"future","ticker":"AAPL","published_at":"2021-06-02",'
                '"content":"AAPL future filing: must be excluded."}\n', encoding="utf-8"
            )
            graph_path.write_text(
                "id,subject,relation,object,observed_at,available_at,source\n"
                "e1,AAPL,SUPPLIES_TO,ExampleCo,2021-05-24,2021-05-31,FinDKG\n"
                "e2,AAPL,COMPETES_WITH,FutureCo,2021-05-31,2021-06-10,FinDKG\n",
                encoding="utf-8",
            )
            bundle = build(text_path, graph_path, args)
    else:
        bundle = build(args.text, args.graph, args)
    if args.output:
        Path(args.output).write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Wrote", args.output)
    else:
        print(json.dumps(bundle, indent=2, ensure_ascii=False))
    print("\n--- Qwen prompt (not sent) ---\n")
    print(render_llm_prompt(bundle))


def build(text_path, graph_path, args):
    return fuse_evidence(args.ticker, args.as_of,
                         retrieve_text_jsonl(text_path, args.ticker, args.as_of),
                         retrieve_findkg_csv(graph_path, args.ticker, args.as_of, hops=args.hops))


if __name__ == "__main__":
    main()
