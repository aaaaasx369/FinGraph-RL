# Dynamic KG event layer (lab Linux deployment)

Pure Python standard library, no GitHub, no GPU, no OpenAI API, no pip dependencies. Copy `dynamic_kg.py` into `~/projects/FinRL-Trading/` on the lab server.

```bash
cd ~/projects/FinRL-Trading
python dynamic_kg.py init-sec --sec rag/data/processed/sec/sec_chunks.jsonl
python dynamic_kg.py stats
# Import REAL extracted ReflectKG relation edges (normalized schema below):
python dynamic_kg.py import-edges --edges kg/data/reflectkg_edges_normalized.csv
python dynamic_kg.py snapshot --ticker AAPL --as-of 2021-06-01 --hops 2
python dynamic_kg.py changes --ticker AAPL --as-of 2021-06-01 --days 90
```

`init-sec` registers **every filing, including zero-edge filings**, but **does not extract relationships**. Import actual SEC-derived relation rows from your existing extraction pipeline. The existing 27-dimensional parquet features are **not** enough to reconstruct source edges. Do not feed native FinDKG numeric ID triples into this importer; their mappings are different. The `example_edges.csv` contains **synthetic** relations only for testing.

Expected CSV columns: `subject,relation,object,published_at,doc_id,chunk_id,confidence,action,text_snippet` (minimum `subject,relation,object,published_at,doc_id`). For REMOVE, require explicit confirming source evidence. Missing mention does not remove a relation.

The as-of cutoff is **exclusive** (`published_at < as_of`), using whole calendar dates. This is conservative at open-of-day. For intraday decisions, preserve timestamps in a future version. `published_at` must be source public availability date, not filing fiscal period. Source time for FinDKG must be validated separately.

The graph is currently a **separate evidence branch**, not merged into LightRAG. Call `snapshot` / `changes` and add resulting sourced relations as `[REFLECTKG_EVIDENCE]` / `[KG_CHANGES]` to your V4 Late Fusion evidence prompt. The precise V4 patch depends on the server's live `late_fusion_v4.py`; this package intentionally doesn't guess its code or overwrite it.

Important limits: one relation state per `(subject,relation,object)`, last evidence determines active state; `REMOVE` has to be explicit. This is an initial research prototype, not verified entity resolution or a production bitemporal graph. Edges imported with an existing doc_id are assumed to use the same publication date; this should be checked in the extraction pipeline.

## Qwen fusion adapter

This adapter runs alongside V4 without guessing V4 internal CLI arguments. Save the SEC and FinDKG evidence returned by your verified V4 query into two UTF-8 text files, then run:

```bash
python fusion_adapter.py --ticker AAPL --as-of 2021-06-01 \
  --question 'What changed in supply-chain risk?' \
  --sec-evidence-file /tmp/sec_evidence.txt \
  --findkg-evidence-file /tmp/findkg_evidence.txt \
  --invoke-qwen --model qwen2.5:3b \
  --ollama-url http://140.113.87.164:11434
```

Without `--invoke-qwen`, the command prints the full prompt for inspection. This is **manual-file fusion**, not yet automatic V4 retrieval hookup. Do not run a claimed full V5 ablation before wiring actual V4 retriever outputs and validating zero-leakage.
