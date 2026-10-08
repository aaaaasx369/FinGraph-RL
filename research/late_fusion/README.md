# SEC / LightRAG × FinDKG: evidence-level late fusion (MVP)

An isolated research prototype. No changes to existing training or 200-d KGTransformer pipeline.

## Run now (Python 3.11+, no extra packages or API key)

From repository root:

```bash
python -m research.late_fusion.demo
python -m unittest discover -s research/late_fusion -p 'test_*.py' -v
```

The default example uses **synthetic** records. It is NOT actual Apple/FinDKG evidence and does not run Qwen or LightRAG yet.

## Use real normalized data

```bash
python -m research.late_fusion.demo --ticker AAPL --as-of 2021-06-01 \
  --text path/to/dated_chunks.jsonl \
  --graph path/to/normalized_findkg_edges.csv \
  --hops 2 --output fusion_20210601.json
```

Text JSONL required columns: `content`, `published_at` (or `filing_date`), recommended `ticker`, `id`, `source`. IMPORTANT: use the genuine **public availability date**; do not assume fiscal period end or timestamp embedded in text is publication date.

Graph CSV required columns: `subject,relation,object,observed_at,available_at`; optional `id,source`. `available_at` is the earliest time the relation was publicly knowable. FinDKG's native data are **not assumed** to use these field names: write an explicit ETL mapping once the dataset release format and time semantics are verified. Weekly graph snapshots cannot automatically be treated as contemporaneously public. If that date cannot be established, exclude edge from backtests.

## Connect actual LightRAG

Current LightRAG supports `rag.aquery_data(query, param=QueryParam(mode="mix"))` returning a dict with `data.chunks`. Pass this response through:

```python
from research.late_fusion.retrieval import normalize_lightrag_query_data
items = normalize_lightrag_query_data(
    response, "AAPL", "2021-06-01",
    metadata={"chunk-id-from-lightrag": {"published_at": "2021-04-29"}}
)
```

**Critical:** build or load a point-in-time LightRAG index with ONLY documents published by the query date. A current/future-inclusive index can leak future information via retrieval ranking even if returned chunks are post-filtered. Store a document manifest with `document_id/source/published_at` and record index build cutoff for audit.

## Next interfaces

1. LightRAG adapter with dated index manifest.
2. FinDKG native-to-normalized converter (requires actual column schema and release cadence).
3. Qwen structured-output validator: score fields nullable with no evidence; track evidence IDs, prompt/model versions and error flags. Never fabricate numeric scores in fallback.
4. FinRL feature table keyed by `(date,ticker)` and missingness flags; compare technical-only, text-only, graph-only, joint retrieval against existing KGTransformer benchmark on same split/seeds.

**Not yet implemented:** ingest/index construction, live Qwen calls, FinRL state integration, backtests. This MVP produces an auditable evidence bundle only.
