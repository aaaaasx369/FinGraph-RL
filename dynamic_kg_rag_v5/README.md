# V5 server prototype

This code is a **prototype**, not a validated financial backtest. Standard-library Python only. It does not change existing files.

On Windows PowerShell upload the ZIP with `scp "C:\Users\user\Downloads\dynamic_kg_rag_v5.zip" doralin@xiangjipai:~/`.

On the Linux server:

```bash
cd ~/projects/FinRL-Trading
unzip -o ~/dynamic_kg_rag_v5.zip
cp dynamic_kg_rag_v5/late_fusion_v5.py .
python late_fusion_v5.py query --ticker AAPL --as-of 2021-06-01 --out outputs/v5_aapl.json
```

The command above uses an explicitly labelled **SEC lexical fallback**; it does **NOT** execute LightRAG. To run LightRAG, use `--run-v4` (may be slow; V4 index must exist), or `--v4-output /path/to/previously_generated_V4.json`. Example:

```bash
python late_fusion_v5.py query --ticker AAPL --as-of 2021-06-01 --run-v4 \
  --llm qwen2.5:3b --out outputs/v5_aapl_20210601.json
```

Add `--invoke-qwen` only after checking evidence and mappings; Qwen calls an existing Ollama server on port 11434. `--save-prompt outputs/v5_prompt.txt` saves the prompt for inspection.

**Important issues:**
- FinDKG `subject_id/object_id` may be opaque. Use `--mapping-dir` with `entity_id,name` mapping CSVs for entity-level retrieval. If no ticker is resolved, the program **does not invent focal edges**.
- FinDKG `date` meaning is not verified as public release date; date-scoped results are only graph observations and may leak future information in backtests.
- ReflectKG `filing_date` is not proven equal to SEC publication time; must join to reliable `published_at` with document ID before any claims of leakage-free backtests.
- ReflectKG `changes` are recent **observed evidence** rather than confirmed edge births/deaths; absence of mention never causes removal.
- V4 JSON is embedded as-is; its exact schema has not been inspected. Claims of end-to-end correctness require inspection of `v4_output` and its actual evidence.
- SEC fallback performs keyword overlap for a limited number of relevant chunks; it is not semantic retrieval or LightRAG. No true incremental text-index updates are implemented here.
- `as_of` is day resolution and strictly excludes same-day evidence. Market hours/timestamps require future work.

Run smoke tests: `python -m unittest -v dynamic_kg_rag_v5/test_v5.py` from repo root after unzipping.
