# SEC-RAG + DRL Portfolio Allocation

This project implements a research-ready starting point for:

1. **Technical-only DRL**
2. **Technical + SEC-RAG feature fusion**
3. **Technical + RAG soft confidence gate**
4. **Technical + RAG soft gate + selective hard veto**

Supported agents: **PPO, A2C, DDPG, SAC**.

The main research question is not "which RL algorithm is coolest?" but:

> Should SEC-RAG information be learned by the policy, used to constrain the
> policy, or both?

---

## 1. Environment

Python 3.10/3.11 is recommended.

```bash
cd sec_rag_drl
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build appropriate for your server/GPU first,
then install the remaining packages.

---

## 2. Required market data

`data/market.csv`

Minimum:

```csv
date,ticker,close
2018-01-08,AAPL,43.59
2018-01-08,MSFT,88.28
...
```

The code calculates these technical features causally from historical prices:

- 1d / 5d / 20d return
- 20d volatility
- SMA10 and SMA20 price ratios
- RSI14
- MACD and MACD signal

Edit the ticker list and train/test dates in `config.yaml`.

---

## 3. SEC input

Create a CSV such as:

```csv
ticker,filing_date,filing_type,text_path
AAPL,2024-11-01,10-K,sec_text/AAPL_2024_10K.txt
MSFT,2024-07-30,10-K,sec_text/MSFT_2024_10K.txt
```

You may use a `text` column instead of `text_path`.

**Important:** `filing_date` must be the public SEC filing/acceptance date that
was actually available to investors. Do not use fiscal period end as the
availability date.

Generate RAG signals:

```bash
python scripts/build_rag_signals.py \
  --sec-csv data/sec_filings.csv \
  --out data/rag_signals.csv
```

Output schema:

```csv
ticker,available_date,filing_type,impact,risk,confidence,rationale,...
AAPL,2024-11-01,10-K,-0.42,0.71,0.88,"..."
```

The generator uses:

- `BAAI/bge-small-en-v1.5` for dense retrieval
- `Qwen/Qwen2.5-1.5B-Instruct` for evidence-grounded structured signals

The downstream loader performs a **backward as-of join**:

`latest available_date <= trading date`

so future filings are never backfilled into the past.

---

## 4. Fast pipeline smoke test without an LLM

Generate fake market/RAG data:

```bash
python scripts/make_demo_data.py
```

Then reduce `total_timesteps` in `config.yaml` to e.g. `5000` and run:

```bash
python scripts/run_experiments.py \
  --agents PPO \
  --modes technical fusion hybrid_soft hybrid_veto \
  --seeds 42
```

This verifies the mechanics only. Do not report synthetic results.

---

## 5. Formal experiment

Recommended first pass:

```bash
python scripts/run_experiments.py \
  --agents PPO A2C \
  --modes technical fusion hybrid_soft hybrid_veto \
  --seeds 42 123 2026
```

Your old DDPG baseline can be added:

```bash
python scripts/run_experiments.py \
  --agents DDPG \
  --modes technical fusion \
  --seeds 42 123 2026
```

SAC is supported if you later want a fourth continuous-control robustness check.

Results are written to:

```text
outputs/
  all_metrics.csv
  summary_by_seed.csv
  PPO__technical__seed42/
  PPO__fusion__seed42/
  PPO__hybrid_soft__seed42/
  PPO__hybrid_veto__seed42/
  ...
```

Each run contains:

- model
- out-of-sample backtest CSV
- target weights
- metrics
- equity curve

---

## 6. What each mode means

### `technical`

```text
Technical -> RL -> weights
```

No RAG in the observation; no gate.

### `fusion`

```text
[Technical + impact + risk + confidence] -> RL -> weights
```

This tests whether the policy learns useful incremental information from RAG.

### `gate_only`

```text
Technical -> RL -> raw weights
RAG -> soft gate -> final weights
```

RAG constrains the policy but is not in the policy observation.

### `hybrid_soft`

```text
[Technical + RAG] -> RL -> raw weights
RAG -> soft gate -> final weights
```

Soft gate:

\[
g_i = 1 + \alpha C_i I_i - \beta C_i R_i
\]

for sufficiently confident signals.

### `hybrid_veto`

```text
[Technical + RAG] -> RL -> raw weights
        -> soft gate
        -> selective negative veto
        -> final weights
```

Negative veto score:

\[
V_i=C_iR_i\max(-I_i,0)
\]

If it exceeds the threshold, the stock is capped at `veto_max_weight`.
Removed exposure is sent to **cash**, so a subsequent normalization step cannot
accidentally violate the cap.

---

## 7. Permutation / reallocated-weight attribution

For a trained fusion model:

```bash
python scripts/permutation_attribution.py \
  --agent PPO \
  --mode fusion \
  --model outputs/PPO__fusion__seed42/model.zip \
  --repeats 10
```

For hybrid models, the default permutation changes the policy observation only,
while leaving gate inputs untouched. This isolates:

> Did the learned RL policy itself use this feature?

If you want total-system influence, including the gate:

```bash
python scripts/permutation_attribution.py \
  --agent PPO \
  --mode hybrid_veto \
  --model outputs/PPO__hybrid_veto__seed42/model.zip \
  --repeats 10 \
  --permute-gate
```

The reported quantity is:

\[
100\times\frac12\sum_i|w_i^{perm}-w_i^{base}|
\]

averaged over test dates. It is the percentage of capital that must be
reallocated to transform one portfolio into the other.

---

## 8. Metrics

The evaluator reports:

- total return
- annualized return
- annualized volatility
- Sharpe
- Sortino
- maximum drawdown
- daily 5% historical CVaR
- average turnover
- transaction-cost drag
- final portfolio value

For a paper/thesis, compare mean and standard deviation over multiple seeds.

---

## 9. Recommended research ablation

Start with the same data, split, transaction cost and seeds for all models:

| Variant | RAG in state | Soft gate | Hard veto |
|---|---:|---:|---:|
| Technical | No | No | No |
| Fusion | Yes | No | No |
| Hybrid-soft | Yes | Yes | No |
| Hybrid-veto | Yes | Yes | Yes |

Optional `gate_only` gives an additional clean test of "constraint without
learning the RAG features."

Do **not** tune each model on the test period. Choose thresholds/hyperparameters
using a training/validation period, then lock them before the final OOS test.

---

## 10. Before using this for your actual results

This is a solid starting implementation, but you should still audit:

1. exact SEC acceptance/availability timestamps,
2. ticker survivorship/universe construction,
3. corporate actions / adjusted prices,
4. train-validation-test or rolling OOS protocol,
5. transaction cost assumptions,
6. RAG signal stability and malformed-output rate,
7. threshold sensitivity (`confidence_threshold`, `veto_threshold`),
8. multiple random seeds.

The most important anti-leakage property is already explicit in the code:
RAG signals are joined only from filings available on or before each trade date.
