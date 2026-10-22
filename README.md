# FinGraph-RL

### Temporal Financial Knowledge Graphs and Event-Driven LightRAG for Deep Reinforcement Learning Portfolio Optimization

**A research framework integrating Unified Temporal Knowledge Graphs, Event-Driven Retrieval-Augmented Generation, and Deep Reinforcement Learning for evidence-aware portfolio allocation.**

Built upon [FinRL-X / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading).

---

## 1. Overview

**FinGraph-RL** is a research-oriented extension of FinRL-X that investigates whether time-aware financial knowledge graphs and retrieval-augmented semantic representations can improve deep reinforcement learning (DRL) for portfolio optimization.

Traditional DRL portfolio allocation primarily relies on numerical market observations, such as historical prices, returns, trading volumes, and technical indicators.

However, financial markets are also influenced by company relationships, industry events, regulatory disclosures, and evolving business risks.

FinGraph-RL integrates these heterogeneous information sources into a unified decision-making framework.

The current research focuses on five technology companies:

- Apple (AAPL)
- Amazon (AMZN)
- Alphabet (GOOGL)
- Microsoft (MSFT)
- NVIDIA (NVDA)

The system combines four feature channels:

| Feature Channel | Information Source | Dimensions per Company |
|---|---|---:|
| Market | Historical prices and technical indicators | 106 |
| SEC | SEC filing-derived features | 64 |
| Dynamic KG | Temporal financial knowledge graph features | 149 |
| Event-Driven LightRAG | Structural and semantic retrieval features | 45 |
| **Total (M2)** | **Combined company observation** | **364** |

For five companies, the expected M2 observation dimension is:

\[
5 \times 364 + 5 = 1825
\]

The additional five dimensions represent the portfolio-state variables in the current environment configuration.

The actual observation dimension should be verified against `observation_space.shape`.

---

## 2. Research Objectives

FinGraph-RL investigates three major research questions.

**RQ1 — Temporal Financial Knowledge Representation**

Can unified temporal financial knowledge graphs provide useful relational information beyond conventional market and SEC features?

**RQ2 — Event-Driven Retrieval and Semantic Representations**

Can event-driven graph retrieval and LLM-generated semantic representations provide additional information for portfolio allocation?

**RQ3 — Evidence Quality and Decision Reliability**

How do knowledge graph quality, retrieval accuracy, temporal validity, and evidence grounding affect downstream DRL performance?

The long-term objective is to establish a traceable evaluation pipeline:

**Knowledge Graph Quality → Retrieval Quality → Semantic Representation → Portfolio Decision Quality**

---

## 3. System Architecture

FinGraph-RL extends the original FinRL-X framework with a temporal financial knowledge graph and an Event-Driven LightRAG pipeline.

```text
                 Historical Financial Data
                            |
          +-----------------+-----------------+
          |                 |                 |
      Market Data       SEC Filings      Financial KG
          |                 |                 |
          v                 v                 v
    Market Features    SEC Features   Unified Temporal KG
        106D               64D               |
                                          14,196 Nodes
                                          19 Relations
                                          224,593 Edges
                                               |
                                  +------------+------------+
                                  |                         |
                                  v                         v
                           Dynamic KG Path          Event-Driven Path
                                  |                         |
                           Temporal KG Features      Event Trigger
                                  |                         |
                                 149D                Historical Evidence
                                                            |
                                                    Official LightRAG
                                                            |
                                                  Local / Global Retrieval
                                                            |
                                                  Merge + Deduplication
                                                            |
                                                  Evidence Chunk Selection
                                                            |
                                                       Qwen Summary
                                                            |
                                                        BGE-M3
                                                            |
                                                      1024D Embedding
                                                            |
                                                      Train-only PCA
                                                            |
                                                      16D Semantic
                                                            |
                                                  Structural 28D + Mask
                                                            |
                                                        RAG 45D
                                                            |
                                               Trading-Day Alignment
                                                            |
                     +--------------------------------------+
                     |
                     v
              Feature Fusion
                     |
       Market + SEC + Dynamic KG + RAG
                     |
              364D per Company
                     |
              5 Companies + 5
               Portfolio States
                     |
                     v
             FinRL-X Observation
                  1825D
                     |
              +------+------+
              |      |      |
             PPO    A2C    DDPG
              |      |      |
              +------+------+
                     |
               Portfolio Weights
                     |
               Backtesting
                     |
              Performance Analysis
```

The graph construction, evidence retrieval, semantic encoding, and portfolio learning stages are separated to support modular evaluation and ablation experiments.

---

## 4. Unified Temporal Financial Knowledge Graph

The Unified Temporal Financial Knowledge Graph integrates financial relationships and events from multiple sources.

The currently reported graph contains:

| Component | Count |
|---|---:|
| Entities | 14,196 |
| Relation Types | 19 |
| Temporal Edges | 224,593 |

The graph represents financial entities and their time-dependent relationships.

Rather than directly feeding raw graph nodes and edges into a neural network, FinGraph-RL transforms the graph into fixed-dimensional representations.

Two complementary processing paths are used.

### 4.1 Dynamic KG Channel

The original Dynamic KG processing pipeline converts temporal financial graph information into numerical features.

The current baseline uses:

**149 dimensions per company per trading day.**

This representation is independent of the total number of graph edges.

### 4.2 Event-Driven LightRAG Channel

The Event-Driven LightRAG pipeline constructs historical graph snapshots, retrieves relevant evidence, generates financial summaries, and converts them into numerical representations.

The final RAG representation contains:

- 28 structural dimensions
- 16 semantic PCA dimensions
- 1 semantic validity mask

**Total: 45 dimensions per company.**

---

## 5. Event-Driven LightRAG

### 5.1 Motivation

The initial Historical LightRAG design used monthly snapshots.

However, financial events do not occur exclusively at monthly boundaries.

A fixed monthly retrieval schedule may therefore delay the incorporation of newly available financial information.

FinGraph-RL introduces an Event-Driven Retrieval Scheduling approach.

Event-Driven scheduling determines **when** historical graph snapshots should be updated.

Local and Global retrieval determine **which evidence** should be retrieved from an already constructed graph.

These are complementary stages rather than competing retrieval strategies.

### 5.2 Event Trigger Planning

The Event-Driven Planner considers financial knowledge graph updates and SEC filing events.

In the initial experiment:

| Stage | Count |
|---|---:|
| Recorded Event Jobs | 612 |
| Candidate Event Jobs | 483 |
| Selected Graph Snapshots | 25 |

The 483 candidate jobs represent potential retrieval-update opportunities, not 483 completed graph snapshots.

### 5.3 Pilot Snapshot Selection

For the initial pipeline validation, 25 graph snapshots were selected across five companies.

Each company contributes:

- Four training snapshots from 2018–2020
- One validation snapshot from 2021

The selection prioritizes dates with higher numbers of newly observed KG events while favoring coverage across different months.

| Company | Training Snapshots | Validation Snapshot |
|---|---|---|
| AAPL | 2018-08-06, 2018-09-17, 2019-01-07, 2019-02-04 | 2021-05-03 |
| AMZN | 2018-03-19, 2018-11-19, 2018-12-17, 2020-02-03 | 2021-02-08 |
| GOOGL | 2018-07-23, 2018-09-24, 2019-09-16, 2020-12-21 | 2021-01-11 |
| MSFT | 2018-02-05, 2018-06-11, 2020-08-10, 2020-09-14 | 2021-05-03 |
| NVDA | 2018-04-02, 2018-05-14, 2019-02-04, 2020-09-14 | 2021-12-06 |

These 25 snapshots were selected primarily to validate the end-to-end integration pipeline.

They do not constitute a comprehensive evaluation over all candidate events.

---

## 6. Historical Evidence Selection

For each selected event date, the system retrieves company-relevant historical KG evidence.

Evidence selection considers:

- Target company
- Historical snapshot date
- Temporal availability restrictions
- Configured evidence retrieval budgets

The current LightRAG graph-generation setting allows up to 80 KG events per snapshot.

Availability-time filtering is used to reduce future-information leakage.

However, independent end-to-end point-in-time certification remains future work.

**Important distinction:** SEC filings may trigger snapshot generation, but the current pilot graph-generation configuration uses `--sec-k 0`. Therefore, an SEC-triggered job does not necessarily include SEC document chunks in the constructed LightRAG graph.

---

## 7. Official LightRAG Graph Construction

Selected historical evidence is used to construct a company-specific, time-dependent LightRAG graph and index.

Each graph snapshot contains entities and relationships derived from eligible historical information.

For example, the AAPL snapshot dated August 6, 2018 contains:

- 58 graph nodes
- 57 graph edges

Graph construction occurs **before** Local and Global retrieval.

The graph serves as a retrieval index and evidence source rather than a raw observation directly supplied to DRL.

---

## 8. Hybrid Retrieval: Local and Global

Once the historical LightRAG graph is constructed, the system performs Hybrid Retrieval.

### 8.1 Local Retrieval

Local Retrieval emphasizes relevant entities and their neighboring relationships.

Example: retrieving Apple-related entities associated with smartphone markets, regional markets, and other companies.

### 8.2 Global Retrieval

Global Retrieval emphasizes relationships and broader financial contexts.

Example: identifying business-risk relationships involving Apple and related financial entities.

### 8.3 Evidence Fusion

The system merges and deduplicates Local and Global retrieval results.

It subsequently retrieves source evidence chunks associated with the selected entities and relationships.

Entity-related and relation-related evidence candidates are integrated using a round-robin selection strategy under a context budget.

### 8.4 Actual Retrieval Example

**AAPL — August 6, 2018**

| Retrieval Stage | Entities | Relations |
|---|---:|---:|
| Local Retrieval | 40 | 57 |
| Global Retrieval | 41 | 40 |
| Merge + Deduplication | 55 | 57 |

The constructed snapshot graph contains 58 nodes and 57 edges.

The 55 retrieved entities and 57 retrieved relations represent the deduplicated retrieval result, not a newly constructed graph.

**Evidence Chunk Processing**

| Stage | Count |
|---|---:|
| Entity-related Candidate Chunks | 54 |
| Additional Relation-related Candidate Chunks | 14 |
| Total Candidate Chunks | 68 |
| Selected Context Chunks | 20 |

The final Qwen context contains 55 entities, 57 relations, and 20 selected evidence chunks.

Document chunking occurs during LightRAG indexing. Retrieval subsequently identifies and selects previously indexed source chunks.

---

## 9. Qwen Semantic Representation

After Hybrid Retrieval, the selected evidence context is provided to Qwen for historical financial summary generation.

The summary is intended to capture relevant company relationships, business conditions, and potential financial risks supported by retrieved evidence.

However, text cannot be directly supplied as a standard numerical DRL observation.

Therefore, FinGraph-RL applies semantic encoding and dimensionality reduction.

### 9.1 BGE-M3 Embedding

Qwen-generated summaries are encoded using BGE-M3.

The original semantic representation contains 1024 dimensions.

### 9.2 PCA Dimensionality Reduction

The semantic embeddings are transformed using PCA fitted only on training data.

In the initial experiment:

| Property | Value |
|---|---:|
| Unique Training Embeddings | 20 |
| Numerical Rank | 19 |
| PCA Components | 16 |

The resulting semantic representation is 16-dimensional.

Using training-only PCA reduces the risk of introducing information from the validation period into the fitted feature transformation.

The small number of unique training embeddings remains a limitation requiring additional robustness evaluation.

---

## 10. RAG Structural and Semantic Features

FinGraph-RL generates two complementary RAG feature channels.

### 10.1 Structural Channel — 28D

| Feature Group | Dimensions |
|---|---:|
| Impact, Risk, Confidence | 3 |
| Corresponding Validity Masks | 3 |
| Evidence Count, Relation Count, Diversity | 3 |
| Retrieval Valid, Summary Present, Freshness | 3 |
| Relation Distribution | 16 |
| **Total** | **28** |

Impact, Risk, and Confidence currently remain zero-valued with invalid masks because complete quote-level grounding has not yet been established.

### 10.2 Semantic Channel — 17D

| Component | Dimensions |
|---|---:|
| PCA-reduced Qwen Summary Embedding | 16 |
| Semantic Validity Mask | 1 |
| **Total** | **17** |

Therefore:

**28D Structural + 16D Semantic + 1D Mask = 45D RAG Features**

---

## 11. Trading-Day Feature Alignment

Graph snapshots are aligned with historical trading dates.

For each company and trading day, the system selects the most recent eligible snapshot within the configured validity window.

The pilot allows snapshot features to be carried forward for up to 35 calendar days.

Snapshots cannot be used before their corresponding availability dates.

When no valid snapshot exists, RAG features are handled as missing or zero-valued inputs with associated validity information.

### Pilot Coverage

| Dataset | Valid Company-Days | Total Company-Days | Coverage |
|---|---:|---:|---:|
| Training | 457 | 3,780 | 12.09% |
| Validation | Approximately 113 | 1,260 | 8.97% |

Sparse snapshot coverage is a major limitation of the initial pilot.

It is not yet established how model performance will change when a substantially larger portion of event-driven candidates is processed.

---

## 12. FinRL-X Integration

FinGraph-RL integrates Market, SEC, Dynamic KG, and Event-Driven RAG features into the portfolio observation.

### Model Configurations

| Model | Components | Dimensions per Company |
|---|---|---:|
| M0 | Market + SEC + Dynamic KG | 319 |
| M1 | M0 + Structural RAG | 347 |
| M2 | M1 + Semantic PCA + Mask | 364 |

The portfolio uses five companies.

With five additional portfolio-state variables, the expected M2 observation has 1,825 dimensions.

The system evaluates portfolio allocation using PPO, A2C, and DDPG.

---

## 13. Preliminary Experimental Results

### Experimental Configuration

| Setting | Value |
|---|---|
| Training | 2018–2020 |
| Validation | 2021 |
| Portfolio | Five technology stocks |
| Algorithms | PPO, A2C, DDPG |
| Initial Reported Seed | 42 |
| Training Timesteps | 100,000 |

### Validation Final Multiple

| Algorithm | M0 | M1 | M2 | M2 vs. M0 |
|---|---:|---:|---:|---:|
| PPO | 1.5992 | 1.5380 | **1.6728** | +4.60% |
| A2C | 1.5033 | 1.5212 | **1.9223** | +27.88% |
| DDPG | 1.3103 | 1.4135 | **1.4562** | +11.14% |

In this preliminary single-seed validation, M2 achieves a higher final asset multiple than M0 and M1 across all three evaluated algorithms.

However, these findings do not establish statistical significance or demonstrate that semantic embeddings independently caused the improvement.

The A2C configuration also exhibits higher volatility and maximum drawdown, while DDPG introduces substantial turnover and transaction-cost concerns.

Further multi-seed experiments and controlled ablations are required.

---

## 14. Knowledge Graph Quality and QA Evaluation

An important ongoing research direction is the independent evaluation of knowledge graph quality and retrieval correctness.

Downstream portfolio performance alone cannot establish whether a knowledge graph contains accurate financial relationships or whether RAG retrieves the correct evidence.

The evaluation framework therefore considers multiple levels.

### 14.1 KG Quality

- Structural consistency
- Duplicate and invalid relations
- Temporal validity
- Semantic correctness
- Evidence provenance

### 14.2 Triple-Based Question Answering

Given a knowledge graph triple:

\[
(h,r,t)
\]

Three question-answering tasks are considered:

| Task | Input | Prediction Target |
|---|---|---|
| Tail Prediction | Head + Relation | Tail |
| Relation Prediction | Head + Tail | Relation |
| Head Prediction | Relation + Tail | Head |

These correspond to HR → T, HT → R, and RT → H.

### 14.3 Retrieval and Answer Evaluation

Potential evaluation metrics include:

- QA Exact-Match Accuracy
- Evidence Recall@K
- Hits@K, where ranked candidate predictions are available
- Temporal Validity Rate
- Unsupported Answer Rate
- Error categories for KG, retrieval, and LLM failures

KG-derived QA evaluation measures whether the system can recover recorded graph information.

It does not independently establish the factual accuracy of every KG triple against external financial sources.

A complete evaluation loop should connect:

**KG Quality → Evidence Retrieval → Qwen Answer → QA Metrics → Error Analysis**

The QA benchmark and grounding evaluation remain active development and validation tasks, rather than completed claims of verified accuracy.

---

## 15. Repository Structure

```text
FinGraph-RL/
├── src/                         # Original FinRL-X framework
├── sec_rag_drl/                 # SEC-RAG + DRL integration
├── rag/                         # SEC retrieval and processing
├── kg/                          # Financial KG construction and features
├── unified_event_memory/        # Unified temporal KG and FinRL-X extensions
├── experiments/
│   └── light_rag_dynamic_kg/    # Event-Driven LightRAG experiments
│       └── evaluation/         # KG/RAG QA evaluation
├── scripts/                     # Historical KG and research utilities
├── docs/                        # Research and framework documentation
├── examples/                    # Original FinRL-X examples
├── figs/                        # Framework figures
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

### Important Research Entry Points

| Component | Script |
|---|---|
| Unified Temporal KG | `unified_event_memory/build_unified_kg_v3.py` |
| Event-Driven Planner | `experiments/light_rag_dynamic_kg/plan_event_driven_v3.py` |
| Pilot Snapshot Batch | `experiments/light_rag_dynamic_kg/run_event_m2_batch.py` |
| Official LightRAG | `experiments/light_rag_dynamic_kg/run_official_lightrag.py` |
| Feature Alignment | `experiments/light_rag_dynamic_kg/build_event_m2_aligned.py` |
| Multi-DRL Training | `experiments/light_rag_dynamic_kg/train_finrlx_multidrl.py` |
| KG QA | `experiments/light_rag_dynamic_kg/evaluation/run_kg_qa.py` |
| Official KG QA | `experiments/light_rag_dynamic_kg/evaluation/run_official_kg_qa_v2.py` |
| Integrated Financial QA | `experiments/light_rag_dynamic_kg/evaluation/run_integrated_financial_qa.py` |

These are research entry points, not a verified end-to-end one-command installation procedure.

The complete research environment requires corresponding historical datasets, model dependencies, and locally available generated artifacts.

---

## 16. Reproducibility

This repository contains the research code for FinGraph-RL and the original FinRL-X framework.

Some historical datasets, trained weights, cached graph indices, and generated embeddings may be excluded from Git due to size or reproducibility considerations.

The recommended reproduction sequence is:

1. Prepare historical Market, SEC, and KG datasets.
2. Construct or load the Unified Temporal KG.
3. Generate Event-Driven candidate jobs.
4. Select historical snapshot dates.
5. Build Official LightRAG graph snapshots.
6. Execute Hybrid Retrieval and Qwen summarization.
7. Generate BGE-M3 embeddings and train-only PCA features.
8. Align RAG features with trading dates.
9. Construct FinRL-X M0/M1/M2 observations.
10. Train and evaluate PPO, A2C, and DDPG.
11. Evaluate KG quality, retrieval accuracy, and QA behavior.

Exact runtime requirements and end-to-end reproduction commands should be validated against the corresponding experiment configuration before use.

---

## 17. Research Limitations

Current limitations include:

- Sparse Event-Driven RAG snapshot coverage
- Non-random initial snapshot selection
- Limited unique semantic embeddings for PCA estimation
- Incomplete quote-level grounding for financial scoring
- Preliminary single-seed performance results
- Incomplete independent point-in-time auditing
- Potential effects of missing-data patterns and feature selection
- Insufficient evidence to establish causal performance improvements

The project is an ongoing research implementation rather than a validated production trading system.

---

## 18. Future Work

The next research stages focus on:

**Expanded Event-Driven Coverage**

Evaluate additional candidate event dates and investigate how snapshot coverage influences downstream DRL performance.

**KG Quality Benchmark**

Assess graph consistency, provenance, financial relationship accuracy, and historical temporal validity.

**RAG Question Answering**

Evaluate triple-based QA tasks, evidence retrieval accuracy, and unsupported-answer behavior.

**Controlled Ablation Studies**

Separate the effects of event-driven scheduling, structural retrieval features, and semantic representations.

**Multi-Seed Robustness**

Evaluate return, Sharpe Ratio, volatility, maximum drawdown, turnover, and transaction costs over multiple random seeds.

**Evidence Grounding**

Improve claim-level and quote-level attribution for LLM-generated financial analysis.

---

## 19. Foundation and Acknowledgments

FinGraph-RL is a research extension built upon the open-source FinRL-X / FinRL-Trading framework.

The original FinRL-X repository is maintained by the AI4Finance Foundation:

[AI4Finance-Foundation / FinRL-Trading](https://github.com/AI4Finance-Foundation/FinRL-Trading)

The project also builds upon concepts from:

- Financial Knowledge Graph research
- Temporal Knowledge Graph representation learning
- Retrieval-Augmented Generation
- LightRAG
- Financial LLM reasoning and semantic embeddings
- Deep Reinforcement Learning for portfolio allocation

Original upstream contributions and third-party components remain subject to their respective licenses and attribution requirements.

See the repository [LICENSE](LICENSE) and upstream documentation for applicable terms.

---

## 20. Research Status

**Status: Active Research and Experimental Development**

Current progress includes:

- Unified Temporal Financial KG construction
- Event-Driven snapshot planning
- Initial Official LightRAG integration
- Structural and semantic feature generation
- Trading-day feature alignment
- FinRL-X M0/M1/M2 integration
- Preliminary PPO/A2C/DDPG validation
- Development of KG quality and QA evaluation

**FinGraph-RL aims to investigate whether historically grounded financial graph evidence can provide reliable, measurable improvements to reinforcement-learning-based portfolio decisions.**
