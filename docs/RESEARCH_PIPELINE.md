# FinGraph-RL Research Pipeline

**Unified Temporal Financial Knowledge Graph × Event-Driven LightRAG × FinRL-X**

This document describes the research pipeline used to integrate temporal financial knowledge graphs, event-driven retrieval-augmented generation, semantic representations, and deep reinforcement learning for portfolio optimization.

The current implementation is an experimental research system. Reported numerical results reflect the initial pilot configuration and require further validation.

---

## 1. Research Overview

FinGraph-RL extends FinRL-X by integrating four complementary information sources:

1. **Market Data:** Historical stock prices and technical indicators.
2. **SEC Filings:** Historical corporate disclosure features.
3. **Dynamic Financial Knowledge Graph:** Time-dependent financial entities and relationships.
4. **Event-Driven LightRAG:** Structural and semantic representations constructed from historically eligible retrieved evidence.

The portfolio consists of five companies:

| Ticker | Company |
|---|---|
| AAPL | Apple |
| AMZN | Amazon |
| GOOGL | Alphabet |
| MSFT | Microsoft |
| NVDA | NVIDIA |

The research investigates whether financial graph information and evidence-grounded semantic representations can provide additional information for deep reinforcement learning portfolio decisions.

---

## 2. End-to-End System Architecture

The pipeline follows this sequence:

**Historical Financial Data → Unified Temporal KG → Event-Driven Planning → Historical Evidence Selection → Official LightRAG → Hybrid Retrieval → Qwen → BGE-M3 → PCA16 → FinRL-X → PPO/A2C/DDPG**

### Architecture Diagram

```text
                       HISTORICAL FINANCIAL DATA
                                  |
             +--------------------+--------------------+
             |                    |                    |
        MARKET DATA           SEC FILINGS        FINANCIAL KG
             |                    |                    |
             v                    v                    v
       MARKET 106D             SEC 64D         UNIFIED TEMPORAL KG
                                                       |
                                         +-------------+-------------+
                                         |                           |
                                         v                           v
                                    DYNAMIC KG                EVENT-DRIVEN
                                      149D                      PLANNER
                                                                     |
                                                                     v
                                                          HISTORICAL EVIDENCE
                                                                     |
                                                                     v
                                                            OFFICIAL LIGHTRAG
                                                                     |
                                                                     v
                                                          LOCAL / GLOBAL SEARCH
                                                                     |
                                                                     v
                                                           MERGE + DEDUPLICATE
                                                                     |
                                                                     v
                                                            RETRIEVED CHUNKS
                                                                     |
                                                                     v
                                                               QWEN SUMMARY
                                                                     |
                                                                     v
                                                              BGE-M3 1024D
                                                                     |
                                                                     v
                                                              TRAIN-ONLY PCA
                                                                     |
                                                                     v
                                                              SEMANTIC 16D
                                                                     |
                                                                     v
                                                       STRUCTURAL 28D + MASK 1D
                                                                     |
                                                                     v
                                                                RAG 45D
                                                                     |
                                                                     v
                                                          TRADING-DAY ALIGNMENT
                                                                     |
             +--------------------+--------------------+-------------+
                                  |
                                  v
                            FEATURE FUSION
                                  |
                                  v
                             M0 / M1 / M2
                                  |
                                  v
                            FINRL-X ENVIRONMENT
                                  |
                                  v
                             PPO / A2C / DDPG
                                  |
                                  v
                           PORTFOLIO EVALUATION
```

The Event-Driven Planner determines when graph snapshots should be created or updated. Local and Global retrieval determine which information is selected from the resulting LightRAG index.

---

## 3. Unified Temporal Financial Knowledge Graph

The initial reported Unified Temporal Financial Knowledge Graph contains:

| Property | Value |
|---|---:|
| Entities | 14,196 |
| Relation Types | 19 |
| Temporal Edges | 224,593 |

The graph represents financial entities, company relationships, and time-dependent financial events.

Its primary purposes are to:

- Represent evolving financial relationships.
- Supply numerical features for the Dynamic KG channel.
- Identify graph-change events for retrieval scheduling.
- Provide historical evidence for LightRAG construction.
- Support knowledge graph quality and QA evaluation.

### 3.1 Dynamic KG Channel

The Dynamic KG channel converts temporal relational information into numerical features.

The current M0 baseline uses **149 Dynamic KG dimensions per company**.

### 3.2 Point-in-Time Restrictions

Historical graph evidence is subject to availability-time restrictions.

For a historical decision date, the intended rule is to use only information available by that date.

This is necessary to reduce look-ahead bias. Full end-to-end point-in-time correctness requires additional auditing.

**Relevant code:** `unified_event_memory/`

---

## 4. Event-Driven Snapshot Planning

The Event-Driven Planner identifies candidate dates on which new financial information may justify updating the RAG graph.

The initial planner considers KG event changes and SEC filing events.

### 4.1 Candidate Statistics

| Stage | Count |
|---|---:|
| Recorded Event Jobs | 612 |
| Candidate Jobs | 483 |
| Selected Pilot Snapshots | 25 |

The 483 candidates represent possible snapshot jobs, not 483 completed LightRAG graphs.

### 4.2 Pilot Selection

The initial pilot selects five snapshots per company:

- Four from the training period, 2018–2020.
- One from the validation period, 2021.

Candidates are prioritized using newly observed KG-event counts while favoring coverage across different months.

### 4.3 Selected Dates

| Company | Training Snapshots | Validation Snapshot |
|---|---|---|
| AAPL | 2018-08-06, 2018-09-17, 2019-01-07, 2019-02-04 | 2021-05-03 |
| AMZN | 2018-03-19, 2018-11-19, 2018-12-17, 2020-02-03 | 2021-02-08 |
| GOOGL | 2018-07-23, 2018-09-24, 2019-09-16, 2020-12-21 | 2021-01-11 |
| MSFT | 2018-02-05, 2018-06-11, 2020-08-10, 2020-09-14 | 2021-05-03 |
| NVDA | 2018-04-02, 2018-05-14, 2019-02-04, 2020-09-14 | 2021-12-06 |

The 25-snapshot selection was intended to validate the integrated pipeline. It is not a full-scale evaluation over all event candidates.

**Relevant code:** `experiments/light_rag_dynamic_kg/plan_event_driven_v3.py`

---

## 5. Historical Evidence Selection

For each selected event date, the system retrieves eligible historical KG evidence for the target company.

### Processing Steps

1. Identify the target company and snapshot date.
2. Retrieve relevant historical KG events.
3. Apply availability-time filtering.
4. Restrict the candidate set to the configured budget.
5. Prepare historical evidence for LightRAG indexing.

The pilot uses a budget of up to **80 KG events per snapshot**.

The pilot batch uses `--sec-k 0`. Therefore, SEC events can act as triggers, but SEC document chunks are not directly selected for indexing in this configuration.

A snapshot triggered by an SEC filing is not necessarily a snapshot containing SEC filing text.

---

## 6. Official LightRAG Graph Construction

The selected evidence is provided to the Official LightRAG implementation for graph construction and indexing.

The graph contains entities and relationships extracted or assembled from historically eligible information.

### Example: AAPL, August 6, 2018

| Item | Value |
|---|---:|
| KG Event Budget | 80 |
| Constructed Nodes | 58 |
| Constructed Edges | 57 |

The constructed graph is the index used for subsequent retrieval.

It should not be confused with the merged Local/Global retrieval results.

**Relevant code:** `experiments/light_rag_dynamic_kg/run_official_lightrag.py`

---

## 7. Hybrid Retrieval

Hybrid Retrieval combines complementary entity-centric and relation-centric retrieval strategies.

### 7.1 Local Retrieval

Local retrieval emphasizes target entities and their neighboring relationships.

### 7.2 Global Retrieval

Global retrieval emphasizes broader relation patterns and contextual connections.

### 7.3 Merge and Deduplication

Local and Global results are merged, and duplicate entities and relations are removed.

For the AAPL snapshot dated August 6, 2018:

| Retrieval Stage | Entities | Relations |
|---|---:|---:|
| Local | 40 | 57 |
| Global | 41 | 40 |
| Merged | 55 | 57 |

The merged result represents the selected retrieval context, not a newly constructed graph.

### 7.4 Evidence Chunk Retrieval

The system retrieves source chunks associated with selected entities and relations.

| Stage | Count |
|---|---:|
| Entity-related Candidate Chunks | 54 |
| Additional Relation-related Chunks | 14 |
| Total Candidate Chunks | 68 |
| Selected Context Chunks | 20 |

Chunks are created during LightRAG indexing. Retrieval selects from those indexed chunks.

Round-robin selection and context-budget constraints determine which evidence chunks enter the final prompt.

The example final context contains:

- 55 entities
- 57 relations
- 20 evidence chunks

---

## 8. Qwen Financial Summarization

The retrieved historical evidence is provided to Qwen to generate a company-specific financial summary.

The purpose is to convert relational evidence into a semantic description that can subsequently be encoded as numerical features.

The summary may cover relevant business relationships, financial conditions, and risks supported by the retrieved context.

However, generating a summary does not by itself establish factual correctness or quote-level grounding.

The research therefore includes separate retrieval and grounding evaluation tasks.

---

## 9. Semantic Embedding and PCA

### 9.1 BGE-M3

Qwen-generated summaries are encoded with BGE-M3 into 1,024-dimensional vectors.

### 9.2 Training-Only PCA

PCA is fitted using training embeddings only.

The initial pilot contains:

| Property | Value |
|---|---:|
| Unique Training Embeddings | 20 |
| Numerical Rank | 19 |
| PCA Components | 16 |

The transformation produces **16-dimensional semantic features**.

The training-fitted PCA model is subsequently applied to validation data without refitting on validation observations.

---

## 10. Structural and Semantic RAG Features

The final RAG channel contains 45 dimensions per company.

### 10.1 Structural Features — 28D

| Feature Category | Dimensions |
|---|---:|
| Impact, Risk, Confidence | 3 |
| Corresponding Validity Masks | 3 |
| Evidence Count, Relation Count, Diversity | 3 |
| Retrieval Valid, Summary Present, Freshness | 3 |
| Relation Distribution | 16 |
| **Total** | **28** |

Impact, risk, and confidence fields currently remain zero-valued with invalid masks where quote-level grounding has not been established.

### 10.2 Semantic Features — 17D

| Component | Dimensions |
|---|---:|
| PCA Embedding | 16 |
| Semantic Validity Mask | 1 |
| **Total** | **17** |

Thus, the total RAG representation is:

**28D Structural + 16D Semantic + 1D Validity Mask = 45D**

---

## 11. Historical Trading-Day Alignment

Each generated graph snapshot is aligned with historical trading dates.

For a company and trading day, the system uses the most recent eligible snapshot satisfying the historical availability rule.

The pilot permits snapshot features to be carried forward for up to **35 calendar days**.

When no eligible snapshot exists, the corresponding RAG features are represented as missing or zero-valued values with validity indicators.

### Pilot Coverage

| Split | Valid Company-Days | Total Company-Days | Coverage |
|---|---:|---:|---:|
| Training | 457 | 3,780 | 12.09% |
| Validation | Approximately 113 | 1,260 | 8.97% |

Sparse snapshot coverage is an important limitation of the initial experiment.

**Relevant code:** `experiments/light_rag_dynamic_kg/build_event_m2_aligned.py`

---

## 12. FinRL-X Feature Fusion

Three model configurations are compared.

| Model | Feature Composition | Dimensions per Company |
|---|---|---:|
| M0 | Market + SEC + Dynamic KG | 319 |
| M1 | M0 + Structural RAG | 347 |
| M2 | M1 + Semantic PCA + Validity Mask | 364 |

The component dimensions are:

| Component | Dimensions |
|---|---:|
| Market | 106 |
| SEC | 64 |
| Dynamic KG | 149 |
| Structural RAG | 28 |
| Semantic PCA | 16 |
| Semantic Validity Mask | 1 |
| **M2 Total** | **364** |

For five stocks and five additional portfolio-state dimensions:

**5 × 364 + 5 = 1,825 dimensions**

This is the expected observation size for the reported configuration. The implemented environment must be checked to verify its actual observation shape.

---

## 13. Deep Reinforcement Learning Experiments

The current pilot evaluates:

- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)
- Deep Deterministic Policy Gradient (DDPG)

### Experimental Settings

| Parameter | Value |
|---|---|
| Portfolio | AAPL, AMZN, GOOGL, MSFT, NVDA |
| Training Period | 2018–2020 |
| Validation Period | 2021 |
| Training Timesteps | 100,000 |
| Initial Reported Seed | 42 |

### Preliminary Validation Results

| Algorithm | M0 Final Multiple | M1 Final Multiple | M2 Final Multiple |
|---|---:|---:|---:|
| PPO | 1.5992 | 1.5380 | 1.6728 |
| A2C | 1.5033 | 1.5212 | 1.9223 |
| DDPG | 1.3103 | 1.4135 | 1.4562 |

These preliminary single-seed results show higher final asset multiples for M2 in the reported validation runs.

They do not establish statistical significance or the independent causal contribution of semantic features.

Multi-seed evaluation, missingness controls, and additional ablations remain necessary.

**Relevant code:** `experiments/light_rag_dynamic_kg/train_finrlx_multidrl.py`

---

## 14. Knowledge Graph and RAG Evaluation

The research evaluates not only portfolio performance, but also the quality of intermediate knowledge and retrieval stages.

### 14.1 KG Quality

Relevant evaluation dimensions include:

- Entity and relation consistency
- Duplicate or invalid relationships
- Temporal availability
- Financial evidence provenance
- Factual support

### 14.2 Triple-Based QA

Given a KG triple `(head, relation, tail)`, three QA tasks are considered.

| Task | Given | Prediction |
|---|---|---|
| HR → T | Head + Relation | Tail |
| HT → R | Head + Tail | Relation |
| RT → H | Relation + Tail | Head |

These tasks test the ability to recover recorded graph relationships.

### 14.3 Retrieval and Answer Quality

Evaluation may include:

- Exact-match QA accuracy
- Evidence Recall@K
- Temporal validity rate
- Unsupported-answer rate
- Error classification by pipeline stage

KG-based QA measures agreement with recorded graph facts; independent external verification is required to establish whether those facts are objectively correct.

### Evaluation Scripts

| Component | Script |
|---|---|
| KG QA | `experiments/light_rag_dynamic_kg/evaluation/run_kg_qa.py` |
| Official LightRAG QA | `experiments/light_rag_dynamic_kg/evaluation/run_official_kg_qa_v2.py` |
| Integrated Financial QA | `experiments/light_rag_dynamic_kg/evaluation/run_integrated_financial_qa.py` |
| Natural QA | `experiments/light_rag_dynamic_kg/evaluation/run_natural_qa.py` |
| Repeated QA | `experiments/light_rag_dynamic_kg/evaluation/run_repeated_qa_v3.py` |

These evaluation components remain under active development and should not be interpreted as evidence that all QA benchmarks have already passed.

---

## 15. Reproducibility Workflow

The intended reproduction sequence is:

1. Prepare historical financial datasets.
2. Construct or load the Unified Temporal KG.
3. Generate Event-Driven candidate jobs.
4. Select historical snapshot dates.
5. Apply availability-time evidence filtering.
6. Construct the Official LightRAG index.
7. Execute Local and Global retrieval.
8. Retrieve and select evidence chunks.
9. Generate Qwen financial summaries.
10. Encode summaries with BGE-M3.
11. Fit PCA using training embeddings only.
12. Construct the 45D RAG feature channel.
13. Align snapshot features with trading dates.
14. Build FinRL-X M0/M1/M2 observations.
15. Train and validate PPO, A2C, and DDPG.
16. Evaluate KG quality, retrieval correctness, and downstream performance.

Exact commands, model requirements, and data preparation depend on the relevant experiment configuration.

The repository does not yet provide a fully validated one-command reproduction workflow for all experiments.

---

## 16. Limitations

The initial research pipeline has several limitations:

- Sparse RAG snapshot coverage.
- Pilot snapshot selection rather than exhaustive event processing.
- Limited unique training summaries for semantic PCA.
- Incomplete quote-level grounding of some financial scoring fields.
- Preliminary single-seed DRL comparisons.
- Incomplete independent point-in-time auditing.
- Potential missing-data and snapshot-selection effects.
- Need for further KG and QA quality evaluation.
- Need for controlled ablations to establish the independent effect of semantic features.

---

## 17. Future Research

Planned research directions include:

**Expanded Event Coverage:** Evaluate additional Event-Driven candidate snapshots and their effects on feature coverage.

**KG Quality Benchmarking:** Assess graph consistency, temporal validity, provenance, and factual correctness.

**QA and Evidence Evaluation:** Measure triple-based QA accuracy, evidence recall, and unsupported-answer behavior.

**Controlled Ablations:** Isolate the contributions of event-driven scheduling, structural retrieval features, and semantic embeddings.

**Multi-Seed DRL Evaluation:** Compare performance across random seeds using return, Sharpe ratio, maximum drawdown, volatility, turnover, and transaction costs.

**Grounded Financial Summarization:** Improve the traceability of generated claims to historical evidence.

---

## 18. Research Status

**Status: Active Research**

Completed or initially integrated components include:

- Unified Temporal KG construction
- Event-Driven candidate planning
- Pilot LightRAG graph construction
- Local and Global retrieval
- Qwen summary generation
- BGE-M3 and PCA semantic representation
- Structural and semantic feature integration
- Trading-day alignment
- Preliminary M0/M1/M2 DRL validation

Knowledge graph evaluation, QA benchmarking, broader event coverage, and robust multi-seed comparisons remain active research tasks.

FinGraph-RL is an experimental research framework, not a production trading or investment recommendation system.
