# FinGraph-RL Experiment Index

This document maps the major research components to their
implementation files.

The current classification is based on the repository structure.
It does not imply that every experiment has been fully reproduced.

## 1. Current Research: Event-Driven LightRAG

| Stage | Script |
|---|---|
| Event Planning | `experiments/light_rag_dynamic_kg/plan_event_driven_v3.py` |
| Official LightRAG | `experiments/light_rag_dynamic_kg/run_official_lightrag.py` |
| Snapshot Batch | `experiments/light_rag_dynamic_kg/run_event_m2_batch.py` |
| Feature Alignment | `experiments/light_rag_dynamic_kg/build_event_m2_aligned.py` |
| FinRL-X Multi-DRL | `experiments/light_rag_dynamic_kg/train_finrlx_multidrl.py` |
| RAG Signal Verification | `experiments/light_rag_dynamic_kg/verify_rag_signals.py` |
| Coverage Audit | `experiments/light_rag_dynamic_kg/audit_coverage.py` |

## 2. Unified Temporal Knowledge Graph

| Component | Script |
|---|---|
| Unified KG Construction | `unified_event_memory/build_unified_kg_v3.py` |
| Event Store | `unified_event_memory/build_event_store.py` |
| Historical DGL | `unified_event_memory/build_asof_dgl.py` |
| KGT Training | `unified_event_memory/train_unified_kgt_edge.py` |
| Dynamic KG DRL | `unified_event_memory/finrl_x/run_dynamic_kg_drl.py` |

## 3. Knowledge Graph and RAG Evaluation

| Evaluation | Script |
|---|---|
| Triple-based KG QA | `experiments/light_rag_dynamic_kg/evaluation/run_kg_qa.py` |
| Official KG QA | `experiments/light_rag_dynamic_kg/evaluation/run_official_kg_qa_v2.py` |
| Integrated Financial QA | `experiments/light_rag_dynamic_kg/evaluation/run_integrated_financial_qa.py` |
| Natural QA | `experiments/light_rag_dynamic_kg/evaluation/run_natural_qa.py` |
| Repeated QA | `experiments/light_rag_dynamic_kg/evaluation/run_repeated_qa_v3.py` |
| QA Diagnostics | `experiments/light_rag_dynamic_kg/evaluation/diagnose_qa.py` |

## 4. DRL Ablation and Comparison

| Experiment | Script |
|---|---|
| M0 / M1 / M2 DRL | `scripts/run_m0_m1_m2_drl.py` |
| RAG/KG Comparison | `experiments/multi_drl_rag_kg_compare.py` |
| Multi-seed Comparison | `experiments/multi_drl_rag_kg_multiseed.py` |
| PPO Behavior Analysis | `experiments/analyze_ppo_behavior.py` |
| RAG Feature Reliance | `experiments/rag_feature_reliance.py` |

## 5. Historical and Supporting Research

The following directories contain earlier methods, comparisons,
utilities, or supporting implementations:

- `scripts/`
- `kg/scripts/`
- `dynamic_kg_rag_server/`
- `dynamic_kg_rag_v5/`
- `experiments/generative_dynamic_kg_rag/`
- `sec_rag_drl/`
- `rag/scripts/`

Historical scripts are preserved for reproducibility and
comparison. They should not be treated as the latest default
pipeline unless explicitly documented.

## 6. Core Framework

The original FinRL-X implementation remains under `src/`.

This includes market data processing, strategies, backtesting,
and trading-related infrastructure.

## 7. Reproducibility Notes

- Verify data availability before executing each experiment.
- Use historically valid data and avoid future leakage.
- Fit feature transformations using training data only.
- Keep generated models and large datasets outside ordinary
  Git tracking where appropriate.
- Treat preliminary single-seed results as exploratory.
- Validate command-line arguments against each script.

## 8. Maintenance Convention

New research scripts should be documented with:

1. Purpose
2. Input dependencies
3. Output artifacts
4. Evaluation period
5. Model configuration
6. Reproduction command
7. Known limitations

Avoid renaming or moving active Python modules without checking
imports and experiment configurations.
