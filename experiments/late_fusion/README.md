# Late Fusion Research

This directory documents the historical Late Fusion
experiments developed for FinGraph-RL.

The original executable scripts currently remain in
the repository root to preserve backward compatibility.

## Versions

| Version | Script | Role |
|---|---|---|
| Initial | `late_fusion.py` | Initial fusion prototype |
| Real-data | `late_fusion_real.py` | Financial evidence integration |
| V3 | `late_fusion_v3.py` | Early KG and SEC integration |
| V4 | `late_fusion_v4.py` | Main indexing/query interface |
| V5 | `late_fusion_v5.py` | V4-based extension |
| V6 | `late_fusion_v6.py` | Later fusion pipeline and output schema |
| V8 | `late_fusion_v8_ablation.py` | Ablation experiments |
| V9 | `late_fusion_v9.py` | Later experimental variant |

All paths above refer to the repository root.

## Important Dependencies

- V3 imports `late_fusion_real` and `findkg_retrieval_v2`.
- V5 references `late_fusion_v4.py`.
- V6 invokes V4 and V5 through subprocess.
- `run_late_fusion_pilot.py` executes V4.
- V8 and V9 consume V6-schema output.

Do not relocate these scripts without updating imports,
subprocess arguments, and execution documentation.

## Relationship to Current Research

The current Event-Driven LightRAG research is located in:

`experiments/light_rag_dynamic_kg/`

The Late Fusion implementations are retained as
historical research and comparison components.

## Maintenance

Before relocating scripts:

1. Identify direct and indirect dependencies.
2. Update executable paths and imports.
3. Validate input and output paths.
4. Run regression and smoke tests.
5. Confirm historical reproducibility.
