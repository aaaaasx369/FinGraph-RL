#!/usr/bin/env python3

from pathlib import Path
import ast
import py_compile
import json

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BASE = ROOT / "outputs/finrl_dkg_pit_2022_2023"

CONFIGS = [
    {
        "name": "FULL_15",
        "source": "export_original_kgtransformer.py",
        "old_base": "finrl_dkg_original_full",
        "folder": "full15",
        "output": "export_dkg_2022_2023_full15.py",
        "relations": 15,
    },
    {
        "name": "FOUR_4",
        "source": "export_four_native_kgtransformer.py",
        "old_base": "finrl_dkg_four_native",
        "folder": "four4",
        "output": "export_dkg_2022_2023_four4.py",
        "relations": 4,
    },
]


def replace_exact(text, before, after, label):
    count = text.count(before)
    if count != 1:
        raise RuntimeError(
            f"{label}: expected 1 match, found {count}"
        )
    return text.replace(before, after, 1)


for cfg in CONFIGS:
    source_path = SCRIPTS / cfg["source"]
    output_path = SCRIPTS / cfg["output"]
    run = BASE / cfg["folder"] / "original_run"

    ckpt = (
        run
        / "result/FinDKG-full/DKG/"
        "FinDKG-full_KGTransformer_"
        "overall_best_checkpoint_opt_edge.pt"
    )

    dataset = (
        run / "FinDKG_dataset/FinDKG-full"
    )

    for path in [
        source_path,
        ckpt,
        dataset / "time2id.txt",
        dataset / "train.txt",
        dataset / "valid.txt",
        dataset / "test.txt",
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    text = source_path.read_text(
        encoding="utf-8"
    )

    old = (
        'BASE = ROOT / '
        f'"outputs/{cfg["old_base"]}"'
    )

    new = (
        'BASE = ROOT / '
        '"outputs/finrl_dkg_pit_2022_2023/'
        f'{cfg["folder"]}"'
    )

    text = replace_exact(
        text, old, new, "Experiment directory"
    )

    # The new KG training period is 2018-2020.
    text = replace_exact(
        text,
        'weekly["week_date"] < "2022-01-01"',
        'weekly["week_date"] < "2021-01-01"',
        "PCA training cutoff",
    )

    text = replace_exact(
        text,
        '"pca_fit_period": "2018-2021"',
        '"pca_fit_period": "2018-2020"',
        "PCA manifest",
    )

    text = replace_exact(
        text,
        '"kg_validation_period": "2022"',
        '"kg_validation_period": "2021"',
        "Validation manifest",
    )

    text = replace_exact(
        text,
        '"kg_test_period": "2023"',
        '"kg_test_period": "2022-2023"',
        "Test manifest",
    )

    # Update explanatory notes, without changing
    # the original EmbeddingUpdater replay logic.
    text = text.replace(
        "selected after full 2018-2021 training and "
        "2022 validation",
        "selected after full 2018-2020 training and "
        "2021 validation",
    )

    # The checkpoint must be loaded strictly.
    required = [
        'RUN = BASE / "original_run"',
        'dataset = RUN / "FinDKG_dataset"',
        '"overall_best_checkpoint_opt_edge.pt"',
        'model.load_state_dict(',
        'strict=True',
        'model.node_latest_event_time.zero_()',
        '"strict_pit_certified": False',
    ]

    for item in required:
        if item not in text:
            raise RuntimeError(
                f"{cfg['name']}: missing expected "
                f"replay logic: {item}"
            )

    relation_check = (
        f"if G.num_relations != "
        f"{cfg['relations']}:"
    )

    if relation_check not in text:
        raise RuntimeError(
            "Wrong expected relation count: "
            + relation_check
        )

    ast.parse(
        text,
        filename=str(output_path)
    )

    if output_path.exists():
        existing = output_path.read_text(
            encoding="utf-8"
        )
        if existing != text:
            raise RuntimeError(
                "Refusing to overwrite existing file: "
                + str(output_path)
            )
        print("UNCHANGED:", output_path.name)
    else:
        output_path.write_text(
            text, encoding="utf-8"
        )
        print("CREATED:", output_path.name)

    py_compile.compile(
        str(output_path),
        doraise=True,
    )

    print("  Checkpoint:", ckpt)
    print("  Dataset:", dataset)
    print("  PCA Train: 2018-2020")
    print("  Validation: 2021")
    print("  Test: 2022-2023")

print("\n=== EXPORTERS READY ===")
print("No KG model retraining performed.")
print("Previous experiments unchanged.")
