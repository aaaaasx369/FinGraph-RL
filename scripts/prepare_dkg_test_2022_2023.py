#!/usr/bin/env python3

from pathlib import Path
import shutil
import re
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "outputs/finrl_dkg_pit_2022_2023"

SOURCES = {
    "FULL_15": {
        "base": ROOT / "outputs/finrl_dkg_original_full",
        "script": "run_original.py",
        "relations": 15,
        "folder": "full15",
    },
    "FOUR_4": {
        "base": ROOT / "outputs/finrl_dkg_four_native",
        "script": "run_four_native.py",
        "relations": 4,
        "folder": "four4",
    },
}

TRAIN_END = pd.Timestamp("2020-12-31")
VAL_END = pd.Timestamp("2021-12-31")
TEST_END = pd.Timestamp("2023-12-31")


def fail(msg):
    raise RuntimeError("PREPARE STOP: " + str(msg))


def load_dates(path):
    df = pd.read_csv(path)

    required = {"TimeID", "DATE_WK"}
    if not required.issubset(df.columns):
        fail("Unexpected time2id format: " + str(path))

    df["TimeID"] = df["TimeID"].astype(int)
    df["DATE_WK"] = pd.to_datetime(
        df["DATE_WK"]
    ).dt.normalize()

    if df["TimeID"].duplicated().any():
        fail("Duplicated TimeID")

    return dict(zip(df["TimeID"], df["DATE_WK"]))


def assign_split(date):
    if date <= TRAIN_END:
        return "train"

    if date <= VAL_END:
        return "valid"

    if date <= TEST_END:
        return "test"

    fail("Event outside experiment period: " + str(date))


def stage(name, cfg):
    source_run = cfg["base"] / "original_run"

    source_data = (
        source_run / "FinDKG_dataset/FinDKG-full"
    )

    destination = OUT / cfg["folder"]
    dest_run = destination / "original_run"

    if destination.exists():
        fail(
            f"{destination} already exists. "
            "Refusing to overwrite staged experiments."
        )

    for path in [
        source_data,
        source_run / "DKG",
        source_run / cfg["script"],
    ]:
        if not path.exists():
            fail("Missing source: " + str(path))

    date_map = load_dates(
        source_data / "time2id.txt"
    )

    events = {
        "train": [],
        "valid": [],
        "test": [],
    }

    source_counts = {}
    extra_values = set()

    for old_split in ["train", "valid", "test"]:
        path = source_data / f"{old_split}.txt"
        n = 0

        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue

                fields = line.strip().split()

                if len(fields) != 5:
                    fail(
                        f"Unexpected five-column format "
                        f"in {path}: {line!r}"
                    )

                try:
                    s, r, o, tid, extra = map(
                        int, fields
                    )
                except ValueError:
                    fail("Noninteger KG event: " + line)

                if not (
                    0 <= r < cfg["relations"]
                ):
                    fail(
                        f"{name}: relation ID {r} "
                        "outside expected range"
                    )

                if tid not in date_map:
                    fail(f"Unknown TimeID: {tid}")

                split = assign_split(date_map[tid])

                events[split].append(
                    (tid, s, r, o, extra)
                )

                extra_values.add(extra)
                n += 1

        source_counts[old_split] = n

    # Chronological order; preserve five columns.
    for split in events:
        events[split].sort(key=lambda row: row[0])

    all_times = {
        split: {
            row[0] for row in rows
        }
        for split, rows in events.items()
    }

    expected_weeks = {
        "train": 156,
        "valid": 52,
        "test": 53,
    }

    for split, expected in expected_weeks.items():
        actual = len(all_times[split])

        if actual != expected:
            fail(
                f"{name}: {split} expected "
                f"{expected} active weeks, got {actual}"
            )

    # Stage locally. Never touch the existing source.
    new_data = (
        dest_run / "FinDKG_dataset/FinDKG-full"
    )
    new_data.parent.mkdir(
        parents=True, exist_ok=True
    )

    shutil.copytree(
        source_data,
        new_data,
    )

    shutil.copytree(
        source_run / "DKG",
        dest_run / "DKG",
        ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc"
        ),
    )

    for split, rows in events.items():
        path = new_data / f"{split}.txt"

        with path.open("w") as f:
            for tid, s, r, o, extra in rows:
                f.write(
                    f"{s}\t{r}\t{o}\t"
                    f"{tid}\t{extra}\n"
                )

    source_script = (
        source_run / cfg["script"]
    ).read_text()

    # Change only the maximum number of epochs.
    patched, count = re.subn(
        r"(?m)^epoch_times\s*=\s*\d+\s*$",
        "epoch_times = 30",
        source_script,
        count=1,
    )

    if count != 1:
        fail(
            f"{name}: cannot locate epoch_times"
        )

    # Require training to be enabled.
    if not re.search(
        r"(?m)^flag_train\s*=\s*True\b",
        patched,
    ):
        fail(
            f"{name}: training flag is not True"
        )

    # The new output directory starts without
    # old checkpoints or optimizer states.
    trainer = (
        dest_run / "run_pit_2021_2023.py"
    )

    trainer.write_text(
        patched,
        encoding="utf-8",
    )

    manifest = {
        "name": name,
        "relations": cfg["relations"],
        "source": str(source_data),
        "train": "2018-2020",
        "validation": "2021",
        "test": "2022-2023",
        "source_counts": source_counts,
        "new_counts": {
            split: len(rows)
            for split, rows in events.items()
        },
        "new_weeks": {
            split: len(times)
            for split, times in all_times.items()
        },
        "extra_column_values": sorted(extra_values),
        "checkpoint_reused": False,
        "strict_pit_certified": False,
    }

    destination.joinpath(
        "split_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n=== {name} ===")

    for split, rows in events.items():
        dates = sorted(
            date_map[t]
            for t in all_times[split]
        )

        print(
            f"{split.upper():5s} "
            f"events={len(rows):7d} "
            f"weeks={len(dates):3d} "
            f"range={dates[0].date()}.."
            f"{dates[-1].date()}"
        )

    print("Trainer:", trainer)


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, cfg in SOURCES.items():
        stage(name, cfg)

    print("\n=== STAGING COMPLETE ===")
    print("Output:", OUT)
    print("Old experiments: UNCHANGED")
    print("Models trained: NO")


if __name__ == "__main__":
    main()
