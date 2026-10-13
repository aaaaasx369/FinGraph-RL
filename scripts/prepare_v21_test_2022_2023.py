#!/usr/bin/env python3

from pathlib import Path
import ast
import py_compile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_v21_dynamic_ablation.py"
TARGET = ROOT / "scripts/run_v21_test_2022_2023.py"

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")

old = '''    "train": ("2018-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2023-12-31"),'''

new = '''    "train": ("2018-01-01", "2020-12-31"),
    "val": ("2021-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2023-12-31"),'''

if text.count(old) != 1:
    raise RuntimeError(
        "Cannot uniquely locate V21 date configuration. "
        "Original file unchanged."
    )

text = text.replace(old, new, 1)

text = text.replace(
    "Train: 2018-2021",
    "Train: 2018-2020",
    1,
).replace(
    "Val:   2022",
    "Val:   2021",
    1,
).replace(
    "Test:  2023",
    "Test:  2022-2023",
    1,
)

ast.parse(text, filename=str(TARGET))

if TARGET.exists():
    if TARGET.read_text(encoding="utf-8") != text:
        raise RuntimeError(
            f"Refusing to overwrite: {TARGET}"
        )
    print("UNCHANGED:", TARGET)
else:
    TARGET.write_text(text, encoding="utf-8")
    print("CREATED:", TARGET)

py_compile.compile(str(TARGET), doraise=True)

print("\n=== DATE CONFIGURATION ===")

for line in text.splitlines():
    if '"train": (' in line:
        print(line)
    elif '"val": (' in line:
        print(line)
    elif '"test": (' in line:
        print(line)

print("\nDRL training executed: NO")
print("Original V21 unchanged: YES")
print("Full timing audit: PENDING")
