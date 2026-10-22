from pathlib import Path
import runpy

TARGET = (
    Path(__file__).resolve().parent
    / "experiments"
    / "late_fusion"
    / "legacy"
    / "late_fusion_v8_ablation.py"
)

if __name__ == "__main__":
    runpy.run_path(str(TARGET), run_name="__main__")
