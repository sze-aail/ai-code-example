"""Exercise starter for 09_hf_pipeline.

Runs the corresponding lecture demo, then prints candidate multilingual models
students can try as a follow-up.
"""

import argparse
from pathlib import Path
import runpy
import sys


LECTURE_SCRIPT = "09_huggingface_pipelines_demo.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: HF pipelines")
    parser.add_argument("--skip-lecture", action="store_true", help="Skip running the lecture script")
    return parser.parse_args()


def _supports_quick_flag(script_path: Path) -> bool:
    return "--quick" in script_path.read_text(encoding="utf-8", errors="ignore")


def run_lecture_script():
    script_path = Path(__file__).resolve().parents[2] / "lecture_code" / LECTURE_SCRIPT
    argv_backup = sys.argv[:]
    try:
        sys.argv = [str(script_path)]
        if _supports_quick_flag(script_path):
            sys.argv.append("--quick")
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = argv_backup


def extension_model_suggestions():
    suggestions = [
        "distilbert-base-multilingual-cased",
        "Helsinki-NLP/opus-mt-en-hu",
        "google/mt5-small",
    ]
    print("Suggested follow-up models:")
    for name in suggestions:
        print(f"- {name}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_model_suggestions()
