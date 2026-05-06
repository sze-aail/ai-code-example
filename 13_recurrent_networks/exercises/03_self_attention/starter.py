"""Exercise starter for 03_self_attention.

Runs the corresponding lecture demo, then verifies the first softmax row manually.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "03_self_attention_numpy_causal_mask.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: self-attention")
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


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def extension_first_row_check():
    q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores = q @ q.T / np.sqrt(2.0)
    row0 = softmax(scores[0])
    print("First-row attention weights:", np.round(row0, 4))
    print("Row sum:", row0.sum())


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_first_row_check()
