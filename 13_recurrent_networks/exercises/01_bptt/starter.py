"""Exercise starter for 01_bptt.

Runs the corresponding lecture demo, then executes a small extension:
measure how a simple recurrent gradient term decays with sequence length.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "01_bptt_tiny_rnn_parity_numpy.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: BPTT")
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


def extension_gradient_decay():
    lengths = [5, 10, 20, 40, 80]
    recurrent_jacobian = 0.9
    values = [abs(recurrent_jacobian) ** t for t in lengths]
    print("\nExtension - gradient decay vs sequence length")
    for t, v in zip(lengths, values):
        print(f"T={t:>3} -> approx_grad_scale={v:.6f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_gradient_decay()
