"""Exercise starter for 02_lstm_cell.

Runs the corresponding lecture demo, then compares forget-gate bias values.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "02_lstm_cell_from_scratch_torch.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: LSTM cell")
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


def extension_forget_bias_demo():
    steps = 10
    c0 = 1.0
    for bias in [0.0, 1.0]:
        # Approximate forget gate activation as sigmoid(bias) with zero input.
        f = 1.0 / (1.0 + np.exp(-bias))
        c_t = c0 * (f ** steps)
        print(f"forget_bias={bias:.1f}, f={f:.3f}, c_after_{steps}={c_t:.4f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_forget_bias_demo()
