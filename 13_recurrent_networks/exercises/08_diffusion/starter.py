"""Exercise starter for 08_diffusion.

Runs the corresponding lecture demo, then compares schedule summaries for
different numbers of diffusion steps.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "08_toy_2d_diffusion_torch.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: toy diffusion")
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


def extension_compare_T():
    for t_steps in [20, 100, 400]:
        betas = np.linspace(1e-4, 0.02, t_steps)
        alpha_bar = np.cumprod(1.0 - betas)
        final_signal = float(np.sqrt(alpha_bar[-1]))
        print(f"T={t_steps:>3} -> sqrt(alpha_bar_T)={final_signal:.6f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_compare_T()
