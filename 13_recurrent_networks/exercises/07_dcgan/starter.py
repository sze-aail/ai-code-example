"""Exercise starter for 07_dcgan.

Runs the corresponding lecture demo, then computes a tiny diversity indicator
that can help detect mode collapse.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "07_mini_dcgan_mnist_torch.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: DCGAN")
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


def extension_mode_collapse_proxy():
    rng = np.random.default_rng(0)
    diverse = rng.normal(size=(32, 16))
    collapsed = np.repeat(diverse[:1], repeats=32, axis=0)

    def mean_pairwise_distance(x: np.ndarray) -> float:
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
        tri = d[np.triu_indices_from(d, k=1)]
        return float(tri.mean())

    print(f"diverse batch distance:   {mean_pairwise_distance(diverse):.4f}")
    print(f"collapsed batch distance: {mean_pairwise_distance(collapsed):.4f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_mode_collapse_proxy()
