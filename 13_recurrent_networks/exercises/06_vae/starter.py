"""Exercise starter for 06_vae.

Runs the corresponding lecture demo, then demonstrates beta-VAE loss tradeoff.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "06_vae_mnist_torch.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: VAE")
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


def extension_beta_sweep():
    # Toy decomposition values to illustrate weighting effect.
    recon = np.array([110.0, 108.0, 106.0])
    kl = np.array([4.0, 6.5, 9.5])
    betas = [0.5, 1.0, 4.0]

    print("beta sweep (toy):")
    for beta in betas:
        total = recon + beta * kl
        print(f"beta={beta:.1f} -> mean total loss={total.mean():.3f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_beta_sweep()
