"""Exercise starter for 05_tiny_transformer.

Runs the corresponding lecture demo, then compares a toy pre-norm/post-norm
block by gradient norm after one backward pass.
"""

import argparse
from pathlib import Path
import runpy
import sys

import torch
import torch.nn as nn


LECTURE_SCRIPT = "05_tiny_transformer_encoder_block_torch.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: tiny transformer")
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


class TinyBlock(nn.Module):
    def __init__(self, d_model: int, prenorm: bool):
        super().__init__()
        self.prenorm = prenorm
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        if self.prenorm:
            return x + self.ff(self.ln(x))
        return self.ln(x + self.ff(x))


def extension_prenorm_vs_postnorm():
    torch.manual_seed(1)
    x = torch.randn(8, 16, 32)

    for prenorm in [True, False]:
        block = TinyBlock(32, prenorm=prenorm)
        y = block(x)
        loss = y.pow(2).mean()
        loss.backward()
        total_grad = 0.0
        for p in block.parameters():
            if p.grad is not None:
                total_grad += p.grad.norm().item()
        name = "pre-norm" if prenorm else "post-norm"
        print(f"{name} grad_norm_sum: {total_grad:.6f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_prenorm_vs_postnorm()
