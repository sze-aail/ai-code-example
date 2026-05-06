"""Exercise starter for 04_multihead_attention.

Runs the corresponding lecture demo, then checks that heads=1 matches
single-head attention in a toy setup.
"""

import argparse
from pathlib import Path
import runpy
import sys

import torch


LECTURE_SCRIPT = "04_multihead_attention_torch.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: multi-head attention")
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


def extension_heads_one_check():
    torch.manual_seed(7)
    x = torch.randn(2, 4, 8)  # (batch, seq, d)
    scores = x @ x.transpose(-2, -1) / (x.size(-1) ** 0.5)
    a = torch.softmax(scores, dim=-1)
    single = a @ x

    # For heads=1, split/merge should be identity in this toy case.
    h = 1
    d = x.size(-1)
    xh = x.view(x.size(0), x.size(1), h, d // h).transpose(1, 2)
    scores_h = xh @ xh.transpose(-2, -1) / ((d // h) ** 0.5)
    ah = torch.softmax(scores_h, dim=-1)
    multi = (ah @ xh).transpose(1, 2).contiguous().view_as(x)

    max_diff = (single - multi).abs().max().item()
    print(f"heads=1 consistency max diff: {max_diff:.8f}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_heads_one_check()
