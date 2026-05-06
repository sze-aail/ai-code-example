"""Exercise starter for 10_mini_rag.

Runs the corresponding lecture demo, then shows a toy top-k retrieval effect.
"""

import argparse
from pathlib import Path
import runpy
import sys

import numpy as np


LECTURE_SCRIPT = "10_mini_rag_sentence_transformers.py"


def parse_args():
    parser = argparse.ArgumentParser(description="Exercise starter: mini RAG")
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


def extension_topk_effect():
    docs = [
        "A self-attention uses softmax(QK^T/sqrt(dk)).",
        "A VAE loss combines reconstruction and KL.",
        "RAG augments prompts with retrieved context.",
        "Diffusion learns to predict noise.",
    ]
    query = "How does RAG use context?"

    rng = np.random.default_rng(42)
    doc_emb = rng.normal(size=(len(docs), 8))
    q_emb = rng.normal(size=(8,))
    sims = doc_emb @ q_emb

    print(f"Query: {query}")
    for k in [1, 2, 3]:
        top_idx = np.argsort(sims)[-k:][::-1]
        print(f"top-{k} docs:")
        for idx in top_idx:
            print(f"  - {docs[idx]}")


if __name__ == "__main__":
    args = parse_args()
    if not args.skip_lecture:
        run_lecture_script()
    extension_topk_effect()
