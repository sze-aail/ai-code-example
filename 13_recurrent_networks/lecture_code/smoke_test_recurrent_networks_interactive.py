"""Smoke test for interactive Lec13 demos."""

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_bptt_tiny_rnn_parity_interactive.py",
    "02_lstm_cell_from_scratch_interactive.py",
    "03_self_attention_numpy_causal_mask_interactive.py",
    "04_multihead_attention_interactive.py",
    "05_tiny_transformer_encoder_block_interactive.py",
    "06_vae_mnist_interactive.py",
    "07_mini_dcgan_mnist_interactive.py",
    "08_toy_2d_diffusion_interactive.py",
    "09_huggingface_pipelines_interactive.py",
    "10_mini_rag_interactive.py",
]


def main():
    base = Path(__file__).resolve().parent
    failed = []
    for script in SCRIPTS:
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, str(base / script), "--smoke"], check=False)
        if result.returncode != 0:
            failed.append(script)

    print("\n=== Interactive smoke test result ===")
    if failed:
        print("FAILED scripts:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)
    print("All interactive scripts finished (optional online/model demos may skip gracefully).")


if __name__ == "__main__":
    main()

