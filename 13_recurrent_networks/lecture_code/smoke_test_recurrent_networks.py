"""
smoke_test_recurrent_networks.py

Runs all Lec13 scripts in quick mode.
"""

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "01_bptt_tiny_rnn_parity_numpy.py",
    "02_lstm_cell_from_scratch_torch.py",
    "03_self_attention_numpy_causal_mask.py",
    "04_multihead_attention_torch.py",
    "05_tiny_transformer_encoder_block_torch.py",
    "06_vae_mnist_torch.py",
    "07_mini_dcgan_mnist_torch.py",
    "08_toy_2d_diffusion_torch.py",
    "09_huggingface_pipelines_demo.py",
    "10_mini_rag_sentence_transformers.py",
]


def main():
    base = Path(__file__).resolve().parent
    failed = []

    for script in SCRIPTS:
        path = base / script
        print(f"\n=== Running {script} ===")
        result = subprocess.run([sys.executable, str(path), "--quick"], check=False)
        if result.returncode != 0:
            failed.append(script)

    print("\n=== Smoke test result ===")
    if failed:
        print("FAILED scripts:")
        for name in failed:
            print(f"- {name}")
        raise SystemExit(1)

    print("All scripts finished (some may skip optional dependencies gracefully).")


if __name__ == "__main__":
    main()

