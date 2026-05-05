"""
03_self_attention_numpy_causal_mask.py

Row-wise softmax(QK^T/sqrt(d_k))V with optional causal masking.
"""

import argparse
import numpy as np


def softmax_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix - matrix.max(axis=1, keepdims=True)
    e = np.exp(matrix)
    return e / e.sum(axis=1, keepdims=True)


def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, causal: bool = False):
    scores = (Q @ K.T) / np.sqrt(Q.shape[1])
    if causal:
        mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = scores.copy()
        scores[mask] = -1e9
    weights = softmax_rows(scores)
    out = weights @ V
    return scores, weights, out


def parse_args():
    parser = argparse.ArgumentParser(description="NumPy self-attention demo")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    _ = parse_args()

    Q = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    K = Q.copy()
    V = np.array([[10, 0], [0, 10], [5, 5]], dtype=float)

    scores, weights, out = attention(Q, K, V, causal=False)
    _, weights_c, out_c = attention(Q, K, V, causal=True)

    np.set_printoptions(precision=3, suppress=True)
    print("Scores (unmasked):")
    print(scores)
    print("\nAttention weights (unmasked):")
    print(weights)
    print("\nOutput (unmasked):")
    print(out)

    print("\nAttention weights (causal):")
    print(weights_c)
    print("\nOutput (causal):")
    print(out_c)


if __name__ == "__main__":
    main()

