"""
01_bptt_tiny_rnn_parity_numpy.py

Tiny vanilla RNN + manual BPTT on parity task.
"""

import argparse
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_batch(batch_size: int, seq_len: int, rng: np.random.Generator):
    xs = rng.integers(0, 2, size=(batch_size, seq_len, 1)).astype(np.float32)
    ys = (xs.sum(axis=1) % 2).astype(np.float32)
    return xs, ys


def forward(xs: np.ndarray, h0: np.ndarray, params):
    w_xh, w_hh, b_h, w_hy, b_y = params
    hs = [h0]
    for t in range(xs.shape[0]):
        x_t = xs[t].reshape(-1)
        h_t = np.tanh(w_xh @ x_t + w_hh @ hs[-1] + b_h)
        hs.append(h_t)
    y_logit = w_hy @ hs[-1] + b_y
    y_hat = sigmoid(y_logit)
    return y_hat, (hs, y_hat)


def bptt(xs: np.ndarray, y_true: float, cache, params):
    w_xh, w_hh, b_h, w_hy, _ = params
    hs, y_hat = cache

    dlogit = float(y_hat.item() - y_true)

    grad_w_hy = dlogit * hs[-1][None, :]
    grad_b_y = np.array([dlogit], dtype=np.float32)
    d_h = dlogit * w_hy.reshape(-1)

    grad_w_xh = np.zeros_like(w_xh)
    grad_w_hh = np.zeros_like(w_hh)
    grad_b_h = np.zeros_like(b_h)

    for t in range(len(hs) - 1, 0, -1):
        h_t = hs[t]
        h_prev = hs[t - 1]
        x_t = xs[t - 1].reshape(-1)

        dtanh = (1.0 - h_t * h_t) * d_h
        grad_w_xh += np.outer(dtanh, x_t)
        grad_w_hh += np.outer(dtanh, h_prev)
        grad_b_h += dtanh

        d_h = w_hh.T @ dtanh

    return grad_w_xh, grad_w_hh, grad_b_h, grad_w_hy, grad_b_y


def bce_loss(y_hat: np.ndarray, y_true: float) -> float:
    eps = 1e-8
    y = float(y_true)
    yh = float(y_hat.item())
    return float(-(y * np.log(yh + eps) + (1.0 - y) * np.log(1.0 - yh + eps)))


def evaluate(params, seq_len: int, samples: int, seed: int):
    rng = np.random.default_rng(seed)
    ok = 0
    h0 = np.zeros(params[0].shape[0], dtype=np.float32)
    for _ in range(samples):
        xs, ys = generate_batch(1, seq_len, rng)
        y_hat, _ = forward(xs[0], h0, params)
        pred = 1 if float(y_hat.item()) >= 0.5 else 0
        if pred == int(ys[0, 0]):
            ok += 1
    return ok / samples


def parse_args():
    parser = argparse.ArgumentParser(description="Manual BPTT on parity task")
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 120)
        args.seq_len = min(args.seq_len, 5)

    rng = np.random.default_rng(args.seed)

    h = args.hidden_size
    w_xh = rng.normal(0, 0.5, size=(h, 1)).astype(np.float32)
    w_hh = rng.normal(0, 0.5, size=(h, h)).astype(np.float32)
    b_h = np.zeros(h, dtype=np.float32)
    w_hy = rng.normal(0, 0.5, size=(1, h)).astype(np.float32)
    b_y = np.zeros(1, dtype=np.float32)
    params = [w_xh, w_hh, b_h, w_hy, b_y]

    h0 = np.zeros(h, dtype=np.float32)
    for epoch in range(1, args.epochs + 1):
        xs, ys = generate_batch(batch_size=1, seq_len=args.seq_len, rng=rng)
        x_seq = xs[0]
        y = float(ys[0, 0])

        y_hat, cache = forward(x_seq, h0, params)
        loss = bce_loss(y_hat, y)
        grads = bptt(x_seq, y, cache, params)

        for p, g in zip(params, grads):
            p -= args.lr * g

        if epoch % max(1, args.epochs // 6) == 0 or epoch == 1:
            grad_norm = float(np.sqrt(sum((g * g).sum() for g in grads)))
            print(f"epoch={epoch:4d} loss={loss:.4f} grad_norm={grad_norm:.5f}")

    acc = evaluate(params, seq_len=args.seq_len, samples=200, seed=args.seed + 1)
    print(f"Final parity accuracy (seq_len={args.seq_len}): {acc:.3f}")


if __name__ == "__main__":
    main()

