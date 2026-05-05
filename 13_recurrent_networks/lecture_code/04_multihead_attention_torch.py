"""
04_multihead_attention_torch.py

Educational multi-head attention implementation.
"""

import argparse
import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor):
        b, s, _ = x.shape
        x = x.view(b, s, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor):
        b, h, s, d = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(b, s, h * d)

    def forward(self, x: torch.Tensor, causal: bool = False):
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal:
            s = x.shape[1]
            mask = torch.triu(torch.ones(s, s, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        out = self.combine_heads(out)
        return self.w_o(out), weights


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-head attention demo")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.batch_size = min(args.batch_size, 1)
        args.seq_len = min(args.seq_len, 4)

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch_size, args.seq_len, args.d_model)

    mha = MultiHeadAttention(args.d_model, args.heads)
    y, w = mha(x, causal=True)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    print(f"Attn shape:   {tuple(w.shape)}  # (batch, heads, seq, seq)")

    # Sanity check: with one head, shape should still be valid.
    one_head = MultiHeadAttention(args.d_model, 1)
    y1, _ = one_head(x, causal=True)
    print(f"One-head output shape: {tuple(y1.shape)}")


if __name__ == "__main__":
    main()

