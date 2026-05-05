"""
02_lstm_cell_from_scratch_torch.py

Single LSTM cell implemented by hand with one Linear for all 4 gates.
"""

import argparse
import torch
from torch import nn


class LSTMCellHand(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim + hid_dim, 4 * hid_dim)
        self.hid_dim = hid_dim
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)
        # Forget gate bias trick for easier memory retention.
        with torch.no_grad():
            self.W.bias[: self.hid_dim].fill_(1.0)

    def forward(self, x, state):
        h, c = state
        gates = self.W(torch.cat([x, h], dim=-1))
        f, i, c_tilde, o = gates.chunk(4, dim=-1)
        f = torch.sigmoid(f)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        c_tilde = torch.tanh(c_tilde)
        c_new = f * c + i * c_tilde
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


def parse_args():
    parser = argparse.ArgumentParser(description="Hand-written LSTM cell forward/backward demo")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--in-dim", type=int, default=3)
    parser.add_argument("--hid-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.batch_size = min(args.batch_size, 2)
        args.seq_len = min(args.seq_len, 4)

    torch.manual_seed(args.seed)

    cell = LSTMCellHand(args.in_dim, args.hid_dim)
    x = torch.randn(args.batch_size, args.seq_len, args.in_dim)

    h = torch.zeros(args.batch_size, args.hid_dim)
    c = torch.zeros(args.batch_size, args.hid_dim)

    for t in range(args.seq_len):
        h, c = cell(x[:, t, :], (h, c))

    # Tiny objective to verify gradients flow.
    target = torch.randn_like(h)
    loss = torch.mean((h - target) ** 2)
    loss.backward()

    grad_norm = cell.W.weight.grad.norm().item()
    print(f"Output shape: {tuple(h.shape)}")
    print(f"Loss: {loss.item():.5f}")
    print(f"Weight grad norm: {grad_norm:.5f}")


if __name__ == "__main__":
    main()

