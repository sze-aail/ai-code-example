"""Interactive multi-head attention inspector."""

import argparse
import math

import torch
from torch import nn

from interactive_common import default_output_path, save_or_show, setup_matplotlib


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

    def split_heads(self, x):
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x, causal=False):
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if causal:
            s = x.shape[1]
            mask = torch.triu(torch.ones(s, s, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        out = w @ v
        return out, w


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive multi-head attention")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_data(seed: int, heads: int):
    torch.manual_seed(seed)
    x = torch.randn(1, 4, 8)
    mha = MultiHeadAttention(8, heads)
    return x, mha


def main():
    args = parse_args()
    plt, _, _, CheckButtons, RadioButtons, _ = setup_matplotlib(args.smoke)

    fig = plt.figure(figsize=(11, 5.5))
    ax_heat = fig.add_axes([0.08, 0.24, 0.38, 0.64])
    ax_out = fig.add_axes([0.56, 0.24, 0.32, 0.64])
    ax_heads = fig.add_axes([0.08, 0.05, 0.18, 0.13])
    ax_head_select = fig.add_axes([0.33, 0.05, 0.18, 0.14])
    ax_mask = fig.add_axes([0.58, 0.06, 0.12, 0.12])

    radio_heads = RadioButtons(ax_heads, ["1 head", "2 heads", "4 heads"], active=2)
    radio_head_select = RadioButtons(ax_head_select, ["head 1", "head 2", "head 3", "head 4"], active=0)
    check_mask = CheckButtons(ax_mask, ["causal"], [True])

    def render(_=None):
        heads = int(radio_heads.value_selected.split()[0])
        head_idx = int(radio_head_select.value_selected.split()[-1]) - 1
        head_idx = min(head_idx, heads - 1)
        causal = check_mask.get_status()[0]
        x, mha = make_data(args.seed, heads)
        out, weights = mha(x, causal=causal)
        weights = weights.detach().numpy()[0]
        out = out.detach().numpy()[0]
        out_slice = out[head_idx]

        ax_heat.clear()
        ax_heat.imshow(weights[head_idx], cmap="magma", vmin=0.0, vmax=1.0)
        ax_heat.set_title(f"Attention heatmap | head {head_idx + 1}/{heads}")
        ax_heat.set_xlabel("key token")
        ax_heat.set_ylabel("query token")

        ax_out.clear()
        limit = float(max(abs(out_slice.min()), abs(out_slice.max()), 1e-6))
        ax_out.imshow(out_slice.T, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
        ax_out.set_title(f"Output features for head {head_idx + 1}")
        ax_out.set_xlabel("token")
        ax_out.set_ylabel("feature")
        fig.canvas.draw_idle()

    radio_heads.on_clicked(render)
    radio_head_select.on_clicked(render)
    check_mask.on_clicked(render)
    render()

    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()



