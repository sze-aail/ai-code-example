"""
05_tiny_transformer_encoder_block_torch.py

Tiny pre-norm Transformer encoder for character next-token prediction.
"""

import argparse
import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        if d_model % heads != 0:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.d_k = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        att = weights @ v
        att = att.transpose(1, 2).contiguous().view(b, s, d)
        return self.out(att)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, mlp_mult: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, heads: int, depth: int, context: int):
        super().__init__()
        self.context = context
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, context, d_model))
        self.blocks = nn.ModuleList([EncoderBlock(d_model, heads) for _ in range(depth)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor):
        b, s = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :s, :]
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def make_dataset(text: str, context: int):
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    xs, ys = [], []
    for i in range(len(data) - context):
        xs.append(data[i : i + context])
        ys.append(data[i + 1 : i + context + 1])

    return torch.stack(xs), torch.stack(ys), stoi, itos


def parse_args():
    parser = argparse.ArgumentParser(description="Tiny Transformer encoder block demo")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--context", type=int, default=32)
    parser.add_argument("--steps", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.steps = min(args.steps, 80)
        args.context = min(args.context, 24)
        args.batch_size = min(args.batch_size, 16)

    torch.manual_seed(args.seed)

    text = (
        "self attention computes token-token relevance. "
        "pre norm transformer blocks are stable for training. "
        "this is a tiny educational corpus. "
    ) * 30

    x_all, y_all, stoi, itos = make_dataset(text, args.context)
    model = TinyTransformer(len(stoi), args.d_model, args.heads, args.depth, args.context)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_all.size(0), (args.batch_size,))
        xb, yb = x_all[idx], y_all[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(f"step={step:4d} loss={loss.item():.4f}")

    # Small sample continuation.
    model.eval()
    prompt = "self attention "
    ctx = torch.tensor([[stoi.get(c, 0) for c in prompt[-args.context :]]], dtype=torch.long)
    with torch.no_grad():
        for _ in range(40):
            logits = model(ctx[:, -args.context :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ctx = torch.cat([ctx, next_token], dim=1)
    generated = "".join(itos[int(i)] for i in ctx[0])
    print("Generated:")
    print(generated)


if __name__ == "__main__":
    main()

