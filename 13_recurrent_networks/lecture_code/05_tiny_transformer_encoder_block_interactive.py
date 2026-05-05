"""Interactive CLI version of the tiny Transformer demo."""

import argparse
import math

import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape
        qkv = self.qkv(x).view(b, s, 3, self.heads, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        weights = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.d_k), dim=-1)
        att = weights @ v
        return self.out(att.transpose(1, 2).contiguous().view(b, s, d))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x):
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

    def forward(self, idx):
        x = self.tok_emb(idx) + self.pos_emb[:, : idx.size(1), :]
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


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


def generate(model, stoi, itos, prompt: str, max_new_tokens: int):
    ctx = torch.tensor([[stoi.get(c, 0) for c in prompt[-model.context :]]], dtype=torch.long)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(ctx[:, -model.context :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ctx = torch.cat([ctx, next_token], dim=1)
    return "".join(itos[int(i)] for i in ctx[0])


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive CLI tiny Transformer")
    parser.add_argument("--prompt", default="self attention ")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--sample-len", type=int, default=50)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick or args.smoke:
        args.steps = min(args.steps, 60)
        args.sample_len = min(args.sample_len, 30)

    torch.manual_seed(args.seed)
    text = (
        "self attention computes token-token relevance. "
        "pre norm transformer blocks are stable for training. "
        "this is a tiny educational corpus. "
    ) * 25
    context = 24
    x_all, y_all, stoi, itos = make_dataset(text, context)
    model = TinyTransformer(len(stoi), 48, 4, 2, context)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x_all.size(0), (16,))
        xb, yb = x_all[idx], y_all[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step % max(1, args.steps // 3) == 0 or step == 1:
            print(f"step={step:3d} loss={loss.item():.4f}")

    if args.smoke:
        print(generate(model, stoi, itos, args.prompt, args.sample_len))
        return

    print("Interactive mode. Empty line exits.")
    while True:
        prompt = input("prompt> ").strip()
        if not prompt:
            break
        print(generate(model, stoi, itos, prompt, args.sample_len))


if __name__ == "__main__":
    main()

