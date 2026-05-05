"""
08_toy_2d_diffusion_torch.py

Toy DDPM-like diffusion on 2D spiral data.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


def make_two_spirals(n: int = 2000, noise: float = 0.05, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    n2 = n // 2
    theta = torch.linspace(0, 4 * torch.pi, n2)
    r = torch.linspace(0.1, 1.0, n2)
    x1 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x2 = torch.stack([-r * torch.cos(theta), -r * torch.sin(theta)], dim=1)
    x = torch.cat([x1, x2], dim=0)
    x = x + noise * torch.randn_like(x, generator=g)
    return x


class TimeMLP(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x_t, t_norm):
        return self.net(torch.cat([x_t, t_norm], dim=-1))


def parse_args():
    parser = argparse.ArgumentParser(description="Toy 2D diffusion demo")
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.steps = min(args.steps, 250)
        args.timesteps = min(args.timesteps, 60)

    torch.manual_seed(args.seed)
    x0 = make_two_spirals(n=3000, seed=args.seed)

    T = args.timesteps
    beta = torch.linspace(1e-4, 2e-2, T)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)

    model = TimeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x0.size(0), (args.batch_size,))
        x = x0[idx]
        t = torch.randint(1, T + 1, (args.batch_size,))
        eps = torch.randn_like(x)

        a_bar_t = alphabar[t - 1].unsqueeze(1)
        x_t = torch.sqrt(a_bar_t) * x + torch.sqrt(1.0 - a_bar_t) * eps
        t_norm = (t.float() / T).unsqueeze(1)

        eps_pred = model(x_t, t_norm)
        loss = torch.mean((eps - eps_pred) ** 2)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(f"step={step:4d} loss={loss.item():.4f}")

    # Reverse sampling.
    model.eval()
    x = torch.randn(1200, 2)
    with torch.no_grad():
        for t in range(T, 0, -1):
            a_t = alpha[t - 1]
            a_bar_t = alphabar[t - 1]
            t_norm = torch.full((x.size(0), 1), t / T)
            eps_pred = model(x, t_norm)

            x = (1.0 / torch.sqrt(a_t)) * (x - ((1.0 - a_t) / torch.sqrt(1.0 - a_bar_t)) * eps_pred)
            if t > 1:
                x = x + torch.sqrt(beta[t - 1]) * torch.randn_like(x)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "08_toy_diffusion_samples.png"

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].scatter(x0[:, 0].numpy(), x0[:, 1].numpy(), s=2, alpha=0.5)
    axes[0].set_title("Training data")
    axes[1].scatter(x[:, 0].numpy(), x[:, 1].numpy(), s=2, alpha=0.5, color="tab:orange")
    axes[1].set_title("Generated samples")
    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()

