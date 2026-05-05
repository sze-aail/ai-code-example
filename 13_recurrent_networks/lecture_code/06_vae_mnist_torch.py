"""
06_vae_mnist_torch.py

Minimal VAE on MNIST: BCE reconstruction + beta * KL.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


def loss_fn(x_hat, x, mu, log_var, beta: float):
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + beta * kl, bce, kl


def parse_args():
    parser = argparse.ArgumentParser(description="VAE MNIST demo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.max_samples = min(args.max_samples, 512)
        args.batch_size = min(args.batch_size, 64)

    torch.manual_seed(args.seed)

    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("Missing optional dependency: torchvision. Install it to run MNIST VAE.")
        return

    transform = transforms.ToTensor()
    ds = datasets.MNIST(root=".data", train=True, transform=transform, download=True)
    if args.max_samples > 0:
        ds = torch.utils.data.Subset(ds, range(min(args.max_samples, len(ds))))

    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    model = VAE(args.latent_dim)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, _ in loader:
            x = x.view(x.size(0), -1)
            x_hat, mu, log_var = model(x)
            loss, bce, kl = loss_fn(x_hat, x, mu, log_var, args.beta)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg = total_loss / len(loader.dataset)
        print(f"epoch={epoch} avg_loss={avg:.4f} (beta={args.beta})")

    model.eval()
    with torch.no_grad():
        x, _ = next(iter(loader))
        x_flat = x.view(x.size(0), -1)
        x_hat, _, _ = model(x_flat)

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "06_vae_reconstruction.png"

    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i in range(8):
        axes[0, i].imshow(x[i, 0].numpy(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i].view(28, 28).numpy(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("real", fontsize=8)
    axes[1, 0].set_title("recon", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()

