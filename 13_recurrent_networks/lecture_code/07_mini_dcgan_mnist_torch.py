"""
07_mini_dcgan_mnist_torch.py

Mini DCGAN on MNIST with non-saturating generator objective.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 28 * 28),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(z.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def parse_args():
    parser = argparse.ArgumentParser(description="Mini DCGAN on MNIST")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def to_minus_one_one(x):
    return x * 2.0 - 1.0


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
        print("Missing optional dependency: torchvision. Install it to run DCGAN MNIST.")
        return

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(to_minus_one_one)])
    ds = datasets.MNIST(root=".data", train=True, transform=transform, download=True)
    if args.max_samples > 0:
        ds = torch.utils.data.Subset(ds, range(min(args.max_samples, len(ds))))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    G = Generator(args.z_dim)
    D = Discriminator()

    bce = nn.BCEWithLogitsLoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(1, args.epochs + 1):
        for real, _ in loader:
            b = real.size(0)
            real_label = torch.ones(b, 1)
            fake_label = torch.zeros(b, 1)

            # D step
            z = torch.randn(b, args.z_dim)
            fake_detached = G(z).detach()
            loss_d = bce(D(real), real_label) + bce(D(fake_detached), fake_label)
            opt_D.zero_grad()
            loss_d.backward()
            opt_D.step()

            # G step (non-saturating)
            z = torch.randn(b, args.z_dim)
            fake = G(z)
            loss_g = bce(D(fake), real_label)
            opt_G.zero_grad()
            loss_g.backward()
            opt_G.step()

        print(f"epoch={epoch} loss_D={loss_d.item():.4f} loss_G={loss_g.item():.4f}")

    with torch.no_grad():
        samples = G(torch.randn(16, args.z_dim)).cpu()

    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "07_dcgan_samples.png"

    fig, axes = plt.subplots(4, 4, figsize=(5, 5))
    for i, ax in enumerate(axes.flatten()):
        img = (samples[i, 0].numpy() + 1.0) / 2.0
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


if __name__ == "__main__":
    main()

