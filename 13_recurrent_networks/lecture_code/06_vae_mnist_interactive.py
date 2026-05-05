"""Interactive 2D latent explorer for a tiny MNIST VAE."""

import argparse
import torch
from torch import nn
import torch.nn.functional as F

from interactive_common import default_output_path, save_or_show, setup_matplotlib


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.mu = nn.Linear(64, 2)
        self.log_var = nn.Linear(64, 2)
        self.decoder = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 28 * 28), nn.Sigmoid()
        )

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = self.mu(h), self.log_var(h)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, log_var


def loss_fn(x_hat, x, mu, log_var):
    bce = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kl


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive MNIST VAE latent explorer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick or args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_samples = min(args.max_samples, 256)
        args.batch_size = min(args.batch_size, 64)

    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("Missing optional dependency: torchvision")
        return

    plt, Slider, _, _, _, _ = setup_matplotlib(args.smoke)
    torch.manual_seed(args.seed)

    ds = datasets.MNIST(root=".data", train=True, download=True, transform=transforms.ToTensor())
    ds = torch.utils.data.Subset(ds, range(min(args.max_samples, len(ds))))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        total_loss = 0.0
        total_items = 0
        for x, _ in loader:
            x = x.view(x.size(0), -1)
            x_hat, mu, log_var = model(x)
            loss = loss_fn(x_hat, x, mu, log_var)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            total_items += x.size(0)
        avg_loss = total_loss / max(total_items, 1)
        print(f"epoch={epoch + 1} loss={avg_loss:.4f}")

    fig = plt.figure(figsize=(7, 4.8))
    ax_img = fig.add_axes([0.12, 0.24, 0.34, 0.62])
    ax_z1 = fig.add_axes([0.60, 0.18, 0.28, 0.04])
    ax_z2 = fig.add_axes([0.60, 0.10, 0.28, 0.04])
    s_z1 = Slider(ax_z1, "z1", -3.0, 3.0, valinit=0.0)
    s_z2 = Slider(ax_z2, "z2", -3.0, 3.0, valinit=0.0)

    def render(_=None):
        z = torch.tensor([[s_z1.val, s_z2.val]], dtype=torch.float32)
        with torch.no_grad():
            img = model.decoder(z).view(28, 28).numpy()
        ax_img.clear()
        ax_img.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
        ax_img.set_title(f"Decoded digit at z=({s_z1.val:.1f}, {s_z2.val:.1f})")
        ax_img.axis("off")
        fig.canvas.draw_idle()

    s_z1.on_changed(render)
    s_z2.on_changed(render)
    render()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()


