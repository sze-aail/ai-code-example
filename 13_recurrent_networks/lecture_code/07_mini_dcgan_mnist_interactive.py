"""Interactive latent explorer for a tiny MNIST GAN."""

import argparse
import torch
from torch import nn

from interactive_common import default_output_path, save_or_show, setup_matplotlib


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 128), nn.LeakyReLU(0.2), nn.Linear(128, 28 * 28), nn.Tanh()
        )

    def forward(self, z):
        return self.net(z).view(z.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128), nn.LeakyReLU(0.2), nn.Linear(128, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive MNIST GAN latent explorer")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_minus_one_one(x):
    return x * 2.0 - 1.0


def main():
    args = parse_args()
    if args.quick or args.smoke:
        args.epochs = min(args.epochs, 1)
        args.max_samples = min(args.max_samples, 256)

    try:
        from torchvision import datasets, transforms
    except ImportError:
        print("Missing optional dependency: torchvision")
        return

    plt, Slider, _, _, _, _ = setup_matplotlib(args.smoke)
    torch.manual_seed(args.seed)

    ds = datasets.MNIST(
        root=".data",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(to_minus_one_one)]),
    )
    ds = torch.utils.data.Subset(ds, range(min(args.max_samples, len(ds))))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    G = Generator()
    D = Discriminator()
    bce = nn.BCEWithLogitsLoss()
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        for real, _ in loader:
            b = real.size(0)
            ones = torch.ones(b, 1)
            zeros = torch.zeros(b, 1)

            z = torch.randn(b, 2)
            fake_detached = G(z).detach()
            loss_d = bce(D(real), ones) + bce(D(fake_detached), zeros)
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

            z = torch.randn(b, 2)
            fake = G(z)
            loss_g = bce(D(fake), ones)
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()
        print(f"epoch={epoch + 1} loss_D={loss_d.item():.4f} loss_G={loss_g.item():.4f}")

    fig = plt.figure(figsize=(7, 4.8))
    ax_img = fig.add_axes([0.12, 0.24, 0.34, 0.62])
    ax_z1 = fig.add_axes([0.60, 0.18, 0.28, 0.04])
    ax_z2 = fig.add_axes([0.60, 0.10, 0.28, 0.04])
    s_z1 = Slider(ax_z1, "z1", -3.0, 3.0, valinit=0.0)
    s_z2 = Slider(ax_z2, "z2", -3.0, 3.0, valinit=0.0)

    def render(_=None):
        with torch.no_grad():
            img = G(torch.tensor([[s_z1.val, s_z2.val]], dtype=torch.float32))[0, 0].numpy()
        ax_img.clear()
        ax_img.imshow((img + 1.0) / 2.0, cmap="gray", vmin=0.0, vmax=1.0)
        ax_img.set_title(f"Generated digit at z=({s_z1.val:.1f}, {s_z2.val:.1f})")
        ax_img.axis("off")
        fig.canvas.draw_idle()

    s_z1.on_changed(render)
    s_z2.on_changed(render)
    render()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()

