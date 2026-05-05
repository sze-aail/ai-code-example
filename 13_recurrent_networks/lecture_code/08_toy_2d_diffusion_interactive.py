"""Interactive toy diffusion demo with timestep slider."""

import argparse

import torch
from torch import nn

from interactive_common import default_output_path, save_or_show, setup_matplotlib


def make_two_spirals(n: int = 1200, noise: float = 0.05, seed: int = 42):
    torch.manual_seed(seed)
    n2 = n // 2
    theta = torch.linspace(0, 4 * torch.pi, n2)
    r = torch.linspace(0.1, 1.0, n2)
    x1 = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=1)
    x2 = torch.stack([-r * torch.cos(theta), -r * torch.sin(theta)], dim=1)
    return torch.cat([x1, x2], dim=0) + noise * torch.randn(n, 2)


class TimeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x_t, t_norm):
        return self.net(torch.cat([x_t, t_norm], dim=-1))


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive toy diffusion")
    parser.add_argument("--timesteps", type=int, default=60)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick or args.smoke:
        args.timesteps = min(args.timesteps, 40)
        args.steps = min(args.steps, 120)

    plt, Slider, Button, _, _, _ = setup_matplotlib(args.smoke)
    torch.manual_seed(args.seed)

    x0 = make_two_spirals(seed=args.seed)
    beta = torch.linspace(1e-4, 2e-2, args.timesteps)
    alpha = 1.0 - beta
    alphabar = torch.cumprod(alpha, dim=0)
    model = TimeMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, x0.size(0), (128,))
        x = x0[idx]
        t = torch.randint(1, args.timesteps + 1, (128,))
        eps = torch.randn_like(x)
        a_bar_t = alphabar[t - 1].unsqueeze(1)
        x_t = torch.sqrt(a_bar_t) * x + torch.sqrt(1.0 - a_bar_t) * eps
        eps_pred = model(x_t, (t.float() / args.timesteps).unsqueeze(1))
        loss = torch.mean((eps - eps_pred) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, args.steps // 3) == 0 or step == 1:
            print(f"step={step:3d} loss={loss.item():.4f}")

    fig = plt.figure(figsize=(10, 5))
    ax_data = fig.add_axes([0.08, 0.24, 0.36, 0.62])
    ax_noisy = fig.add_axes([0.56, 0.24, 0.36, 0.62])
    ax_t = fig.add_axes([0.20, 0.11, 0.55, 0.04])
    ax_btn = fig.add_axes([0.78, 0.08, 0.12, 0.08])
    slider = Slider(ax_t, "t", 1, args.timesteps, valinit=args.timesteps // 2, valstep=1)
    button = Button(ax_btn, "Sample")
    sampled = None

    def sample_reverse():
        x = torch.randn(1200, 2)
        with torch.no_grad():
            for t in range(args.timesteps, 0, -1):
                a_t = alpha[t - 1]
                a_bar_t = alphabar[t - 1]
                eps_pred = model(x, torch.full((x.size(0), 1), t / args.timesteps))
                x = (1.0 / torch.sqrt(a_t)) * (x - ((1.0 - a_t) / torch.sqrt(1.0 - a_bar_t)) * eps_pred)
                if t > 1:
                    x = x + torch.sqrt(beta[t - 1]) * torch.randn_like(x)
        return x

    def render(_=None):
        nonlocal sampled
        t = int(slider.val)
        eps = torch.randn_like(x0)
        a_bar_t = alphabar[t - 1]
        x_t = torch.sqrt(a_bar_t) * x0 + torch.sqrt(1.0 - a_bar_t) * eps

        ax_data.clear()
        ax_data.scatter(x0[:, 0].numpy(), x0[:, 1].numpy(), s=3, alpha=0.45)
        ax_data.set_title("Training data")

        ax_noisy.clear()
        current = sampled if sampled is not None else x_t
        ax_noisy.scatter(current[:, 0].numpy(), current[:, 1].numpy(), s=3, alpha=0.45, color="tab:orange")
        ax_noisy.set_title("Reverse samples" if sampled is not None else f"Forward noising at t={t}")

        for ax in (ax_data, ax_noisy):
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
        fig.canvas.draw_idle()

    def on_sample(_event):
        nonlocal sampled
        sampled = sample_reverse()
        render()

    slider.on_changed(render)
    button.on_clicked(on_sample)
    render()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()

