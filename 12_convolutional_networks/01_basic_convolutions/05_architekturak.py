"""
05_architekturak.py — CNN architekturak es skip connection
Neuralis halok II. (CNN) — Hajdu Csaba
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


DEPTHS = [2, 4, 6]


class ResidualBlock(nn.Module):
    """ResNet alap blokk: F(x) + x (skip connection)."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class PlainBlock(nn.Module):
    """Ugyanaz skip nelkul."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)


class CNNConfigurable(nn.Module):
    def __init__(self, n_blocks, use_skip=True, channels=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        block = ResidualBlock if use_skip else PlainBlock
        self.blocks = nn.Sequential(*[block(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
    y = torch.tensor(digits.target, dtype=torch.long)
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y.numpy())
    Xt, Xv = torch.tensor(np.array(Xt)), torch.tensor(np.array(Xv))
    yt, yv = torch.tensor(np.array(yt)), torch.tensor(np.array(yv))
    return Xt, Xv, yt, yv


def build_checkpoint_path(checkpoint_dir, n_blocks, use_skip):
    tag = "resnet" if use_skip else "plain"
    return Path(checkpoint_dir) / f"05_architekturak_{tag}_{n_blocks}b.pt"


def save_checkpoint(path, model, train_losses, val_accs, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses": train_losses,
            "val_accs": val_accs,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint_if_exists(path, model):
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def train_eval(model, Xt, Xv, yt, yv, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accs = [], []
    for _ in range(epochs):
        model.train()
        loss = criterion(model(Xt), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        val_accs.append(acc)
    return train_losses, val_accs


def train_or_load_model(Xt, Xv, yt, yv, n_blocks, use_skip, checkpoint_dir, load_pt, save_pt, epochs):
    torch.manual_seed(42)
    model = CNNConfigurable(n_blocks, use_skip=use_skip)
    ckpt = build_checkpoint_path(checkpoint_dir, n_blocks, use_skip)
    payload = load_checkpoint_if_exists(ckpt, model) if load_pt else None
    if payload is not None:
        tl = payload.get("train_losses", [])
        va = payload.get("val_accs", [])
        if not va:
            model.eval()
            with torch.no_grad():
                va = [(model(Xv).argmax(1) == yv).float().mean().item()]
        if not tl:
            tl = [np.nan] * len(va)
    else:
        tl, va = train_eval(model, Xt, Xv, yt, yv, epochs=epochs)
        if save_pt:
            save_checkpoint(
                ckpt,
                model,
                tl,
                va,
                extra={"n_blocks": n_blocks, "use_skip": use_skip, "epochs": epochs},
            )
    return model, tl, va


def run_skip_experiment(Xt, Xv, yt, yv, checkpoint_dir=".", load_pt=True, save_pt=True, epochs=50):
    print("=" * 60)
    print("1. SKIP CONNECTION HATASA: Plain vs. ResNet")
    print("=" * 60)
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))
    models = {}

    for ax, n_blocks in zip(axes1, DEPTHS):
        for use_skip, name, color in [(False, "Plain", "#E91E63"), (True, "ResNet", "#4CAF50")]:
            model, _tl, va = train_or_load_model(
                Xt,
                Xv,
                yt,
                yv,
                n_blocks,
                use_skip,
                checkpoint_dir,
                load_pt,
                save_pt,
                epochs,
            )
            models[(n_blocks, "resnet" if use_skip else "plain")] = model
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {n_blocks} blokk, {name}: acc={va[-1]*100:.1f}%, params={n_params}")
            ax.plot(va, label=f"{name} ({va[-1]*100:.0f}%)", color=color, linewidth=2)

        ax.set_title(f"{n_blocks} blokk ({n_blocks*4+2} conv reteg)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val accuracy")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.02)

    fig1.suptitle("Skip connection hatasa: melyebb halonal kritikus a kulonbseg", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("05_skip_connection.png", dpi=150)
    print("\nAbra mentve: 05_skip_connection.png")
    return models


def create_architecture_scatter_figure():
    print("\n" + "=" * 60)
    print("2. ARCHITEKTURAK OSSZEHASONLITASA")
    print("=" * 60)
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    archs = [
        ("LeNet-5\n(1998)", 0.06, 99.2, 1998),
        ("AlexNet\n(2012)", 61, 84.7, 2012),
        ("VGG-16\n(2014)", 138, 92.7, 2014),
        ("GoogLeNet\n(2014)", 6.8, 93.3, 2014),
        ("ResNet-50\n(2015)", 25.6, 96.4, 2015),
        ("MobileNetV2\n(2018)", 3.4, 92.1, 2018),
        ("EfficientNet-B0\n(2019)", 5.3, 93.3, 2019),
        ("ConvNeXt-T\n(2022)", 28.6, 96.2, 2022),
        ("ViT-B/16\n(2020)", 86.6, 97.8, 2020),
    ]
    names, params, accs, years = zip(*archs)
    scatter = ax2.scatter(params, accs, c=years, s=200, cmap="RdYlGn", edgecolors="k", linewidth=1, zorder=3)
    for i, name in enumerate(names):
        ax2.annotate(name, (params[i], accs[i]), fontsize=8, ha="center", xytext=(0, 15), textcoords="offset points", fontweight="bold")
    plt.colorbar(scatter, ax=ax2, label="Ev")
    ax2.set_xlabel("Parameterek (millio)", fontsize=12)
    ax2.set_ylabel("ImageNet top-5 accuracy (%)", fontsize=12)
    ax2.set_title("CNN architekturak: parameterek vs. pontossag", fontsize=14, fontweight="bold")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("05_architekturak.png", dpi=150)
    print("Abra mentve: 05_architekturak.png")


def calc_receptive_field(layers):
    rf = 1
    stride_prod = 1
    for k, s, d in layers:
        rf = rf + (k - 1) * d * stride_prod
        stride_prod *= s
    return rf


def create_receptive_field_figure():
    print("\n" + "=" * 60)
    print("3. RECEPTIVE FIELD SZAMITAS")
    print("=" * 60)
    architectures_rf = {
        "LeNet-5": [(5, 1, 1), (2, 2, 1), (5, 1, 1), (2, 2, 1)],
        "VGG-16 (elso 5)": [(3, 1, 1)] * 2 + [(2, 2, 1)] + [(3, 1, 1)] * 2 + [(2, 2, 1)],
        "ResNet (6 blokk)": [(3, 1, 1)] * 12,
        "Dilated (d=1,2,4,8)": [(3, 1, 1), (3, 1, 2), (3, 1, 4), (3, 1, 8)],
    }
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for name, layers in architectures_rf.items():
        rfs = [1]
        for i in range(1, len(layers) + 1):
            rfs.append(calc_receptive_field(layers[:i]))
        ax3.plot(range(len(rfs)), rfs, "o-", linewidth=2, markersize=6, label=f"{name} (RF={rfs[-1]})")
        print(f"  {name}: RF = {rfs[-1]}")

    ax3.set_xlabel("Reteg sorszama", fontsize=12)
    ax3.set_ylabel("Receptive field (pixel)", fontsize=12)
    ax3.set_title("Receptive field novekedes retegenkent", fontsize=13, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("05_receptive_field.png", dpi=150)
    print("Abra mentve: 05_receptive_field.png")


def load_or_train_gui_models(Xt, Xv, yt, yv, models, checkpoint_dir, load_pt, save_pt):
    gui_models = dict(models)
    for n_blocks in DEPTHS:
        for use_skip, tag in [(False, "plain"), (True, "resnet")]:
            key = (n_blocks, tag)
            if key not in gui_models:
                model, _tl, _va = train_or_load_model(
                    Xt,
                    Xv,
                    yt,
                    yv,
                    n_blocks,
                    use_skip,
                    checkpoint_dir,
                    load_pt,
                    save_pt,
                    epochs=30,
                )
                gui_models[key] = model
    return gui_models


def launch_gui(Xv, yv, models):
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[8, 1, 1], width_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, :])
    ax_depth = fig.add_subplot(gs[2, :])

    slider = Slider(ax_slider, "Minta index", 0, len(Xv) - 1, valinit=0, valstep=1)
    depth_selector = RadioButtons(ax_depth, [str(d) for d in DEPTHS], active=len(DEPTHS) - 1)
    ax_text.axis("off")
    txt = ax_text.text(0.0, 0.98, "", va="top", fontsize=11, family="monospace")
    state = {"depth": DEPTHS[-1]}

    for model in models.values():
        model.eval()

    def pred_line(model, x):
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
        return pred, conf

    def update(_):
        i = int(slider.val)
        depth = state["depth"]
        x = Xv[i : i + 1]
        true_label = int(yv[i].item())
        plain_model = models[(depth, "plain")]
        resnet_model = models[(depth, "resnet")]
        p_pred, p_conf = pred_line(plain_model, x)
        r_pred, r_conf = pred_line(resnet_model, x)

        ax_img.clear()
        ax_img.imshow(Xv[i, 0].detach().numpy(), cmap="gray_r", vmin=0, vmax=1)
        ax_img.set_title(f"Digits minta #{i} (true={true_label})", fontweight="bold")
        ax_img.axis("off")

        txt.set_text(
            "\n".join(
                [
                    f"Depth: {depth} blokk",
                    f"True label: {true_label}",
                    "",
                    f"Plain : pred={p_pred}  conf={p_conf*100:5.1f}%",
                    f"ResNet: pred={r_pred}  conf={r_conf*100:5.1f}%",
                ]
            )
        )
        fig.canvas.draw_idle()

    def set_depth(label):
        state["depth"] = int(label)
        update(None)

    slider.on_changed(update)
    depth_selector.on_clicked(set_depth)
    update(None)
    fig.suptitle("Skip connection GUI: Plain vs ResNet", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="CNN architekturak demo checkpoint es GUI tamogatassal")
    parser.add_argument("--checkpoint-dir", default=".", help="Checkpoint konyvtar (.pt)")
    parser.add_argument("--no-load-pt", action="store_true", help="Ne toltsen be meglevo checkpointot")
    parser.add_argument("--no-save-pt", action="store_true", help="Ne mentsen checkpointot")
    parser.add_argument("--gui", action="store_true", help="GUI inditasa")
    parser.add_argument("--gui-only", action="store_true", help="Csak GUI (statikus abrak kihagyasa)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochok szama a skip kiserlethez")
    return parser.parse_args()


def main():
    args = parse_args()
    Xt, Xv, yt, yv = load_data()
    load_pt = not args.no_load_pt
    save_pt = not args.no_save_pt

    models = {}
    if not args.gui_only:
        models = run_skip_experiment(
            Xt,
            Xv,
            yt,
            yv,
            checkpoint_dir=args.checkpoint_dir,
            load_pt=load_pt,
            save_pt=save_pt,
            epochs=args.epochs,
        )
        create_architecture_scatter_figure()
        create_receptive_field_figure()

    if args.gui:
        gui_models = load_or_train_gui_models(
            Xt,
            Xv,
            yt,
            yv,
            models,
            checkpoint_dir=args.checkpoint_dir,
            load_pt=load_pt,
            save_pt=save_pt,
        )
        launch_gui(Xv, yv, gui_models)

    plt.close("all")
    print("\nKesz!")


if __name__ == "__main__":
    main()
