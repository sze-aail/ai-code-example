"""
09_convnext.py — ConvNeXt: A ConvNet for the 2020s
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - ConvNeXt (Liu et al., 2022): CNN modernizálva Transformer trükkökkel
  - ResNet → ConvNeXt lépésről lépésre átalakítás
  - Minden modernizációs lépés hatása a pontosságra
  - ConvNeXt blokk implementáció és összehasonlítás
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from pathlib import Path
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


BLOCK_DEFS = [
    ("resnet", "0. ResNet\n(baseline)", "0. ResNet"),
    ("step1", "1. + Depthwise\nSeparable", "1. + Depthwise"),
    ("step2", "2. + 7x7 kernel\n(nagyobb RF)", "2. + 7x7"),
    ("step3", "3. + Inverted\nBottleneck", "3. + Inverted"),
    ("convnext", "4. + LayerNorm\n+ GELU (ConvNeXt)", "4. ConvNeXt"),
]


# ══════════════════════════════════════════════════════════════
# BLOKKOK: ResNet → ConvNeXt lépésenként
# ══════════════════════════════════════════════════════════════

class ResNetBlock(nn.Module):
    """Klasszikus ResNet blokk."""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + x)


class Step1_DepthwiseBlock(nn.Module):
    """Lépés 1: 3×3 conv → depthwise separable."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn1 = nn.BatchNorm2d(dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        return F.relu(self.bn2(self.pwconv(F.relu(self.bn1(self.dwconv(x)))))) + x


class Step2_LargerKernelBlock(nn.Module):
    """Lépés 2: 3×3 → 7×7 depthwise kernel (nagyobb RF)."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.pwconv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return F.relu(self.pwconv(F.relu(self.bn(self.dwconv(x))))) + x


class Step3_InvertedBottleneck(nn.Module):
    """Lépés 3: inverted bottleneck (dim → 4×dim → dim)."""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)  # expand
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)   # project

    def forward(self, x):
        h = self.dwconv(x)
        h = self.norm(h)
        h = F.relu(self.pwconv1(h))
        h = self.pwconv2(h)
        return h + x


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt blokk (Liu et al., 2022): a teljes modernizáció.

    ResNet-hez képest:
    1. Depthwise separable conv (MobileNet-ből)
    2. Nagyobb kernel (7×7, Swin Transformer-ből)
    3. Inverted bottleneck (dim → 4×dim → dim)
    4. LayerNorm (BatchNorm helyett, Transformer-ből)
    5. GELU aktiváció (ReLU helyett, Transformer-ből)
    6. Kevesebb aktiváció (csak egy, Transformer-ből)
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)  # ← LayerNorm!
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()  # ← GELU!
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B,C,H,W) → (B,H,W,C) LayerNorm-hoz
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # vissza (B,C,H,W)
        return x + residual


class TestNet(nn.Module):
    def __init__(self, block_cls, channels=16, n_blocks=3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[block_cls(channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, 10))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def load_data(test_size=0.2, random_state=42):
    digits = load_digits()
    X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
    y = torch.tensor(digits.target, dtype=torch.long)
    Xt, Xv, yt, yv = [
        torch.tensor(np.array(a))
        for a in train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y.numpy())
    ]
    return Xt, Xv, yt, yv


def build_checkpoint_path(checkpoint_dir, model_id):
    return Path(checkpoint_dir) / f"09_convnext_{model_id}.pt"


def save_checkpoint(path, model, val_accs, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
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


def train_eval(model, Xt, Xv, yt, yv, epochs=80):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    val_accs = []
    for epoch in range(epochs):
        model.train()
        loss = criterion(model(Xt), yt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        val_accs.append(acc)
    return val_accs


def get_block_cls(model_id):
    mapping = {
        "resnet": ResNetBlock,
        "step1": Step1_DepthwiseBlock,
        "step2": Step2_LargerKernelBlock,
        "step3": Step3_InvertedBottleneck,
        "convnext": ConvNeXtBlock,
    }
    return mapping[model_id]


def run_modernization_experiment(Xt, Xv, yt, yv, checkpoint_dir=".", load_pt=True, save_pt=True, epochs=80):
    print("=" * 60)
    print("ResNet -> ConvNeXt: LEPESENKENTI MODERNIZACIO")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    results = []
    trained_models = {}

    for model_id, name, short_name in BLOCK_DEFS:
        torch.manual_seed(42)
        model = TestNet(get_block_cls(model_id))
        ckpt_path = build_checkpoint_path(checkpoint_dir, model_id)
        payload = load_checkpoint_if_exists(ckpt_path, model) if load_pt else None

        if payload is not None:
            va = payload.get("val_accs", [])
            if not va:
                model.eval()
                with torch.no_grad():
                    va = [(model(Xv).argmax(1) == yv).float().mean().item()]
            print(f"  {short_name:25s}: checkpoint betoltve ({ckpt_path.name})")
        else:
            va = train_eval(model, Xt, Xv, yt, yv, epochs=epochs)
            if save_pt:
                save_checkpoint(
                    ckpt_path,
                    model,
                    va,
                    extra={"model_id": model_id, "epochs": epochs},
                )
                print(f"  checkpoint mentve: {ckpt_path.name}")

        n_params = sum(p.numel() for p in model.parameters())
        trained_models[model_id] = model
        results.append((name, va[-1], n_params, model_id))
        axes[0].plot(va, linewidth=2, label=f"{short_name} ({va[-1]*100:.0f}%)")
        print(f"  {short_name:25s}: acc={va[-1]*100:.1f}%, params={n_params}")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Val accuracy")
    axes[0].set_title("Lepesenkenti modernizacio hatasa", fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    names, accs, params, _ = zip(*results)
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
    axes[1].barh(range(len(results)), [a * 100 for a in accs], color=colors, edgecolor="k")
    axes[1].set_yticks(range(len(results)))
    axes[1].set_yticklabels([n.replace("\n", " ") for n in names], fontsize=9)
    axes[1].set_xlabel("Val accuracy (%)")
    axes[1].set_title("Vegso pontossag", fontweight="bold")
    for i, (a, p) in enumerate(zip(accs, params)):
        axes[1].text(a * 100 + 0.3, i, f"{a*100:.1f}% ({p}p)", va="center", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="x")

    fig.suptitle(
        "ResNet -> ConvNeXt: 5 modernizacios lepes (Liu et al., 2022)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig("09_convnext_lepesek.png", dpi=150)
    print("\nAbra mentve: 09_convnext_lepesek.png")
    return trained_models


def create_summary_figure():
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.axis("off")
    data = [
        ["Elem", "ResNet (2015)", "Swin Transformer (2021)", "ConvNeXt (2022)"],
        ["Kernel meret", "3x3", "7x7 ablak (attention)", "7x7 depthwise conv"],
        ["Normalizacio", "BatchNorm", "LayerNorm", "LayerNorm"],
        ["Aktivacio", "ReLU (minden conv utan)", "GELU (1x per blokk)", "GELU (1x per blokk)"],
        ["Bottleneck", "dim -> dim/4 -> dim\n(standard)", "-", "dim -> 4xdim -> dim\n(inverted!)"],
        ["Stem", "7x7 conv, stride=2\n+ MaxPool", "4x4 patch embed\nstride=4", "4x4 conv,\nstride=4"],
        ["ImageNet\ntop-1", "76.1% (ResNet-50)", "83.5% (Swin-T)", "82.1% (ConvNeXt-T)\n-> CNN versenykepes!"],
    ]
    table = ax2.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.2)
    for j in range(4):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ECEFF1")
        table[i, 0].set_text_props(fontweight="bold")
    ax2.set_title(
        "ConvNeXt: Transformer trukkok CNN-ben - a CNN nem halott!",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    fig2.tight_layout()
    fig2.savefig("09_convnext_vs_swin.png", dpi=150)
    print("Abra mentve: 09_convnext_vs_swin.png")


def launch_gui(Xv, yv, models):
    if not models:
        print("GUI nem indithato: nincs elerheto modell.")
        return

    order = [model_id for model_id, _, _ in BLOCK_DEFS if model_id in models]
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[8, 1, 1], width_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, :])
    ax_radio = fig.add_subplot(gs[2, :])

    slider = Slider(ax_slider, "Minta index", 0, len(Xv) - 1, valinit=0, valstep=1)
    radio_labels = [short_name for model_id, _, short_name in BLOCK_DEFS if model_id in order]
    radio = RadioButtons(ax_radio, radio_labels, active=len(radio_labels) - 1)
    ax_text.axis("off")
    txt = ax_text.text(0.0, 0.98, "", va="top", fontsize=11, family="monospace")

    for model in models.values():
        model.eval()

    label_to_model = {
        short_name: model_id
        for model_id, _, short_name in BLOCK_DEFS
        if model_id in models
    }
    state = {"model_id": label_to_model[radio_labels[-1]]}

    def update(_):
        i = int(slider.val)
        x = Xv[i : i + 1]
        img = Xv[i, 0].detach().numpy()
        true_label = int(yv[i].item())
        model_id = state["model_id"]
        model = models[model_id]

        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())

        ax_img.clear()
        ax_img.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax_img.set_title(f"Digits minta #{i}", fontweight="bold")
        ax_img.axis("off")

        top3_idx = torch.argsort(probs, descending=True)[:3].tolist()
        top3_text = "\n".join([f"  {k}. {int(c)} -> {float(probs[c])*100:5.1f}%" for k, c in enumerate(top3_idx, 1)])
        txt.set_text(
            "\n".join(
                [
                    f"Model: {model_id}",
                    f"True label: {true_label}",
                    f"Pred: {pred} ({conf*100:5.1f}%)",
                    "",
                    "Top-3:",
                    top3_text,
                ]
            )
        )
        fig.canvas.draw_idle()

    def on_model_change(label):
        state["model_id"] = label_to_model[label]
        update(None)

    slider.on_changed(update)
    radio.on_clicked(on_model_change)
    update(None)
    fig.suptitle("ConvNeXt modernizacios lepesek - interaktiv teszt", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def load_or_train_models_for_gui(Xt, Xv, yt, yv, checkpoint_dir, load_pt, save_pt):
    models = {}
    for model_id, _, _ in BLOCK_DEFS:
        model = TestNet(get_block_cls(model_id))
        ckpt_path = build_checkpoint_path(checkpoint_dir, model_id)
        payload = load_checkpoint_if_exists(ckpt_path, model) if load_pt else None
        if payload is None:
            va = train_eval(model, Xt, Xv, yt, yv, epochs=30)
            if save_pt:
                save_checkpoint(
                    ckpt_path,
                    model,
                    va,
                    extra={"model_id": model_id, "epochs": 30, "purpose": "gui"},
                )
        models[model_id] = model
    return models


def parse_args():
    parser = argparse.ArgumentParser(description="ConvNeXt demo checkpoint es GUI tamogatassal")
    parser.add_argument("--checkpoint-dir", default=".", help="Checkpoint konyvtar (.pt fajlok)")
    parser.add_argument("--no-load-pt", action="store_true", help="Ne toltsen be meglevo checkpointot")
    parser.add_argument("--no-save-pt", action="store_true", help="Ne mentsen checkpointot")
    parser.add_argument("--gui", action="store_true", help="GUI inditasa")
    parser.add_argument("--gui-only", action="store_true", help="Csak GUI (fo kiserlet kihagyasa)")
    parser.add_argument("--epochs", type=int, default=80, help="Epochok szama a fo kiserlethez")
    return parser.parse_args()


def main():
    args = parse_args()
    load_pt = not args.no_load_pt
    save_pt = not args.no_save_pt
    Xt, Xv, yt, yv = load_data()

    trained_models = {}
    if not args.gui_only:
        trained_models = run_modernization_experiment(
            Xt,
            Xv,
            yt,
            yv,
            checkpoint_dir=args.checkpoint_dir,
            load_pt=load_pt,
            save_pt=save_pt,
            epochs=args.epochs,
        )
        create_summary_figure()

    if args.gui:
        gui_models = dict(trained_models)
        missing_ids = [model_id for model_id, _, _ in BLOCK_DEFS if model_id not in gui_models]
        if missing_ids:
            loaded = load_or_train_models_for_gui(Xt, Xv, yt, yv, args.checkpoint_dir, load_pt, save_pt)
            for model_id in missing_ids:
                gui_models[model_id] = loaded[model_id]
        launch_gui(Xv, yv, gui_models)

    plt.close("all")
    print("\nKesz!")


if __name__ == "__main__":
    main()
