"""
07_highway_vs_resnet.py — Highway Network vs. ResNet
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - Highway Network (Srivastava et al., 2015): tanulható kapu T(x) szabályozza az áteresztést
  - ResNet (He et al., 2015): fix identity skip, egyszerűbb és stabilabb
  - Összehasonlítás: kapu-aktivációk vizualizáció, konvergencia, gradiens-áramlás
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from pathlib import Path
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════
# BLOKKOK
# ══════════════════════════════════════════════════════════════

class HighwayBlock(nn.Module):
    """
    Highway Network (Srivastava, Greff & Schmidhuber, 2015).

    y = T(x) ⊙ H(x) + (1 - T(x)) ⊙ x

    T(x) = σ(W_T·x + b_T)  : transform gate (mennyit engedünk át a transzformációból)
    H(x) = ReLU(W_H·x + b_H): transzformáció
    (1-T(x))·x              : carry gate (mennyit tartunk meg az eredetiből)

    Ha T→0: y≈x (identity, mint ResNet)
    Ha T→1: y≈H(x) (teljes transzformáció)
    A háló TANULJA, mikor melyik réteg fontos!
    """
    def __init__(self, channels):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.Sigmoid(),
        )

    def forward(self, x):
        H = self.transform(x)
        T = self.gate(x)
        return T * H + (1 - T) * x  # Highway formula

    def get_gate_values(self, x):
        """Kapu-értékek visszaadása vizualizációhoz."""
        return self.gate(x)


class ResBlock(nn.Module):
    """ResNet: y = F(x) + x (fix identity skip)."""
    def __init__(self, channels):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(self.transform(x) + x)


class ConfigurableNet(nn.Module):
    def __init__(self, n_blocks, block_type='resnet', channels=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        Block = ResBlock if block_type == 'resnet' else HighwayBlock
        self.blocks = nn.ModuleList([Block(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)
        self.block_type = block_type

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.pool(x).flatten(1))

    def get_gate_map(self, x):
        """Highway kapu-értékek rétegenként."""
        if self.block_type != 'highway':
            return None
        x = self.stem(x)
        gates = []
        for block in self.blocks:
            gates.append(block.get_gate_values(x).mean(dim=(0, 2, 3)).detach().numpy())
            x = block(x)
        return gates


def load_data(test_size=0.2, random_state=42):
    """Digits adathalmaz betöltése és train/validation split."""
    digits = load_digits()
    X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
    y = torch.tensor(digits.target, dtype=torch.long)
    Xt, Xv, yt, yv = [
        torch.tensor(np.array(a))
        for a in train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y.numpy())
    ]
    return Xt, Xv, yt, yv


def train_model(model, Xt, Xv, yt, yv, epochs=100, lr=0.005):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        loss = criterion(model(Xt), yt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        val_accs.append(acc)
    return train_losses, val_accs


def build_checkpoint_path(checkpoint_dir, block_type, n_blocks, tag="main"):
    suffix = f"{block_type}_{n_blocks}b"
    if tag != "main":
        suffix = f"{suffix}_{tag}"
    return Path(checkpoint_dir) / f"07_highway_vs_resnet_{suffix}.pt"


def save_checkpoint(path, model, train_losses=None, val_accs=None, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "train_losses": train_losses or [],
        "val_accs": val_accs or [],
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint_if_exists(path, model):
    if not path.exists():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    return payload


def run_highway_vs_resnet_experiment(Xt, Xv, yt, yv, checkpoint_dir=".", load_pt=True, save_pt=True):
    """Highway és ResNet modellek összehasonlítása, valamint kapu-vizualizáció."""
    print("=" * 60)
    print("1. HIGHWAY NETWORK vs. RESNET")
    print("=" * 60)

    fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5))
    trained_models = {}

    for n_blocks in [3, 6]:
        for btype, color, ls in [('resnet', '#4CAF50', '-'), ('highway', '#E91E63', '--')]:
            torch.manual_seed(42)
            model = ConfigurableNet(n_blocks, btype)
            ckpt_path = build_checkpoint_path(checkpoint_dir, btype, n_blocks)

            payload = load_checkpoint_if_exists(ckpt_path, model) if load_pt else None
            if payload is not None:
                tl = payload.get("train_losses", [])
                va = payload.get("val_accs", [])
                if not va:
                    model.eval()
                    with torch.no_grad():
                        va = [(model(Xv).argmax(1) == yv).float().mean().item()]
                if not tl:
                    tl = [np.nan] * len(va)
                print(f"  {btype:8s} {n_blocks} blokk: checkpoint betöltve ({ckpt_path.name})")
            else:
                tl, va = train_model(model, Xt, Xv, yt, yv, epochs=50)
                if save_pt:
                    save_checkpoint(
                        ckpt_path,
                        model,
                        tl,
                        va,
                        extra={"block_type": btype, "n_blocks": n_blocks, "epochs": 50},
                    )
                    print(f"  checkpoint mentve: {ckpt_path.name}")

            trained_models[(btype, n_blocks)] = model
            n_params = sum(p.numel() for p in model.parameters())
            axes1[0].plot(tl, color=color, linestyle=ls, linewidth=1.5,
                          label=f"{btype} {n_blocks}b ({va[-1]*100:.0f}%)")
            axes1[1].plot(va, color=color, linestyle=ls, linewidth=1.5,
                          label=f"{btype} {n_blocks}b ({n_params}p)")
            print(f"  {btype:8s} {n_blocks} blokk: acc={va[-1]*100:.1f}%, params={n_params}")

    axes1[0].set_title("Tanítási veszteség", fontweight="bold")
    axes1[0].set_xlabel("Epoch"); axes1[0].legend(fontsize=8); axes1[0].grid(True, alpha=0.3)
    axes1[1].set_title("Validációs pontosság", fontweight="bold")
    axes1[1].set_xlabel("Epoch"); axes1[1].legend(fontsize=8); axes1[1].grid(True, alpha=0.3)

    torch.manual_seed(42)
    hw_model = ConfigurableNet(6, 'highway')
    hw_ckpt = build_checkpoint_path(checkpoint_dir, "highway", 6, tag="gate80")
    hw_payload = load_checkpoint_if_exists(hw_ckpt, hw_model) if load_pt else None
    if hw_payload is None:
        hw_tl, hw_va = train_model(hw_model, Xt, Xv, yt, yv, epochs=80)
        if save_pt:
            save_checkpoint(
                hw_ckpt,
                hw_model,
                hw_tl,
                hw_va,
                extra={"block_type": "highway", "n_blocks": 6, "epochs": 80, "purpose": "gate_map"},
            )
            print(f"  checkpoint mentve: {hw_ckpt.name}")
    else:
        print(f"  highway gate checkpoint betöltve ({hw_ckpt.name})")
    hw_model.eval()

    with torch.no_grad():
        gates = hw_model.get_gate_map(Xv[:1])

    ax = axes1[2]
    gate_matrix = np.array(gates)  # (n_blocks, channels)
    im = ax.imshow(gate_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Csatorna"); ax.set_ylabel("Blokk (réteg)")
    ax.set_title("Highway kapu-értékek T(x)\n(zöld=áteresztés, piros=identity)", fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)

    fig1.suptitle("Highway Network vs. ResNet: tanulható kapu vs. fix identity skip",
                  fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("07_highway_vs_resnet.png", dpi=150)
    print("\nÁbra mentve: 07_highway_vs_resnet.png")
    return trained_models


def create_skip_summary_figure():
    """Összefoglaló táblázat különböző skip connection variánsokról."""
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.axis("off")
    data = [
        ["", "Highway Network", "ResNet", "DenseNet"],
        ["Év", "2015 (Srivastava,\nSchmidhuber)", "2015 (He et al.)", "2017 (Huang et al.)"],
        ["Skip típus", "y = T·H(x) + (1-T)·x\nTANULHATÓ kapu", "y = F(x) + x\nFIX identity", "y = [x, F1(x), F2(x),...]\nKONKATENÁCIÓ"],
        ["Extra param.", "Igen (kapu hálózat)", "Nincs", "Nincs (de nő a\ncsatornaszám)"],
        ["Gradiens", "T(x) szabályozza", "Mindig 1 (direkt út)", "Minden rétegből\ndirekt út"],
        ["Erősség", "Adaptív: a háló\ndönti el mi fontos", "Egyszerű, stabil,\njól skálázódik", "Feature reuse,\nkevés paraméter"],
    ]
    table = ax2.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 2.4)
    for j in range(4):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ECEFF1")
        table[i, 0].set_text_props(fontweight="bold")
    ax2.set_title("Skip connection variánsok összehasonlítása", fontsize=14, fontweight="bold", pad=20)
    fig2.tight_layout()
    fig2.savefig("07_skip_variansok.png", dpi=150)
    print("Ábra mentve: 07_skip_variansok.png")


def load_or_train_for_gui(block_type, n_blocks, Xt, Xv, yt, yv, checkpoint_dir, load_pt, save_pt):
    model = ConfigurableNet(n_blocks, block_type)
    ckpt_path = build_checkpoint_path(checkpoint_dir, block_type, n_blocks)
    payload = load_checkpoint_if_exists(ckpt_path, model) if load_pt else None
    if payload is None:
        tl, va = train_model(model, Xt, Xv, yt, yv, epochs=30)
        if save_pt:
            save_checkpoint(
                ckpt_path,
                model,
                tl,
                va,
                extra={"block_type": block_type, "n_blocks": n_blocks, "epochs": 30, "purpose": "gui"},
            )
    return model


def launch_gui(Xv, yv, models_by_depth, initial_depth=6):
    """Interaktív GUI minták közti léptetéssel és két architektúra predikcióval."""
    if not models_by_depth:
        print("GUI nem indítható: egyik modell sem elérhető.")
        return

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(4, 2, height_ratios=[8, 1, 1, 1], width_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, :])
    ax_depth = fig.add_subplot(gs[2, 0])
    ax_hint = fig.add_subplot(gs[2, 1])
    ax_prev = fig.add_subplot(gs[3, 0])
    ax_next = fig.add_subplot(gs[3, 1])

    slider = Slider(ax_slider, "Minta index", 0, len(Xv) - 1, valinit=0, valstep=1)
    depth_selector = RadioButtons(ax_depth, ["3", "6"], active=1 if int(initial_depth) == 6 else 0)
    btn_prev = Button(ax_prev, "Előző")
    btn_next = Button(ax_next, "Következő")

    ax_text.axis("off")
    ax_hint.axis("off")
    ax_hint.text(0.0, 0.45, "Depth selector: 3b / 6b", fontsize=10)
    txt = ax_text.text(0.0, 0.98, "", va="top", fontsize=11, family="monospace")

    for depth_models in models_by_depth.values():
        for model in depth_models.values():
            if model is not None:
                model.eval()

    state = {"depth": int(initial_depth) if int(initial_depth) in models_by_depth else sorted(models_by_depth.keys())[0]}

    def predict_line(name, model, x):
        if model is None:
            return f"{name}: N/A"
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
        return f"{name}: pred={pred}  conf={conf*100:5.1f}%"

    def update(idx):
        i = int(idx)
        depth = state["depth"]
        depth_models = models_by_depth.get(depth, {})
        resnet_model = depth_models.get("resnet")
        highway_model = depth_models.get("highway")
        x = Xv[i:i + 1]
        img = Xv[i, 0].detach().numpy()
        true_label = int(yv[i].item())

        ax_img.clear()
        ax_img.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        ax_img.set_title(f"Digits minta #{i} (true={true_label})", fontweight="bold")
        ax_img.axis("off")

        lines = [
            f"True label: {true_label} (depth={depth}b)",
            "",
            predict_line(f"ResNet {depth}b ", resnet_model, x),
            predict_line(f"Highway {depth}b", highway_model, x),
        ]
        txt.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def set_depth(label):
        state["depth"] = int(label)
        update(slider.val)

    def go_prev(_):
        slider.set_val(max(0, int(slider.val) - 1))

    def go_next(_):
        slider.set_val(min(len(Xv) - 1, int(slider.val) + 1))

    slider.on_changed(update)
    depth_selector.on_clicked(set_depth)
    btn_prev.on_clicked(go_prev)
    btn_next.on_clicked(go_next)
    update(0)
    fig.suptitle("Highway vs ResNet teszt GUI", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Highway vs ResNet kísérlet checkpoint és GUI támogatással")
    parser.add_argument("--checkpoint-dir", default=".", help="Checkpoint könyvtár (.pt fájlok)")
    parser.add_argument("--no-load-pt", action="store_true", help="Ne töltsön be meglévő checkpointot")
    parser.add_argument("--no-save-pt", action="store_true", help="Ne mentsen checkpointot")
    parser.add_argument("--gui", action="store_true", help="GUI indítása a két architektúra teszteléséhez")
    parser.add_argument("--gui-only", action="store_true", help="Csak GUI (fő kísérlet és ábrák kihagyása)")
    parser.add_argument("--gui-depth", type=int, default=6, choices=[3, 6], help="Kezdő mélység a GUI-ban (3 vagy 6)")
    return parser.parse_args()


def main():
    args = parse_args()
    load_pt = not args.no_load_pt
    save_pt = not args.no_save_pt

    Xt, Xv, yt, yv = load_data()

    trained_models = {}
    if not args.gui_only:
        trained_models = run_highway_vs_resnet_experiment(
            Xt,
            Xv,
            yt,
            yv,
            checkpoint_dir=args.checkpoint_dir,
            load_pt=load_pt,
            save_pt=save_pt,
        )
        create_skip_summary_figure()

    if args.gui:
        gui_models = {3: {}, 6: {}}
        for depth in [3, 6]:
            for btype in ["resnet", "highway"]:
                model = trained_models.get((btype, depth))
                if model is None:
                    model = load_or_train_for_gui(
                        btype, depth, Xt, Xv, yt, yv, args.checkpoint_dir, load_pt, save_pt
                    )
                gui_models[depth][btype] = model
        launch_gui(Xv, yv, gui_models, initial_depth=args.gui_depth)

    plt.close("all")
    print("\nKész!")


if __name__ == '__main__':
    main()

