"""
06_data_augmentation.py — Data augmentation vizualizacio
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


def load_data(train_size=100, random_state=42):
    digits = load_digits()
    X_all = digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0
    y_all = digits.target
    X_train_small, X_test, y_train_small, y_test = train_test_split(
        X_all,
        y_all,
        train_size=train_size,
        random_state=random_state,
        stratify=y_all,
    )
    Xt = torch.tensor(X_train_small)
    yt = torch.tensor(y_train_small, dtype=torch.long)
    Xv = torch.tensor(X_test)
    yv = torch.tensor(y_test, dtype=torch.long)
    return X_all, y_all, Xt, Xv, yt, yv


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 2 * 2, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return self.fc(x.view(-1, 16 * 2 * 2))


def build_checkpoint_path(checkpoint_dir, tag):
    return Path(checkpoint_dir) / f"06_data_augmentation_{tag}.pt"


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


def rotate_img(img, angle_deg):
    rad = np.radians(angle_deg)
    h, w = img.shape
    cy, cx = h / 2, w / 2
    result = np.zeros_like(img)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    for y in range(h):
        for x in range(w):
            src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx
            src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy
            if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                x0, y0 = int(src_x), int(src_y)
                result[y, x] = img[y0, x0]
    return result


def apply_named_augmentation(img, name):
    if name == "Eredeti":
        return img
    if name == "Forgatas +15":
        return rotate_img(img, 15)
    if name == "Forgatas -15":
        return rotate_img(img, -15)
    if name == "Vizszintes tukrozes":
        return img[:, ::-1]
    if name == "Fuggoleges tukrozes":
        return img[::-1, :]
    if name == "Gauss zaj":
        return np.clip(img + np.random.randn(*img.shape) * 0.1, 0, 1)
    if name == "SaltPepper":
        r1 = np.random.random(img.shape)
        r2 = np.random.random(img.shape)
        return np.where(r1 < 0.1, 1, np.where(r2 < 0.1, 0, img))
    if name == "Fenyero +0.3":
        return np.clip(img + 0.3, 0, 1)
    if name == "Fenyero -0.3":
        return np.clip(img - 0.3, 0, 1)
    if name == "Kontraszt 1.5x":
        return np.clip((img - 0.5) * 1.5 + 0.5, 0, 1)
    if name == "Random crop+pad":
        return np.pad(img[1:7, 1:7], ((1, 1), (1, 1)), constant_values=0)
    if name == "Kombinacio":
        return np.clip(rotate_img(img[:, ::-1], 10) + np.random.randn(*img.shape) * 0.05, 0, 1)
    return img


def build_augmentation_grid(base_img):
    names = [
        "Eredeti",
        "Forgatas +15",
        "Forgatas -15",
        "Vizszintes tukrozes",
        "Fuggoleges tukrozes",
        "Gauss zaj",
        "SaltPepper",
        "Fenyero +0.3",
        "Fenyero -0.3",
        "Kontraszt 1.5x",
        "Random crop+pad",
        "Kombinacio",
    ]
    return [(name, apply_named_augmentation(base_img, name)) for name in names]


def create_augmentation_figure(X_all):
    print("=" * 60)
    print("1. AUGMENTACIOS TECHNIKAK")
    print("=" * 60)
    img = X_all[0, 0]
    augmentations = build_augmentation_grid(img)
    fig1, axes = plt.subplots(2, 6, figsize=(18, 6))
    for ax, (name, aug_img) in zip(axes.flat, augmentations):
        ax.imshow(aug_img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(name, fontsize=9, fontweight="bold")
        ax.axis("off")
    fig1.suptitle("Data augmentation technikak - egy '0' szamjegy 12 variacioja", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("06_augmentacio_tipusok.png", dpi=150)
    print("Abra mentve: 06_augmentacio_tipusok.png")


def augment_batch(X, strength=0.15):
    X_aug = X.clone()
    X_aug += torch.randn_like(X_aug) * strength * 0.5
    shifts = torch.randint(-1, 2, (X.size(0), 2))
    for i in range(X.size(0)):
        X_aug[i] = torch.roll(torch.roll(X_aug[i], shifts[i, 0].item(), -2), shifts[i, 1].item(), -1)
    return torch.clamp(X_aug, 0, 1)


def train_with_aug(Xt, Xv, yt, yv, use_aug, epochs=300):
    torch.manual_seed(42)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accs = [], []

    for _ in range(epochs):
        model.train()
        x_batch = augment_batch(Xt) if use_aug else Xt
        loss = criterion(model(x_batch), yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        val_accs.append(acc)

    return model, train_losses, val_accs


def train_or_load_model(Xt, Xv, yt, yv, use_aug, checkpoint_dir, load_pt, save_pt, epochs):
    model = SmallCNN()
    tag = "with_aug" if use_aug else "no_aug"
    ckpt = build_checkpoint_path(checkpoint_dir, tag)
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
        model, tl, va = train_with_aug(Xt, Xv, yt, yv, use_aug, epochs=epochs)
        if save_pt:
            save_checkpoint(
                ckpt,
                model,
                tl,
                va,
                extra={"use_aug": use_aug, "epochs": epochs},
            )
    return model, tl, va


def run_training_comparison(Xt, Xv, yt, yv, checkpoint_dir=".", load_pt=True, save_pt=True, epochs=300):
    print("\n" + "=" * 60)
    print("2. AUGMENTACIO HATASA: keves adat, augmentacioval vs. nelkul")
    print("=" * 60)

    model_no, tl_no, va_no = train_or_load_model(Xt, Xv, yt, yv, False, checkpoint_dir, load_pt, save_pt, epochs)
    model_aug, tl_aug, va_aug = train_or_load_model(Xt, Xv, yt, yv, True, checkpoint_dir, load_pt, save_pt, epochs)

    print(f"  Augmentacio NELKUL: train_loss={tl_no[-1]:.3f}, val_acc={va_no[-1]*100:.1f}%")
    print(f"  Augmentacioval:     train_loss={tl_aug[-1]:.3f}, val_acc={va_aug[-1]*100:.1f}%")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    axes2[0].plot(tl_no, label="Nincs augm.", color="#E91E63", linewidth=1.5)
    axes2[0].plot(tl_aug, label="Augmentacioval", color="#4CAF50", linewidth=1.5)
    axes2[0].set_title("Tanitasi veszteseg", fontweight="bold")
    axes2[0].set_xlabel("Epoch")
    axes2[0].legend()
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(va_no, label=f"Nincs augm. ({va_no[-1]*100:.0f}%)", color="#E91E63", linewidth=1.5)
    axes2[1].plot(va_aug, label=f"Augmentacioval ({va_aug[-1]*100:.0f}%)", color="#4CAF50", linewidth=1.5)
    axes2[1].set_title("Validacios pontossag", fontweight="bold")
    axes2[1].set_xlabel("Epoch")
    axes2[1].legend()
    axes2[1].grid(True, alpha=0.3)

    fig2.suptitle(f"Data augmentation hatasa (csak {len(Xt)} tanitominta)", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("06_augmentacio_hatas.png", dpi=150)
    print("Abra mentve: 06_augmentacio_hatas.png")

    return {"no_aug": model_no, "with_aug": model_aug}


def launch_gui(Xv, yv, models):
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(3, 3, height_ratios=[8, 1, 1], width_ratios=[1, 1, 1])
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_aug = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])
    ax_slider = fig.add_subplot(gs[1, :])
    ax_radio = fig.add_subplot(gs[2, :])

    slider = Slider(ax_slider, "Minta index", 0, len(Xv) - 1, valinit=0, valstep=1)
    aug_names = [
        "Eredeti",
        "Forgatas +15",
        "Forgatas -15",
        "Vizszintes tukrozes",
        "Gauss zaj",
        "Random crop+pad",
        "Kombinacio",
    ]
    radio = RadioButtons(ax_radio, aug_names, active=0)
    ax_text.axis("off")
    txt = ax_text.text(0.0, 0.98, "", va="top", fontsize=10, family="monospace")
    state = {"aug": aug_names[0]}

    for model in models.values():
        model.eval()

    def predict(model, image_np):
        x = torch.tensor(image_np[None, None, :, :], dtype=torch.float32)
        with torch.no_grad():
            probs = F.softmax(model(x), dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
        return pred, conf

    def update(_):
        i = int(slider.val)
        orig = Xv[i, 0].detach().numpy()
        aug = apply_named_augmentation(orig, state["aug"])
        true_label = int(yv[i].item())

        ax_orig.clear()
        ax_orig.imshow(orig, cmap="gray_r", vmin=0, vmax=1)
        ax_orig.set_title("Eredeti", fontweight="bold")
        ax_orig.axis("off")

        ax_aug.clear()
        ax_aug.imshow(aug, cmap="gray_r", vmin=0, vmax=1)
        ax_aug.set_title(state["aug"], fontweight="bold")
        ax_aug.axis("off")

        no_pred, no_conf = predict(models["no_aug"], aug)
        ag_pred, ag_conf = predict(models["with_aug"], aug)
        txt.set_text(
            "\n".join(
                [
                    f"True label: {true_label}",
                    f"Tesztelt kep: {state['aug']}",
                    "",
                    f"No-aug modell : {no_pred} ({no_conf*100:5.1f}%)",
                    f"Aug modell    : {ag_pred} ({ag_conf*100:5.1f}%)",
                ]
            )
        )
        fig.canvas.draw_idle()

    def on_aug_change(label):
        state["aug"] = label
        update(None)

    slider.on_changed(update)
    radio.on_clicked(on_aug_change)
    update(None)
    fig.suptitle("Data augmentation interaktiv GUI", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Data augmentation demo checkpoint es GUI tamogatassal")
    parser.add_argument("--checkpoint-dir", default=".", help="Checkpoint konyvtar (.pt)")
    parser.add_argument("--no-load-pt", action="store_true", help="Ne toltsen be meglevo checkpointot")
    parser.add_argument("--no-save-pt", action="store_true", help="Ne mentsen checkpointot")
    parser.add_argument("--gui", action="store_true", help="GUI inditasa")
    parser.add_argument("--gui-only", action="store_true", help="Csak GUI (statikus abrak kihagyasa)")
    parser.add_argument("--epochs", type=int, default=300, help="Epochok szama a tanulasi kiserlethez")
    return parser.parse_args()


def main():
    args = parse_args()
    X_all, _y_all, Xt, Xv, yt, yv = load_data()
    load_pt = not args.no_load_pt
    save_pt = not args.no_save_pt

    models = {}
    if not args.gui_only:
        create_augmentation_figure(X_all)
        models = run_training_comparison(
            Xt,
            Xv,
            yt,
            yv,
            checkpoint_dir=args.checkpoint_dir,
            load_pt=load_pt,
            save_pt=save_pt,
            epochs=args.epochs,
        )

    if args.gui:
        if not models:
            models = {
                "no_aug": train_or_load_model(Xt, Xv, yt, yv, False, args.checkpoint_dir, load_pt, save_pt, epochs=40)[0],
                "with_aug": train_or_load_model(Xt, Xv, yt, yv, True, args.checkpoint_dir, load_pt, save_pt, epochs=40)[0],
            }
        launch_gui(Xv, yv, models)

    plt.close("all")
    print("\nKesz!")


if __name__ == "__main__":
    main()
