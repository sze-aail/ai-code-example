"""
06_data_augmentation.py — Data augmentation vizualizáció
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - Képaugmentációs technikák (forgatás, tükrözés, crop, szín, zaj)
  - PyTorch transforms használata
  - Augmentáció hatása a tanulásra (ugyanaz a háló, augmentációval vs. nélkül)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ── Adat ──
digits = load_digits()
X_all = digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0
y_all = digits.target

# ══════════════════════════════════════════════════════════════
# 1. AUGMENTÁCIÓS TECHNIKÁK VIZUALIZÁCIÓ
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("1. AUGMENTÁCIÓS TECHNIKÁK")
print("=" * 60)

img = X_all[0, 0]  # egy '0' számjegy

def rotate_img(img, angle_deg):
    rad = np.radians(angle_deg)
    h, w = img.shape
    cy, cx = h/2, w/2
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

augmentations = [
    ("Eredeti", img),
    ("Forgatás 15°", rotate_img(img, 15)),
    ("Forgatás -15°", rotate_img(img, -15)),
    ("Vízszintes tükrözés", img[:, ::-1]),
    ("Függőleges tükrözés", img[::-1, :]),
    ("Gauss zaj (σ=0.1)", np.clip(img + np.random.randn(*img.shape) * 0.1, 0, 1)),
    ("Salt & pepper", (lambda i: (np.where(np.random.random(i.shape) < 0.1, 1, np.where(np.random.random(i.shape) < 0.1, 0, i))))(img)),
    ("Fényerő +0.3", np.clip(img + 0.3, 0, 1)),
    ("Fényerő -0.3", np.clip(img - 0.3, 0, 1)),
    ("Kontraszt 1.5×", np.clip((img - 0.5) * 1.5 + 0.5, 0, 1)),
    ("Random crop+pad", np.pad(img[1:7, 1:7], ((1,1),(1,1)), constant_values=0)),
    ("Kombináció", np.clip(rotate_img(img[:, ::-1], 10) + np.random.randn(*img.shape) * 0.05, 0, 1)),
]

fig1, axes = plt.subplots(2, 6, figsize=(18, 6))
for ax, (name, aug_img) in zip(axes.flat, augmentations):
    ax.imshow(aug_img, cmap="gray_r", vmin=0, vmax=1)
    ax.set_title(name, fontsize=9, fontweight="bold")
    ax.axis("off")

fig1.suptitle("Data augmentation technikák — egy '0' számjegy 12 variációja",
              fontsize=14, fontweight="bold")
fig1.tight_layout()
fig1.savefig("06_augmentacio_tipusok.png", dpi=150)
print("Ábra mentve: 06_augmentacio_tipusok.png")


# ══════════════════════════════════════════════════════════════
# 2. AUGMENTÁCIÓ HATÁSA A TANULÁSRA
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. AUGMENTÁCIÓ HATÁSA: kevés adat, augmentációval vs. nélkül")
print("=" * 60)

# Szándékosan KEVÉS tanítóadat (100 minta)
X_train_small, X_test, y_train_small, y_test = train_test_split(
    X_all, y_all, train_size=100, random_state=42, stratify=y_all)

Xt = torch.tensor(X_train_small); yt = torch.tensor(y_train_small, dtype=torch.long)
Xv = torch.tensor(X_test); yv = torch.tensor(y_test, dtype=torch.long)

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


def augment_batch(X, strength=0.15):
    """Egyszerű online augmentáció: zaj + kis eltolás."""
    X_aug = X.clone()
    # Gauss zaj
    X_aug += torch.randn_like(X_aug) * strength * 0.5
    # Random eltolás (1 pixel)
    shifts = torch.randint(-1, 2, (X.size(0), 2))
    for i in range(X.size(0)):
        X_aug[i] = torch.roll(torch.roll(X_aug[i], shifts[i, 0].item(), -2), shifts[i, 1].item(), -1)
    return torch.clamp(X_aug, 0, 1)


def train_with_aug(use_aug, epochs=300):
    torch.manual_seed(42)
    model = SmallCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        x_batch = augment_batch(Xt) if use_aug else Xt
        loss = criterion(model(x_batch), yt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        val_accs.append(acc)

    return train_losses, val_accs


tl_no, va_no = train_with_aug(False)
tl_aug, va_aug = train_with_aug(True)

print(f"  Augmentáció NÉLKÜL: train_loss={tl_no[-1]:.3f}, val_acc={va_no[-1]*100:.1f}%")
print(f"  Augmentációval:     train_loss={tl_aug[-1]:.3f}, val_acc={va_aug[-1]*100:.1f}%")

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

axes2[0].plot(tl_no, label="Nincs augm.", color="#E91E63", linewidth=1.5)
axes2[0].plot(tl_aug, label="Augmentációval", color="#4CAF50", linewidth=1.5)
axes2[0].set_title("Tanítási veszteség", fontweight="bold")
axes2[0].set_xlabel("Epoch"); axes2[0].legend(); axes2[0].grid(True, alpha=0.3)

axes2[1].plot(va_no, label=f"Nincs augm. ({va_no[-1]*100:.0f}%)", color="#E91E63", linewidth=1.5)
axes2[1].plot(va_aug, label=f"Augmentációval ({va_aug[-1]*100:.0f}%)", color="#4CAF50", linewidth=1.5)
axes2[1].set_title("Validációs pontosság", fontweight="bold")
axes2[1].set_xlabel("Epoch"); axes2[1].legend(); axes2[1].grid(True, alpha=0.3)

fig2.suptitle(f"Data augmentation hatása (csak {len(Xt)} tanítóminta!)",
              fontsize=14, fontweight="bold")
fig2.tight_layout()
fig2.savefig("06_augmentacio_hatas.png", dpi=150)
print("Ábra mentve: 06_augmentacio_hatas.png")

plt.close("all")
print("\nKész!")
