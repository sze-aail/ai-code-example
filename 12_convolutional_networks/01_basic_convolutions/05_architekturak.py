"""
05_architekturak.py — CNN architektúrák és skip connection
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - ResNet skip connection implementáció és hatása
  - Architektúrák paraméterszám összehasonlítása
  - Mélység hatása skip connection nélkül vs. skip-pel
  - Receptive field számítás különböző architektúráknál
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
X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
y = torch.tensor(digits.target, dtype=torch.long)
Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.numpy())
Xt, Xv = torch.tensor(np.array(Xt)), torch.tensor(np.array(Xv))
yt, yv = torch.tensor(np.array(yt)), torch.tensor(np.array(yv))


# ══════════════════════════════════════════════════════════════
# ÉPÍTŐELEMEK
# ══════════════════════════════════════════════════════════════

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
        return F.relu(out + residual)  # ← SKIP CONNECTION


class PlainBlock(nn.Module):
    """Ugyanaz skip nélkül."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out)  # NINCS skip


class CNN_Configurable(nn.Module):
    def __init__(self, n_blocks, use_skip=True, channels=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU())
        Block = ResidualBlock if use_skip else PlainBlock
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def train_eval(model, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
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


# ══════════════════════════════════════════════════════════════
# 1. SKIP CONNECTION HATÁSA
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("1. SKIP CONNECTION HATÁSA: Plain vs. ResNet")
print("=" * 60)

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))

depths = [2, 4, 6]
for ax, n_blocks in zip(axes1, depths):
    results = {}
    for use_skip, name, color in [(False, "Plain", "#E91E63"), (True, "ResNet", "#4CAF50")]:
        torch.manual_seed(42)
        model = CNN_Configurable(n_blocks, use_skip=use_skip)
        n_params = sum(p.numel() for p in model.parameters())
        tl, va = train_eval(model, epochs=50)
        results[name] = (tl, va)
        print(f"  {n_blocks} blokk, {name}: acc={va[-1]*100:.1f}%, params={n_params}")
        ax.plot(va, label=f"{name} ({va[-1]*100:.0f}%)", color=color, linewidth=2)

    ax.set_title(f"{n_blocks} blokk ({n_blocks*4+2} conv réteg)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val accuracy")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_ylim(0.5, 1.02)

fig1.suptitle("Skip connection hatása: mélyebb hálónál kritikus a különbség",
              fontsize=14, fontweight="bold")
fig1.tight_layout()
fig1.savefig("05_skip_connection.png", dpi=150)
print("\nÁbra mentve: 05_skip_connection.png")


# ══════════════════════════════════════════════════════════════
# 2. ARCHITEKTÚRÁK PARAMÉTERSZÁM ÖSSZEHASONLÍTÁS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. ARCHITEKTÚRÁK ÖSSZEHASONLÍTÁSA")
print("=" * 60)

fig2, ax2 = plt.subplots(figsize=(12, 6))

# Híres architektúrák (ImageNet adatok)
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
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(archs)))
scatter = ax2.scatter(params, accs, c=years, s=200, cmap="RdYlGn", edgecolors="k", linewidth=1, zorder=3)
for i, name in enumerate(names):
    ax2.annotate(name, (params[i], accs[i]), fontsize=8, ha="center",
                 xytext=(0, 15), textcoords="offset points", fontweight="bold")

plt.colorbar(scatter, ax=ax2, label="Év")
ax2.set_xlabel("Paraméterek (millió)", fontsize=12)
ax2.set_ylabel("ImageNet top-5 accuracy (%)", fontsize=12)
ax2.set_title("CNN architektúrák: paraméterek vs. pontosság", fontsize=14, fontweight="bold")
ax2.set_xscale("log")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("05_architekturak.png", dpi=150)
print("Ábra mentve: 05_architekturak.png")


# ══════════════════════════════════════════════════════════════
# 3. RECEPTIVE FIELD SZÁMÍTÁS
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. RECEPTIVE FIELD SZÁMÍTÁS")
print("=" * 60)

def calc_receptive_field(layers):
    """layers: list of (kernel_size, stride, dilation)"""
    rf = 1
    stride_prod = 1
    for k, s, d in layers:
        rf = rf + (k - 1) * d * stride_prod
        stride_prod *= s
    return rf

architectures_rf = {
    "LeNet-5": [(5,1,1), (2,2,1), (5,1,1), (2,2,1)],
    "VGG-16 (első 5)": [(3,1,1)]*2 + [(2,2,1)] + [(3,1,1)]*2 + [(2,2,1)],
    "ResNet (6 blokk)": [(3,1,1)]*12,
    "Dilated (d=1,2,4,8)": [(3,1,1), (3,1,2), (3,1,4), (3,1,8)],
}

fig3, ax3 = plt.subplots(figsize=(10, 5))
for name, layers in architectures_rf.items():
    rfs = [1]
    for i in range(1, len(layers)+1):
        rfs.append(calc_receptive_field(layers[:i]))
    ax3.plot(range(len(rfs)), rfs, "o-", linewidth=2, markersize=6, label=f"{name} (RF={rfs[-1]})")
    print(f"  {name}: RF = {rfs[-1]}")

ax3.set_xlabel("Réteg sorszáma", fontsize=12)
ax3.set_ylabel("Receptive field (pixel)", fontsize=12)
ax3.set_title("Receptive field növekedés rétegenként", fontsize=13, fontweight="bold")
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("05_receptive_field.png", dpi=150)
print("Ábra mentve: 05_receptive_field.png")

plt.close("all")
print("\nKész!")
