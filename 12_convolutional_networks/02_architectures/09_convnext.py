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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ── Adat ──
digits = load_digits()
X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
y = torch.tensor(digits.target, dtype=torch.long)
Xt, Xv, yt, yv = [torch.tensor(np.array(a)) for a in train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.numpy())]


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


def train_eval(model, epochs=80):
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


# ══════════════════════════════════════════════════════════════
# KÍSÉRLET: lépésenkénti modernizáció
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("ResNet → ConvNeXt: LÉPÉSENKÉNTI MODERNIZÁCIÓ")
print("=" * 60)

blocks = [
    ("0. ResNet\n(baseline)", ResNetBlock),
    ("1. + Depthwise\nSeparable", Step1_DepthwiseBlock),
    ("2. + 7×7 kernel\n(nagyobb RF)", Step2_LargerKernelBlock),
    ("3. + Inverted\nBottleneck", Step3_InvertedBottleneck),
    ("4. + LayerNorm\n+ GELU (ConvNeXt)", ConvNeXtBlock),
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
results = []

for name, block_cls in blocks:
    torch.manual_seed(42)
    model = TestNet(block_cls)
    n_params = sum(p.numel() for p in model.parameters())
    va = train_eval(model, epochs=80)
    results.append((name, va[-1], n_params))
    axes[0].plot(va, linewidth=2, label=f"{name.split(chr(10))[0]} ({va[-1]*100:.0f}%)")
    print(f"  {name.split(chr(10))[0]:25s}: acc={va[-1]*100:.1f}%, params={n_params}")

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val accuracy")
axes[0].set_title("Lépésenkénti modernizáció hatása", fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

# Parameterszám vs pontosság
names, accs, params = zip(*results)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))
axes[1].barh(range(len(results)), [a*100 for a in accs], color=colors, edgecolor="k")
axes[1].set_yticks(range(len(results)))
axes[1].set_yticklabels([n.replace('\n', ' ') for n in names], fontsize=9)
axes[1].set_xlabel("Val accuracy (%)")
axes[1].set_title("Végső pontosság", fontweight="bold")
for i, (a, p) in enumerate(zip(accs, params)):
    axes[1].text(a*100 + 0.3, i, f"{a*100:.1f}% ({p}p)", va="center", fontsize=9)
axes[1].grid(True, alpha=0.3, axis="x")

fig.suptitle("ResNet → ConvNeXt: 5 modernizációs lépés (Liu et al., 2022)",
             fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig("09_convnext_lepesek.png", dpi=150)
print("\nÁbra mentve: 09_convnext_lepesek.png")


# ══════════════════════════════════════════════════════════════
# ÖSSZEFOGLALÓ: mit vett át a ConvNeXt a Transformertől
# ══════════════════════════════════════════════════════════════

fig2, ax2 = plt.subplots(figsize=(14, 5))
ax2.axis("off")
data = [
    ["Elem", "ResNet (2015)", "Swin Transformer (2021)", "ConvNeXt (2022)"],
    ["Kernel méret", "3×3", "7×7 ablak (attention)", "7×7 depthwise conv"],
    ["Normalizáció", "BatchNorm", "LayerNorm", "LayerNorm"],
    ["Aktiváció", "ReLU (minden conv után)", "GELU (1× per blokk)", "GELU (1× per blokk)"],
    ["Bottleneck", "dim → dim/4 → dim\n(standard)", "—", "dim → 4×dim → dim\n(inverted!)"],
    ["Stem", "7×7 conv, stride=2\n+ MaxPool", "4×4 patch embed\nstride=4", "4×4 conv,\nstride=4"],
    ["ImageNet\ntop-1", "76.1% (ResNet-50)", "83.5% (Swin-T)", "82.1% (ConvNeXt-T)\n→ CNN versenyképes!"],
]
table = ax2.table(cellText=data, loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1.0, 2.2)
for j in range(4):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(data)):
    table[i, 0].set_facecolor("#ECEFF1")
    table[i, 0].set_text_props(fontweight="bold")
ax2.set_title("ConvNeXt: Transformer trükkök CNN-ben — a CNN nem halott!",
              fontsize=14, fontweight="bold", pad=20)
fig2.tight_layout()
fig2.savefig("09_convnext_vs_swin.png", dpi=150)
print("Ábra mentve: 09_convnext_vs_swin.png")

plt.close("all")
print("\nKész!")
