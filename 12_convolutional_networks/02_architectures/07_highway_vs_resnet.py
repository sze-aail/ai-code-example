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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ── Adat ──
digits = load_digits()
X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
y = torch.tensor(digits.target, dtype=torch.long)
Xt, Xv, yt, yv = [torch.tensor(np.array(a)) for a in train_test_split(X, y, test_size=0.2, random_state=42, stratify=y.numpy())]


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


def train_model(model, epochs=100, lr=0.005):
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


# ══════════════════════════════════════════════════════════════
# 1. KÍSÉRLET: Highway vs. ResNet
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("1. HIGHWAY NETWORK vs. RESNET")
print("=" * 60)

fig1, axes1 = plt.subplots(1, 3, figsize=(17, 5))

for n_blocks in [3, 6]:
    for btype, color, ls in [('resnet', '#4CAF50', '-'), ('highway', '#E91E63', '--')]:
        torch.manual_seed(42)
        model = ConfigurableNet(n_blocks, btype)
        n_params = sum(p.numel() for p in model.parameters())
        tl, va = train_model(model, epochs=50)
        axes1[0].plot(tl, color=color, linestyle=ls, linewidth=1.5,
                      label=f"{btype} {n_blocks}b ({va[-1]*100:.0f}%)")
        axes1[1].plot(va, color=color, linestyle=ls, linewidth=1.5,
                      label=f"{btype} {n_blocks}b ({n_params}p)")
        print(f"  {btype:8s} {n_blocks} blokk: acc={va[-1]*100:.1f}%, params={n_params}")

axes1[0].set_title("Tanítási veszteség", fontweight="bold")
axes1[0].set_xlabel("Epoch"); axes1[0].legend(fontsize=8); axes1[0].grid(True, alpha=0.3)
axes1[1].set_title("Validációs pontosság", fontweight="bold")
axes1[1].set_xlabel("Epoch"); axes1[1].legend(fontsize=8); axes1[1].grid(True, alpha=0.3)

# 2. Kapu-értékek vizualizáció
torch.manual_seed(42)
hw_model = ConfigurableNet(6, 'highway')
train_model(hw_model, epochs=80)
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

# ══════════════════════════════════════════════════════════════
# 2. ÖSSZEFOGLALÓ: SKIP-VARIÁNSOK
# ══════════════════════════════════════════════════════════════

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

plt.close("all")
print("\nKész!")
