"""
02_xor_mlp.py — XOR probléma megoldása többrétegű neurális hálóval
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Miért kell rejtett réteg az XOR-hoz
  - MLP felépítése PyTorch-ban
  - Tanítási ciklus (forward → loss → backward → step)
  - Döntési felület vizualizáció
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class XOR_MLP(nn.Module):
    """Kétrétegű MLP az XOR probléma megoldásához."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),   # rejtett réteg: 4 neuron
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Adatok ──
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# ── Modell, veszteségfüggvény, optimalizáló ──
model = XOR_MLP()
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# ── Tanítás ──
losses = []
print("Tanítás indul...")
for epoch in range(2000):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 500 == 0:
        print(f"  Epoch {epoch+1:4d} | Loss: {loss.item():.6f}")

# ── Eredmények ──
print("\nEredmények:")
with torch.no_grad():
    preds = model(X)
    for xi, yi, pi in zip(X, y, preds):
        label = 1 if pi.item() > 0.5 else 0
        print(f"  {xi.numpy()} -> pred={pi.item():.4f} (osztály={label}), elvárt={int(yi.item())}")

# ── 1. ábra: Tanulási görbe ──
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(losses, color="#E91E63", linewidth=1.5)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("BCE Loss")
ax1.set_title("XOR MLP — Tanulási görbe")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("02_xor_loss.png", dpi=150)
print("\nÁbra mentve: 02_xor_loss.png")

# ── 2. ábra: Döntési felület ──
fig2, ax2 = plt.subplots(figsize=(6, 5))
xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    zz = model(grid).numpy().reshape(xx.shape)

contour = ax2.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.8)
plt.colorbar(contour, ax=ax2, label="P(y=1)")
ax2.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)

colors = ["#2196F3" if yi == 0 else "#4CAF50" for yi in y.numpy().flatten()]
ax2.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=colors, s=150, edgecolors="k", zorder=3)
ax2.set_xlabel("x₁")
ax2.set_ylabel("x₂")
ax2.set_title("XOR MLP — Döntési felület")
fig2.tight_layout()
fig2.savefig("02_xor_decision.png", dpi=150)
print("Ábra mentve: 02_xor_decision.png")

# ── 3. ábra: Rejtett réteg aktivációk ──
fig3, axes = plt.subplots(1, 2, figsize=(10, 4))

# Az első lineáris réteg kimenetének vizualizációja
hidden_layer = model.net[:2]  # Linear + ReLU
with torch.no_grad():
    h = hidden_layer(grid).numpy()

for idx, ax in enumerate(axes):
    zz_h = h[:, idx].reshape(xx.shape)
    c = ax.contourf(xx, yy, zz_h, levels=30, cmap="viridis", alpha=0.8)
    plt.colorbar(c, ax=ax)
    ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=colors, s=100, edgecolors="k", zorder=3)
    ax.set_title(f"Rejtett neuron {idx + 1} aktivációja")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

fig3.suptitle("Rejtett réteg reprezentáció — a háló 'széthajtja' a teret", fontsize=12)
fig3.tight_layout()
fig3.savefig("02_xor_hidden.png", dpi=150)
print("Ábra mentve: 02_xor_hidden.png")

plt.close("all")
print("\nKész!")
