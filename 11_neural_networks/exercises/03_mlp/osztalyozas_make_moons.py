"""
osztályozas_make_moons.py — MLP osztályozás félhold adathalmazon
Forrás: edu-machine-learning (03_MLP_Classification_SimpleDataset), átdolgozva

Demonstrálja:
  - Nemlineáris döntési határ tanulása
  - Rejtett réteg méretének hatása
  - Train/test split és generalizáció
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


# ── Adat ──
X, y = make_moons(n_samples=1000, random_state=42, noise=0.15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_test, dtype=torch.float32)
yv = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


class MoonMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ── Különböző kapacitások ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
hidden_sizes = [2, 16, 128]

xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

for ax, h in zip(axes, hidden_sizes):
    torch.manual_seed(42)
    model = MoonMLP(hidden=h)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    for epoch in range(500):
        pred = model(Xt)
        loss = criterion(pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        zz = model(grid).numpy().reshape(xx.shape)
        test_pred = (model(Xv) > 0.5).float()
        acc = (test_pred == yv).float().mean().item()

    ax.contourf(xx, yy, zz, levels=50, cmap="RdYlBu", alpha=0.7)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    colors_test = ["#2196F3" if yi == 0 else "#E91E63" for yi in y_test]
    ax.scatter(X_test[:, 0], X_test[:, 1], c=colors_test, s=20, edgecolors="k", linewidth=0.5, zorder=3)
    ax.set_title(f"Rejtett: {h} neuron\nTest acc: {acc*100:.1f}%", fontsize=12, fontweight="bold")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

fig.suptitle("MLP kapacitás hatása — make_moons adathalmaz", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("make_moons_kapacitas.png", dpi=150)
print(f"Ábra mentve: make_moons_kapacitas.png")

plt.close("all")
print("Kész!")
