"""
veszteségfuggvenyek_3d.py — Veszteségfüggvények 3D vizualizáció
Forrás: edu-machine-learning (00_Alapok), átdolgozva

Demonstrálja:
  - MSE, Huber, Hinge loss felületek 3D-ben
  - Cross-Entropy loss
  - A különböző hibamértékek viselkedése outlierek esetén
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
xv, yv = np.meshgrid(x, y, sparse=False, indexing="ij")

# ── Veszteségfüggvények ──
def mse(X, Y):
    return (X - Y) ** 2

def mae(X, Y):
    return np.abs(X - Y)

def huber(X, Y, delta=0.3):
    diff = np.abs(Y - X)
    return np.where(diff <= delta, diff**2 / 2, delta * (diff - delta / 2))

def binary_crossentropy(X, Y, eps=1e-7):
    X_clip = np.clip(X, eps, 1 - eps)
    return -(Y * np.log(X_clip) + (1 - Y) * np.log(1 - X_clip))

losses = [
    ("MSE (négyzetes hiba)", mse(xv, yv)),
    ("MAE (abszolút hiba)", mae(xv, yv)),
    ("Huber (δ=0.3)", huber(xv, yv, delta=0.3)),
    ("Binary Cross-Entropy", binary_crossentropy(xv, yv)),
]

fig = plt.figure(figsize=(16, 12))
for i, (name, Z) in enumerate(losses):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    ax.plot_surface(xv, yv, Z, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.set_xlabel("Predikció (ŷ)")
    ax.set_ylabel("Valódi (y)")
    ax.set_zlabel("Loss")
    ax.set_title(name, fontsize=13, fontweight="bold")
    ax.view_init(elev=25, azim=135)

fig.suptitle("Veszteségfüggvények összehasonlítása", fontsize=16, fontweight="bold")
fig.tight_layout()
fig.savefig("vesztesgfuggvenyek_3d.png", dpi=150)
print("Ábra mentve: vesztesgfuggvenyek_3d.png")

# ── 1D összehasonlítás: outlier-érzékenység ──
fig2, ax2 = plt.subplots(figsize=(8, 5))
err = np.linspace(-2, 2, 300)
ax2.plot(err, err**2, label="MSE", linewidth=2, color="#E91E63")
ax2.plot(err, np.abs(err), label="MAE", linewidth=2, color="#2196F3")
delta = 0.5
huber_1d = np.where(np.abs(err) <= delta, err**2 / 2, delta * (np.abs(err) - delta / 2))
ax2.plot(err, huber_1d, label=f"Huber (δ={delta})", linewidth=2, color="#4CAF50")
ax2.set_xlabel("Hiba (y - ŷ)", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Outlier-érzékenység: MSE vs. MAE vs. Huber", fontsize=13, fontweight="bold")
ax2.legend(fontsize=11)
ax2.set_ylim(0, 3)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("vesztesgfuggvenyek_1d.png", dpi=150)
print("Ábra mentve: vesztesgfuggvenyek_1d.png")

plt.close("all")
print("Kész!")
