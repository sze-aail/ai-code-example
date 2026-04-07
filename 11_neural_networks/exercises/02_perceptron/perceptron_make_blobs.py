"""
perceptron_make_blobs.py — Perceptron nagy adathalmazon (make_blobs)
Forrás: mesterseges-intelligencia-kodok (01_perceptrons), átdolgozva

Demonstrálja:
  - Perceptron sklearn generált adaton
  - 3D döntési határ vizualizáció
  - Konvergencia 1000 mintán
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from mpl_toolkits.mplot3d import Axes3D


# ── Adatgenerálás ──
N = 1000
X, y = make_blobs(n_samples=N, centers=2, n_features=2, random_state=988)
y[y == 0] = -1  # {0,1} → {-1,1}

# ── Aktivációs függvények ──
def unit_step(x):
    return np.where(x > 0, 1, -1)

# ── Perceptron ──
def perceptron(X, y, lr=0.001, max_iter=1000, eps=1e-3):
    theta = np.zeros(X.shape[1] + 1)
    X_ = np.hstack((np.ones((X.shape[0], 1)), X))  # bias oszlop
    errors_history = []

    for it in range(max_iter):
        pred = unit_step(X_ @ theta)
        errors = np.sum(pred != y)
        errors_history.append(errors)

        if errors == 0:
            print(f"Konvergált {it+1} iteráció után!")
            break

        # Frissítés hibás mintákon
        misclassified = pred != y
        theta += lr * (X_[misclassified].T @ (y[misclassified] - pred[misclassified]))

    return theta, errors_history

theta, hist = perceptron(X, y)
print(f"Súlyvektor: θ = {theta}")

# ── 1. ábra: 2D döntési határ ──
fig1, ax1 = plt.subplots(figsize=(8, 6))
colors = ["#2196F3" if yi == -1 else "#4CAF50" for yi in y]
ax1.scatter(X[:, 0], X[:, 1], c=colors, s=15, alpha=0.6, edgecolors="none")

# Döntési vonal: theta[0] + theta[1]*x1 + theta[2]*x2 = 0
x1_range = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
if abs(theta[2]) > 1e-8:
    x2_line = -(theta[0] + theta[1] * x1_range) / theta[2]
    ax1.plot(x1_range, x2_line, "r-", linewidth=2, label="Döntési határ")
    ax1.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

ax1.set_xlabel("x₁", fontsize=12)
ax1.set_ylabel("x₂", fontsize=12)
ax1.set_title(f"Perceptron — {N} minta, make_blobs", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("perceptron_blobs_2d.png", dpi=150)
print("Ábra mentve: perceptron_blobs_2d.png")

# ── 2. ábra: 3D döntési sík ──
fig2 = plt.figure(figsize=(10, 7))
ax3d = fig2.add_subplot(111, projection="3d")

# Adatpontok z=0 síkon
ax3d.scatter(X[y == -1, 0], X[y == -1, 1], zs=0, s=10, c="#2196F3", alpha=0.4, label="Osztály -1")
ax3d.scatter(X[y == 1, 0], X[y == 1, 1], zs=0, s=10, c="#4CAF50", alpha=0.4, label="Osztály +1")

# Döntési sík
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 30),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 30),
)
zz = theta[0] + theta[1] * xx + theta[2] * yy
ax3d.plot_surface(xx, yy, zz, alpha=0.3, color="red")
ax3d.set_xlabel("x₁")
ax3d.set_ylabel("x₂")
ax3d.set_zlabel("θ₀ + θ₁x₁ + θ₂x₂")
ax3d.set_title("Perceptron — 3D döntési sík", fontsize=13, fontweight="bold")
ax3d.legend()
fig2.tight_layout()
fig2.savefig("perceptron_blobs_3d.png", dpi=150)
print("Ábra mentve: perceptron_blobs_3d.png")

# ── 3. ábra: Konvergencia ──
fig3, ax3 = plt.subplots(figsize=(7, 4))
ax3.plot(hist, color="#E91E63", linewidth=1.5)
ax3.set_xlabel("Iteráció")
ax3.set_ylabel("Hibásan osztályozott minták")
ax3.set_title("Perceptron konvergencia", fontweight="bold")
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("perceptron_blobs_convergence.png", dpi=150)
print("Ábra mentve: perceptron_blobs_convergence.png")

plt.close("all")
print("Kész!")
