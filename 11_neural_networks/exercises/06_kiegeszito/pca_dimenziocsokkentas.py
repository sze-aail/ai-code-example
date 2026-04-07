"""
pca_dimenziocsokkentas.py — Főkomponens-analízis (PCA) szemléltetés
Forrás: edu-machine-learning (03_SVM/PCA), átdolgozva

Demonstrálja:
  - PCA dimenziócsökkentés Iris adathalmazon (4D → 2D)
  - Megőrzött variancia hányad (explained variance ratio)
  - Kapcsolat a Hebb/Oja-szabállyal
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


# ── Adat ──
X, y = load_iris(return_X_y=True)
target_names = load_iris().target_names
print(f"Eredeti dimenzió: {X.shape[1]}")

# ── PCA: 4D → 2D ──
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print(f"\nMegőrzött variancia:")
for i, (ratio, comp) in enumerate(zip(pca.explained_variance_ratio_, pca.components_)):
    print(f"  PC{i+1}: {ratio*100:.1f}%  irány = {comp}")
print(f"  Összesen: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ── 1. ábra: PCA projekció ──
fig1, ax1 = plt.subplots(figsize=(8, 6))
colors = ["#2196F3", "#4CAF50", "#E91E63"]
for i, (name, color) in enumerate(zip(target_names, colors)):
    mask = y == i
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=name, s=40, edgecolors="k", linewidth=0.5, alpha=0.7)

ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=12)
ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=12)
ax1.set_title("PCA — Iris adathalmaz (4D → 2D)", fontsize=13, fontweight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("pca_iris_2d.png", dpi=150)
print("\nÁbra mentve: pca_iris_2d.png")

# ── 2. ábra: Scree plot (variancia megőrzés) ──
pca_full = PCA().fit(X)

fig2, ax2 = plt.subplots(figsize=(7, 5))
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
ax2.bar(range(1, 5), pca_full.explained_variance_ratio_, color="#2196F3", alpha=0.7, label="Egyéni")
ax2.step(range(1, 5), cumvar, where="mid", color="#E91E63", linewidth=2, label="Kumulatív")
ax2.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95% küszöb")
ax2.set_xlabel("Főkomponens", fontsize=12)
ax2.set_ylabel("Megőrzött variancia hányad", fontsize=12)
ax2.set_title("Scree plot — hány komponens elég?", fontsize=13, fontweight="bold")
ax2.set_xticks(range(1, 5))
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis="y")
fig2.tight_layout()
fig2.savefig("pca_scree.png", dpi=150)
print("Ábra mentve: pca_scree.png")

plt.close("all")
print("Kész!")
