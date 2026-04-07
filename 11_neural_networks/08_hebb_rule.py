"""
08_hebb_rule.py — Hebb-szabály demonstráció
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Hebb-tanulás: Δw = η · x · y
  - Oja-szabály: normalizált Hebb (megakadályozza a súlyok divergenciáját)
  - Főkomponens kinyerése (PCA-szerű) felügyeletlen tanulással
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(42)

    # ── 2D korrelált adat generálása ──
    n = 300
    angle = np.pi / 6  # 30 fokos főtengely
    cov = np.array([[2.0, 1.2], [1.2, 0.8]])
    mean = [0, 0]
    data = np.random.multivariate_normal(mean, cov, n)

    # ── Hebb-tanulás (naiv) ──
    w_hebb = np.random.randn(2) * 0.1
    lr = 0.001
    w_hebb_history = [w_hebb.copy()]

    for xi in data:
        y = np.dot(w_hebb, xi)
        w_hebb += lr * y * xi  # Δw = η · y · x
        w_hebb_history.append(w_hebb.copy())

    print("Hebb (naiv) végső súly:", w_hebb)
    print("  FIGYELEM: a súlyok divergálnak (nincs normalizáció)!")

    # ── Oja-szabály (normalizált Hebb) ──
    w_oja = np.random.randn(2) * 0.1
    w_oja = w_oja / np.linalg.norm(w_oja)
    lr_oja = 0.01
    w_oja_history = [w_oja.copy()]

    for epoch in range(10):
        for xi in data:
            y = np.dot(w_oja, xi)
            w_oja += lr_oja * y * (xi - y * w_oja)  # Oja: Δw = η·y·(x - y·w)
        w_oja = w_oja / np.linalg.norm(w_oja)
        w_oja_history.append(w_oja.copy())

    print(f"\nOja végső irány: {w_oja}")

    # Valódi PC1 (eigenvektor)
    eigvals, eigvecs = np.linalg.eigh(np.cov(data.T))
    pc1 = eigvecs[:, np.argmax(eigvals)]
    if np.dot(pc1, w_oja) < 0:
        pc1 = -pc1
    print(f"Valódi PC1:      {pc1}")
    print(f"Szögeltérés:     {np.degrees(np.arccos(np.clip(np.dot(pc1, w_oja), -1, 1))):.2f}°")

    # ── Ábra ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Bal: Oja-szabály eredménye
    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], s=8, alpha=0.3, color="#90CAF9")
    scale = 4
    ax.arrow(0, 0, w_oja[0]*scale, w_oja[1]*scale, head_width=0.15, head_length=0.1,
             fc="#E91E63", ec="#E91E63", linewidth=2.5, label="Oja-tanult irány")
    ax.arrow(0, 0, pc1[0]*scale, pc1[1]*scale, head_width=0.15, head_length=0.1,
             fc="#4CAF50", ec="#4CAF50", linewidth=2, linestyle="--", label="Valódi PC1")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    ax.set_aspect("equal")
    ax.set_title("Oja-szabály: főkomponens kinyerése", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Jobb: Hebb naiv súlyok divergenciája
    ax2 = axes[1]
    wh = np.array(w_hebb_history)
    norm_h = np.linalg.norm(wh, axis=1)
    ax2.plot(norm_h, color="#E91E63", linewidth=2)
    ax2.set_xlabel("Lépés", fontsize=11)
    ax2.set_ylabel("||w|| (súlyvektor normája)", fontsize=11)
    ax2.set_title("Naiv Hebb-tanulás: súlyok divergálnak!", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Hebb-szabály és Oja-szabály összehasonlítása", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("08_hebb_oja.png", dpi=150)
    print("\nÁbra mentve: 08_hebb_oja.png")

    plt.close("all")
    print("Kész!")

if __name__ == "__main__":
    main()
