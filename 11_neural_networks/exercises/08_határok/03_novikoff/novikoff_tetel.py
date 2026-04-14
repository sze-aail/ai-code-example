"""
novikoff_tetel.py — Perceptron konvergencia-tétel (Novikoff, 1962)
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Felső korlát: a perceptron legfeljebb (R/γ)² frissítést végez
  - R = max ||xᵢ|| (adatpontok sugara), γ = margin (minimális távolság a döntési síktól)
  - Empirikus vs. elméleti korlát összehasonlítása
  - Margin hatása a konvergencia sebességére
"""
import numpy as np
import matplotlib.pyplot as plt


# ── Perceptron frissítés-számlálóval ──
def perceptron_train(X, y, max_iter=50000):
    """Perceptron tanítás, visszaadja a frissítések számát."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    updates = 0

    for _ in range(max_iter):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) <= 0:
                w += yi * xi
                b += yi
                updates += 1
                errors += 1
        if errors == 0:
            return updates, w, b
    return updates, w, b


def compute_margin_and_radius(X, y):
    """Kiszámítja az optimális margint (SVM-mel) és az adatsugarat."""
    from sklearn.svm import SVC
    svm = SVC(kernel='linear', C=1e6)
    svm.fit(X, y)
    w_opt = svm.coef_[0]
    b_opt = svm.intercept_[0]
    norm_w = np.linalg.norm(w_opt)

    # Margin: a legközelebbi pont távolsága a döntési síktól
    distances = np.abs(X @ w_opt + b_opt) / norm_w
    gamma = distances.min()

    # Sugár: max ||x||
    R = np.max(np.linalg.norm(X, axis=1))

    return R, gamma, w_opt / norm_w, b_opt / norm_w


def generate_separable_data(n=100, margin=1.0, dim=2, seed=42):
    """Lineárisan szeparálható adat adott marginnal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim)
    # Vetítés az első tengelyre, margin biztosítása
    y = np.sign(X[:, 0])
    # Tooljuk szét a pontokat a marginnál
    X[:, 0] += y * margin / 2
    # Szűrjük a marginon belülieket
    mask = np.abs(X[:, 0]) > margin / 4
    return X[mask], y[mask].astype(int)


def main():
    # ── 1. kísérlet: Margin hatása ──
    print("=" * 60)
    print("1. KÍSÉRLET: Margin (γ) hatása a konvergenciára")
    print("=" * 60)

    margins = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    results = []

    for m in margins:
        X, y = generate_separable_data(n=200, margin=m, seed=42)
        R, gamma, _, _ = compute_margin_and_radius(X, y)
        updates, _, _ = perceptron_train(X, y)
        bound = (R / gamma) ** 2
        results.append((m, R, gamma, updates, bound))
        print(f"  margin={m:.1f} | R={R:.2f}, γ={gamma:.3f} | "
              f"Frissítések: {updates:5d} | Korlát (R/γ)²: {bound:10.1f}")

    fig1, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax1 = axes[0]
    ms, Rs, gammas, upds, bounds = zip(*results)
    ax1.semilogy(ms, bounds, "rs--", markersize=8, linewidth=2, label="Elméleti korlát (R/γ)²")
    ax1.semilogy(ms, upds, "bo-", markersize=8, linewidth=2, label="Tényleges frissítések")
    ax1.set_xlabel("Margin beállítás", fontsize=12)
    ax1.set_ylabel("Frissítések száma (log)", fontsize=12)
    ax1.set_title("Novikoff-tétel: margin hatása", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.fill_between(ms, upds, bounds, alpha=0.15, color="green", label="Rés (korlát vs. valós)")

    # ── 2. kísérlet: Dimenzió hatása ──
    print("\n" + "=" * 60)
    print("2. KÍSÉRLET: Dimenzió hatása")
    print("=" * 60)

    dims = [2, 3, 5, 10, 20, 50]
    dim_results = []

    for d in dims:
        X, y = generate_separable_data(n=300, margin=1.0, dim=d, seed=42)
        R, gamma, _, _ = compute_margin_and_radius(X, y)
        updates, _, _ = perceptron_train(X, y)
        bound = (R / gamma) ** 2
        dim_results.append((d, R, gamma, updates, bound))
        print(f"  dim={d:2d} | R={R:.2f}, γ={gamma:.3f} | "
              f"Frissítések: {updates:5d} | Korlát: {bound:10.1f}")

    ax2 = axes[1]
    ds, Rs2, gammas2, upds2, bounds2 = zip(*dim_results)
    ax2.semilogy(ds, bounds2, "rs--", markersize=8, linewidth=2, label="Elméleti korlát (R/γ)²")
    ax2.semilogy(ds, upds2, "bo-", markersize=8, linewidth=2, label="Tényleges frissítések")
    ax2.set_xlabel("Bemenet dimenziója (d)", fontsize=12)
    ax2.set_ylabel("Frissítések száma (log)", fontsize=12)
    ax2.set_title("Novikoff-tétel: dimenzió hatása", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig1.suptitle("Perceptron konvergencia-tétel (Novikoff, 1962)", fontsize=15, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("novikoff_tetel.png", dpi=150)
    print("\nÁbra mentve: novikoff_tetel.png")

    # ── 3. ábra: Margin vizualizáció 2D-ben ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    for ax, m in zip(axes2, [0.3, 1.0, 3.0]):
        X, y = generate_separable_data(n=150, margin=m, seed=42)
        R, gamma, w_opt, b_opt = compute_margin_and_radius(X, y)
        updates, w_p, b_p = perceptron_train(X, y)

        colors = ["#2196F3" if yi == -1 else "#4CAF50" for yi in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=25, edgecolors="k", linewidth=0.3, alpha=0.7)

        # Optimális döntési határ (SVM) + margin sáv
        xlim = ax.get_xlim()
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
        if abs(w_opt[1]) > 1e-8:
            y_line = -(w_opt[0] * x_line + b_opt) / w_opt[1]
            y_margin_p = -(w_opt[0] * x_line + b_opt - gamma) / w_opt[1]
            y_margin_n = -(w_opt[0] * x_line + b_opt + gamma) / w_opt[1]
            ax.plot(x_line, y_line, "r-", linewidth=2, label="Optimális határ")
            ax.plot(x_line, y_margin_p, "r--", linewidth=1, alpha=0.5)
            ax.plot(x_line, y_margin_n, "r--", linewidth=1, alpha=0.5)
            ax.fill_between(x_line, y_margin_n, y_margin_p, alpha=0.1, color="red")

        # Perceptron döntési határ
        norm_p = np.linalg.norm([w_p[0], w_p[1]])
        if norm_p > 1e-8 and abs(w_p[1]) > 1e-8:
            y_perc = -(w_p[0] * x_line + b_p) / w_p[1]
            ax.plot(x_line, y_perc, "k--", linewidth=1.5, alpha=0.7, label="Perceptron határ")

        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        ax.set_title(f"γ = {gamma:.2f} | Frissítések: {updates}\nKorlát: {(R/gamma)**2:.0f}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

        # R sugár kör
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R * np.cos(theta), R * np.sin(theta), "gray", linewidth=0.8, linestyle=":", alpha=0.5)

    fig2.suptitle("Margin és konvergencia: kis γ → sok frissítés, nagy γ → kevés", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("novikoff_margin_viz.png", dpi=150)
    print("Ábra mentve: novikoff_margin_viz.png")

    plt.close("all")
    print("\nKész!")


if __name__ == "__main__":
    main()
