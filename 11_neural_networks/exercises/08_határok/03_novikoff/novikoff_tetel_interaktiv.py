"""
novikoff_tetel_interaktiv.py — Perceptron konvergencia-tétel (Novikoff, 1962)
Neurális hálók I. — Hajdu Csaba

Interaktív vizualizáció:
  A csúszkák segítségével állítható a margin (γ) és az adatpontok száma (N).
  Az ábra valós időben mutatja a tényleges frissítések számát és az elméleti korlátot (R/γ)²,
  valamint felrajzolja a sugarat (R), az optimális és a perceptron döntési határt.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.svm import SVC

def perceptron_train(X, y, max_iter=50000):
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
    svm = SVC(kernel='linear', C=1e6)
    svm.fit(X, y)
    w_opt = svm.coef_[0]
    b_opt = svm.intercept_[0]
    norm_w = np.linalg.norm(w_opt)
    distances = np.abs(X @ w_opt + b_opt) / norm_w
    gamma = distances.min()
    R = np.max(np.linalg.norm(X, axis=1))
    return R, gamma, w_opt / norm_w, b_opt / norm_w

def generate_separable_data(n=100, margin=1.0, dim=2, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, dim)
    y = np.sign(X[:, 0])
    y[y == 0] = 1
    X[:, 0] += y * margin / 2
    mask = np.abs(X[:, 0]) > margin / 4
    return X[mask], y[mask].astype(int)

def main():
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.subplots_adjust(bottom=0.35)

    # Sliders
    ax_margin = plt.axes([0.15, 0.2, 0.65, 0.03])
    ax_n = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_seed = plt.axes([0.15, 0.1, 0.65, 0.03])

    s_margin = Slider(ax_margin, 'Margin (Szeparáció)', 0.1, 5.0, valinit=1.0, valstep=0.1)
    s_n = Slider(ax_n, 'Generált pontok (N)', 20, 500, valinit=100, valstep=10)
    s_seed = Slider(ax_seed, 'Random Seed', 0, 100, valinit=42, valstep=1)

    def update(val):
        ax.clear()

        margin = s_margin.val
        n = int(s_n.val)
        seed = int(s_seed.val)

        X, y = generate_separable_data(n=n, margin=margin, seed=seed)
        if len(np.unique(y)) < 2:
            ax.set_title("Nincs elég adat mindkét osztályból, állíts a seed-en vagy N-en!")
            fig.canvas.draw_idle()
            return

        R, gamma, w_opt, b_opt = compute_margin_and_radius(X, y)
        updates, w_p, b_p = perceptron_train(X, y)

        colors = ["#2196F3" if yi == -1 else "#4CAF50" for yi in y]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=30, edgecolors="k", zorder=3)

        # Optimális margin vonalak
        xlim = ax.get_xlim()
        x_line = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)

        if abs(w_opt[1]) > 1e-8:
            y_line = -(w_opt[0] * x_line + b_opt) / w_opt[1]
            y_margin_p = -(w_opt[0] * x_line + b_opt - gamma) / w_opt[1]
            y_margin_n = -(w_opt[0] * x_line + b_opt + gamma) / w_opt[1]
            ax.plot(x_line, y_line, "r-", linewidth=2, label="Optimális határ (SVM)")
            ax.plot(x_line, y_margin_p, "r--", linewidth=1, alpha=0.5)
            ax.plot(x_line, y_margin_n, "r--", linewidth=1, alpha=0.5)
            ax.fill_between(x_line, y_margin_n, y_margin_p, alpha=0.1, color="red")

        # Perceptron konvergált vonala
        norm_p = np.linalg.norm([w_p[0], w_p[1]])
        if norm_p > 1e-8 and abs(w_p[1]) > 1e-8:
            y_perc = -(w_p[0] * x_line + b_p) / w_p[1]
            ax.plot(x_line, y_perc, "k--", linewidth=1.5, alpha=0.7, label="Perceptron határ")

        # R sugár jelölése
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(R * np.cos(theta), R * np.sin(theta), "gray", linewidth=0.8, linestyle=":", alpha=0.5, label="Legnagyobb sugár (R)")

        ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

        bound = (R/gamma)**2
        ax.set_title(f"Valós γ = {gamma:.2f}, R = {R:.2f}\nFrissítések: {updates} | Elméleti korlát (R/γ)²: {bound:.1f}", fontsize=12)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    s_margin.on_changed(update)
    s_n.on_changed(update)
    s_seed.on_changed(update)

    update(0)
    plt.show()

if __name__ == "__main__":
    main()
