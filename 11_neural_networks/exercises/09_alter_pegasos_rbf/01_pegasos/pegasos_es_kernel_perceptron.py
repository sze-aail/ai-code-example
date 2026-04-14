"""
pegasos_es_kernel_perceptron.py — Pegasos, távolságmetrikák és RBF kernel perceptron
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Pegasos (Primal Estimated sub-GrAdient SOlver for SVM): online SVM tanulás
  - Perceptron → Pegasos fejlődési ív (margin maximalizálás + regularizáció)
  - Távolságmetrikák hatása (euklideszi, Manhattan, Csebisev, Mahalanobis)
  - Kernel perceptron: nemlineáris döntési határ rejtett réteg nélkül
  - RBF kernel perceptron: XOR megoldása egyetlen réteggel!
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs


# ══════════════════════════════════════════════════════════════
# 1. RÉSZ: PEGASOS — Online SVM
# ══════════════════════════════════════════════════════════════

class Pegasos:
    """
    Pegasos (Shalev-Shwartz et al., 2007):
    Online / mini-batch SVM a primal térben.

    Frissítési szabály t-edik lépésben:
      η_t = 1 / (λ · t)
      w_{t+1} = (1 - η_t·λ)·w_t + η_t·y_i·x_i   ha y_i·⟨w_t, x_i⟩ < 1
      w_{t+1} = (1 - η_t·λ)·w_t                    egyébként

    A perceptrontól eltérően:
      - Van regularizáció (λ): a súlyok nem nőnek korlátlanul
      - A margin-t maximalizálja (hinge loss): nem csak szétválaszt, hanem JÓLOL választ szét
      - A learning rate csökken (1/λt): garantált konvergencia O(1/t)
    """

    def __init__(self, lam: float = 0.01):
        self.lam = lam
        self.w = None
        self.b = 0.0
        self.history = []

    def fit(self, X, y, n_epochs=20):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        t = 1

        for epoch in range(n_epochs):
            perm = np.random.permutation(n)
            epoch_loss = 0
            for i in perm:
                eta = 1.0 / (self.lam * t)
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)

                # Regularizáció (weight decay)
                self.w *= (1 - eta * self.lam)

                # Hinge loss gradiens
                if margin < 1:
                    self.w += eta * y[i] * X[i]
                    self.b += eta * y[i]
                    epoch_loss += 1 - margin

                t += 1

            self.history.append(epoch_loss / n)

        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def decision_function(self, X):
        return X @ self.w + self.b


class SimplePerceptron:
    """Összehasonlításhoz: alap perceptron."""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = 0.0

    def fit(self, X, y, n_epochs=20):
        n, d = X.shape
        self.w = np.zeros(d)
        for epoch in range(n_epochs):
            for i in np.random.permutation(n):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
        return self

    def predict(self, X):
        return np.sign(X @ self.w + self.b)

    def decision_function(self, X):
        return X @ self.w + self.b


def plot_decision_2d(ax, model, X, y, title, resolution=200):
    """Döntési határ és margin vizualizáció."""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.4)
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["blue", "black", "blue"],
               linestyles=["--", "-", "--"], linewidths=[1, 2, 1])

    colors = ["#E91E63" if yi == -1 else "#4CAF50" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=30, edgecolors="k", linewidth=0.5, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)


# ── Perceptron vs. Pegasos összehasonlítás ──
print("=" * 60)
print("1. PERCEPTRON vs. PEGASOS")
print("=" * 60)

np.random.seed(42)
X_blobs, y_blobs = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=42)
y_blobs = 2 * y_blobs - 1  # {0,1} → {-1,1}

fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))

# Perceptron
perc = SimplePerceptron(lr=0.01).fit(X_blobs, y_blobs, n_epochs=30)
plot_decision_2d(axes1[0], perc, X_blobs, y_blobs, "Perceptron\n(bármely szétválasztó sík)")

# Pegasos λ=0.01
peg1 = Pegasos(lam=0.01).fit(X_blobs, y_blobs, n_epochs=30)
plot_decision_2d(axes1[1], peg1, X_blobs, y_blobs, "Pegasos (λ=0.01)\n(margin-maximalizáló)")

# Pegasos λ=0.001
peg2 = Pegasos(lam=0.001).fit(X_blobs, y_blobs, n_epochs=30)
plot_decision_2d(axes1[2], peg2, X_blobs, y_blobs, "Pegasos (λ=0.001)\n(lazább regularizáció)")

fig1.suptitle("Perceptron vs. Pegasos: a szaggatott vonal a margin-sávot jelöli",
              fontsize=14, fontweight="bold")
fig1.tight_layout()
fig1.savefig("pegasos_vs_perceptron.png", dpi=150)
print("Ábra mentve: pegasos_vs_perceptron.png")


# ══════════════════════════════════════════════════════════════
# 2. RÉSZ: TÁVOLSÁGMETRIKÁK
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. TÁVOLSÁGMETRIKÁK VIZUALIZÁCIÓ")
print("=" * 60)

def dist_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=-1))

def dist_manhattan(x, y):
    return np.sum(np.abs(x - y), axis=-1)

def dist_chebyshev(x, y):
    return np.max(np.abs(x - y), axis=-1)

def dist_cosine(x, y):
    dot = np.sum(x * y, axis=-1)
    norm_x = np.sqrt(np.sum(x ** 2, axis=-1))
    norm_y = np.sqrt(np.sum(y ** 2, axis=-1))
    return 1 - dot / (norm_x * norm_y + 1e-8)

metrics = [
    ("Euklideszi (L2)", dist_euclidean),
    ("Manhattan (L1)", dist_manhattan),
    ("Csebisev (L∞)", dist_chebyshev),
    ("Koszinusz", dist_cosine),
]

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5))
center = np.array([0.0, 0.0])
xx, yy = np.meshgrid(np.linspace(-3, 3, 300), np.linspace(-3, 3, 300))
grid_points = np.stack([xx, yy], axis=-1)

for ax, (name, dist_fn) in zip(axes2, metrics):
    dist_map = dist_fn(grid_points, center)
    contour = ax.contourf(xx, yy, dist_map, levels=20, cmap="YlOrRd_r", alpha=0.8)
    ax.contour(xx, yy, dist_map, levels=[1.0], colors="k", linewidths=2)
    ax.plot(0, 0, "k*", markersize=12)
    ax.set_title(f"{name}\n(egységkör feketén)", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.grid(True, alpha=0.2)

fig2.suptitle("Távolságmetrikák: azonos távolságú pontok halmazai (izometrikus görbék)",
              fontsize=14, fontweight="bold")
fig2.tight_layout()
fig2.savefig("tavolsagmetrikak.png", dpi=150)
print("Ábra mentve: tavolsagmetrikak.png")


# ══════════════════════════════════════════════════════════════
# 3. RÉSZ: KERNEL PERCEPTRON + RBF
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. KERNEL PERCEPTRON (RBF)")
print("=" * 60)

class KernelPerceptron:
    """
    Kernel perceptron: a belső szorzatot k(x_i, x_j)-vel helyettesítjük.

    Predikció: y = sign(Σ α_i · y_i · k(x_i, x))
    Frissítés: ha hibás, α_i += 1

    RBF kernel: k(x, y) = exp(-γ · ||x - y||²)
      - γ kicsi: simább határ (alulillesztés)
      - γ nagy: bonyolultabb határ (túlillesztés)
    """

    def __init__(self, kernel='rbf', gamma=1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.alphas = None
        self.X_train = None
        self.y_train = None

    def _kernel_fn(self, X1, X2):
        if self.kernel == 'rbf':
            # ||x - y||² = ||x||² + ||y||² - 2·x·y
            sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dist_sq = sq1 + sq2 - 2 * X1 @ X2.T
            return np.exp(-self.gamma * dist_sq)
        elif self.kernel == 'linear':
            return X1 @ X2.T
        elif self.kernel == 'poly':
            return (1 + X1 @ X2.T) ** 3

    def fit(self, X, y, n_epochs=50):
        n = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.alphas = np.zeros(n)

        K = self._kernel_fn(X, X)

        for epoch in range(n_epochs):
            errors = 0
            for i in range(n):
                decision = np.sum(self.alphas * self.y_train * K[:, i])
                if y[i] * decision <= 0:
                    self.alphas[i] += 1
                    errors += 1
            if errors == 0:
                break

        return self

    def decision_function(self, X):
        K = self._kernel_fn(self.X_train, X)
        return (self.alphas * self.y_train) @ K

    def predict(self, X):
        return np.sign(self.decision_function(X))


def plot_kernel_decision(ax, model, X, y, title, resolution=200):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=2)
    colors = ["#E91E63" if yi == -1 else "#4CAF50" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=40, edgecolors="k", linewidth=0.5, zorder=3)

    # Support vectorok kiemelése (α > 0)
    sv_mask = model.alphas > 0
    ax.scatter(X[sv_mask, 0], X[sv_mask, 1], s=120, facecolors="none",
               edgecolors="gold", linewidths=2, zorder=4, label=f"SV: {sv_mask.sum()}")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.2)


def main():
    # ── 3a: XOR megoldása kernel perceptronnal ──
    print("\nXOR megoldása RBF kernel perceptronnal:")
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([-1, 1, 1, -1])

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (name, kernel, gamma) in zip(axes3, [
        ("Lineáris kernel\n(NEM oldja meg)", "linear", 1.0),
        ("RBF kernel (γ=2)\nMEGOLDJA!", "rbf", 2.0),
        ("Polinom kernel (fokszám=3)\nMEGOLDJA!", "poly", 1.0),
    ]):
        kp = KernelPerceptron(kernel=kernel, gamma=gamma).fit(X_xor, y_xor, n_epochs=100)
        preds = kp.predict(X_xor)
        acc = np.mean(preds == y_xor)
        plot_kernel_decision(ax, kp, X_xor, y_xor, f"{name}\nPontosság: {acc*100:.0f}%")

    fig3.suptitle("XOR probléma: kernel perceptron megoldja rejtett réteg nélkül!",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("kernel_xor.png", dpi=150)
    print("Ábra mentve: kernel_xor.png")

    # ── 3b: Nemlineáris adathalmazok ──
    print("\nNemlineáris adathalmazok:")
    datasets = [
        ("Make Moons", *make_moons(n_samples=200, noise=0.15, random_state=42)),
        ("Make Circles", *make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)),
    ]

    fig4, axes4 = plt.subplots(2, 4, figsize=(18, 9))

    for row, (ds_name, X_ds, y_ds) in enumerate(datasets):
        y_ds = 2 * y_ds - 1  # {0,1} → {-1,1}

        for col, (gamma, title) in enumerate([
            (0.1, "γ=0.1 (sima)"),
            (1.0, "γ=1.0"),
            (5.0, "γ=5.0"),
            (50.0, "γ=50 (túlillesztés)"),
        ]):
            ax = axes4[row, col]
            kp = KernelPerceptron(kernel='rbf', gamma=gamma).fit(X_ds, y_ds, n_epochs=30)
            acc = np.mean(kp.predict(X_ds) == y_ds)
            plot_kernel_decision(ax, kp, X_ds, y_ds, f"{ds_name}: {title}\nacc={acc*100:.0f}%")

    fig4.suptitle("RBF kernel perceptron: γ hatása (bias-variancia tradeoff rejtett réteg nélkül)",
                  fontsize=14, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig("rbf_gamma_hatas.png", dpi=150)
    print("Ábra mentve: rbf_gamma_hatas.png")

    # ── 3c: Összehasonlító ábra ──
    print("\nPerceptron → Pegasos → Kernel perceptron fejlődési ív:")
    X_moons, y_moons = make_moons(n_samples=300, noise=0.15, random_state=42)
    y_moons_signed = 2 * y_moons - 1

    fig5, axes5 = plt.subplots(1, 4, figsize=(18, 4.5))

    # Perceptron
    perc_m = SimplePerceptron(lr=0.01).fit(X_moons, y_moons_signed, n_epochs=50)
    plot_decision_2d(axes5[0], perc_m, X_moons, y_moons_signed, "1. Perceptron\n(lineáris)")

    # Pegasos
    peg_m = Pegasos(lam=0.01).fit(X_moons, y_moons_signed, n_epochs=50)
    plot_decision_2d(axes5[1], peg_m, X_moons, y_moons_signed, "2. Pegasos (SVM)\n(lineáris + margin)")

    # RBF kernel perceptron γ=2
    kp_m = KernelPerceptron(kernel='rbf', gamma=2.0).fit(X_moons, y_moons_signed, n_epochs=30)
    acc_rbf = np.mean(kp_m.predict(X_moons) == y_moons_signed)
    plot_kernel_decision(axes5[2], kp_m, X_moons, y_moons_signed, f"3. RBF kernel (γ=2)\nacc={acc_rbf*100:.0f}%")

    # RBF kernel perceptron γ=10
    kp_m2 = KernelPerceptron(kernel='rbf', gamma=10.0).fit(X_moons, y_moons_signed, n_epochs=30)
    acc_rbf2 = np.mean(kp_m2.predict(X_moons) == y_moons_signed)
    plot_kernel_decision(axes5[3], kp_m2, X_moons, y_moons_signed, f"4. RBF kernel (γ=10)\nacc={acc_rbf2*100:.0f}%")

    fig5.suptitle("Fejlődési ív: Perceptron → Pegasos (SVM) → Kernel perceptron (nemlineáris)",
                  fontsize=14, fontweight="bold")
    fig5.tight_layout()
    fig5.savefig("fejlodesi_iv.png", dpi=150)
    print("Ábra mentve: fejlodesi_iv.png")

    plt.close("all")
    print("\nKész!")


if __name__=="__main__":
    main()
