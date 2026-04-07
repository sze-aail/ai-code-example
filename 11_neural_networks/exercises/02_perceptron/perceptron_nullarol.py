"""
01_perceptron.py — Perceptron megvalósítás nulláról
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Perceptron tanulási algoritmus
  - Lineáris szeparálhatóság
  - Döntési határ vizualizáció
"""
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """Egyszerű perceptron bináris osztályozáshoz."""

    def __init__(self, n_features: int, lr: float = 0.1):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.history: list[dict] = []

    def predict(self, x: np.ndarray) -> int:
        return 1 if np.dot(self.w, x) + self.b > 0 else 0

    def train(self, X: np.ndarray, y: np.ndarray, max_epochs: int = 100) -> int:
        """Perceptron tanulási szabály. Visszaadja az iterációk számát."""
        for epoch in range(max_epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                err = yi - pred
                if err != 0:
                    self.w += self.lr * err * xi
                    self.b += self.lr * err
                    errors += 1
            self.history.append({"epoch": epoch, "errors": errors, "w": self.w.copy(), "b": self.b})
            if errors == 0:
                print(f"Konvergált {epoch + 1} epoch után!")
                return epoch + 1
        print(f"Nem konvergált {max_epochs} epoch alatt (utolsó hibák: {errors})")
        return max_epochs


def plot_decision_boundary(model: Perceptron, X: np.ndarray, y: np.ndarray, title: str = ""):
    """Döntési határ ábrázolása."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    colors = ["#2196F3" if yi == 0 else "#4CAF50" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=100, edgecolors="k", zorder=3)

    # Döntési határ: w0*x0 + w1*x1 + b = 0  =>  x1 = -(w0*x0 + b) / w1
    xlim = ax.get_xlim()
    if abs(model.w[1]) > 1e-8:
        x_line = np.linspace(xlim[0] - 0.5, xlim[1] + 0.5, 100)
        y_line = -(model.w[0] * x_line + model.b) / model.w[1]
        ax.plot(x_line, y_line, "r-", linewidth=2, label="Döntési határ")
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(title or "Perceptron döntési határ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# ── AND kapu ──
print("=" * 50)
print("AND kapu tanítása")
print("=" * 50)
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(n_features=2, lr=0.1)
p_and.train(X_and, y_and)
print(f"Súlyok: w={p_and.w}, b={p_and.b:.2f}")
for xi, yi in zip(X_and, y_and):
    print(f"  {xi} -> pred={p_and.predict(xi)}, elvárt={yi}")

fig_and = plot_decision_boundary(p_and, X_and, y_and, "AND kapu — Perceptron")
fig_and.savefig("01_perceptron_and.png", dpi=150)
print("Ábra mentve: 01_perceptron_and.png\n")

# ── OR kapu ──
print("=" * 50)
print("OR kapu tanítása")
print("=" * 50)
y_or = np.array([0, 1, 1, 1])

p_or = Perceptron(n_features=2, lr=0.1)
p_or.train(X_and, y_or)
print(f"Súlyok: w={p_or.w}, b={p_or.b:.2f}")

fig_or = plot_decision_boundary(p_or, X_and, y_or, "OR kapu — Perceptron")
fig_or.savefig("01_perceptron_or.png", dpi=150)
print("Ábra mentve: 01_perceptron_or.png\n")

# ── XOR kapu (NEM konvergál!) ──
print("=" * 50)
print("XOR kapu tanítása (lineárisan NEM szeparálható)")
print("=" * 50)
y_xor = np.array([0, 1, 1, 0])

p_xor = Perceptron(n_features=2, lr=0.1)
p_xor.train(X_and, y_xor, max_epochs=20)

# Tanulási görbe
fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
ax_hist.plot([h["epoch"] for h in p_xor.history], [h["errors"] for h in p_xor.history], "ro-")
ax_hist.set_xlabel("Epoch")
ax_hist.set_ylabel("Hibák száma")
ax_hist.set_title("XOR — Perceptron nem konvergál")
ax_hist.grid(True, alpha=0.3)
fig_hist.tight_layout()
fig_hist.savefig("01_perceptron_xor_history.png", dpi=150)
print("Ábra mentve: 01_perceptron_xor_history.png")

plt.close("all")
print("\nKész!")
