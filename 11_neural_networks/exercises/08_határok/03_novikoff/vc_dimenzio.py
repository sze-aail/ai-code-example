"""
vc_dimenzio.py — VC-dimenzió szemléltetés
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - VC-dimenzió definíciója: a legnagyobb n, amelyre minden 2ⁿ címkézés szétválasztható
  - Perceptron VC-dimenziója = d+1 (d bemeneti dimenzió)
  - 2D-ben: 3 pont szétválasztható (2³=8 eset mind), 4 pont NEM (van XOR-szerű eset)
  - Kapcsolat a generalizációhoz (több paraméter → nagyobb VC → több adat kell)
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# ── Perceptron megoldhatóság tesztelése ──
def is_separable(X, y, max_iter=10000, lr=0.1):
    """Ellenőrzi, hogy a perceptron meg tudja-e oldani a feladatot."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(max_iter):
        errors = 0
        for xi, yi in zip(X, y):
            if yi * (np.dot(w, xi) + b) <= 0:
                w += lr * yi * xi
                b += lr * yi
                errors += 1
        if errors == 0:
            return True, w, b
    return False, w, b


def plot_with_boundary(ax, X, y, w, b, separable, title=""):
    """Adatpontok és döntési határ ábrázolása."""
    colors = ["#E91E63" if yi == -1 else "#4CAF50" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=120, edgecolors="k", linewidth=1.5, zorder=3)

    if separable and np.linalg.norm(w) > 1e-8 and abs(w[1]) > 1e-8:
        x_line = np.linspace(X[:, 0].min() - 1.5, X[:, 0].max() + 1.5, 100)
        y_line = -(w[0] * x_line + b) / w[1]
        ax.plot(x_line, y_line, "b-", linewidth=2, alpha=0.7)

    status = "✓" if separable else "✗"
    color = "#4CAF50" if separable else "#E91E63"
    ax.set_title(title, fontsize=9, fontweight="bold", color=color)
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=7)


# ── 1. ábra: 3 pont — mind a 2³=8 címkézés szétválasztható ──
print("=" * 60)
print("VC-DIMENZIÓ: 2D perceptron (d=2) → VC = d+1 = 3")
print("=" * 60)
print("\n3 pont (általános helyzetben): MINDEN 2³=8 címkézés szétválasztható")

# 3 pont általános helyzetben (nem kollineáris)
X3 = np.array([[0, 1.5], [-1.3, -0.8], [1.3, -0.8]])

fig1, axes1 = plt.subplots(2, 4, figsize=(14, 7))
all_labels_3 = list(product([-1, 1], repeat=3))

all_sep = True
for ax, labels in zip(axes1.flat, all_labels_3):
    y = np.array(labels)
    sep, w, b = is_separable(X3, y)
    if not sep:
        all_sep = False
    label_str = "".join(["+" if l == 1 else "−" for l in labels])
    plot_with_boundary(ax, X3, y, w, b, sep, f"[{label_str}] {'✓' if sep else '✗'}")

fig1.suptitle("VC-dimenzió: 3 pont 2D-ben — mind a 8 címkézés szétválasztható (VC ≥ 3)",
              fontsize=14, fontweight="bold")
fig1.tight_layout()
fig1.savefig("vc_3pont.png", dpi=150)
print(f"  Mind szétválasztható: {all_sep}")
print("  Ábra mentve: vc_3pont.png")

# ── 2. ábra: 4 pont — VAN nem szétválasztható címkézés (XOR) ──
print("\n4 pont: VAN nem szétválasztható címkézés (XOR-típusú)")

X4 = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float)

# Csak a 16 címkézésből mutassuk a legérdekesebbeket
fig2, axes2 = plt.subplots(2, 4, figsize=(14, 7))
all_labels_4 = list(product([-1, 1], repeat=4))

sep_count = 0
nonsep_examples = []
sep_examples = []

for labels in all_labels_4:
    y = np.array(labels)
    sep, w, b = is_separable(X4, y)
    if sep:
        sep_count += 1
        if len(sep_examples) < 4:
            sep_examples.append((y, w, b))
    else:
        if len(nonsep_examples) < 4:
            nonsep_examples.append((y, np.zeros(2), 0))

print(f"  Szétválasztható: {sep_count}/16, Nem szétválasztható: {16 - sep_count}/16")

# Felső sor: szétválasztható példák
for ax, (y, w, b) in zip(axes2[0], sep_examples):
    label_str = "".join(["+" if l == 1 else "−" for l in y])
    plot_with_boundary(ax, X4, y, w, b, True, f"[{label_str}] ✓ szétválasztható")

# Alsó sor: NEM szétválasztható példák
for ax, (y, w, b) in zip(axes2[1], nonsep_examples):
    label_str = "".join(["+" if l == 1 else "−" for l in y])
    plot_with_boundary(ax, X4, y, w, b, False, f"[{label_str}] ✗ NEM szétválasztható")

fig2.suptitle("4 pont 2D-ben — nem minden címkézés szétválasztható → VC < 4, tehát VC = 3",
              fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig("vc_4pont.png", dpi=150)
print("  Ábra mentve: vc_4pont.png")

# ── 3. ábra: VC-dimenzió és generalizáció ──
print("\n" + "=" * 60)
print("VC-DIMENZIÓ ÉS GENERALIZÁCIÓ")
print("=" * 60)

fig3, ax3 = plt.subplots(figsize=(9, 6))

# Elméleti generalizációs korlát: ε ≤ sqrt((VC * (ln(2n/VC) + 1) - ln(δ/4)) / n)
n_range = np.arange(10, 1000, 5)
delta = 0.05  # 95% konfidencia

for vc, color, label in [(3, "#2196F3", "VC=3 (2D perceptron)"),
                          (10, "#FF9800", "VC=10 (9D perceptron)"),
                          (50, "#E91E63", "VC=50 (49D perceptron)"),
                          (200, "#9C27B0", "VC=200 (MLP ~64 param)")]:
    # Vapnik-Chervonenkis generalizációs korlát
    eps = np.sqrt(np.maximum(0, (vc * (np.log(2 * n_range / vc) + 1) - np.log(delta / 4)) / n_range))
    eps = np.clip(eps, 0, 2)
    ax3.plot(n_range, eps, color=color, linewidth=2, label=label)

ax3.set_xlabel("Tanítóminták száma (n)", fontsize=12)
ax3.set_ylabel("Generalizációs hiba felső korlátja (ε)", fontsize=12)
ax3.set_title("VC-dimenzió és generalizáció: több paraméter → több adat kell",
              fontsize=13, fontweight="bold")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.axhline(0.1, color="gray", linestyle="--", alpha=0.4)
ax3.text(800, 0.12, "ε = 0.1", color="gray", fontsize=9)

fig3.tight_layout()
fig3.savefig("vc_generalizacio.png", dpi=150)
print("Ábra mentve: vc_generalizacio.png")

# ── Összefoglaló ──
print("\n" + "=" * 60)
print("ÖSSZEFOGLALÁS")
print("=" * 60)
print("  • Perceptron VC-dimenziója d bemeneten: VC = d + 1")
print("  • 2D-ben: 3 pont szétválasztható, 4 pont NEM mindig")
print("  • Nagyobb VC → expresszívebb modell, DE több adat kell")
print("  • MLP VC-dimenziója ≈ O(W·log(W)), ahol W a súlyok száma")
print("  • Ez motiválja a regularizációt: csökkentsük az effektív VC-t!")

plt.close("all")
print("\nKész!")
