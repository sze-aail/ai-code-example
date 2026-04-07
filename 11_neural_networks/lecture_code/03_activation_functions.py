"""
03_activation_functions.py — Aktivációs függvények összehasonlítása
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Klasszikus és modern aktivációs függvények
  - Függvényértékek és deriváltak vizualizációja
  - Vanishing gradient probléma szemléltetése
"""
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-5, 5, 500)

# ── Aktivációs függvények ──
def sigmoid(x):     return 1 / (1 + np.exp(-x))
def tanh_(x):       return np.tanh(x)
def relu(x):        return np.maximum(0, x)
def leaky_relu(x):  return np.where(x > 0, x, 0.01 * x)
def elu(x, a=1.0):  return np.where(x > 0, x, a * (np.exp(x) - 1))
def gelu(x):        return x * 0.5 * (1 + np.vectorize(lambda v: np.tanh(np.sqrt(2/np.pi)*(v + 0.044715*v**3)))(x))
def swish(x):       return x * sigmoid(x)
def mish(x):        return x * np.tanh(np.log(1 + np.exp(x)))

# ── Deriváltak (numerikus) ──
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

activations = [
    ("Sigmoid",    sigmoid,    "#E91E63"),
    ("Tanh",       tanh_,      "#9C27B0"),
    ("ReLU",       relu,       "#2196F3"),
    ("Leaky ReLU", leaky_relu, "#00BCD4"),
    ("ELU",        elu,        "#4CAF50"),
    ("GELU",       gelu,       "#FF9800"),
    ("Swish/SiLU", swish,      "#795548"),
    ("Mish",       mish,       "#607D8B"),
]


def main():
    # ── 1. ábra: Függvényértékek ──
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, func, color) in zip(axes1.flat, activations):
        ax.plot(x, func(x), color=color, linewidth=2.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(name, fontsize=13, fontweight="bold")
        ax.set_xlim(-5, 5)
        ax.grid(True, alpha=0.2)
    fig1.suptitle("Aktivációs függvények", fontsize=16, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("03_activations.png", dpi=150)
    print("Ábra mentve: 03_activations.png")

    # ── 2. ábra: Deriváltak ──
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (name, func, color) in zip(axes2.flat, activations):
        dx = numerical_derivative(func, x)
        ax.plot(x, dx, color=color, linewidth=2.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_title(f"{name} derivált", fontsize=12)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(True, alpha=0.2)
    fig2.suptitle("Aktivációs függvények deriváltjai", fontsize=16, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("03_derivatives.png", dpi=150)
    print("Ábra mentve: 03_derivatives.png")

    # ── 3. ábra: Vanishing gradient demonstráció ──
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    n_layers = 20
    grad_sigmoid = np.array([sigmoid(0.5)] * n_layers)  # max derivált sigmoid = 0.25
    grad_tanh    = np.array([1 - np.tanh(0.5)**2] * n_layers)
    grad_relu    = np.array([1.0] * n_layers)

    cumgrad_sig  = np.cumprod(grad_sigmoid * 0.25)  # sigmoid derivált max ~0.25
    cumgrad_tanh = np.cumprod(grad_tanh * 0.5)
    cumgrad_relu = np.cumprod(grad_relu)

    ax3.semilogy(range(1, n_layers + 1), cumgrad_sig,  "o-", color="#E91E63", label="Sigmoid", linewidth=2)
    ax3.semilogy(range(1, n_layers + 1), cumgrad_tanh, "s-", color="#9C27B0", label="Tanh", linewidth=2)
    ax3.semilogy(range(1, n_layers + 1), cumgrad_relu, "^-", color="#2196F3", label="ReLU", linewidth=2)
    ax3.set_xlabel("Réteg mélység (visszafelé)", fontsize=12)
    ax3.set_ylabel("Kumulált gradiens (log skála)", fontsize=12)
    ax3.set_title("Vanishing Gradient probléma", fontsize=14, fontweight="bold")
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig("03_vanishing_gradient.png", dpi=150)
    print("Ábra mentve: 03_vanishing_gradient.png")

    # ── 4. ábra: Kumulált gradiens minden aktivációra ──
    # Számoljuk minden aktiváció deriváltját egy reprezentatív pre-aktivációs értéken (pl. 0.5)
    # majd a rétegek mentén a kumulált gradiens (szorzat) log-skálán.
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    n_layers_all = 20
    v0 = 0.5
    markers = ["o", "s", "^", "d", "v", "p", "h", "x"]
    for (name, func, color), m in zip(activations, markers):
        # numerical_derivative visszaad array-et, ezért adunk át numpy array-t
        g = numerical_derivative(func, np.array([v0]))
        # ha g egy array, vegyük az első elemet; ha skalar, úgyis működik
        try:
            g_val = float(g[0])
        except Exception:
            g_val = float(g)
        # Készítsünk n_layers_all hosszú tömböt a deriváltból, majd kumulatív szorzat
        cum = np.cumprod(np.full(n_layers_all, g_val))
        ax4.semilogy(
            range(1, n_layers_all + 1),
            cum,
            marker=m,
            linestyle="-",
            color=color,
            linewidth=2,
            label=f"{name} (g={g_val:.3f})",
        )

    ax4.set_xlabel("Réteg mélység (visszafelé)", fontsize=12)
    ax4.set_ylabel("Kumulált gradiens (log skála)", fontsize=12)
    ax4.set_title("Kumulált gradiens minden aktivációhoz (reprezentatív derivált = 0.5)", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    fig4.tight_layout()
    fig4.savefig("03_vanishing_gradient_all.png", dpi=150)
    print("Ábra mentve: 03_vanishing_gradient_all.png")

    plt.close("all")
    print("\nKész!")

if __name__ == "__main__":
    main()
