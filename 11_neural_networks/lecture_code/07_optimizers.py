"""
07_optimizers.py — Optimalizálók összehasonlítása
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - SGD, Momentum, RMSProp, Adam trajektóriái 2D loss surface-en
  - Konvergencia sebesség összehasonlítás
"""
import numpy as np
import matplotlib.pyplot as plt


# ── Rosenbrock-szerű loss: nehéz felület szűk völggyel ──
def loss_fn(w):
    x, y = w
    return (1 - x)**2 + 10 * (y - x**2)**2

def grad_fn(w):
    x, y = w
    dx = -2 * (1 - x) + 10 * 2 * (y - x**2) * (-2 * x)
    dy = 10 * 2 * (y - x**2)
    return np.array([dx, dy])


# ── Optimalizálók implementációja ──
def sgd(grad, state, lr=0.001):
    return -lr * grad, state

def momentum(grad, state, lr=0.001, beta=0.9):
    v = state.get("v", np.zeros_like(grad))
    v = beta * v + lr * grad
    state["v"] = v
    return -v, state

def rmsprop(grad, state, lr=0.01, beta=0.99, eps=1e-8):
    s = state.get("s", np.zeros_like(grad))
    s = beta * s + (1 - beta) * grad**2
    state["s"] = s
    return -lr * grad / (np.sqrt(s) + eps), state

def adam(grad, state, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    t = state.get("t", 0) + 1
    m = state.get("m", np.zeros_like(grad))
    v = state.get("v", np.zeros_like(grad))
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    state.update({"t": t, "m": m, "v": v})
    return -lr * m_hat / (np.sqrt(v_hat) + eps), state


optimizers = [
    ("SGD (η=0.001)",         sgd,      {"lr": 0.001}),
    ("Momentum (β=0.9)",      momentum, {"lr": 0.001, "beta": 0.9}),
    ("RMSProp (η=0.01)",      rmsprop,  {"lr": 0.01}),
    ("Adam (η=0.01)",         adam,      {"lr": 0.01}),
]

def main():
    # ── Futtatás ──
    w0 = np.array([-1.5, 2.0])
    n_steps = 500

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Loss surface háttér
    xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-1, 3, 200))
    zz = np.array([[loss_fn([x, y]) for x, y in zip(xr, yr)] for xr, yr in zip(xx, yy)])

    colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3"]

    for ax, (name, opt_fn, kwargs), color in zip(axes, optimizers, colors):
        ax.contourf(xx, yy, np.log1p(zz), levels=30, cmap="YlOrRd", alpha=0.6)
        ax.contour(xx, yy, np.log1p(zz), levels=15, colors="k", linewidths=0.3, alpha=0.3)

        w = w0.copy()
        state = {}
        path = [w.copy()]
        for _ in range(n_steps):
            g = grad_fn(w)
            step, state = opt_fn(g, state, **kwargs)
            w = w + step
            w = np.clip(w, -3, 4)
            path.append(w.copy())

        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color=color, markersize=1.5, linewidth=1, alpha=0.8)
        ax.plot(path[0, 0], path[0, 1], "go", markersize=8, zorder=5)
        ax.plot(1, 1, "r*", markersize=12, zorder=5)  # optimum
        ax.set_title(f"{name}\nVégérték: L={loss_fn(path[-1]):.4f}", fontsize=10, fontweight="bold")
        ax.set_xlabel("θ₁")
        ax.set_ylabel("θ₂")

    fig.suptitle("Optimalizálók a Rosenbrock-felületen (zöld=start, piros★=optimum)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("07_optimizers.png", dpi=150)
    print("Ábra mentve: 07_optimizers.png")

    # ── Loss görbék összehasonlítás ──
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for (name, opt_fn, kwargs), color in zip(optimizers, colors):
        w = w0.copy()
        state = {}
        loss_history = []
        for _ in range(n_steps):
            loss_history.append(loss_fn(w))
            g = grad_fn(w)
            step, state = opt_fn(g, state, **kwargs)
            w = w + step
            w = np.clip(w, -3, 4)
        ax2.semilogy(loss_history, color=color, linewidth=2, label=name)

    ax2.set_xlabel("Iteráció", fontsize=12)
    ax2.set_ylabel("Loss (log)", fontsize=12)
    ax2.set_title("Konvergencia sebesség összehasonlítás", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("07_optimizers_loss.png", dpi=150)
    print("Ábra mentve: 07_optimizers_loss.png")

    plt.close("all")
    print("Kész!")

if __name__ == "__main__":
    main()