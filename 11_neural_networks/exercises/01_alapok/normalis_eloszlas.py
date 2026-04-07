"""
normalis_eloszlas.py — Normális eloszlás és súlyinicializáció
Forrás: mesterseges-intelligencia-kodok (06_normal_distribution), kibővítve

Demonstrálja:
  - Normális eloszlás és annak tulajdonságai
  - Miért fontos a súlyinicializáció neurális hálókban
  - Xavier és He inicializáció összehasonlítása
"""
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(42)

# ── 1. Normális eloszlás alap ──
fig1, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (mu, sigma) in zip(axes, [(0, 1), (0, 0.5), (2, 1.5)]):
    samples = np.random.normal(mu, sigma, 5000)
    ax.hist(samples, bins=50, density=True, alpha=0.7, color="#2196F3", edgecolor="white")
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    pdf = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, pdf, "r-", linewidth=2.5)
    ax.set_title(f"μ={mu}, σ={sigma}", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

fig1.suptitle("Normális eloszlás", fontsize=14, fontweight="bold")
fig1.tight_layout()
fig1.savefig("normalis_eloszlas.png", dpi=150)
print("Ábra mentve: normalis_eloszlas.png")

# ── 2. Súlyinicializáció hatása ──
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))

n_in, n_out = 256, 256
inits = [
    ("Túl nagy (σ=1.0)", np.random.randn(n_in, n_out) * 1.0),
    ("Xavier (σ=√(2/(n_in+n_out)))", np.random.randn(n_in, n_out) * np.sqrt(2.0 / (n_in + n_out))),
    ("He/Kaiming (σ=√(2/n_in))", np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)),
]

# Szimuláljuk 10 réteg forward pass-ét ReLU-val
for ax, (name, W_init) in zip(axes2, inits):
    x = np.random.randn(1, n_in)
    activations = []
    for layer in range(10):
        W = W_init.copy() if layer == 0 else np.random.randn(n_in, n_out) * W_init.std()
        x = x @ W
        x = np.maximum(0, x)  # ReLU
        activations.append(x.flatten())

    # Az utolsó réteg aktivációinak eloszlása
    final = activations[-1]
    ax.hist(final[final != 0], bins=50, density=True, alpha=0.7, color="#4CAF50", edgecolor="white")
    ax.set_title(f"{name}\nstd={final.std():.2e}, nulla%={100*np.mean(final==0):.0f}%",
                 fontsize=10, fontweight="bold")
    ax.set_xlim(-0.1, max(0.1, np.percentile(final[final > 0], 99) if np.any(final > 0) else 0.1))
    ax.grid(True, alpha=0.3)

fig2.suptitle("Súlyinicializáció: 10 ReLU réteg utáni aktiváció-eloszlás", fontsize=13, fontweight="bold")
fig2.tight_layout()
fig2.savefig("sulyinicializacio.png", dpi=150)
print("Ábra mentve: sulyinicializacio.png")

plt.close("all")
print("Kész!")
