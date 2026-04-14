"""
retegszam_interaktiv.py — Rétegszám geometriai szabálya interaktívan
Neurális hálók I. — Hajdu Csaba

Interaktív vizualizáció: a felhasználó csúszkákkal állíthatja be
a rejtett rétegek méretét (0 esetén nincs az a réteg), és választhat
különböző adathalmazok közül, majd a "Tanítás és Frissítés" gombra
kattintva megtekintheti az új döntési határt.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# ══════════════════════════════════════════════════════════════
# Adatgenerátorok
# ══════════════════════════════════════════════════════════════

def make_linear_data(n=400, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y

def make_convex_data(n=400, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
    def in_triangle(p):
        v0, v1, v2 = np.array([0, 1.5]), np.array([-1.5, -1]), np.array([1.5, -1])
        d00 = np.dot(v2 - v0, v2 - v0)
        d01 = np.dot(v2 - v0, v1 - v0)
        d11 = np.dot(v1 - v0, v1 - v0)
        d20 = np.dot(p - v0, v2 - v0)
        d21 = np.dot(p - v0, v1 - v0)
        denom = d00 * d11 - d01 * d01
        u = (d11 * d20 - d01 * d21) / denom
        v = (d00 * d21 - d01 * d20) / denom
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    y = np.array([in_triangle(p) for p in X], dtype=np.float32)
    return X, y

def make_nonconvex_data(n=400, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, (n, 2)).astype(np.float32)
    in_box1 = (X[:, 0] > -2.5) & (X[:, 0] < -0.5) & (X[:, 1] > 0.3) & (X[:, 1] < 2.3)
    in_box2 = (X[:, 0] > 0.5) & (X[:, 0] < 2.5) & (X[:, 1] > -2.3) & (X[:, 1] < -0.3)
    y = (in_box1 | in_box2).astype(np.float32)
    return X, y

def make_xor_regions(n=400, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32)
    return X, y

def make_donut_data(n=400, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.5, 2.5, (n, 2)).astype(np.float32)
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = ((r > 0.8) & (r < 1.8)).astype(np.float32)
    return X, y

DATASETS = {
    'Lineáris (félsík)': make_linear_data,
    'Konvex (háromszög)': make_convex_data,
    'Nem-konvex (két doboz)': make_nonconvex_data,
    'XOR (4 negyed)': make_xor_regions,
    'Gyűrű': make_donut_data
}

# ══════════════════════════════════════════════════════════════
# Modell és tanítás
# ══════════════════════════════════════════════════════════════

def make_mlp(hidden_layers):
    layers = []
    prev = 2
    for h in hidden_layers:
        if h > 0:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

def train_model(model, X, y, epochs=500, lr=0.01):
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        pred = model(X_t)
        loss = criterion(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(X_t)) > 0.5).float() == y_t).float().mean().item()
    return acc

# ══════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(12, 7))
    ax_plot = plt.axes([0.35, 0.1, 0.6, 0.8])

    # Vezérlők helye
    ax_radio = plt.axes([0.05, 0.6, 0.25, 0.3])
    ax_slider1 = plt.axes([0.05, 0.45, 0.25, 0.03])
    ax_slider2 = plt.axes([0.05, 0.35, 0.25, 0.03])
    ax_button = plt.axes([0.05, 0.2, 0.25, 0.08])

    radio = RadioButtons(ax_radio, list(DATASETS.keys()))
    slider1 = Slider(ax_slider1, '1. Réteg', 0, 64, valinit=16, valstep=1)
    slider2 = Slider(ax_slider2, '2. Réteg', 0, 64, valinit=0, valstep=1)
    btn_train = Button(ax_button, 'Tanítás és Frissítés', color='lightgreen', hovercolor='palegreen')

    contour_coll = None
    scatter_coll = None

    def update_plot(event=None):
        nonlocal contour_coll, scatter_coll

        ax_plot.clear()

        # Adatok lekérése
        ds_name = radio.value_selected
        X, y = DATASETS[ds_name]()

        # Architektúra beolvasása
        h1 = int(slider1.val)
        h2 = int(slider2.val)
        hidden = []
        if h1 > 0: hidden.append(h1)
        if h2 > 0: hidden.append(h2)

        # Modell építése és tanítása
        ax_plot.set_title("Tanítás folyamatban...", color='red')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        torch.manual_seed(42)
        model = make_mlp(hidden)
        # Bonyolultabb struktúránál több epoch
        epochs = 1500 if sum(hidden) > 0 else 500
        acc = train_model(model, X, y, epochs=epochs, lr=0.01)
        n_params = sum(p.numel() for p in model.parameters())

        # Kirajzolás
        xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                             np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

        contour_coll = ax_plot.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.6)
        ax_plot.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)

        colors = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y]
        scatter_coll = ax_plot.scatter(X[:, 0], X[:, 1], c=colors, s=15, edgecolors="k", zorder=3)

        arch_str = f"2 → {' → '.join(map(str, hidden))} → 1" if hidden else "2 → 1"
        ax_plot.set_title(f"Adat: {ds_name} | Arch: [{arch_str}]\nPontosság: {acc*100:.1f}% | Paraméterek: {n_params}",
                          fontsize=12, fontweight='bold', color='black')

        fig.canvas.draw_idle()

    btn_train.on_clicked(update_plot)

    # Kezdeti rajz
    update_plot()
    plt.show()

if __name__ == "__main__":
    main()
