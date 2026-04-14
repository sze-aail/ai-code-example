"""
tobbretegu_rbf_interaktiv.py — Interaktív többrétegű RBF neurális háló
Neurális hálók I. — Hajdu Csaba

Interaktív vizualizáció:
A felhasználó kiválaszthatja az adathalmazt, a távolságmetrikát, valamint
a két lehetséges RBF réteg méretét (0 esetén a réteg kimarad).
A "Tanítás és Frissítés" gombra kattintva a háló betanul, majd az ábrán
megjelenik a döntési határ és az első RBF réteg centrumai.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.datasets import make_moons, make_circles, make_blobs

# ══════════════════════════════════════════════════════════════
# Modellosztályok (másolat tobbretegu_rbf.py-ből)
# ══════════════════════════════════════════════════════════════

class RBFLayer(nn.Module):
    def __init__(self, in_features, n_centers, distance='l2'):
        super().__init__()
        self.in_features = in_features
        self.n_centers = n_centers
        self.distance = distance

        self.centers = nn.Parameter(torch.randn(n_centers, in_features) * 0.5)
        self.log_sigmas = nn.Parameter(torch.zeros(n_centers))

        if distance == 'mahalanobis':
            self.L = nn.Parameter(torch.eye(in_features).unsqueeze(0).repeat(n_centers, 1, 1) * 0.5)

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        c_expanded = self.centers.unsqueeze(0)
        diff = x_expanded - c_expanded

        if self.distance == 'l2':
            dist_sq = torch.sum(diff ** 2, dim=-1)
        elif self.distance == 'l1':
            eps = 1e-6
            dist_sq = torch.sum(torch.sqrt(diff ** 2 + eps), dim=-1) ** 2
        elif self.distance == 'linf':
            alpha = 10.0
            dist_sq = (torch.logsumexp(alpha * diff.abs(), dim=-1) / alpha) ** 2
        elif self.distance == 'cosine':
            cos_sim = F.cosine_similarity(x_expanded, c_expanded, dim=-1)
            dist_sq = (1 - cos_sim) ** 2
        elif self.distance == 'mahalanobis':
            Ldiff = torch.einsum('cij,bcj->bci', self.L, diff)
            dist_sq = torch.sum(Ldiff ** 2, dim=-1)
        else:
            raise ValueError(f"Ismeretlen távolság: {self.distance}")

        sigmas = torch.exp(self.log_sigmas)
        return torch.exp(-dist_sq / (2 * sigmas ** 2 + 1e-8))

class MultiLayerRBF(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, distance='l2'):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            if h > 0:
                layers.append(RBFLayer(prev, h, distance=distance))
                prev = h
        self.rbf_layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev, out_features)

    def forward(self, x):
        for layer in self.rbf_layers:
            x = layer(x)
        return self.output(x)

# ══════════════════════════════════════════════════════════════
# Adatgenerátorok
# ══════════════════════════════════════════════════════════════

def get_data(name):
    np.random.seed(42)
    if name == 'Blobs':
        X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.2, random_state=42)
    elif name == 'Moons':
        X, y = make_moons(n_samples=300, noise=0.15, random_state=42)
    elif name == 'Circles':
        X, y = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)
    elif name == 'Spirál':
        n_per_class = 150
        theta1 = np.linspace(0, 4 * np.pi, n_per_class) + np.random.randn(n_per_class) * 0.3
        r1 = np.linspace(0.2, 2, n_per_class)
        theta2 = np.linspace(0, 4 * np.pi, n_per_class) + np.pi + np.random.randn(n_per_class) * 0.3
        r2 = np.linspace(0.2, 2, n_per_class)
        X = np.vstack([
            np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)],
            np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)],
        ]).astype(np.float32)
        y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.float32)

    return X.astype(np.float32), y.astype(np.float32)

# ══════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(13, 7))
    ax_plot = plt.axes([0.35, 0.1, 0.6, 0.8])

    # Vezérlők
    ax_radio_data = plt.axes([0.05, 0.70, 0.20, 0.20])
    ax_radio_dist = plt.axes([0.05, 0.45, 0.20, 0.20])

    ax_slider_h1 = plt.axes([0.05, 0.35, 0.20, 0.03])
    ax_slider_h2 = plt.axes([0.05, 0.25, 0.20, 0.03])
    ax_slider_epochs = plt.axes([0.05, 0.15, 0.20, 0.03])

    ax_btn = plt.axes([0.05, 0.05, 0.20, 0.06])

    radio_data = RadioButtons(ax_radio_data, ['Blobs', 'Moons', 'Circles', 'Spirál'])
    radio_dist = RadioButtons(ax_radio_dist, ['l2', 'l1', 'linf', 'cosine', 'mahalanobis'])

    s_h1 = Slider(ax_slider_h1, 'RBF 1. réteg', 1, 64, valinit=16, valstep=1)
    s_h2 = Slider(ax_slider_h2, 'RBF 2. réteg', 0, 32, valinit=0, valstep=1)
    s_epochs = Slider(ax_slider_epochs, 'Epochs', 50, 2000, valinit=500, valstep=50)

    btn_train = Button(ax_btn, 'Tanítás és Frissítés', color='lightgreen', hovercolor='palegreen')

    def update_plot(event=None):
        ax_plot.clear()

        data_name = radio_data.value_selected
        dist_name = radio_dist.value_selected
        h1 = int(s_h1.val)
        h2 = int(s_h2.val)
        epochs = int(s_epochs.val)

        X, y = get_data(data_name)
        X_t = torch.tensor(X)
        y_t = torch.tensor(y).unsqueeze(1)

        ax_plot.set_title("Tanítás folyamatban...", color='red')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        hidden = [h1]
        if h2 > 0: hidden.append(h2)

        torch.manual_seed(42)
        model = MultiLayerRBF(2, hidden, 1, distance=dist_name)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        for p in range(epochs):
            pred = model(X_t)
            loss = criterion(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = ((torch.sigmoid(model(X_t)) > 0.5).float() == y_t).float().mean().item()

        # Kirajzolás
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                             np.linspace(y_min, y_max, 150))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

        ax_plot.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.5)
        ax_plot.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)

        colors = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y]
        ax_plot.scatter(X[:, 0], X[:, 1], c=colors, s=30, edgecolors="k", zorder=3)

        if len(model.rbf_layers) > 0:
            centers = model.rbf_layers[0].centers.detach().numpy()
            ax_plot.scatter(centers[:, 0], centers[:, 1], c="gold", s=120, marker="*",
                            edgecolors="k", linewidth=1, zorder=4, label=f"Centrumok ({len(centers)})")
            ax_plot.legend(loc="upper right")

        n_params = sum(p.numel() for p in model.parameters())
        arch_str = f"2 → {' → '.join(map(str, hidden))} → 1"
        ax_plot.set_title(f"Adat: {data_name} | Arch: [{arch_str}] | Táv.: {dist_name}\nPontosság: {acc*100:.1f}% | Paraméterek: {n_params}",
                          fontsize=12, fontweight='bold')
        ax_plot.set_xlim(x_min, x_max)
        ax_plot.set_ylim(y_min, y_max)

        fig.canvas.draw_idle()

    btn_train.on_clicked(update_plot)
    update_plot(None)

    plt.show()

if __name__ == "__main__":
    main()
