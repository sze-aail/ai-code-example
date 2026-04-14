"""
tobbretegu_rbf_halo.py — Többrétegű RBF neurális háló backpropagation-nel
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - RBF neuron: φ(||x - c||) belső szorzat (⟨w,x⟩) helyett távolságalapú aktiváció
  - Tanulható centrumok (c), szélességek (σ) és kimeneti súlyok (W) — mind backproppal
  - Többrétegű RBF: RBF→RBF→Linear (nem csak a klasszikus egyrétegű!)
  - Cserélhető távolságmetrikák: L2, L1 (smooth), Mahalanobis, koszinusz
  - Összehasonlítás klasszikus MLP-vel azonos paraméterszám mellett
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_swiss_roll


# ══════════════════════════════════════════════════════════════
# RBF RÉTEG: differenciálható, backprop-kompatibilis
# ══════════════════════════════════════════════════════════════

class RBFLayer(nn.Module):
    """
    Radiális bázisfüggvény réteg.

    Minden neuron j egy centroid c_j és szélesség σ_j:
      h_j(x) = φ(d(x, c_j) / σ_j)

    ahol φ a Gauss-függvény: exp(-z²), és d() a választott távolság.
    Minden paraméter (c, σ) tanulható backprop-pal.
    """

    def __init__(self, in_features, n_centers, distance='l2'):
        super().__init__()
        self.in_features = in_features
        self.n_centers = n_centers
        self.distance = distance

        # Tanulható paraméterek
        self.centers = nn.Parameter(torch.randn(n_centers, in_features) * 0.5)
        self.log_sigmas = nn.Parameter(torch.zeros(n_centers))  # log(σ) a pozitivitásért

        # Mahalanobis-hoz: tanulható L mátrix (σ_j helyett L_j @ L_j^T)
        if distance == 'mahalanobis':
            self.L = nn.Parameter(torch.eye(in_features).unsqueeze(0).repeat(n_centers, 1, 1) * 0.5)

    def forward(self, x):
        """
        x: (batch, in_features)
        kimenet: (batch, n_centers) — minden centroidhoz egy aktiváció
        """
        # x: (B, D) → (B, 1, D), centers: (1, C, D)
        x_expanded = x.unsqueeze(1)  # (B, 1, D)
        c_expanded = self.centers.unsqueeze(0)  # (1, C, D)
        diff = x_expanded - c_expanded  # (B, C, D)

        # Távolság számítása
        if self.distance == 'l2':
            dist_sq = torch.sum(diff ** 2, dim=-1)  # (B, C)

        elif self.distance == 'l1':
            # Sima L1 approximáció (differenciálható): √(x² + ε)
            eps = 1e-6
            dist_sq = torch.sum(torch.sqrt(diff ** 2 + eps), dim=-1) ** 2

        elif self.distance == 'linf':
            # Smooth L∞: logsumexp approximáció
            alpha = 10.0
            dist_sq = (torch.logsumexp(alpha * diff.abs(), dim=-1) / alpha) ** 2

        elif self.distance == 'cosine':
            # Koszinusz-alapú: 1 - cos(x, c)
            cos_sim = F.cosine_similarity(x_expanded, c_expanded, dim=-1)
            dist_sq = (1 - cos_sim) ** 2

        elif self.distance == 'mahalanobis':
            # d² = (x-c)ᵀ M (x-c), ahol M = LLᵀ (pozitív definit)
            # diff: (B, C, D) → matmul with L: (C, D, D)
            Ldiff = torch.einsum('cij,bcj->bci', self.L, diff)  # (B, C, D)
            dist_sq = torch.sum(Ldiff ** 2, dim=-1)  # (B, C)

        else:
            raise ValueError(f"Ismeretlen távolság: {self.distance}")

        # Gauss aktiváció: exp(-d² / (2σ²))
        sigmas = torch.exp(self.log_sigmas)  # pozitív σ
        return torch.exp(-dist_sq / (2 * sigmas ** 2 + 1e-8))


class MultiLayerRBF(nn.Module):
    """
    Többrétegű RBF háló: RBF → RBF → ... → Linear

    Ez NEM a klasszikus egyrétegű RBF hálózat!
    Az egymás utáni RBF rétegek hierarchikus távollságteret tanulnak.
    """

    def __init__(self, in_features, hidden_sizes, out_features, distance='l2'):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(RBFLayer(prev, h, distance=distance))
            prev = h
        self.rbf_layers = nn.ModuleList(layers)
        self.output = nn.Linear(prev, out_features)

    def forward(self, x):
        for layer in self.rbf_layers:
            x = layer(x)
        return self.output(x)


class StandardMLP(nn.Module):
    """Összehasonlításhoz: hagyományos MLP azonos struktúrával."""

    def __init__(self, in_features, hidden_sizes, out_features):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# SEGÉDFÜGGVÉNYEK
# ══════════════════════════════════════════════════════════════

def train_model(model, X, y, epochs=500, lr=0.01, task='classification'):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if task == 'classification':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def plot_decision(ax, model, X, y, title, resolution=150):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                         np.linspace(y_min, y_max, resolution))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

    ax.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    colors = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y.numpy().flatten()]
    ax.scatter(X[:, 0].numpy(), X[:, 1].numpy(), c=colors, s=25,
               edgecolors="k", linewidth=0.3, zorder=3)

    # Centrumok kirajzolása (ha RBF modell)
    if hasattr(model, 'rbf_layers'):
        centers = model.rbf_layers[0].centers.detach().numpy()
        ax.scatter(centers[:, 0], centers[:, 1], c="gold", s=80, marker="*",
                   edgecolors="k", linewidth=0.8, zorder=4, label=f"Centrumok ({len(centers)})")
        ax.legend(fontsize=7, loc="lower right")

    # Paraméterszám
    n_params = sum(p.numel() for p in model.parameters())
    acc = ((torch.sigmoid(model(X)) > 0.5).float() == y).float().mean().item()
    ax.set_title(f"{title}\nacc={acc*100:.1f}%, params={n_params}", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.2)


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. KÍSÉRLET: RBF vs MLP — Make Moons
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. KÍSÉRLET: Egyrétegű RBF vs. MLP (Make Moons)")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    X_moons, y_moons = make_moons(n_samples=500, noise=0.15, random_state=42)
    X_m = torch.tensor(X_moons, dtype=torch.float32)
    y_m = torch.tensor(y_moons, dtype=torch.float32).unsqueeze(1)

    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 5))

    configs_1 = [
        ("MLP [2→16→16→1]", StandardMLP(2, [16, 16], 1)),
        ("RBF 1 réteg [2→(16)→1]", MultiLayerRBF(2, [16], 1, distance='l2')),
        ("RBF 2 réteg [2→(16)→(8)→1]", MultiLayerRBF(2, [16, 8], 1, distance='l2')),
    ]

    for ax, (name, model) in zip(axes1, configs_1):
        losses = train_model(model, X_m, y_m, epochs=600, lr=0.01)
        plot_decision(ax, model, X_m, y_m, name)
        print(f"  {name}: loss={losses[-1]:.4f}")

    fig1.suptitle("Egyrétegű vs. többrétegű RBF (L2) — Make Moons", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("rbf_vs_mlp_moons.png", dpi=150)
    print("Ábra mentve: rbf_vs_mlp_moons.png\n")


    # ══════════════════════════════════════════════════════════════
    # 2. KÍSÉRLET: Különböző távolságmetrikák
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("2. KÍSÉRLET: Távolságmetrikák hatása (Make Circles)")
    print("=" * 60)

    X_circles, y_circles = make_circles(n_samples=500, noise=0.1, factor=0.4, random_state=42)
    X_c = torch.tensor(X_circles, dtype=torch.float32)
    y_c = torch.tensor(y_circles, dtype=torch.float32).unsqueeze(1)

    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 4.5))

    distances = [
        ("L2 (euklideszi)", "l2"),
        ("L1 (smooth Manhattan)", "l1"),
        ("L∞ (smooth Csebisev)", "linf"),
        ("Koszinusz", "cosine"),
        ("Mahalanobis", "mahalanobis"),
    ]

    for ax, (name, dist) in zip(axes2, distances):
        torch.manual_seed(42)
        model = MultiLayerRBF(2, [20], 1, distance=dist)
        losses = train_model(model, X_c, y_c, epochs=800, lr=0.01)
        plot_decision(ax, model, X_c, y_c, f"RBF ({name})")
        print(f"  {name}: loss={losses[-1]:.4f}")

    fig2.suptitle("Távolságmetrikák hatása az RBF döntési határra — Make Circles",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("rbf_tavolsagmetrikak.png", dpi=150)
    print("Ábra mentve: rbf_tavolsagmetrikak.png\n")


    # ══════════════════════════════════════════════════════════════
    # 3. KÍSÉRLET: Többrétegű RBF — mély vs. sekély
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("3. KÍSÉRLET: Mélység hatása — spirál adathalmaz")
    print("=" * 60)

    # Spirál adat (nehezebb feladat)
    np.random.seed(42)
    n_per_class = 250
    theta1 = np.linspace(0, 4 * np.pi, n_per_class) + np.random.randn(n_per_class) * 0.3
    r1 = np.linspace(0.2, 2, n_per_class)
    theta2 = np.linspace(0, 4 * np.pi, n_per_class) + np.pi + np.random.randn(n_per_class) * 0.3
    r2 = np.linspace(0.2, 2, n_per_class)

    X_spiral = np.vstack([
        np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)],
        np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)],
    ]).astype(np.float32)
    y_spiral = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.float32)

    X_s = torch.tensor(X_spiral)
    y_s = torch.tensor(y_spiral).unsqueeze(1)

    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.5))

    configs_3 = [
        ("MLP [2→32→32→1]", StandardMLP(2, [32, 32], 1)),
        ("RBF 1 réteg [2→(32)→1]", MultiLayerRBF(2, [32], 1)),
        ("RBF 2 réteg [2→(24)→(16)→1]", MultiLayerRBF(2, [24, 16], 1)),
        ("RBF 3 réteg [2→(20)→(16)→(8)→1]", MultiLayerRBF(2, [20, 16, 8], 1)),
    ]

    for ax, (name, model) in zip(axes3, configs_3):
        torch.manual_seed(42)
        losses = train_model(model, X_s, y_s, epochs=1500, lr=0.005)
        plot_decision(ax, model, X_s, y_s, name)
        print(f"  {name}: loss={losses[-1]:.4f}")

    fig3.suptitle("Mélység hatása: spirál adathalmaz (nehéz nemlineáris probléma)",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("rbf_melyseg_spiral.png", dpi=150)
    print("Ábra mentve: rbf_melyseg_spiral.png\n")


    # ══════════════════════════════════════════════════════════════
    # 4. KÍSÉRLET: Regresszió — RBF mint univerzális approximátor
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("4. KÍSÉRLET: Regresszió — RBF mint univerzális approximátor")
    print("=" * 60)

    x_reg = torch.linspace(-3, 3, 300).unsqueeze(1)
    y_true = torch.sin(x_reg) + 0.5 * torch.sin(3 * x_reg)
    y_reg = y_true + torch.randn_like(y_true) * 0.15

    fig4, axes4 = plt.subplots(1, 3, figsize=(16, 5))

    reg_configs = [
        ("MLP [1→32→32→1]", StandardMLP(1, [32, 32], 1)),
        ("RBF L2 [1→(20)→(10)→1]", MultiLayerRBF(1, [20, 10], 1, distance='l2')),
        ("RBF Mahalanobis [1→(20)→(10)→1]", MultiLayerRBF(1, [20, 10], 1, distance='mahalanobis')),
    ]

    for ax, (name, model) in zip(axes4, reg_configs):
        torch.manual_seed(42)
        losses = train_model(model, x_reg, y_reg, epochs=1000, lr=0.005, task='regression')

        model.eval()
        with torch.no_grad():
            y_pred = model(x_reg).numpy().flatten()

        ax.plot(x_reg.numpy(), y_true.numpy(), "g--", linewidth=1.5, alpha=0.7, label="Valódi f(x)")
        ax.plot(x_reg.numpy(), y_pred, "r-", linewidth=2, label="Modell")
        ax.scatter(x_reg.numpy()[::8], y_reg.numpy()[::8], s=10, c="#2196F3", alpha=0.4)

        n_params = sum(p.numel() for p in model.parameters())
        ax.set_title(f"{name}\nMSE={losses[-1]:.4f}, params={n_params}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        print(f"  {name}: MSE={losses[-1]:.4f}")

    fig4.suptitle("Regresszió: RBF háló mint univerzális approximátor", fontsize=14, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig("rbf_regresszio.png", dpi=150)
    print("Ábra mentve: rbf_regresszio.png\n")


    # ══════════════════════════════════════════════════════════════
    # 5. TANULÁSI DINAMIKA: centrumok mozgása
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("5. CENTRUMOK MOZGÁSA TANÍTÁS KÖZBEN")
    print("=" * 60)

    torch.manual_seed(42)
    model_track = MultiLayerRBF(2, [12], 1, distance='l2')
    optimizer = torch.optim.Adam(model_track.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    snapshots = []
    snapshot_epochs = [0, 20, 50, 100, 200, 500]

    for epoch in range(501):
        if epoch in snapshot_epochs:
            centers = model_track.rbf_layers[0].centers.detach().clone().numpy()
            sigmas = torch.exp(model_track.rbf_layers[0].log_sigmas).detach().clone().numpy()
            snapshots.append((epoch, centers, sigmas))

        pred = model_track(X_m)
        loss = criterion(pred, y_m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fig5, axes5 = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (epoch, centers, sigmas) in zip(axes5.flat, snapshots):
        colors_data = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y_m.numpy().flatten()]
        ax.scatter(X_m[:, 0].numpy(), X_m[:, 1].numpy(), c=colors_data, s=10, alpha=0.3)

        for j in range(len(centers)):
            circle = plt.Circle(centers[j], sigmas[j], fill=False, color="gold",
                                linewidth=1.5, linestyle="--", alpha=0.7)
            ax.add_patch(circle)
            ax.plot(centers[j, 0], centers[j, 1], "k*", markersize=8)

        ax.set_xlim(-2, 3)
        ax.set_ylim(-1.5, 2)
        ax.set_aspect("equal")
        ax.set_title(f"Epoch {epoch}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig5.suptitle("RBF centrumok és szélességek (σ) mozgása tanítás közben — backprop optimalizálja!",
                  fontsize=14, fontweight="bold")
    fig5.tight_layout()
    fig5.savefig("rbf_centrum_mozgas.png", dpi=150)
    print("Ábra mentve: rbf_centrum_mozgas.png")

    plt.close("all")
    print("\nKész!")

if __name__ == "__main__":
    main()