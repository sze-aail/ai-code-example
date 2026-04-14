"""
retegszam_geometria.py — Rétegszám geometriai szabálya és mélység-szélesség tradeoff
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - 0 rejtett réteg: félsík (lineáris határ)
  - 1 rejtett réteg: konvex régiók (félsíkok metszete)
  - 2 rejtett réteg: tetszőleges régiók (konvex régiók uniója)
  - Mélység vs. szélesség tradeoff azonos paraméterszám mellett
  - Kolmogorov-reprezentációs tétel szemléltetése
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ══════════════════════════════════════════════════════════════
# Adatgenerátorok a három geometriai szinthez
# ══════════════════════════════════════════════════════════════

def make_linear_data(n=400, seed=42):
    """Lineárisan szeparálható (félsík)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 2).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)
    return X, y

def make_convex_data(n=600, seed=42):
    """Konvex régió: háromszög belseje."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
    # Háromszög csúcsai: (0, 1.5), (-1.5, -1), (1.5, -1)
    def in_triangle(p):
        v0 = np.array([0, 1.5])
        v1 = np.array([-1.5, -1])
        v2 = np.array([1.5, -1])
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

def make_nonconvex_data(n=800, seed=42):
    """Nem-konvex régió: két szétválasztott klaszter (konvex régiók uniója)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-3, 3, (n, 2)).astype(np.float32)
    # Két négyzet: bal-felső + jobb-alsó (nem konvex unió)
    in_box1 = (X[:, 0] > -2.5) & (X[:, 0] < -0.5) & (X[:, 1] > 0.3) & (X[:, 1] < 2.3)
    in_box2 = (X[:, 0] > 0.5) & (X[:, 0] < 2.5) & (X[:, 1] > -2.3) & (X[:, 1] < -0.3)
    y = (in_box1 | in_box2).astype(np.float32)
    return X, y

def make_xor_regions(n=800, seed=42):
    """XOR-szerű: 4 sarok, átlós pozitív (nem konvex, nem összefüggő)."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(np.float32)
    return X, y

def make_donut_data(n=600, seed=42):
    """Gyűrű: nem konvex, körkörös határ."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(-2.5, 2.5, (n, 2)).astype(np.float32)
    r = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    y = ((r > 0.8) & (r < 1.8)).astype(np.float32)
    return X, y


# ══════════════════════════════════════════════════════════════
# Modellek
# ══════════════════════════════════════════════════════════════

def make_mlp(hidden_layers):
    """MLP építése réteglistából."""
    layers = []
    prev = 2
    for h in hidden_layers:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_and_get_acc(model, X, y, epochs=1000, lr=0.01):
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        pred = model(X_t)
        loss = criterion(pred, y_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        acc = ((torch.sigmoid(model(X_t)) > 0.5).float() == y_t).float().mean().item()
    return acc


def plot_result(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                         np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.6)
    ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)
    colors = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y]
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=12, edgecolors="k", linewidth=0.2, alpha=0.7, zorder=3)
    n_params = sum(p.numel() for p in model.parameters())
    X_t = torch.tensor(X)
    y_t = torch.tensor(y).unsqueeze(1)
    with torch.no_grad():
        acc = ((torch.sigmoid(model(X_t)) > 0.5).float() == y_t).float().mean().item()
    ax.set_title(f"{title}\nacc={acc*100:.1f}% | {n_params} param", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.2)


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. KÍSÉRLET: A geometriai szabály demonstrációja
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. GEOMETRIAI RÉTEGSZABÁLY")
    print("   0 rejtett = félsík | 1 rejtett = konvex | 2 rejtett = tetszőleges")
    print("=" * 60)

    datasets = [
        ("Félsík\n(lineáris)", make_linear_data()),
        ("Háromszög\n(konvex)", make_convex_data()),
        ("Két doboz\n(nem-konvex)", make_nonconvex_data()),
        ("XOR régiók\n(nem-konvex)", make_xor_regions()),
        ("Gyűrű\n(nem-konvex)", make_donut_data()),
    ]

    architectures = [
        ("0 rejtett\n[2→1]", []),
        ("1 rejtett\n[2→16→1]", [16]),
        ("2 rejtett\n[2→16→16→1]", [16, 16]),
    ]

    fig1, axes1 = plt.subplots(len(architectures), len(datasets), figsize=(22, 13))

    for col, (ds_name, (X, y)) in enumerate(datasets):
        for row, (arch_name, hidden) in enumerate(architectures):
            ax = axes1[row, col]
            torch.manual_seed(42)
            model = make_mlp(hidden) if hidden else nn.Sequential(nn.Linear(2, 1))
            train_and_get_acc(model, X, y, epochs=1500 if hidden else 500)
            plot_result(ax, model, X, y, f"{arch_name}" if col == 0 else "")
            if row == 0:
                ax.set_title(f"{ds_name}\n{ax.get_title()}", fontsize=9, fontweight="bold")

    # Sor címkék
    for row, (arch_name, _) in enumerate(architectures):
        axes1[row, 0].set_ylabel(arch_name, fontsize=11, fontweight="bold", rotation=0,
                                  labelpad=80, va="center")

    fig1.suptitle("Geometriai rétegszabály: félsík → konvex → tetszőleges régió",
                  fontsize=16, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("retegszam_geometria.png", dpi=150)
    print("Ábra mentve: retegszam_geometria.png\n")


    # ══════════════════════════════════════════════════════════════
    # 2. KÍSÉRLET: Mélység vs. szélesség tradeoff
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("2. MÉLYSÉG vs. SZÉLESSÉG (hasonló paraméterszám)")
    print("=" * 60)

    # Spirál adat (nehéz)
    np.random.seed(42)
    n_per = 300
    t1 = np.linspace(0, 4 * np.pi, n_per) + np.random.randn(n_per) * 0.25
    r1 = np.linspace(0.3, 2, n_per)
    t2 = t1 + np.pi
    r2 = r1
    X_sp = np.vstack([np.c_[r1*np.cos(t1), r1*np.sin(t1)],
                       np.c_[r2*np.cos(t2), r2*np.sin(t2)]]).astype(np.float32)
    y_sp = np.array([0]*n_per + [1]*n_per, dtype=np.float32)

    # ~200 paraméter mindegyikben
    configs_depth = [
        ("Sekély-széles\n[2→200→1]", [200]),            # 200*2+200+200*1+1 = 801
        ("Közepes\n[2→32→32→1]", [32, 32]),             # 32*2+32+32*32+32+32*1+1 = 1185
        ("Mély-keskeny\n[2→16→16→16→1]", [16, 16, 16]), # 16*2+16+16*16+16+16*16+16+16*1+1 = 609
        ("Nagyon mély\n[2→8→8→8→8→8→1]", [8]*5),        # sok réteg, kevés neuron
    ]

    fig2, axes2 = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, (name, hidden) in zip(axes2, configs_depth):
        torch.manual_seed(42)
        model = make_mlp(hidden)
        n_params = sum(p.numel() for p in model.parameters())
        train_and_get_acc(model, X_sp, y_sp, epochs=2000, lr=0.005)
        plot_result(ax, model, X_sp, y_sp, f"{name}\n({n_params} param)")
        with torch.no_grad():
            X_t = torch.tensor(X_sp)
            y_t = torch.tensor(y_sp).unsqueeze(1)
            acc = ((torch.sigmoid(model(X_t)) > 0.5).float() == y_t).float().mean().item()
        print(f"  {name.replace(chr(10), ' ')}: {n_params} params, acc={acc*100:.1f}%")

    fig2.suptitle("Mélység vs. szélesség: spirál adathalmaz (azonos feladat, eltérő architektúra)",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("melyseg_vs_szelesseg.png", dpi=150)
    print("Ábra mentve: melyseg_vs_szelesseg.png\n")


    # ══════════════════════════════════════════════════════════════
    # 3. ELVI ÁBRA: Rétegszám → régió bonyolultság
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("3. ELVI ÁBRA")
    print("=" * 60)

    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.5))

    # 0 rejtett: félsík
    ax = axes3[0]
    xx, yy = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
    zz = xx + 0.5 * yy
    ax.contourf(xx, yy, zz, levels=[-10, 0, 10], colors=["#FFCDD2", "#C8E6C9"], alpha=0.8)
    ax.contour(xx, yy, zz, levels=[0], colors="k", linewidths=2)
    ax.set_title("0 rejtett réteg\nFÉLSÍK", fontsize=12, fontweight="bold")
    ax.text(0.7, 1.2, "+", fontsize=20, fontweight="bold", color="#4CAF50")
    ax.text(-1.2, -1.0, "−", fontsize=20, fontweight="bold", color="#E91E63")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    # 1 rejtett: konvex metszet
    ax = axes3[1]
    z1 = xx + yy - 0.5
    z2 = -xx + yy - 0.5
    z3 = -yy - 0.5
    inside = (z1 < 0) & (z2 < 0) & (z3 < 0)
    ax.contourf(xx, yy, inside.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["#FFCDD2", "#C8E6C9"], alpha=0.8)
    ax.contour(xx, yy, inside.astype(float), levels=[0.5], colors="k", linewidths=2)
    ax.set_title("1 rejtett réteg\nKONVEX RÉGIÓ\n(félsíkok ∩ metszete)", fontsize=12, fontweight="bold")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    # 2 rejtett: tetszőleges (két konvex uniója)
    ax = axes3[2]
    box1 = (xx > -1.8) & (xx < -0.3) & (yy > 0.2) & (yy < 1.7)
    box2 = (xx > 0.3) & (xx < 1.8) & (yy > -1.7) & (yy < -0.2)
    inside2 = box1 | box2
    ax.contourf(xx, yy, inside2.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=["#FFCDD2", "#C8E6C9"], alpha=0.8)
    ax.contour(xx, yy, inside2.astype(float), levels=[0.5], colors="k", linewidths=2)
    ax.set_title("2 rejtett réteg\nTETSZŐLEGES RÉGIÓ\n(konvex régiók ∪ uniója)", fontsize=12, fontweight="bold")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2); ax.set_aspect("equal"); ax.grid(True, alpha=0.2)

    # Összefoglaló szöveg
    ax = axes3[3]
    ax.axis("off")
    summary = [
        ("0 rejtett", "félsík", "w₁x₁+w₂x₂+b > 0", "1 hipersík"),
        ("1 rejtett\n(n neuron)", "konvex régió", "∩ félsíkok", "n hipersík\nmetszete"),
        ("2 rejtett\n(n×m neuron)", "tetszőleges", "∪ konvex\nrégiók", "m konvex\nrégió uniója"),
    ]
    for i, (depth, region, formula, desc) in enumerate(summary):
        y_pos = 0.82 - i * 0.32
        ax.text(0.05, y_pos, depth, fontsize=11, fontweight="bold", transform=ax.transAxes, va="top")
        ax.text(0.35, y_pos, f"→ {region}", fontsize=11, transform=ax.transAxes, va="top", color="#1565C0")
        ax.text(0.35, y_pos - 0.1, f"   {formula}", fontsize=9, transform=ax.transAxes, va="top",
                color="gray", style="italic")

    ax.text(0.05, 0.05, "Lippmann (1987)\nCybenko (1989)\nHornik (1991)",
            fontsize=8, transform=ax.transAxes, va="bottom", color="gray")
    ax.set_title("Összefoglalás", fontsize=12, fontweight="bold")

    fig3.suptitle("Rétegszám → geometriai komplexitás", fontsize=15, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("retegszam_elvi.png", dpi=150)
    print("Ábra mentve: retegszam_elvi.png")

    plt.close("all")
    print("\nKész!")

if __name__ == "__main__":
    main()
