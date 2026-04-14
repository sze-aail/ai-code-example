"""
meretezesi_szabalyok.py — MLP méretezési szabályok: hány réteg, hány neuron?
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Kolmogorov / Hecht-Nielsen (1987): 2n+1 rejtett neuron
  - Kapacitás-szabály (Baum & Haussler, 1989): W ≤ N/α
  - Hüvelykujj-szabályok: (in+out)/2, √(in×out), 2×in
  - Osztályozási szabály: K(K-1)/2 neuron K osztályhoz
  - Empirikus validáció: túl kevés / ajánlott / túl sok neuron hatása
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_classification


# ══════════════════════════════════════════════════════════════
# Méretezési kalkulátor
# ══════════════════════════════════════════════════════════════

def sizing_rules(n_input, n_output, n_samples, n_classes=2):
    """Kiszámítja az összes méretezési ajánlást."""
    rules = {}

    # 1. Kolmogorov / Hecht-Nielsen
    rules["Kolmogorov (2n+1)"] = 2 * n_input + 1

    # 2. Hüvelykujj-szabályok
    rules["(in+out)/2"] = max(1, (n_input + n_output) // 2)
    rules["√(in×out)"] = max(1, int(np.sqrt(n_input * n_output)))
    rules["2×in"] = 2 * n_input

    # 3. Kapacitás-szabály (α=5 és α=10)
    for alpha in [5, 10]:
        n_h = max(1, n_samples // (alpha * (n_input + n_output)))
        rules[f"N/(α(in+out)), α={alpha}"] = n_h

    # 4. Osztályozási szabály
    if n_classes > 2:
        rules[f"K(K-1)/2, K={n_classes}"] = n_classes * (n_classes - 1) // 2

    return rules


def print_rules(n_input, n_output, n_samples, n_classes=2):
    """Szép kiírás."""
    rules = sizing_rules(n_input, n_output, n_samples, n_classes)
    print(f"\n  Bemenet: {n_input}, Kimenet: {n_output}, "
          f"Minták: {n_samples}, Osztályok: {n_classes}")
    print(f"  {'─' * 50}")
    for name, val in rules.items():
        print(f"  {name:30s} → {val:4d} neuron")
    return rules


# ══════════════════════════════════════════════════════════════
# 1. KÍSÉRLET: Szabályok összehasonlítása különböző feladatokon
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("1. MÉRETEZÉSI AJÁNLÁSOK KONKRÉT FELADATOKRA")
print("=" * 60)

scenarios = [
    ("Iris-szerű (kicsi)", 4, 1, 150, 3),
    ("MNIST-szerű (közepes)", 784, 10, 60000, 10),
    ("Tabular (jellemző)", 20, 1, 5000, 2),
    ("Kép regr. (nagy)", 1024, 1, 10000, 0),
]

fig1, ax1 = plt.subplots(figsize=(12, 6))
bar_data = {}

for name, n_in, n_out, n_samp, n_cls in scenarios:
    rules = print_rules(n_in, n_out, n_samp, n_cls)
    bar_data[name] = rules

# Vizualizáció
x_labels = list(bar_data.keys())
rule_names = ["Kolmogorov (2n+1)", "(in+out)/2", "√(in×out)", "2×in", "N/(α(in+out)), α=10"]
colors = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
x = np.arange(len(x_labels))
width = 0.15

for i, (rname, color) in enumerate(zip(rule_names, colors)):
    vals = []
    for scenario in x_labels:
        vals.append(bar_data[scenario].get(rname, 0))
    ax1.bar(x + i * width, vals, width, label=rname, color=color, alpha=0.8, edgecolor="k", linewidth=0.5)

ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_ylabel("Ajánlott rejtett neuronok száma", fontsize=11)
ax1.set_title("Méretezési szabályok összehasonlítása", fontsize=14, fontweight="bold")
ax1.legend(fontsize=9, loc="upper left")
ax1.set_yscale("log")
ax1.grid(True, alpha=0.3, axis="y")
fig1.tight_layout()
fig1.savefig("meretezesi_szabalyok.png", dpi=150)
print("\nÁbra mentve: meretezesi_szabalyok.png")


# ══════════════════════════════════════════════════════════════
# 2. KÍSÉRLET: Empirikus validáció — neuronszám hatása
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. EMPIRIKUS VALIDÁCIÓ: neuronszám hatása")
print("=" * 60)

def make_mlp(n_in, hidden, n_out):
    layers = []
    prev = n_in
    for h in hidden:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


def train_eval(model, X_train, y_train, X_test, y_test, epochs=500, lr=0.01, task='cls'):
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt_cls = torch.tensor(y_train, dtype=torch.long)
    yt_reg = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X_test, dtype=torch.float32)
    yv_cls = torch.tensor(y_test, dtype=torch.long)

    if task == 'cls':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_scores = [], []

    for epoch in range(epochs):
        model.train()
        pred = model(Xt)
        loss = criterion(pred, yt_cls if task == 'cls' else yt_reg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            if task == 'cls':
                val_pred = model(Xv).argmax(dim=1)
                score = (val_pred == yv_cls).float().mean().item()
            else:
                val_pred = model(Xv)
                score = 1 - criterion(val_pred, torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)).item()
        val_scores.append(score)

    return train_losses, val_scores

def main():
    # Feladat: 20 bemeneti jellemző, 4 osztály, 2000 minta
    n_in, n_out, n_samples, n_classes = 20, 4, 2000, 4
    X, y = make_classification(n_samples=n_samples, n_features=n_in, n_informative=12,
                               n_classes=n_classes, n_clusters_per_class=2, random_state=42)
    X = X.astype(np.float32)

    # Train/test split
    split = int(0.7 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    rules = sizing_rules(n_in, n_out, split, n_classes)
    print(f"\n  Feladat: {n_in} bemenet, {n_classes} osztály, {split} tanítóminta")
    for name, val in rules.items():
        print(f"    {name}: {val} neuron")

    # Teszteljük különböző méretekkel
    hidden_sizes = [2, 5, 10, int(rules["(in+out)/2"]), int(rules["Kolmogorov (2n+1)"]),
                    64, 128, 256, 512]
    hidden_sizes = sorted(set(hidden_sizes))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.5))
    test_accs = []
    train_final_losses = []

    for h in hidden_sizes:
        torch.manual_seed(42)
        model = make_mlp(n_in, [h], n_out)
        n_params = sum(p.numel() for p in model.parameters())
        tl, vs = train_eval(model, X_train, y_train, X_test, y_test, epochs=500)
        test_accs.append(vs[-1])
        train_final_losses.append(tl[-1])
        status = ""
        # Jelöljük meg a szabályokból származó értékeket
        for rname, rval in rules.items():
            if abs(h - rval) < 2:
                status = f" ← {rname}"
                break
        print(f"  h={h:4d} | params={n_params:6d} | test_acc={vs[-1]*100:.1f}%{status}")

    # Bal ábra: pontosság vs. neuronszám
    ax = axes2[0]
    ax.semilogx(hidden_sizes, [a * 100 for a in test_accs], "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Rejtett neuronok száma (log)", fontsize=12)
    ax.set_ylabel("Teszt pontosság (%)", fontsize=12)
    ax.set_title("Neuronszám hatása a pontosságra", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Szabályok jelölése
    rule_colors = {"Kolmogorov (2n+1)": "#E91E63", "(in+out)/2": "#2196F3",
                   "2×in": "#FF9800", "N/(α(in+out)), α=10": "#9C27B0"}
    for rname, rval in rules.items():
        if rname in rule_colors:
            ax.axvline(rval, color=rule_colors[rname], linestyle="--", alpha=0.7, linewidth=1.5)
            ax.text(rval * 1.1, ax.get_ylim()[0] + 2, rname, fontsize=7, rotation=90,
                    color=rule_colors[rname], va="bottom")

    # Jobb ábra: train loss vs. neuronszám
    ax = axes2[1]
    ax.semilogx(hidden_sizes, train_final_losses, "rs-", linewidth=2, markersize=8)
    ax.set_xlabel("Rejtett neuronok száma (log)", fontsize=12)
    ax.set_ylabel("Tanítási veszteség (végső)", fontsize=12)
    ax.set_title("Neuronszám hatása a tanítási hibára", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    for rname, rval in rules.items():
        if rname in rule_colors:
            ax.axvline(rval, color=rule_colors[rname], linestyle="--", alpha=0.7, linewidth=1.5)

    fig2.suptitle(f"Empirikus validáció: {n_in}D bemenet, {n_classes} osztály, {split} minta",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("meretezesi_empirikus.png", dpi=150)
    print("\nÁbra mentve: meretezesi_empirikus.png")


    # ══════════════════════════════════════════════════════════════
    # 3. KÍSÉRLET: Kapacitás-szabály — túl sok param = overfitting
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("3. KAPACITÁS: túl kevés / épp elég / túl sok paraméter")
    print("=" * 60)

    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
    X_moons = X_moons.astype(np.float32)
    split_m = 140

    fig3, axes3 = plt.subplots(1, 4, figsize=(18, 4.5))

    capacity_configs = [
        ("Túl kevés\n[2→2→1]", [2]),
        ("Ajánlott: (in+out)/2\n[2→2→1]... hm, √(in×out)\n[2→1→1]", [3]),
        ("Kolmogorov 2n+1=5\n[2→5→1]", [5]),
        ("Túl sok (overfit)\n[2→256→256→1]", [256, 256]),
    ]

    for ax, (name, hidden) in zip(axes3, capacity_configs):
        torch.manual_seed(42)
        model = make_mlp(2, hidden, 1)
        n_params = sum(p.numel() for p in model.parameters())

        # Tanítás
        Xt = torch.tensor(X_moons[:split_m])
        yt = torch.tensor(y_moons[:split_m], dtype=torch.float32).unsqueeze(1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(800):
            loss = criterion(model(Xt), yt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        # Döntési határ
        xx, yy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
        grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            zz = torch.sigmoid(model(grid)).numpy().reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=50, cmap="RdYlGn", alpha=0.6)
        ax.contour(xx, yy, zz, levels=[0.5], colors="k", linewidths=2)

        # Train + test pontok
        c_train = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y_moons[:split_m]]
        c_test = ["#E91E63" if yi == 0 else "#4CAF50" for yi in y_moons[split_m:]]
        ax.scatter(X_moons[:split_m, 0], X_moons[:split_m, 1], c=c_train, s=20,
                   edgecolors="k", linewidth=0.3, zorder=3, label="Train")
        ax.scatter(X_moons[split_m:, 0], X_moons[split_m:, 1], c=c_test, s=40,
                   edgecolors="k", linewidth=1, zorder=3, marker="^", label="Test")

        # Pontosságok
        with torch.no_grad():
            train_acc = ((torch.sigmoid(model(Xt)) > 0.5).float() == yt).float().mean().item()
            Xv = torch.tensor(X_moons[split_m:])
            yv = torch.tensor(y_moons[split_m:], dtype=torch.float32).unsqueeze(1)
            test_acc = ((torch.sigmoid(model(Xv)) > 0.5).float() == yv).float().mean().item()

        ax.set_title(f"{name}\n{n_params} param | train={train_acc*100:.0f}% test={test_acc*100:.0f}%",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)
        print(f"  {hidden}: {n_params} params, train={train_acc*100:.1f}%, test={test_acc*100:.1f}%")

    fig3.suptitle("Kapacitás-szabály: W ≤ N/α — túl sok paraméter = overfitting",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("meretezesi_kapacitas.png", dpi=150)
    print("Ábra mentve: meretezesi_kapacitas.png")

    plt.close("all")
    print("\nKész!")

if __name__=="__main__":
    main()
