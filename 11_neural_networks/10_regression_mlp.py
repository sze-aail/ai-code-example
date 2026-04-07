"""
10_regression_mlp.py — Regresszió MLP-vel: univerzális approximáció
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Folytonos függvény közelítése (sin, kompozit)
  - Modellkapacitás hatása (kevés vs. sok neuron)
  - Univerzális approximációs tétel szemléltetése
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ── Célfüggvény: összetett, nemlineáris ──
def target_fn(x):
    return np.sin(x) + 0.5 * np.sin(3 * x) + 0.3 * np.cos(5 * x)


def main():
    torch.manual_seed(42)
    np.random.seed(42)



    n = 200
    X_np = np.sort(np.random.uniform(-4, 4, n)).astype(np.float32)
    y_np = target_fn(X_np).astype(np.float32) + np.random.randn(n).astype(np.float32) * 0.15

    Xt = torch.tensor(X_np).unsqueeze(1)
    yt = torch.tensor(y_np).unsqueeze(1)
    x_dense = torch.linspace(-4.5, 4.5, 500).unsqueeze(1)
    y_true = target_fn(x_dense.numpy().flatten())


    # ── Különböző kapacitású MLP-k ──
    configs = [
        ("4 neuron (alultanulás)", [4]),
        ("16 neuron", [16]),
        ("64 neuron (2 réteg)", [64, 64]),
        ("256 neuron (3 réteg)", [128, 128, 64]),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, hidden_sizes) in zip(axes.flat, configs):
        layers = []
        in_dim = 1
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()

        for epoch in range(1500):
            pred = model(Xt)
            loss = criterion(pred, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        print(f"{name}: MSE = {final_loss:.5f}")

        model.eval()
        with torch.no_grad():
            y_pred = model(x_dense).numpy().flatten()

        ax.plot(x_dense.numpy(), y_true, "g--", linewidth=1.5, alpha=0.7, label="Valódi f(x)")
        ax.plot(x_dense.numpy(), y_pred, "r-", linewidth=2, label="MLP közelítés")
        ax.scatter(X_np[::5], y_np[::5], s=15, c="#2196F3", alpha=0.5, label="Tanítóadatok")
        ax.set_title(f"{name}\nMSE = {final_loss:.5f}", fontsize=11, fontweight="bold")
        ax.set_ylim(-2.5, 2.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Univerzális approximáció: modellkapacitás hatása", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig("10_regression.png", dpi=150)
    print("\nÁbra mentve: 10_regression.png")

    plt.close("all")
    print("Kész!")

if __name__ == "__main__":
    main()