"""
06_overfitting_regularization.py — Túltanulás és regularizáció
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Overfitting vs. underfitting vizuálisan
  - L2 regularizáció, Dropout, Early stopping hatása
  - Train/val loss görbék összehasonlítása
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def main():
    # ── Szintetikus adat: zajos szinusz ──
    np.random.seed(42)
    torch.manual_seed(42)
    n = 60
    X_all = np.sort(np.random.uniform(-3, 3, n)).astype(np.float32)
    y_all = (np.sin(X_all) + np.random.randn(n).astype(np.float32) * 0.3)

    # Train/val split
    split = 40
    X_train, y_train = X_all[:split], y_all[:split]
    X_val, y_val = X_all[split:], y_all[split:]

    Xt = torch.tensor(X_train).unsqueeze(1)
    yt = torch.tensor(y_train).unsqueeze(1)
    Xv = torch.tensor(X_val).unsqueeze(1)
    yv = torch.tensor(y_val).unsqueeze(1)


    def make_mlp(hidden=64, dropout=0.0):
        layers = [nn.Linear(1, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, 1))
        return nn.Sequential(*layers)


    def train_model(model, epochs=800, lr=0.01, weight_decay=0.0, early_stop=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        train_losses, val_losses = [], []
        best_val, patience, counter = float("inf"), 50, 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            pred = model(Xt)
            loss = criterion(pred, yt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(Xv), yv).item()
            val_losses.append(val_loss)

            if early_stop:
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"  Early stopping: epoch {epoch + 1}")
                        model.load_state_dict(best_state)
                        break

        return train_losses, val_losses


    # ── 4 kísérlet ──
    configs = [
        ("Nincs regularizáció\n(overfitting)", {"hidden": 128, "dropout": 0.0}, {"weight_decay": 0.0}),
        ("L2 regularizáció\n(weight_decay=0.01)", {"hidden": 128, "dropout": 0.0}, {"weight_decay": 0.01}),
        ("Dropout (p=0.3)", {"hidden": 128, "dropout": 0.3}, {"weight_decay": 0.0}),
        ("Early Stopping\n(patience=50)", {"hidden": 128, "dropout": 0.0}, {"weight_decay": 0.0, "early_stop": True}),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    x_dense = torch.linspace(-3.5, 3.5, 300).unsqueeze(1)
    y_true_curve = np.sin(x_dense.numpy().flatten())

    for col, (title, model_kw, train_kw) in enumerate(configs):
        print(f"Tanítás: {title.replace(chr(10), ' ')}")
        model = make_mlp(**model_kw)
        tl, vl = train_model(model, **train_kw)

        # Loss görbék
        ax_loss = axes[0, col]
        ax_loss.plot(tl, label="Train", color="#2196F3", linewidth=1.5)
        ax_loss.plot(vl, label="Val", color="#E91E63", linewidth=1.5)
        ax_loss.set_title(title, fontsize=11, fontweight="bold")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("MSE")
        ax_loss.legend(fontsize=9)
        ax_loss.set_ylim(0, max(0.5, min(max(vl), 3)))
        ax_loss.grid(True, alpha=0.3)

        # Illesztés
        ax_fit = axes[1, col]
        model.eval()
        with torch.no_grad():
            y_pred = model(x_dense).numpy().flatten()

        ax_fit.plot(x_dense.numpy(), y_true_curve, "g--", linewidth=1.5, label="sin(x)", alpha=0.7)
        ax_fit.plot(x_dense.numpy(), y_pred, "r-", linewidth=2, label="MLP")
        ax_fit.scatter(X_train, y_train, c="#2196F3", s=30, label="Train", zorder=3, edgecolors="k", linewidth=0.5)
        ax_fit.scatter(X_val, y_val, c="#E91E63", s=30, marker="^", label="Val", zorder=3, edgecolors="k", linewidth=0.5)
        ax_fit.set_ylim(-2, 2)
        ax_fit.set_xlabel("x")
        ax_fit.set_ylabel("y")
        ax_fit.legend(fontsize=8)
        ax_fit.grid(True, alpha=0.3)

    fig.suptitle("Regularizáció hatása — túltanulás kezelése", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig("06_regularization.png", dpi=150)
    print("\nÁbra mentve: 06_regularization.png")

    plt.close("all")
    print("Kész!")

if __name__ == "__main__":
    main()