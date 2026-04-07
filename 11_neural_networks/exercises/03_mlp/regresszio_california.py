"""
regresszio_california.py — MLP regresszió valós adaton (California Housing)
Forrás: edu-machine-learning (02_MLP_regression) + mesterseges-intelligencia-kodok (03_mlp_regression), egyesítve

Demonstrálja:
  - Valós adathalmaz előfeldolgozása (StandardScaler)
  - MLP regresszió PyTorch-ban
  - Train/test loss görbék, predikció vs. valódi értékek
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Adat ──
try:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = data.data, data.target
    print(f"Jellemzők: {data.feature_names}")
except Exception:
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=5000, n_features=8, n_informative=6, noise=15, random_state=42)
    y = (y - y.min()) / (y.max() - y.min()) * 5
    print("(California Housing nem elérhető, szintetikus adat)")
print(f"Minták: {X.shape[0]}, Jellemzők: {X.shape[1]}, Cél: [{y.min():.2f}, {y.max():.2f}]")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
Xv = torch.tensor(X_test, dtype=torch.float32)
yv = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


# ── Modell ──
class HousingMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(42)
model = HousingMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# ── Tanítás (mini-batch) ──
batch_size = 256
n_epochs = 100
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    model.train()
    perm = torch.randperm(Xt.size(0))
    epoch_loss = 0
    n_batches = 0
    for i in range(0, Xt.size(0), batch_size):
        idx = perm[i:i + batch_size]
        pred = model(Xt[idx])
        loss = criterion(pred, yt[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    train_losses.append(epoch_loss / n_batches)

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(Xv), yv).item()
    val_losses.append(val_loss)

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Train MSE: {train_losses[-1]:.4f} | Val MSE: {val_loss:.4f}")

# ── Ábrák ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss görbék
axes[0].plot(train_losses, label="Train", color="#2196F3", linewidth=1.5)
axes[0].plot(val_losses, label="Validation", color="#E91E63", linewidth=1.5)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE")
axes[0].set_title("Tanulási görbék", fontweight="bold")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Predikció vs. valódi
model.eval()
with torch.no_grad():
    y_pred = model(Xv).numpy().flatten()

axes[1].scatter(y_test, y_pred, s=8, alpha=0.3, color="#4CAF50")
lims = [0, max(y_test.max(), y_pred.max()) + 0.5]
axes[1].plot(lims, lims, "r--", linewidth=1.5, label="Tökéletes predikció")
axes[1].set_xlabel("Valódi ár ($100k)")
axes[1].set_ylabel("Predikált ár ($100k)")
axes[1].set_title(f"Predikció vs. valódi (R² = {1 - np.sum((y_test-y_pred)**2)/np.sum((y_test-y_test.mean())**2):.3f})",
                   fontweight="bold")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("California Housing — MLP regresszió", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig("california_regresszio.png", dpi=150)
print("\nÁbra mentve: california_regresszio.png")

plt.close("all")
print("Kész!")
