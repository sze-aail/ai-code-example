"""
02_lenet_szurok.py — LeNet-5 tanítás és szűrő-vizualizáció
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - LeNet-5 architektúra (LeCun 1998) PyTorch-ban
  - Tanítás sklearn digits (8×8) adathalmazon
  - Tanult konvolúciós szűrők vizualizáció (mit tanul a háló?)
  - Feature map-ek vizualizáció rétegenként
  - CNN vs. MLP összehasonlítás azonos feladaton
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Adat: 8×8 számjegyek ──
digits = load_digits()
X = digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

Xt = torch.tensor(X_train); yt = torch.tensor(y_train, dtype=torch.long)
Xv = torch.tensor(X_test); yv = torch.tensor(y_test, dtype=torch.long)


# ══════════════════════════════════════════════════════════════
# LeNet-5 (adaptálva 8×8-ra)
# ══════════════════════════════════════════════════════════════

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)   # 8×8 → 8×8
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, padding=1)  # 4×4 → 4×4
        self.pool = nn.MaxPool2d(2, 2)                            # 8→4→2
        self.fc1 = nn.Linear(16 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B,1,8,8) → (B,6,4,4)
        x = self.pool(F.relu(self.conv2(x)))  # (B,6,4,4) → (B,16,2,2)
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_feature_maps(self, x):
        """Köztes feature mapok visszaadása vizualizációhoz."""
        fm1 = F.relu(self.conv1(x))
        fm1_pool = self.pool(fm1)
        fm2 = F.relu(self.conv2(fm1_pool))
        return fm1, fm2


class SimpleMLP(nn.Module):
    """Összehasonlításhoz: MLP azonos feladaton."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# Tanítás
# ══════════════════════════════════════════════════════════════

def train_model(model, name, epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accs = [], []

    for epoch in range(epochs):
        model.train()
        pred = model(Xt)
        loss = criterion(pred, yt)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv).argmax(dim=1)
            acc = (val_pred == yv).float().mean().item()
        val_accs.append(acc)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {name}: acc={val_accs[-1]*100:.1f}%, params={n_params}")
    return train_losses, val_accs


def main():
    print("=" * 60)
    print("TANÍTÁS: LeNet-5 vs. MLP")
    print("=" * 60)

    torch.manual_seed(42)
    lenet = LeNet5()
    tl_cnn, va_cnn = train_model(lenet, "LeNet-5")

    torch.manual_seed(42)
    mlp = SimpleMLP()
    tl_mlp, va_mlp = train_model(mlp, "MLP")


    # ══════════════════════════════════════════════════════════════
    # ÁBRÁK
    # ══════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. Tanulási görbék
    ax = axes[0, 0]
    ax.plot(tl_cnn, label="LeNet-5", color="#E91E63", linewidth=1.5)
    ax.plot(tl_mlp, label="MLP", color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Tanítási veszteség", fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)

    # 2. Validációs pontosság
    ax = axes[0, 1]
    ax.plot(va_cnn, label="LeNet-5", color="#E91E63", linewidth=1.5)
    ax.plot(va_mlp, label="MLP", color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Validációs pontosság", fontweight="bold"); ax.legend(); ax.grid(True, alpha=0.3)

    # 3. Tanult szűrők (conv1: 6 db 3×3 szűrő)
    ax = axes[0, 2]
    filters = lenet.conv1.weight.detach().numpy()
    for i in range(6):
        r, c = divmod(i, 3)
        sub = fig.add_axes([0.68 + c*0.1, 0.72 - r*0.15, 0.08, 0.12])
        sub.imshow(filters[i, 0], cmap="RdBu_r", vmin=-0.5, vmax=0.5)
        sub.set_title(f"F{i+1}", fontsize=8); sub.axis("off")
    ax.axis("off")
    ax.set_title("Conv1 tanult szűrők (6 db, 3×3)", fontweight="bold")

    # 4-5. Feature map-ek egy '5'-ös számjegyre
    idx_5 = np.where(y_test == 5)[0][0]
    sample = Xv[idx_5:idx_5+1]
    lenet.eval()
    with torch.no_grad():
        fm1, fm2 = lenet.get_feature_maps(sample)

    ax = axes[1, 0]
    ax.imshow(sample.squeeze().numpy(), cmap="gray_r")
    ax.set_title(f"Bemenet (címke: {y_test[idx_5]})", fontweight="bold"); ax.axis("off")

    ax = axes[1, 1]
    fm1_grid = np.zeros((2 * 8 + 2, 3 * 8 + 4))
    for i in range(6):
        r, c = divmod(i, 3)
        fm1_grid[r*(8+2):r*(8+2)+8, c*(8+2):c*(8+2)+8] = fm1[0, i].numpy()
    ax.imshow(fm1_grid, cmap="viridis"); ax.axis("off")
    ax.set_title("Conv1 feature map-ek (6 csatorna)", fontweight="bold")

    ax = axes[1, 2]
    fm2_np = fm2[0].numpy()
    fm2_grid = np.zeros((4 * (4+1), 4 * (4+1)))
    for i in range(16):
        r, c = divmod(i, 4)
        fm2_grid[r*5:r*5+4, c*5:c*5+4] = fm2_np[i]
    ax.imshow(fm2_grid, cmap="viridis"); ax.axis("off")
    ax.set_title("Conv2 feature map-ek (16 csatorna)", fontweight="bold")

    fig.suptitle("LeNet-5: tanítás, szűrők és feature map-ek", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig("02_lenet_szurok.png", dpi=150)
    print("\nÁbra mentve: 02_lenet_szurok.png")

    plt.close("all")
    print("Kész!")


if __name__ == "__main__":
    main()
