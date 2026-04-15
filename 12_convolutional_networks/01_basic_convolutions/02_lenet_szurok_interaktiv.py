"""
02_lenet_szurok_interaktiv.py — LeNet-5 interaktív feature map vizualizáció
Neurális hálók II. (CNN) — Hajdu Csaba

Interaktív vizualizáció: a felhasználó végiglapozhatja a teszthalmaz
számjegyeit egy csúszka segítségével, és megvizsgálhatja, hogyan
változnak a modell rejtett rétegeinek (Conv1 és Conv2) aktivitásai
(feature map-jei) a különböző bemenetek hatására.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 2 * 2)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def get_feature_maps(self, x):
        """Köztes feature mapok visszaadása vizualizációhoz."""
        fm1 = F.relu(self.conv1(x))
        fm1_pool = self.pool(fm1)
        fm2 = F.relu(self.conv2(fm1_pool))
        return fm1, fm2


def train_lenet(model, epochs=150):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        pred = model(Xt)
        loss = criterion(pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    print("LeNet-5 modell tanítása folyamatban...")
    torch.manual_seed(42)
    model = LeNet5()
    train_lenet(model, epochs=150)
    model.eval()
    print("Tanítás befejeződött. Ablak megnyitása...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    plt.subplots_adjust(bottom=0.25)

    ax_in = axes[0]
    ax_fm1 = axes[1]
    ax_fm2 = axes[2]

    # Csúszka a teszthalmaz indexéhez
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.04])
    s_idx = Slider(ax_slider, 'Minta index', 0, len(Xv) - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(s_idx.val)
        sample = Xv[idx:idx+1]
        label = yv[idx].item()

        with torch.no_grad():
            pred_out = model(sample)
            pred_label = pred_out.argmax(dim=1).item()
            fm1, fm2 = model.get_feature_maps(sample)

        # Bemenet
        ax_in.clear()
        ax_in.imshow(sample.squeeze().numpy(), cmap="gray_r")
        color = "green" if pred_label == label else "red"
        ax_in.set_title(f"Bemenet\nValós: {label} | Jósolt: {pred_label}", color=color, fontweight="bold")
        ax_in.axis("off")

        # Conv1 Feature maps (6 csatorna, 8×8)
        ax_fm1.clear()
        fm1_grid = np.zeros((2 * 8 + 2, 3 * 8 + 4))
        for i in range(6):
            r, c = divmod(i, 3)
            fm1_grid[r*(8+2):r*(8+2)+8, c*(8+2):c*(8+2)+8] = fm1[0, i].numpy()
        ax_fm1.imshow(fm1_grid, cmap="viridis")
        ax_fm1.set_title("Conv1 Feature Map (6 ch)", fontweight="bold")
        ax_fm1.axis("off")

        # Conv2 Feature maps (16 csatorna, 4×4)
        ax_fm2.clear()
        fm2_np = fm2[0].numpy()
        fm2_grid = np.zeros((4 * (4+1), 4 * (4+1)))
        for i in range(16):
            r, c = divmod(i, 4)
            fm2_grid[r*5:r*5+4, c*5:c*5+4] = fm2_np[i]
        ax_fm2.imshow(fm2_grid, cmap="viridis")
        ax_fm2.set_title("Conv2 Feature Map (16 ch)", fontweight="bold")
        ax_fm2.axis("off")

        fig.canvas.draw_idle()

    s_idx.on_changed(update)
    update(0)  # Kezdeti állapot

    plt.show()


if __name__ == "__main__":
    main()
