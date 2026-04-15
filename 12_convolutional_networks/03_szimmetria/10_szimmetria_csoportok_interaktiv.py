"""
10_szimmetria_csoportok_interaktiv.py — Forgási invariancia interaktív vizualizációja
Neurális hálók II. (CNN) — Hajdu Csaba

Interaktív vizualizáció, ahol a felhasználó egy csúszka segítségével
forgathat egy számjegyet (0-360 fok), és valós időben megfigyelheti
a Standard CNN és a Group Equivariant CNN (G-CNN) jóslatainak változását.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# ══════════════════════════════════════════════════════════════
# Modellek
# ══════════════════════════════════════════════════════════════

class StandardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).flatten(1))

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_rotations=4):
        super().__init__()
        self.n_rotations = n_rotations
        self.base_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
        self.padding = kernel_size // 2

    def forward(self, x):
        outputs = []
        for k in range(self.n_rotations):
            rotated_weight = torch.rot90(self.base_weight, k, dims=(-2, -1))
            outputs.append(F.conv2d(x, rotated_weight, padding=self.padding))
        stacked = torch.stack(outputs, dim=-1)
        pooled, _ = stacked.max(dim=-1)
        return pooled

class GroupEquivariantCNN(nn.Module):
    def __init__(self, n_rotations=4):
        super().__init__()
        self.gconv1 = GroupConv2d(1, 8, 3, n_rotations=n_rotations)
        self.gconv2 = GroupConv2d(8, 16, 3, n_rotations=n_rotations)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.gconv1(x))
        x = F.relu(self.gconv2(x))
        return self.fc(self.pool(x).flatten(1))

# ══════════════════════════════════════════════════════════════
# Segédfüggvények
# ══════════════════════════════════════════════════════════════

def rotate_image_arbitrary(img, angle_deg):
    h, w = img.shape
    cy, cx = h / 2, w / 2
    rad = np.radians(angle_deg)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    result = np.zeros_like(img)
    for y in range(h):
        for x in range(w):
            src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx
            src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy
            ix, iy = int(src_x), int(src_y)
            if 0 <= ix < w and 0 <= iy < h:
                result[y, x] = img[iy, ix]
    return result

def train_model(model, Xt, yt, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(Xt), yt)
        loss.backward()
        optimizer.step()
    model.eval()

# ══════════════════════════════════════════════════════════════
# Fő iteratív program
# ══════════════════════════════════════════════════════════════

def main():
    print("Adatok betöltése és modellek gyors tanítása...")
    digits = load_digits()
    X_all = digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0
    y_all = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    Xt = torch.tensor(X_train); yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_test); yv = torch.tensor(y_test, dtype=torch.long)

    torch.manual_seed(42)
    std_cnn = StandardCNN()
    train_model(std_cnn, Xt, yt, epochs=80)

    torch.manual_seed(42)
    g_cnn = GroupEquivariantCNN(4)
    train_model(g_cnn, Xt, yt, epochs=80)
    print("Modellek kész! Ablak megnyitása...")

    # Interfész
    fig = plt.figure(figsize=(14, 6))

    ax_img = plt.axes([0.05, 0.25, 0.25, 0.6])
    ax_std = plt.axes([0.35, 0.25, 0.25, 0.6])
    ax_gcnn = plt.axes([0.65, 0.25, 0.25, 0.6])

    ax_slider_a = plt.axes([0.2, 0.1, 0.6, 0.03])
    ax_slider_i = plt.axes([0.2, 0.05, 0.6, 0.03])

    s_angle = Slider(ax_slider_a, 'Forgatás (°)', 0, 360, valinit=0, valstep=5)
    s_idx = Slider(ax_slider_i, 'Teszteset', 0, len(Xv)-1, valinit=0, valstep=1)

    def update(val=None):
        idx = int(s_idx.val)
        angle = s_angle.val

        orig_img = Xv[idx, 0].numpy()
        true_label = yv[idx].item()

        # Kép forgatása
        rot_img = rotate_image_arbitrary(orig_img, angle)
        rot_tensor = torch.tensor(rot_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # Jóslatok
        with torch.no_grad():
            std_logits = std_cnn(rot_tensor)
            std_probs = F.softmax(std_logits, dim=1).numpy().flatten()
            std_pred = std_probs.argmax()

            gcnn_logits = g_cnn(rot_tensor)
            gcnn_probs = F.softmax(gcnn_logits, dim=1).numpy().flatten()
            gcnn_pred = gcnn_probs.argmax()

        # Rajzolás -> Kép
        ax_img.clear()
        ax_img.imshow(rot_img, cmap="gray_r", vmin=0, vmax=1)
        ax_img.set_title(f"Bemenet (Forgatás: {angle}°)\nValódi: {true_label}", fontweight="bold")
        ax_img.axis("off")

        # Rajzolás -> Std CNN
        ax_std.clear()
        colors_std = ["#2196F3"] * 10
        colors_std[std_pred] = "#4CAF50" if std_pred == true_label else "#E91E63"
        ax_std.bar(range(10), std_probs, color=colors_std, alpha=0.8)
        ax_std.set_ylim(0, 1)
        ax_std.set_xticks(range(10))
        ax_std.set_title(f"Standard CNN\nJóslat: {std_pred} ({std_probs.max()*100:.0f}%)", fontweight="bold")

        # Rajzolás -> G-CNN
        ax_gcnn.clear()
        colors_gcnn = ["#2196F3"] * 10
        colors_gcnn[gcnn_pred] = "#4CAF50" if gcnn_pred == true_label else "#E91E63"
        ax_gcnn.bar(range(10), gcnn_probs, color=colors_gcnn, alpha=0.8)
        ax_gcnn.set_ylim(0, 1)
        ax_gcnn.set_xticks(range(10))
        ax_gcnn.set_title(f"G-CNN (C4 szimmetria)\nJóslat: {gcnn_pred} ({gcnn_probs.max()*100:.0f}%)", fontweight="bold")

        fig.canvas.draw_idle()

    s_angle.on_changed(update)
    s_idx.on_changed(update)

    update()
    plt.show()

if __name__ == "__main__":
    main()
