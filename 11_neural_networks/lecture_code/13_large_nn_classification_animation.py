import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import TensorDataset, DataLoader

def generate_complex_classification_data(n_samples=20000):
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate interleaved spirals
    n = n_samples // 2
    theta = np.sqrt(np.random.rand(n)) * 4 * np.pi

    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n, 2) * 1.5

    r_b = -2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n, 2) * 1.5

    X = np.vstack([x_a, x_b])
    y = np.hstack([np.zeros(n), np.ones(n)])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor

class LargeNetCls(nn.Module):
    def __init__(self):
        super(LargeNetCls, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

def main():
    X, y = generate_complex_classification_data(20000)

    indices = torch.randperm(20000)
    train_idx, val_idx = indices[:16000], indices[16000:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)

    model = LargeNetCls()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    epochs = 150
    frames_data = []

    x_min, x_max = -35, 35
    y_min, y_max = -35, 35
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                         np.linspace(y_min, y_max, 150))
    X_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    train_losses = []
    val_losses = []

    print("Hálózat tanítása osztályozásra (20000 minta)...")
    for epoch in range(epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss /= len(X_train)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
            val_losses.append(val_loss)

            if epoch % 3 == 0:
                grid_pred = model(X_grid).numpy().reshape(xx.shape)
                w1 = model.net[2].weight.data.clone().numpy()
                w2 = model.net[4].weight.data.clone().numpy()
                g1 = model.net[2].weight.grad.clone().numpy() if model.net[2].weight.grad is not None else np.zeros_like(w1)
                g2 = model.net[4].weight.grad.clone().numpy() if model.net[4].weight.grad is not None else np.zeros_like(w2)

                frames_data.append({
                    'epoch': epoch,
                    'train_loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'grid_pred': grid_pred,
                    'train_losses': list(train_losses),
                    'val_losses': list(val_losses),
                    'w1': w1, 'w2': w2,
                    'g1': g1, 'g2': g2
                })

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    for _ in range(10):
        frames_data.append(frames_data[-1])

    # 1. Animáció: Döntési határ (Decision Boundary)
    print("Első animáció (Decision boundary) készítése...")
    fig1, ax1 = plt.subplots(figsize=(8, 8))

    # Ritkított pontok ábrázolása
    X_val_np = X_val.numpy()[:2000]
    y_val_np = y_val.numpy()[:2000].flatten()

    def update_decision(frame_idx):
        data = frames_data[frame_idx]
        ax1.clear()

        ax1.contourf(xx, yy, data['grid_pred'], levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=0.6)

        ax1.scatter(X_val_np[y_val_np==0, 0], X_val_np[y_val_np==0, 1], color='red', alpha=0.6, s=15, edgecolor='k', label='Osztály 0')
        ax1.scatter(X_val_np[y_val_np==1, 0], X_val_np[y_val_np==1, 1], color='blue', alpha=0.6, s=15, edgecolor='k', label='Osztály 1')

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title(f"A modell döntési határának evolúciója (Epoch: {data['epoch']})", fontsize=14)
        ax1.legend(loc='upper right')
        return ax1,

    anim1 = FuncAnimation(fig1, update_decision, frames=len(frames_data), blit=False)
    anim1.save("13_large_nn_cls_decision_boundary.gif", writer='pillow', fps=10)
    plt.close(fig1)

    # 2. Animáció: Hiba görbék
    print("Második animáció (Error curves) készítése...")
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    def update_errors(frame_idx):
        data = frames_data[frame_idx]
        ax2.clear()

        curr_epochs = range(len(data['train_losses']))
        ax2.plot(curr_epochs, data['train_losses'], 'b-', linewidth=2, label='Training Loss')
        ax2.plot(curr_epochs, data['val_losses'], 'orange', linewidth=2, label='Validation Loss')

        ax2.set_xlim(0, epochs)
        ax2.set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)
        ax2.set_title(f"Tanítási és Validációs Hiba (Epoch: {data['epoch']})", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Binary Cross Entropy Loss")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        return ax2,

    anim2 = FuncAnimation(fig2, update_errors, frames=len(frames_data), blit=False)
    anim2.save("13_large_nn_cls_error_curves.gif", writer='pillow', fps=10)
    plt.close(fig2)

    # 3. Animáció: Súlyok és Gradiensek (Weights & Grads)
    print("Harmadik animáció (Weights & Grads) készítése...")
    fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
    (ax_w1, ax_w2), (ax_g1, ax_g2) = axes3

    cmap = 'RdBu_r'

    def update_weights(frame_idx):
        data = frames_data[frame_idx]
        for ax in axes3.flat:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        vmax_w = 0.5
        ax_w1.imshow(data['w1'], cmap=cmap, vmin=-vmax_w, vmax=vmax_w)
        ax_w1.set_title("W2 Súlyok (128x128)")

        ax_w2.imshow(data['w2'], cmap=cmap, vmin=-vmax_w, vmax=vmax_w)
        ax_w2.set_title("W3 Súlyok (64x128)")

        g1 = data['g1']
        g2 = data['g2']
        vmax_g1 = max(np.max(np.abs(g1)), 1e-5)
        vmax_g2 = max(np.max(np.abs(g2)), 1e-5)

        ax_g1.imshow(g1, cmap=cmap, vmin=-vmax_g1, vmax=vmax_g1)
        ax_g1.set_title("W2 Gradiensek (Normalizálva)")

        ax_g2.imshow(g2, cmap=cmap, vmin=-vmax_g2, vmax=vmax_g2)
        ax_g2.set_title("W3 Gradiensek (Normalizálva)")

        fig3.suptitle(f"Osztályozó: Súlyok és Gradiensek (Epoch: {data['epoch']})", fontsize=16)
        return axes3.flat

    anim3 = FuncAnimation(fig3, update_weights, frames=len(frames_data), blit=False)
    anim3.save("13_large_nn_cls_weights_animation.gif", writer='pillow', fps=10)
    plt.close(fig3)

    print("Minden animáció sikeresen lementve!")

if __name__ == "__main__":
    main()
