import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from torch.utils.data import TensorDataset, DataLoader

def generate_complex_data(n_samples=20000):
    # Generálunk 20000 pontot egy komplex nem-lineáris függvény körül, zajjal
    np.random.seed(42)
    torch.manual_seed(42)

    # X értékek egyenletes eloszlásban -5 és 5 között
    X = np.random.uniform(-5, 5, n_samples)

    # Sokkal komplexebb struktúra: magas frekvenciás szinuszok, koszinuszok, polinom és exponenciális
    y = 3 * np.sin(5 * X) + 2 * np.cos(13 * X) + X * np.sin(2 * X) + 0.2 * X**2 - 0.05 * X**3 + 5 * np.exp(-0.5 * X**2)

    # Különböző szórású zaj (heteroszkedasztikus zaj) a valósághűbb adatért
    noise_std = np.abs(np.sin(X)) * 1.5 + 0.5
    noise = np.random.normal(0, noise_std, n_samples)
    y = y + noise

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X_tensor, y_tensor

class LargeNet(nn.Module):
    def __init__(self):
        super(LargeNet, self).__init__()
        # Nagyobb hálózat (több réteg, több paraméter)
        self.net = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

def main():
    X, y = generate_complex_data(20000)

    # Keverés és felosztás
    indices = torch.randperm(20000)
    train_idx, val_idx = indices[:16000], indices[16000:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)

    model = LargeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()

    epochs = 150
    frames_data = []

    # Grid a predikció ábrázolásához
    X_grid = torch.linspace(-5.5, 5.5, 500).unsqueeze(1)

    # Előre kiszámítjuk a "True" függvényt a gridre
    y_true_grid = 3 * torch.sin(5 * X_grid) + 2 * torch.cos(13 * X_grid) + X_grid * torch.sin(2 * X_grid) + 0.2 * X_grid**2 - 0.05 * X_grid**3 + 5 * torch.exp(-0.5 * X_grid**2)

    train_losses = []
    val_losses = []

    print("Hálózat tanítása (20000 minta)...")
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
                grid_pred = model(X_grid).numpy()
                w1 = model.net[2].weight.data.clone().numpy()
                w2 = model.net[4].weight.data.clone().numpy()
                g1 = model.net[2].weight.grad.clone().numpy() if model.net[2].weight.grad is not None else np.zeros_like(w1)
                g2 = model.net[4].weight.grad.clone().numpy() if model.net[4].weight.grad is not None else np.zeros_like(w2)

                frames_data.append({
                    'epoch': epoch,
                    'train_loss': epoch_train_loss,
                    'val_loss': val_loss,
                    'grid_pred': grid_pred,
                    'train_losses': list(train_losses),  # másolat
                    'val_losses': list(val_losses),
                    'w1': w1, 'w2': w2,
                    'g1': g1, 'g2': g2
                })

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")

    for _ in range(10):
        frames_data.append(frames_data[-1])

    # ----------------------------------------------------
    # Animáció 1: Becslés pontossága (Estimation Precision)
    # ----------------------------------------------------
    print("Első animáció (Estimation precision) készítése...")
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Ritkított pontokat ábrázolunk a gyorsabb plotolás/renderelés miatt
    X_val_np = X_val.numpy()[:2000]
    y_val_np = y_val.numpy()[:2000]
    X_grid_np = X_grid.numpy()
    y_true_np = y_true_grid.numpy()

    def update_precision(frame_idx):
        data = frames_data[frame_idx]
        ax1.clear()

        # Scatter pontok
        ax1.scatter(X_val_np, y_val_np, color='gray', alpha=0.3, s=10, label='Minta adatok (Val)')

        # Valódi függvény
        ax1.plot(X_grid_np, y_true_np, 'g--', linewidth=2, label='Valódi elméleti függvény')

        # Modell predikció
        ax1.plot(X_grid_np, data['grid_pred'], 'r-', linewidth=3, label='Modell predikciója')

        ax1.set_ylim(-15, 15)
        ax1.set_xlim(-5.2, 5.2)
        ax1.set_title(f"A modell predikciójának evolúciója (Epoch: {data['epoch']})", fontsize=14)
        ax1.legend(loc='lower left')
        ax1.grid(True, alpha=0.3)
        return ax1,

    anim1 = FuncAnimation(fig1, update_precision, frames=len(frames_data), blit=False)
    anim1.save("12_large_nn_estimation_precision.gif", writer='pillow', fps=10)
    plt.close(fig1)

    # ----------------------------------------------------
    # Animáció 2: Train & Val hiba görbék dinamikája
    # ----------------------------------------------------
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
        ax2.set_ylabel("Mean Squared Error (MSE)")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        return ax2,

    anim2 = FuncAnimation(fig2, update_errors, frames=len(frames_data), blit=False)
    anim2.save("12_large_nn_error_curves.gif", writer='pillow', fps=10)
    plt.close(fig2)

    # ----------------------------------------------------
    # Animáció 3: Súlyok és Gradiensek (Weights & Grads)
    # ----------------------------------------------------
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
        ax_w1.set_title("W2 Súlyok (256x256)")

        ax_w2.imshow(data['w2'], cmap=cmap, vmin=-vmax_w, vmax=vmax_w)
        ax_w2.set_title("W3 Súlyok (128x256)")

        g1 = data['g1']
        g2 = data['g2']
        vmax_g1 = max(np.max(np.abs(g1)), 1e-5)
        vmax_g2 = max(np.max(np.abs(g2)), 1e-5)

        ax_g1.imshow(g1, cmap=cmap, vmin=-vmax_g1, vmax=vmax_g1)
        ax_g1.set_title("W2 Gradiensek (Normalizálva)")

        ax_g2.imshow(g2, cmap=cmap, vmin=-vmax_g2, vmax=vmax_g2)
        ax_g2.set_title("W3 Gradiensek (Normalizálva)")

        fig3.suptitle(f"Súlyok és Gradiensek alakulása (Epoch: {data['epoch']})", fontsize=16)
        return axes3.flat

    anim3 = FuncAnimation(fig3, update_weights, frames=len(frames_data), blit=False)
    anim3.save("12_large_nn_weights_animation.gif", writer='pillow', fps=10)
    plt.close(fig3)

    print("Minden animáció sikeresen lementve!")

if __name__ == "__main__":
    main()
