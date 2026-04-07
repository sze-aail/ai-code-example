import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1D Regression adat nem-lineáris trenddel (szinusz) és megnövelt zajjal
    X_train = torch.linspace(-3, 3, 100).unsqueeze(1)
    y_train = torch.sin(2 * X_train) + 0.3 * torch.randn(X_train.size())

    # Véletlenszerű, sokkal extrémebb kiugró értékek (outliers) hozzáadása
    outlier_indices = torch.randperm(100)[:15] # 15 outlier
    y_train[outlier_indices] += (torch.rand(15, 1) * 20 - 10) * 1.5

    # Neurális hálózat modell definiálása
    def create_model():
        return nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # Modellek (MLP, hogy képes legyen a nem-lineáris görbületet felvenni)
    model_mse = create_model()
    model_huber = create_model()

    # Kezdősúlyok pontosan megegyeznek
    model_huber.load_state_dict(model_mse.state_dict())

    opt_mse = torch.optim.Adam(model_mse.parameters(), lr=0.01)
    opt_huber = torch.optim.Adam(model_huber.parameters(), lr=0.01)

    # Hibafüggvények
    criterion_mse = nn.MSELoss()
    criterion_huber = nn.HuberLoss(delta=1.0)

    epochs = 300
    frames_data = []

    # Külön grid a szép vizualizációhoz
    X_grid = torch.linspace(-3.5, 3.5, 200).unsqueeze(1)

    print("Modellek tanítása (MSE vs Huber Neurális Hálóval)...")
    for epoch in range(epochs + 1):
        # MSE Train
        opt_mse.zero_grad()
        out_mse = model_mse(X_train)
        loss_mse = criterion_mse(out_mse, y_train)
        loss_mse.backward()
        opt_mse.step()

        # Huber Train
        opt_huber.zero_grad()
        out_huber = model_huber(X_train)
        loss_huber = criterion_huber(out_huber, y_train)
        loss_huber.backward()
        opt_huber.step()

        if epoch % 3 == 0:
            with torch.no_grad():
                frames_data.append({
                    'epoch': epoch,
                    'pred_mse': model_mse(X_grid).numpy(),
                    'pred_huber': model_huber(X_grid).numpy()
                })

    # Kitartjuk az utolsó frame-et
    for _ in range(15):
        frames_data.append(frames_data[-1])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def update(frame_idx):
        data = frames_data[frame_idx]
        ax1.clear()
        ax2.clear()

        X_np = X_train.numpy()
        y_np = y_train.numpy()
        X_g = X_grid.numpy()

        configs = [
            (ax1, "MSE Loss (L2)\n(Érzékeny a kiugró pontokra)", data['pred_mse'], 'red'),
            (ax2, "Huber Loss (Smooth L1)\n(Robusztus a kiugró pontokkal szemben)", data['pred_huber'], 'green')
        ]

        for ax, title, pred, color in configs:
            ax.scatter(X_np, y_np, color='blue', alpha=0.5, label='Adat (zajos + outlierek)')
            ax.plot(X_g, pred, color=color, linewidth=3, label='Illesztett hálózat')
            ax.set_ylim(-15, 15)
            ax.set_xlim(-3.5, 3.5)
            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')

        fig.suptitle(f"MSE vs Huber Loss Evolúció - Epocha: {data['epoch']}", fontsize=16)
        return ax1, ax2

    print("Animáció készítése...")
    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "10_huber_vs_mse_animation.gif"
    anim.save(output_file, writer='pillow', fps=10)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
