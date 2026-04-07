import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # Fix seed
    torch.manual_seed(42)
    np.random.seed(42)

    # 1D Regression dataset
    X_train = torch.linspace(-1, 1, 50).unsqueeze(1)
    y_train = torch.sin(3 * X_train) + 0.1 * torch.randn(X_train.size())

    # Create two identical models
    model_normal = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    model_large_lr = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

    # Initialize with the same weights
    model_large_lr.load_state_dict(model_normal.state_dict())

    # Optimizers
    opt_normal = torch.optim.SGD(model_normal.parameters(), lr=0.1)
    opt_large = torch.optim.SGD(model_large_lr.parameters(), lr=1.5)  # Deliberately large learning rate

    criterion = nn.MSELoss()

    epochs = 150
    frames_data = []

    print("Training regression models...")
    for epoch in range(epochs + 1):
        # Train normal model
        opt_normal.zero_grad()
        out_normal = model_normal(X_train)
        loss_normal = criterion(out_normal, y_train)
        loss_normal.backward()
        opt_normal.step()

        # Train model with exploding learning rate
        opt_large.zero_grad()
        out_large = model_large_lr(X_train)
        loss_large = criterion(out_large, y_train)
        loss_large.backward()
        opt_large.step()

        # Save frames
        if epoch % 3 == 0:
            with torch.no_grad():
                frames_data.append({
                    'epoch': epoch,
                    'loss_normal': loss_normal.item(),
                    'pred_normal': model_normal(X_train).numpy(),
                    'loss_large': loss_large.item(),
                    'pred_large': model_large_lr(X_train).numpy()
                })

    # Pad at the end
    for _ in range(10):
        frames_data.append(frames_data[-1])

    # Plot setup
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    def update(frame_idx):
        data = frames_data[frame_idx]
        ax1.clear()
        ax2.clear()

        X_np = X_train.numpy()
        y_np = y_train.numpy()

        configs = [
            (ax1, "Normál LR (0.1)", data['pred_normal'], data['loss_normal']),
            (ax2, "Szándékosan nagy LR (1.5) - Divergencia", data['pred_large'], data['loss_large'])
        ]

        for ax, title, pred, loss in configs:
            ax.scatter(X_np, y_np, color='blue', alpha=0.5, label='Adat')
            ax.plot(X_np, pred, color='red', linewidth=2, label='Modell predikció')
            ax.set_ylim(-3, 3)
            ax.set_xlim(-1.1, 1.1)
            ax.set_title(f"{title}\nLoss (MSE): {loss:.4f}")
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        fig.suptitle(f"Regressziós tanulás - Epocha: {data['epoch']}", fontsize=16)

    print("Generating regression animation...")
    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "10_regression_wrong_learning_animation.gif"
    anim.save(output_file, writer='pillow', fps=10)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
