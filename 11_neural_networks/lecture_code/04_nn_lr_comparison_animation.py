import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_model():
    # Make sure all models start with the exact same initialization for a fair comparison
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # XOR Dataset
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # We will compare these learning rates
    lrs = [0.01, 0.1, 1.0]

    # Create identical models but different optimizers
    models = [create_model() for _ in lrs]
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m, lr in zip(models, lrs)]
    criterion = nn.BCELoss()

    # Grid for background contour
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    epochs = 400
    frames_data = []

    print("Training MLPs with different Learning Rates...")
    for epoch in range(epochs + 1):
        epoch_data = []
        for i in range(len(lrs)):
            optimizers[i].zero_grad()
            out = models[i](X)
            loss = criterion(out, y)
            loss.backward()
            optimizers[i].step()

            # Save frame data
            if epoch % 10 == 0:
                with torch.no_grad():
                    Z = models[i](grid).reshape(xx.shape).numpy()
                    epoch_data.append((loss.item(), Z))

        if epoch % 10 == 0:
            frames_data.append((epoch, epoch_data))

    # Add a few identical frames at the end
    for _ in range(10):
        frames_data.append(frames_data[-1])

    # Plot Setup
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def update(frame_idx):
        epoch, epoch_data = frames_data[frame_idx]

        for i, ax in enumerate(axes):
            ax.clear()
            loss_val, Z = epoch_data[i]

            ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=0.6)

            X_np = X.numpy()
            y_np = y.numpy().flatten()
            ax.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], color='red', marker='o', s=100, edgecolor='k')
            ax.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], color='blue', marker='s', s=100, edgecolor='k')

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(f"SGD LR = {lrs[i]}\nLoss: {loss_val:.4f}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"XOR Learning progression with different Learning Rates - Epoch {epoch}", fontsize=16)

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "04_nn_lr_comparison_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
