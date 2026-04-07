import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # Fix seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # XOR Dataset
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # Simple MLP Model
    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()

    # Grid for background contour
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    # Train and collect frames
    epochs = 300
    frames_data = []

    print("Training MLP for XOR...")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # Save frame every 5 epochs
        if epoch % 5 == 0:
            with torch.no_grad():
                Z = model(grid).reshape(xx.shape).numpy()
                frames_data.append((epoch, loss.item(), Z))

    # Add a few identical frames at the end to hold the final decision boundary
    for _ in range(10):
        frames_data.append(frames_data[-1])

    # Plot Setup
    fig, ax = plt.subplots(figsize=(6, 5))

    def update(frame_idx):
        ax.clear()
        epoch, loss_val, Z = frames_data[frame_idx]

        # Draw contour
        contour = ax.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=0.6)

        # Plot points
        X_np = X.numpy()
        y_np = y.numpy().flatten()
        ax.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], color='red', marker='o', s=150, edgecolor='k', label='Class 0')
        ax.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], color='blue', marker='s', s=150, edgecolor='k', label='Class 1')

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"XOR Problem Learning (MLP)\nEpoch {epoch} | Loss: {loss_val:.4f}")
        ax.grid(True, alpha=0.3)

        # Only add legend to the first frame to avoid a bit of overhead, or just do it inside.
        # Since we clear ax, we must re-add it.
        ax.legend(loc="upper right")

    # Animate
    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "02_xor_learning_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
