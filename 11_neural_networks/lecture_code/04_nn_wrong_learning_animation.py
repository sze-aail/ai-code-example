import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_mlp_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(2, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    )

def create_linear_model():
    torch.manual_seed(42)
    return nn.Sequential(
        nn.Linear(2, 1),
        nn.Sigmoid()
    )

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # XOR Dataset
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    # 1. Gradient Ascent (Wrong direction - maximizing loss)
    model_ga = create_mlp_model()
    opt_ga = torch.optim.SGD(model_ga.parameters(), lr=0.1)

    # 2. Linear model on XOR (Capacity too small - underfitting)
    model_lin = create_linear_model()
    opt_lin = torch.optim.SGD(model_lin.parameters(), lr=0.1)

    # 3. Exploding LR
    model_exp = create_mlp_model()
    opt_exp = torch.optim.SGD(model_exp.parameters(), lr=10.0)

    criterion = nn.BCELoss()

    titles = ["Gradient Ascent (Rosszirányú tanulás)",
              "Lineáris modell (Alulillesztés - XOR)",
              "Túl nagy LR (Felvevő/Felrobbanó)"]

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    epochs = 200
    frames_data = []

    print("Training deliberately wrong models...")
    for epoch in range(epochs + 1):
        epoch_data = []

        # 1. Gradient Ascent
        opt_ga.zero_grad()
        out_ga = model_ga(X)
        loss_ga = criterion(out_ga, y)
        loss_ga_inv = -loss_ga  # Negative loss for ascent
        loss_ga_inv.backward()
        opt_ga.step()

        # 2. Linear model
        opt_lin.zero_grad()
        out_lin = model_lin(X)
        loss_lin = criterion(out_lin, y)
        loss_lin.backward()
        opt_lin.step()

        # 3. Exploding LR
        opt_exp.zero_grad()
        out_exp = model_exp(X)
        loss_exp = criterion(out_exp, y)
        loss_exp.backward()
        opt_exp.step()

        # Save frame data
        if epoch % 5 == 0:
            with torch.no_grad():
                Z_ga = model_ga(grid).reshape(xx.shape).numpy()
                Z_lin = model_lin(grid).reshape(xx.shape).numpy()
                Z_exp = model_exp(grid).reshape(xx.shape).numpy()

                epoch_data.append((loss_ga.item(), Z_ga))
                epoch_data.append((loss_lin.item(), Z_lin))
                epoch_data.append((loss_exp.item(), Z_exp))

            frames_data.append((epoch, epoch_data))

    # Pad at end
    for _ in range(10):
        frames_data.append(frames_data[-1])

    # Plot
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
            ax.set_title(f"{titles[i]}\nLoss: {loss_val:.4f}")
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Szándékosan elrontott tanulási folyamatok - Epocha: {epoch}", fontsize=16)

    print("Generating animation...")
    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "04_nn_wrong_learning_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
