import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_data(n_samples=500):
    np.random.seed(42)
    torch.manual_seed(42)

    # Custom make_moons-like data generation without sklearn dependency
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    # Add noise
    X += np.random.normal(scale=0.15, size=X.shape)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# --- Define Different Architectures ---
class NetLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

class NetShallow(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class NetMedium(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ReLU(),
            nn.Linear(16, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class NetDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def main():
    X_train, y_train = get_data(500)

    models = {
        "Nincs Rejtett Réteg (Lineáris)": NetLinear(),
        "Sekély (1x8 Rejtett)": NetShallow(),
        "Közepes (2x16 Rejtett)": NetMedium(),
        "Mély (4x32 Rejtett)": NetDeep()
    }

    optimizers = {name: torch.optim.Adam(m.parameters(), lr=0.02) for name, m in models.items()}
    criterion = nn.BCELoss()

    epochs = 200
    frames_data = []

    # Grid for decision boundaries
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    X_grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    print("Különböző Architektúrák Tanítása...")
    for epoch in range(epochs + 1):
        epoch_preds = {}
        for name, model in models.items():
            model.train()
            opt = optimizers[name]

            opt.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            opt.step()

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    epoch_preds[name] = model(X_grid).numpy().reshape(xx.shape)

        if epoch % 5 == 0:
            frames_data.append({
                'epoch': epoch,
                'preds': epoch_preds
            })

    for _ in range(10):
        frames_data.append(frames_data[-1])

    print("Animáció készítése...")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    X_np = X_train.numpy()
    y_np = y_train.numpy().flatten()

    def update(frame_idx):
        data = frames_data[frame_idx]

        for ax, name in zip(axes.flat, models.keys()):
            ax.clear()
            ax.contourf(xx, yy, data['preds'][name], levels=np.linspace(0, 1, 11), cmap='RdBu', alpha=0.6)

            ax.scatter(X_np[y_np==0, 0], X_np[y_np==0, 1], color='red', edgecolor='k', s=20)
            ax.scatter(X_np[y_np==1, 0], X_np[y_np==1, 1], color='blue', edgecolor='k', s=20)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_title(name, fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Neurális Háló Rétegeinek és Kapacitásának Összehasonlítása\nEpocha: {data['epoch']}", fontsize=16)
        return axes.flat

    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "16_nn_architecture_comparison_animation.gif"
    anim.save(output_file, writer='pillow', fps=10)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
