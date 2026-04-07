import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_data(n_samples=1000):
    torch.manual_seed(42)
    np.random.seed(42)
    # 10 dimenziós bemenet, bináris osztályozás
    X = torch.randn(n_samples, 10)
    # Beiktatunk szándékos eltolást (covariate shift)
    X = X * 2.0 + 5.0
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 25).float().unsqueeze(1)
    return X, y

class NetNormMethod(nn.Module):
    def __init__(self, norm_type='none'):
        super().__init__()
        self.norm_type = norm_type

        self.fc1 = nn.Linear(10, 100)
        self.norm1 = self._get_norm_layer(norm_type, 100)
        self.act1 = nn.Sigmoid()

        self.fc2 = nn.Linear(100, 100)
        self.norm2 = self._get_norm_layer(norm_type, 100)
        self.act2 = nn.Sigmoid()

        self.fc3 = nn.Linear(100, 100)
        self.norm3 = self._get_norm_layer(norm_type, 100)
        self.act3 = nn.Sigmoid()

        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

        self.saved_acts = {}

    def _get_norm_layer(self, norm_type, dim):
        if norm_type == 'batch':
            return nn.BatchNorm1d(dim)
        elif norm_type == 'layer':
            return nn.LayerNorm(dim)
        elif norm_type == 'instance':
            # Instance norm expects 3D input (N, C, L), so we just wrap LayerNorm for 1D/2D
            return nn.InstanceNorm1d(dim)
        else:
            return nn.Identity()

    def forward(self, x):
        x = self.fc1(x)

        if self.norm_type == 'instance':
            x = x.unsqueeze(1)
            x = self.norm1(x)
            x = x.squeeze(1)
        else:
            x = self.norm1(x)

        self.saved_acts['fc1'] = x.detach()
        x = self.act1(x)

        x = self.fc2(x)
        if self.norm_type == 'instance':
            x = x.unsqueeze(1)
            x = self.norm2(x)
            x = x.squeeze(1)
        else:
            x = self.norm2(x)

        self.saved_acts['fc2'] = x.detach()
        x = self.act2(x)

        x = self.fc3(x)
        if self.norm_type == 'instance':
            x = x.unsqueeze(1)
            x = self.norm3(x)
            x = x.squeeze(1)
        else:
            x = self.norm3(x)

        self.saved_acts['fc3'] = x.detach()
        x = self.act3(x)

        x = self.out(x)
        return self.out_act(x)

def main():
    # Kis batch size a zajosabb normálásokhoz
    batch_size = 64
    X_train, y_train = get_data(1000)
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(42)
    models = {
        "Normálás Nélkül": NetNormMethod('none'),
        "Batch Normalization": NetNormMethod('batch'),
        "Layer Normalization": NetNormMethod('layer'),
        "Instance Normalization": NetNormMethod('instance'),
    }

    optimizers = {name: torch.optim.Adam(m.parameters(), lr=0.005) for name, m in models.items()}
    criterion = nn.BCELoss()
    epochs = 40

    frames_data = []

    print("Modellek tanítása (Különböző Normalizációs módszerek összehasonlítása)...")
    for epoch in range(epochs + 1):
        epoch_data = {'epoch': epoch, 'acts': {}, 'losses': {}}

        # Reset epoch loss
        for name in models:
            epoch_data['losses'][name] = 0.0

        for batch_x, batch_y in loader:
            for name, model in models.items():
                model.train()
                opt = optimizers[name]

                opt.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                opt.step()

                epoch_data['losses'][name] += loss.item() * batch_x.size(0)

        # Átlag loss és aktivációk kimentése (teljes batch-re epocha végén)
        for name, model in models.items():
            epoch_data['losses'][name] /= 1000

            model.eval()
            with torch.no_grad():
                model(X_train)
                epoch_data['acts'][name] = model.saved_acts['fc3'].numpy().flatten()

        frames_data.append(epoch_data)

    for _ in range(5):
        frames_data.append(frames_data[-1])

    print("Animáció készítése...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['red', 'green', 'purple', 'brown']

    def update(frame_idx):
        data = frames_data[frame_idx]

        for ax, name, color in zip(axes.flat, models.keys(), colors):
            ax.clear()
            bins = np.linspace(-6, 6, 60)
            ax.hist(data['acts'][name], bins=bins, color=color, alpha=0.7, density=True)
            ax.set_title(f"{name}\nLoss: {data['losses'][name]:.4f}")
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-6, 6)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Normalizációs Rétegek Hatása (Epoch: {data['epoch']})", fontsize=16)
        return axes.flat

    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "09_normalization_methods_animation.gif"
    anim.save(output_file, writer='pillow', fps=5)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
