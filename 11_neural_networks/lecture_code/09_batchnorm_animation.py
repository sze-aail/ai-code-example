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
    y = (X[:, 0] * X[:, 1] + X[:, 2] > 0).float().unsqueeze(1)
    return X, y

class NetWithoutBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 100)
        self.act1 = nn.Sigmoid()
        self.fc2 = nn.Linear(100, 100)
        self.act2 = nn.Sigmoid()
        self.fc3 = nn.Linear(100, 100)
        self.act3 = nn.Sigmoid()
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

        self.saved_acts = {}

    def forward(self, x):
        x = self.fc1(x)
        self.saved_acts['fc1'] = x.detach()
        x = self.act1(x)

        x = self.fc2(x)
        self.saved_acts['fc2'] = x.detach()
        x = self.act2(x)

        x = self.fc3(x)
        self.saved_acts['fc3'] = x.detach()
        x = self.act3(x)

        x = self.out(x)
        return self.out_act(x)

class ParameterizedNetBN(nn.Module):
    def __init__(self, momentum=0.1, affine=True):
        super().__init__()
        self.fc1 = nn.Linear(10, 100)
        self.bn1 = nn.BatchNorm1d(100, momentum=momentum, affine=affine)
        self.act1 = nn.Sigmoid()

        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, momentum=momentum, affine=affine)
        self.act2 = nn.Sigmoid()

        self.fc3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100, momentum=momentum, affine=affine)
        self.act3 = nn.Sigmoid()

        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

        self.saved_acts = {}

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        self.saved_acts['fc1'] = x.detach()
        x = self.act1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        self.saved_acts['fc2'] = x.detach()
        x = self.act2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        self.saved_acts['fc3'] = x.detach()
        x = self.act3(x)

        x = self.out(x)
        return self.out_act(x)

def main():
    X_train, y_train = get_data(1000)

    torch.manual_seed(42)
    models = {
        "BN Nélkül": NetWithoutBN(),
        "BN (Alap, Mom=0.1, Affine=True)": ParameterizedNetBN(),
        "BN (Magas Mom=0.9)": ParameterizedNetBN(momentum=0.9),
        "BN (Affine=False)": ParameterizedNetBN(affine=False),
    }

    optimizers = {name: torch.optim.Adam(m.parameters(), lr=0.01) for name, m in models.items()}
    criterion = nn.BCELoss()
    epochs = 100

    frames_data = []

    print("Modellek tanítása (Batch Norm paraméterek összehasonlítása)...")
    for epoch in range(epochs + 1):
        epoch_data = {'epoch': epoch, 'acts': {}, 'losses': {}}
        for name, model in models.items():
            model.train()
            opt = optimizers[name]

            opt.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            opt.step()

            if epoch % 2 == 0:
                epoch_data['acts'][name] = model.saved_acts['fc3'].numpy().flatten()
                epoch_data['losses'][name] = loss.item()

        if epoch % 2 == 0:
            frames_data.append(epoch_data)

    for _ in range(10):
        frames_data.append(frames_data[-1])

    print("Animáció készítése...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['red', 'green', 'blue', 'orange']

    def update(frame_idx):
        data = frames_data[frame_idx]

        for ax, name, color in zip(axes.flat, models.keys(), colors):
            ax.clear()
            bins = np.linspace(-5, 5, 50)
            ax.hist(data['acts'][name], bins=bins, color=color, alpha=0.7, density=True)
            ax.set_title(f"{name}\nLoss: {data['losses'][name]:.4f}")
            ax.set_ylim(0, 1.0)
            ax.set_xlim(-5, 5)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Batch Norm Paraméterek Hatása (Epoch: {data['epoch']})", fontsize=16)
        return axes.flat

    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "09_batchnorm_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
