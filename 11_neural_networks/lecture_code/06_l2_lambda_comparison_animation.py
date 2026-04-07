import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_data(n_samples=30):
    torch.manual_seed(42)
    np.random.seed(42)
    # Szűk, zajos adat
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.sin(X) + 0.4 * torch.randn(X.size())
    return X, y

class SimpleRegNet(nn.Module):
    def __init__(self):
        super(SimpleRegNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return self.net(x)

def create_model_and_optim(weight_decay=0.0):
    torch.manual_seed(42)
    model = SimpleRegNet()
    # High learning rate to overfit quickly on the unregularized one
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    return model, opt

def main():
    X_train, y_train = get_data(30)
    X_grid = torch.linspace(-4, 4, 200).unsqueeze(1)
    y_true = torch.sin(X_grid)

    lambdas = [0.0, 0.01, 0.1, 1.0]

    # Modellek létrehozása különböző lambda (weight_decay) értékekkel
    models = {
        f'Lambda = {l}': create_model_and_optim(weight_decay=l)
        for l in lambdas
    }

    criterion = nn.MSELoss()
    epochs = 200

    frames_data = []

    print("Modellek tanítása (L2 Lambda összehasonlítás)...")
    for epoch in range(epochs + 1):
        epoch_preds = {}

        for name, (model, opt) in models.items():
            model.train()
            opt.zero_grad()
            out = model(X_train)
            loss = criterion(out, y_train)
            loss.backward()
            opt.step()

            # Save predictions for animation
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    epoch_preds[name] = model(X_grid).numpy()

        if epoch % 5 == 0:
            frames_data.append({
                'epoch': epoch,
                'preds': epoch_preds
            })

    # Pad animation
    for _ in range(10):
        frames_data.append(frames_data[-1])

    print("Animáció készítése...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    colors = ['red', 'orange', 'green', 'blue']

    def update(frame_idx):
        data = frames_data[frame_idx]

        for idx, (name, ax) in enumerate(zip(models.keys(), axes.flat)):
            ax.clear()
            ax.scatter(X_train.numpy(), y_train.numpy(), color='black', alpha=0.5, label='Képzési adat')
            ax.plot(X_grid.numpy(), y_true.numpy(), 'k--', label='Valódi függvény', alpha=0.5)

            pred = data['preds'][name]
            ax.plot(X_grid.numpy(), pred, color=colors[idx], lw=3, label=name)

            ax.set_xlim(-4, 4)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f"{name}", fontsize=14)
            if idx == 0:
                ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"L2 Regularizáció Lambda (Weight Decay) hatása (Epocha: {data['epoch']})", fontsize=16)
        return axes.flat

    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "06_l2_lambda_comparison_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
