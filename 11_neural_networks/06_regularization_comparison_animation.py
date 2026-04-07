import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_data(n_samples=50):
    torch.manual_seed(42)
    np.random.seed(42)
    # Szűk, zajos adat
    # Alulreprezentált, ezért nagyon könnyű túltanulni (overfit)
    X = torch.linspace(-3, 3, n_samples).unsqueeze(1)
    y = torch.sin(X) + 0.3 * torch.randn(X.size())
    return X, y

class RegNet(nn.Module):
    def __init__(self, use_dropout=False):
        super(RegNet, self).__init__()
        self.use_dropout = use_dropout

        self.fc1 = nn.Linear(1, 100)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        self.fc2 = nn.Linear(100, 100)
        self.act2 = nn.ReLU()
        self.drop2 = nn.Dropout(0.5) if use_dropout else nn.Identity()

        self.out = nn.Linear(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)

        x = self.out(x)
        return x

def create_model_and_optim(weight_decay=0.0, use_dropout=False):
    torch.manual_seed(42)
    model = RegNet(use_dropout=use_dropout)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    return model, opt

def main():
    X_train, y_train = get_data(30)
    X_grid = torch.linspace(-4, 4, 200).unsqueeze(1)
    y_true = torch.sin(X_grid)

    # Modellek
    models = {
        'Alap (No Reg)': create_model_and_optim(weight_decay=0.0, use_dropout=False),
        'L2 (Weight Decay)': create_model_and_optim(weight_decay=0.1, use_dropout=False),
        'Dropout (p=0.5)': create_model_and_optim(weight_decay=0.0, use_dropout=True)
    }

    criterion = nn.MSELoss()
    epochs = 200

    frames_data = []

    print("Modellek tanítása (Regularizáció összehasonlítás)...")
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {
        'Alap (No Reg)': 'red',
        'L2 (Weight Decay)': 'green',
        'Dropout (p=0.5)': 'purple'
    }

    def update(frame_idx):
        data = frames_data[frame_idx]

        for idx, (name, ax) in enumerate(zip(models.keys(), axes)):
            ax.clear()
            ax.scatter(X_train.numpy(), y_train.numpy(), color='black', alpha=0.5, label='Képzési adat')
            ax.plot(X_grid.numpy(), y_true.numpy(), 'k--', label='Valódi függvény', alpha=0.5)

            pred = data['preds'][name]
            ax.plot(X_grid.numpy(), pred, color=colors[name], lw=3, label=name)

            ax.set_xlim(-4, 4)
            ax.set_ylim(-2.5, 2.5)
            ax.set_title(f"{name}", fontsize=14)
            if idx == 2:
                ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Regularizációs módszerek összehasonlítása (Epocha: {data['epoch']})", fontsize=16)
        return axes

    anim = FuncAnimation(fig, update, frames=len(frames_data), blit=False)

    output_file = "06_regularization_comparison_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
