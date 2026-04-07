import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # Cél: Beállítani a determinizmust a reprodukálhatóságért
    torch.manual_seed(42)

    # Egyszerű feladat: 8x8-as identitásmátrix kódolása egy szűk (6 neuronos) rétegen át
    X = torch.eye(8)
    y = torch.eye(8)

    # Létrehozunk egy egyszerű Autoencoder-szerű hálózatot biais (torzítás) nélkül az egyszerűbb vizualizációért
    model = nn.Sequential(
        nn.Linear(8, 6, bias=False),  # W1: 6x8 mátrix
        nn.Tanh(),
        nn.Linear(6, 8, bias=False)   # W2: 8x6 mátrix
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    criterion = nn.MSELoss()

    frames = []
    epochs = 150

    print("Hálózat tanítása és tensorok rögzítése...")
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()

        # Minden páros epochában lementjük a súlyokat és a fast backward által kiszámolt gradienseket
        if epoch % 2 == 0:
            w1 = model[0].weight.data.clone().numpy()
            w2 = model[2].weight.data.clone().numpy()
            g1 = model[0].weight.grad.clone().numpy() if model[0].weight.grad is not None else np.zeros_like(w1)
            g2 = model[2].weight.grad.clone().numpy() if model[2].weight.grad is not None else np.zeros_like(w2)

            frames.append({
                'epoch': epoch,
                'loss': loss.item(),
                'w1': w1, 'w2': w2,
                'g1': g1, 'g2': g2
            })

        optimizer.step()

    for _ in range(10):
        frames.append(frames[-1])

    # Plot Setup
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    (ax_w1, ax_w2), (ax_g1, ax_g2) = axes

    # Színskálák korlátai
    vmax_w = 1.0
    cmap = 'RdBu_r'

    def update(frame_idx):
        data = frames[frame_idx]

        for ax in axes.flat:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Súlyok ábrázolása
        im_w1 = ax_w1.imshow(data['w1'], cmap=cmap, vmin=-vmax_w, vmax=vmax_w)
        ax_w1.set_title("W1 Súlyok (6x8)")

        im_w2 = ax_w2.imshow(data['w2'], cmap=cmap, vmin=-vmax_w, vmax=vmax_w)
        ax_w2.set_title("W2 Súlyok (8x6)")

        # Gradiensek ábrázolása - Normalizálva per frame a láthatóságért
        g1 = data['g1']
        g2 = data['g2']
        vmax_g1 = max(np.max(np.abs(g1)), 1e-5)
        vmax_g2 = max(np.max(np.abs(g2)), 1e-5)

        im_g1 = ax_g1.imshow(g1, cmap=cmap, vmin=-vmax_g1, vmax=vmax_g1)
        ax_g1.set_title("W1 Gradiensek (Normalizálva)")

        im_g2 = ax_g2.imshow(g2, cmap=cmap, vmin=-vmax_g2, vmax=vmax_g2)
        ax_g2.set_title("W2 Gradiensek (Normalizálva)")

        fig.suptitle(f"Súlyok és Gradiensek (Backprop) Alakulása\nEpocha: {data['epoch']} | Loss: {data['loss']:.4f}", fontsize=16)

    print("Animáció készítése...")
    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)

    output_file = "05_weight_tensor_animation.gif"
    anim.save(output_file, writer='pillow', fps=8)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
