"""
03_pooling.py — Pooling típusok összehasonlítása
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - MaxPool vs. AvgPool vs. Strided Convolution
  - Pooling hatása a feature map-ekre vizuálisan
  - Információveszítés és invariancia
  - Adaptive pooling és Global Average Pooling (GAP)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def main():
    # ── Adat ──
    digits = load_digits()
    img = digits.data[0].reshape(8, 8).astype(np.float32) / 16.0  # egy '0'
    img_big = np.kron(img, np.ones((4, 4)))  # 32×32-re felnagyítás

    # ══════════════════════════════════════════════════════════════
    # 1. MaxPool vs AvgPool vs Strided Conv
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. POOLING TÍPUSOK ÖSSZEHASONLÍTÁSA")
    print("=" * 60)

    img_t = torch.tensor(img_big, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,32,32)

    maxpool = nn.MaxPool2d(2, 2)
    avgpool = nn.AvgPool2d(2, 2)
    strided = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
    nn.init.ones_(strided.weight); strided.weight.data /= 9  # átlagoló

    gap = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling → 1×1

    with torch.no_grad():
        mp = maxpool(img_t).squeeze().numpy()
        ap = avgpool(img_t).squeeze().numpy()
        sc = strided(img_t).squeeze().numpy()
        gp = gap(img_t).squeeze().item()

    results = [
        ("Eredeti (32×32)", img_big),
        (f"MaxPool 2×2\n({mp.shape[0]}×{mp.shape[1]})", mp),
        (f"AvgPool 2×2\n({ap.shape[0]}×{ap.shape[1]})", ap),
        (f"Strided Conv (s=2)\n({sc.shape[0]}×{sc.shape[1]})", sc),
    ]

    fig1, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (name, data) in zip(axes, results):
        ax.imshow(data, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.axis("off")

    fig1.suptitle("Pooling típusok: MaxPool megőrzi az éleket, AvgPool simít", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("03_pooling_tipusok.png", dpi=150)
    print("Ábra mentve: 03_pooling_tipusok.png")

    # ══════════════════════════════════════════════════════════════
    # 2. ISMÉTELT POOLING: információveszítés
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("2. ISMÉTELT POOLING: információveszítés")
    print("=" * 60)

    fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
    titles_row = ["MaxPool", "AvgPool"]

    for row, (pool_fn, title) in enumerate([(maxpool, "MaxPool"), (avgpool, "AvgPool")]):
        x = img_t.clone()
        for col in range(5):
            data = x.squeeze().numpy()
            axes2[row, col].imshow(data, cmap="gray_r", vmin=0, vmax=1)
            axes2[row, col].set_title(f"{title}: {data.shape[0]}×{data.shape[1]}", fontsize=10, fontweight="bold")
            axes2[row, col].axis("off")
            if data.shape[0] > 1:
                x = pool_fn(x)

    fig2.suptitle("Ismételt pooling: minden lépés felezi a méretet → információveszítés",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("03_pooling_ismetelt.png", dpi=150)
    print("Ábra mentve: 03_pooling_ismetelt.png")

    # ══════════════════════════════════════════════════════════════
    # 3. ELTOLÁS-INVARIANCIA DEMONSTRÁCIÓ
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("3. POOLING → ELTOLÁS-INVARIANCIA")
    print("=" * 60)

    fig3, axes3 = plt.subplots(2, 4, figsize=(18, 8))

    # Eredeti és eltolt kép
    orig = np.zeros((16, 16), dtype=np.float32)
    orig[4:10, 4:10] = 1.0  # négyzet

    shifts = [(0, 0), (1, 0), (0, 1), (1, 1)]

    for col, (dy, dx) in enumerate(shifts):
        shifted = np.roll(np.roll(orig, dy, axis=0), dx, axis=1)
        axes3[0, col].imshow(shifted, cmap="gray_r")
        axes3[0, col].set_title(f"Eltolás ({dy},{dx})", fontweight="bold")
        axes3[0, col].axis("off")

        # MaxPool után
        shifted_t = torch.tensor(shifted).unsqueeze(0).unsqueeze(0)
        pooled = nn.MaxPool2d(2, 2)(shifted_t).squeeze().numpy()
        axes3[1, col].imshow(pooled, cmap="gray_r")
        axes3[1, col].set_title(f"MaxPool 2×2 után", fontweight="bold")
        axes3[1, col].axis("off")

    # Hasonlóság mérése pooling után
    pool_results = []
    for dy, dx in shifts:
        shifted = np.roll(np.roll(orig, dy, axis=0), dx, axis=1)
        shifted_t = torch.tensor(shifted).unsqueeze(0).unsqueeze(0)
        pooled = nn.MaxPool2d(2, 2)(shifted_t).squeeze().numpy()
        pool_results.append(pooled)

    diffs = [np.mean((pool_results[0] - p)**2) for p in pool_results]
    print("  Pooling utáni MSE az eredetihez képest:")
    for (dy, dx), d in zip(shifts, diffs):
        print(f"    Eltolás ({dy},{dx}): MSE = {d:.4f}")

    fig3.suptitle("Pooling → eltolás-invariancia: kis elmozdulás eltűnik a pooling után",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("03_pooling_invariancia.png", dpi=150)
    print("Ábra mentve: 03_pooling_invariancia.png")

    plt.close("all")
    print("\nKész!")

if __name__ == "__main__":
    main()