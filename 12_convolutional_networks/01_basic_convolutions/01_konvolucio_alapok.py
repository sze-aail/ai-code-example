"""
01_konvolucio_alapok.py — Konvolúció alapok: 1D és 2D
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - 1D konvolúció kézzel és numpy-val (full, valid, same)
  - 2D konvolúció képfeldolgozásban (Sobel, Gauss, élesítés, domborítás)
  - PyTorch nn.Conv2d összehasonlítás
  - Stride és padding hatása a kimeneti méretre
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. 1D KONVOLÚCIÓ
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. 1D KONVOLÚCIÓ")
    print("=" * 60)

    x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    h = np.array([1, 0, -1], dtype=float)  # éldetektáló kernel

    conv_full = np.convolve(x, h, mode='full')
    conv_valid = np.convolve(x, h, mode='valid')
    conv_same = np.convolve(x, h, mode='same')

    print(f"  Bemenet x:     {x}")
    print(f"  Kernel h:      {h}")
    print(f"  Full ({len(conv_full)}):   {conv_full}")
    print(f"  Valid ({len(conv_valid)}):  {conv_valid}")
    print(f"  Same ({len(conv_same)}):   {conv_same}")

    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4))
    modes = [("Full", conv_full), ("Valid", conv_valid), ("Same", conv_same)]
    for ax, (name, result) in zip(axes1, modes):
        ax.stem(range(len(x)), x, linefmt="b-", markerfmt="bo", basefmt="k-", label="Bemenet")
        ax.stem(range(len(result)), result, linefmt="r-", markerfmt="r^", basefmt="k-", label=f"Conv ({name})")
        ax.set_title(f"mode='{name.lower()}' (kimenet: {len(result)})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    fig1.suptitle("1D konvolúció: éldetektáló kernel [1, 0, -1]", fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("01_conv1d.png", dpi=150)
    print("Ábra mentve: 01_conv1d.png\n")

    # Második 1D példa: zajszűrés / simítás
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    x_noisy = np.sin(t) + np.random.normal(0, 0.2, len(t))
    h_smooth = np.ones(5) / 5  # Mozgóátlag (Moving average) kernel

    conv_smooth = np.convolve(x_noisy, h_smooth, mode='same')

    fig1b, ax1b = plt.subplots(figsize=(10, 4))
    ax1b.plot(t, x_noisy, 'k-', alpha=0.5, label="Zajos idősor")
    ax1b.plot(t, conv_smooth, 'r-', linewidth=2, label="Szűrt (simított) kimenet")
    ax1b.set_title("1D konvolúció: Zajos jel simítása (moving average kernel)", fontsize=12, fontweight="bold")
    ax1b.legend()
    ax1b.grid(True, alpha=0.3)

    fig1b.tight_layout()
    fig1b.savefig("01_conv1d_smoothing.png", dpi=150)
    print("Ábra mentve: 01_conv1d_smoothing.png\n")


    # ══════════════════════════════════════════════════════════════
    # 2. 2D KONVOLÚCIÓ: KÉPSZŰRŐK
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("2. 2D KONVOLÚCIÓ: képszűrők")
    print("=" * 60)

    # Szintetikus kép: sakktábla + gradiens + kör
    img = np.zeros((64, 64), dtype=np.float32)
    # Sakktábla
    for i in range(0, 64, 8):
        for j in range(0, 64, 8):
            if (i // 8 + j // 8) % 2 == 0:
                img[i:i+8, j:j+8] = 1.0
    # Kör
    yy, xx = np.mgrid[:64, :64]
    circle = ((xx - 40)**2 + (yy - 20)**2) < 100
    img[circle] = 0.7

    kernels = {
        "Identitás": np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
        "Éldetektálás\n(Laplacian)": np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        "Sobel (vízsz.)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
        "Sobel (függ.)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
        "Gauss (simítás)": np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16,
        "Élesítés": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
        "Domborítás\n(emboss)": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
        "Box blur\n(átlagoló)": np.ones((3,3), dtype=np.float32) / 9,
    }

    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))

    for ax, (name, kernel) in zip(axes2.flat, kernels.items()):
        # Konvolúció scipy nélkül: PyTorch-cal
        img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        k_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)
        result = F.conv2d(img_t, k_t, padding=1).squeeze().numpy()

        ax.imshow(result, cmap="gray")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig2.suptitle("2D konvolúciós szűrők hatása (3×3 kernelek)", fontsize=15, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("01_conv2d_szurok.png", dpi=150)
    print("Ábra mentve: 01_conv2d_szurok.png\n")


    # ══════════════════════════════════════════════════════════════
    # 3. STRIDE ÉS PADDING HATÁSA
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("3. STRIDE ÉS PADDING")
    print("=" * 60)

    fig3, axes3 = plt.subplots(2, 3, figsize=(16, 10))
    configs = [
        ("padding=0, stride=1", 0, 1),
        ("padding=1, stride=1\n(same)", 1, 1),
        ("padding=0, stride=2", 0, 2),
        ("padding=1, stride=2", 1, 2),
        ("padding=2, stride=1", 2, 1),
        ("padding=2, stride=3", 2, 3),
    ]

    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    sobel = torch.tensor(kernels["Sobel (vízsz.)"]).unsqueeze(0).unsqueeze(0)

    for ax, (name, pad, stride) in zip(axes3.flat, configs):
        result = F.conv2d(img_t, sobel, padding=pad, stride=stride).squeeze().numpy()
        ax.imshow(result, cmap="gray")
        h_out, w_out = result.shape
        ax.set_title(f"{name}\nKimenet: {h_out}×{w_out}", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig3.suptitle("Stride és padding hatása a kimeneti méretre\nKéplet: out = ⌊(in + 2p - k) / s⌋ + 1",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("01_stride_padding.png", dpi=150)
    print("Ábra mentve: 01_stride_padding.png")

    # Kimeneti méret képlet
    print("\n  Kimeneti méret képlet: out = floor((in + 2*padding - kernel) / stride) + 1")
    for name, pad, stride in configs:
        out = (64 + 2*pad - 3) // stride + 1
        print(f"    {name.split(chr(10))[0]:25s} → ({64}+2*{pad}-3)/{stride}+1 = {out}")

    plt.close("all")
    print("\nKész!")

if __name__ == "__main__":
    main()