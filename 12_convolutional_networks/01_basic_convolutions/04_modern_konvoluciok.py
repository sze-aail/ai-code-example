"""
04_modern_konvoluciok.py — Modern konvolúciós típusok
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - 1×1 konvolúció (Network in Network): csatorna-dimenzió csökkentés
  - Depthwise Separable Convolution (MobileNet): paraméter-hatékonyság
  - Dilated/Atrous Convolution: receptive field növelés pooling nélkül
  - Transposed Convolution: upsampling szegmentációhoz
  - Paraméterszám összehasonlítás
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def main():
    # ══════════════════════════════════════════════════════════════
    # 1. 1×1 KONVOLÚCIÓ
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. 1×1 KONVOLÚCIÓ (Network in Network, Lin et al. 2013)")
    print("=" * 60)

    # 256 csatornás feature map → 64 csatornára csökkentés
    conv_1x1 = nn.Conv2d(256, 64, kernel_size=1)
    x = torch.randn(1, 256, 16, 16)
    out_1x1 = conv_1x1(x)

    # Összehasonlítás 3×3 konvolúcióval
    conv_3x3 = nn.Conv2d(256, 64, kernel_size=3, padding=1)
    out_3x3 = conv_3x3(x)

    params_1x1 = sum(p.numel() for p in conv_1x1.parameters())
    params_3x3 = sum(p.numel() for p in conv_3x3.parameters())

    print(f"  Bemenet:  {list(x.shape)}")
    print(f"  1×1 conv: {list(out_1x1.shape)}, params = {params_1x1:,}")
    print(f"  3×3 conv: {list(out_3x3.shape)}, params = {params_3x3:,}")
    print(f"  Megtakarítás: {(1-params_1x1/params_3x3)*100:.0f}%")


    # ══════════════════════════════════════════════════════════════
    # 2. DEPTHWISE SEPARABLE CONVOLUTION
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("2. DEPTHWISE SEPARABLE CONV (MobileNet, Howard et al. 2017)")
    print("=" * 60)

    class DepthwiseSeparable(nn.Module):
        """Depthwise (csatornánként) + Pointwise (1×1)."""
        def __init__(self, in_ch, out_ch, kernel_size=3):
            super().__init__()
            self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, padding=kernel_size//2, groups=in_ch)
            self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        def forward(self, x):
            return self.pointwise(self.depthwise(x))

    standard = nn.Conv2d(64, 128, 3, padding=1)
    depthwise_sep = DepthwiseSeparable(64, 128)

    p_std = sum(p.numel() for p in standard.parameters())
    p_dws = sum(p.numel() for p in depthwise_sep.parameters())

    print(f"  Standard 3×3 conv (64→128): {p_std:,} params")
    print(f"  Depthwise separable (64→128): {p_dws:,} params")
    print(f"  Megtakarítás: {(1-p_dws/p_std)*100:.1f}% kevesebb paraméter!")
    print(f"  Arány: 1/out_ch + 1/k² = 1/{128} + 1/{9} ≈ {1/128 + 1/9:.3f}")


    # ══════════════════════════════════════════════════════════════
    # 3. DILATED (ATROUS) CONVOLUTION
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("3. DILATED CONVOLUTION (Yu & Koltun, 2016)")
    print("=" * 60)

    # Receptive field különbség
    dilations = [1, 2, 4, 8]
    print("  3×3 kernel különböző dilation-nel:")
    for d in dilations:
        effective_k = 3 + (3-1)*(d-1)  # effektív kernel méret
        print(f"    dilation={d}: effektív kernel = {effective_k}×{effective_k}, "
              f"receptive field = {effective_k}")


    # ══════════════════════════════════════════════════════════════
    # 4. TRANSPOSED CONVOLUTION
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("4. TRANSPOSED CONVOLUTION (upsampling)")
    print("=" * 60)

    x_small = torch.randn(1, 1, 4, 4)
    trans_conv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
    upsample_bilinear = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    out_trans = trans_conv(x_small)
    out_bilin = upsample_bilinear(x_small)

    print(f"  Bemenet: {list(x_small.shape)}")
    print(f"  Transposed conv (s=2): {list(out_trans.shape)} — TANULHATÓ upsampling")
    print(f"  Bilinear upsample (2×): {list(out_bilin.shape)} — fix interpoláció")


    # ══════════════════════════════════════════════════════════════
    # ÖSSZEFOGLALÓ ÁBRA
    # ══════════════════════════════════════════════════════════════

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    # 1. 1×1 conv vizualizáció
    ax = axes[0, 0]
    ax.bar(["Standard\n3×3 conv", "1×1 conv"], [params_3x3, params_1x1],
           color=["#E91E63", "#4CAF50"], edgecolor="k")
    ax.set_ylabel("Paraméterek"); ax.set_title("1×1 conv: dimenziócsökkentés\n(256→64 csatorna)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([params_3x3, params_1x1]):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # 2. Depthwise separable
    ax = axes[0, 1]
    ax.bar(["Standard\n3×3", "Depthwise\nSeparable"], [p_std, p_dws],
           color=["#E91E63", "#4CAF50"], edgecolor="k")
    ax.set_ylabel("Paraméterek"); ax.set_title("Depthwise Separable Conv\n(64→128 csatorna)", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([p_std, p_dws]):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontweight="bold")

    # 3. Dilated conv receptive field
    ax = axes[0, 2]
    colors_d = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    for d, color in zip(dilations, colors_d):
        eff_k = 3 + (3-1)*(d-1)
        # Rajzoljuk ki a mintázatot
        grid = np.zeros((eff_k, eff_k))
        for i in range(3):
            for j in range(3):
                grid[i*d, j*d] = 1
        ax_sub = fig.add_axes([0.67 + (d-1)*0.065, 0.55 + (0 if d<=2 else -0.22),
                               0.06, 0.06 * eff_k/max(3+(3-1)*7, 1)])
        ax_sub.imshow(grid, cmap="Blues", vmin=0, vmax=1)
        ax_sub.set_title(f"d={d}", fontsize=7); ax_sub.axis("off")

    rf_sizes = [3 + (3-1)*(d-1) for d in dilations]
    ax.bar([f"d={d}" for d in dilations], rf_sizes, color=colors_d, edgecolor="k")
    ax.set_ylabel("Receptive field méret"); ax.set_title("Dilated Conv: RF növelés\npooling nélkül", fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Transposed conv vizualizáció
    ax = axes[1, 0]
    with torch.no_grad():
        small_img = torch.zeros(1, 1, 4, 4)
        small_img[0, 0, 1, 1] = 1; small_img[0, 0, 2, 2] = 1
        up_img = nn.ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1, bias=False)
        nn.init.ones_(up_img.weight); up_img.weight.data /= 9
        result = up_img(small_img).squeeze().numpy()

    ax_sub1 = fig.add_axes([0.06, 0.12, 0.08, 0.15])
    ax_sub1.imshow(small_img.squeeze(), cmap="Blues"); ax_sub1.set_title("4×4", fontsize=8); ax_sub1.axis("off")
    ax_sub2 = fig.add_axes([0.17, 0.08, 0.14, 0.22])
    ax_sub2.imshow(result, cmap="Blues"); ax_sub2.set_title("8×8", fontsize=8); ax_sub2.axis("off")
    ax.axis("off")
    ax.set_title("Transposed Conv: tanulható upsampling\n(szegmentációhoz, decoder-hez)", fontweight="bold")

    # 5. Összefoglaló táblázat
    ax = axes[1, 1]
    ax.axis("off")
    table_data = [
        ["Típus", "Cél", "Param."],
        ["1×1 conv", "Csatorna-csökkentés", "c_in × c_out"],
        ["Depthwise Sep.", "Param. csökkentés", "~1/k² × standard"],
        ["Dilated (d)", "RF növelés", "= standard"],
        ["Transposed", "Upsampling", "= standard"],
    ]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.0, 2.0)
    for j in range(3):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Összefoglalás", fontweight="bold")

    # 6. Receptive field növekedés rétegenként
    ax = axes[1, 2]
    # Klasszikus 3×3 conv-ok egymás után
    layers = list(range(1, 11))
    rf_standard = [2*l + 1 for l in layers]  # RF = 2L+1 (L réteg 3×3 conv)
    rf_dilated = []
    rf = 1
    for l in layers:
        d = min(2**(l-1), 8)
        rf = rf + 2*d  # minden réteg hozzáad 2*dilation-t
        rf_dilated.append(rf)

    ax.plot(layers, rf_standard, "o-", color="#2196F3", linewidth=2, label="Standard 3×3")
    ax.plot(layers, rf_dilated, "s-", color="#E91E63", linewidth=2, label="Dilated (d=1,2,4,8...)")
    ax.set_xlabel("Rétegek száma"); ax.set_ylabel("Receptive field")
    ax.set_title("Receptive field növekedés", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("Modern konvolúciós típusok", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig("04_modern_konvoluciok.png", dpi=150)
    print("\nÁbra mentve: 04_modern_konvoluciok.png")

    plt.close("all")
    print("\nKész!")

if __name__ == "__main__":
    main()