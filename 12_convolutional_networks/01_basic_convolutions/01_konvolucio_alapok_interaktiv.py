"""
01_konvolució_alapok_interaktiv.py — Konvolúció alapok interaktív vizualizációja
Neurális hálók II. (CNN) — Hajdu Csaba

Interaktív vizualizáció, ahol a felhasználó valós időben tesztelheti
a különböző konvolúciós kerneleket, valamint a stride (lépésköz) és a
padding (kitöltés) hatását a kimeneti képre és annak felbontására.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def create_synthetic_image():
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
    return img

def main():
    img = create_synthetic_image()
    img_t = torch.tensor(img).unsqueeze(0).unsqueeze(0)

    kernels = {
        "Identitás": np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32),
        "Laplacian (Él)": np.array([[0,1,0],[1,-4,1],[0,1,0]], dtype=np.float32),
        "Sobel (vízsz.)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32),
        "Sobel (függ.)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32),
        "Gauss (simítás)": np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float32) / 16,
        "Élesítés": np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32),
        "Domborítás": np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32),
        "Box blur": np.ones((3,3), dtype=np.float32) / 9,
    }

    fig, (ax_orig, ax_kernel, ax_conv) = plt.subplots(1, 3, figsize=(15, 6), gridspec_kw={'width_ratios': [2, 1, 2]})
    plt.subplots_adjust(left=0.3, bottom=0.3)

    ax_orig.imshow(img, cmap="gray")
    ax_orig.set_title(f"Bemeneti kép ({img.shape[0]}×{img.shape[1]})", fontweight="bold")
    ax_orig.axis("off")

    ax_kernel.axis("off")
    ax_conv.axis("off")

    # GUI elemek
    ax_radio = plt.axes([0.05, 0.4, 0.2, 0.4])
    radio_kernel = RadioButtons(ax_radio, list(kernels.keys()))

    ax_slider_pad = plt.axes([0.35, 0.15, 0.5, 0.03])
    ax_slider_stride = plt.axes([0.35, 0.08, 0.5, 0.03])

    s_pad = Slider(ax_slider_pad, 'Padding', 0, 5, valinit=1, valstep=1)
    s_stride = Slider(ax_slider_stride, 'Stride', 1, 5, valinit=1, valstep=1)

    # Frissítő függvény
    def update(val=None):
        kernel_name = radio_kernel.value_selected
        pad = int(s_pad.val)
        stride = int(s_stride.val)

        kernel = kernels[kernel_name]
        k_t = torch.tensor(kernel).unsqueeze(0).unsqueeze(0)

        result_t = F.conv2d(img_t, k_t, padding=pad, stride=stride)
        result = result_t.squeeze().numpy()

        ax_kernel.clear()
        ax_kernel.imshow(kernel, cmap="coolwarm")
        ax_kernel.set_title("Kernel Mátrix", fontweight="bold")
        ax_kernel.axis("off")
        for i in range(kernel.shape[0]):
            for j in range(kernel.shape[1]):
                val = kernel[i, j]
                ax_kernel.text(j, i, f"{val:.2g}", ha="center", va="center", 
                               color="black" if abs(val) < max(0.5, np.max(np.abs(kernel))*0.5) else "white", fontsize=10)

        ax_conv.clear()
        ax_conv.imshow(result, cmap="gray")
        h_out, w_out = result.shape

        # Képlet megjelenítése
        in_size = img.shape[0]
        k_size = kernel.shape[0]
        formula = f"out = ⌊({in_size} + 2×{pad} - {k_size}) / {stride}⌋ + 1 = {h_out}"

        ax_conv.set_title(f"Konvolúció ({kernel_name})\nKimenet: {h_out}×{w_out}\n{formula}",
                          fontweight="bold", fontsize=11)
        ax_conv.axis("off")

        fig.canvas.draw_idle()

    radio_kernel.on_clicked(update)
    s_pad.on_changed(update)
    s_stride.on_changed(update)

    # Kezdeti kirajzolás
    update()

    plt.show()

if __name__ == "__main__":
    main()
