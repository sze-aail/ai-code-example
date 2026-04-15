"""
03_pooling_interaktiv.py — Pooling paraméterek interaktív vizualizációja
Neurális hálók II. (CNN) — Hajdu Csaba

Interaktív vizualizáció: a felhasználó csúszkák és gombok segítségével
kipróbálhatja a MaxPool és az AvgPool működését, miközben módosítja
a kernel méretet, a stride-ot és a paddinget. Látható, hogyan csökken
vagy változik a bemeneti kép (egy számjegy) felbontása.
"""
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from sklearn.datasets import load_digits

def main():
    digits = load_digits()
    img = digits.data[0].reshape(8, 8).astype(np.float32) / 16.0
    img_big = np.kron(img, np.ones((8, 8)))  # 64x64-re felnagyítjuk, hogy jobban látszódjon a hatás

    img_t = torch.tensor(img_big).unsqueeze(0).unsqueeze(0)

    fig, (ax_orig, ax_pool) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(left=0.3, bottom=0.3)

    ax_orig.imshow(img_big, cmap="gray_r")
    ax_orig.set_title(f"Bemeneti kép ({img_big.shape[0]}×{img_big.shape[1]})", fontweight="bold")
    ax_orig.axis("off")

    ax_pool.axis("off")

    # GUI elemek
    ax_radio = plt.axes([0.05, 0.4, 0.2, 0.2])
    radio_pool = RadioButtons(ax_radio, ["MaxPool", "AvgPool"])

    ax_slider_k = plt.axes([0.35, 0.20, 0.5, 0.03])
    ax_slider_s = plt.axes([0.35, 0.13, 0.5, 0.03])
    ax_slider_p = plt.axes([0.35, 0.06, 0.5, 0.03])

    s_kernel = Slider(ax_slider_k, 'Kernel méret', 2, 16, valinit=2, valstep=1)
    s_stride = Slider(ax_slider_s, 'Stride', 1, 16, valinit=2, valstep=1)
    s_pad = Slider(ax_slider_p, 'Padding', 0, 8, valinit=0, valstep=1)

    def update(val=None):
        pool_type = radio_pool.value_selected
        k = int(s_kernel.val)
        s = int(s_stride.val)
        p = int(s_pad.val)

        if pool_type == "MaxPool":
            result_t = F.max_pool2d(img_t, kernel_size=k, stride=s, padding=p)
        else:
            # AvgPool a paddingkel együtt kiszámolva
            result_t = F.avg_pool2d(img_t, kernel_size=k, stride=s, padding=p)

        result = result_t.squeeze().numpy()

        ax_pool.clear()
        ax_pool.imshow(result, cmap="gray_r")

        h_in = img_big.shape[0]
        h_out, w_out = result.shape

        formula = f"out = ⌊({h_in} + 2×{p} - {k}) / {s}⌋ + 1 = {h_out}"
        ax_pool.set_title(f"{pool_type}\nKimenet: {h_out}×{w_out}\n{formula}",
                          fontweight="bold", fontsize=11)
        ax_pool.axis("off")

        fig.canvas.draw_idle()

    radio_pool.on_clicked(update)
    s_kernel.on_changed(update)
    s_stride.on_changed(update)
    s_pad.on_changed(update)

    update()

    plt.show()

if __name__ == "__main__":
    main()
