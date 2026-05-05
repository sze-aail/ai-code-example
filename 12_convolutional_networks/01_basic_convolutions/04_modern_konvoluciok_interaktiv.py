"""
04_modern_konvoluciok_interaktiv.py — Modern konvolúciós típusok interaktív vizualizációja
Neurális hálók II. (CNN) — Hajdu Csaba

Interaktív vizualizáció, ahol a felhasználó megtapasztalhatja:
  1. A Depthwise Separable Convolution (MobileNet) paraméterszám-megtakarítását
  2. A Dilated (Atrous) Convolution effektív receptív mezejének (látóterének) növekedését
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.35, bottom=0.4)

    # ─── GUI elemek ───
    ax_radio = plt.axes([0.05, 0.6, 0.22, 0.2])
    radio = RadioButtons(ax_radio, ['Depthwise Separable', 'Dilated (Atrous)'])

    # Paraméterek a Depthwise Separable-höz
    ax_slider_in = plt.axes([0.1, 0.3, 0.8, 0.03])
    ax_slider_out = plt.axes([0.1, 0.2, 0.8, 0.03])
    ax_slider_k = plt.axes([0.1, 0.1, 0.8, 0.03])

    s_in = Slider(ax_slider_in, 'In Channels', 1, 512, valinit=64, valstep=1)
    s_out = Slider(ax_slider_out, 'Out Channels', 1, 512, valinit=128, valstep=1)
    s_k = Slider(ax_slider_k, 'Kernel/Dilation', 1, 10, valinit=3, valstep=1)

    def set_slider_state(slider, active):
        slider.set_active(active)
        alpha = 1.0 if active else 0.3
        slider.valtext.set_alpha(alpha)
        slider.label.set_alpha(alpha)
        slider.poly.set_alpha(alpha)
        slider.vline.set_alpha(alpha)

    def update(val=None):
        mode = radio.value_selected
        ax.clear()

        c_in = int(s_in.val)
        c_out = int(s_out.val)

        if mode == 'Depthwise Separable':
            s_k.label.set_text('Kernel Size (K)')
            set_slider_state(s_in, True)
            set_slider_state(s_out, True)

            k = int(s_k.val)

            # Standard conv params: C_in * C_out * K * K
            p_std = c_in * c_out * k * k
            # Depthwise Separable params: (C_in * 1 * K * K) + (C_in * C_out * 1 * 1)
            p_dws = (c_in * k * k) + (c_in * c_out)

            bars = ax.bar(["Standard Conv", "Depthwise Separable"], [p_std, p_dws],
                          color=["#E91E63", "#4CAF50"], edgecolor="k")
            ax.set_ylabel("Tanulható paraméterek száma")
            ax.set_title(f"Paraméter-megtakarítás (K={k}, C_in={c_in}, C_out={c_out})\nMegtakarítás: {(1 - p_dws/p_std)*100:.1f}%" if p_std > 0 else "Paraméter-megtakarítás")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_ylim(0, max(p_std, p_dws) * 1.15)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold')

        else:
            # Dilated Convolution mód
            s_k.label.set_text('Dilation Rate (d)')
            # Inaktívvá tesszük a csatorna-csúszkákat vizuálisan
            set_slider_state(s_in, False)
            set_slider_state(s_out, False)

            d = int(s_k.val)
            k = 3 # Rögzített 3x3 kernel ehhez a vizualizációhoz

            eff_k = k + (k - 1) * (d - 1)

            # Rajzoljuk fel a mátrixot rácsként
            grid_size = eff_k + 2
            grid = np.zeros((grid_size, grid_size))

            center = grid_size // 2
            start = center - (k//2)*d

            for i in range(k):
                for j in range(k):
                    grid[start + i*d, start + j*d] = 1

            ax.imshow(grid, cmap="Blues", vmin=0, vmax=1)

            # Formázás
            ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", size=0)
            ax.set_xticks([]); ax.set_yticks([])

            ax.set_title(f"Dilated Convolution (K=3×3, dilation={d})\nEffektív Receptive Field: {eff_k}×{eff_k}", fontweight='bold')

        fig.canvas.draw_idle()

    radio.on_clicked(update)
    s_in.on_changed(update)
    s_out.on_changed(update)
    s_k.on_changed(update)

    # Kezdeti kirajzolás
    update()

    plt.show()


if __name__ == "__main__":
    main()
