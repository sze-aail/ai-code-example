"""
01_konvolucio_1d_interaktiv.py — 1D Konvolúció interaktív vizualizációja
Neurális hálók II. (CNN) — Hajdu Csaba

Interaktív vizualizáció az 1D konvolúció működésének megértéséhez.
A felhasználó kiválaszthatja a bemeneti jelet, a konvolúciós kernelt,
és a konvolúció módját (full, valid, same).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons

def get_signal(name):
    if name == "Lépésfüggvény (Step)":
        x = np.zeros(100)
        x[30:70] = 1.0
        return x
    elif name == "Zajos szinusz":
        np.random.seed(42)
        t = np.linspace(0, 10, 100)
        return np.sin(t) + np.random.normal(0, 0.2, len(t))
    elif name == "Impulzus":
        x = np.zeros(100)
        x[50] = 1.0
        return x
    return np.zeros(100)

def get_kernel(name):
    if name == "Éldetektor [1, 0, -1]":
        return np.array([1.0, 0.0, -1.0])
    elif name == "Simítás (3-as)":
        return np.ones(3) / 3.0
    elif name == "Simítás (7-es)":
        return np.ones(7) / 7.0
    elif name == "Gauss-szerű (5-ös)":
        return np.array([0.1, 0.2, 0.4, 0.2, 0.1])
    return np.array([1.0])

def main():
    fig, (ax_sig, ax_kernel, ax_conv) = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(left=0.35, hspace=0.6)

    # Radio buttons - Signal
    ax_radio_sig = plt.axes([0.05, 0.65, 0.22, 0.2])
    radio_sig = RadioButtons(ax_radio_sig, ["Lépésfüggvény (Step)", "Zajos szinusz", "Impulzus"])

    # Radio buttons - Kernel
    ax_radio_kernel = plt.axes([0.05, 0.35, 0.22, 0.25])
    radio_kernel = RadioButtons(ax_radio_kernel,
        ["Éldetektor [1, 0, -1]", "Simítás (3-as)", "Simítás (7-es)", "Gauss-szerű (5-ös)"])

    # Radio buttons - Mode
    ax_radio_mode = plt.axes([0.05, 0.1, 0.22, 0.2])
    radio_mode = RadioButtons(ax_radio_mode, ["full", "valid", "same"], active=2)

    def update(val=None):
        sig_name = radio_sig.value_selected
        kernel_name = radio_kernel.value_selected
        mode = radio_mode.value_selected

        x = get_signal(sig_name)
        h = get_kernel(kernel_name)
        y = np.convolve(x, h, mode=mode)

        ax_sig.clear()
        ax_sig.stem(range(len(x)), x, linefmt="b-", markerfmt="bo", basefmt="k-")
        ax_sig.set_title(f"Bemeneti jel: {sig_name} (Hossz: {len(x)})")
        ax_sig.grid(True, alpha=0.3)

        ax_kernel.clear()
        ax_kernel.stem(range(len(h)), h, linefmt="g-", markerfmt="go", basefmt="k-")
        ax_kernel.set_title(f"Kernel: {kernel_name} (Hossz: {len(h)})")
        ax_kernel.grid(True, alpha=0.3)

        ax_conv.clear()
        ax_conv.stem(range(len(y)), y, linefmt="r-", markerfmt="ro", basefmt="k-")
        ax_conv.set_title(f"Kimeneti jel (Mode: {mode}, Hossz: {len(y)})")
        ax_conv.grid(True, alpha=0.3)

        fig.canvas.draw_idle()

    radio_sig.on_clicked(update)
    radio_kernel.on_clicked(update)
    radio_mode.on_clicked(update)

    update()
    plt.show()

if __name__ == "__main__":
    main()
