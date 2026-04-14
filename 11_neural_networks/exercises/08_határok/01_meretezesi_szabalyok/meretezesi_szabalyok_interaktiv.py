"""
meretezesi_szabalyok_interaktiv.py — MLP méretezési szabályok interaktív vizualizációja
Neurális hálók I. — Hajdu Csaba

Kiszámítja az összes méretezési ajánlást interaktívan, matplotlib Csatúszkákkal.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def sizing_rules(n_input, n_output, n_samples, n_classes=2):
    """Kiszámítja az összes méretezési ajánlást."""
    rules = {}

    # 1. Kolmogorov / Hecht-Nielsen
    rules["Kolmogorov (2n+1)"] = 2 * n_input + 1

    # 2. Hüvelykujj-szabályok
    rules["(in+out)/2"] = max(1, (n_input + n_output) // 2)
    rules["√(in×out)"] = max(1, int(np.sqrt(n_input * n_output)))
    rules["2×in"] = 2 * n_input

    # 3. Kapacitás-szabály (α=5 és α=10)
    for alpha in [5, 10]:
        n_h = max(1, n_samples // (alpha * (n_input + n_output)))
        rules[f"N/(α(in+out)), α={alpha}"] = n_h

    # 4. Osztályozási szabály
    if n_classes > 2:
        rules[f"K(K-1)/2, K={n_classes}"] = n_classes * (n_classes - 1) // 2

    return rules

def main():
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.35, bottom=0.35)

    # Kezdeti értékek
    init_in = 20
    init_out = 4
    init_samples = 2000
    init_classes = 4

    rules = sizing_rules(init_in, init_out, init_samples, init_classes)
    labels = list(rules.keys())
    values = list(rules.values())

    # Létrehozunk egy vízszintes sávdiagramot
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, align='center', color='skyblue', edgecolor='k')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # fentről lefelé
    ax.set_xlabel("Ajánlott rejtett neuronok száma")
    ax.set_title("Interaktív méretezési szabályok")

    # Logaritmikus skála hogy a nagy különbségek is látszódjanak
    # ax.set_xscale('log')
    # ax.set_xlim(1, max(values) * 1.5)

    # Szöveges értékek a sávok mellett
    texts = []
    for i, v in enumerate(values):
        texts.append(ax.text(v + 1, i, str(v), va='center', fontweight='bold'))

    # Csúszkák hozzáadása
    ax_in = plt.axes([0.15, 0.20, 0.65, 0.03])
    ax_out = plt.axes([0.15, 0.15, 0.65, 0.03])
    ax_samp = plt.axes([0.15, 0.10, 0.65, 0.03])
    ax_cls = plt.axes([0.15, 0.05, 0.65, 0.03])

    s_in = Slider(ax_in, 'Bemenet', 1, 1000, valinit=init_in, valstep=1)
    s_out = Slider(ax_out, 'Kimenet', 1, 1000, valinit=init_out, valstep=1)
    s_samp = Slider(ax_samp, 'Minták', 10, 100000, valinit=init_samples, valstep=10)
    s_cls = Slider(ax_cls, 'Osztályok', 2, 100, valinit=init_classes, valstep=1)

    def update(val):
        n_in = int(s_in.val)
        n_out = int(s_out.val)
        n_samp = int(s_samp.val)
        n_cls = int(s_cls.val)

        new_rules = sizing_rules(n_in, n_out, n_samp, n_cls)
        new_values = list(new_rules.values())

        for i, bar in enumerate(bars):
            if i < len(new_values):
                bar.set_width(new_values[i])
                texts[i].set_position((new_values[i] + max(1, new_values[i]*0.05), i))
                texts[i].set_text(str(new_values[i]))

        ax.set_xlim(0, max(new_values) * 1.2)
        fig.canvas.draw_idle()

    s_in.on_changed(update)
    s_out.on_changed(update)
    s_samp.on_changed(update)
    s_cls.on_changed(update)

    update(0)  # kezdeti frissítés a határok beállításához

    plt.show()

if __name__ == "__main__":
    main()
