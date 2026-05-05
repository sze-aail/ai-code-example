"""Interactive scalar LSTM cell demo."""

import argparse
import numpy as np

from interactive_common import default_output_path, save_or_show, setup_matplotlib


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def lstm_scalar(x_t, h_prev, c_prev, forget_bias):
    f = sigmoid(1.2 * h_prev + 0.8 * x_t + forget_bias)
    i = sigmoid(-0.4 * h_prev + 1.0 * x_t)
    c_tilde = np.tanh(0.7 * x_t + 0.5 * h_prev)
    o = sigmoid(0.9 * h_prev + 0.6 * x_t)
    c_new = f * c_prev + i * c_tilde
    h_new = o * np.tanh(c_new)
    return {"f": f, "i": i, "c_tilde": c_tilde, "o": o, "c_new": c_new, "h_new": h_new}


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive scalar LSTM cell")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    plt, Slider, _, _, _, _ = setup_matplotlib(args.smoke)

    fig = plt.figure(figsize=(10, 5))
    ax_bar = fig.add_axes([0.08, 0.28, 0.84, 0.64])
    ax_x = fig.add_axes([0.12, 0.16, 0.25, 0.04])
    ax_h = fig.add_axes([0.12, 0.10, 0.25, 0.04])
    ax_c = fig.add_axes([0.58, 0.16, 0.25, 0.04])
    ax_fb = fig.add_axes([0.58, 0.10, 0.25, 0.04])

    s_x = Slider(ax_x, "x_t", -2.0, 2.0, valinit=0.5)
    s_h = Slider(ax_h, "h_{t-1}", -2.0, 2.0, valinit=0.0)
    s_c = Slider(ax_c, "c_{t-1}", -2.0, 2.0, valinit=0.5)
    s_fb = Slider(ax_fb, "forget bias", -1.0, 2.0, valinit=1.0)

    labels = ["f", "i", "c_tilde", "o", "c_new", "h_new"]

    def render(_=None):
        vals = lstm_scalar(s_x.val, s_h.val, s_c.val, s_fb.val)
        numbers = [vals[k] for k in labels]
        colors = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:red", "tab:brown"]

        ax_bar.clear()
        ax_bar.bar(labels, numbers, color=colors)
        ax_bar.axhline(0.0, color="black", linewidth=1)
        ax_bar.set_ylim(-1.2, 1.2)
        ax_bar.set_title("LSTM gates and updated states")
        for idx, value in enumerate(numbers):
            ax_bar.text(idx, value + (0.05 if value >= 0 else -0.1), f"{value:.2f}", ha="center")
        ax_bar.grid(axis="y", alpha=0.25)
        fig.canvas.draw_idle()

    for slider in (s_x, s_h, s_c, s_fb):
        slider.on_changed(render)
    render()

    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()

