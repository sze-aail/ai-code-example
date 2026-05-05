"""
Interactive BPTT demo for a tiny vanilla RNN.

- Toggle input bits
- Adjust recurrent weight scale
- Inspect hidden-state and gradient-flow over time
"""

import argparse
import numpy as np

from interactive_common import default_output_path, save_or_show, setup_matplotlib


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_params(hidden_size: int, scale: float, seed: int):
    rng = np.random.default_rng(seed)
    w_xh = rng.normal(0, 0.8, size=(hidden_size, 1))
    w_hh = rng.normal(0, scale, size=(hidden_size, hidden_size))
    b_h = np.zeros(hidden_size)
    w_hy = rng.normal(0, 0.6, size=(1, hidden_size))
    b_y = np.zeros(1)
    return w_xh, w_hh, b_h, w_hy, b_y


def forward(xs, params):
    w_xh, w_hh, b_h, w_hy, b_y = params
    hs = [np.zeros(w_hh.shape[0])]
    for x_t in xs:
        h_t = np.tanh(w_xh[:, 0] * x_t + w_hh @ hs[-1] + b_h)
        hs.append(h_t)
    y_hat = sigmoid(w_hy @ hs[-1] + b_y)
    return float(y_hat.item()), hs


def backward_grad_norms(xs, y_true, params):
    w_xh, w_hh, b_h, w_hy, b_y = params
    y_hat, hs = forward(xs, params)
    d_h = (y_hat - y_true) * w_hy.reshape(-1)
    norms = []
    for t in range(len(xs), 0, -1):
        dtanh = (1.0 - hs[t] * hs[t]) * d_h
        norms.append(float(np.linalg.norm(dtanh)))
        d_h = w_hh.T @ dtanh
    norms.reverse()
    return y_hat, hs, norms


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive BPTT parity demo")
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    plt, Slider, Button, CheckButtons, _, _ = setup_matplotlib(args.smoke)

    xs = np.array([1, 0, 1, 0, 1][: args.seq_len], dtype=float)
    scale = 0.45
    params = make_params(args.hidden_size, scale, args.seed)

    fig = plt.figure(figsize=(11, 6))
    ax_hidden = fig.add_axes([0.08, 0.33, 0.38, 0.56])
    ax_grad = fig.add_axes([0.56, 0.33, 0.36, 0.56])
    ax_bits = fig.add_axes([0.08, 0.08, 0.18, 0.18])
    ax_scale = fig.add_axes([0.34, 0.14, 0.22, 0.04])
    ax_seed = fig.add_axes([0.61, 0.08, 0.15, 0.08])

    check = CheckButtons(ax_bits, [f"x{idx}" for idx in range(1, args.seq_len + 1)], [bool(v) for v in xs])
    scale_slider = Slider(ax_scale, "W_hh scale", 0.05, 1.4, valinit=scale, valstep=0.05)
    seed_button = Button(ax_seed, "New weights")

    def render(_=None):
        nonlocal params, xs
        scale_now = float(scale_slider.val)
        params = make_params(args.hidden_size, scale_now, args.seed)
        y_true = int(xs.sum() % 2)
        y_hat, hs, norms = backward_grad_norms(xs, y_true, params)
        hs_arr = np.stack(hs[1:], axis=0)

        ax_hidden.clear()
        ax_hidden.imshow(hs_arr.T, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax_hidden.set_title(f"Hidden states over time | target parity={y_true}, pred={int(y_hat >= 0.5)}")
        ax_hidden.set_xlabel("time step")
        ax_hidden.set_ylabel("hidden unit")

        ax_grad.clear()
        ax_grad.plot(range(1, len(norms) + 1), norms, "o-", color="tab:red", label="||dL/dh_t||")
        ax_grad.set_title(f"Gradient flow backward in time | y_hat={y_hat:.3f}")
        ax_grad.set_xlabel("time step")
        ax_grad.set_ylabel("gradient norm")
        ax_grad.grid(alpha=0.25)
        ax_grad.legend(loc="upper right")
        fig.canvas.draw_idle()

    def on_toggle(label):
        idx = int(label[1:]) - 1
        xs[idx] = 1.0 - xs[idx]
        render()

    def on_seed(_event):
        args.seed += 1
        render()

    check.on_clicked(on_toggle)
    scale_slider.on_changed(render)
    seed_button.on_clicked(on_seed)
    render()

    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()

