"""Interactive self-attention heatmap demo."""

import argparse
import numpy as np

from interactive_common import default_output_path, save_or_show, setup_matplotlib


Q = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
K = Q.copy()
V = np.array([[10, 0], [0, 10], [5, 5]], dtype=float)


def softmax_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix - matrix.max(axis=1, keepdims=True)
    e = np.exp(matrix)
    return e / e.sum(axis=1, keepdims=True)


def attention(q_scale: float, causal: bool):
    scores = (q_scale * Q @ K.T) / np.sqrt(Q.shape[1])
    if causal:
        mask = np.triu(np.ones_like(scores, dtype=bool), k=1)
        scores = scores.copy()
        scores[mask] = -1e9
    weights = softmax_rows(scores)
    out = weights @ V
    return weights, out


def parse_args():
    parser = argparse.ArgumentParser(description="Interactive self-attention demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    plt, Slider, _, CheckButtons, RadioButtons, _ = setup_matplotlib(args.smoke)

    fig = plt.figure(figsize=(11, 5.5))
    ax_heat = fig.add_axes([0.08, 0.25, 0.38, 0.62])
    ax_out = fig.add_axes([0.56, 0.25, 0.32, 0.62])
    ax_scale = fig.add_axes([0.08, 0.10, 0.28, 0.04])
    ax_mask = fig.add_axes([0.42, 0.06, 0.12, 0.12])
    ax_token = fig.add_axes([0.84, 0.06, 0.10, 0.14])

    s_scale = Slider(ax_scale, "Q scale", 0.2, 2.0, valinit=1.0, valstep=0.1)
    check_mask = CheckButtons(ax_mask, ["causal"], [False])
    radio = RadioButtons(ax_token, ["token 1", "token 2", "token 3"], active=0)

    def render(_=None):
        token_idx = int(radio.value_selected.split()[-1]) - 1
        causal = check_mask.get_status()[0]
        weights, out = attention(s_scale.val, causal)

        ax_heat.clear()
        ax_heat.imshow(weights, cmap="viridis", vmin=0.0, vmax=1.0)
        ax_heat.set_title("Attention matrix")
        ax_heat.set_xlabel("key token")
        ax_heat.set_ylabel("query token")
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                ax_heat.text(j, i, f"{weights[i, j]:.2f}", ha="center", va="center", color="white")

        ax_out.clear()
        ax_out.bar(["v1", "v2"], out[token_idx], color=["tab:blue", "tab:orange"])
        ax_out.set_ylim(0, max(11, out.max() + 1))
        ax_out.set_title(f"Output vector for token {token_idx + 1}")
        ax_out.grid(axis="y", alpha=0.25)
        fig.canvas.draw_idle()

    s_scale.on_changed(render)
    check_mask.on_clicked(render)
    radio.on_clicked(render)
    render()

    save_or_show(fig, plt, args.smoke, default_output_path(__file__))


if __name__ == "__main__":
    main()

