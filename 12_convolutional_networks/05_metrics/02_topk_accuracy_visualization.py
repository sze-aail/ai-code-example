"""
02_topk_accuracy_visualization.py
Top-1 and Top-5 style probability bar visualization.
"""
import argparse
import numpy as np
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="Top-k accuracy visualization demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    class_names = [f"class_{i}" for i in range(10)]
    rng = np.random.default_rng(42)
    logits = rng.normal(0, 1, size=len(class_names))
    probs = softmax(logits)
    true_label = 7
    top_idx = np.argsort(probs)[-5:][::-1]
    top_vals = probs[top_idx]
    top_labels = [class_names[i] for i in top_idx]
    true_in_top5 = int(true_label) in top_idx.tolist()
    top1_ok = int(top_idx[0] == true_label)
    if true_in_top5:
        true_pos = top_idx.tolist().index(true_label)
    else:
        true_pos = -1
    colors = ["#1E2761"] * 5
    if true_in_top5:
        colors[true_pos] = "#5A8F3D"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top_labels[::-1], top_vals[::-1], color=colors[::-1])
    ax.set_xlabel("Probability")
    ax.set_title(
        f"Top-5 predictions | true={class_names[true_label]} | Top-1={'yes' if top1_ok else 'no'} | Top-5={'yes' if true_in_top5 else 'no'}"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
if __name__ == "__main__":
    main()
