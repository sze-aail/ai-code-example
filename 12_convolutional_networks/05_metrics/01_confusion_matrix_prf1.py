"""
01_confusion_matrix_prf1.py
Confusion matrix visualization with per-cell counts and row-normalized percentages.
"""
import argparse
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="Confusion matrix + P/R/F1 demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    rng = np.random.default_rng(42)
    n = 320
    y_true = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    y_pred = np.where(y_true == 1, rng.choice([0, 1], size=n, p=[0.25, 0.75]), rng.choice([0, 1], size=n, p=[0.9, 0.1]))
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    titles = ["Confusion matrix (count)", "Confusion matrix (row-normalized)"]
    matrices = [cm, cm_norm]
    try:
        import seaborn as sns
        for ax, title, mat in zip(axes, titles, matrices):
            fmt = "d" if mat.dtype.kind in {"i", "u"} else ".2f"
            sns.heatmap(
                mat,
                annot=True,
                fmt=fmt,
                cmap="Blues",
                xticklabels=["neg", "pos"],
                yticklabels=["neg", "pos"],
                ax=ax,
            )
            ax.set_title(title)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
    except ImportError:
        for ax, title, mat in zip(axes, titles, matrices):
            im = ax.imshow(mat, cmap="Blues")
            ax.set_title(title)
            ax.set_xticks([0, 1], ["neg", "pos"])
            ax.set_yticks([0, 1], ["neg", "pos"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(2):
                for j in range(2):
                    text = f"{mat[i, j]:.2f}" if mat.dtype.kind == "f" else f"{int(mat[i, j])}"
                    ax.text(j, i, text, ha="center", va="center", color="black")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["neg", "pos"], digits=3))
if __name__ == "__main__":
    main()
