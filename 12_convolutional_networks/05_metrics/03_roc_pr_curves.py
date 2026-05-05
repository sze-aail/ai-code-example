"""
03_roc_pr_curves.py
ROC and Precision-Recall curves for model comparison.
"""
import argparse
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="ROC and PR curve demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def make_scores(seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 500
    y_true = rng.choice([0, 1], size=n, p=[0.85, 0.15])
    score_a = np.clip(rng.normal(0.25 + 0.5 * y_true, 0.25, size=n), 0, 1)
    score_b = np.clip(rng.normal(0.2 + 0.65 * y_true, 0.22, size=n), 0, 1)
    return y_true, score_a, score_b
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    y_true, score_a, score_b = make_scores()
    fpr_a, tpr_a, _ = roc_curve(y_true, score_a)
    fpr_b, tpr_b, _ = roc_curve(y_true, score_b)
    prec_a, rec_a, _ = precision_recall_curve(y_true, score_a)
    prec_b, rec_b, _ = precision_recall_curve(y_true, score_b)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(fpr_a, tpr_a, lw=2, label=f"Model A AUC={auc(fpr_a, tpr_a):.3f}", color="#1E2761")
    axes[0].plot(fpr_b, tpr_b, lw=2, label=f"Model B AUC={auc(fpr_b, tpr_b):.3f}", color="#D4A574")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(rec_a, prec_a, lw=2, label=f"Model A AP~{auc(rec_a, prec_a):.3f}", color="#1E2761")
    axes[1].plot(rec_b, prec_b, lw=2, label=f"Model B AP~{auc(rec_b, prec_b):.3f}", color="#D4A574")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
if __name__ == "__main__":
    main()
