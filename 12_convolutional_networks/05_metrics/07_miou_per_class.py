"""
07_miou_per_class.py
Per-class IoU bar chart with mean IoU marker.
"""
import argparse
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="mIoU per-class bar chart")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    ious = {
        "ut": 0.95,
        "jarda": 0.71,
        "auto": 0.83,
        "gyalogos": 0.46,
        "biciklista": 0.31,
        "fa": 0.88,
        "eg": 0.97,
    }
    classes = list(ious.keys())
    values = list(ious.values())
    miou = sum(values) / len(values)
    colors = ["#B85042" if v < 0.5 else "#5A8F3D" for v in values]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.barh(classes, values, color=colors)
    ax.axvline(miou, ls="--", color="#1E2761", label=f"mIoU={miou:.3f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("IoU")
    ax.set_title("Per-class IoU")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
if __name__ == "__main__":
    main()
