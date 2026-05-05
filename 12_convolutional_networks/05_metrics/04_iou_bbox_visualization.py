"""
04_iou_bbox_visualization.py
Bounding-box IoU visualization with overlap area and shift sweep.
"""
import argparse
import matplotlib.patches as patches
import numpy as np
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def iou(box_a, box_b):
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union
def parse_args():
    parser = argparse.ArgumentParser(description="BBox IoU visualization demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    gt = np.array([50, 50, 200, 180], dtype=float)
    pred = np.array([110, 100, 240, 240], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].set_xlim(0, 300)
    axes[0].set_ylim(280, 0)
    axes[0].set_title(f"GT vs Pred IoU={iou(gt, pred):.3f}")
    for box, color, label in [(gt, "#5A8F3D", "GT"), (pred, "#D4A574", "Pred")]:
        rect = patches.Rectangle(
            (box[0], box[1]),
            box[2] - box[0],
            box[3] - box[1],
            linewidth=2,
            edgecolor=color,
            facecolor=color,
            alpha=0.3,
        )
        axes[0].add_patch(rect)
        axes[0].text(box[0], box[1] - 5, label, color=color, fontweight="bold")
    axes[0].grid(alpha=0.2)
    shifts = np.arange(-80, 81, 5)
    ious = []
    for dx in shifts:
        moved = pred.copy()
        moved[[0, 2]] += dx
        ious.append(iou(gt, moved))
    axes[1].plot(shifts, ious, color="#1E2761", marker="o", markersize=3)
    axes[1].set_xlabel("Pred box horizontal shift (px)")
    axes[1].set_ylabel("IoU")
    axes[1].set_title("IoU sensitivity to translation")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
if __name__ == "__main__":
    main()
