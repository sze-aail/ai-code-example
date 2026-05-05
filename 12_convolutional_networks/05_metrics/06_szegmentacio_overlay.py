"""
06_szegmentacio_overlay.py
Mask overlay visualization with IoU and Dice.
"""
import argparse
import numpy as np
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def iou_dice(gt_mask, pred_mask):
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    iou = inter / union if union else 1.0
    dice = 2 * inter / (gt_mask.sum() + pred_mask.sum() + 1e-9)
    return float(iou), float(dice)
def overlay(img, mask, color, alpha=0.4):
    out = img.copy().astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)
def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation overlay demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def make_toy_data():
    h, w = 160, 220
    image = np.zeros((h, w, 3), dtype=np.uint8)
    image[..., 0] = 45
    image[..., 1] = 55
    image[..., 2] = 70
    yy, xx = np.mgrid[:h, :w]
    gt_mask = ((xx - 90) ** 2 / (40 ** 2) + (yy - 80) ** 2 / (30 ** 2)) < 1.0
    pred_mask = ((xx - 105) ** 2 / (43 ** 2) + (yy - 86) ** 2 / (32 ** 2)) < 1.0
    return image, gt_mask, pred_mask
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    image, gt_mask, pred_mask = make_toy_data()
    iou, dice = iou_dice(gt_mask, pred_mask)
    inter = gt_mask & pred_mask
    diff_gt = gt_mask & ~pred_mask
    diff_pred = pred_mask & ~gt_mask
    vis = overlay(image, inter, (200, 180, 50))
    vis = overlay(vis, diff_gt, (90, 143, 61))
    vis = overlay(vis, diff_pred, (184, 80, 66))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(overlay(image, gt_mask, (90, 143, 61)))
    axes[1].set_title("GT overlay")
    axes[2].imshow(vis)
    axes[2].set_title(f"Pred vs GT | IoU={iou:.3f}, Dice={dice:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
if __name__ == "__main__":
    main()
