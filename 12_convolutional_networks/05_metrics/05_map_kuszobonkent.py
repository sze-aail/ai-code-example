"""
05_map_kuszobonkent.py
Educational mAP threshold sweep visualization.
"""
import argparse
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
    parser = argparse.ArgumentParser(description="mAP threshold sweep demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def toy_pr_at_iou(iou_threshold: float):
    gt = np.array([[30, 30, 90, 100], [130, 50, 210, 150], [230, 120, 310, 210]], dtype=float)
    preds = np.array([[32, 35, 89, 99], [120, 40, 214, 160], [240, 128, 300, 205], [10, 10, 60, 60]], dtype=float)
    scores = np.array([0.95, 0.8, 0.7, 0.4])
    order = np.argsort(scores)[::-1]
    preds = preds[order]
    matched = np.zeros(len(gt), dtype=bool)
    tp, fp = [], []
    for pred in preds:
        best_j = -1
        best_iou = 0.0
        for j, target in enumerate(gt):
            if matched[j]:
                continue
            val = iou(pred, target)
            if val > best_iou:
                best_iou = val
                best_j = j
        if best_iou >= iou_threshold and best_j >= 0:
            matched[best_j] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / len(gt)
    precision = tp / np.maximum(tp + fp, 1)
    ap = np.trapezoid(precision, recall)
    return recall, precision, float(ap)
def torchmetrics_map_if_available():
    try:
        import torch
        from torchmetrics.detection import MeanAveragePrecision
        metric = MeanAveragePrecision(iou_thresholds=[0.5, 0.75, 0.95], class_metrics=True)
        preds = [
            {
                "boxes": torch.tensor([[32.0, 35.0, 89.0, 99.0], [120.0, 40.0, 214.0, 160.0], [10.0, 10.0, 60.0, 60.0]]),
                "scores": torch.tensor([0.95, 0.8, 0.4]),
                "labels": torch.tensor([0, 0, 0]),
            }
        ]
        targets = [{"boxes": torch.tensor([[30.0, 30.0, 90.0, 100.0], [130.0, 50.0, 210.0, 150.0]]), "labels": torch.tensor([0, 0])}]
        metric.update(preds, targets)
        out = metric.compute()
        print(f"torchmetrics map@[0.5:0.95]: {float(out['map']):.3f}")
        print(f"torchmetrics map_50: {float(out['map_50']):.3f}")
    except Exception as exc:
        print(f"torchmetrics not available or failed: {exc}")
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    iou_thresholds = [0.5, 0.6, 0.75, 0.95]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    aps = []
    for thr in iou_thresholds:
        rec, prec, ap = toy_pr_at_iou(thr)
        aps.append(ap)
        axes[0].plot(rec, prec, marker="o", label=f"IoU={thr:.2f}, AP={ap:.3f}")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].set_title("PR curves at different IoU thresholds")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    class_ap = {"person": 0.62, "car": 0.78, "bike": 0.41, "dog": 0.58}
    axes[1].bar(class_ap.keys(), class_ap.values(), color=["#1E2761", "#5A8F3D", "#B85042", "#D4A574"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-class AP (toy)")
    axes[1].set_ylabel("AP")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
    print(f"Toy mAP over thresholds: {np.mean(aps):.3f}")
    torchmetrics_map_if_available()
if __name__ == "__main__":
    main()
