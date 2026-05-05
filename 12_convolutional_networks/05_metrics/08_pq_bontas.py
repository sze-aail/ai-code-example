"""
08_pq_bontas.py
Panoptic Quality decomposition: SQ, RQ, PQ by class.
"""
import argparse
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="PQ decomposition demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    classes = ["ut", "jarda", "auto(things)", "gyalogos(things)", "eg"]
    sq = [0.92, 0.78, 0.81, 0.65, 0.95]
    rq = [1.00, 0.85, 0.74, 0.60, 1.00]
    pq = [a * b for a, b in zip(sq, rq)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    x = range(len(classes))
    width = 0.26
    axes[0].bar([i - width for i in x], sq, width=width, label="SQ", color="#5A8F3D")
    axes[0].bar(x, rq, width=width, label="RQ", color="#D4A574")
    axes[0].bar([i + width for i in x], pq, width=width, label="PQ", color="#1E2761")
    axes[0].set_xticks(list(x), classes, rotation=18)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("PQ decomposition by class")
    axes[0].set_ylabel("score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    things_pq = sum(v for c, v in zip(classes, pq) if "things" in c) / 2
    stuff_pq = sum(v for c, v in zip(classes, pq) if "things" not in c) / 3
    axes[1].pie(
        [stuff_pq, things_pq],
        labels=[f"stuff PQ={stuff_pq:.2f}", f"things PQ={things_pq:.2f}"],
        colors=["#5A8F3D", "#D4A574"],
        startangle=90,
        wedgeprops={"width": 0.45},
    )
    axes[1].set_title("Stuff vs Things PQ")
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
if __name__ == "__main__":
    main()
