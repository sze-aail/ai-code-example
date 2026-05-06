"""Toy FID demo using clean-fid when available."""
import argparse
from metrics_common import default_output_path, missing_dep, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="FID toy demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    args = parse_args()
    plt = setup_matplotlib(args.smoke)
    fid_per_epoch = [78, 63, 49, 42, 37, 34, 31]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(fid_per_epoch, color="#D4A574", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.set_title("Toy training curve: FID drops")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save_or_show(fig, plt, args.smoke, default_output_path(__file__))
    try:
        from cleanfid import fid
        print("clean-fid available. Example call:")
        print("fid.compute_fid('real_images/', 'generated_images/', mode='clean')")
        _ = fid
    except ImportError:
        missing_dep("clean-fid")
if __name__ == "__main__":
    main()
