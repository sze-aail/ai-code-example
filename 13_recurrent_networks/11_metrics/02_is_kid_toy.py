"""Toy IS and KID examples via torchmetrics."""
import argparse
import torch
from metrics_common import missing_dep
def parse_args():
    parser = argparse.ArgumentParser(description="IS/KID toy demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def main():
    _ = parse_args()

    try:
        from torchmetrics.image.inception import InceptionScore
        from torchmetrics.image.kid import KernelInceptionDistance
    except Exception:
        missing_dep("torchmetrics")
        return

    imgs_fake = torch.randint(0, 256, (24, 3, 299, 299), dtype=torch.uint8)
    imgs_real = torch.randint(0, 256, (24, 3, 299, 299), dtype=torch.uint8)

    try:
        inception = InceptionScore()
        inception.update(imgs_fake)
        is_mean, is_std = inception.compute()
        print(f"IS = {float(is_mean):.2f} +/- {float(is_std):.2f}")
    except Exception as exc:
        print(f"InceptionScore unavailable: {exc}")

    try:
        kid = KernelInceptionDistance(subset_size=12)
        kid.update(imgs_real, real=True)
        kid.update(imgs_fake, real=False)
        kid_mean, kid_std = kid.compute()
        print(f"KID = {float(kid_mean):.4f} +/- {float(kid_std):.4f}")
    except Exception as exc:
        print(f"KID unavailable: {exc}")
if __name__ == "__main__":
    main()
