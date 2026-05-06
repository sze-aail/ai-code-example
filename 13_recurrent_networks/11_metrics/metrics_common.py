"""Shared helpers for Lec13 metrics scripts."""

from pathlib import Path


def setup_matplotlib(smoke: bool = False):
    import matplotlib

    if smoke:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    return plt


def default_output_path(script_file: str, suffix: str = "_output.png") -> Path:
    path = Path(script_file).resolve()
    return path.with_name(path.stem + suffix)


def save_or_show(fig, plt, smoke: bool, output_path):
    if smoke:
        fig.savefig(output_path, dpi=140, bbox_inches="tight")
        print(f"Saved: {Path(output_path).name}")
        plt.close(fig)
    else:
        plt.show()


def missing_dep(name: str):
    print(f"Missing optional dependency: {name}")

