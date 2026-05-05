"""Shared helpers for Lec13 interactive demos."""

from pathlib import Path


def setup_matplotlib(smoke: bool = False):
    import matplotlib

    if smoke:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

    return plt, Slider, Button, CheckButtons, RadioButtons, TextBox


def default_output_path(script_file: str, suffix: str = "_smoke.png") -> Path:
    path = Path(script_file).resolve()
    return path.with_name(path.stem + suffix)


def save_or_show(fig, plt, smoke: bool, output_path=None):
    if smoke:
        if output_path is not None:
            fig.savefig(output_path, dpi=140, bbox_inches="tight")
            print(f"Saved: {Path(output_path).name}")
        plt.close(fig)
    else:
        plt.show()

