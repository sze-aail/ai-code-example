"""
09_interaktiv_kuszob.py
Interactive threshold sweep (Plotly) with fallback static matplotlib chart.
"""
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from metrics_common import default_output_path, save_or_show, setup_matplotlib
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive threshold demo")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()
def make_data(seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = rng.choice([0, 1], size=300, p=[0.82, 0.18])
    y_score = np.clip(rng.normal(0.25 + 0.55 * y_true, 0.24, size=300), 0, 1)
    return y_true, y_score
def run_plotly(y_true, y_score, smoke: bool):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    thresholds = np.linspace(0, 1, 21)
    frames = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        frames.append(
            go.Frame(
                data=[go.Heatmap(z=cm, text=cm, texttemplate="%{text}", colorscale="Blues")],
                name=f"{t:.2f}",
            )
        )
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(frames[0].data[0])
    fig.frames = frames
    fig.update_layout(
        title="Confusion matrix vs threshold",
        sliders=[dict(steps=[dict(method="animate", args=[[f.name]], label=f.name) for f in frames])],
    )
    if smoke:
        out_html = default_output_path(__file__, suffix="_output.html")
        fig.write_html(str(out_html))
        print(f"Saved: {out_html.name}")
    else:
        fig.show()
def run_matplotlib_fallback(y_true, y_score, smoke: bool):
    plt = setup_matplotlib(smoke)
    thresholds = np.linspace(0.0, 1.0, 41)
    prec, rec, f1 = [], [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        prec.append(precision_score(y_true, y_pred, zero_division=0))
        rec.append(recall_score(y_true, y_pred, zero_division=0))
        f1.append(f1_score(y_true, y_pred, zero_division=0))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(thresholds, prec, label="precision", color="#1E2761")
    ax.plot(thresholds, rec, label="recall", color="#5A8F3D")
    ax.plot(thresholds, f1, label="f1", color="#D4A574")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title("Threshold tuning curves (fallback)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, plt, smoke, default_output_path(__file__))
def main():
    args = parse_args()
    y_true, y_score = make_data()
    try:
        run_plotly(y_true, y_score, args.smoke)
    except Exception as exc:
        print(f"Plotly unavailable, using matplotlib fallback: {exc}")
        run_matplotlib_fallback(y_true, y_score, args.smoke)
if __name__ == "__main__":
    main()
