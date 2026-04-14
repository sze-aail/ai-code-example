"""
kohonen_som_interaktiv.py — Kohonen önszervező térkép (SOM) hálózat interaktív tanulása
Neurális hálók I. — Hajdu Csaba

Interaktív vizualizáció: a felügyeletlen tanulás lépéseit (epochonként) mutatja be.
A "Start/Stop" gombbal elindítható és megállítható az animáció, a csúszkákkal
pedig állíthatók a tanulási ráta és a topológiai szignál.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider, RadioButtons
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.preprocessing import StandardScaler

class InteraktioSOM:
    def __init__(self, grid_h, grid_w, n_features, random_state=42):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)
        self.grid_coords = np.array([[i, j] for i in range(grid_h) for j in range(grid_w)])
        self.weights = None

    def init_weights(self, X):
        self.weights = np.zeros((self.grid_h, self.grid_w, self.n_features))
        for d in range(self.n_features):
            self.weights[:, :, d] = self.rng.uniform(
                X[:, d].min(), X[:, d].max(), (self.grid_h, self.grid_w))

    def _find_bmu(self, x):
        diff = self.weights - x
        distances = np.sum(diff ** 2, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.grid_h, self.grid_w))
        return bmu_idx

    def _neighborhood(self, bmu_idx, sigma):
        bmu_coord = np.array(bmu_idx)
        grid_distances = np.sum((self.grid_coords - bmu_coord) ** 2, axis=1)
        grid_distances = grid_distances.reshape(self.grid_h, self.grid_w)
        return np.exp(-grid_distances / (2 * max(sigma, 0.01) ** 2))

    def predict_bmus(self, X):
        w_flat = self.weights.reshape(-1, self.n_features)
        diff = X[:, np.newaxis, :] - w_flat[np.newaxis, :, :]
        distances = np.sum(diff ** 2, axis=-1)
        return np.argmin(distances, axis=1)

    def step(self, X, eta, sigma):
        n = X.shape[0]
        perm = self.rng.permutation(n)
        for idx in perm:
            x = X[idx]
            bmu = self._find_bmu(x)
            h = self._neighborhood(bmu, sigma)
            for d in range(self.n_features):
                self.weights[:, :, d] += eta * h * (x[d] - self.weights[:, :, d])

def get_data(name):
    np.random.seed(42)
    if name == 'Blobs':
        X, y = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
    elif name == 'Moons':
        X, y = make_moons(n_samples=500, noise=0.08, random_state=42)
    elif name == 'Circles':
        X, y = make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=42)
    else:
        # Spiral
        n_per_class = 250
        theta1 = np.linspace(0, 4 * np.pi, n_per_class) + np.random.randn(n_per_class) * 0.3
        r1 = np.linspace(0.2, 2, n_per_class)
        X = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]
        y = np.zeros(n_per_class)

    return StandardScaler().fit_transform(X).astype(np.float32), y

def main():
    fig = plt.figure(figsize=(12, 8))
    ax_plot = plt.axes([0.35, 0.1, 0.6, 0.8])

    # Vezérlők
    ax_radio_data = plt.axes([0.05, 0.65, 0.20, 0.20])
    radio_data = RadioButtons(ax_radio_data, ['Blobs', 'Moons', 'Circles', 'Spirál'])

    ax_slider_eta = plt.axes([0.05, 0.45, 0.20, 0.03])
    ax_slider_sigma = plt.axes([0.05, 0.35, 0.20, 0.03])
    s_eta = Slider(ax_slider_eta, 'Tanulási ráta (η)', 0.01, 1.0, valinit=0.5)
    s_sigma = Slider(ax_slider_sigma, 'Szomszédság (σ)', 0.1, 5.0, valinit=3.0)

    ax_btn_reset = plt.axes([0.05, 0.20, 0.09, 0.06])
    ax_btn_play = plt.axes([0.16, 0.20, 0.09, 0.06])
    btn_reset = Button(ax_btn_reset, 'Reset', color='lightcoral')
    btn_play = Button(ax_btn_play, 'Start/Stop', color='lightgreen')

    som = InteraktioSOM(10, 10, 2)
    is_playing = False
    epoch = 0

    # Adatok
    X, y = get_data('Blobs')
    som.init_weights(X)

    def draw_som():
        ax_plot.clear()

        # Adatpontok klaszterezése a legújabb térkép alapján
        bmu_indices = som.predict_bmus(X)
        ax_plot.scatter(X[:, 0], X[:, 1], c=bmu_indices, cmap="tab20", s=15, alpha=0.4, zorder=1)

        # SOM rács
        w = som.weights
        for i in range(w.shape[0]):
            ax_plot.plot(w[i, :, 0], w[i, :, 1], "k-", linewidth=1.5, alpha=0.6, zorder=2)
        for j in range(w.shape[1]):
            ax_plot.plot(w[:, j, 0], w[:, j, 1], "k-", linewidth=1.5, alpha=0.6, zorder=2)

        ax_plot.scatter(w[:, :, 0].ravel(), w[:, :, 1].ravel(),
                   c="gold", s=40, edgecolors="k", linewidth=1, zorder=3)

        ax_plot.set_title(f"SOM önszerveződés | Epoch: {epoch}", fontsize=14, fontweight='bold')
        ax_plot.grid(True, alpha=0.2)
        fig.canvas.draw_idle()

    def update_data(label):
        nonlocal X, y, epoch, is_playing
        is_playing = False
        X, y = get_data(label)
        som.init_weights(X)
        epoch = 0
        draw_som()

    def reset_som(event):
        nonlocal epoch, is_playing
        is_playing = False
        som.init_weights(X)
        epoch = 0
        draw_som()

    def toggle_play(event):
        nonlocal is_playing
        is_playing = not is_playing
        if is_playing:
            run_animation()

    def run_animation():
        nonlocal epoch
        while is_playing and plt.fignum_exists(fig.number):
            som.step(X, s_eta.val, s_sigma.val)
            epoch += 1
            draw_som()
            plt.pause(0.05)

    radio_data.on_clicked(update_data)
    btn_reset.on_clicked(reset_som)
    btn_play.on_clicked(toggle_play)

    draw_som()
    plt.show()

if __name__ == "__main__":
    main()
