"""
pegasos_es_kernel_perceptron_interaktiv.py — Interaktív Pegasos és Kernel Perceptron
Neurális hálók I. — Hajdu Csaba

Interaktív vizualizáció:
A felhasználó kiválaszthatja a modellt (Perceptron, Pegasos, Kernel Perceptron),
az adathalmazt, valamint a hiperparamétereket (Pegasos λ, RBF γ, epoch szám),
majd a "Tanítás és Frissítés" gombra kattintva megtekintheti az új döntési határt.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from sklearn.datasets import make_moons, make_circles, make_blobs

# ══════════════════════════════════════════════════════════════
# Modellek (Perceptron, Pegasos, Kernel Perceptron)
# ══════════════════════════════════════════════════════════════

class SimplePerceptron:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.w = None
        self.b = 0.0

    def fit(self, X, y, n_epochs=20):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for epoch in range(n_epochs):
            for i in np.random.permutation(n):
                if y[i] * (np.dot(self.w, X[i]) + self.b) <= 0:
                    self.w += self.lr * y[i] * X[i]
                    self.b += self.lr * y[i]
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class Pegasos:
    def __init__(self, lam=0.01):
        self.lam = lam
        self.w = None
        self.b = 0.0

    def fit(self, X, y, n_epochs=20):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        t = 1
        for epoch in range(n_epochs):
            for i in np.random.permutation(n):
                eta = 1.0 / (self.lam * t)
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                self.w *= (1 - eta * self.lam)
                if margin < 1:
                    self.w += eta * y[i] * X[i]
                    self.b += eta * y[i]
                t += 1
        return self

    def decision_function(self, X):
        return X @ self.w + self.b

    def predict(self, X):
        return np.sign(self.decision_function(X))


class KernelPerceptron:
    def __init__(self, gamma=1.0):
        self.gamma = gamma
        self.alphas = None
        self.X_train = None
        self.y_train = None

    def _kernel_fn(self, X1, X2):
        sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = sq1 + sq2 - 2 * X1 @ X2.T
        return np.exp(-self.gamma * dist_sq)

    def fit(self, X, y, n_epochs=50):
        n = X.shape[0]
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.alphas = np.zeros(n)
        K = self._kernel_fn(X, X)
        for epoch in range(n_epochs):
            errors = 0
            for i in range(n):
                decision = np.sum(self.alphas * self.y_train * K[:, i])
                if y[i] * decision <= 0:
                    self.alphas[i] += 1
                    errors += 1
            if errors == 0:
                break
        return self

    def decision_function(self, X):
        K = self._kernel_fn(self.X_train, X)
        return (self.alphas * self.y_train) @ K

    def predict(self, X):
        return np.sign(self.decision_function(X))

# ══════════════════════════════════════════════════════════════
# Adatgenerátorok
# ══════════════════════════════════════════════════════════════

def get_data(name):
    np.random.seed(42)
    if name == 'Blobs':
        X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.2, random_state=42)
        y = 2 * y - 1
    elif name == 'Moons':
        X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
        y = 2 * y - 1
    elif name == 'Circles':
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=42)
        y = 2 * y - 1
    elif name == 'XOR':
        X = np.random.uniform(-2, 2, (300, 2))
        y = np.sign(X[:, 0] * X[:, 1])
        y[y == 0] = 1
    return X, y

# ══════════════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════════════

def main():
    fig = plt.figure(figsize=(12, 7))
    ax_plot = plt.axes([0.35, 0.1, 0.6, 0.8])

    # Vezérlők
    ax_radio_data = plt.axes([0.05, 0.70, 0.20, 0.20])
    ax_radio_model = plt.axes([0.05, 0.45, 0.20, 0.20])

    ax_slider_epochs = plt.axes([0.05, 0.35, 0.20, 0.03])
    ax_slider_lam = plt.axes([0.05, 0.25, 0.20, 0.03])
    ax_slider_gamma = plt.axes([0.05, 0.15, 0.20, 0.03])

    ax_btn = plt.axes([0.05, 0.05, 0.20, 0.06])

    radio_data = RadioButtons(ax_radio_data, ['Blobs', 'Moons', 'Circles', 'XOR'])
    radio_model = RadioButtons(ax_radio_model, ['Perceptron', 'Pegasos (SVM)', 'RBF Kernel Perceptron'])

    s_epochs = Slider(ax_slider_epochs, 'Epochs', 1, 200, valinit=30, valstep=1)
    s_lam = Slider(ax_slider_lam, 'Pegasos λ', 0.0001, 0.1, valinit=0.01, valstep=0.001)
    s_gamma = Slider(ax_slider_gamma, 'RBF γ', 0.1, 20.0, valinit=2.0, valstep=0.1)

    btn_train = Button(ax_btn, 'Tanítás és Frissítés', color='lightgreen', hovercolor='palegreen')

    def update_plot(event=None):
        ax_plot.clear()

        data_name = radio_data.value_selected
        model_name = radio_model.value_selected
        epochs = int(s_epochs.val)
        lam = s_lam.val
        gamma = s_gamma.val

        X, y = get_data(data_name)

        ax_plot.set_title("Tanítás folyamatban...", color='red')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        if model_name == 'Perceptron':
            model = SimplePerceptron(lr=0.01)
        elif model_name == 'Pegasos (SVM)':
            model = Pegasos(lam=lam)
        else:
            model = KernelPerceptron(gamma=gamma)

        model.fit(X, y, n_epochs=epochs)
        preds = model.predict(X)
        acc = np.mean(preds == y)

        # Kirajzolás
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 150),
                             np.linspace(y_min, y_max, 150))
        grid = np.c_[xx.ravel(), yy.ravel()]

        zz = model.decision_function(grid).reshape(xx.shape)

        ax_plot.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.4)
        ax_plot.contour(xx, yy, zz, levels=[0], colors="k", linewidths=2)

        if model_name == 'Pegasos (SVM)':
            ax_plot.contour(xx, yy, zz, levels=[-1, 1], colors=["blue", "blue"],
                            linestyles=["--", "--"], linewidths=[1, 1], alpha=0.5)

        colors = ["#E91E63" if yi == -1 else "#4CAF50" for yi in y]
        ax_plot.scatter(X[:, 0], X[:, 1], c=colors, s=30, edgecolors="k", zorder=3)

        # Ha Kernel Perceptron, SV-k jelölése
        if model_name == 'RBF Kernel Perceptron':
            sv_mask = model.alphas > 0
            ax_plot.scatter(X[sv_mask, 0], X[sv_mask, 1], s=100, facecolors="none",
                            edgecolors="gold", linewidths=1.5, zorder=4, label=f"SV: {sv_mask.sum()}")
            ax_plot.legend(loc="upper right")

        ax_plot.set_title(f"{model_name} on {data_name} | Pontosság: {acc*100:.1f}%", fontsize=12, fontweight='bold')
        ax_plot.set_xlim(x_min, x_max)
        ax_plot.set_ylim(y_min, y_max)

        fig.canvas.draw_idle()

    btn_train.on_clicked(update_plot)
    update_plot(None)

    plt.show()

if __name__ == "__main__":
    main()
