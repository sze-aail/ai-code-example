import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Skewed bowl function to highlight differenes in optimizers
def f(x, y):
    return 0.1 * x**2 + 2 * y**2

def grad_f(x, y):
    return np.array([0.2 * x, 4 * y])

def main():
    n_steps = 50
    start_point = np.array([-4.0, 3.0])

    lrs = {
        'SGD': 0.1,
        'Momentum': 0.1,
        'RMSprop': 0.1,
        'Adam': 0.2
    }

    histories = {k: [start_point.copy()] for k in lrs.keys()}

    # 1. Run SGD
    p = start_point.copy()
    for _ in range(n_steps):
        p = p - lrs['SGD'] * grad_f(p[0], p[1])
        histories['SGD'].append(p.copy())

    # 2. Run Momentum
    p = start_point.copy()
    v = np.zeros(2)
    mu = 0.9
    for _ in range(n_steps):
        v = mu * v - lrs['Momentum'] * grad_f(p[0], p[1])
        p = p + v
        histories['Momentum'].append(p.copy())

    # 3. Run RMSprop
    p = start_point.copy()
    cache = np.zeros(2)
    decay = 0.99
    eps = 1e-8
    for _ in range(n_steps):
        g = grad_f(p[0], p[1])
        cache = decay * cache + (1 - decay) * g**2
        p = p - lrs['RMSprop'] * g / (np.sqrt(cache) + eps)
        histories['RMSprop'].append(p.copy())

    # 4. Run Adam
    p = start_point.copy()
    m = np.zeros(2)
    v = np.zeros(2)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    for t in range(1, n_steps + 1):
        g = grad_f(p[0], p[1])
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        p = p - lrs['Adam'] * m_hat / (np.sqrt(v_hat) + eps)
        histories['Adam'].append(p.copy())

    # Plot Setup
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Optimalizálók összehasonlítása\n$f(x,y) = 0.1 x^2 + 2 y^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    x_grid = np.linspace(-5, 5, 100)
    y_grid = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    ax.contour(X, Y, Z, levels=np.logspace(-1.5, 2, 20), cmap='viridis', alpha=0.6)

    colors = {'SGD': 'red', 'Momentum': 'blue', 'RMSprop': 'green', 'Adam': 'purple'}
    lines = {}
    points = {}

    for name in histories.keys():
        lines[name], = ax.plot([], [], 'o-', color=colors[name], lw=2, markersize=4, label=name)
        points[name], = ax.plot([], [], '*', color=colors[name], markersize=10)

    ax.legend(loc="upper right")

    def init():
        for name in histories.keys():
            lines[name].set_data([], [])
            points[name].set_data([], [])
        return tuple(lines.values()) + tuple(points.values())

    def update(frame):
        for name in histories.keys():
            h = np.array(histories[name])
            lines[name].set_data(h[:frame+1, 0], h[:frame+1, 1])
            # Highligh current point with error handling (array vs scalar)
            points[name].set_data([h[frame, 0]], [h[frame, 1]])
        ax.set_title(f"Optimalizálók összehasonlítása - Lépés {frame}\n$f(x,y) = 0.1 x^2 + 2 y^2$")
        return tuple(lines.values()) + tuple(points.values())

    anim = FuncAnimation(fig, update, frames=n_steps+1, init_func=init, blit=True)

    output_file = "07_optimizers_animation.gif"
    anim.save(output_file, writer='pillow', fps=6)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
