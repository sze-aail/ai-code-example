import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Rosenbrock-style valley function for testing optimizers
# Global minimum is at (1, 1)
def f(x, y):
    return (y - x**2)**2 + (x - 1)**2

def grad_f(x, y):
    df_dx = -4 * x * (y - x**2) + 2 * (x - 1)
    df_dy = 2 * (y - x**2)
    return np.array([df_dx, df_dy])

def main():
    n_steps = 150
    start_point = np.array([-2.0, 2.0])

    # Tuned learning rates for the Rosenbrock function
    lrs = {
        'SGD': 0.02,
        'Momentum': 0.02,
        'RMSprop': 0.1,
        'Adam': 0.15
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
    ax.set_title("Optimalizálók a Rosenbrock völgyben\n$f(x,y) = (y - x^2)^2 + (x - 1)^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    x_grid = np.linspace(-3, 3, 200)
    y_grid = np.linspace(-2, 5, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    # Generate background contours
    levels = np.logspace(-1, 3, 25)
    ax.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)

    # Mark the global minimum
    ax.plot(1, 1, 'r*', markersize=15, label="Global Min (1,1)")

    colors = {'SGD': 'red', 'Momentum': 'blue', 'RMSprop': 'green', 'Adam': 'purple'}
    lines = {}
    points = {}

    for name in histories.keys():
        lines[name], = ax.plot([], [], '-', color=colors[name], lw=2, label=name)
        points[name], = ax.plot([], [], 'o', color=colors[name], markersize=6)

    ax.legend(loc="upper left")

    def init():
        for name in histories.keys():
            lines[name].set_data([], [])
            points[name].set_data([], [])
        return tuple(lines.values()) + tuple(points.values())

    def update(frame):
        for name in histories.keys():
            h = np.array(histories[name])
            lines[name].set_data(h[:frame+1, 0], h[:frame+1, 1])
            # Highlight current point
            points[name].set_data([h[frame, 0]], [h[frame, 1]])

        ax.set_title(f"Optimalizálók a Rosenbrock völgyben - Lépés {frame}\n$f(x,y) = (y - x^2)^2 + (x - 1)^2$")
        return tuple(lines.values()) + tuple(points.values())

    anim = FuncAnimation(fig, update, frames=n_steps+1, init_func=init, blit=True)

    output_file = "07_optimizers_rosenbrock.gif"
    anim.save(output_file, writer='pillow', fps=10)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
