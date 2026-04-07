import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the objective function and its gradient
def f(x, y):
    return x**2 + 2 * y**2

def grad_f(x, y):
    return np.array([2 * x, 4 * y])

def plot_different_lrs():
    lrs = [0.05, 0.1, 0.25]
    n_steps = 20
    point_init = np.array([-4.0, 3.0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Gradient Descent - Különböző tanulási ráták (LR)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    x_grid = np.linspace(-5, 5, 100)
    y_grid = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    ax.contour(X, Y, Z, levels=np.logspace(-1.5, 2, 20), cmap='viridis', alpha=0.6)

    colors = ['blue', 'green', 'red']
    for lr, color in zip(lrs, colors):
        point = point_init.copy()
        history = [point.copy()]
        for _ in range(n_steps):
            grad = grad_f(point[0], point[1])
            point = point - lr * grad
            history.append(point.copy())
        history = np.array(history)
        ax.plot(history[:, 0], history[:, 1], 'o-', color=color, label=f'LR={lr}', markersize=4)

    ax.legend()
    output_file = "04_gradient_descent_lrs.png"
    plt.savefig(output_file, dpi=150)
    print(f"Ábra mentve: {output_file}")
    plt.close(fig)

def main():
    # Először készítsünk egy statikus ábrát több LR összehasonlításával
    plot_different_lrs()

    # Gradient Descent Parameters
    lr = 0.1
    n_steps = 30

    # Initial point
    point = np.array([-4.0, 3.0])
    history = [point.copy()]

    # Run Gradient Descent
    for _ in range(n_steps):
        grad = grad_f(point[0], point[1])
        point = point - lr * grad
        history.append(point.copy())

    history = np.array(history)

    # Setup plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Gradient Descent Animation\n$f(x, y) = x^2 + 2y^2$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Create contour background
    x_grid = np.linspace(-5, 5, 100)
    y_grid = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    ax.contour(X, Y, Z, levels=np.logspace(-1.5, 2, 20), cmap='viridis', alpha=0.6)

    # Initialize animation elements
    line, = ax.plot([], [], 'ro-', lw=2, markersize=5, label='Útvonal (Path)')
    current_point, = ax.plot([], [], 'r*', markersize=10)
    ax.legend(loc="upper right")

    def init():
        line.set_data([], [])
        current_point.set_data([], [])
        return line, current_point

    def update(frame):
        # Plot history up to the current frame
        line.set_data(history[:frame+1, 0], history[:frame+1, 1])
        # Highligh current point
        current_point.set_data([history[frame, 0]], [history[frame, 1]])
        ax.set_title(f"Gradient Descent Lépés {frame}\n$f(x, y) = x^2 + 2y^2$ | x={history[frame, 0]:.2f}, y={history[frame, 1]:.2f}")
        return line, current_point

    anim = FuncAnimation(fig, update, frames=len(history), init_func=init, blit=True)

    output_file = "04_gradient_descent_animation.gif"
    anim.save(output_file, writer='pillow', fps=5)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
