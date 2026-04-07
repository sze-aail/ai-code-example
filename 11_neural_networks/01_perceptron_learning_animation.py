import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation(lr, output_file):
    # Dataset: AND gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, -1, 1])

    # Initialize weights [bias, w1, w2]
    np.random.seed(42)
    w = np.random.rand(3) * 0.2 - 0.1

    # Keep track of weights history for the animation
    history_w = [w.copy()]

    # Train perceptron and record state after each update
    max_epochs = 20
    for epoch in range(max_epochs):
        errors = 0
        for i in range(len(X)):
            x_i = np.insert(X[i], 0, 1) # Add bias term: [1, x1, x2]
            y_hat = 1 if np.dot(w, x_i) >= 0 else -1

            if y_hat != y[i]:
                w += lr * y[i] * x_i
                history_w.append(w.copy())
                errors += 1

        if errors == 0:
            # Pad with a few identical frames at the end to hold the final state
            for _ in range(5):
                history_w.append(w.copy())
            break

    # Setup plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_title(f"Perceptron Learning (AND Gate) - LR={lr}")
    ax.grid(True, alpha=0.3)

    # Scatter plot for points
    ax.scatter(X[y==-1, 0], X[y==-1, 1], color='red', marker='o', s=100, label='Class -1 (0)')
    ax.scatter(X[y==1, 0], X[y==1, 1], color='blue', marker='s', s=100, label='Class 1 (1)')

    # Line for decision boundary
    line, = ax.plot([], [], 'g-', lw=2, label='Decision Boundary')
    ax.legend(loc='lower right')

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        curr_w = history_w[frame]
        # Decision boundary: w0 + w1*x1 + w2*x2 = 0
        # => x2 = -(w1*x1 + w0) / w2
        x_vals = np.array([-0.5, 1.5])

        if abs(curr_w[2]) > 1e-5:
            y_vals = -(curr_w[1] * x_vals + curr_w[0]) / curr_w[2]
        else:
            y_vals = np.zeros_like(x_vals)

        line.set_data(x_vals, y_vals)
        ax.set_title(f"LR={lr} | Step {frame}\nW = [{curr_w[0]:.2f}, {curr_w[1]:.2f}, {curr_w[2]:.2f}]")
        return line,

    # Animate
    anim = FuncAnimation(fig, update, frames=len(history_w), init_func=init, blit=True)

    # Save as GIF
    anim.save(output_file, writer='pillow', fps=2)
    print(f"Sikeresen legenerálva és mentve: {output_file} (LR={lr}, lépések száma: {len(history_w)-6})")
    plt.close(fig)

def main():
    # Original version
    create_animation(0.1, '01_perceptron_learning.gif')

    # Try different learning rates
    lrs = [0.01, 0.5, 1.0]
    for lr in lrs:
        output_file = f'01_perceptron_learning_lr_{lr}.gif'
        create_animation(lr, output_file)

if __name__ == "__main__":
    main()