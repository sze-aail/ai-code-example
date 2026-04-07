import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    np.random.seed(42)

    # 2D Gaussian dataset extended along a specific axis (e.g. y = x)
    n_samples = 300
    mean = [0, 0]
    cov = [[3.0, 2.5],
           [2.5, 3.0]]

    X = np.random.multivariate_normal(mean, cov, n_samples)

    # Initialize weight randomly (normalized)
    w = np.random.randn(2)
    w = w / np.linalg.norm(w)

    lr = 0.01
    epochs = 4

    history_w = [w.copy()]

    # Unsupervised learning using Oja's Rule (stable Hebbian learning)
    # w(t+1) = w(t) + eta * y * (x - y * w(t))
    # where y = w.T * x

    for epoch in range(epochs):
        # We shuffle the data each epoch
        indices = np.random.permutation(n_samples)
        for i in indices:
            x_i = X[i]
            y = np.dot(w, x_i)
            # Oja's rule step
            dw = lr * y * (x_i - y * w)
            w = w + dw
            history_w.append(w.copy())

    # Sample down to e.g. 100 frames for smooth animation
    total_steps = len(history_w)
    frame_indices = np.linspace(0, total_steps - 1, 100, dtype=int)
    frames_w = [history_w[i] for i in frame_indices]

    # Pad end to linger on final frame
    for _ in range(15):
        frames_w.append(frames_w[-1])

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(X[:, 0], X[:, 1], color='lightblue', edgecolor='k', alpha=0.7, label='Data')

    # Calculate True First Principal Component for reference
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(X.T))
    pc1 = eigenvectors[:, np.argmax(eigenvalues)]

    ax.plot([0, pc1[0]*4], [0, pc1[1]*4], 'g--', lw=2, label='True PC1')
    ax.plot([0, -pc1[0]*4], [0, -pc1[1]*4], 'g--', lw=2)

    # Vector for plotting weight
    quiver = ax.quiver(0, 0, frames_w[0][0], frames_w[0][1],
                       angles='xy', scale_units='xy', scale=0.25, color='red', label='Weight (Hebb/Oja)')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    def update(frame):
        w_current = frames_w[frame]
        quiver.set_UVC(w_current[0], w_current[1])
        ax.set_title(f"Hebb (Oja's) Rule Learning\nStep: {frame_indices[min(frame, len(frame_indices)-1)]}/{total_steps} | w = [{w_current[0]:.2f}, {w_current[1]:.2f}]")
        return quiver,

    anim = FuncAnimation(fig, update, frames=len(frames_w), blit=True)

    output_file = "08_hebb_learning_animation.gif"
    anim.save(output_file, writer='pillow', fps=10)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
