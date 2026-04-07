import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

def main():
    # Setup plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define node positions for f(x,y,z) = (x+y)*z
    # x (-2), y (5), z (-4)
    nodes = {
        'x': (0.1, 0.8),
        'y': (0.1, 0.6),
        'z': (0.3, 0.2),
        '+': (0.4, 0.7),
        '*': (0.7, 0.45),
        'out': (0.9, 0.45)
    }

    # Values for forward pass
    vals = {
        'x': -2,
        'y': 5,
        'z': -4,
        'q': 3,   # x + y
        'f': -12  # q * z
    }

    # Values for backward pass (gradients)
    grads = {
        'f': 1,
        'z': 3,   # df/dq * dq/dx ? No, df/dz = q = 3
        'q': -4,  # df/dq = z = -4
        'x': -4,  # df/dx = df/dq * dq/dx = -4 * 1 = -4
        'y': -4   # df/dy = df/dq * dq/dy = -4 * 1 = -4
    }

    def draw_base_graph():
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title("Backpropagation algoritmus: $f(x,y,z) = (x + y) \cdot z$", fontsize=16, pad=20)

        # Draw edges
        ax.annotate('', xy=nodes['+'], xytext=nodes['x'], arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
        ax.annotate('', xy=nodes['+'], xytext=nodes['y'], arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
        ax.annotate('', xy=nodes['*'], xytext=nodes['+'], arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
        ax.annotate('', xy=nodes['*'], xytext=nodes['z'], arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
        ax.annotate('', xy=nodes['out'], xytext=nodes['*'], arrowprops=dict(arrowstyle="->", lw=2, color='gray'))

        # Draw nodes
        circle_plus = patches.Circle(nodes['+'], 0.05, facecolor='lightblue', edgecolor='black', lw=2, zorder=10)
        circle_mul = patches.Circle(nodes['*'], 0.05, facecolor='lightgreen', edgecolor='black', lw=2, zorder=10)

        ax.add_patch(circle_plus)
        ax.add_patch(circle_mul)

        ax.text(nodes['+'][0], nodes['+'][1], '+', fontsize=20, ha='center', va='center', zorder=11)
        ax.text(nodes['*'][0], nodes['*'][1], '$\\times$', fontsize=20, ha='center', va='center', zorder=11)

        ax.text(nodes['x'][0]-0.05, nodes['x'][1], 'x', fontsize=16, ha='right', va='center')
        ax.text(nodes['y'][0]-0.05, nodes['y'][1], 'y', fontsize=16, ha='right', va='center')
        ax.text(nodes['z'][0]-0.05, nodes['z'][1], 'z', fontsize=16, ha='right', va='center')
        ax.text(nodes['out'][0]+0.02, nodes['out'][1], 'f', fontsize=16, ha='left', va='center')

    def add_text(x, y, text, color, is_grad=False):
        offset_y = -0.08 if is_grad else 0.05
        ax.text(x, y + offset_y, text, fontsize=14, color=color, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor=color, boxstyle='round,pad=0.2', alpha=0.8, lw=2), zorder=15)

    def update(frame):
        draw_base_graph()

        # FORWARD PASS
        if frame >= 1:
            add_text(nodes['x'][0] + 0.05, nodes['x'][1], f"{vals['x']}", 'green')
            add_text(nodes['y'][0] + 0.05, nodes['y'][1], f"{vals['y']}", 'green')
            add_text(nodes['z'][0] + 0.1, nodes['z'][1], f"{vals['z']}", 'green')

        if frame >= 2:
            add_text(nodes['+'][0], nodes['+'][1] + 0.05, f"q={vals['q']}", 'green')

        if frame >= 3:
            add_text(nodes['*'][0], nodes['*'][1] + 0.05, f"f={vals['f']}", 'green')

        # BACKWARD PASS
        if frame >= 4:
            ax.text(0.5, 0.9, "Visszaterjesztés (Backpropagation)", fontsize=16, color='red', ha='center', fontweight='bold')
            add_text(nodes['*'][0] + 0.1, nodes['*'][1], f"grad={grads['f']}", 'red', is_grad=True)

        if frame >= 5:
            add_text(nodes['+'][0] + 0.15, nodes['+'][1]-0.15, f"grad={grads['q']}", 'red', is_grad=True)
            add_text(nodes['z'][0] + 0.2, nodes['z'][1], f"grad={grads['z']}", 'red', is_grad=True)

        if frame >= 6:
            add_text(nodes['x'][0] + 0.1, nodes['x'][1], f"grad={grads['x']}", 'red', is_grad=True)
            add_text(nodes['y'][0] + 0.1, nodes['y'][1], f"grad={grads['y']}", 'red', is_grad=True)

        return ax,

    # Frames:
    # 0: Empty structure
    # 1: Init X, Y, Z (Forward)
    # 2: Calc q (Forward)
    # 3: Calc f (Forward)
    # 4: Calc grad f (Backward)
    # 5: Calc grad q, z (Backward)
    # 6: Calc grad x, y (Backward)
    sequence = [0, 1, 2, 3, 4, 5, 6]
    # Hold the final frame
    sequence.extend([6] * 5)

    anim = FuncAnimation(fig, update, frames=sequence, blit=False)

    output_file = "05_backpropagation_animation.gif"
    anim.save(output_file, writer='pillow', fps=1)
    print(f"Animáció mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
