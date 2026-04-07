import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def main():
    # Simulation parameters
    dt = 1.0     # Time step (ms)
    T = 100.0    # Total time (ms)
    time = np.arange(0, T, dt)
    n_steps = len(time)

    # LIF Neuron parameters
    tau_m = 10.0      # Membrane time constant
    R = 1.0           # Membrane resistance
    v_th = 1.0        # Spike threshold
    v_reset = 0.0     # Reset potential

    # Two neurons with different input currents
    # Neuron 1: Constant input current
    I1 = np.ones(n_steps) * 1.5
    # Neuron 2: Oscillating input current
    I2 = np.sin(time / 5.0) + 1.0

    # Initialize variables
    v1 = np.zeros(n_steps)
    v2 = np.zeros(n_steps)
    spikes1 = []
    spikes2 = []

    # Simulate
    for i in range(1, n_steps):
        # Neuron 1
        dv1 = (-(v1[i-1] - v_reset) + R * I1[i]) / tau_m * dt
        v1[i] = v1[i-1] + dv1
        if v1[i] >= v_th:
            v1[i] = v_reset
            spikes1.append(time[i])

        # Neuron 2
        dv2 = (-(v2[i-1] - v_reset) + R * I2[i]) / tau_m * dt
        v2[i] = v2[i-1] + dv2
        if v2[i] >= v_th:
            v2[i] = v_reset
            spikes2.append(time[i])

    # Plot Setup
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Spiking Neural Network (LIF Model) Szimuláció', fontsize=16)

    # Plot Input currents
    ax_i1 = axes[0, 0]
    ax_i2 = axes[0, 1]
    ax_v1 = axes[1, 0]
    ax_v2 = axes[1, 1]

    # Initialize plots
    line_i1, = ax_i1.plot([], [], 'g-', lw=2)
    ax_i1.set_xlim(0, T)
    ax_i1.set_ylim(0, 3)
    ax_i1.set_title("Neuron 1 Bemeneti Áram (I)", fontsize=12)
    ax_i1.set_ylabel("Áram")

    line_i2, = ax_i2.plot([], [], 'g-', lw=2)
    ax_i2.set_xlim(0, T)
    ax_i2.set_ylim(0, 3)
    ax_i2.set_title("Neuron 2 Bemeneti Áram (I)", fontsize=12)

    line_v1, = ax_v1.plot([], [], 'b-', lw=2)
    scatter_s1 = ax_v1.scatter([], [], color='red', marker='*')
    ax_v1.set_xlim(0, T)
    ax_v1.set_ylim(-0.2, 1.2)
    ax_v1.set_title("Neuron 1 Membrán Potenciál (V)", fontsize=12)
    ax_v1.set_xlabel("Idő (ms)")
    ax_v1.set_ylabel("Potenciál (V)")
    ax_v1.axhline(v_th, color='r', linestyle='--', label='Küszöb')
    ax_v1.legend(loc="upper right")

    line_v2, = ax_v2.plot([], [], 'b-', lw=2)
    scatter_s2 = ax_v2.scatter([], [], color='red', marker='*')
    ax_v2.set_xlim(0, T)
    ax_v2.set_ylim(-0.2, 1.2)
    ax_v2.set_title("Neuron 2 Membrán Potenciál (V)", fontsize=12)
    ax_v2.set_xlabel("Idő (ms)")
    ax_v2.axhline(v_th, color='r', linestyle='--', label='Küszöb')

    def update(frame):
        t_curr = time[:frame]

        # Update input currents
        line_i1.set_data(t_curr, I1[:frame])
        line_i2.set_data(t_curr, I2[:frame])

        # Update potentials
        line_v1.set_data(t_curr, v1[:frame])
        # Gather spikes up to current frame
        spks1 = [s for s in spikes1 if s <= time[frame]]
        if spks1:
            scatter_s1.set_offsets(np.c_[spks1, [v_th]*len(spks1)])

        line_v2.set_data(t_curr, v2[:frame])
        spks2 = [s for s in spikes2 if s <= time[frame]]
        if spks2:
            scatter_s2.set_offsets(np.c_[spks2, [v_th]*len(spks2)])

        return line_i1, line_i2, line_v1, scatter_s1, line_v2, scatter_s2

    print("Animáció készítése...")
    anim = FuncAnimation(fig, update, frames=n_steps, blit=False, interval=20)

    output_file = "14_spiking_neural_network_animation.gif"
    anim.save(output_file, writer='pillow', fps=20)
    print(f"Mentve: {output_file}")
    plt.close(fig)

if __name__ == "__main__":
    main()
