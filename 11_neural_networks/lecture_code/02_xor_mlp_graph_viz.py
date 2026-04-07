import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class XOR_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

def main():
    torch.manual_seed(42)
    model = XOR_MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

    print("Hálózat gyorstanítása...")
    for _ in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

    w1 = model.net[0].weight.data.numpy()  # (4, 2)
    b1 = model.net[0].bias.data.numpy()    # (4,)
    w2 = model.net[2].weight.data.numpy()  # (1, 4)
    b2 = model.net[2].bias.data.numpy()    # (1,)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    layer_x = [0, 1.5, 3]

    input_nodes = [(layer_x[0], 0.6), (layer_x[0], 0.4)]
    hidden_nodes = [(layer_x[1], 0.8), (layer_x[1], 0.6), (layer_x[1], 0.4), (layer_x[1], 0.2)]
    output_nodes = [(layer_x[2], 0.5)]

    bias_nodes = [(layer_x[0], 0.8), (layer_x[1], 0.95)] # Bias input, Bias hidden

    max_w = max(np.max(np.abs(w1)), np.max(np.abs(b1)), np.max(np.abs(w2)), np.max(np.abs(b2)))

    def draw_edge(n1, n2, weight):
        color = '#E91E63' if weight < 0 else '#2196F3' # Red negative, Blue positive
        lw = 1 + 5 * (abs(weight) / max_w)
        ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color=color, linewidth=lw, alpha=0.6, zorder=1)

        # Eltoljuk a feliratot picit az első Node irányába, nehogy átfedjék egymást középen
        x_text = n1[0] + (n2[0] - n1[0]) * 0.35
        # Kicsit szórjuk a magasságot a jobb olvashatóságért y tengely mentén
        y_text = n1[1] + (n2[1] - n1[1]) * 0.35 + np.random.uniform(-0.03, 0.03)

        ax.text(x_text, y_text, f"{weight:.2f}", fontsize=10, color=color,
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1), zorder=2)

    # Draw w1
    for i, h_node in enumerate(hidden_nodes):
        for j, i_node in enumerate(input_nodes):
            draw_edge(i_node, h_node, w1[i, j])
        # Draw b1
        draw_edge(bias_nodes[0], h_node, b1[i])

    # Draw w2
    for i, o_node in enumerate(output_nodes):
        for j, h_node in enumerate(hidden_nodes):
            draw_edge(h_node, o_node, w2[i, j])
        # Draw b2
        draw_edge(bias_nodes[1], o_node, b2[i])

    # Draw nodes
    for node in input_nodes + hidden_nodes + output_nodes:
        circle = plt.Circle(node, 0.08, color='#EEEEEE', ec='#333333', lw=2, zorder=3)
        ax.add_patch(circle)

    for node in bias_nodes:
        circle = plt.Circle(node, 0.06, color='#FFF59D', ec='#333333', lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(node[0], node[1], "B=1", ha='center', va='center', zorder=4, fontsize=9)

    # Labels inside nodes
    ax.text(input_nodes[0][0], input_nodes[0][1], "x1", ha='center', va='center', zorder=4, fontsize=12)
    ax.text(input_nodes[1][0], input_nodes[1][1], "x2", ha='center', va='center', zorder=4, fontsize=12)

    for idx, node in enumerate(hidden_nodes):
        ax.text(node[0], node[1], f"h{idx+1}", ha='center', va='center', zorder=4, fontsize=12)

    ax.text(output_nodes[0][0], output_nodes[0][1], "y", ha='center', va='center', zorder=4, fontsize=12)

    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#2196F3', lw=4),
                    Line2D([0], [0], color='#E91E63', lw=4)]
    ax.legend(custom_lines, ['Pozitív súly', 'Negatív súly'], loc='upper right')

    ax.set_title("XOR MLP Architektúra és Súlyok a betanulás után", fontsize=16)

    output_filename = "02_xor_mlp_graph.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Mentve: {output_filename}")

if __name__ == "__main__":
    main()
