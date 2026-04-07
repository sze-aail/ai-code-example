import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# --- 1. Adatgenerálás (Egy egyszerű 2D forma, pl. egy svájci tekercs vagy kör) ---
def get_data(n_samples=1000):
    torch.manual_seed(42)
    np.random.seed(42)
    # Generáljunk egy kört
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    r = 2.0
    x = r * np.cos(theta) + np.random.normal(0, 0.1, n_samples)
    y = r * np.sin(theta) + np.random.normal(0, 0.1, n_samples)
    data = np.vstack([x, y]).T
    return torch.tensor(data, dtype=torch.float32)

# --- 2. Zaj időzítés (Beta schedule) ---
T = 40
beta = torch.linspace(0.001, 0.05, T)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# Előretejesztés (Forward) függvény
def q_sample(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    a_bar_t = alpha_bar[t].view(-1, 1)
    x_t = torch.sqrt(a_bar_t) * x_0 + torch.sqrt(1 - a_bar_t) * noise
    return x_t, noise

# --- 3. Modell (idő-kondicionált MLP a zaj becslésére) ---
class DenoiseMLP(nn.Module):
    def __init__(self):
        super(DenoiseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, t):
        # normalize t
        t_norm = (t.float() / T).view(-1, 1)
        # concatenate x and t
        x_t = torch.cat([x, t_norm], dim=1)
        return self.net(x_t)

def main():
    X_0 = get_data(1000)

    # --- 4. Modell Tanítása ---
    model = DenoiseMLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    epochs = 2000
    batch_size = 256

    print("Egyszerű Diffusion Modell Tanítása...")
    model.train()
    for ev in range(epochs):
        # Véletlen batch
        idx = torch.randint(0, X_0.shape[0], (batch_size,))
        x_batch = X_0[idx]

        # Véletlen t minden elemhez
        t = torch.randint(0, T, (batch_size,))

        # Forward diffúzió
        noise = torch.randn_like(x_batch)
        x_t, true_noise = q_sample(x_batch, t, noise)

        # Zaj becslése
        pred_noise = model(x_t, t)

        # Loss (MSE)
        loss = nn.MSELoss()(pred_noise, true_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ev+1) % 500 == 0:
            print(f"Iters: {ev+1}/{epochs} | Loss: {loss.item():.4f}")

    # --- 5. Animáció generálása ---
    # Készítünk egy sorozatot:
    # Fázis 1: Előrefelé (Forward) -> Adatból Zaj (T lépés)
    # Fázis 2: Visszafelé (Reverse) -> Zajból generált Adat (T lépés)

    print("Trajektóriák generálása az animációhoz...")
    num_samples_anim = 500
    x_anim_data = X_0[:num_samples_anim]

    forward_frames = []
    # Fázis 1: Forward
    for t_step in range(T):
        t_tensor = torch.full((num_samples_anim,), t_step, dtype=torch.long)
        x_t, _ = q_sample(x_anim_data, t_tensor)
        forward_frames.append(x_t.detach().numpy())

    # Utolsó forward állapot = tiszta zaj
    x_t = torch.randn((num_samples_anim, 2))

    backward_frames = [x_t.detach().numpy()]
    # Fázis 2: Reverse
    model.eval()
    with torch.no_grad():
        for t_step in reversed(range(T)):
            t_tensor = torch.full((num_samples_anim,), t_step, dtype=torch.long)

            # x_t -> noise
            pred_noise = model(x_t, t_tensor)

            a_t = alpha[t_step]
            a_bar_t = alpha_bar[t_step]

            # Sampling logika: x_{t-1} = 1/sqrt(a_t) * (x_t - (1-a_t)/sqrt(1-a_bar_t) * pred_noise) + sigma * z
            var = beta[t_step] if t_step > 0 else 0.0

            term1 = (x_t - ((1.0 - a_t) / torch.sqrt(1.0 - a_bar_t)) * pred_noise)
            mean = term1 / torch.sqrt(a_t)

            if t_step > 0:
                z = torch.randn_like(x_t)
            else:
                z = torch.zeros_like(x_t)

            x_t = mean + math.sqrt(var) * z
            backward_frames.append(x_t.detach().numpy())

    # --- 6. Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))
    scatter = ax.scatter([], [], s=10, c='blue', alpha=0.6)
    title = ax.set_title("")

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.grid(True, alpha=0.3)

    # Teljes frame lista: Forward + kis várakozás + Backward + kis várakozás
    all_frames = []
    for _ in range(10): all_frames.append((forward_frames[0], "1: Eredeti adat (Kör)"))
    for f in forward_frames: all_frames.append((f, "1: Forward pass (Zajosítás...)"))
    for _ in range(10): all_frames.append((forward_frames[-1], "2: Tiszta normál zaj"))
    for f in backward_frames: all_frames.append((f, "2: Reverse pass (Zajszűrés...)"))
    for _ in range(15): all_frames.append((backward_frames[-1], "3: Generált adat (Kör)"))

    def update(frame_data):
        pts, txt = frame_data
        scatter.set_offsets(pts)
        title.set_text(txt)

        # Színváltás fázistól függően
        if "Forward" in txt: scatter.set_color('red')
        elif "Reverse" in txt: scatter.set_color('green')
        elif "Generált" in txt: scatter.set_color('blue')
        else: scatter.set_color('gray')

        return scatter, title

    anim = FuncAnimation(fig, update, frames=all_frames, blit=False, interval=50)

    output_file = "15_simple_diffusion_model.gif"
    print(f"Animáció mentése: {output_file} ...")
    anim.save(output_file, writer='pillow', fps=15)
    print("Sikeresen mentve!")
    plt.close(fig)

if __name__ == "__main__":
    main()
