"""
10_szimmetria_csoportok.py — Forgási invariancia és szimmetriacsoportok CNN-ekben
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - A probléma: standard CNN NEM forgás-invariáns (csak eltolás-ekviváriáns)
  - Szimmetriacsoportok: C4, C8, p4, p4m — mit jelent az ekvivariancia
  - Group Equivariant CNN (G-CNN, Cohen & Welling, 2016): csoportkonvolúció
  - Data augmentation vs. beépített szimmetria összehasonlítás
  - Steerable szűrők: forgás-ekviváriáns bázisfüggvények
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. A PROBLÉMA: CNN NEM FORGÁS-INVARIÁNS
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. A PROBLÉMA: standard CNN forgás-érzékenysége")
    print("=" * 60)

    def rotate_image_90(img, k=1):
        """90 fokos forgatás k-szor (numpy 2D)."""
        return np.rot90(img, k)

    def rotate_image_arbitrary(img, angle_deg):
        """Tetszőleges szögű forgatás bilineáris interpolációval."""
        h, w = img.shape
        cy, cx = h / 2, w / 2
        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        result = np.zeros_like(img)
        for y in range(h):
            for x in range(w):
                src_x = cos_a * (x - cx) + sin_a * (y - cy) + cx
                src_y = -sin_a * (x - cx) + cos_a * (y - cy) + cy
                ix, iy = int(src_x), int(src_y)
                if 0 <= ix < w and 0 <= iy < h:
                    result[y, x] = img[iy, ix]
        return result

    # Egy számjegy különböző forgatásokkal
    digits = load_digits()
    sample = digits.data[0].reshape(8, 8) / 16.0  # egy '0'

    fig1, axes = plt.subplots(2, 6, figsize=(18, 6))
    angles = [0, 30, 45, 60, 90, 180]

    # Felső sor: forgatott képek
    for ax, angle in zip(axes[0], angles):
        rotated = rotate_image_arbitrary(sample, angle)
        ax.imshow(rotated, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{angle}°", fontsize=12, fontweight="bold")
        ax.axis("off")

    # Alsó sor: standard CNN válasza
    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(8, 10)
        def forward(self, x):
            return self.fc(self.pool(F.relu(self.conv1(x))).flatten(1))

    torch.manual_seed(42)
    cnn = TinyCNN()
    # Tanítás egyenes (0°) képeken
    X = torch.tensor(digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0)
    y_digits = torch.tensor(digits.target, dtype=torch.long)
    opt = torch.optim.Adam(cnn.parameters(), lr=0.01)
    for _ in range(100):
        loss = nn.CrossEntropyLoss()(cnn(X), y_digits)
        opt.zero_grad(); loss.backward(); opt.step()

    cnn.eval()
    probs_by_angle = []
    for ax, angle in zip(axes[1], angles):
        rotated = rotate_image_arbitrary(sample, angle)
        inp = torch.tensor(rotated, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            logits = cnn(inp)
            probs = F.softmax(logits, dim=1).numpy().flatten()
        pred = probs.argmax()
        conf = probs.max()
        probs_by_angle.append((angle, pred, conf))
        ax.bar(range(10), probs, color="#2196F3", alpha=0.7)
        ax.set_title(f"Pred: {pred} ({conf*100:.0f}%)",
                     fontweight="bold", color="#4CAF50" if pred == 0 else "#E91E63")
        ax.set_ylim(0, 1); ax.set_xticks(range(10))

        print(f"  {angle:3d}°: pred={pred}, conf={conf*100:.1f}%")

    axes[0, 0].set_ylabel("Bemeneti kép", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("CNN osztály-\nvalószínűségek", fontsize=11, fontweight="bold")

    fig1.suptitle("Standard CNN forgás-érzékenysége: 30° forgatás már elrontja a predikciót!",
                  fontsize=14, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("10_forgatas_problema.png", dpi=150)
    print("Ábra mentve: 10_forgatas_problema.png\n")


    # ══════════════════════════════════════════════════════════════
    # 2. SZIMMETRIACSOPORTOK VIZUALIZÁCIÓ
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("2. SZIMMETRIACSOPORTOK")
    print("=" * 60)

    fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))

    # Alap szűrő (aszimmetrikus)
    base_filter = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, 1, 1]], dtype=float)

    # C1: nincs szimmetria (standard CNN)
    group_c1 = [base_filter]

    # C4: 4 forgás (0°, 90°, 180°, 270°)
    group_c4 = [np.rot90(base_filter, k) for k in range(4)]

    # C8: 8 forgás (45° lépésekkel, közelítve)
    group_c8 = []
    for k in range(8):
        angle = k * 45
        rotated = rotate_image_arbitrary(base_filter, angle)
        group_c8.append(rotated)

    # p4m: 4 forgás × 2 tükrözés = 8 elem
    group_p4m = []
    for mirror in [False, True]:
        f = np.fliplr(base_filter) if mirror else base_filter
        for k in range(4):
            group_p4m.append(np.rot90(f, k))

    groups = [
        ("C1 (trivális)\n1 elem = standard CNN", group_c1),
        ("C4 (ciklikus)\n4 elem = 90° forgatások", group_c4),
        ("C8 (ciklikus)\n8 elem = 45° forgatások", group_c8),
        ("p4m (diéderes)\n8 elem = 4 forgatás × 2 tükrözés", group_p4m),
    ]

    for col, (name, group) in enumerate(groups):
        # Felső sor: csoport elemei
        ax = axes2[0, col]
        n = len(group)
        cols_g = min(4, n)
        rows_g = (n + cols_g - 1) // cols_g
        grid = np.zeros((rows_g * 4, cols_g * 4))
        for i, g in enumerate(group[:8]):
            r, c = divmod(i, cols_g)
            h, w = g.shape
            grid[r*4:r*4+h, c*4:c*4+w] = g
        ax.imshow(grid, cmap="Blues", vmin=-0.5, vmax=1.5)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

        # Alsó sor: csoport hatása egy jellemzőre
        ax = axes2[1, col]
        # A csoport által generált összes szűrőválasz összege
        responses = np.zeros((8, 8))
        test_img = digits.data[0].reshape(8, 8) / 16.0
        for g in group:
            h, w = g.shape
            padded = np.pad(test_img, ((1, 1), (1, 1)), mode='constant')
            for i in range(8):
                for j in range(8):
                    responses[i, j] += np.sum(padded[i:i+h, j:j+w] * g)
        ax.imshow(responses, cmap="viridis")
        ax.set_title(f"Csoport-konvolúció\nválasz ({n} szűrő összege)", fontsize=10)
        ax.axis("off")

    fig2.suptitle("Szimmetriacsoportok: egy szűrő → csoport összes transzformáltja",
                  fontsize=15, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("10_szimmetria_csoportok.png", dpi=150)
    print("Ábra mentve: 10_szimmetria_csoportok.png\n")


    # ══════════════════════════════════════════════════════════════
    # 3. GROUP EQUIVARIANT CNN (G-CNN) IMPLEMENTÁCIÓ
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("3. GROUP EQUIVARIANT CNN (G-CNN)")
    print("=" * 60)

    class GroupConv2d(nn.Module):
        """
        C4 csoportkonvolúció (Cohen & Welling, 2016).

        Ahelyett, hogy egy szűrőt tanulnánk, a szűrő ÖSSZES 90°-os
        forgatottját is alkalmazzuk. Ez garantálja az ekvivarienciát:

        f(ρ·x) = ρ·f(x)  (a kimenet úgy fordul, ahogy a bemenet)

        Paraméterszám: UGYANANNYI mint a standard conv!
        (csak 1 szűrőt tanulunk, a 4 forgatottat a csoport generálja)
        """
        def __init__(self, in_channels, out_channels, kernel_size=3, n_rotations=4):
            super().__init__()
            self.n_rotations = n_rotations
            self.base_weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.1)
            self.padding = kernel_size // 2

        def forward(self, x):
            outputs = []
            for k in range(self.n_rotations):
                # Szűrő forgatása k×90°-kal
                rotated_weight = torch.rot90(self.base_weight, k, dims=(-2, -1))
                out = F.conv2d(x, rotated_weight, padding=self.padding)
                outputs.append(out)

            # Stack: (B, out_ch, H, W) × n_rot → max-pool a forgatások felett
            stacked = torch.stack(outputs, dim=-1)  # (B, out_ch, H, W, n_rot)
            # Max-pool a csoporton → forgás-invariáns
            pooled, _ = stacked.max(dim=-1)
            return pooled


    class GroupEquivariantCNN(nn.Module):
        """G-CNN: csoportkonvolúciós rétegek + group pooling."""
        def __init__(self, n_rotations=4):
            super().__init__()
            self.gconv1 = GroupConv2d(1, 8, 3, n_rotations=n_rotations)
            self.gconv2 = GroupConv2d(8, 16, 3, n_rotations=n_rotations)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)

        def forward(self, x):
            x = F.relu(self.gconv1(x))
            x = F.relu(self.gconv2(x))
            return self.fc(self.pool(x).flatten(1))


    class StandardCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 10)
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return self.fc(self.pool(x).flatten(1))


    # ── Adat: tanítás egyenes + teszt forgatott ──
    X_all = digits.data.astype(np.float32).reshape(-1, 1, 8, 8) / 16.0
    y_all = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)

    Xt = torch.tensor(X_train); yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_test); yv = torch.tensor(y_test, dtype=torch.long)

    # Forgatott tesztadat
    def make_rotated_test(X_test, y_test, angle):
        X_rot = np.zeros_like(X_test)
        for i in range(len(X_test)):
            X_rot[i, 0] = rotate_image_arbitrary(X_test[i, 0], angle)
        return torch.tensor(X_rot), torch.tensor(y_test, dtype=torch.long)


    def train_and_test_rotations(model, name, epochs=100, use_aug=False):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            model.train()
            x_batch = Xt.clone()
            if use_aug:
                # Random 90° forgatás augmentáció
                for i in range(len(x_batch)):
                    k = torch.randint(0, 4, (1,)).item()
                    x_batch[i] = torch.rot90(x_batch[i], k, dims=(-2, -1))
            loss = criterion(model(x_batch), yt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        model.eval()
        results = {}
        for angle in [0, 45, 90, 135, 180, 270]:
            Xr, yr = make_rotated_test(X_test, y_test, angle)
            with torch.no_grad():
                acc = (model(Xr).argmax(1) == yr).float().mean().item()
            results[angle] = acc

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:30s} ({n_params:5d}p): " + " | ".join(f"{a}°={v*100:.0f}%" for a, v in results.items()))
        return results


    # Tanítás és tesztelés
    torch.manual_seed(42); r_std = train_and_test_rotations(StandardCNN(), "Standard CNN")
    torch.manual_seed(42); r_aug = train_and_test_rotations(StandardCNN(), "CNN + Aug (90° random)", use_aug=True)
    torch.manual_seed(42); r_c4 = train_and_test_rotations(GroupEquivariantCNN(4), "G-CNN (C4, 4 forgatás)")
    torch.manual_seed(42); r_c8 = train_and_test_rotations(GroupEquivariantCNN(4), "G-CNN (C4, pool max)")

    # ── Összehasonlító ábra ──
    fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5.5))

    angles_plot = [0, 45, 90, 135, 180, 270]
    for results, name, color, marker in [
        (r_std, "Standard CNN", "#E91E63", "o"),
        (r_aug, "CNN + 90° augm.", "#FF9800", "s"),
        (r_c4, "G-CNN (C4)", "#4CAF50", "^"),
    ]:
        accs = [results[a] * 100 for a in angles_plot]
        axes3[0].plot(angles_plot, accs, f"{marker}-", color=color, linewidth=2, markersize=8, label=name)

    axes3[0].set_xlabel("Tesztelési forgatás (°)", fontsize=12)
    axes3[0].set_ylabel("Pontosság (%)", fontsize=12)
    axes3[0].set_title("Forgás-robosztusság: standard vs. augmentáció vs. G-CNN", fontweight="bold")
    axes3[0].legend(fontsize=10); axes3[0].grid(True, alpha=0.3)
    axes3[0].set_xticks(angles_plot)

    # Összehasonlító táblázat
    ax = axes3[1]; ax.axis("off")
    data = [
        ["", "0°", "90°", "180°", "Átlag"],
        ["Standard CNN", f"{r_std[0]*100:.0f}%", f"{r_std[90]*100:.0f}%", f"{r_std[180]*100:.0f}%",
         f"{np.mean([r_std[a] for a in angles_plot])*100:.0f}%"],
        ["+ 90° augm.", f"{r_aug[0]*100:.0f}%", f"{r_aug[90]*100:.0f}%", f"{r_aug[180]*100:.0f}%",
         f"{np.mean([r_aug[a] for a in angles_plot])*100:.0f}%"],
        ["G-CNN (C4)", f"{r_c4[0]*100:.0f}%", f"{r_c4[90]*100:.0f}%", f"{r_c4[180]*100:.0f}%",
         f"{np.mean([r_c4[a] for a in angles_plot])*100:.0f}%"],
    ]
    table = ax.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 2.0)
    for j in range(5):
        table[0, j].set_facecolor("#37474F"); table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, 4):
        table[i, 0].set_facecolor("#ECEFF1"); table[i, 0].set_text_props(fontweight="bold")
    ax.set_title("Pontosság összehasonlítás", fontweight="bold")

    fig3.suptitle("Szimmetriacsoport-ekvivariáns CNN vs. standard megközelítések",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("10_gcnn_osszehasonlitas.png", dpi=150)
    print("\nÁbra mentve: 10_gcnn_osszehasonlitas.png\n")


    # ══════════════════════════════════════════════════════════════
    # 4. STEERABLE SZŰRŐK VIZUALIZÁCIÓ
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("4. STEERABLE SZŰRŐK")
    print("=" * 60)

    fig4, axes4 = plt.subplots(2, 4, figsize=(18, 9))

    # Steerable bázis: kör-harmonikusok (e^(imθ) = cos(mθ) + i·sin(mθ))
    xx, yy = np.meshgrid(np.linspace(-1, 1, 32), np.linspace(-1, 1, 32))
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)
    gauss = np.exp(-r**2 / 0.3)

    basis_functions = [
        ("m=0 (izotróp)\nGauss", gauss),
        ("m=1 cos\n(x-irányú él)", gauss * np.cos(theta)),
        ("m=1 sin\n(y-irányú él)", gauss * np.sin(theta)),
        ("m=2 cos\n(átlós)", gauss * np.cos(2 * theta)),
        ("m=2 sin\n(átlós)", gauss * np.sin(2 * theta)),
        ("m=3 cos\n(háromszög)", gauss * np.cos(3 * theta)),
        ("m=4 cos\n(négyzet)", gauss * np.cos(4 * theta)),
        ("m=0 + m=2\n(kombinált)", gauss * (1 + 0.5 * np.cos(2 * theta))),
    ]

    for ax, (name, basis) in zip(axes4.flat, basis_functions):
        ax.imshow(basis, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig4.suptitle("Steerable szűrők: kör-harmonikus bázisfüggvények\nBármely forgatás = bázisfüggvények LINEÁRIS kombinációja!",
                  fontsize=14, fontweight="bold")
    fig4.tight_layout()
    fig4.savefig("10_steerable_szurok.png", dpi=150)
    print("Ábra mentve: 10_steerable_szurok.png")


    # ══════════════════════════════════════════════════════════════
    # 5. ÖSSZEFOGLALÓ: MEGKÖZELÍTÉSEK HIERARCHIÁJA
    # ══════════════════════════════════════════════════════════════

    fig5, ax5 = plt.subplots(figsize=(14, 7))
    ax5.axis("off")
    data = [
        ["Megközelítés", "Szimmetria", "Garantált?", "Paraméter\ntöbblet", "Példa"],
        ["Data augmentation", "Bármilyen\n(tetszőleges tfm.)", "NEM\n(közelítés)", "Nincs\n(több adat kell)", "Random forgatás,\ntükrözés, zaj"],
        ["Test-Time Aug.\n(TTA)", "Bármilyen", "Közelítő\n(átlagolás)", "Nincs (lassabb\ninferencia)", "5-crop, flip\nátlagolás"],
        ["Group Equiv. CNN\n(G-CNN)", "Diszkrét csoport\n(C4, p4m, ...)", "IGEN\n(egzakt)", "Nincs (ugyanaz\na szűrő!)", "Cohen & Welling\n2016"],
        ["Steerable CNN", "Folytonos SO(2)\nvagy SO(3)", "IGEN\n(egzakt)", "Minimális\n(bázis koeff.)", "Weiler & Cesa\n2019 (e2cnn)"],
        ["Harmonic Networks", "Folytonos SO(2)", "IGEN", "Komplex\nszűrők", "Worrall et al.\n2017"],
        ["Capsule Networks", "Affinitás\n(forgás+nyújtás)", "Részben", "Routing\nparaméterek", "Sabour, Hinton\n2017"],
        ["Invariáns pooling\n(pl. max over rot.)", "Tetszőleges", "IGEN\n(invariancia!)", "n× szűrő\nalkalmazás", "Laptev et al.\n(klasszikus)"],
    ]
    table = ax5.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 2.2)
    for j in range(5):
        table[0, j].set_facecolor("#37474F"); table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ECEFF1"); table[i, 0].set_text_props(fontweight="bold")

    ax5.set_title("Forgási invariancia megközelítések: data aug → group equiv → steerable → capsule",
                  fontsize=14, fontweight="bold", pad=20)
    fig5.tight_layout()
    fig5.savefig("10_megkozelitesek.png", dpi=150)
    print("Ábra mentve: 10_megkozelitesek.png")

    plt.close("all")
    print("\nKész!")


if __name__=="__main__":
    main()