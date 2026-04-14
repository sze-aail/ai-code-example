"""
kohonen_som.py — Kohonen önszervező térkép (SOM) és összehasonlítás DBSCAN-nel
Neurális hálók I. — Hajdu Csaba

Demonstrálja:
  - Kohonen SOM felügyeletlen tanulás: topológia-megőrző leképezés
  - SOM rács önszerveződése tanítás közben (animáció-szerű snapshot-ok)
  - Összehasonlítás DBSCAN-nel és k-Means-szel
  - Különböző adatformákon: klaszterek, félholdak, gyűrűk, svájci tekercs
  - U-mátrix: klaszterhatárok vizualizáció a SOM rácson
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.collections import LineCollection


# ══════════════════════════════════════════════════════════════
# KOHONEN SOM IMPLEMENTÁCIÓ
# ══════════════════════════════════════════════════════════════

class KohonenSOM:
    """
    Kohonen Self-Organizing Map (1982).

    Felügyeletlen tanulás: a rács neuronjai versenyeznek a bemenetért.
    A győztes (BMU = Best Matching Unit) és szomszédai a bemenet felé mozdulnak.

    Tanulási szabály:
      w_j(t+1) = w_j(t) + η(t) · h(j, bmu, t) · (x - w_j(t))

    ahol:
      η(t) = η₀ · exp(-t/τ_η)           ← csökkenő learning rate
      h(j, bmu, t) = exp(-d²_grid / 2σ²(t))  ← szomszédsági függvény (Gauss)
      σ(t) = σ₀ · exp(-t/τ_σ)           ← csökkenő szomszédság sugár
    """

    def __init__(self, grid_h, grid_w, n_features, random_state=42):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.n_features = n_features
        self.rng = np.random.RandomState(random_state)

        # Súlyok inicializálása (kis véletlenszerű értékek)
        self.weights = self.rng.randn(grid_h, grid_w, n_features) * 0.1

        # Rácspontok pozíciói (a szomszédság számításához)
        self.grid_coords = np.array([[i, j] for i in range(grid_h) for j in range(grid_w)])

    def _find_bmu(self, x):
        """Best Matching Unit keresése (legközelebbi neuron)."""
        diff = self.weights - x
        distances = np.sum(diff ** 2, axis=-1)
        bmu_idx = np.unravel_index(np.argmin(distances), (self.grid_h, self.grid_w))
        return bmu_idx

    def _neighborhood(self, bmu_idx, sigma):
        """Gauss szomszédsági függvény."""
        bmu_coord = np.array(bmu_idx)
        grid_distances = np.sum((self.grid_coords - bmu_coord) ** 2, axis=1)
        grid_distances = grid_distances.reshape(self.grid_h, self.grid_w)
        return np.exp(-grid_distances / (2 * sigma ** 2))

    def fit(self, X, n_epochs=100, eta0=0.5, sigma0=None, snapshot_epochs=None):
        """SOM tanítás."""
        n = X.shape[0]
        if sigma0 is None:
            sigma0 = max(self.grid_h, self.grid_w) / 2

        tau_eta = n_epochs / 2
        tau_sigma = n_epochs / np.log(sigma0)

        # Inicializáljuk az adattartományba
        for d in range(self.n_features):
            self.weights[:, :, d] = self.rng.uniform(
                X[:, d].min(), X[:, d].max(), (self.grid_h, self.grid_w))

        snapshots = []
        self.quantization_errors = []

        for epoch in range(n_epochs):
            eta = eta0 * np.exp(-epoch / tau_eta)
            sigma = max(0.5, sigma0 * np.exp(-epoch / tau_sigma))

            # Snapshot mentése
            if snapshot_epochs and epoch in snapshot_epochs:
                snapshots.append((epoch, self.weights.copy(), eta, sigma))

            # Véletlenszerű sorrend
            perm = self.rng.permutation(n)
            qe = 0

            for idx in perm:
                x = X[idx]
                bmu = self._find_bmu(x)
                h = self._neighborhood(bmu, sigma)

                # Súlyfrissítés
                for d in range(self.n_features):
                    self.weights[:, :, d] += eta * h * (x[d] - self.weights[:, :, d])

                qe += np.sum((x - self.weights[bmu]) ** 2)

            self.quantization_errors.append(qe / n)

        if snapshot_epochs and n_epochs - 1 not in snapshot_epochs:
            snapshots.append((n_epochs - 1, self.weights.copy(), eta, sigma))

        return snapshots

    def predict(self, X):
        """BMU hozzárendelés minden adatponthoz."""
        labels = []
        for x in X:
            bmu = self._find_bmu(x)
            labels.append(bmu[0] * self.grid_w + bmu[1])
        return np.array(labels)

    def u_matrix(self):
        """U-mátrix: szomszédos neuronok közötti átlagos távolság."""
        um = np.zeros((self.grid_h, self.grid_w))
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                neighbors = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_h and 0 <= nj < self.grid_w:
                        neighbors.append(np.linalg.norm(
                            self.weights[i, j] - self.weights[ni, nj]))
                um[i, j] = np.mean(neighbors)
        return um


def main():
    # ══════════════════════════════════════════════════════════════
    # 1. KÍSÉRLET: SOM önszerveződés vizualizáció
    # ══════════════════════════════════════════════════════════════

    print("=" * 60)
    print("1. SOM ÖNSZERVEZŐDÉS — rács evolúciója")
    print("=" * 60)

    np.random.seed(42)
    X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)
    X_blobs = StandardScaler().fit_transform(X_blobs).astype(np.float32)

    som = KohonenSOM(8, 8, 2, random_state=42)
    snaps = som.fit(X_blobs, n_epochs=200, eta0=0.5,
                    snapshot_epochs=[0, 5, 15, 50, 100, 199])

    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 10))

    for ax, (epoch, weights, eta, sigma) in zip(axes1.flat, snaps):
        ax.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap="Set1", s=15, alpha=0.3, zorder=1)

        # Rács vonalak
        for i in range(weights.shape[0]):
            ax.plot(weights[i, :, 0], weights[i, :, 1], "k-", linewidth=0.8, alpha=0.6, zorder=2)
        for j in range(weights.shape[1]):
            ax.plot(weights[:, j, 0], weights[:, j, 1], "k-", linewidth=0.8, alpha=0.6, zorder=2)

        # Neuronok
        ax.scatter(weights[:, :, 0].ravel(), weights[:, :, 1].ravel(),
                   c="gold", s=30, edgecolors="k", linewidth=0.5, zorder=3)

        ax.set_title(f"Epoch {epoch}\nη={eta:.3f}, σ={sigma:.2f}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig1.suptitle("Kohonen SOM: 8×8 rács önszerveződése", fontsize=15, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("som_onszervezodes.png", dpi=150)
    print("Ábra mentve: som_onszervezodes.png")


    # ══════════════════════════════════════════════════════════════
    # 2. KÍSÉRLET: SOM vs. DBSCAN vs. k-Means — 4 adathalmaz
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("2. SOM vs. DBSCAN vs. k-Means — összehasonlítás")
    print("=" * 60)

    datasets = [
        ("Klaszterek", *make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)),
        ("Félholdak", *make_moons(n_samples=500, noise=0.08, random_state=42)),
        ("Gyűrűk", *make_circles(n_samples=500, noise=0.05, factor=0.4, random_state=42)),
        ("Anizotróp", *make_blobs(n_samples=500, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)),
    ]

    methods = ["SOM (6×6)", "DBSCAN", "k-Means"]

    fig2, axes2 = plt.subplots(len(datasets), len(methods) + 1, figsize=(20, 16))

    for row, (ds_name, X_raw, y_true) in enumerate(datasets):
        X = StandardScaler().fit_transform(X_raw).astype(np.float32)
        n_true = len(np.unique(y_true))

        # Eredeti adat
        ax = axes2[row, 0]
        ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap="Set1", s=20, edgecolors="k", linewidth=0.2)
        ax.set_title(f"{ds_name}\n(valódi: {n_true} klaszter)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)
        if row == 0:
            ax.set_ylabel("Eredeti", fontsize=11, fontweight="bold")

        # SOM
        ax = axes2[row, 1]
        som = KohonenSOM(6, 6, 2, random_state=42)
        som.fit(X, n_epochs=200, eta0=0.5)
        som_labels = som.predict(X)
        try:
            sil_som = silhouette_score(X, som_labels)
        except:
            sil_som = -1
        ax.scatter(X[:, 0], X[:, 1], c=som_labels, cmap="tab20", s=20, edgecolors="k", linewidth=0.2)
        # Rács overlay
        w = som.weights
        for i in range(w.shape[0]):
            ax.plot(w[i, :, 0], w[i, :, 1], "k-", linewidth=0.5, alpha=0.4)
        for j in range(w.shape[1]):
            ax.plot(w[:, j, 0], w[:, j, 1], "k-", linewidth=0.5, alpha=0.4)
        ax.set_title(f"SOM (6×6)\nsil={sil_som:.2f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # DBSCAN
        ax = axes2[row, 2]
        db = DBSCAN(eps=0.3, min_samples=8).fit(X)
        db_labels = db.labels_
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = (db_labels == -1).sum()
        try:
            sil_db = silhouette_score(X[db_labels >= 0], db_labels[db_labels >= 0]) if n_clusters_db > 1 else -1
        except:
            sil_db = -1
        ax.scatter(X[:, 0], X[:, 1], c=db_labels, cmap="tab20", s=20, edgecolors="k", linewidth=0.2)
        ax.scatter(X[db_labels == -1, 0], X[db_labels == -1, 1], c="gray", s=10, marker="x", alpha=0.5)
        ax.set_title(f"DBSCAN (ε=0.3)\n{n_clusters_db} klaszter, {n_noise} zaj, sil={sil_db:.2f}",
                     fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

        # k-Means
        ax = axes2[row, 3]
        km = KMeans(n_clusters=n_true, random_state=42, n_init=10).fit(X)
        km_labels = km.labels_
        try:
            sil_km = silhouette_score(X, km_labels)
        except:
            sil_km = -1
        ax.scatter(X[:, 0], X[:, 1], c=km_labels, cmap="Set1", s=20, edgecolors="k", linewidth=0.2)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                   c="gold", s=150, marker="*", edgecolors="k", linewidth=1, zorder=5)
        ax.set_title(f"k-Means (K={n_true})\nsil={sil_km:.2f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.2)

        print(f"  {ds_name:12s} | SOM sil={sil_som:.2f} | DBSCAN sil={sil_db:.2f} ({n_clusters_db} cl) | "
              f"k-Means sil={sil_km:.2f}")

    fig2.suptitle("Felügyeletlen tanulás: SOM vs. DBSCAN vs. k-Means", fontsize=16, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("som_vs_dbscan_kmeans.png", dpi=150)
    print("Ábra mentve: som_vs_dbscan_kmeans.png")


    # ══════════════════════════════════════════════════════════════
    # 3. KÍSÉRLET: U-mátrix — klaszterhatárok vizualizáció
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("3. U-MÁTRIX: klaszterhatárok a SOM rácson")
    print("=" * 60)

    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (ds_name, X_raw, y_true) in zip(axes3, datasets[:3]):
        X = StandardScaler().fit_transform(X_raw).astype(np.float32)

        som = KohonenSOM(10, 10, 2, random_state=42)
        som.fit(X, n_epochs=300, eta0=0.5)
        um = som.u_matrix()

        im = ax.imshow(um, cmap="bone_r", interpolation="bilinear")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(f"{ds_name}\nU-mátrix (sötét = klaszterhatár)", fontsize=11, fontweight="bold")
        ax.set_xlabel("SOM oszlop")
        ax.set_ylabel("SOM sor")

    fig3.suptitle("U-mátrix: a SOM klaszterhatárai (nagy távolság = határ neuronok között)",
                  fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("som_u_matrix.png", dpi=150)
    print("Ábra mentve: som_u_matrix.png")


    # ══════════════════════════════════════════════════════════════
    # 4. ÖSSZEFOGLALÓ ÁBRA: mikor melyik?
    # ══════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("4. ÖSSZEFOGLALÁS")
    print("=" * 60)

    fig4, ax4 = plt.subplots(figsize=(12, 6))
    ax4.axis("off")

    table_data = [
        ["", "SOM (Kohonen)", "DBSCAN", "k-Means"],
        ["Típus", "Versengő tanulás\n(neurális)", "Sűrűség-alapú", "Centroid-alapú"],
        ["Klaszterszám", "Rács méretéből\n(előre fix)", "Automatikus\n(ε, min_samples)", "Előre megadott\n(K)"],
        ["Klaszter alak", "Voronoi-cellák\na rácson", "Tetszőleges alak\n(sűrűségfüggő)", "Gömb alakú\n(izotróp)"],
        ["Zaj kezelése", "Nincs explicit\nzajszűrés", "Zaj automatikusan\nkiszűrt (-1)", "Nincs\n(minden klaszterbe kerül)"],
        ["Dimenzió-\ncsökkentés", "Igen! 2D rács\n← nD adat", "Nem", "Nem"],
        ["Topológia-\nmegőrzés", "Igen\n(szomszédság)", "Részben\n(összefüggőség)", "Nem"],
        ["Tanulás", "Online\n(inkrementális)", "Batch", "Batch\n(EM-szerű)"],
        ["Erősség", "Vizualizáció,\ntopológia", "Tetszőleges alak,\nzajszűrés", "Egyszerű,\ngyors, skálázható"],
    ]

    table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 2.2)

    # Fejléc formázás
    for j in range(4):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data)):
        table[i, 0].set_facecolor("#ECEFF1")
        table[i, 0].set_text_props(fontweight="bold")

    ax4.set_title("SOM vs. DBSCAN vs. k-Means — mikor melyik?", fontsize=14,
                  fontweight="bold", pad=20)
    fig4.tight_layout()
    fig4.savefig("som_osszefoglalas.png", dpi=150)
    print("Ábra mentve: som_osszefoglalas.png")

    plt.close("all")
    print("\nKész!")


if __name__ == "__main__":
    main()