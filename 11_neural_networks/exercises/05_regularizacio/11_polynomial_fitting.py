import numpy as np
import matplotlib.pyplot as plt
import warnings

# Hagyjuk figyelmen kívül a magas fokszámú illesztésnél jelentkező RankWarning-ot
warnings.simplefilter('ignore')

def true_fun(X):
    # Egy hullámzó függvény (pl. koszinusz)
    return np.cos(1.5 * np.pi * X)

def main():
    np.random.seed(0)

    # Készítsünk zajos adatokat
    n_samples = 30
    X = np.sort(np.random.rand(n_samples))
    y = true_fun(X) + np.random.randn(n_samples) * 0.1

    degrees = [1, 4, 15]
    titles = [
        "Alulillesztés (Underfitting)\nFokszám = 1",
        "Megfelelő illesztés\nFokszám = 4",
        "Túlillesztés (Overfitting)\nFokszám = 15"
    ]

    plt.figure(figsize=(14, 5))
    X_test = np.linspace(0, 1, 100)

    for i, degree in enumerate(degrees):
        ax = plt.subplot(1, 3, i + 1)

        # Polinom illesztése
        coefficients = np.polyfit(X, y, degree)
        p = np.poly1d(coefficients)

        # Eredmények ábrázolása
        plt.plot(X_test, p(X_test), color='red', label="Tanult modell", linewidth=2)
        plt.plot(X_test, true_fun(X_test), color='green', label="Valódi függvény", linestyle='--', linewidth=2)
        plt.scatter(X, y, color='blue', edgecolor='k', s=40, label="Gyakorló adatok")

        plt.xlim((0, 1))
        plt.ylim((-1.5, 1.5))
        plt.title(titles[i], fontsize=14)
        plt.legend(loc="best", fontsize=10)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_filename = "11_polynomial_under_over_fitting.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Ábra mentve: {output_filename}")
    plt.close()

if __name__ == "__main__":
    main()
