"""
logisztikus_regresszio.py — Logisztikus regresszió sklearn-nel
Forrás: mesterseges-intelligencia-kodok (07_logistic_regression), átdolgozva

Demonstrálja:
  - Logisztikus regresszió mint egyszerű osztályozó
  - Döntési határ megjelenítése
  - Sigmoid függvény illesztése bináris esetre
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import datasets
from scipy.special import expit


# ── 1. Iris adathalmaz — többosztályos döntési határ ──
iris = datasets.load_iris()
X = iris.data[:, :2]  # csak az első két jellemző
y = iris.target

logreg = LogisticRegression(C=1e5, max_iter=1000)
logreg.fit(X, y)

fig1, ax1 = plt.subplots(figsize=(8, 6))
h = 0.02
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax1.contourf(xx, yy, Z, cmap="RdYlBu", alpha=0.4)
scatter = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="RdYlBu", edgecolors="k", s=40)
ax1.set_xlabel("Csésze hossz (cm)")
ax1.set_ylabel("Csésze szélesség (cm)")
ax1.set_title("Logisztikus regresszió — Iris döntési határok", fontweight="bold")
ax1.legend(*scatter.legend_elements(), title="Osztály")
fig1.tight_layout()
fig1.savefig("logisztikus_iris.png", dpi=150)
print("Ábra mentve: logisztikus_iris.png")

# ── 2. Bináris eset: sigmoid illesztés ──
np.random.seed(42)
n = 100
x1d = np.random.randn(n) * 2
y1d = (x1d + np.random.randn(n) * 0.5 > 0).astype(int)

lr = LogisticRegression()
lr.fit(x1d.reshape(-1, 1), y1d)

fig2, ax2 = plt.subplots(figsize=(8, 5))
x_plot = np.linspace(-6, 6, 300)
prob = expit(lr.coef_[0][0] * x_plot + lr.intercept_[0])

ax2.scatter(x1d, y1d, c=["#2196F3" if yi == 0 else "#4CAF50" for yi in y1d],
            s=40, edgecolors="k", alpha=0.7, zorder=3)
ax2.plot(x_plot, prob, "r-", linewidth=2.5, label="σ(wx + b)")
ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Döntési küszöb (0.5)")
ax2.set_xlabel("x", fontsize=12)
ax2.set_ylabel("P(y=1|x)", fontsize=12)
ax2.set_title("Logisztikus regresszió — Sigmoid illesztés", fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("logisztikus_sigmoid.png", dpi=150)
print("Ábra mentve: logisztikus_sigmoid.png")

plt.close("all")
print("Kész!")
