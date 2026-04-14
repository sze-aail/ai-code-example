# MLP méretezési szabályok: hány réteg, hány neuron?

**Neurális hálók I. — Hajdu Csaba**

## Gyors referencia

| Szabály | Képlet | Forrás |
|---|---|---|
| Kolmogorov / Hecht-Nielsen | **2n+1** rejtett neuron (n = bemenet dim.) | Kolmogorov (1957), Hecht-Nielsen (1987) |
| Kapacitás-szabály | **W ≤ N / α**, α ≈ 5–10 | Baum & Haussler (1989) |
| Hüvelykujj #1 | **(n_in + n_out) / 2** | gyakorlati tapasztalat |
| Hüvelykujj #2 | **√(n_in × n_out)** | geometriai közép |
| Hüvelykujj #3 | **≤ 2 × n_in** | felső korlát |
| Osztályozás | **K(K-1)/2** neuron K osztályhoz | Lippmann (1987) |
| Rétegszám (geometriai) | 0 = félsík, 1 = konvex, **2 = tetszőleges** | Lippmann (1987) |

## Részletes leírás

### 1. Kolmogorov-szuperpozíciós tétel (1957)

Bármely n-változós folytonos függvény előállítható:

```
f(x₁,...,xₙ) = Σⱼ₌₀²ⁿ gⱼ( Σᵢ₌₁ⁿ φᵢⱼ(xᵢ) )
```

Ebből következik: **egy rejtett réteg 2n+1 neuronnal elméletileg elegendő** bármely folytonos függvény közelítéséhez. A gyakorlatban a φ és g függvények nem konstruktívak, de az elméleti korlát érvényes.

**Mikor használd:** kiindulópontnak regresszióhoz, ha nincs más tájékoztatás.

### 2. Kapacitás-szabály (Baum & Haussler, 1989)

A generalizációs hiba korlátos, ha:

```
N_minták ≥ α × W_súlyok / ε
```

ahol ε a megengedett hibaarány. Megfordítva: **a súlyok száma legyen legfeljebb N/α**, ahol α ≈ 5–10. Egy rejtett réteges hálóra (n_in → h → n_out):

```
W = h × (n_in + 1) + n_out × (h + 1)
```

Ebből az ajánlott neuronszám:

```
h ≈ N / (α × (n_in + n_out))
```

**Mikor használd:** ha az overfitting a fő kockázat (kevés adat, sok jellemző).

### 3. Hüvelykujj-szabályok

Ezek nem tételek, hanem évtizedes gyakorlati tapasztalat:

| Szabály | Mikor jó | Mikor rossz |
|---|---|---|
| (in+out)/2 | Kis/közepes feladatok | Nagy bemeneti dimenzió (pl. képek) |
| √(in×out) | Geometriai közép, konzervatív | Túl kicsi lehet nagy feladatoknál |
| 2×in | Felső korlát, bőséges kapacitás | Overfitting kockázat kis adatnál |

### 4. Osztályozási szabály: K(K-1)/2

K osztály szétválasztásához **páronként** kell döntési határt húzni. K osztály között K(K-1)/2 pár van, tehát ennyi neuron szükséges a páronkénti szétválasztáshoz.

| K osztály | K(K-1)/2 neuron |
|---|---|
| 2 | 1 |
| 3 | 3 |
| 5 | 10 |
| 10 | 45 |

**Megjegyzés:** ez a szükséges minimum — a gyakorlatban több neuron javítja a döntési határ minőségét.

## Konkrét példák

### Iris-szerű feladat (4 bemenet, 3 osztály, 150 minta)

| Szabály | Ajánlás |
|---|---|
| Kolmogorov (2×4+1) | 9 |
| (4+1)/2 | 2 |
| Kapacitás (α=10) | 3 |
| K(K-1)/2 | 3 |
| **Gyakorlati ajánlás** | **3–9 neuron, 1 rejtett réteg** |

### MNIST-szerű feladat (784 bemenet, 10 osztály, 60000 minta)

| Szabály | Ajánlás |
|---|---|
| Kolmogorov (2×784+1) | 1569 |
| √(784×10) | 88 |
| Kapacitás (α=5) | 15 |
| K(K-1)/2 | 45 |
| **Gyakorlati ajánlás** | **128–512 neuron, 1–2 rejtett réteg** |

A Kolmogorov-ajánlás itt irreálisan nagy; a kapacitás-szabály túl konzervatív. A gyakorlat: 128–512 neuron, de **CNN-nel jobb** (a térbeli struktúrát kihasználva).

## Mikor melyiket használd?

```
                   Kevés adat (<1000)?
                    /              \
                  Igen             Nem
                  /                  \
        Kapacitás-szabály     Sok jellemző (>100)?
        h ≈ N/(10×(in+out))    /              \
                             Igen             Nem
                             /                  \
                     √(in×out) +          Kolmogorov (2n+1)
                     2 rejtett réteg      vagy 2×in
```

## A legfontosabb tanulság

Ezek **kiindulópontok**, nem végleges válaszok. A gyakorlatban:

1. Kezdd egy szabály szerinti mérettel
2. Figyeld a train/val görbéket
3. Ha underfitting → több neuron / mélyebb háló
4. Ha overfitting → kevesebb neuron / regularizáció / több adat
5. Hyperparameter keresés (grid search, Bayesian optimization) a végső finomhangoláshoz

## Hivatkozások

- Kolmogorov, A. N. (1957). On the representation of continuous functions. *Doklady*.
- Hecht-Nielsen, R. (1987). Kolmogorov's mapping neural network existence theorem. *IEEE ICNN*.
- Lippmann, R. P. (1987). An Introduction to Computing with Neural Nets. *IEEE ASSP Magazine*.
- Baum, E. B. & Haussler, D. (1989). What size net gives valid generalization? *NeurIPS*.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *MCSS*.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*.
