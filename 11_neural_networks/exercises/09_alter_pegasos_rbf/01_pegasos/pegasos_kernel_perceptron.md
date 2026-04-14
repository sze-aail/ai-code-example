# Perceptron kiterjesztések: Pegasos, távolságmetrikák, kernel perceptron

**Neurális hálók I. — Hajdu Csaba**

## Motiváció

A klasszikus perceptron három korlátja:

1. **Bármely** szétválasztó síkot megtalálja, nem az optimálisat (nincs margin-maximalizálás)
2. Csak **lineárisan szeparálható** problémákat old meg
3. A távolság fogalma rögzített (euklideszi belső szorzat)

Ez a három irány — Pegasos, távolságmetrikák, kernel perceptron — mindhárom korlátot feloldja.

---

## 1. Pegasos: a perceptrontól az SVM-ig

### Mi a Pegasos?

**Primal Estimated sub-GrAdient SOlver for SVM** (Shalev-Shwartz, Singer & Srebro, 2007).

Lényegében egy online SVM: a perceptron frissítési ciklusát kiegészíti regularizációval és hinge loss-szal.

### Frissítési szabály

A *t*-edik lépésben:

```
η_t = 1 / (λ · t)                          ← csökkenő learning rate
w_{t+1} = (1 - η_t·λ)·w_t + η_t·y_i·x_i   ha y_i·⟨w_t, x_i⟩ < 1
w_{t+1} = (1 - η_t·λ)·w_t                   egyébként
```

### Mi a különbség a perceptronhoz képest?

| | Perceptron | Pegasos |
|---|---|---|
| Frissítés feltétele | `y·⟨w,x⟩ ≤ 0` (hibás) | `y·⟨w,x⟩ < 1` (marginon belül) |
| Regularizáció | nincs | `(1 - η·λ)·w` (L2 weight decay) |
| Learning rate | fix | csökkenő: `1/(λ·t)` |
| Célfüggvény | nincs explicit | hinge loss + L2 |
| Konvergencia | (R/γ)² frissítés | O(1/(λ·t)) |
| Eredmény | bármely szétválasztó sík | **margin-maximalizáló** sík |

### Kapcsolat a Novikoff-tételhez

A Novikoff-tétel azt mondja, hogy a perceptron konvergál, de a talált sík nem optimális. A Pegasos explicit módon a **maximális marginú** síkot keresi — pont azt, ami a legjobb generalizációt adja (VC-elmélet).

---

## 2. Távolságmetrikák

### Miért számít a távolság?

A perceptron (és az SVM) döntése a `⟨w, x⟩` belső szorzaton alapul, ami az euklideszi geometriát feltételezi. De nem minden adatnál ez a természetes metrika.

### Összehasonlítás

| Metrika | Képlet | Egységkör alakja | Mikor használjuk |
|---|---|---|---|
| **Euklideszi (L2)** | `√(Σ(xᵢ-yᵢ)²)` | kör | alapértelmezett, izotróp adatok |
| **Manhattan (L1)** | `Σ|xᵢ-yᵢ|` | rombusz (45°-ban forgatott négyzet) | ritka jellemzők, grid-távolság |
| **Csebisev (L∞)** | `max|xᵢ-yᵢ|` | négyzet | sakktábla-távolság, worst-case |
| **Koszinusz** | `1 - cos(θ)` | szög alapú (nem norma!) | szöveg/NLP, irány számít |
| **Mahalanobis** | `√((x-y)ᵀΣ⁻¹(x-y))` | ellipszis | korrelált jellemzők |

### Kapcsolat az RBF kernelhez

Az RBF kernel pont az euklideszi távolságra épül:

```
k(x, y) = exp(-γ · ||x - y||²)
```

A γ paraméter azt szabályozza, milyen "messzire lát" a kernel: kis γ → tág hatókör (sima határ), nagy γ → szűk hatókör (lokális döntés).

---

## 3. Kernel perceptron és RBF kiterjesztés

### A kernel-trükk

A perceptron döntése: `sign(Σ αᵢ · yᵢ · ⟨xᵢ, x⟩)`

A **kernel-trükk**: cseréljük ki a belső szorzatot egy kernel függvénnyel:

```
sign(Σ αᵢ · yᵢ · k(xᵢ, x))
```

Ez **implicit módon** magasabb dimenziós térbe vetíti az adatot anélkül, hogy kiszámolnánk a vetítést!

### Kernelek

| Kernel | Képlet | Implicit dimenzió |
|---|---|---|
| Lineáris | `⟨x, y⟩` | d (változatlan) |
| Polinom (p-ed fokú) | `(1 + ⟨x, y⟩)ᵖ` | O(dᵖ) |
| RBF / Gauss | `exp(-γ·\|\|x-y\|\|²)` | **végtelen!** |

### Miért oldja meg az XOR-t?

Az RBF kernel minden tanítópont köré egy "Gauss-dombot" helyez. A döntés lokális: egy új pont aszerint kap címkét, hogy mely tanítópontokhoz van közel. Ez elég az XOR-hoz (és bármely nemlineáris problémához, ha γ elég nagy).

### γ hatása (bias-variancia tradeoff)

| γ | Hatás | Analógia |
|---|---|---|
| Kicsi (0.1) | Sima döntési határ, távoli pontok is számítanak | Underfitting |
| Közepes (1–5) | Jó kompromisszum | Optimális |
| Nagy (50+) | Minden pont körül szűk "buborék" | Overfitting |

Ez ugyanaz a bias-variancia tradeoff, amit az MLP-nél a rejtett neuronok számával (vagy regularizációval) szabályozunk.

### Support vektorok

A kernel perceptron tanítása után azok a pontok a **support vektorok**, amelyeknél `αᵢ > 0` — vagyis amelyeken a perceptron hibázott és frissített. Ezek a döntési határ "tartóoszlopai". Az ábrán sárga körrel jelölve.

---

## A fejlődési ív összefoglalása

```
Perceptron (1958, Rosenblatt)
    │  + regularizáció + hinge loss + csökkenő η
    ▼
Pegasos / Online SVM (2007, Shalev-Shwartz)
    │  + kernel-trükk: ⟨x,y⟩ → k(x,y)
    ▼
Kernel perceptron / Kernel SVM
    │  + RBF kernel: végtelen dimenziós vetítés
    ▼
RBF Kernel Perceptron
    → nemlineáris döntési határ
    → XOR megoldható rejtett réteg NÉLKÜL
    → de: O(n) predikciós idő (minden SV-t végig kell nézni)
```

### Miért nem elég a kernel perceptron?

A kernel módszerek hátránya, hogy a predikciós idő **O(n·d)** (n = support vektorok száma), míg az MLP-é **O(W)** (W = súlyok száma, fix). Nagy adatnál az MLP gyorsabb. Ezért a neurális hálók "nyertek" a gyakorlatban — de a kernel elmélet alapvető fontosságú a megértéshez.

---

## Hivatkozások

- Rosenblatt, F. (1958). The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain.
- Novikoff, A. (1962). On convergence proofs on perceptrons.
- Aizerman, M., Braverman, E. & Rozonoér, L. (1964). Theoretical foundations of the potential function method (kernel módszer eredete).
- Shalev-Shwartz, S., Singer, Y. & Srebro, N. (2007). Pegasos: Primal Estimated sub-GrAdient SOlver for SVM.
- Schölkopf, B. & Smola, A. (2002). Learning with Kernels.
