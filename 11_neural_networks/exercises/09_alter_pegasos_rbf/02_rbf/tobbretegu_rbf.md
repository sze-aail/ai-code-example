# Többrétegű RBF neurális háló backpropagation-nel

**Neurális hálók I. — Hajdu Csaba**

## Mi az RBF neuron?

A klasszikus (MLP) neuron **belső szorzatot** számol: `σ(⟨w, x⟩ + b)` — megméri, mennyire "hasonlít" a bemenet egy irányhoz.

Az **RBF neuron** ezzel szemben **távolságot** mér: `φ(||x - c|| / σ)` — megméri, mennyire "közel" van a bemenet egy tanult centrumhoz.

```
MLP neuron:   h(x) = σ(wᵀx + b)           ← irányérzékeny (félsík)
RBF neuron:   h(x) = exp(-||x - c||² / 2σ²)  ← távolságérzékeny (Gauss-buborék)
```

## Miért érdekes a többrétegű RBF?

A klasszikus RBF hálózat (Broomhead & Lowe, 1988; Moody & Darken, 1989) **egyrétegű**: RBF réteg + lineáris kimenet. Ez a legtöbb tankönyv változata.

De mi történik, ha **több RBF réteget** egymásra rakunk? A kísérletünk megmutatja:

| Architektúra | Spirál acc | Paraméterek | Megjegyzés |
|---|---|---|---|
| MLP [2→32→32→1] | 100.0% | 1185 | Referencia |
| RBF 1 réteg [2→(32)→1] | 99.4% | 129 | Küszködik a spirállal |
| RBF 2 réteg [2→(24)→(16)→1] | 100.0% | **489** | Megoldja kevesebb paraméterrel! |
| RBF 3 réteg [2→(20)→(16)→(8)→1] | 100.0% | 541 | Szintén megoldja |

A kulcs: **a mélység az RBF hálóknál is segít**, nem csak az MLP-knél. A második RBF réteg az első réteg "távolság-reprezentációjából" további távolság-mintákat tanul — hierarchikus lokalizáció.

## Tanulható paraméterek

Minden RBF rétegben 3 paraméter-csoport van, mind backproppal tanulható:

| Paraméter | Jelölés | Szerep | Méret |
|---|---|---|---|
| Centrumok | `c_j` | A Gauss-buborék közepe | (n_centers, d) |
| Szélességek | `σ_j` | A buborék mérete (log-skálán tárolva) | (n_centers,) |
| Kimeneti súlyok | `W` | A lineáris kombináció súlyai | (n_centers, out) |

A `log(σ)` trükk biztosítja, hogy σ mindig pozitív maradjon: `σ = exp(log_σ)`.

## Távolságmetrikák cserélhetősége

A kód 5 különböző differenciálható távolságmetrikát implementál:

### L2 (euklideszi) — alapértelmezett
```
d²(x, c) = Σ(xᵢ - cᵢ)²
```
Izotróp buborékok. Az RBF hálók klasszikus választása.

### L1 (smooth Manhattan)
```
d(x, c) = Σ √((xᵢ - cᵢ)² + ε)
```
Rombusz alakú hatókör. A sima approximáció (ε > 0) biztosítja a differenciálhatóságot.

### L∞ (smooth Csebisev)
```
d(x, c) = logsumexp(α · |xᵢ - cᵢ|) / α
```
Négyzet alakú hatókör. A logsumexp a max() sima közelítése.

### Koszinusz
```
d(x, c) = 1 - cos(x, c)
```
Szögtávolság — az irány számít, nem a nagyság. NLP-ben és ajánlórendszereknél gyakori.

### Mahalanobis
```
d²(x, c) = (x - c)ᵀ M (x - c),   M = LLᵀ
```
Tanulható metrikus tenzor! Anizotróp buborékok: a centroid "különböző irányokba" különböző mértékben érzékeny. Az L mátrix alsó háromszög-felbontás (Cholesky), ami garantálja M pozitív definitségét.

## Centrumok mozgása

Az 5. kísérlet vizualizálja, hogyan mozognak a centrumok (★) és hogyan változnak a szélességek (σ, szaggatott körök) tanítás közben:

- **Epoch 0**: véletlenszerű pozíciók, egyforma σ
- **Epoch 50**: kezdenek a döntési határ felé húzódni
- **Epoch 200**: a centrumok a két osztály határvonalára koncentrálódnak
- **Epoch 500**: stabil konfiguráció, a σ-k az adatsűrűséghez igazodnak

Ez a backpropagation ereje: a centrumok **nem rögzítettek** (mint a k-means inicializálásban), hanem végig-végig optimalizálódnak a feladathoz.

## Kapcsolat más módszerekhez

```
                    Perceptron (belső szorzat)
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       MLP           Kernel           RBF háló
   (mély, ⟨w,x⟩)   (implicit φ)   (explicit φ, távolság)
          │              │              │
          ▼              ▼              ▼
    Transzformer    SVM / GP      Többrétegű RBF
   (attention)    (Gauss-foly.)   (ez a kód!)
```

## Mikor érdemes RBF-et használni?

| Szempont | RBF háló | MLP |
|---|---|---|
| Lokalizáció | Természetes (Gauss-buborékok) | Globális (félsík döntések) |
| Interpoláció | Kiváló (radiális bázisfüggvények) | Jó |
| Extrapolaráció | Gyenge (minden buborékon kívül → 0) | Jobb |
| Interpretálhatóság | Jobb (centrumok = prototípusok) | Gyengébb |
| Skálázhatóság | Korlátos (O(n·C) per réteg) | Jobb (mátrixszorzás) |
| Fizikai modellek | Kiváló (RBF a PDE-megoldók alapja) | Jó (PINN) |

## Hivatkozások

- Broomhead, D. & Lowe, D. (1988). Radial Basis Functions, Multi-Variable Functional Interpolation and Adaptive Networks.
- Moody, J. & Darken, C. (1989). Fast Learning in Networks of Locally-Tuned Processing Units.
- Park, J. & Sandberg, I. (1991). Universal Approximation Using Radial-Basis-Function Networks.
- Schwenker, F., Kestler, H. & Palm, G. (2001). Three learning phases for radial-basis-function networks.
- Que, Q. & Belkin, M. (2016). Back to the Future: Radial Basis Function Networks Revisited.
