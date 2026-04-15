# Forgási invariancia és szimmetriacsoportok CNN-ekben

**Neurális hálók II. (CNN) — Hajdu Csaba**

## A probléma

A standard CNN **eltolás-ekviváriáns** (a konvolúció természetéből), de **NEM forgás-invariáns**. Egy 30°-kal elforgatott számjegy teljesen más predikciót adhat.

```
Ekvivariancia:  f(T·x) = T·f(x)   (a kimenet együtt transzformálódik a bemenettel)
Invariancia:    f(T·x) = f(x)      (a kimenet NEM változik a transzformációval)
```

A CNN-nek az eltolás-ekvivariancia "ingyen" jön (súlymegosztás + konvolúció). A forgás-ekvivariancia NEM — ezt külön kell biztosítani.

## Szimmetriacsoportok

| Csoport | Elemek | Leírás |
|---|---|---|
| **C1** | {e} | Triviális — standard CNN |
| **C4** | {0°, 90°, 180°, 270°} | Ciklikus, 4 forgás |
| **C8** | {0°, 45°, ..., 315°} | Ciklikus, 8 forgás |
| **D4 (p4m)** | C4 × {id, tükrözés} = 8 elem | Diéderes, forgás + tükrözés |
| **SO(2)** | Folytonos forgás [0°, 360°) | Teljes forgáscsoport |
| **SE(2)** | SO(2) × ℝ² | Forgás + eltolás |

## Megközelítések

### 1. Data augmentation (ad hoc)
Tanítás közben véletlenszerűen forgatjuk a képeket. **Nem garantálja** az invarianciát, csak közelíti. Több adat kell, lassabb konvergencia.

### 2. Group Equivariant CNN (Cohen & Welling, 2016)
A szűrőt nem egyszer, hanem a **csoport összes elemével** alkalmazzuk:

```python
for k in range(4):  # C4 csoport
    rotated_filter = rot90(base_filter, k)
    output_k = conv2d(input, rotated_filter)
```

Kulcs: **nem tanulunk több szűrőt** — a csoport generálja az összeset! Paraméterszám = standard CNN.

A csoport feletti max-pool invarianciát ad, sum-pool ekvivarienciát.

### 3. Steerable CNN (Weiler & Cesa, 2019)
A szűrőket **kör-harmonikus bázisfüggvényekből** építjük fel:

```
ψ(r, θ) = R(r) · e^{imθ} = R(r) · (cos(mθ) + i·sin(mθ))
```

Bármely forgatás = bázisfüggvények **lineáris kombinációja**. Tehát a forgatás nem kell, hogy explicit legyen — a bázis-együtthatók transzformálása elég.

Az `e2cnn` Python könyvtár implementálja: `pip install e2cnn`.

### 4. Harmonic Networks (Worrall et al., 2017)
Komplex szűrők, amelyek a kör-harmonikusok m-edik rendjét használják. A forgatás fázis-eltolásként jelenik meg a komplex válaszban.

### 5. Capsule Networks (Sabour & Hinton, 2017)
Nem csak a jellemző *jelenlétét*, hanem a *pózát* is kódolják (pozíció, orientáció, skála). A routing-by-agreement mechanizmus a részt-egész kapcsolatot tanulja.

## Kísérleti eredmények

Digits adathalmaz, tanítás 0°-os képeken, tesztelés forgatott képeken:

| Módszer | 0° | 90° | 180° | Átlag |
|---|---|---|---|---|
| Standard CNN | 64% | 11% | 17% | 22% |
| + 90° augmentáció | 36% | 36% | 27% | 29% |
| G-CNN (C4) | 62% | **46%** | **39%** | **37%** |

A G-CNN a 90° és 180° teszten egyértelműen jobb, mert a C4 csoport ezeket a szimmetriákat tartalmazza. A 45°-on mindhárom gyenge — ehhez C8 vagy SO(2) kellene.

## Kapcsolat a G-SPHF keretrendszerhez

A szimmetriacsoport-ekvivariancia természetesen kapcsolódik a hipergráf-formalizmushoz:

- A **csoport hatása** a szűrőn = **csúcsfüggvény-transzformáció** a számítási gráfban
- A G-CNN szűrő-orbita = egy hipergráf csúcs összes **szimmetrikus vetülete**
- Az **incidencia-struktúra** (melyik szűrő melyik bemenetre hat) invariáns a csoport-hatás alatt

## Hivatkozások

- Cohen, T. & Welling, M. (2016). Group Equivariant Convolutional Networks. *ICML*.
- Worrall, D. et al. (2017). Harmonic Networks: Deep Translation and Rotation Equivariance. *CVPR*.
- Weiler, M. & Cesa, G. (2019). General E(2)-Equivariant Steerable CNNs. *NeurIPS*.
- Sabour, S., Frosst, N. & Hinton, G. (2017). Dynamic Routing Between Capsules. *NeurIPS*.
- Bronstein, M. et al. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. *arXiv*.
- Serre, J.-P. (1977). Linear Representations of Finite Groups. Springer.
