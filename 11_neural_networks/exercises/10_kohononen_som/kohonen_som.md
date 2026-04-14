# Kohonen SOM és felügyeletlen tanulás összehasonlítás

**Neurális hálók I. — Hajdu Csaba**

## Mi a Kohonen-háló?

A **Self-Organizing Map** (Kohonen, 1982) egy felügyeletlen neurális háló, amely magas dimenziós adatot egy alacsony dimenziós (tipikusan 2D) rácson szervez el, megőrizve a topológiát: ami az adattérben közel van, az a rácson is közel lesz.

### Tanulási mechanizmus: versengő tanulás

```
1. Bemenet: x
2. Verseny: BMU = argmin_j ||x - w_j||  (legközelebbi neuron nyer)
3. Kooperáció: a BMU szomszédai is frissülnek (Gauss szomszédság)
4. Adaptáció: w_j ← w_j + η · h(j, BMU) · (x - w_j)
```

Ez **nem backpropagation** — nincs hibafüggvény, nincs gradiens. A tanulás lokális és versenyalapú, biológiailag plauzibilisebb, mint a backprop.

### Hiperparaméterek és ütemezés

| Paraméter | Kezdőérték | Változás | Hatás |
|---|---|---|---|
| η (learning rate) | 0.5 | exponenciális csökkenés | Kezdetben nagy lépések, később finomhangolás |
| σ (szomszédság sugár) | rácsméret/2 | exponenciális csökkenés | Kezdetben globális szerveződés, később lokális |
| Rács méret | √(5×√N) × √(5×√N) | fix | Túl kicsi → alulreprezentáció, túl nagy → üres neuronok |

A tanulásnak két fázisa van: **szervező fázis** (nagy η, nagy σ → globális topológia) és **konvergencia fázis** (kis η, kis σ → lokális finomhangolás).

## U-mátrix

Az **Unified Distance Matrix** a szomszédos neuronok közötti átlagos távolság. Sötét területek = nagy ugrás a súlyokban = klaszterhatár. Világos területek = homogén régió = klaszter belseje.

Ez a SOM egyedülálló előnye: a klaszterstruktúra vizualizálható anélkül, hogy explicit klaszterszámot adnánk meg.

## Összehasonlítás: SOM vs. DBSCAN vs. k-Means

### Silhouette score az adathalmazokon

| Adathalmaz | SOM (6×6) | DBSCAN (ε=0.3) | k-Means |
|---|---|---|---|
| Klaszterek (gömb) | 0.27 | **0.84** | **0.84** |
| Félholdak | 0.31 | 0.38 | **0.49** |
| Gyűrűk | **0.34** | 0.16 | 0.33 |
| Anizotróp | 0.24 | 0.69 | **0.77** |

A SOM silhouette score-ja alacsonyabb, mert 36 (6×6) mikroklasztert csinál a valódi 2–4 helyett. A SOM ereje nem a klaszterezés, hanem a **topológia-megőrző vizualizáció**.

### Mikor melyik?

| Szempont | SOM | DBSCAN | k-Means |
|---|---|---|---|
| **Cél** | Vizualizáció, topológia | Tetszőleges alakú klaszterek | Gömb-klaszterek |
| **Klaszterszám** | Rács → implicit | Automatikus | Előre kell tudni |
| **Zaj** | Nincs explicit kezelés | Automatikus zajszűrés | Nincs |
| **Magas dimenzió** | Kiváló (dimenziócsökkentés!) | Curse of dimensionality | OK |
| **Online tanulás** | Igen (inkrementális) | Nem | Nem (de van mini-batch) |
| **Skálázhatóság** | O(N × G × epochs) | O(N × log N) | O(N × K × iters) |

### Klasszikus alkalmazások

- **SOM**: génexpresszió vizualizáció, szövegtopológia, ipari folyamat-monitoring
- **DBSCAN**: térinformatika (GPS klaszterek), anomália-detekció, képszegmentálás
- **k-Means**: ügyfélszegmentáció, képtömörítés (VQ), feature learning

## Kapcsolat a Hebb-szabályhoz

A SOM tanulási szabálya Hebb-rokon: a BMU és a bemenet „együtt aktívak", tehát a köztük lévő kapcsolat erősödik (a súly a bemenet felé mozdul). A szomszédsági függvény ezt kiterjeszti a topológiai szomszédokra — ez a **laterális gátlás** neurobiológiai modellje.

```
Hebb:     Δw = η · x · y         (korreláció)
Oja:      Δw = η · y · (x - y·w) (normalizált Hebb)
Kohonen:  Δw = η · h · (x - w)   (versengő + szomszédság)
```

## Hivatkozások

- Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. *Biological Cybernetics*.
- Kohonen, T. (2001). *Self-Organizing Maps.* 3rd edition, Springer.
- Ester, M. et al. (1996). A density-based algorithm for discovering clusters (DBSCAN). *KDD*.
- Vesanto, J. & Alhoniemi, E. (2000). Clustering of the Self-Organizing Map. *IEEE TNN*.
