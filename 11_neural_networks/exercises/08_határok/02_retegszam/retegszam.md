# Rétegszám geometriai szabálya

**Neurális hálók I. — Hajdu Csaba**

## A szabály

| Rejtett rétegek | Döntési régió típusa | Geometriai leírás |
|---|---|---|
| **0** | félsík | egyetlen hipersík: `w₁x₁ + w₂x₂ + b > 0` |
| **1** (n neuron) | konvex régió | n félsík **metszete** (∩) |
| **2** (n×m neuron) | tetszőleges régió | m konvex régió **uniója** (∪) |

Forrás: Lippmann, R. P. (1987). "An Introduction to Computing with Neural Nets." *IEEE ASSP Magazine*.

### Miért?

Egy ReLU (vagy sigmoid) neuron egy **félsíkot** vág ki a bemeneti térből: az egyik oldalon aktív, a másikon nem. Az első rejtett réteg n neuronja n félsíkot definiál, a kimeneti neuron ezek **metszetét** veszi (AND-művelet) — ez konvex. A második rejtett réteg több ilyen konvex régiót kombinál (OR-művelet) — ez tetszőleges alakzat.

## Kapcsolódó tételek

### Cybenko (1989): Univerzális approximációs tétel
Egyetlen rejtett réteges háló (sigmoid aktiváció) tetszőleges folytonos függvényt közelíthet — de **exponenciálisan** sok neuront igényelhet.

### Hornik (1991): Kiterjesztett univerzális approximáció
Bármely mérhető függvény közelíthető egyetlen rejtett réteggel. Az aktivációs függvény nem kell sigmoid legyen — bármely nem-konstans, korlátos, monoton növekedő folytonos függvény alkalmas.

### Kolmogorov (1957): Szuperpozíciós tétel
Bármely n-változós folytonos függvény kifejezhető:

```
f(x₁, ..., xₙ) = Σⱼ gⱼ(Σᵢ φᵢⱼ(xᵢ))
```

ahol gⱼ és φᵢⱼ egyváltozós folytonos függvények, és mindössze **2n+1** belső tag szükséges. Ez a neurális hálók elméleti alapja, de a gyakorlatban az φ függvények nem konstruktívak.

### Telgarsky (2016): Mélység-szeparációs tétel
Léteznek olyan függvények, amelyeket O(k) mélységű ReLU háló polinomiális szélességgel kifejez, de O(k−1) mélységű hálónak **exponenciális** szélességre van szüksége. Ez formálisan igazolja, hogy a mélység nem pusztán kényelmi kérdés.

## Mélység vs. szélesség — kísérleti eredmények

Spirál adathalmazon (nehéz nemlineáris feladat):

| Architektúra | Paraméterek | Pontosság | Megjegyzés |
|---|---|---|---|
| [2→200→1] sekély-széles | 801 | 100% | Sok neuron, de egy réteg elég |
| [2→32→32→1] közepes | 1185 | 100% | Klasszikus választás |
| [2→16→16→16→1] mély-keskeny | 609 | 100% | **Legkevesebb param 100%-kal** |
| [2→8→8→8→8→8→1] nagyon mély | 321 | 91.8% | Túl keskeny → információveszítés |

A tanulság: a mélység hatékonyabb a szélességnél, de van egy alsó korlát — ha egy réteg túl keskeny (bottleneck), az információ nem jut át.

## A gyűrű-probléma érdekessége

Az 1 rejtett réteges háló 99%-ot ér el a gyűrűn, holott az nem konvex. Miért?

A ReLU aktiváció **darabonként lineáris** határt hoz létre. Elég neuronnal a konvex korlát megkerülhető: sok kis szegmensből közelíthető a görbe határ. A szabály tehát **elméleti worst-case** — a gyakorlatban a ReLU hálók rugalmasabbak, mint amit a tétel garantál.

## Hivatkozások

- Lippmann, R. P. (1987). An Introduction to Computing with Neural Nets. *IEEE ASSP Magazine*.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*.
- Kolmogorov, A. N. (1957). On the representation of continuous functions. *Doklady Akademii Nauk*.
- Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT*.
- Eldan, R. & Shamir, O. (2016). The power of depth for feedforward neural networks. *COLT*.
