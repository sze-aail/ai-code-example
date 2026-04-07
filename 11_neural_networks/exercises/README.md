# Neurális hálók I. — Teljes kódgyűjtemény

**Hajdu Csaba — Mesterséges Intelligencia kurzus**

Összeszervezett kódgyűjtemény: 41 Python script (17 statikus demó + 20 animáció + 4 haladó téma).

## Telepítés

```bash
pip install torch numpy matplotlib scikit-learn Pillow
```

## Mappaszerkezet

```
neural_halok_I/
├── 01_alapok/                                    # Előismeretek
│   ├── vesztesgfuggvenyek_3d.py                  # MSE/Huber/BCE 3D + outlier         [dia 18]
│   ├── normalis_eloszlas.py                       # Normális eloszlás + Xavier/He       [dia 26]
│   └── logisztikus_regresszio.py                  # Logisztikus regr. + sigmoid         [dia 16]
│
├── 02_perceptron/                                # Perceptron
│   ├── perceptron_nullarol.py                     # AND/OR/XOR nulláról                 [dia 14-21]
│   ├── perceptron_make_blobs.py                   # 1000 minta + 3D döntési sík         [dia 14-17]
│   └── animaciok/
│       └── 01_perceptron_learning_animation.py    # GIF: tanulás lépésről lépésre
│
├── 03_mlp/                                       # Többrétegű hálók
│   ├── xor_megoldas.py                            # XOR + rejtett réteg vizualizáció    [dia 20-24]
│   ├── osztalyozas_make_moons.py                  # Félhold, kapacitás hatása            [dia 23]
│   ├── iris_osztalyozas.py                        # Iris + BatchNorm összehasonlítás     [dia 37, 45]
│   ├── regresszio_approximacio.py                 # Univerzális approximáció              [dia 25, 36]
│   ├── regresszio_california.py                   # California Housing valós adat        [dia 36]
│   └── animaciok/
│       ├── 02_xor_learning_animation.py           # GIF: XOR döntési határ tanulása
│       ├── 02_xor_mlp_graph_viz.py                # PNG: MLP számítási gráf
│       └── 13_large_nn_classification_animation.py# GIF: nagy háló döntési határ
│
├── 04_tanulas/                                   # Tanulási algoritmusok
│   ├── aktivacios_fuggvenyek.py                   # 8 fv. + deriváltak + vanishing      [dia 18,33,40]
│   ├── gradiens_ereszkedas.py                     # LR hatás, saddle point, variánsok   [dia 41]
│   ├── backpropagation.py                         # Kézi backprop + autograd             [dia 29-31]
│   ├── optimalizalok.py                           # SGD->Adam Rosenbrock-felületen       [dia 42]
│   └── animaciok/
│       ├── 04_gradient_descent_animation.py       # PNG: GD learning rate hatás
│       ├── 04_nn_lr_comparison_animation.py       # GIF: LR összehasonlítás hálón
│       ├── 04_nn_wrong_learning_animation.py      # GIF: rossz LR hatása tanításra
│       ├── 05_backpropagation_animation.py        # GIF: backprop vizualizáció
│       ├── 05_weight_tensor_animation.py          # GIF: súlytenzor változás
│       ├── 05_weight_tensor_animation_large.py    # GIF: nagy háló súlyváltozás
│       ├── 07_optimizers_animation.py             # GIF: optimalizálók 2D-ben
│       └── 07_optimizers_rosenbrock.py            # GIF: optimalizálók Rosenbrock-on
│
├── 05_regularizacio/                             # Regularizáció
│   ├── overfitting_regularizacio.py               # L2, Dropout, Early stopping         [dia 43-44]
│   ├── 11_polynomial_fitting.py                   # Polinom: under/over/jó illesztés     [dia 43]
│   └── animaciok/
│       ├── 06_l2_lambda_comparison_animation.py   # GIF: L2 lambda hatása
│       ├── 06_regularization_comparison_animation.py # GIF: regularizáció módszerek
│       ├── 10_huber_vs_mse_animation.py           # GIF: Huber vs MSE outlierrel
│       └── 10_regression_wrong_learning_animation.py # GIF: rossz tanulás regresszión
│
├── 06_kiegeszito/                                # Kiegészítő témák
│   ├── hebb_oja.py                                # Hebb + Oja (PCA kinyerés)           [dia 39]
│   ├── pca_dimenziocsokkentas.py                  # PCA Iris-en + scree plot
│   └── animaciok/
│       ├── 08_hebb_learning_animation.py          # GIF: Hebb-tanulás lépései
│       ├── 09_batchnorm_animation.py              # GIF: BatchNorm hatása
│       └── 09_normalization_methods_animation.py  # GIF: BN/LN/GN/IN összehasonlítás
│
├── 07_halado/                                    # Haladó / modern irányok
│   ├── 12_large_nn_training_animation.py          # GIF: 20k pontos regresszió          [dia 48-50]
│   ├── 14_spiking_neural_network_animation.py     # GIF: LIF spiking neuron             [dia 12, 49]
│   ├── 15_simple_diffusion_animation.py           # GIF: diffúziós modell alapok         [dia 49]
│   └── 16_nn_architecture_comparison_animation.py # GIF: architektúrák összehasonlítás  [dia 48]
│
├── abrak/                                        # 31 generált PNG (statikus demók)
└── README.md
```

## Ajánlott sorrend

### Kezdőknek
1. `01_alapok/normalis_eloszlas.py`
2. `01_alapok/vesztesgfuggvenyek_3d.py`
3. `02_perceptron/perceptron_nullarol.py` + animáció
4. `03_mlp/xor_megoldas.py` + animáció
5. `04_tanulas/backpropagation.py` + animáció
6. `04_tanulas/gradiens_ereszkedas.py`

### Haladóknak
- `04_tanulas/animaciok/07_optimizers_rosenbrock.py` — optimalizálók versenye
- `05_regularizacio/animaciok/06_regularization_comparison_animation.py`
- `06_kiegeszito/animaciok/09_normalization_methods_animation.py`

### Modern irányok (diák 48-50)
- `07_halado/14_spiking_neural_network_animation.py` — SNN / LIF neuron
- `07_halado/15_simple_diffusion_animation.py` — diffúziós modell
- `07_halado/16_nn_architecture_comparison_animation.py` — architektúrák

## Futtatás

Statikus scriptek PNG ábrákat generálnak, animációk GIF-et:
```bash
python 02_perceptron/perceptron_nullarol.py          # -> PNG
python 02_perceptron/animaciok/01_perceptron_learning_animation.py  # -> GIF
```
