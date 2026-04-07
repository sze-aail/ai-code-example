# Neurális hálók I. — Példakódok

**Hajdu Csaba — Mesterséges Intelligencia kurzus**

## Fájlok

| # | Fájl | Téma | Kapcsolódó dia |
|---|------|------|----------------|
| 01 | `01_perceptron.py` | Perceptron nulláról (AND, OR, XOR) | 14–21 |
| 01 | `01_perceptron_learning_animation.py` | Perceptron tanulásának bemutatása animált GIF formában az AND kapun (*új*) | 14–21 |
| 02 | `02_xor_mlp.py` | XOR megoldása MLP-vel (PyTorch) | 20–24 |
| 03 | `03_activation_functions.py` | Aktivációs fv.-ek + vanishing gradient | 18, 33, 40 |
| 04 | `04_gradient_descent.py` | Gradiens-ereszkedés, learning rate, saddle point | 41 |
| 05 | `05_backpropagation.py` | Backprop kézzel vs. PyTorch autograd | 29–31 |
| 06 | `06_overfitting_regularization.py` | Overfitting, L2, Dropout, Early stopping | 43–44 |
| 07 | `07_optimizers.py` | SGD → Momentum → RMSProp → Adam | 42 |
| 08 | `08_hebb_rule.py` | Hebb- és Oja-szabály, PCA kinyerés | 39 |
| 09 | `09_iris_classification.py` | Iris MLP + BatchNorm + konfúziós mátrix | 37, 45 |
| 10 | `10_regression_mlp.py` | Univerzális approximáció, kapacitás | 25, 36 |
| 11 | `11_polynomial_fitting.py` | Alul- és túlillesztés szemléltetése polinom regresszióval (*új*) | |

## Követelmények

```bash
pip install torch numpy matplotlib scikit-learn
```

## Futtatás

```bash
python 01_perceptron.py
python 01_perceptron_learning_animation.py
python 02_xor_mlp.py
python 11_polynomial_fitting.py
# ... stb.
```

Minden script PNG ábrákat generál az aktuális mappába.
