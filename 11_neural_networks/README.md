# Neurális hálók I. — Példakódok

**Hajdu Csaba — Mesterséges Intelligencia kurzus**

## Fájlok

| # | Fájl | Téma | Kapcsolódó dia |
|---|------|------|----------------|
| 01 | `01_perceptron.py` | Perceptron nulláról (AND, OR, XOR) | 14–21 |
| 01 | `01_perceptron_learning_animation.py` | Perceptron tanulásának bemutatása animált GIF formában az AND kapun (*új*) | 14–21 |
| 02 | `02_xor_mlp.py` | XOR megoldása MLP-vel (PyTorch) | 20–24 |
| 02 | `02_xor_learning_animation.py` | XOR probléma tanulásának bemutatása MLP-vel, animált döntési hátárvonallal (*új*) | 20–24 |
| 02 | `02_xor_mlp_graph_viz.py` | Tanított XOR MLP súlyainak hálózati gráf-alapú vizualizációja (*új*) | 20–24 |
| 03 | `03_activation_functions.py` | Aktivációs fv.-ek + vanishing gradient | 18, 33, 40 |
| 04 | `04_gradient_descent.py` | Gradiens-ereszkedés, learning rate, saddle point | 41 |
| 04 | `04_gradient_descent_animation.py` | Gradiens-ereszkedés iterációinak animálása 2D felületen (*új*) | 41 |
| 04 | `04_nn_lr_comparison_animation.py` | Neurális háló (MLP) betanulási folyamatának animációja különböző tanulási rátákkal (*új*) | 41 |
| 04 | `04_nn_wrong_learning_animation.py` | Szándékosan elrontott tanulási folyamatok animációja (Rossz irány, Lineáris XOR, Túl nagy LR) (*új*) | 41 |
| 05 | `05_backpropagation.py` | Backprop kézzel vs. PyTorch autograd | 29–31 |
| 05 | `05_backpropagation_animation.py` | A backpropagation algoritmus lépéseinek számítási gráfos animációja (Előreterjesztés és Visszaterjesztés) (*új*) | 29–31 |
| 05 | `05_weight_tensor_animation.py` | Neurális háló (Autoencoder) súlymátrixainak és gradienseinek interaktív változása backpropagation hatására (Tenzió nézet) (*új*) | 29–31 |
| 05 | `05_weight_tensor_animation_large.py` | Nagyobb neurális háló súlymátrixainak animációja (Tenzió nézet) (*új*) | 29–31 |
| 06 | `06_overfitting_regularization.py` | Overfitting, L2, Dropout, Early stopping | 43–44 |
| 06 | `06_regularization_comparison_animation.py` | Animáció a L2 és Dropout regularizációs módszerek hatásának összehasonlításáról az overfitting elkerülésében (*új*) | 43–44 |
| 06 | `06_l2_lambda_comparison_animation.py` | Különböző L2 regularizációs (Lambda) paraméterek hatásának animált összehasonlítása túlillesztésre hajlamos modellen (*új*) | 43–44 |
| 07 | `07_optimizers.py` | SGD → Momentum → RMSProp → Adam | 42 |
| 07 | `07_optimizers_animation.py` | Különböző optimalizálók (SGD, Momentum, RMSprop, Adam) összehasonlítása animációban (*új*) | 42 |
| 07 | `07_optimizers_rosenbrock.py` | Optimalizálók viselkedése a hírhedt Rosenbrock völgyben (banánfüggvény) animálva (*új*) | 42 |
| 08 | `08_hebb_rule.py` | Hebb- és Oja-szabály, PCA kinyerés | 39 |
| 08 | `08_hebb_learning_animation.py` | Tanítatlan, Hebb (Oja) szabályon alapuló tanulás animációja főkomponens-elemzés (PCA) vizualizálásával (*új*) | 39 |
| 09 | `09_iris_classification.py` | Iris MLP + BatchNorm + konfúziós mátrix | 37, 45 |
| 09 | `09_batchnorm_animation.py` | Batch Normalization hatásának animálása a belső rétegek aktivációs eloszlására (Mély hálózatok stabilizálása) (*új*) | 37, 45 |
| 09 | `09_normalization_methods_animation.py` | Különféle Normalizációs módszerek (Batch, Layer, Instance) hatásának animált összehasonlítása (*új*) | 37, 45 |
| 10 | `10_regression_mlp.py` | Univerzális approximáció, kapacitás | 25, 36 |
| 10 | `10_regression_wrong_learning_animation.py` | Szándékosan felrobbanó regressziós betanítás (túl nagy LR miatt divergáló modell) (*új*) | 25, 36 |
| 10 | `10_huber_vs_mse_animation.py` | Huber Loss vs. MSE Loss összehasonlító animáció kiugró értékekkel (outliers) teli adathalmazon (*új*) | 25, 36 |
| 11 | `11_polynomial_fitting.py` | Alul- és túlillesztés szemléltetése polinom regresszióval (*új*) | |
| 12 | `12_large_nn_training_animation.py` | Nagy neurális háló tanítása nagy adathalmazon (20.000 sor): Becslési precizitás és Hibagörbe animációk (*új*) | |
| 13 | `13_large_nn_classification_animation.py` | Nagy neurális háló tanítása komplex osztályozási feladaton (20.000 adatpont): Döntési határ, Hiba görbék és Súlymátrixok animációja (*új*) | |
| 14 | `14_spiking_neural_network_animation.py` | Biológiailag inspirált Tüzelő Neurális Hálózat (Spiking, LIF modell) szimulációja dinamikus bemenetekkel (*új*) | |
| 15 | `15_simple_diffusion_animation.py` | Nagyon egyszerű Diffúziós Modell animációja: előre (zajosítás) és visszafelé (generálás) folyamatok egy 2D alakon (*új*) | |
| 16 | `16_nn_architecture_comparison_animation.py` | Animáció a különböző neurális háló architektúrák (rétegek és neuronok száma) döntési határára gyakorolt hatásáról (*új*) | |

## Követelmények

```bash
pip install torch numpy matplotlib scikit-learn
```

## Futtatás

```bash
python 01_perceptron.py
python 01_perceptron_learning_animation.py
python 02_xor_mlp.py
python 02_xor_learning_animation.py
python 02_xor_mlp_graph_viz.py
python 04_gradient_descent_animation.py
python 04_nn_lr_comparison_animation.py
python 04_nn_wrong_learning_animation.py
python 05_backpropagation_animation.py
python 05_weight_tensor_animation.py
python 05_weight_tensor_animation_large.py
python 06_l2_lambda_comparison_animation.py
python 06_regularization_comparison_animation.py
python 07_optimizers_animation.py
python 07_optimizers_rosenbrock.py
python 08_hebb_learning_animation.py
python 09_batchnorm_animation.py
python 09_normalization_methods_animation.py
python 10_regression_wrong_learning_animation.py
python 10_huber_vs_mse_animation.py
python 11_polynomial_fitting.py
python 12_large_nn_training_animation.py
python 13_large_nn_classification_animation.py
python 14_spiking_neural_network_animation.py
python 15_simple_diffusion_animation.py
python 16_nn_architecture_comparison_animation.py
# ... stb.
```

Minden script PNG ábrákat generál az aktuális mappába.
