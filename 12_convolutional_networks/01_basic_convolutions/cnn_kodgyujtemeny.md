# Konvolúciós Neurális Hálózatok — Kódgyűjtemény

**Neurális hálók II. (CNN) — Hajdu Csaba**

## Kódok áttekintése

| # | Script | Téma | Kapcsolódó dia |
|---|---|---|---|
| 01 | `01_konvolucio_alapok.py` | 1D/2D konvolúció, 8 szűrő, stride/padding képlet | 13–18 |
| 02 | `02_lenet_szurok.py` | LeNet-5 tanítás, tanult szűrők, feature map-ek, CNN vs MLP | 30, 38 |
| 03 | `03_pooling.py` | MaxPool vs AvgPool vs Strided Conv, invariancia, ismételt pooling | 34–36 |
| 04 | `04_modern_konvoluciok.py` | 1×1 conv, depthwise separable, dilated, transposed conv | (új) |
| 05 | `05_architekturak.py` | Skip connection (ResNet), architektúrák összehasonlítás, receptive field | 46–47 |
| 06 | `06_data_augmentation.py` | 12 augmentációs technika, hatás kevés adaton | (új) |

## Kulcsfogalmak

### Konvolúció kimeneti mérete
```
out = ⌊(in + 2×padding - dilation×(kernel-1) - 1) / stride⌋ + 1
```
Egyszerűsítve (dilation=1): `out = ⌊(in + 2p - k) / s⌋ + 1`

### Receptive field
Az a terület a bemeneti képen, amit egy neuron "lát" egy adott rétegben. Rétegenként nő:
```
RF_l = RF_{l-1} + (k_l - 1) × dilation_l × Π_{i<l} stride_i
```

### Paraméterszám
- Standard Conv2d(in, out, k): `in × out × k² + out`
- 1×1 Conv: `in × out + out` (k²=1)
- Depthwise Separable: `in × k² + in × out + out` (arány: ~1/k² a standardhoz)

## Modern konvolúciós típusok

| Típus | Cél | Kulcs innovació |
|---|---|---|
| **1×1 conv** (NiN, 2013) | Csatorna-dimenzió csökkentés | Pontosan egy pixelt dolgoz, de összes csatornát kombinálja |
| **Depthwise Separable** (MobileNet, 2017) | Paraméter-hatékonyság | Szétválasztja a térbeli és csatorna-szűrést |
| **Dilated/Atrous** (2016) | RF növelés pooling nélkül | Lyukacsos kernel, exponenciálisan növő RF |
| **Transposed Conv** | Tanulható upsampling | Decoder / szegmentáció hálókban (U-Net) |
| **Grouped Conv** (AlexNet→ResNeXt) | Párhuzamosítás | Csatornákat csoportokra bontja |

## Architektúra evolúció

```
Neocognitron (1980, Fukushima)
    │  + backpropagation
    ▼
LeNet-5 (1998, LeCun)         ← MNIST, kézírás
    │  + GPU + mély hálók
    ▼
AlexNet (2012, Krizhevsky)    ← ImageNet áttörés, ReLU, Dropout
    │  + mélyebb, egyszerűbb
    ▼
VGG-16 (2014, Simonyan)       ← csak 3×3 convok, 138M param
    │  + skip connection
    ▼
ResNet (2015, He et al.)       ← 152 réteg!, skip = F(x)+x
    │  + hatékony conv
    ▼
MobileNet (2017, Howard)       ← Depthwise sep., mobil eszközök
    │
    ├── EfficientNet (2019)    ← compound scaling (mélység×szélesség×felbontás)
    ├── ConvNeXt (2022)        ← CNN modernizálva Transformer trükkökkel
    └── ViT (2020)             ← figyelmi mechanizmus konvolúció HELYETT
```

## Neocognitron → CNN kapcsolat

| Neocognitron (1980) | CNN (modern) |
|---|---|
| S-cella (Simple) | Conv réteg |
| C-cella (Complex) | Pooling réteg |
| Hebb-tanulás | Backpropagation |
| Rétegenkénti tanítás | End-to-end tanítás |
| Önszervező | Felügyelt |

A koncepció ugyanaz: hierarchikus jellemző-detektálás (élek → textúrák → alakzatok → objektumok).

## Hivatkozások

- Fukushima, K. (1980). Neocognitron: A Self-organizing Neural Network Model. *Biological Cybernetics*.
- LeCun, Y. et al. (1998). Gradient-Based Learning Applied to Document Recognition. *Proc. IEEE*.
- Lin, M. et al. (2013). Network In Network. *arXiv:1312.4400*.
- Simonyan, K. & Zisserman, A. (2014). Very Deep Convolutional Networks (VGG). *arXiv:1409.1556*.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
- Howard, A. et al. (2017). MobileNets: Efficient CNNs for Mobile Vision. *arXiv:1704.04861*.
- Yu, F. & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. *ICLR*.
- Redmon, J. et al. (2016). You Only Look Once (YOLO). *CVPR*.
- Dosovitskiy, A. et al. (2020). An Image is Worth 16x16 Words (ViT). *ICLR*.
- Liu, Z. et al. (2022). A ConvNet for the 2020s (ConvNeXt). *CVPR*.
