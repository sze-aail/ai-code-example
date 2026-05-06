# 05_metrics - Lec12 metrikak vizualizacioja
Minimal, futtathato mintakodok klaszifikacios, detektalasi es szegmentacios metrikakhoz.
## Script lista
1. `01_confusion_matrix_prf1.py` - confusion matrix + classification report
2. `02_topk_accuracy_visualization.py` - top-1/top-5 bar chart
3. `03_roc_pr_curves.py` - ROC es PR gorbek (2 modell)
4. `04_iou_bbox_visualization.py` - bbox IoU + eltolasi erzekenyseg
5. `05_map_kuszobonkent.py` - PR gorbek IoU kuszobonkent + optional torchmetrics
6. `06_szegmentacio_overlay.py` - IoU es Dice mask overlay
7. `07_miou_per_class.py` - per-class IoU bar chart
8. `08_pq_bontas.py` - PQ bontas: SQ, RQ, PQ
9. `09_interaktiv_kuszob.py` - Plotly kuszob-csuszka (fallback matplotlib)
## Notebook lista

Minden scripthez tartozik egy paros notebook, ugyanazzal a sorszamozassal:

- `01_confusion_matrix_prf1.ipynb`
- `02_topk_accuracy_visualization.ipynb`
- `03_roc_pr_curves.ipynb`
- `04_iou_bbox_visualization.ipynb`
- `05_map_kuszobonkent.ipynb`
- `06_szegmentacio_overlay.ipynb`
- `07_miou_per_class.ipynb`
- `08_pq_bontas.ipynb`
- `09_interaktiv_kuszob.ipynb`

## Notebook validalas

```bash
python -c "import json,glob; [json.load(open(p,encoding='utf-8')) for p in glob.glob('12_convolutional_networks/05_metrics/*.ipynb')]; print('Notebook JSON valid')"
```

## Gyors ellenorzes
```bash
python 12_convolutional_networks/05_metrics/smoke_test_metrics.py
```
## Megjegyzesek
- A scriptek synthetic adatokat hasznalnak, tantermi celra.
- `seaborn`, `plotly`, `torchmetrics` hianya eseten fallback vagy figyelmeztetes jelenik meg.
- `--smoke` modban minden script fajlba ment, nem nyit interaktiv ablakot.
