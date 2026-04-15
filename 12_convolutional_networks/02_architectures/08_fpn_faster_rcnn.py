"""
08_fpn_faster_rcnn.py — Feature Pyramid Network és Faster R-CNN
Neurális hálók II. (CNN) — Hajdu Csaba

Demonstrálja:
  - FPN (Lin et al., 2017): multi-skálás jellemző-piramis top-down útvonallal
  - Faster R-CNN (Ren et al., 2015): Region Proposal Network + RoI Pooling
  - Az FPN integrálása Faster R-CNN-be
  - Piramis szintek vizualizáció
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ══════════════════════════════════════════════════════════════
# FPN IMPLEMENTÁCIÓ
# ══════════════════════════════════════════════════════════════

class SimpleBackbone(nn.Module):
    """Egyszerű 4 szintű backbone (ResNet-szerű)."""
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.stage2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.stage3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.stage4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))

    def forward(self, x):
        c1 = self.stage1(x)   # /2
        c2 = self.stage2(c1)  # /4
        c3 = self.stage3(c2)  # /8
        c4 = self.stage4(c3)  # /16
        return [c1, c2, c3, c4]


class FPN(nn.Module):
    """
    Feature Pyramid Network (Lin et al., 2017).

    Bottom-up: backbone (C1→C2→C3→C4) — egyre kisebb, egyre szemantikusabb
    Top-down: upsampling + laterális összekötés → azonos szemantika MINDEN skálán

    P4 = Conv1×1(C4)
    P3 = Conv1×1(C3) + Upsample(P4)
    P2 = Conv1×1(C2) + Upsample(P3)
    P1 = Conv1×1(C1) + Upsample(P2)

    Minden P szinten: 3×3 conv simítás (aliasing csökkentés)
    """
    def __init__(self, in_channels_list, out_channels=64):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        self.smooth_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])

    def forward(self, features):
        # Top-down útvonal
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        # Felülről lefelé: upsample + összeadás
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], size=laterals[i].shape[2:], mode='nearest')

        # Simítás
        outputs = [conv(lat) for conv, lat in zip(self.smooth_convs, laterals)]
        return outputs


# ══════════════════════════════════════════════════════════════
# RPN (Region Proposal Network) — egyszerűsített
# ══════════════════════════════════════════════════════════════

class SimpleRPN(nn.Module):
    """
    Region Proposal Network (Faster R-CNN, Ren et al., 2015).

    Minden pozíción k anchor box-ot generál, és mindegyikre:
    - Objektum valószínűség (objectness score)
    - Bounding box finomítás (dx, dy, dw, dh)
    """
    def __init__(self, in_channels, n_anchors=9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls = nn.Conv2d(256, n_anchors * 2, 1)  # obj/no-obj per anchor
        self.reg = nn.Conv2d(256, n_anchors * 4, 1)  # dx,dy,dw,dh per anchor

    def forward(self, feature_map):
        x = F.relu(self.conv(feature_map))
        cls_scores = self.cls(x)
        bbox_deltas = self.reg(x)
        return cls_scores, bbox_deltas


# ══════════════════════════════════════════════════════════════
# 1. FPN PIRAMIS VIZUALIZÁCIÓ
# ══════════════════════════════════════════════════════════════

print("=" * 60)
print("1. FPN PIRAMIS SZINTEK")
print("=" * 60)

torch.manual_seed(42)
backbone = SimpleBackbone()
fpn = FPN([32, 64, 128, 256], out_channels=64)

# Szintetikus kép (64×64 RGB)
img = torch.randn(1, 3, 64, 64)
features = backbone(img)
fpn_features = fpn(features)

fig1, axes = plt.subplots(2, 4, figsize=(18, 9))

# Felső sor: backbone kimenetek (bottom-up)
for i, (feat, name) in enumerate(zip(features, ["C1 (32×32)", "C2 (16×16)", "C3 (8×8)", "C4 (4×4)"])):
    ax = axes[0, i]
    fm = feat[0, 0].detach().numpy()
    ax.imshow(fm, cmap="viridis")
    ax.set_title(f"Backbone: {name}\n{feat.shape[1]}ch", fontsize=10, fontweight="bold")
    ax.axis("off")

# Alsó sor: FPN kimenetek (top-down + lateral)
for i, (feat, name) in enumerate(zip(fpn_features, ["P1 (32×32)", "P2 (16×16)", "P3 (8×8)", "P4 (4×4)"])):
    ax = axes[1, i]
    fm = feat[0, 0].detach().numpy()
    ax.imshow(fm, cmap="viridis")
    ax.set_title(f"FPN: {name}\n64ch (egységes!)", fontsize=10, fontweight="bold",
                 color="#4CAF50")
    ax.axis("off")

axes[0, 0].set_ylabel("Bottom-up\n(backbone)", fontsize=12, fontweight="bold")
axes[1, 0].set_ylabel("Top-down\n(FPN)", fontsize=12, fontweight="bold")

fig1.suptitle("FPN: Feature Pyramid Network — azonos szemantika minden skálán",
              fontsize=15, fontweight="bold")
fig1.tight_layout()
fig1.savefig("08_fpn_piramis.png", dpi=150)
print("Ábra mentve: 08_fpn_piramis.png")


# ══════════════════════════════════════════════════════════════
# 2. FASTER R-CNN PIPELINE VIZUALIZÁCIÓ
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("2. FASTER R-CNN PIPELINE")
print("=" * 60)

fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))

# 1. Bemeneti kép szimulálás
ax = axes2[0]
np.random.seed(42)
canvas = np.ones((64, 64, 3)) * 0.9
# "Objektumok"
for cx, cy, w, h, color in [(15, 20, 12, 15, [0.8, 0.2, 0.2]),
                              (45, 40, 10, 10, [0.2, 0.6, 0.8]),
                              (30, 50, 18, 8, [0.3, 0.8, 0.3])]:
    y1, y2 = max(0, cy-h//2), min(64, cy+h//2)
    x1, x2 = max(0, cx-w//2), min(64, cx+w//2)
    canvas[y1:y2, x1:x2] = color
ax.imshow(canvas)
ax.set_title("1. Bemeneti kép\n(3 objektum)", fontweight="bold")
ax.axis("off")

# 2. Feature map
ax = axes2[1]
ax.imshow(features[1][0, 0].detach().numpy(), cmap="viridis")
ax.set_title("2. Backbone feature map\n(C2: 16×16)", fontweight="bold")
ax.axis("off")

# 3. RPN anchor box-ok
ax = axes2[2]
ax.imshow(canvas)
# Anchor boxok rajzolása
anchor_scales = [4, 8, 16]
anchor_ratios = [0.5, 1.0, 2.0]
n_shown = 0
for sy in range(4, 64, 16):
    for sx in range(4, 64, 16):
        for scale in anchor_scales[:1]:
            for ratio in anchor_ratios:
                w = scale * np.sqrt(ratio)
                h = scale / np.sqrt(ratio)
                rect = patches.Rectangle((sx - w/2, sy - h/2), w, h,
                                         linewidth=0.5, edgecolor='yellow', facecolor='none', alpha=0.6)
                ax.add_patch(rect)
                n_shown += 1
ax.set_title(f"3. RPN: anchor box-ok\n({n_shown} db megjelenítve)", fontweight="bold")
ax.axis("off")

# 4. Detektálás eredmény
ax = axes2[3]
ax.imshow(canvas)
detections = [(9, 12, 21, 28, "obj A", 0.95, "#E91E63"),
              (35, 30, 55, 50, "obj B", 0.88, "#2196F3"),
              (21, 46, 48, 58, "obj C", 0.72, "#4CAF50")]
for x1, y1, x2, y2, label, conf, color in detections:
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                              linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-2, f"{label}: {conf:.0%}", fontsize=8, color=color, fontweight="bold")
ax.set_title("4. Detektált objektumok\n(NMS után)", fontweight="bold")
ax.axis("off")

fig2.suptitle("Faster R-CNN pipeline: kép → backbone → RPN (region proposals) → detektálás",
              fontsize=14, fontweight="bold")
fig2.tight_layout()
fig2.savefig("08_faster_rcnn_pipeline.png", dpi=150)
print("Ábra mentve: 08_faster_rcnn_pipeline.png")


# ══════════════════════════════════════════════════════════════
# 3. ANCHOR BOX-OK RÉSZLETESEN
# ══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("3. ANCHOR BOX-OK")
print("=" * 60)

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))

# Különböző skálák
for ax, (scale, title) in zip(axes3, [(4, "Kis skála (4px)"),
                                        (8, "Közepes skála (8px)"),
                                        (16, "Nagy skála (16px)")]):
    ax.set_xlim(0, 64); ax.set_ylim(64, 0); ax.set_aspect("equal")
    ax.imshow(canvas, extent=[0, 64, 64, 0])

    cx, cy = 32, 32
    colors_a = ["#E91E63", "#2196F3", "#4CAF50"]
    for ratio, color, label in zip([0.5, 1.0, 2.0], colors_a, ["1:2", "1:1", "2:1"]):
        w = scale * np.sqrt(ratio)
        h = scale / np.sqrt(ratio)
        rect = patches.Rectangle((cx - w/2, cy - h/2), w, h,
                                  linewidth=2, edgecolor=color, facecolor=color, alpha=0.2)
        ax.add_patch(rect)
        rect2 = patches.Rectangle((cx - w/2, cy - h/2), w, h,
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect2)
        ax.text(cx + w/2 + 1, cy, label, fontsize=8, color=color, va="center")

    ax.plot(cx, cy, "k+", markersize=10, markeredgewidth=2)
    ax.set_title(f"{title}\n3 arány × 1 skála = 3 anchor", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2)

fig3.suptitle("Anchor box-ok: skála × arány = anchor készlet (tipikusan 3×3=9 per pozíció)",
              fontsize=14, fontweight="bold")
fig3.tight_layout()
fig3.savefig("08_anchor_boxok.png", dpi=150)
print("Ábra mentve: 08_anchor_boxok.png")


# ══════════════════════════════════════════════════════════════
# 4. DETEKCIÓS ARCHITEKTÚRÁK ÖSSZEFOGLALÁS
# ══════════════════════════════════════════════════════════════

fig4, ax4 = plt.subplots(figsize=(14, 6))
ax4.axis("off")
data = [
    ["", "R-CNN\n(2014)", "Fast R-CNN\n(2015)", "Faster R-CNN\n(2015)", "YOLO\n(2016)", "Mask R-CNN\n(2017)"],
    ["Region\nproposal", "Selective\nSearch (külső)", "Selective\nSearch (külső)", "RPN\n(tanult!)", "Nincs\n(grid-alapú)", "RPN\n(tanult)"],
    ["Gerinc", "CNN per\nrégió", "Közös CNN\n(megosztott)", "Közös CNN\n+ FPN", "DarkNet /\nCSPNet", "ResNet\n+ FPN"],
    ["Sebesség", "~50s / kép\n(nagyon lassú)", "~2s / kép", "~0.2s / kép\n(5 FPS)", "~0.02s / kép\n(45+ FPS)", "~0.2s / kép"],
    ["Kimenet", "Osztály +\nbbox", "Osztály +\nbbox", "Osztály +\nbbox", "Osztály +\nbbox", "Osztály +\nbbox + MASZK"],
    ["Fő újítás", "CNN jellemzők\ndetektáláshoz", "RoI Pooling\n(megosztott feat.)", "RPN: end-to-end\nregion proposal", "Egylépéses\n(one-shot)", "Instance\nszegmentáció"],
]
table = ax4.table(cellText=data, loc="center", cellLoc="center")
table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1.0, 2.2)
for j in range(6):
    table[0, j].set_facecolor("#37474F")
    table[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(data)):
    table[i, 0].set_facecolor("#ECEFF1")
    table[i, 0].set_text_props(fontweight="bold")
ax4.set_title("Objektumdetektálás: R-CNN család evolúciója", fontsize=14, fontweight="bold", pad=20)
fig4.tight_layout()
fig4.savefig("08_detekcios_architekturak.png", dpi=150)
print("Ábra mentve: 08_detekcios_architekturak.png")

plt.close("all")
print("\nKész!")
