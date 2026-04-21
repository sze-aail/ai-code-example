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
from matplotlib.widgets import RadioButtons, Slider
from pathlib import Path
import argparse


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


def build_checkpoint_path(checkpoint_dir, name="demo"):
    return Path(checkpoint_dir) / f"08_fpn_faster_rcnn_{name}.pt"


def save_checkpoint(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint_if_exists(path):
    if not path.exists():
        return None
    # PyTorch 2.6+ defaults to weights_only=True; these demo checkpoints store numpy objects too.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_canvas(seed=42):
    np.random.seed(seed)
    canvas = np.ones((64, 64, 3), dtype=np.float32) * 0.9
    for cx, cy, w, h, color in [
        (15, 20, 12, 15, [0.8, 0.2, 0.2]),
        (45, 40, 10, 10, [0.2, 0.6, 0.8]),
        (30, 50, 18, 8, [0.3, 0.8, 0.3]),
    ]:
        y1, y2 = max(0, cy - h // 2), min(64, cy + h // 2)
        x1, x2 = max(0, cx - w // 2), min(64, cx + w // 2)
        canvas[y1:y2, x1:x2] = color
    return canvas


def anchor_boxes():
    boxes = []
    n_shown = 0
    anchor_scales = [4, 8, 16]
    anchor_ratios = [0.5, 1.0, 2.0]
    for sy in range(4, 64, 16):
        for sx in range(4, 64, 16):
            for scale in anchor_scales[:1]:
                for ratio in anchor_ratios:
                    w = scale * np.sqrt(ratio)
                    h = scale / np.sqrt(ratio)
                    boxes.append((sx - w / 2, sy - h / 2, w, h))
                    n_shown += 1
    return boxes, n_shown


def demo_detections():
    return [
        (9, 12, 21, 28, "obj A", 0.95, "#E91E63"),
        (35, 30, 55, 50, "obj B", 0.88, "#2196F3"),
        (21, 46, 48, 58, "obj C", 0.72, "#4CAF50"),
    ]


def build_or_load_demo_state(checkpoint_dir=".", load_pt=True, save_pt=True, seed=42):
    torch.manual_seed(seed)
    backbone = SimpleBackbone()
    fpn = FPN([32, 64, 128, 256], out_channels=64)
    rpn = SimpleRPN(64)
    ckpt_path = build_checkpoint_path(checkpoint_dir, "state")

    payload = load_checkpoint_if_exists(ckpt_path) if load_pt else None
    if payload is not None:
        backbone.load_state_dict(payload["backbone_state_dict"])
        fpn.load_state_dict(payload["fpn_state_dict"])
        rpn.load_state_dict(payload["rpn_state_dict"])
        img = payload["img"]
        features = payload["features"]
        fpn_features = payload["fpn_features"]
        canvas = payload["canvas"]
        print(f"Checkpoint betoltve: {ckpt_path.name}")
    else:
        img = torch.randn(1, 3, 64, 64)
        features = backbone(img)
        fpn_features = fpn(features)
        canvas = build_canvas(seed)
        if save_pt:
            save_checkpoint(
                ckpt_path,
                {
                    "backbone_state_dict": backbone.state_dict(),
                    "fpn_state_dict": fpn.state_dict(),
                    "rpn_state_dict": rpn.state_dict(),
                    "img": img,
                    "features": features,
                    "fpn_features": fpn_features,
                    "canvas": canvas,
                    "meta": {"seed": seed},
                },
            )
            print(f"Checkpoint mentve: {ckpt_path.name}")

    with torch.no_grad():
        rpn_cls, rpn_reg = rpn(fpn_features[1])
    return {
        "backbone": backbone,
        "fpn": fpn,
        "rpn": rpn,
        "img": img,
        "features": features,
        "fpn_features": fpn_features,
        "canvas": canvas,
        "rpn_cls": rpn_cls,
        "rpn_reg": rpn_reg,
    }


def create_fpn_figure(features, fpn_features):
    print("=" * 60)
    print("1. FPN PIRAMIS SZINTEK")
    print("=" * 60)
    fig1, axes = plt.subplots(2, 4, figsize=(18, 9))
    for i, (feat, name) in enumerate(zip(features, ["C1 (32x32)", "C2 (16x16)", "C3 (8x8)", "C4 (4x4)"])):
        ax = axes[0, i]
        ax.imshow(feat[0, 0].detach().numpy(), cmap="viridis")
        ax.set_title(f"Backbone: {name}\n{feat.shape[1]}ch", fontsize=10, fontweight="bold")
        ax.axis("off")

    for i, (feat, name) in enumerate(zip(fpn_features, ["P1 (32x32)", "P2 (16x16)", "P3 (8x8)", "P4 (4x4)"])):
        ax = axes[1, i]
        ax.imshow(feat[0, 0].detach().numpy(), cmap="viridis")
        ax.set_title(f"FPN: {name}\n64ch (egyseges)", fontsize=10, fontweight="bold", color="#4CAF50")
        ax.axis("off")

    axes[0, 0].set_ylabel("Bottom-up\n(backbone)", fontsize=12, fontweight="bold")
    axes[1, 0].set_ylabel("Top-down\n(FPN)", fontsize=12, fontweight="bold")
    fig1.suptitle("FPN: Feature Pyramid Network - azonos szemantika minden skalan", fontsize=15, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig("08_fpn_piramis.png", dpi=150)
    print("Abra mentve: 08_fpn_piramis.png")


def create_pipeline_figure(features, canvas):
    print("\n" + "=" * 60)
    print("2. FASTER R-CNN PIPELINE")
    print("=" * 60)
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    axes2[0].imshow(canvas)
    axes2[0].set_title("1. Bemeneti kep\n(3 objektum)", fontweight="bold")
    axes2[0].axis("off")

    axes2[1].imshow(features[1][0, 0].detach().numpy(), cmap="viridis")
    axes2[1].set_title("2. Backbone feature map\n(C2: 16x16)", fontweight="bold")
    axes2[1].axis("off")

    boxes, n_shown = anchor_boxes()
    axes2[2].imshow(canvas)
    for x, y, w, h in boxes:
        axes2[2].add_patch(
            patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor="yellow", facecolor="none", alpha=0.6)
        )
    axes2[2].set_title(f"3. RPN anchor boxok\n({n_shown} db)", fontweight="bold")
    axes2[2].axis("off")

    axes2[3].imshow(canvas)
    for x1, y1, x2, y2, label, conf, color in demo_detections():
        axes2[3].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none"))
        axes2[3].text(x1, y1 - 2, f"{label}: {conf:.0%}", fontsize=8, color=color, fontweight="bold")
    axes2[3].set_title("4. Detektalt objektumok\n(NMS utan)", fontweight="bold")
    axes2[3].axis("off")

    fig2.suptitle("Faster R-CNN pipeline: kep -> backbone -> RPN -> detektalas", fontsize=14, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig("08_faster_rcnn_pipeline.png", dpi=150)
    print("Abra mentve: 08_faster_rcnn_pipeline.png")


def create_anchor_figure(canvas):
    print("\n" + "=" * 60)
    print("3. ANCHOR BOX-OK")
    print("=" * 60)
    fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (scale, title) in zip(axes3, [(4, "Kis skala (4px)"), (8, "Kozepes skala (8px)"), (16, "Nagy skala (16px)")]):
        ax.set_xlim(0, 64)
        ax.set_ylim(64, 0)
        ax.set_aspect("equal")
        ax.imshow(canvas, extent=[0, 64, 64, 0])
        cx, cy = 32, 32
        for ratio, color, label in zip([0.5, 1.0, 2.0], ["#E91E63", "#2196F3", "#4CAF50"], ["1:2", "1:1", "2:1"]):
            w = scale * np.sqrt(ratio)
            h = scale / np.sqrt(ratio)
            ax.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.2))
            ax.add_patch(patches.Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=2, edgecolor=color, facecolor="none"))
            ax.text(cx + w / 2 + 1, cy, label, fontsize=8, color=color, va="center")
        ax.plot(cx, cy, "k+", markersize=10, markeredgewidth=2)
        ax.set_title(f"{title}\n3 arany x 1 skala = 3 anchor", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2)

    fig3.suptitle("Anchor boxok: skala x arany = anchor keszlet (tipikusan 3x3=9)", fontsize=14, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig("08_anchor_boxok.png", dpi=150)
    print("Abra mentve: 08_anchor_boxok.png")


def create_summary_figure():
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    ax4.axis("off")
    data = [
        ["", "R-CNN\n(2014)", "Fast R-CNN\n(2015)", "Faster R-CNN\n(2015)", "YOLO\n(2016)", "Mask R-CNN\n(2017)"],
        ["Region\nproposal", "Selective\nSearch (kulso)", "Selective\nSearch (kulso)", "RPN\n(tanult!)", "Nincs\n(grid-alapu)", "RPN\n(tanult)"],
        ["Gerinc", "CNN per\nregio", "Kozos CNN\n(megosztott)", "Kozos CNN\n+ FPN", "DarkNet /\nCSPNet", "ResNet\n+ FPN"],
        ["Sebesseg", "~50s / kep\n(nagyon lassu)", "~2s / kep", "~0.2s / kep\n(5 FPS)", "~0.02s / kep\n(45+ FPS)", "~0.2s / kep"],
        ["Kimenet", "Osztaly +\nbbox", "Osztaly +\nbbox", "Osztaly +\nbbox", "Osztaly +\nbbox", "Osztaly +\nbbox + MASZK"],
        ["Fo ujitas", "CNN jellemzok\ndetektalashoz", "RoI Pooling\n(megosztott feat.)", "RPN: end-to-end\nregion proposal", "Egylepeses\n(one-shot)", "Instance\nszegmentacio"],
    ]
    table = ax4.table(cellText=data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 2.2)
    for j in range(6):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(data)):
        table[i, 0].set_facecolor("#ECEFF1")
        table[i, 0].set_text_props(fontweight="bold")
    ax4.set_title("Objektumdetektalas: R-CNN csalad evolucioja", fontsize=14, fontweight="bold", pad=20)
    fig4.tight_layout()
    fig4.savefig("08_detekcios_architekturak.png", dpi=150)
    print("Abra mentve: 08_detekcios_architekturak.png")


def launch_gui(state):
    features = state["features"]
    fpn_features = state["fpn_features"]
    canvas = state["canvas"]
    rpn_cls = state["rpn_cls"]
    rpn_reg = state["rpn_reg"]

    fig = plt.figure(figsize=(11, 7))
    gs = fig.add_gridspec(3, 2, height_ratios=[8, 1, 1], width_ratios=[1, 1])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_view = fig.add_subplot(gs[1, :])
    ax_channel = fig.add_subplot(gs[2, :])

    view_labels = ["Input", "C1", "C2", "C3", "C4", "P1", "P2", "P3", "P4", "Anchors", "Detections"]
    radio = RadioButtons(ax_view, view_labels, active=0)
    slider = Slider(ax_channel, "Csatorna", 0, 255, valinit=0, valstep=1)
    ax_text.axis("off")
    text_box = ax_text.text(0.0, 0.98, "", va="top", fontsize=10, family="monospace")

    fmap_dict = {
        "C1": features[0],
        "C2": features[1],
        "C3": features[2],
        "C4": features[3],
        "P1": fpn_features[0],
        "P2": fpn_features[1],
        "P3": fpn_features[2],
        "P4": fpn_features[3],
    }
    current = {"view": "Input"}

    def update(_):
        view = current["view"]
        ch = int(slider.val)
        ax_img.clear()
        details = [f"View: {view}"]

        if view == "Input":
            ax_img.imshow(canvas)
            ax_img.set_title("Szintetikus bemeneti kep", fontweight="bold")
        elif view in fmap_dict:
            feat = fmap_dict[view]
            use_ch = min(ch, feat.shape[1] - 1)
            ax_img.imshow(feat[0, use_ch].detach().numpy(), cmap="viridis")
            ax_img.set_title(f"{view} feature map - channel {use_ch}", fontweight="bold")
            details.extend([
                f"Shape: {tuple(feat.shape)}",
                f"Min/Max: {float(feat.min()):.3f} / {float(feat.max()):.3f}",
            ])
        elif view == "Anchors":
            ax_img.imshow(canvas)
            boxes, n_shown = anchor_boxes()
            for x, y, w, h in boxes:
                ax_img.add_patch(
                    patches.Rectangle((x, y), w, h, linewidth=0.5, edgecolor="yellow", facecolor="none", alpha=0.6)
                )
            ax_img.set_title(f"RPN anchor boxok ({n_shown} db)", fontweight="bold")
            details.append(f"RPN cls shape: {tuple(rpn_cls.shape)}")
            details.append(f"RPN reg shape: {tuple(rpn_reg.shape)}")
        else:
            ax_img.imshow(canvas)
            for x1, y1, x2, y2, label, conf, color in demo_detections():
                ax_img.add_patch(
                    patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor="none")
                )
                ax_img.text(x1, y1 - 2, f"{label}: {conf:.0%}", fontsize=8, color=color, fontweight="bold")
            ax_img.set_title("Detektalt objektumok", fontweight="bold")
            details.append(f"Detekciok: {len(demo_detections())}")

        ax_img.axis("off")
        text_box.set_text("\n".join(details))
        fig.canvas.draw_idle()

    def on_view_change(label):
        current["view"] = label
        update(None)

    radio.on_clicked(on_view_change)
    slider.on_changed(update)
    update(None)
    fig.suptitle("FPN + Faster R-CNN interaktiv GUI", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="FPN/Faster R-CNN demo checkpoint es GUI tamogatassal")
    parser.add_argument("--checkpoint-dir", default=".", help="Checkpoint konyvtar (.pt)")
    parser.add_argument("--no-load-pt", action="store_true", help="Ne toltsen be meglevo checkpointot")
    parser.add_argument("--no-save-pt", action="store_true", help="Ne mentsen checkpointot")
    parser.add_argument("--gui", action="store_true", help="GUI inditasa")
    parser.add_argument("--gui-only", action="store_true", help="Csak GUI, statikus abrak kihagyasa")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    state = build_or_load_demo_state(
        checkpoint_dir=args.checkpoint_dir,
        load_pt=not args.no_load_pt,
        save_pt=not args.no_save_pt,
        seed=args.seed,
    )

    if not args.gui_only:
        create_fpn_figure(state["features"], state["fpn_features"])
        create_pipeline_figure(state["features"], state["canvas"])
        create_anchor_figure(state["canvas"])
        create_summary_figure()

    if args.gui:
        launch_gui(state)

    plt.close("all")
    print("\nKesz!")


if __name__ == "__main__":
    main()
