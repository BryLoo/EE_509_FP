"""
Qualitative inspection of detections (for the report figures).

Supports all three models behind one renderer, so figures are directly comparable
(same images, same confidence threshold, same matplotlib drawing code -- only the
model differs):

    --model faster_rcnn | ssd   -> loads outputs/<model>/best.pt (torchvision)
    --model yolo                -> loads an Ultralytics best.pt (pass --weights)

Modes:

1) Per-image: individual annotated PNGs for the first N val images.
       python visualize.py --model faster_rcnn --n 8

2) Grid: a tiled 4x4 figure of specific images (YOLO val_batch0_pred.jpg analog).
       python visualize.py --model faster_rcnn --grid --score-thresh 0.25
       python visualize.py --model yolo --grid --score-thresh 0.25 \
           --weights runs/detect/outputs/yolo/yolo11n-3/weights/best.pt

For a FAIR comparison, run two models with the SAME --score-thresh and the SAME
images (the default 16, or your own via --ids). Grid mode flags any requested
image that is NOT in the validation split (the model trained on it).
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image

import config
from data.prepare_data import load_split
from models import build

ID_TO_NAME = {v: k for k, v in config.CLASS_TO_TVID.items()}  # torchvision: 1->car ...
COLORS = {"car": "#FF3B30", "pedestrian": "#34C759", "cyclist": "#00C7BE"}

DEFAULT_GRID_IDS = [
    "007474",
    "007462",
    "007439",
    "000394",
    "007471",
    "007461",
    "007437",
    "000090",
    "007465",
    "007451",
    "007435",
    "000087",
    "007463",
    "007447",
    "007403",
    "000085",
]


def _normalize(img_id):
    return img_id.strip().zfill(6)


def make_predictor(model_type, weights, device, iou=0.7):
    """Return a function predict(pil_image, score_thresh) -> (boxes, names, scores).
    Boxes are [x1,y1,x2,y2] in the ORIGINAL image's pixel coordinates for all models."""
    if model_type in ("faster_rcnn", "ssd"):
        model = build(model_type)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device).eval()

        @torch.no_grad()
        def predict(img, score_thresh):
            x = F.to_tensor(img).to(device)
            out = model([x])[0]
            keep = out["scores"] >= score_thresh
            boxes = out["boxes"][keep].cpu().tolist()
            names = [ID_TO_NAME.get(int(l), "?") for l in out["labels"][keep].cpu()]
            scores = out["scores"][keep].cpu().tolist()
            return boxes, names, scores

        return predict

    if model_type == "yolo":
        from ultralytics import YOLO

        model = YOLO(weights)
        dev = 0 if device.type == "cuda" else "cpu"

        def predict(img, score_thresh):
            res = model.predict(
                source=img,
                conf=score_thresh,
                iou=iou,
                verbose=False,
                save=False,
                device=dev,
            )
            r = res[0]
            if r.boxes is None or len(r.boxes) == 0:
                return [], [], []
            boxes = r.boxes.xyxy.cpu().tolist()
            cls = r.boxes.cls.cpu().int().tolist()
            scores = r.boxes.conf.cpu().tolist()
            names = [
                config.CLASSES[i] if 0 <= i < len(config.CLASSES) else "?" for i in cls
            ]
            return boxes, names, scores

        return predict

    raise ValueError(f"Unknown model '{model_type}'")


def run_grid(predict, img_ids, cols, score_thresh, out_path, model_label):
    import math
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img_ids = [_normalize(i) for i in img_ids]
    val_set = set(load_split("val"))
    not_val = [i for i in img_ids if i not in val_set]
    print(f"{len(img_ids) - len(not_val)}/{len(img_ids)} requested images are in val.")
    if not_val:
        print("  NOT in val (drawn but flagged red): " + ", ".join(not_val))

    n = len(img_ids)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 1.7))
    axes = np.array(axes).reshape(-1)

    for ax, img_id in zip(axes, img_ids):
        img_path = config.TRAIN_IMAGES / f"{img_id}.png"
        if not img_path.exists():
            ax.text(
                0.5,
                0.5,
                f"{img_id}\n(not found)",
                ha="center",
                va="center",
                fontsize=9,
                color="red",
            )
            ax.axis("off")
            continue

        img = Image.open(img_path).convert("RGB")
        boxes, names, scores = predict(img, score_thresh)

        ax.imshow(np.asarray(img))
        for (x1, y1, x2, y2), name, score in zip(boxes, names, scores):
            color = COLORS.get(name, "#FFCC00")
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.4,
                )
            )
            ax.text(
                x1,
                y1 - 3,
                f"{name} {score:.2f}",
                fontsize=12,
                color="white",
                bbox=dict(facecolor=color, edgecolor="none", pad=0.6),
            )

        flagged = img_id not in val_set
        ax.set_title(
            img_id + ("  (NOT VAL)" if flagged else ""),
            fontsize=8,
            color="red" if flagged else "black",
        )
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"{model_label} — validation predictions (conf \u2265 {score_thresh})",
        fontsize=12,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved grid figure to {out_path}")


def run_individual(predict, n, score_thresh, out_dir):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    out_dir.mkdir(parents=True, exist_ok=True)
    for img_id in load_split("val")[:n]:
        img = Image.open(config.TRAIN_IMAGES / f"{img_id}.png").convert("RGB")
        boxes, names, scores = predict(img, score_thresh)
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.imshow(np.asarray(img))
        for (x1, y1, x2, y2), name, score in zip(boxes, names, scores):
            color = COLORS.get(name, "#FFCC00")
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.4,
                )
            )
            ax.text(
                x1,
                y1 - 3,
                f"{name} {score:.2f}",
                fontsize=13,
                color="white",
                bbox=dict(facecolor=color, edgecolor="none", pad=0.6),
            )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{img_id}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {n} visualizations to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["faster_rcnn", "ssd", "yolo"], required=True)
    ap.add_argument(
        "--weights",
        default=None,
        help="torchvision default: outputs/<model>/best.pt; "
        "YOLO: pass the Ultralytics best.pt path explicitly",
    )
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(config.DEVICE)

    if args.model == "yolo":
        if not args.weights:
            ap.error(
                "--model yolo requires --weights, e.g. "
                "--weights runs/detect/outputs/yolo/yolo11n-3/weights/best.pt"
            )
        weights = args.weights
    else:
        weights = args.weights or str(config.OUTPUT_DIR / args.model / "best.pt")

    predict = make_predictor(args.model, weights, device, iou=args.iou)

    if args.grid:
        ids = args.ids or DEFAULT_GRID_IDS
        out_path = (
            Path(args.out)
            if args.out
            else config.OUTPUT_DIR / "viz" / f"{args.model}_val_grid.png"
        )
        run_grid(predict, ids, args.cols, args.score_thresh, out_path, args.model)
    else:
        run_individual(
            predict, args.n, args.score_thresh, config.OUTPUT_DIR / "viz" / args.model
        )


if __name__ == "__main__":
    main()
