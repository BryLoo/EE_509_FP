# """
# Qualitative inspection: draw predicted boxes on val images (for the report figures).

# Example:
#     python visualize.py --model faster_rcnn --weights outputs/faster_rcnn/best.pt --n 8

# YOLO already saves prediction images under outputs/yolo/<name>/ during val(); you can
# also run `yolo predict model=outputs/yolo/<name>/weights/best.pt source=...` for more.
# """
# import argparse

# import torch
# import torchvision.transforms.functional as F
# from PIL import Image
# from torchvision.utils import draw_bounding_boxes

# import config
# from data.prepare_data import load_split
# from models import build

# ID_TO_NAME = {v: k for k, v in config.CLASS_TO_TVID.items()}
# COLORS = {"car": "red", "pedestrian": "lime", "cyclist": "cyan"}


# @torch.no_grad()
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--model", choices=["faster_rcnn", "ssd"], required=True)
#     ap.add_argument("--weights", default=None)
#     ap.add_argument("--n", type=int, default=8)
#     ap.add_argument("--score-thresh", type=float, default=0.5)
#     args = ap.parse_args()

#     device = torch.device(config.DEVICE)
#     weights = args.weights or str(config.OUTPUT_DIR / args.model / "best.pt")

#     model = build(args.model)
#     model.load_state_dict(torch.load(weights, map_location=device))
#     model.to(device).eval()

#     out_dir = config.OUTPUT_DIR / "viz" / args.model
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for img_id in load_split("val")[: args.n]:
#         img = Image.open(config.TRAIN_IMAGES / f"{img_id}.png").convert("RGB")
#         x = F.to_tensor(img).to(device)
#         out = model([x])[0]

#         keep = out["scores"] >= args.score_thresh
#         boxes = out["boxes"][keep].cpu()
#         names = [ID_TO_NAME.get(int(l), "?") for l in out["labels"][keep].cpu()]
#         scores = out["scores"][keep].cpu().tolist()
#         labels = [f"{n} {s:.2f}" for n, s in zip(names, scores)]
#         colors = [COLORS.get(n, "yellow") for n in names]

#         canvas = (x.cpu() * 255).to(torch.uint8)
#         drawn = draw_bounding_boxes(canvas, boxes, labels=labels,
#                                     colors=colors, width=2)
#         Image.fromarray(drawn.permute(1, 2, 0).numpy()).save(out_dir / f"{img_id}.png")

#     print(f"Saved {args.n} visualizations to {out_dir}")


# if __name__ == "__main__":
#     main()

"""
Qualitative inspection of detections (for the report figures).

Two modes:

1) Per-image (original): saves individual annotated PNGs for the first N val images.
       python visualize.py --model faster_rcnn --n 8

2) Grid (new): renders a list of specific images as one tiled figure, the
   torchvision equivalent of YOLO's val_batch0_pred.jpg.
       python visualize.py --model faster_rcnn --grid
       python visualize.py --model faster_rcnn --grid --ids 007474 007462 ... --cols 4

The grid mode checks each requested image against the validation split. Images
NOT in the val set are still drawn, but flagged in red in the tile title and
listed in the terminal, so you don't accidentally present predictions on images
the model trained on.
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from PIL import Image

import config
from data.prepare_data import load_split
from models import build

ID_TO_NAME = {v: k for k, v in config.CLASS_TO_TVID.items()}
# RGB tuples for matplotlib; matches the per-image (PIL) color scheme.
COLORS = {"car": "#FF3B30", "pedestrian": "#34C759", "cyclist": "#00C7BE"}

# The 16 images requested for the 4x4 grid (KITTI IDs zero-padded to 6 digits).
DEFAULT_GRID_IDS = [
    "007474",
    "007462",
    "007439",
    "000391",
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
    """Zero-pad to KITTI's 6-digit convention ('07447' -> '007447')."""
    return img_id.strip().zfill(6)


@torch.no_grad()
def _predict(model, device, img, score_thresh):
    """Run the model on one PIL image; return (boxes, names, scores) above threshold."""
    x = F.to_tensor(img).to(device)
    out = model([x])[0]
    keep = out["scores"] >= score_thresh
    boxes = out["boxes"][keep].cpu().tolist()
    names = [ID_TO_NAME.get(int(l), "?") for l in out["labels"][keep].cpu()]
    scores = out["scores"][keep].cpu().tolist()
    return boxes, names, scores


def run_grid(model, device, img_ids, cols, score_thresh, out_path, model_label):
    import math
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    img_ids = [_normalize(i) for i in img_ids]
    val_set = set(load_split("val"))

    # Report val membership before rendering.
    in_val = [i for i in img_ids if i in val_set]
    not_val = [i for i in img_ids if i not in val_set]
    print(f"{len(in_val)}/{len(img_ids)} requested images are in the validation split.")
    if not_val:
        print(
            "  NOT in val (drawn but flagged red in the figure): " + ", ".join(not_val)
        )

    n = len(img_ids)
    rows = math.ceil(n / cols)

    # KITTI frames are ~1242x375 (aspect ~3.3); size the figure so tiles aren't squashed.
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
        boxes, names, scores = _predict(model, device, img, score_thresh)

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
                fontsize=5.5,
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

    # Hide any unused tiles (if n is not a perfect multiple of cols).
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


@torch.no_grad()
def run_individual(model, device, n, score_thresh, out_dir):
    from torchvision.utils import draw_bounding_boxes

    out_dir.mkdir(parents=True, exist_ok=True)
    for img_id in load_split("val")[:n]:
        img = Image.open(config.TRAIN_IMAGES / f"{img_id}.png").convert("RGB")
        x = F.to_tensor(img).to(device)
        out = model([x])[0]
        keep = out["scores"] >= score_thresh
        boxes = out["boxes"][keep].cpu()
        names = [ID_TO_NAME.get(int(l), "?") for l in out["labels"][keep].cpu()]
        scores = out["scores"][keep].cpu().tolist()
        labels = [f"{nm} {s:.2f}" for nm, s in zip(names, scores)]
        colors = [COLORS.get(nm, "#FFCC00") for nm in names]
        canvas = (x.cpu() * 255).to(torch.uint8)
        drawn = draw_bounding_boxes(
            canvas, boxes, labels=labels, colors=colors, width=2
        )
        Image.fromarray(drawn.permute(1, 2, 0).numpy()).save(out_dir / f"{img_id}.png")
    print(f"Saved {n} visualizations to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["faster_rcnn", "ssd"], required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument(
        "--grid",
        action="store_true",
        help="render a tiled grid (default: the 16 requested images)",
    )
    ap.add_argument(
        "--ids",
        nargs="+",
        default=None,
        help="image IDs for grid mode (overrides the default list)",
    )
    ap.add_argument("--cols", type=int, default=4, help="grid columns (default 4)")
    ap.add_argument(
        "--n", type=int, default=8, help="per-image mode: number of val images"
    )
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="grid mode output path")
    args = ap.parse_args()

    device = torch.device(config.DEVICE)
    weights = args.weights or str(config.OUTPUT_DIR / args.model / "best.pt")
    model = build(args.model)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    if args.grid:
        ids = args.ids or DEFAULT_GRID_IDS
        out_path = (
            Path(args.out)
            if args.out
            else config.OUTPUT_DIR / "viz" / f"{args.model}_val_grid.png"
        )
        run_grid(model, device, ids, args.cols, args.score_thresh, out_path, args.model)
    else:
        run_individual(
            model,
            device,
            args.n,
            args.score_thresh,
            config.OUTPUT_DIR / "viz" / args.model,
        )


if __name__ == "__main__":
    main()
