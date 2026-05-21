"""
Qualitative inspection: draw predicted boxes on val images (for the report figures).

Example:
    python visualize.py --model faster_rcnn --weights outputs/faster_rcnn/best.pt --n 8

YOLO already saves prediction images under outputs/yolo/<name>/ during val(); you can
also run `yolo predict model=outputs/yolo/<name>/weights/best.pt source=...` for more.
"""
import argparse

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.utils import draw_bounding_boxes

import config
from data.prepare_data import load_split
from models import build

ID_TO_NAME = {v: k for k, v in config.CLASS_TO_TVID.items()}
COLORS = {"car": "red", "pedestrian": "lime", "cyclist": "cyan"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["faster_rcnn", "ssd"], required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--score-thresh", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device(config.DEVICE)
    weights = args.weights or str(config.OUTPUT_DIR / args.model / "best.pt")

    model = build(args.model)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device).eval()

    out_dir = config.OUTPUT_DIR / "viz" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in load_split("val")[: args.n]:
        img = Image.open(config.TRAIN_IMAGES / f"{img_id}.png").convert("RGB")
        x = F.to_tensor(img).to(device)
        out = model([x])[0]

        keep = out["scores"] >= args.score_thresh
        boxes = out["boxes"][keep].cpu()
        names = [ID_TO_NAME.get(int(l), "?") for l in out["labels"][keep].cpu()]
        scores = out["scores"][keep].cpu().tolist()
        labels = [f"{n} {s:.2f}" for n, s in zip(names, scores)]
        colors = [COLORS.get(n, "yellow") for n in names]

        canvas = (x.cpu() * 255).to(torch.uint8)
        drawn = draw_bounding_boxes(canvas, boxes, labels=labels,
                                    colors=colors, width=2)
        Image.fromarray(drawn.permute(1, 2, 0).numpy()).save(out_dir / f"{img_id}.png")

    print(f"Saved {args.n} visualizations to {out_dir}")


if __name__ == "__main__":
    main()
