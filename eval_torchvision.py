"""
Evaluate a trained Faster R-CNN / SSD checkpoint: mAP (accuracy) + FPS (speed).

Example:
    python eval_torchvision.py --model faster_rcnn --weights outputs/faster_rcnn/best.pt

Writes outputs/results_<model>.json, consumed by compare.py.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader

import config
from data.prepare_data import load_split
from data.kitti_dataset import KittiDetectionDataset, collate_fn
from models import build, count_params_millions
from engine import evaluate_map, measure_fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["faster_rcnn", "ssd"], required=True)
    ap.add_argument("--weights", default=None,
                    help="default: outputs/<model>/best.pt")
    args = ap.parse_args()

    device = torch.device(config.DEVICE)
    weights = args.weights or str(config.OUTPUT_DIR / args.model / "best.pt")

    val_ds = KittiDetectionDataset(load_split("val"), train=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=config.NUM_WORKERS,
                            collate_fn=collate_fn, pin_memory=True)

    model = build(args.model)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.to(device)

    print(f"Evaluating {args.model} ({weights}) on {device} ...")
    metrics = evaluate_map(model, val_loader, device)
    fps = measure_fps(model, val_loader, device)

    per_class = {}
    if isinstance(metrics.get("map_per_class"), list):
        for name, ap in zip(config.CLASSES, metrics["map_per_class"]):
            per_class[name] = ap

    result = {
        "model": args.model,
        "map":    float(metrics.get("map", -1)),
        "map_50": float(metrics.get("map_50", -1)),
        "map_75": float(metrics.get("map_75", -1)),
        "map_per_class": per_class,
        "fps": float(fps),
        "params_millions": round(count_params_millions(model), 2),
    }
    out = config.OUTPUT_DIR / f"results_{args.model}.json"
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
