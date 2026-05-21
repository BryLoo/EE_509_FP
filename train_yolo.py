"""
Train + evaluate YOLO on KITTI using Ultralytics (CUDA GPU).

The proposal asks how YOLO versions improved over time, so this script accepts any
Ultralytics checkpoint and can be run several times to compare versions/sizes:

    python train_yolo.py --weights yolov8n.pt     # YOLOv8 nano
    python train_yolo.py --weights yolov8s.pt     # YOLOv8 small
    python train_yolo.py --weights yolov8m.pt     # YOLOv8 medium
    python train_yolo.py --weights yolo11n.pt     # YOLO11 nano
    python train_yolo.py --weights yolov5su.pt    # YOLOv5 (ultralytics build)

Weights download automatically on first use. Writes outputs/results_yolo_<name>.json.
"""
import argparse
import json

import config
from data.kitti_to_yolo import convert


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--imgsz", type=int, default=config.YOLO_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0", help="CUDA device id, or 'cpu'")
    args = ap.parse_args()

    from ultralytics import YOLO  # imported here so the rest of the repo runs without it

    data_yaml = str(convert())  # builds YOLO dataset from the SHARED split
    tag = args.weights.replace(".pt", "")

    model = YOLO(args.weights)
    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(config.OUTPUT_DIR / "yolo"),
        name=tag,
        seed=config.SPLIT_SEED,
    )

    metrics = model.val(data=data_yaml, imgsz=args.imgsz, device=args.device)

    # metrics.speed is ms per image: preprocess + inference + postprocess
    speed = metrics.speed
    infer_ms = speed.get("inference", float("nan"))
    total_ms = sum(v for v in speed.values() if isinstance(v, (int, float)))
    per_class = {name: float(metrics.box.maps[i]) for i, name in enumerate(config.CLASSES)} \
        if hasattr(metrics.box, "maps") else {}

    result = {
        "model": f"yolo_{tag}",
        "map":    float(metrics.box.map),     # mAP@[.5:.95]
        "map_50": float(metrics.box.map50),
        "map_75": float(metrics.box.map75),
        "map_per_class": per_class,
        "fps": (1000.0 / infer_ms) if infer_ms else float("nan"),  # inference-only FPS
        "fps_end_to_end": (1000.0 / total_ms) if total_ms else float("nan"),
        "speed_ms": speed,
    }
    out = config.OUTPUT_DIR / f"results_yolo_{tag}.json"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
