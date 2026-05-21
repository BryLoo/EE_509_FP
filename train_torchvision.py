"""
Train Faster R-CNN or SSD on KITTI (CUDA GPU).

Examples:
    python train_torchvision.py --model faster_rcnn --epochs 20 --batch-size 4
    python train_torchvision.py --model ssd          --epochs 30 --batch-size 16

Saves to outputs/<model>/:  best.pt, last.pt, history.json
"""
import argparse
import json
import time

import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader

import config
from data.prepare_data import make_splits, load_split
from data.kitti_dataset import KittiDetectionDataset, collate_fn
from models import build, count_params_millions
from engine import train_one_epoch, evaluate_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["faster_rcnn", "ssd"], required=True)
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = ap.parse_args()

    device = torch.device(config.DEVICE)
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: CUDA not available -- training will be extremely slow on CPU.")

    # ---- data (split is created on first run, then reused) ----
    if not (config.SPLIT_DIR / "train.txt").exists():
        make_splits()
    train_ds = KittiDetectionDataset(load_split("train"), train=True)
    val_ds   = KittiDetectionDataset(load_split("val"),   train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- model / optimizer ----
    model = build(args.model).to(device)
    print(f"{args.model}: {count_params_millions(model):.1f}M parameters")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=config.MOMENTUM,
                                weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.LR_STEP_SIZE, gamma=config.LR_GAMMA)

    use_amp = config.USE_AMP and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    ckpt_dir = config.OUTPUT_DIR / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----
    best_map, history = -1.0, []
    for epoch in range(args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, optimizer, train_loader, device,
                               scaler if use_amp else None, epoch)
        scheduler.step()
        metrics = evaluate_map(model, val_loader, device)
        dt = time.time() - t0

        mAP   = float(metrics.get("map", -1))
        mAP50 = float(metrics.get("map_50", -1))
        print(f"[{args.model}] epoch {epoch}: loss={loss:.4f}  "
              f"mAP={mAP:.4f}  mAP@50={mAP50:.4f}  ({dt:.0f}s)")

        history.append({"epoch": epoch, "loss": loss,
                        "map": mAP, "map_50": mAP50, "time_s": dt})
        torch.save(model.state_dict(), ckpt_dir / "last.pt")
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), ckpt_dir / "best.pt")

    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. Best val mAP for {args.model}: {best_map:.4f}  "
          f"(weights: {ckpt_dir/'best.pt'})")


if __name__ == "__main__":
    main()
