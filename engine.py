"""
Training / evaluation engine for the torchvision detectors.

  * train_one_epoch : one pass over the training set (AMP optional)
  * evaluate_map    : COCO-style mAP on the val set via torchmetrics
  * measure_fps     : pure inference throughput (batch=1, GPU-synced, warmed up)
"""
import time
import torch
from torch.amp import autocast
from torchmetrics.detection import MeanAveragePrecision


def train_one_epoch(model, optimizer, loader, device, scaler=None,
                    epoch=0, print_freq=50):
    model.train()
    running = 0.0
    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        if scaler is not None:                      # mixed-precision path (CUDA)
            with autocast("cuda"):
                loss_dict = model(images, targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:                                       # full-precision path
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

        running += loss.item()
        if i % print_freq == 0:
            print(f"    epoch {epoch}  iter {i:>4}/{len(loader)}  "
                  f"loss {loss.item():.4f}")
    return running / max(len(loader), 1)


@torch.no_grad()
def evaluate_map(model, loader, device):
    """Returns the torchmetrics result dict (map, map_50, map_75, per-class, ...)."""
    model.eval()
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        preds = [{k: v.detach().cpu() for k, v in o.items()} for o in outputs]
        tgts  = [{"boxes": t["boxes"], "labels": t["labels"]} for t in targets]
        metric.update(preds, tgts)
    result = metric.compute()
    # Convert tensors to plain Python so the dict is JSON-serializable.
    return {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in result.items()}


@torch.no_grad()
def measure_fps(model, loader, device, n_warmup=10, n_measure=200):
    """Frames per second at batch size 1 (inference only)."""
    model.eval()
    it = iter(loader)

    for _ in range(n_warmup):                        # warm up CUDA kernels / caches
        try:
            images, _ = next(it)
        except StopIteration:
            it = iter(loader); images, _ = next(it)
        model([img.to(device) for img in images])
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0, count = time.time(), 0
    for _ in range(n_measure):
        try:
            images, _ = next(it)
        except StopIteration:
            break
        model([img.to(device) for img in images])
        count += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return count / dt if dt > 0 else float("nan")
