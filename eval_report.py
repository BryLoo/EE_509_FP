import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import config
from data.prepare_data import load_split
from models import build

ID_TO_NAME = {v: k for k, v in config.CLASS_TO_TVID.items()}
CLASSES = list(config.CLASSES)
NC = len(CLASSES)
BG = NC  # background index


def xywhn_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [x1, y1, x2, y2]


def load_yolo_labels(img_id, img_w, img_h):
    label_path = Path("data/kitti_yolo/labels/val") / f"{img_id}.txt"
    boxes, labels = [], []

    if not label_path.exists():
        return boxes, labels

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])
            boxes.append(xywhn_to_xyxy(xc, yc, w, h, img_w, img_h))
            labels.append(cls_id)

    return boxes, labels


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def plot_f1_confidence_curve(all_preds, all_gts, out_path, iou_thresh=0.5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    conf_grid = np.linspace(0.001, 1.0, 200)

    for cls_id, cls_name in enumerate(CLASSES):
        f1_scores = []

        for conf_thresh in conf_grid:
            tp = 0
            fp = 0
            fn = 0

            for preds, gts in zip(all_preds, all_gts):
                pred_boxes, pred_labels, pred_scores = preds
                gt_boxes, gt_labels = gts

                # Keep only predictions for this class above confidence threshold
                cls_pred_indices = [
                    i for i, (label, score) in enumerate(zip(pred_labels, pred_scores))
                    if label == cls_id and score >= conf_thresh
                ]

                cls_gt_indices = [
                    i for i, label in enumerate(gt_labels)
                    if label == cls_id
                ]

                matched_gt = set()

                # Sort predictions by confidence
                cls_pred_indices = sorted(
                    cls_pred_indices,
                    key=lambda i: pred_scores[i],
                    reverse=True
                )

                for pi in cls_pred_indices:
                    best_iou = 0.0
                    best_gi = -1

                    for gi in cls_gt_indices:
                        if gi in matched_gt:
                            continue

                        iou = box_iou(pred_boxes[pi], gt_boxes[gi])
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi

                    if best_iou >= iou_thresh and best_gi >= 0:
                        tp += 1
                        matched_gt.add(best_gi)
                    else:
                        fp += 1

                fn += len(cls_gt_indices) - len(matched_gt)

            precision = tp / max(tp + fp, 1e-9)
            recall = tp / max(tp + fn, 1e-9)
            f1 = 2 * precision * recall / max(precision + recall, 1e-9)

            f1_scores.append(f1)

        ax.plot(conf_grid, f1_scores, label=cls_name)

    ax.set_title(f"F1-Confidence Curve at IoU={iou_thresh}")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("F1 Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def make_predictor(model_type, weights, device, iou=0.7):
    if model_type in ("faster_rcnn", "ssd"):
        model = build(model_type)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device).eval()

        # Lower internal torchvision filtering so PR curve sees low-confidence detections too
        if model_type == "faster_rcnn" and hasattr(model, "roi_heads"):
            model.roi_heads.score_thresh = 0.001
            model.roi_heads.detections_per_img = 1000
            model.roi_heads.nms_thresh = iou

        if model_type == "ssd":
            if hasattr(model, "score_thresh"):
                model.score_thresh = 0.001
            if hasattr(model, "detections_per_img"):
                model.detections_per_img = 1000
            if hasattr(model, "nms_thresh"):
                model.nms_thresh = iou

        @torch.no_grad()
        def predict(img, score_thresh):
            x = F.to_tensor(img).to(device)
            out = model([x])[0]
            keep = out["scores"] >= score_thresh

            boxes = out["boxes"][keep].cpu().numpy()
            labels = out["labels"][keep].cpu().numpy()
            scores = out["scores"][keep].cpu().numpy()

            mapped_labels = []
            for label in labels:
                name = ID_TO_NAME.get(int(label), "?")
                mapped_labels.append(CLASSES.index(name) if name in CLASSES else -1)

            keep_valid = [i for i, label in enumerate(mapped_labels) if label >= 0]

            return (
                boxes[keep_valid].tolist(),
                [mapped_labels[i] for i in keep_valid],
                scores[keep_valid].tolist(),
            )

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

            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            labels = r.boxes.cls.cpu().int().numpy().tolist()
            scores = r.boxes.conf.cpu().numpy().tolist()
            return boxes, labels, scores

        return predict

    raise ValueError(f"Unknown model type: {model_type}")


def build_confusion_matrix(all_preds, all_gts, iou_thresh):
    """
    Matrix layout:
    rows = predicted class
    cols = true class

    Last row = background prediction / missed object
    Last col = true background / false positive
    """
    cm = np.zeros((NC + 1, NC + 1), dtype=np.float64)

    for preds, gts in zip(all_preds, all_gts):
        pred_boxes, pred_labels, pred_scores = preds
        gt_boxes, gt_labels = gts

        matched_gt = set()

        order = np.argsort(-np.array(pred_scores)) if len(pred_scores) else []

        for pi in order:
            pbox = pred_boxes[pi]
            plabel = pred_labels[pi]

            best_iou = 0.0
            best_gi = -1

            for gi, gbox in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue

                iou = box_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_gi >= 0 and best_iou >= iou_thresh:
                true_label = gt_labels[best_gi]
                cm[plabel, true_label] += 1
                matched_gt.add(best_gi)
            else:
                cm[plabel, BG] += 1

        for gi, true_label in enumerate(gt_labels):
            if gi not in matched_gt:
                cm[BG, true_label] += 1

    return cm


def plot_confusion_matrix(cm, out_path, normalize=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = CLASSES + ["background"]

    if normalize:
        denom = cm.sum(axis=0, keepdims=True)
        plot_cm = np.divide(cm, denom, out=np.zeros_like(cm), where=denom != 0)
        title = "Normalized Confusion Matrix"
        fmt = ".2f"
    else:
        plot_cm = cm
        title = "Confusion Matrix"
        fmt = ".0f"

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        plot_cm,
        interpolation="nearest",
        cmap="Blues",
        vmin=0,
        vmax=1 if normalize else None,
    )
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    thresh = plot_cm.max() / 2.0 if plot_cm.max() > 0 else 0.5
    for i in range(plot_cm.shape[0]):
        for j in range(plot_cm.shape[1]):
            value = plot_cm[i, j]
            if value == 0:
                continue
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, recalls, precisions), x)
    return float(ap)


def evaluate_map(all_preds, all_gts, iou_thresholds):
    results = {}

    for iou_thresh in iou_thresholds:
        aps = []

        for cls_id in range(NC):
            cls_preds = []
            total_gt = 0

            for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
                pred_boxes, pred_labels, pred_scores = preds
                gt_boxes, gt_labels = gts

                gt_for_cls = [i for i, label in enumerate(gt_labels) if label == cls_id]
                total_gt += len(gt_for_cls)

                for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                    if label == cls_id:
                        cls_preds.append((img_idx, score, box))

            cls_preds.sort(key=lambda x: x[1], reverse=True)

            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            matched = {}

            for pred_idx, (img_idx, score, pbox) in enumerate(cls_preds):
                gt_boxes, gt_labels = all_gts[img_idx]
                gt_indices = [i for i, label in enumerate(gt_labels) if label == cls_id]

                best_iou = 0.0
                best_gt = -1

                for gi in gt_indices:
                    iou = box_iou(pbox, gt_boxes[gi])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gi

                key = (img_idx, best_gt)

                if best_iou >= iou_thresh and best_gt >= 0 and key not in matched:
                    tp[pred_idx] = 1
                    matched[key] = True
                else:
                    fp[pred_idx] = 1

            if total_gt == 0:
                aps.append(0.0)
                continue

            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            recalls = tp_cum / max(total_gt, 1)
            precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

            ap = compute_ap(recalls, precisions)
            aps.append(ap)

        results[float(iou_thresh)] = {
            "map": float(np.mean(aps)),
            "ap_per_class": {CLASSES[i]: float(aps[i]) for i in range(NC)},
        }

    map50 = results[0.5]["map"]
    map5095 = float(np.mean([results[float(t)]["map"] for t in iou_thresholds]))

    return map50, map5095, results


def plot_pr_curve(all_preds, all_gts, out_path, iou_thresh=0.5):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    for cls_id, cls_name in enumerate(CLASSES):
        cls_preds = []
        total_gt = 0

        for img_idx, (preds, gts) in enumerate(zip(all_preds, all_gts)):
            pred_boxes, pred_labels, pred_scores = preds
            gt_boxes, gt_labels = gts

            total_gt += sum(1 for label in gt_labels if label == cls_id)

            for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
                if label == cls_id:
                    cls_preds.append((img_idx, score, box))

        cls_preds.sort(key=lambda x: x[1], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        matched = {}

        for pred_idx, (img_idx, score, pbox) in enumerate(cls_preds):
            gt_boxes, gt_labels = all_gts[img_idx]
            gt_indices = [i for i, label in enumerate(gt_labels) if label == cls_id]

            best_iou = 0.0
            best_gt = -1

            for gi in gt_indices:
                iou = box_iou(pbox, gt_boxes[gi])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            key = (img_idx, best_gt)

            if best_iou >= iou_thresh and best_gt >= 0 and key not in matched:
                tp[pred_idx] = 1
                matched[key] = True
            else:
                fp[pred_idx] = 1

        if total_gt == 0 or len(cls_preds) == 0:
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recalls = tp_cum / max(total_gt, 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

        # Create PR envelope like common object-detection plots
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))

        # Make precision monotonically decreasing as recall increases
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])

        # Interpolate onto fixed recall axis
        recall_grid = np.linspace(0, 1, 1000)
        precision_grid = np.interp(recall_grid, recalls, precisions)

        ax.plot(recall_grid, precision_grid, label=cls_name)

    ax.set_title(f"Precision-Recall Curve at IoU={iou_thresh}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_results_csv(results_csv, out_path):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    possible_cols = [
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    cols = [c for c in possible_cols if c in df.columns]

    if not cols:
        print(f"No YOLO-style metric columns found in {results_csv}")
        return

    rows, cols_count = 2, 5
    fig, axes = plt.subplots(rows, cols_count, figsize=(15, 6))
    axes = axes.reshape(-1)

    x = df["epoch"] if "epoch" in df.columns else np.arange(1, len(df) + 1)

    for ax, col in zip(axes, cols):
        y = df[col].astype(float)
        ax.plot(x, y, marker="o", linewidth=1.5, label="results")

        if len(y) >= 3:
            smooth = y.rolling(window=3, center=True, min_periods=1).mean()
            ax.plot(x, smooth, linestyle="--", linewidth=1.2, label="smooth")

        ax.set_title(col)
        ax.set_xlabel("epoch")
        ax.grid(True)

    for ax in axes[len(cols) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["faster_rcnn", "ssd", "yolo"], required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--score-thresh", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--img-limit", type=int, default=None)
    ap.add_argument("--results-csv", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = torch.device(config.DEVICE)
    predict = make_predictor(args.model, args.weights, device, iou=args.iou)

    out_dir = (
        Path(args.out_dir) if args.out_dir else Path("outputs") / "eval" / args.model
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    img_ids = load_split("val")
    if args.img_limit is not None:
        img_ids = img_ids[: args.img_limit]

    all_preds = []
    all_gts = []

    for idx, img_id in enumerate(img_ids, start=1):
        img_path = config.TRAIN_IMAGES / f"{img_id}.png"
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        gt_boxes, gt_labels = load_yolo_labels(img_id, img_w, img_h)
        pred_boxes, pred_labels, pred_scores = predict(img, args.score_thresh)

        all_gts.append((gt_boxes, gt_labels))
        all_preds.append((pred_boxes, pred_labels, pred_scores))

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(img_ids)} images")

    cm = build_confusion_matrix(all_preds, all_gts, args.iou)

    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", normalize=False)
    plot_confusion_matrix(
        cm, out_dir / "confusion_matrix_normalized.png", normalize=True
    )
    plot_pr_curve(all_preds, all_gts, out_dir / "PR_curve.png", iou_thresh=0.5)
    plot_f1_confidence_curve(all_preds, all_gts, out_dir / "F1_curve.png", iou_thresh=args.iou)

    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    map50, map5095, map_detail = evaluate_map(all_preds, all_gts, iou_thresholds)

    tp = np.trace(cm[:NC, :NC])
    pred_object_total = cm[:NC, :].sum()
    true_object_total = cm[:, :NC].sum()

    precision = float(tp / pred_object_total) if pred_object_total > 0 else 0.0
    recall = float(tp / true_object_total) if true_object_total > 0 else 0.0

    summary = {
        "model": args.model,
        "weights": args.weights,
        "score_threshold": args.score_thresh,
        "iou_threshold_for_confusion": args.iou,
        "num_val_images": len(img_ids),
        "precision_at_threshold": precision,
        "recall_at_threshold": recall,
        "mAP50": map50,
        "mAP50-95": map5095,
        "mAP_detail": map_detail,
        "confusion_matrix_raw": cm.tolist(),
        "classes": CLASSES + ["background"],
    }

    with open(out_dir / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if args.results_csv:
        plot_results_csv(args.results_csv, out_dir / "results.png")

    print(f"\nSaved evaluation outputs to: {out_dir}")
    print(
        json.dumps(
            {
                "precision_at_threshold": precision,
                "recall_at_threshold": recall,
                "mAP50": map50,
                "mAP50-95": map5095,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
