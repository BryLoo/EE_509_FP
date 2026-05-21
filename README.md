# EE 509 Final Project — Object Detection Pipeline Comparison

Compare a **two-stage** detector (Faster R-CNN) against two **single-stage**
detectors (SSD, YOLO) on the **KITTI 2D Object Detection** benchmark for the three
driving-relevant classes: **car, pedestrian, cyclist**. Metrics: **mAP** (accuracy),
**FPS** (speed), and qualitative detection images.

> **Course-policy note.** The assignment prohibits AI tools for **writing the
> project report**. This repository is *simulation code* (a separate deliverable),
> not the report. Write the report and presentation yourselves, and check with the
> instructor if you're unsure how the policy applies to code.

---

## 1. Install

A CUDA-capable GPU is required for reasonable training times.

```bash
python -m venv .venv && source .venv/bin/activate     # optional
# Install the CUDA build of torch matching your driver (see pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Get the KITTI dataset

Register and download from the official site (free, requires an email):
<https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d> — you need
**"left color images of object data set"** and **"training labels of object data set."**

Unzip so the layout matches `config.KITTI_ROOT` (default `data/kitti`):

```
data/kitti/
  training/
    image_2/   000000.png 000001.png ...   (7481 images)
    label_2/   000000.txt 000001.txt ...
```

(The official `testing/` set has no public labels, so we split the labelled
`training/` set into our own train/val — standard practice for KITTI experiments.)

## 3. Build the shared split

```bash
python -m data.prepare_data        # writes data/splits/{train,val}.txt
```

Every pipeline reads these files, so all three train and evaluate on **identical
images** — the main fairness control from the proposal.

## 4. Train

```bash
# Two-stage
python train_torchvision.py --model faster_rcnn --epochs 20 --batch-size 4

# Single-stage (SSD can use a larger batch)
python train_torchvision.py --model ssd --epochs 30 --batch-size 16

# Single-stage (YOLO — weights auto-download; run several for the version study)
python train_yolo.py --weights yolov8n.pt
python train_yolo.py --weights yolov8s.pt
python train_yolo.py --weights yolo11n.pt
```

## 5. Evaluate (mAP + FPS)

```bash
python eval_torchvision.py --model faster_rcnn
python eval_torchvision.py --model ssd
# YOLO writes its results_*.json automatically during train_yolo.py
```

## 6. Qualitative figures + final comparison

```bash
python visualize.py --model faster_rcnn --n 8     # boxes drawn on val images
python visualize.py --model ssd --n 8
python compare.py                                  # comparison.csv + comparison.png
```

`compare.py` aggregates every `outputs/results_*.json` into one table and a
three-panel chart (mAP bar, FPS bar, and the speed-vs-accuracy plane) — ready to
drop into the report and slides.

---

## File map

| File | Purpose |
|------|---------|
| `config.py` | Paths, class mapping, hyper-parameters (edit this first) |
| `data/prepare_data.py` | Deterministic train/val split, reused everywhere |
| `data/kitti_dataset.py` | KITTI label parser + torchvision detection dataset |
| `data/kitti_to_yolo.py` | KITTI → YOLO format conversion (same split) |
| `models.py` | Faster R-CNN & SSD factories (transfer learning) |
| `engine.py` | Train loop, mAP eval (torchmetrics), FPS benchmark |
| `train_torchvision.py` | Train Faster R-CNN / SSD |
| `train_yolo.py` | Train + eval YOLO (Ultralytics) |
| `eval_torchvision.py` | mAP + FPS for a trained checkpoint |
| `visualize.py` | Draw predictions for qualitative figures |
| `compare.py` | Aggregate results → table + chart |

## Notes / honest caveats (worth discussing in the report)

- **Transfer learning.** ~6000 training images is far too few to train detectors
  from scratch, so Faster R-CNN starts from COCO weights and SSD from an
  ImageNet VGG16 backbone. State this in the report — it affects the absolute
  numbers (not the qualitative ranking).
- **"Same input resolution" is only approximate.** SSD300's anchors fix its input
  at 300×300; Faster R-CNN and YOLO use larger inputs. True pixel parity across all
  three is architecturally impossible — call this out when interpreting the
  speed/accuracy gap.
- **FPS is reported inference-only**, batch size 1, after GPU warm-up, and depends
  heavily on the specific GPU. Report which GPU you used.
- **Reproducibility.** The split seed lives in `config.py` (`SPLIT_SEED = 42`).
