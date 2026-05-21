"""
Central configuration shared by every script in this project.

Edit KITTI_ROOT to point at your downloaded dataset and the rest will follow.
The same train/val split (defined here + written by data/prepare_data.py) is used
by the torchvision detectors (Faster R-CNN, SSD) AND by the YOLO converter, so all
three pipelines see exactly the same images -- this is the key fairness control
from the proposal.
"""
from pathlib import Path

try:
    import torch
    _CUDA = torch.cuda.is_available()
except Exception:  # torch not importable at config-read time
    _CUDA = False

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Root of the downloaded KITTI 2D object detection dataset.
# Expected layout after you unzip the official KITTI files:
#   KITTI_ROOT/
#     training/
#       image_2/   000000.png 000001.png ...   (7481 images, all have labels)
#       label_2/   000000.txt 000001.txt ...
#     testing/
#       image_2/   ...                          (no public labels -> not used for mAP)
KITTI_ROOT   = Path("data/kitti")                       # <-- CHANGE if needed
TRAIN_IMAGES = KITTI_ROOT / "training" / "image_2"
TRAIN_LABELS = KITTI_ROOT / "training" / "label_2"

SPLIT_DIR  = Path("data/splits")        # train.txt / val.txt written here
YOLO_ROOT  = Path("data/kitti_yolo")    # YOLO-format dataset built here
OUTPUT_DIR = Path("outputs")            # checkpoints, logs, figures, results

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #
# The three driving-relevant classes from the proposal.
CLASSES = ["car", "pedestrian", "cyclist"]

# Map raw KITTI "type" field -> our class name. Anything not listed is ignored.
# Raw KITTI types: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram,
#                  Misc, DontCare.
# The defaults below keep the three classes strict. Uncomment lines to merge
# related categories (a common choice on the KITTI benchmark).
KITTI_TO_CLASS = {
    "Car": "car",
    # "Van": "car",
    # "Truck": "car",
    "Pedestrian": "pedestrian",
    # "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
}

# torchvision detectors reserve label 0 for "background", so object ids start at 1.
# YOLO uses 0-indexed ids with no background class.
CLASS_TO_TVID   = {name: i + 1 for i, name in enumerate(CLASSES)}   # car -> 1 ...
CLASS_TO_YOLOID = {name: i     for i, name in enumerate(CLASSES)}   # car -> 0 ...
NUM_CLASSES_TV  = len(CLASSES) + 1   # +1 for background (torchvision convention)

# --------------------------------------------------------------------------- #
# Train / val split
# --------------------------------------------------------------------------- #
VAL_FRACTION = 0.2
SPLIT_SEED   = 42

# --------------------------------------------------------------------------- #
# Training hyper-parameters (torchvision models)
# --------------------------------------------------------------------------- #
DEVICE        = "cuda" if _CUDA else "cpu"
NUM_WORKERS   = 4
EPOCHS        = 20
BATCH_SIZE    = 4          # per GPU; lower if Faster R-CNN OOMs, raise for SSD
LEARNING_RATE = 0.005
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
LR_STEP_SIZE  = 8
LR_GAMMA      = 0.1
USE_AMP       = True       # automatic mixed precision (faster, less VRAM on CUDA)

# Faster R-CNN internal resize bounds. KITTI images are ~1242x375, so a wider
# max_size than the COCO default (1333) is unnecessary; 1280 keeps full width.
FRCNN_MIN_SIZE = 600
FRCNN_MAX_SIZE = 1280

# --------------------------------------------------------------------------- #
# YOLO
# --------------------------------------------------------------------------- #
YOLO_IMGSZ = 640
