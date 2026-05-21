"""
KITTI dataset utilities for the torchvision detectors (Faster R-CNN and SSD).

`parse_kitti_label` is shared with the YOLO converter so both pipelines apply the
exact same class mapping and box filtering.
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

import config


def parse_kitti_label(label_path):
    """Parse one KITTI label_2 .txt file.

    KITTI line format (space separated):
        type truncated occluded alpha  x1 y1 x2 y2  h w l  X Y Z  rotation_y
    We only need `type` (field 0) and the 2D box (fields 4..7, pixel coords).

    Returns a list of (class_name, [x1, y1, x2, y2]) for classes we keep.
    """
    objs = []
    if not label_path.exists():
        return objs
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            cls = config.KITTI_TO_CLASS.get(parts[0])
            if cls is None:
                continue
            x1, y1, x2, y2 = map(float, parts[4:8])
            if x2 <= x1 or y2 <= y1:        # drop degenerate boxes
                continue
            objs.append((cls, [x1, y1, x2, y2]))
    return objs


class KittiDetectionDataset(Dataset):
    """Returns (image_tensor, target_dict) in the format torchvision expects.

    image_tensor : float CHW in [0, 1] (the detector normalizes internally)
    target_dict  : {boxes [N,4] xyxy, labels [N], image_id, area, iscrowd}
    """

    def __init__(self, image_ids, train=False):
        self.image_ids = list(image_ids)
        self.train = train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img = Image.open(config.TRAIN_IMAGES / f"{img_id}.png").convert("RGB")
        W, H = img.size

        objs = parse_kitti_label(config.TRAIN_LABELS / f"{img_id}.txt")
        boxes  = torch.as_tensor([o[1] for o in objs], dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor([config.CLASS_TO_TVID[o[0]] for o in objs], dtype=torch.int64)

        img = F.to_tensor(img)  # CHW float in [0, 1]

        # Light augmentation: random horizontal flip (mirror image + boxes).
        if self.train and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])
            if boxes.numel() > 0:
                x1 = boxes[:, 0].clone()
                x2 = boxes[:, 2].clone()
                boxes[:, 0] = W - x2
                boxes[:, 2] = W - x1

        if boxes.numel() > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((labels.shape[0],), dtype=torch.int64),
        }
        return img, target


def collate_fn(batch):
    """Detection batches have variable #objects per image, so we keep them as
    tuples rather than stacking."""
    return tuple(zip(*batch))
