"""
Convert the KITTI labels into YOLO format and build the folder structure +
data.yaml that Ultralytics expects -- using the SAME split as the torchvision
detectors so the comparison stays fair.

YOLO label format, one line per box, normalized to [0, 1]:
    <class_id> <x_center> <y_center> <width> <height>

Run directly:  python -m data.kitti_to_yolo
"""
import shutil
from PIL import Image

import config
from data.prepare_data import load_split
from data.kitti_dataset import parse_kitti_label


def _link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())          # cheap; no duplicate disk usage
    except OSError:
        shutil.copy(src, dst)                   # fallback (e.g. Windows w/o admin)


def convert():
    for split in ("train", "val"):
        ids = load_split(split)
        img_out = config.YOLO_ROOT / "images" / split
        lbl_out = config.YOLO_ROOT / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_id in ids:
            src_img = config.TRAIN_IMAGES / f"{img_id}.png"
            with Image.open(src_img) as im:
                W, H = im.size
            _link_or_copy(src_img, img_out / f"{img_id}.png")

            lines = []
            for cls, (x1, y1, x2, y2) in parse_kitti_label(
                config.TRAIN_LABELS / f"{img_id}.txt"
            ):
                cid = config.CLASS_TO_YOLOID[cls]
                cx = ((x1 + x2) / 2) / W
                cy = ((y1 + y2) / 2) / H
                bw = (x2 - x1) / W
                bh = (y2 - y1) / H
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (lbl_out / f"{img_id}.txt").write_text("\n".join(lines))

    yaml_path = config.YOLO_ROOT / "kitti.yaml"
    yaml_path.write_text(
        f"path: {config.YOLO_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(config.CLASSES)}\n"
        f"names: {config.CLASSES}\n"
    )
    print(f"YOLO dataset ready at {config.YOLO_ROOT}  (data file: {yaml_path})")
    return yaml_path


if __name__ == "__main__":
    convert()
