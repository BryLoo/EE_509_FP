"""
Build the train/val split ONCE and reuse it everywhere.

The official KITTI test set has no public labels, so (as is standard in the KITTI
literature) we split the 7481 labelled *training* images into our own train/val
sets. Both the torchvision dataset and the YOLO converter read these same files,
guaranteeing identical splits across all three pipelines.

Run directly to (re)generate:  python -m data.prepare_data
"""
import random
import config


def make_splits():
    config.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in config.TRAIN_IMAGES.glob("*.png"))
    if not ids:
        raise FileNotFoundError(
            f"No .png images found in {config.TRAIN_IMAGES}. "
            "Download the KITTI 2D object detection dataset first (see README)."
        )
    rng = random.Random(config.SPLIT_SEED)
    rng.shuffle(ids)
    n_val = int(len(ids) * config.VAL_FRACTION)
    val_ids   = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    (config.SPLIT_DIR / "train.txt").write_text("\n".join(train_ids))
    (config.SPLIT_DIR / "val.txt").write_text("\n".join(val_ids))
    print(f"Wrote splits to {config.SPLIT_DIR}: "
          f"{len(train_ids)} train / {len(val_ids)} val")
    return train_ids, val_ids


def load_split(name):
    path = config.SPLIT_DIR / f"{name}.txt"
    if not path.exists():
        make_splits()
    return path.read_text().split()


if __name__ == "__main__":
    make_splits()
