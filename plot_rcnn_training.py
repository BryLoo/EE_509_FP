import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HISTORY = Path("outputs/faster_rcnn/history.json")
OUT_DIR = Path("outputs/eval/faster_rcnn")

with open(HISTORY) as f:
    history = json.load(f)

epochs   = [e["epoch"] + 1 for e in history]
loss     = [e["loss"] for e in history]
map50    = [e["map_50"] for e in history]
map5095  = [e["map"] for e in history]

OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_plot(x, y, title, xlabel, ylabel, path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, y, marker="o", linewidth=1.8, markersize=4, color=color)

    if len(y) >= 3:
        smooth = np.convolve(y, np.ones(3) / 3, mode="same")
        smooth[0] = y[0]
        smooth[-1] = y[-1]
        ax.plot(x, smooth, linestyle="--", linewidth=1.2, color=color, alpha=0.5, label="3-epoch avg")
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

save_plot(epochs, loss,    "Faster R-CNN – Training Loss vs Epoch",    "Epoch", "Loss",      OUT_DIR / "train_loss.png",   color="steelblue")
save_plot(epochs, map50,   "Faster R-CNN – mAP@50 vs Epoch",           "Epoch", "mAP@50",    OUT_DIR / "map50.png",        color="darkorange")
save_plot(epochs, map5095, "Faster R-CNN – mAP@50-95 vs Epoch",        "Epoch", "mAP@50-95", OUT_DIR / "map50_95.png",     color="seagreen")
