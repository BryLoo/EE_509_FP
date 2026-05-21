"""
Collect every outputs/results_*.json into one comparison table + a chart.

Run after evaluating all models:
    python compare.py

Produces:
    outputs/comparison.csv
    outputs/comparison.png   (mAP and FPS bar charts, plus the speed/accuracy plane)
"""
import csv
import json

import config


def load_results():
    rows = []
    for path in sorted(config.OUTPUT_DIR.glob("results_*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def write_csv(rows):
    out = config.OUTPUT_DIR / "comparison.csv"
    fields = ["model", "map", "map_50", "map_75", "fps", "params_millions"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def print_table(rows):
    print(f"\n{'model':<22}{'mAP':>8}{'mAP@50':>9}{'FPS':>9}{'params(M)':>12}")
    print("-" * 60)
    for r in rows:
        print(f"{r['model']:<22}"
              f"{r.get('map', float('nan')):>8.3f}"
              f"{r.get('map_50', float('nan')):>9.3f}"
              f"{r.get('fps', float('nan')):>9.1f}"
              f"{r.get('params_millions', float('nan')):>12}")
    print()


def make_chart(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping chart.")
        return None

    names = [r["model"] for r in rows]
    maps  = [r.get("map", 0) for r in rows]
    fps   = [r.get("fps", 0) for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].bar(names, maps, color="steelblue")
    axes[0].set_title("Accuracy (mAP@[.5:.95])")
    axes[0].set_ylabel("mAP")
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].bar(names, fps, color="indianred")
    axes[1].set_title("Speed (FPS, inference)")
    axes[1].set_ylabel("frames / sec")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].scatter(fps, maps, s=80)
    for n, x, y in zip(names, fps, maps):
        axes[2].annotate(n, (x, y), fontsize=8,
                         xytext=(5, 5), textcoords="offset points")
    axes[2].set_title("Speed vs Accuracy trade-off")
    axes[2].set_xlabel("FPS")
    axes[2].set_ylabel("mAP")

    fig.tight_layout()
    out = config.OUTPUT_DIR / "comparison.png"
    fig.savefig(out, dpi=150)
    return out


def main():
    rows = load_results()
    if not rows:
        print("No results_*.json files found in outputs/. "
              "Run the eval scripts first.")
        return
    print_table(rows)
    csv_path = write_csv(rows)
    png_path = make_chart(rows)
    print(f"Wrote {csv_path}")
    if png_path:
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
