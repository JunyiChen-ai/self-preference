#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

MODEL_DIRS = [
    "Qwen_Qwen3-4B-Instruct-2507",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "Qwen_Qwen3-Next-80B-A3B-Instruct",
]

MODEL_LABELS = {
    "Qwen_Qwen3-4B-Instruct-2507": "4B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "30B",
    "Qwen_Qwen3-Next-80B-A3B-Instruct": "80B",
}


def canonical(name: str) -> str:
    if not name:
        return ""
    return name.replace("/", "_")


def collect_scores(root: Path):
    wins = []
    totals = []
    for model_name in MODEL_DIRS:
        model_dir = root / model_name
        files = sorted(model_dir.glob("*.json")) if model_dir.exists() else []
        total = len(files)
        win = 0
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if canonical(data.get("recognizer")) == canonical(data.get("selected_source")):
                win += 1
        wins.append(win)
        totals.append(total)
    return wins, totals


def plot_bar(root: Path, output: Path):
    wins, totals = collect_scores(root)
    ratios = [w / t if t else 0 for w, t in zip(wins, totals)]
    labels = [MODEL_LABELS.get(name, name) for name in MODEL_DIRS]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#fdae61", "#fee08b", "#d73027"]
    bars = ax.bar(labels, ratios, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recognition rate")
    ax.set_title("Qwen3 Triple Recognition")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio * 100:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot triple recognition results")
    parser.add_argument("--input-dir", default="data/recognition_triple")
    parser.add_argument("--output", default="plots/recognition_triple.png")
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    plot_bar(root, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
