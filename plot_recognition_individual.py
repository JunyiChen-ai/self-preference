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


def load_files(model_dir: Path):
    return sorted(model_dir.rglob("*.json"))


def compute_accuracy(root: Path):
    wins = []
    totals = []
    for model_name in MODEL_DIRS:
        model_dir = root / model_name
        files = load_files(model_dir) if model_dir.exists() else []
        total = len(files)
        win = 0
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            src = canonical(data.get("source"))
            recog = canonical(data.get("recognizer"))
            is_yes = bool(data.get("is_yes"))
            if src == recog and is_yes:
                win += 1
            elif src != recog and not is_yes:
                win += 1
        wins.append(win)
        totals.append(total)
    return wins, totals


def plot_bar(root: Path, output: Path):
    wins, totals = compute_accuracy(root)
    ratios = [w / t if t else 0 for w, t in zip(wins, totals)]
    labels = [MODEL_LABELS.get(name, name) for name in MODEL_DIRS]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ratios, color=["#80b1d3", "#8dd3c7", "#fb8072"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recognition accuracy")
    ax.set_title("Qwen3 Individual Recognition")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio * 100:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot recognition individual results")
    parser.add_argument("--input-dir", default="data/recognition_individual")
    parser.add_argument("--output", default="plots/recognition_individual.png")
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    plot_bar(root, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
