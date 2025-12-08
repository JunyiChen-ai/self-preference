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


def gather_scores(root: Path):
    totals = []
    wins = []
    for name in MODEL_DIRS:
        model_dir = root / name
        if not model_dir.exists():
            totals.append(0)
            wins.append(0)
            continue
        files = sorted(model_dir.glob("*.json"))
        total = len(files)
        win = 0
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if canonical(data.get("comparison_model")) != name:
                continue
            if canonical(data.get("selected_source")) == canonical(data.get("comparison_model")):
                win += 1
        totals.append(total)
        wins.append(win)
    return wins, totals


def plot_bar(root: Path, output: Path):
    wins, totals = gather_scores(root)
    ratios = [w / t if t else 0 for w, t in zip(wins, totals)]
    labels = [MODEL_LABELS.get(name, name) for name in MODEL_DIRS]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ratios, color=["#fdae61", "#fee08b", "#d73027"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Self-preference rate")
    ax.set_title("Qwen3 Triple Comparison")
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{ratio * 100:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot triple comparison results")
    parser.add_argument("--input-dir", default="data/comparison_triple")
    parser.add_argument("--output", default="plots/comparison_triple.png")
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    plot_bar(root, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
