#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set unified theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

DEFAULT_MODEL_DIRS = [
    "Qwen_Qwen3-4B-Instruct-2507",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "Qwen_Qwen3-Next-80B-A3B-Instruct",
]

DEFAULT_MODEL_LABELS = {
    "Qwen_Qwen3-4B-Instruct-2507": "4B",
    "Qwen_Qwen3-30B-A3B-Instruct-2507": "30B",
    "Qwen_Qwen3-Next-80B-A3B-Instruct": "80B",
}

# Unified color palette
DEFAULT_COLORS = ["#4E79A7", "#F28E2B", "#E15759"]

SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([bBkK])")


def pick_model_dirs(root: Path, keyword: str) -> list[str]:
    keyword = (keyword or "").strip().lower()
    if keyword:
        matches = sorted(
            [p.name for p in root.iterdir() if p.is_dir() and keyword in p.name.lower()]
        )
        if not matches:
            raise SystemExit(
                f"No model directories under {root} match pattern '{keyword}'"
            )
        return matches
    prioritized = [name for name in DEFAULT_MODEL_DIRS if (root / name).exists()]
    if prioritized:
        return prioritized
    fallback = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if fallback:
        return fallback
    raise SystemExit(f"No model directories found under {root}")


def format_model_label(name: str) -> str:
    if name in DEFAULT_MODEL_LABELS:
        return DEFAULT_MODEL_LABELS[name]
    match = SIZE_RE.search(name)
    if match:
        return f"{match.group(1)}{match.group(2).upper()}"
    return name


def palette(count: int):
    if count <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:count]
    return sns.color_palette("tab10", count)


def canonical(name: str) -> str:
    if not name:
        return ""
    return name.replace("/", "_")


def load_files(model_dir: Path):
    return sorted(model_dir.rglob("*.json"))


def compute_accuracy(root: Path, model_dirs: list[str]):
    wins = []
    totals = []
    for model_name in model_dirs:
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


def dataset_label(root: Path) -> str:
    return "News" if "news" in str(root).lower() else "Paper"


def plot_bar(root: Path, output: Path, label: str, model_dirs: list[str]):
    wins, totals = compute_accuracy(root, model_dirs)
    ratios = [w / t if t else 0 for w, t in zip(wins, totals)]
    labels = [format_model_label(name) for name in model_dirs]
    colors = palette(len(model_dirs))

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, ratios, color=colors, width=0.6, edgecolor='white', linewidth=1)
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recognition Accuracy")
    ax.set_title(f"Qwen3 Individual Recognition ({label})")
    
    sns.despine()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar, ratio in zip(bars, ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                f"{ratio * 100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight='medium')

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot recognition individual results")
    parser.add_argument("--input-dir", default="data/recognition_individual")
    parser.add_argument("--output", default="plots/recognition_individual.png")
    parser.add_argument(
        "--model",
        default="",
        help="Optional substring to filter model directories inside the input folder",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    label = dataset_label(root)
    model_dirs = pick_model_dirs(root, args.model)
    plot_bar(root, Path(args.output), label, model_dirs)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
