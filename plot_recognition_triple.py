#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
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


def format_source_label(name: str) -> str:
    return "Human" if name.lower() == "human" else format_model_label(name)


def palette(count: int):
    if count <= len(DEFAULT_COLORS):
        return DEFAULT_COLORS[:count]
    return sns.color_palette("tab10", count)


def canonical(name: str) -> str:
    if not name:
        return ""
    return name.replace("/", "_")


def collect_scores(root: Path, model_dirs: list[str]):
    wins = []
    totals = []
    sources = list(model_dirs) + ["human"]
    source_index = {name: idx for idx, name in enumerate(model_dirs)}
    human_idx = len(model_dirs)
    source_counts = np.zeros((len(sources), len(model_dirs)))
    for col, model_name in enumerate(model_dirs):
        model_dir = root / model_name
        files = sorted(model_dir.glob("*.json")) if model_dir.exists() else []
        total = len(files)
        win = 0
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            recognizer = canonical(data.get("recognizer"))
            if recognizer != model_name:
                continue
            selected = canonical(data.get("selected_source"))
            if selected in source_index:
                source_counts[source_index[selected], col] += 1
            elif selected == "human":
                source_counts[human_idx, col] += 1
            if selected == recognizer:
                win += 1
        wins.append(win)
        totals.append(total)
    return wins, totals, source_counts


def dataset_label(root: Path) -> str:
    return "News" if "news" in str(root).lower() else "Paper"


def plot_bar(root: Path, output: Path, label: str, model_dirs: list[str]):
    wins, totals, _ = collect_scores(root, model_dirs)
    ratios = [w / t if t else 0 for w, t in zip(wins, totals)]
    labels = [format_model_label(name) for name in model_dirs]
    colors = palette(len(model_dirs))

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, ratios, color=colors, width=0.6, edgecolor='white', linewidth=1)
    
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recognition Rate")
    ax.set_title(f"Qwen3 Triple Recognition ({label})")
    
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


def plot_matrix(root: Path, output: Path, label: str, model_dirs: list[str]):
    _, totals, source_counts = collect_scores(root, model_dirs)
    denom = np.array(totals, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(source_counts, denom, out=np.zeros_like(source_counts), where=denom > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    col_labels = [format_model_label(name) for name in model_dirs]
    row_labels = [format_source_label(name) for name in list(model_dirs) + ["human"]]
    
    sns.heatmap(ratios, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=col_labels, yticklabels=row_labels,
                vmin=0, vmax=1, cbar_kws={'label': 'Selection Ratio'}, ax=ax,
                square=True, linewidths=.5, linecolor='white')
    
    ax.set_xlabel("Recognizer Model")
    ax.set_ylabel("Selected Source")
    ax.set_title(f"Qwen3 Triple Recognition Matrix ({label})")
    
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot triple recognition results")
    parser.add_argument("--input-dir", default="data/recognition_triple")
    parser.add_argument("--output", default="plots/recognition_triple.png")
    parser.add_argument("--mode", choices=["bar", "matrix"], default="bar")
    parser.add_argument(
        "--model",
        default="",
        help="Optional substring to filter model directories inside the input folder",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    output_path = Path(args.output)
    label = dataset_label(root)
    model_dirs = pick_model_dirs(root, args.model)
    if args.mode == "bar":
        plot_bar(root, output_path, label, model_dirs)
    else:
        plot_matrix(root, output_path, label, model_dirs)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
