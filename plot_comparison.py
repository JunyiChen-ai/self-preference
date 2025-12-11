#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set unified theme
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


MODEL_ORDER = [
    "Qwen_Qwen3-4B-Instruct-2507",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "Qwen_Qwen3-Next-80B-A3B-Instruct",
]


def list_models(root: Path, pattern: str = "") -> List[str]:
    pattern = pattern.strip().lower()
    if pattern:
        matches = sorted(
            [p.name for p in root.iterdir() if p.is_dir() and pattern in p.name.lower()]
        )
        if not matches:
            raise SystemExit(
                f"No model directories under {root} match pattern '{pattern}'"
            )
        return matches
    if all((root / name).is_dir() for name in MODEL_ORDER):
        return MODEL_ORDER
    fallback = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if fallback:
        return fallback
    raise SystemExit(f"No model directories found under {root}")


def accumulate_matrix(root: Path, model_names: List[str]):
    size = len(model_names)
    wins = np.zeros((size, size), dtype=float)  # rows: ai, cols: human
    totals = np.zeros((size, size), dtype=float)
    for c, human in enumerate(model_names):
        human_dir = root / human
        if not human_dir.is_dir():
            continue
        for r, ai in enumerate(model_names):
            ai_dir = human_dir / ai
            if not ai_dir.is_dir():
                continue
            json_files = sorted(ai_dir.glob("*.json"))
            total_rounds = len(json_files) * 2
            win_count = 0
            for file_path in json_files:
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for key in ("ai_first_win", "ai_second_win"):
                    if data.get(key):
                        win_count += 1
            wins[r, c] = win_count
            totals[r, c] = total_rounds
    return wins, totals


def dataset_label(root: Path) -> str:
    return "News" if "news" in str(root).lower() else "Paper"


def plot_matrix(wins, totals, model_names, output_path, label: str):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(wins, totals, out=np.zeros_like(wins), where=totals != 0)
    
    def to_size_label(name: str) -> str:
        for token in ("4B", "30B", "80B"):
            if token in name:
                return token
        return name

    fig, ax = plt.subplots(figsize=(6, 5))
    short_labels = [to_size_label(name) for name in model_names]
    
    sns.heatmap(ratios, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=short_labels, yticklabels=short_labels,
                vmin=0, vmax=1, cbar_kws={'label': 'Win Rate'}, ax=ax,
                square=True, linewidths=.5, linecolor='white')
    
    ax.set_title(f"Qwen3 Self-Preference (Pairwise, {label})")
    # Keep axis labels optional or minimal as per original
    ax.set_xlabel("Model B")
    ax.set_ylabel("Model A")
    
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix across model comparisons")
    parser.add_argument("--input-dir", default="data/comparison")
    parser.add_argument("--output", default="plots/comparison_matrix.png")
    parser.add_argument(
        "--model",
        default="",
        help="Optional substring to filter model directories inside the input folder",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    model_names = list_models(root, args.model)
    wins, totals = accumulate_matrix(root, model_names)
    label = dataset_label(root)
    plot_matrix(wins, totals, model_names, Path(args.output), label)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
