#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


MODEL_ORDER = [
    "Qwen_Qwen3-4B-Instruct-2507",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "Qwen_Qwen3-Next-80B-A3B-Instruct",
]


def list_models(root: Path) -> List[str]:
    if all((root / name).is_dir() for name in MODEL_ORDER):
        return MODEL_ORDER
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


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


def plot_matrix(wins, totals, model_names, output_path):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(wins, totals, out=np.zeros_like(wins), where=totals != 0)
    def to_size_label(name: str) -> str:
        for token in ("4B", "30B", "80B"):
            if token in name:
                return token
        return name

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.color_palette("rocket", as_cmap=True)
    im = ax.imshow(ratios, vmin=0, vmax=1, cmap=cmap)
    short_labels = [to_size_label(name) for name in model_names]
    ax.set_xticks(range(len(model_names)), labels=short_labels)
    ax.set_yticks(range(len(model_names)), labels=short_labels)
    ax.set_title("Qwen3 Self-Preference")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for i in range(len(model_names)):
        for j in range(len(model_names)):
            if totals[i, j] == 0:
                label = "-"
            else:
                label = f"{ratios[i, j]:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ratio")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Confusion matrix across model comparisons")
    parser.add_argument("--input-dir", default="data/comparison")
    parser.add_argument("--output", default="plots/comparison_matrix.png")
    args = parser.parse_args()

    root = Path(args.input_dir)
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    model_names = list_models(root)
    wins, totals = accumulate_matrix(root, model_names)
    plot_matrix(wins, totals, model_names, Path(args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
