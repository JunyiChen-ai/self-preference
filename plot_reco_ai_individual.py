#!/usr/bin/env python3
"""Plot AI-recognition preference matrix for individual evaluator/source pairs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

DATASET_PATHS: Dict[str, Path] = {
    "paper": Path("self-preference/data/paper/reco_ai_individual"),
    "news": Path("self-preference/data/news_segment/reco_ai_individual"),
    "news_segment": Path("self-preference/data/news_segment/reco_ai_individual"),
}


def canonical(name: str) -> str:
    return name.replace("/", "_") if name else ""


def format_label(name: str) -> str:
    if not name:
        return name
    clean = canonical(name)
    lowered = clean.lower()
    if lowered == "human":
        return "Human"
    tokens = lowered.replace("-", "_").split("_")
    size = ""
    for tok in tokens:
        if tok.endswith("b") or tok.endswith("k"):
            size = tok.upper()
            break
    if "qwen" in lowered:
        version = "Qwen"
        if "qwen3" in lowered:
            version = "Qwen3"
        return f"{version} {size}".strip()
    if "gemma" in lowered:
        version = "Gemma"
        return f"{version} {size}".strip()
    if "gpt" in lowered:
        return clean.replace("_", " ")
    return clean


def dataset_label(name: str) -> str:
    lowered = name.lower()
    if "news" in lowered:
        return "News"
    if "paper" in lowered:
        return "Paper"
    return name.title()


def discover_axes(root: Path) -> Tuple[List[str], List[str]]:
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")
    evaluators = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not evaluators:
        raise SystemExit(f"No evaluator directories found under {root}")
    sources: List[str] = []
    seen = set()
    for evaluator in evaluators:
        subdirs = sorted([p.name for p in (root / evaluator).iterdir() if p.is_dir()])
        for sub in subdirs:
            if sub not in seen:
                sources.append(sub)
                seen.add(sub)
    ordered_sources: List[str] = []
    added = set()
    for name in evaluators:
        if name in seen and name not in added:
            ordered_sources.append(name)
            added.add(name)
    for name in sources:
        if name not in added and name.lower() != "human":
            ordered_sources.append(name)
            added.add(name)
    if "human" in seen and "human" not in added:
        ordered_sources.append("human")
    return evaluators, ordered_sources


def compute_matrix(root: Path, evaluators: List[str], sources: List[str]) -> np.ndarray:
    matrix = np.zeros((len(sources), len(evaluators)))
    for col, evaluator in enumerate(evaluators):
        for row, source in enumerate(sources):
            pair_dir = root / evaluator / source
            if not pair_dir.exists():
                continue
            files = list(pair_dir.glob("*.json"))
            if not files:
                continue
            total = len(files)
            ai_count = 0
            for file_path in files:
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if str(data.get("label", "")).lower() == "ai":
                    ai_count += 1
            matrix[row, col] = ai_count / total if total else 0.0
    return matrix


def plot_heatmap(matrix: np.ndarray, evaluators: List[str], sources: List[str], label: str, output: Path):
    fig, ax = plt.subplots(figsize=(10, 8))
    col_labels = [format_label(name) for name in evaluators]
    row_labels = [format_label(name) for name in sources]

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Label=AI Ratio"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=False,
    )
    ax.set_xlabel("Recognizer")
    ax.set_ylabel("Generator")
    ax.set_title(f"Individual AI Recognition Heatmap ({label})")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reco_ai_individual confusion matrix")
    parser.add_argument("dataset", help="Dataset name (e.g., paper or news)")
    parser.add_argument(
        "--input-dir",
        default="",
        help="Override input directory; otherwise dataset must be in the predefined map",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output image path; defaults to self-preference/plots/indi_reco_ai_<dataset>.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input_dir:
        root = Path(args.input_dir)
    else:
        key = args.dataset.lower()
        if key not in DATASET_PATHS:
            raise SystemExit(
                f"Unknown dataset '{args.dataset}'. Provide --input-dir or choose from: {', '.join(DATASET_PATHS)}"
            )
        root = DATASET_PATHS[key]
    evaluators, sources = discover_axes(root)
    matrix = compute_matrix(root, evaluators, sources)
    label = dataset_label(args.dataset)
    output = Path(args.output) if args.output else Path("self-preference/plots") / f"indi_reco_ai_{args.dataset}.png"
    plot_heatmap(matrix, evaluators, sources, label, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
