#!/usr/bin/env python3
"""Plot recognition preference matrix for 9 LLMs plus human."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)

SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*-?)?([bBkK])")


def canonical(name: str) -> str:
    if not name:
        return ""
    return name.replace("/", "_")


def extract_size(text: str) -> str:
    match = SIZE_RE.search(text)
    if match:
        return f"{match.group(1)}{match.group(2).upper()}"
    return ""


def format_model_label(name: str) -> str:
    clean = canonical(name)
    lowered = clean.lower()
    if lowered == "human":
        return "Human"

    size = extract_size(clean)

    if "qwen" in lowered:
        version = "Qwen"
        version_match = re.search(r"qwen(\d+)", lowered)
        if version_match:
            version = f"Qwen{version_match.group(1)}"
        return f"{version} {size}".strip()

    if "gemma" in lowered:
        version_match = re.search(r"gemma[-_]?([0-9]+)", lowered)
        version = f"Gemma{version_match.group(1)}" if version_match else "Gemma"
        return f"{version} {size}".strip()

    if size:
        return f"{clean} ({size})"
    return clean


def dataset_label(dataset: str) -> str:
    lowered = dataset.lower()
    if "news" in lowered:
        return "News"
    if "paper" in lowered:
        return "Paper"
    return dataset.title()


def list_recognizers(root: Path) -> list[str]:
    recognizers = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not recognizers:
        raise SystemExit(f"No recognizer directories found under {root}")
    return recognizers


def gather_counts(root: Path, recognizers: list[str]):
    counters: list[Counter[str]] = []
    totals: list[int] = []
    generator_set: set[str] = set()

    for name in recognizers:
        rec_dir = root / name
        files = sorted(rec_dir.glob("*.json"))
        totals.append(len(files))
        col_counter: Counter[str] = Counter()
        for file_path in files:
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            selected = canonical(data.get("selected_source"))
            if selected:
                col_counter[selected] += 1
                generator_set.add(selected)
            for option in data.get("options") or []:
                src = canonical(option.get("source"))
                if src:
                    generator_set.add(src)
        counters.append(col_counter)
    for name in recognizers:
        generator_set.add(canonical(name))
    generator_set.add("human")
    return counters, np.array(totals, dtype=float), generator_set


def arrange_generators(recognizers: list[str], generators: set[str]) -> list[str]:
    ordered = []
    seen = set()
    for name in recognizers:
        can = canonical(name)
        if can in generators and can not in seen:
            ordered.append(can)
            seen.add(can)
    remaining = sorted(src for src in generators if src not in seen and src.lower() != "human")
    ordered.extend(remaining)
    if "human" in generators:
        ordered.append("human")
    return ordered


def build_ratio_matrix(counters: list[Counter[str]], totals: np.ndarray, generator_order: list[str]):
    matrix = np.zeros((len(generator_order), len(counters)), dtype=float)
    generator_index = {name: idx for idx, name in enumerate(generator_order)}
    for col, counter in enumerate(counters):
        for name, count in counter.items():
            idx = generator_index.get(name)
            if idx is None:
                continue
            matrix[idx, col] = count
    totals_row = totals[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.divide(matrix, totals_row, out=np.zeros_like(matrix), where=totals_row > 0)
    return ratios


def plot_heatmap(ratios: np.ndarray, recognizers: list[str], generators: list[str], label: str, output: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    row_labels = [format_model_label(name) for name in generators]
    col_labels = [format_model_label(name) for name in recognizers]

    sns.heatmap(
        ratios,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=col_labels,
        yticklabels=row_labels,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Selection Ratio"},
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        square=False,
    )

    ax.set_xlabel("Recognizer")
    ax.set_ylabel("Generator")
    ax.set_title(f"Recognition Matrix ({label})")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot recognition_multi selection ratios")
    parser.add_argument("dataset", help="Dataset folder name under the data directory (e.g., paper)")
    parser.add_argument(
        "--data-root",
        default="data",
        help="Base directory that contains dataset folders (default: data)",
    )
    parser.add_argument(
        "--subdir",
        default="recognition_multi/recognition_9",
        help="Subdirectory inside the dataset folder (default: recognition_multi/recognition_9)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output image path; defaults to plots/recognition_multi_<dataset>.png",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.data_root) / args.dataset / args.subdir
    if not root.exists():
        raise SystemExit(f"Input directory {root} does not exist")

    output = Path(args.output) if args.output else Path("plots") / f"recognition_multi_{args.dataset}.png"
    label = dataset_label(args.dataset)

    recognizers = list_recognizers(root)
    counters, totals, generators = gather_counts(root, recognizers)
    generator_order = arrange_generators(recognizers, generators)
    ratios = build_ratio_matrix(counters, totals, generator_order)
    plot_heatmap(ratios, recognizers, generator_order, label, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
