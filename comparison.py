#!/usr/bin/env python3
"""Pairwise evaluation between human and generated abstracts using an evaluator LLM."""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from openai import OpenAI

PAPER_SYSTEM_PROMPT = (
    "You are an expert peer reviewer evaluating how well abstracts summarize a paper.\n"
    "Given the paper content and two candidate abstracts, choose the better one.\n"
    "Return only '1' if the first abstract is better, or '2' if the second is better.\n"
)

PAPER_PAIR_TEMPLATE = (
    "<ARTICLE>\n{article}\n</ARTICLE>\n\n"
    "Abstract 1:\n{abstract1}\n\n"
    "Abstract 2:\n{abstract2}"
)

TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert translator evaluating Chinese translations of an English passage.\n"
    "Given the English source and two candidate translations, choose the better translation.\n"
    "Return only '1' if the first translation is better, or '2' if the second is better.\n"
)

TRANSLATION_PAIR_TEMPLATE = (
    "<SOURCE>\n{article}\n</SOURCE>\n\n"
    "Translation 1:\n{abstract1}\n\n"
    "Translation 2:\n{abstract2}"
)


@dataclass
class Sample:
    name: str
    article: str
    human_abstract: str
    ai_abstract: str


def list_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.json")):
        if path.is_file():
            yield path


def apply_prefix(prefix: Path, path: Path) -> Path:
    rel = Path(str(path).lstrip("/"))
    return prefix / rel


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def extract_human_fields(payload: Dict, dataset: str) -> Tuple[Optional[str], Optional[str]]:
    if dataset == "paper":
        article = payload.get("article")
        human_desc = payload.get("abstract") or payload.get("descriptions", [""])[0]
    else:
        article = payload.get("article") or payload.get("source")
        human_desc = payload.get("target") or payload.get("descriptions", [""])[0]
    return article, human_desc


def build_messages(dataset: str, article: str, abstract1: str, abstract2: str) -> list:
    if dataset == "paper":
        system_prompt = PAPER_SYSTEM_PROMPT
        pair_template = PAPER_PAIR_TEMPLATE
    else:
        system_prompt = TRANSLATION_SYSTEM_PROMPT
        pair_template = TRANSLATION_PAIR_TEMPLATE

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": pair_template.format(article=article.strip(), abstract1=abstract1.strip(), abstract2=abstract2.strip()),
        },
    ]


def evaluate_pair(
    client: OpenAI,
    model: str,
    dataset: str,
    article: str,
    abstract1: str,
    abstract2: str,
    temperature: float,
    max_tokens: Optional[int],
) -> int:
    messages = build_messages(dataset, article, abstract1, abstract2)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content.strip()
    return 1 if text.startswith("1") else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", default="paper", help="Dataset key (e.g., paper, trans_seg).")
    parser.add_argument("--human-dir", type=Path)
    parser.add_argument("--generator-model", required=True, help="Name of the generator model folder under output-root.")
    parser.add_argument("--generator-root", type=Path, help="Relative root for generator outputs.")
    parser.add_argument("--prefix", default="/mnt/blob_output/v-junyichen", type=Path, help="Prefix applied to generator and output writes.")
    parser.add_argument("--evaluator-model", required=True, help="Model name served by evaluator endpoint.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = repo_root / "data"

    def dataset_subdir() -> Optional[str]:
        if args.dataset == "paper":
            return None
        if args.dataset == "trans_seg":
            return "news_segment"
        return args.dataset

    subdir = dataset_subdir()

    def resolve_path(sub: str, value: Optional[Path]) -> Path:
        if value is not None:
            return value
        if subdir is None:
            return data_root / sub
        return data_root / subdir / sub

    args.human_dir = resolve_path("human", args.human_dir)
    args.generator_root = resolve_path("llm", args.generator_root)
    args.output_root = resolve_path("comparison", args.output_root)

    return args


def main() -> None:
    args = parse_args()
    client = None if args.dry_run else OpenAI(base_url=args.base_url, api_key=args.api_key)

    human_files = list(list_json_files(args.human_dir))
    if not human_files:
        raise SystemExit(f"No human files under {args.human_dir}")

    generator_dir = apply_prefix(args.prefix, args.generator_root / args.generator_model)
    evaluator_folder = sanitize_model_name(args.evaluator_model)
    output_dir = apply_prefix(args.prefix, args.output_root / evaluator_folder / args.generator_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = 0
    ai_votes = 0
    human_votes = 0

    for human_path in human_files:
        gen_path = generator_dir / human_path.name
        if not gen_path.exists():
            print(f"[skip-missing] {gen_path} not found")
            continue

        result_path = output_dir / human_path.name
        if result_path.exists():
            print(f"[skip-existing] {human_path.name}")
            continue

        human_json = load_json(human_path)
        generator_json = load_json(gen_path)

        article, human_desc = extract_human_fields(human_json, args.dataset)
        ai_desc = generator_json.get("descriptions", [""])[0]
        if not article or not human_desc or not ai_desc:
            print(f"[skip-empty] {human_path}")
            continue

        samples += 1
        if args.dry_run:
            print(f"[dry-run] Would evaluate {human_path.name}")
            continue

        first = evaluate_pair(
            client,
            args.evaluator_model,
            args.dataset,
            article,
            ai_desc,
            human_desc,
            args.temperature,
            args.max_tokens,
        )
        second = evaluate_pair(
            client,
            args.evaluator_model,
            args.dataset,
            article,
            human_desc,
            ai_desc,
            args.temperature,
            args.max_tokens,
        )

        ai_first_win = first == 1
        ai_second_win = second == 2
        ai_votes += int(ai_first_win) + int(ai_second_win)
        human_votes += 2 - (int(ai_first_win) + int(ai_second_win))

        result = {
            "file": human_path.name,
            "ai_first_choice": first,
            "human_first_choice": second,
            "ai_first_win": ai_first_win,
            "ai_second_win": ai_second_win,
        }
        with result_path.open("w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[ok] {human_path.name} -> {result}")

    if samples and not args.dry_run:
        ratio = ai_votes / (ai_votes + human_votes) if (ai_votes + human_votes) else 0
        print(
            f"AI votes: {ai_votes}, Human votes: {human_votes}, "
            f"AI preference ratio: {ratio:.2%} over {samples} samples"
        )


if __name__ == "__main__":
    main()
