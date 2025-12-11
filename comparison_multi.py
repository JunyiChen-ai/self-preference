#!/usr/bin/env python3
"""Generalized comparison (human + N LLM) using multiple evaluator LLMs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

PAPER_SYSTEM_PROMPT = (
    "You are an expert peer reviewer evaluating how well abstracts summarize a paper. "
    "Given the paper content and multiple candidate abstracts, choose the single best abstract."
)
PAPER_USER_TEMPLATE = (
    "<ARTICLE>\n{article}\n</ARTICLE>\n\n"
    "Candidate abstracts:\n\n"
    "{options}\n"
    "Respond with only the number (1-{count}) of the best abstract."
)
TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert translator evaluating Chinese translations of an English passage. "
    "Given the English source and multiple candidate translations, choose the best translation."
)
TRANSLATION_USER_TEMPLATE = (
    "<SOURCE>\n{article}\n</SOURCE>\n\n"
    "Candidate translations:\n\n"
    "{options}\n"
    "Respond with only the number (1-{count}) of the best translation."
)
FALLBACK_SUFFIX = (
    " Don't worry about perfection—trust your judgment and reply with only the number of the best option."
)


def list_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.json")):
        if path.is_file():
            yield path


def sanitize_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def apply_prefix(prefix: Path, path: Path) -> Path:
    rel = Path(str(path).lstrip("/"))
    return prefix / rel


def load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def extract_description(payload: Dict) -> Optional[str]:
    descriptions = payload.get("descriptions")
    if isinstance(descriptions, list) and descriptions:
        return descriptions[0]
    text = payload.get("abstract")
    if isinstance(text, str) and text.strip():
        return text
    target = payload.get("target")
    if isinstance(target, str) and target.strip():
        return target
    return None


def extract_article(payload: Dict, dataset: str) -> Optional[str]:
    if dataset == "paper":
        return payload.get("article")
    return payload.get("article") or payload.get("source")


def build_prompt(dataset: str, article: str, options: List[Dict], reinforce: bool = False) -> tuple[str, str]:
    if dataset == "paper":
        system_prompt = PAPER_SYSTEM_PROMPT
        user_template = PAPER_USER_TEMPLATE
    else:
        system_prompt = TRANSLATION_SYSTEM_PROMPT
        user_template = TRANSLATION_USER_TEMPLATE

    option_lines = []
    for idx, opt in enumerate(options, start=1):
        option_lines.append(f"Option {idx}:\n{opt['description'].strip()}\n")
    user_content = user_template.format(article=article.strip(), options="\n".join(option_lines), count=len(options))
    if reinforce:
        user_content = f"{user_content}{FALLBACK_SUFFIX}"
    return system_prompt, user_content


def request_choice(
    client: OpenAI,
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def normalize_choice(text: str, option_count: int) -> Optional[int]:
    for match in re.finditer(r"(\d+)", text):
        idx = int(match.group(1))
        if 1 <= idx <= option_count:
            return idx
    return None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", default="paper", help="Dataset key (e.g., paper, trans_seg).")
    parser.add_argument(
        "--comparison-models",
        nargs="+",
        required=True,
        help="Model names acting as evaluators.",
    )
    parser.add_argument(
        "--generator-models",
        nargs="+",
        required=True,
        help="Generator model names whose outputs will be compared (human is implicit).",
    )
    parser.add_argument("--human-dir", type=Path)
    parser.add_argument("--generator-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prefix", default=Path("/mnt/blob_output/v-junyichen"), type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--overwrite-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--variant-suffix",
        help="Optional name for a subdirectory that segregates outputs (default uses generator count).",
    )
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
    args.output_root = resolve_path("comparison_multi", args.output_root)
    default_suffix = f"comparison_{len(args.generator_models)}"
    args.variant_suffix = args.variant_suffix or default_suffix
    args.output_root = args.output_root / args.variant_suffix

    return args


def load_human_files(args: argparse.Namespace) -> List[Path]:
    human_files = list(list_json_files(args.human_dir))
    if not human_files:
        raise SystemExit(f"No JSON files under {args.human_dir}")
    return human_files


def gather_generators(args: argparse.Namespace) -> Dict[str, Path]:
    return {
        model: apply_prefix(args.prefix, args.generator_root / sanitize_name(model))
        for model in args.generator_models
    }


def process_comparison(
    comparison_model: str,
    human_files: List[Path],
    generator_paths: Dict[str, Path],
    args: argparse.Namespace,
    client: Optional[OpenAI],
    rng: random.Random,
    max_tokens: Optional[int],
) -> None:
    output_dir = apply_prefix(args.prefix, args.output_root / sanitize_name(comparison_model))
    ensure_dir(output_dir)

    for human_path in human_files:
        output_path = output_dir / human_path.name
        if args.skip_existing and output_path.exists():
            print(f"[skip-existing] {comparison_model} {output_path}")
            continue

        human_json = load_json(human_path)
        article = extract_article(human_json, args.dataset)
        human_desc = extract_description(human_json)
        if not article or not human_desc:
            print(f"[skip-empty] {human_path} missing context")
            continue

        entries = [{"source": "human", "description": human_desc}]

        missing = False
        for model, folder in generator_paths.items():
            gen_path = folder / human_path.name
            if not gen_path.exists():
                print(f"[skip-missing] {comparison_model} missing {gen_path}")
                missing = True
                break
            payload = load_json(gen_path)
            desc = extract_description(payload)
            if not desc:
                print(f"[skip-empty] {gen_path} has no description")
                missing = True
                break
            entries.append({"source": model, "description": desc})
        if missing:
            continue

        shuffled = entries[:]
        rng.shuffle(shuffled)

        if args.dry_run:
            print(f"[dry-run] Would compare {human_path.name} using {comparison_model}")
            continue

        system_prompt, prompt = build_prompt(args.dataset, article, shuffled)
        prompt_variant = "base"

        assert client is not None
        response = request_choice(
            client,
            comparison_model,
            system_prompt,
            prompt,
            args.temperature,
            max_tokens,
        )
        choice = normalize_choice(response, len(shuffled))
        if choice is None:
            prompt_variant = "fallback"
            fallback_system_prompt, fallback_prompt = build_prompt(
                args.dataset, article, shuffled, reinforce=True
            )
            response = request_choice(
                client,
                comparison_model,
                fallback_system_prompt,
                fallback_prompt,
                args.temperature,
                max_tokens,
            )
            choice = normalize_choice(response, len(shuffled))

        options_meta = [{"index": idx, "source": opt["source"]} for idx, opt in enumerate(shuffled, start=1)]
        selected_source = None
        if choice is not None and 1 <= choice <= len(shuffled):
            selected_source = options_meta[choice - 1]["source"]

        result = {
            "file": human_path.name,
            "comparison_model": comparison_model,
            "options": options_meta,
            "selected_index": choice,
            "selected_source": selected_source,
            "response": response,
            "prompt_variant": prompt_variant,
        }

        ensure_dir(output_path.parent)
        with output_path.open("w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[ok] {comparison_model} {output_path}")


def main() -> None:
    args = parse_args()
    human_files = load_human_files(args)
    generator_paths = gather_generators(args)

    rng = random.Random(args.shuffle_seed) if args.shuffle_seed is not None else random.Random()

    client: Optional[OpenAI] = None
    if not args.dry_run:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    max_tokens = args.max_tokens if args.max_tokens > 0 else None

    for comparison_model in args.comparison_models:
        process_comparison(
            comparison_model,
            human_files,
            generator_paths,
            args,
            client,
            rng,
            max_tokens,
        )


if __name__ == "__main__":
    main()
