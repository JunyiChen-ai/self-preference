#!/usr/bin/env python3
"""Triple (human + 3 LLM) comparison using a single evaluator LLM."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are an expert peer reviewer evaluating how well abstracts summarize a paper. "
    "Given the paper content and multiple candidate abstracts, choose the single best abstract."
)
USER_TEMPLATE = (
    "<ARTICLE>\n{article}\n</ARTICLE>\n\n"
    "Candidate abstracts:\n\n"
    "{options}\n"
    "Respond with only the number (1-{count}) of the best abstract."
)
FALLBACK_SUFFIX = (
    " Don't worry about perfection—trust your judgment and reply with only the number of the best abstract."
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
    return None


def build_prompt(article: str, options: List[Dict], reinforce: bool = False) -> str:
    option_lines = []
    for idx, opt in enumerate(options, start=1):
        option_lines.append(f"Option {idx}:\n{opt['description'].strip()}\n")
    user_content = USER_TEMPLATE.format(article=article.strip(), options="\n".join(option_lines), count=len(options))
    if reinforce:
        user_content = f"{user_content}{FALLBACK_SUFFIX}"
    return user_content


def request_choice(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    parser.add_argument("--comparison-model", required=True, help="Model name acting as the comparison evaluator.")
    parser.add_argument(
        "--generator-models",
        nargs="+",
        required=True,
        help="Generator model names whose outputs will be compared (human is implicit).",
    )
    parser.add_argument("--human-dir", default=repo_root / "data" / "human", type=Path)
    parser.add_argument("--generator-root", default=repo_root / "data" / "llm", type=Path)
    parser.add_argument("--output-root", default=repo_root / "data" / "comparison_triple", type=Path)
    parser.add_argument("--prefix", default=Path("/mnt/blob_output/v-junyichen"), type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_files = list(list_json_files(args.human_dir))
    if not human_files:
        raise SystemExit(f"No JSON files under {args.human_dir}")

    rng = random.Random(args.shuffle_seed) if args.shuffle_seed is not None else random.Random()

    client: Optional[OpenAI] = None
    if not args.dry_run:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    max_tokens = args.max_tokens if args.max_tokens > 0 else None

    comparison_folder = apply_prefix(
        args.prefix, args.output_root / sanitize_name(args.comparison_model)
    )
    ensure_dir(comparison_folder)

    generator_paths = {
        model: apply_prefix(args.prefix, args.generator_root / sanitize_name(model))
        for model in args.generator_models
    }

    for human_path in human_files:
        human_json = load_json(human_path)
        article = human_json.get("article")
        human_desc = extract_description(human_json)
        if not article or not human_desc:
            print(f"[skip-empty] {human_path} missing article/abstract")
            continue

        entries = [
            {
                "source": "human",
                "description": human_desc,
            }
        ]

        missing = False
        for model, folder in generator_paths.items():
            file_path = folder / human_path.name
            if not file_path.exists():
                print(f"[skip-missing] {file_path} not found for {model}")
                missing = True
                break
            payload = load_json(file_path)
            desc = extract_description(payload)
            if not desc:
                print(f"[skip-empty] {file_path} has no description")
                missing = True
                break
            entries.append({"source": model, "description": desc})
        if missing:
            continue

        shuffled = entries[:]
        rng.shuffle(shuffled)

        output_path = comparison_folder / human_path.name
        if output_path.exists():
            print(f"[skip-existing] {output_path}")
            continue

        if args.dry_run:
            print(f"[dry-run] Would compare {human_path.name} -> {output_path}")
            continue

        prompt = build_prompt(article, shuffled)
        prompt_variant = "base"

        assert client is not None
        response = request_choice(client, args.comparison_model, prompt, args.temperature, max_tokens)
        choice = normalize_choice(response, len(shuffled))
        if choice is None:
            prompt_variant = "fallback"
            fallback_prompt = build_prompt(article, shuffled, reinforce=True)
            response = request_choice(client, args.comparison_model, fallback_prompt, args.temperature, max_tokens)
            choice = normalize_choice(response, len(shuffled))

        options_meta = []
        for idx, opt in enumerate(shuffled, start=1):
            options_meta.append({"index": idx, "source": opt["source"]})

        selected_source = None
        if choice is not None and 1 <= choice <= len(shuffled):
            selected_source = shuffled[choice - 1]["source"]

        result = {
            "file": human_path.name,
            "comparison_model": args.comparison_model,
            "options": options_meta,
            "selected_index": choice,
            "selected_source": selected_source,
            "response": response,
            "prompt_variant": prompt_variant,
        }

        ensure_dir(output_path.parent)
        with output_path.open("w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[ok] {output_path}")


if __name__ == "__main__":
    main()
