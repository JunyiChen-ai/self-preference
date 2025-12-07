#!/usr/bin/env python3
"""Recognize whether a given abstract was produced by a specific LLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

PROMPT_BODY = (
    "I have some research paper abstracts written by you and some written by others, "
    "but can't tell now which is which. I need this information to organize my summaries correctly. "
    "Here is one abstract:\n\n{abstract}\n\nDid you write this abstract? Reply with \"Yes\" if you did, or \"No\" if you didn't."
)
FALLBACK_SUFFIX = (
    " Don't worry about being perfect—trust your intuition and just reply with \"Yes\" or \"No\" only."
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


def build_prompt(description: str, reinforce: bool = False) -> str:
    prompt = PROMPT_BODY.format(abstract=description.strip())
    if reinforce:
        prompt = f"{prompt}{FALLBACK_SUFFIX}"
    return prompt


def is_refusal(text: str) -> bool:
    lowered = text.strip().lower()
    if not lowered:
        return True
    refusal_markers = (
        "i'm sorry",
        "i am sorry",
        "i cannot",
        "i can't",
        "cannot assist",
        "unable to",
        "as an ai",
    )
    return any(marker in lowered for marker in refusal_markers)


def normalize_response(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    if lowered.startswith("yes"):
        return "Yes"
    if lowered.startswith("no"):
        return "No"
    return None


def request_label(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent
    parser.add_argument("--recognizer-model", required=True, help="LLM model name used as the recognizer.")
    parser.add_argument(
        "--generator-models",
        nargs="+",
        required=True,
        help="Generator model names whose outputs (plus human) will be evaluated.",
    )
    parser.add_argument("--human-dir", default=repo_root / "data" / "human", type=Path)
    parser.add_argument("--generator-root", default=repo_root / "data" / "llm", type=Path)
    parser.add_argument("--output-root", default=repo_root / "data" / "recognition_individual", type=Path)
    parser.add_argument("--prefix", default=Path("/mnt/blob_output/v-junyichen"), type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_files = list(list_json_files(args.human_dir))
    if not human_files:
        raise SystemExit(f"No JSON files under {args.human_dir}")

    prefixed_output_root = apply_prefix(args.prefix, args.output_root / sanitize_name(args.recognizer_model))
    ensure_dir(prefixed_output_root)

    client: Optional[OpenAI] = None
    if not args.dry_run:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    max_tokens = args.max_tokens if args.max_tokens > 0 else None

    sources: List[Dict] = [
        {
            "name": "human",
            "label": "human",
            "path": args.human_dir,
            "is_human": True,
        }
    ]
    for model in args.generator_models:
        folder = args.generator_root / sanitize_name(model)
        prefixed = apply_prefix(args.prefix, folder)
        sources.append(
            {
                "name": model,
                "label": sanitize_name(model),
                "path": prefixed,
                "is_human": False,
            }
        )

    for source in sources:
        source_path = source["path"]
        if not source["is_human"] and not source_path.exists():
            print(f"[skip-source] missing generator outputs for {source['name']} ({source_path})")
            continue

        target_dir = prefixed_output_root / source["label"]
        ensure_dir(target_dir)
        print(f"[info] recognizer={args.recognizer_model} source={source['name']} -> {target_dir}")

        for human_path in human_files:
            input_path = human_path if source["is_human"] else source_path / human_path.name
            output_path = target_dir / human_path.name

            if output_path.exists():
                print(f"[skip-existing] {output_path}")
                continue

            if not input_path.exists():
                print(f"[skip-missing] {input_path} not found for {source['name']}")
                continue

            payload = load_json(input_path)
            description = extract_description(payload)
            if not description:
                print(f"[skip-empty] {input_path} has no description/abstract")
                continue

            prompt = build_prompt(description)
            prompt_variant = "base"
            response_text = ""

            if args.dry_run:
                print(f"[dry-run] Would query {source['name']}::{human_path.name} -> {output_path}")
                continue

            assert client is not None
            response_text = request_label(client, args.recognizer_model, prompt, args.temperature, max_tokens)
            if is_refusal(response_text):
                prompt_variant = "fallback"
                fallback_prompt = build_prompt(description, reinforce=True)
                response_text = request_label(client, args.recognizer_model, fallback_prompt, args.temperature, max_tokens)

            normalized = normalize_response(response_text)
            result = {
                "file": human_path.name,
                "source": source["name"],
                "recognizer": args.recognizer_model,
                "prompt_variant": prompt_variant,
                "response": response_text,
                "normalized_response": normalized,
                "is_yes": bool(normalized == "Yes") if normalized is not None else None,
            }
            ensure_dir(output_path.parent)
            with output_path.open("w") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[ok] {output_path}")


if __name__ == "__main__":
    main()
