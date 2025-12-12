#!/usr/bin/env python3
"""Recognize whether a given abstract was produced by a specific LLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import importlib.util
import time

from openai import OpenAI

try:
    from openai import RateLimitError  # type: ignore
except ImportError:  # pragma: no cover
    RateLimitError = Exception  # type: ignore

PAPER_PROMPT_BODY = (
    "I have some research paper abstracts written by you and some written by others, "
    "but can't tell now which is which. I need this information to organize my summaries correctly. "
    "Here is one abstract:\n\n{abstract}\n\nDid you write this abstract? Reply with \"Yes\" if you did, or \"No\" if you didn't."
)
TRANSLATION_PROMPT_BODY = (
    "I have some Chinese translations of an English passage written by you and some written by others. "
    "I can't tell which translations are yours. Here is one translation:\n\n{abstract}\n\n"
    "Did you produce this translation? Reply with \"Yes\" if you did, or \"No\" if you didn't."
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


def is_gpt_model(name: str) -> bool:
    return "gpt" in name.lower()


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


def build_prompt(description: str, dataset: str, reinforce: bool = False) -> str:
    base = PAPER_PROMPT_BODY if dataset == "paper" else TRANSLATION_PROMPT_BODY
    prompt = base.format(abstract=description.strip())
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


def load_gpt_client(module_path: Path):
    spec = importlib.util.spec_from_file_location("recognition_individual_gpt", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import GPT client from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GPT4oClient"):
        raise SystemExit("GPT client module missing GPT4oClient class")
    return module.GPT4oClient()


def request_gpt_label(
    client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    retries: int = 3,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {"model": model, "messages": messages}
    if "gpt-5" not in model.lower():
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    attempt = 0
    while True:
        try:
            response = client.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content.strip()
        except RateLimitError as exc:  # type: ignore
            attempt += 1
            if attempt <= retries:
                time.sleep(30)
                continue
            raise
        except Exception:
            raise


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", default="paper", help="Dataset key (e.g., paper, trans_seg).")
    parser.add_argument("--recognizer-model", required=True, help="LLM model name used as the recognizer.")
    parser.add_argument(
        "--generator-models",
        nargs="+",
        required=True,
        help="Generator model names whose outputs (plus human) will be evaluated.",
    )
    parser.add_argument("--human-dir", type=Path)
    parser.add_argument("--generator-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prefix", default=Path("/mnt/blob_output/v-junyichen"), type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--gpt-client-path",
        type=Path,
        default=Path("/home/v-junyichen/LLaMA-Factory/fairness/GPT.py"),
    )
    args = parser.parse_args()

    data_root = repo_root / "data"

    def dataset_subdir() -> Optional[str]:
        if args.dataset == "paper":
            return "paper"
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
    args.output_root = resolve_path("recognition_individual", args.output_root)

    return args


def main() -> None:
    args = parse_args()
    human_files = list(list_json_files(args.human_dir))
    if not human_files:
        raise SystemExit(f"No JSON files under {args.human_dir}")

    recognizer_is_gpt = is_gpt_model(args.recognizer_model)
    base_output = args.output_root / sanitize_name(args.recognizer_model)
    prefixed_output_root = base_output if recognizer_is_gpt else apply_prefix(args.prefix, base_output)
    ensure_dir(prefixed_output_root)

    client: Optional[OpenAI] = None
    gpt_client = None
    if not args.dry_run and not recognizer_is_gpt:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if not args.dry_run and recognizer_is_gpt:
        gpt_client = load_gpt_client(args.gpt_client_path)
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
        if recognizer_is_gpt or is_gpt_model(model):
            prefixed = folder
        else:
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

            prompt = build_prompt(description, args.dataset)
            prompt_variant = "base"
            response_text = ""

            if args.dry_run:
                print(f"[dry-run] Would query {source['name']}::{human_path.name} -> {output_path}")
                continue

            if recognizer_is_gpt:
                assert gpt_client is not None
                response_text = request_gpt_label(
                    gpt_client,
                    args.recognizer_model,
                    prompt,
                    args.temperature,
                    max_tokens,
                )
            else:
                assert client is not None
                response_text = request_label(client, args.recognizer_model, prompt, args.temperature, max_tokens)
            if is_refusal(response_text):
                prompt_variant = "fallback"
                fallback_prompt = build_prompt(description, args.dataset, reinforce=True)
                if recognizer_is_gpt:
                    response_text = request_gpt_label(
                        gpt_client,
                        args.recognizer_model,
                        fallback_prompt,
                        args.temperature,
                        max_tokens,
                    )
                else:
                    response_text = request_label(
                        client,
                        args.recognizer_model,
                        fallback_prompt,
                        args.temperature,
                        max_tokens,
                    )

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
