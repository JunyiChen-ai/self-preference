#!/usr/bin/env python3
"""Batch description generation via local vLLM or remote GPT endpoints."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

try:  # Prefer modern OpenAI SDK style
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - legacy SDK fallback
    OpenAI = None  # type: ignore
    import openai as legacy_openai  # type: ignore
else:
    legacy_openai = None  # type: ignore

try:  # Backwards-compatible BadRequestError import for older openai packages.
    from openai import BadRequestError  # type: ignore
except ImportError:  # pragma: no cover - older SDK fallback
    try:
        from openai.error import InvalidRequestError as BadRequestError  # type: ignore
    except ImportError:  # pragma: no cover
        class BadRequestError(Exception):
            """Fallback BadRequestError if SDK does not expose one."""

            pass

PROMPT_TEMPLATE = (
    "Read the following academic paper provided in XML format and create an abstract for it.\n\n"
    "Please limit the response to {max_words} words or less.\n\n"
    "---\n\n"
    "**Description:**\n\n"
    "{article}\n"
)

TRANSLATION_PROMPT_TEMPLATE = (
    "You are a professional English-to-Chinese translator. Translate the following passage into fluent Simplified Chinese.\n\n"
    "Source passage:\n{source}\n\n"
    "Return only the Chinese translation."
)


def list_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.json")):
        if path.is_file():
            yield path


def sanitize_folder_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def is_gpt_model(model_name: str) -> bool:
    return "gpt" in model_name.lower()


def apply_prefix(prefix: Path, path: Path) -> Path:
    rel = Path(str(path).lstrip("/"))
    return prefix / rel


def build_prompt(human_payload: dict, dataset: str, max_words: int) -> Optional[str]:
    if dataset == "paper":
        article = human_payload.get("article")
        if not article:
            return None
        return PROMPT_TEMPLATE.format(max_words=max_words, article=article.strip())

    source = human_payload.get("source")
    if not source:
        return None
    return TRANSLATION_PROMPT_TEMPLATE.format(source=source.strip())


def request_vllm_summary(
    client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    # Legacy openai module interface
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    response = client.ChatCompletion.create(**kwargs)
    # Legacy responses are dict-like
    return response["choices"][0]["message"]["content"].strip()


def load_gpt_client(module_path: Path):
    if not module_path.exists():
        raise SystemExit(f"GPT client module not found at {module_path}")
    spec = importlib.util.spec_from_file_location("fairness_gpt_client", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to load GPT client from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GPT4oClient"):
        raise SystemExit("GPT client module missing GPT4oClient class")
    return module.GPT4oClient()


def request_gpt_summary(
    gpt_client,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
) -> str:
    messages = [{"role": "user", "content": prompt}]
    kwargs = {
        "model": model,
        "messages": messages,
    }
    if "gpt-5" not in model.lower():
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    response = gpt_client.client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


def prepare_output_record(
    human_payload: dict,
    prompt: str,
    summary: str,
    model_name: str,
    temperature: float,
    max_words: int,
    dataset: str,
) -> dict:
    article_text = human_payload.get("article") or human_payload.get("source")
    record = {
        "item_type": human_payload.get("item_type", "paper"),
        "title": human_payload.get("title"),
        "origin": "LLM",
        "llm_engine": model_name,
        "temperature": temperature,
        "max_words": max_words,
        "generation_prompt_text": prompt,
        "descriptions": [summary],
        "article": article_text,
        "source_file_name": human_payload.get("title"),
    }
    if dataset != "paper":
        record["source_text"] = human_payload.get("source")
    return record


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent
    parser.add_argument("--dataset", default="paper", help="Dataset key (e.g., paper, trans_seg).")
    parser.add_argument(
        "--human-dir",
        type=Path,
        help="Directory containing human-authored JSON files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Base directory (before prefix) for generated files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="/mnt/blob_output/v-junyichen",
        type=Path,
        help="Prefix to prepend to every output path.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more model names.",
    )
    parser.add_argument(
        "--temperature",
        default=0.2,
        type=float,
        help="Sampling temperature for the chat completion API.",
    )
    parser.add_argument(
        "--max-words",
        default=100,
        type=int,
        help="Word limit inserted into the prompt template.",
    )
    parser.add_argument(
        "--max-tokens",
        default=512,
        type=int,
        help="max_tokens parameter sent to the API (set to 0 to omit).",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="Base URL of the OpenAI-compatible endpoint (e.g., vLLM).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for the endpoint (defaults to OPENAI_API_KEY env).",
    )
    parser.add_argument(
        "--overwrite-existing",
        dest="skip_existing",
        action="store_false",
        help="Regenerate even if the target file already exists (default is to skip).",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without contacting the endpoint.",
    )
    parser.add_argument(
        "--gpt-client-path",
        type=Path,
        default=Path("/home/v-junyichen/LLaMA-Factory/fairness/GPT.py"),
        help="Path to the GPT4oClient implementation.",
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

    def resolve_path(sub: str, override: Optional[Path]) -> Path:
        if override is not None:
            return override
        if subdir is None:
            return data_root / sub
        return data_root / subdir / sub

    args.human_dir = resolve_path("human", args.human_dir)
    args.output_root = resolve_path("llm", args.output_root)

    return args


def main() -> None:
    args = parse_args()
    human_files = list(list_json_files(args.human_dir))
    if not human_files:
        raise SystemExit(f"No JSON files found under {args.human_dir}")

    needs_vllm = any(not is_gpt_model(m) for m in args.models)
    needs_gpt = any(is_gpt_model(m) for m in args.models)

    vllm_client = None
    if not args.dry_run and needs_vllm:
        if OpenAI is not None:
            vllm_client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        else:
            legacy_openai.api_key = args.api_key  # type: ignore
            legacy_openai.api_base = args.base_url  # type: ignore
            vllm_client = legacy_openai  # type: ignore

    gpt_client = None
    if not args.dry_run and needs_gpt:
        gpt_client = load_gpt_client(args.gpt_client_path)

    max_tokens = args.max_tokens if args.max_tokens > 0 else None

    for model in args.models:
        model_folder = sanitize_folder_name(model)
        base_target_dir = args.output_root / model_folder
        if is_gpt_model(model):
            final_target_dir = base_target_dir
        else:
            final_target_dir = apply_prefix(args.output_prefix, base_target_dir)
        ensure_dir(final_target_dir)
        print(f"Processing {len(human_files)} files for model '{model}' -> {final_target_dir}")

        for human_path in human_files:
            rel_name = human_path.name
            output_path = final_target_dir / rel_name
            if args.skip_existing and output_path.exists():
                print(f"[skip] {output_path}")
                continue

            with human_path.open() as f:
                human_payload = json.load(f)

            prompt = build_prompt(human_payload, args.dataset, args.max_words)
            if not prompt:
                print(f"[warn] missing required fields in {human_path}, skipping")
                continue

            if args.dry_run:
                print(f"[dry-run] Would request {human_path.name} -> {output_path}")
                continue

            try:
                if is_gpt_model(model):
                    summary = request_gpt_summary(gpt_client, model, prompt, args.temperature, max_tokens)
                else:
                    summary = request_vllm_summary(vllm_client, model, prompt, args.temperature, max_tokens)
            except BadRequestError as exc:
                message = str(exc)
                if "maximum context length" in message:
                    print(f"[skip-length] {human_path} ({len(prompt.split())} words) -> {message}")
                    continue
                raise
            record = prepare_output_record(
                human_payload,
                prompt,
                summary,
                model,
                args.temperature,
                args.max_words,
                args.dataset,
            )

            ensure_dir(output_path.parent)
            with output_path.open("w") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            print(f"[ok] {output_path}")


if __name__ == "__main__":
    main()
