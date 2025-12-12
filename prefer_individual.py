#!/usr/bin/env python3
"""Evaluate individual descriptions/translations for preference."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from openai import OpenAI

try:  # Optional import for retry handling.
    from openai import RateLimitError  # type: ignore
except ImportError:  # pragma: no cover
    RateLimitError = Exception  # type: ignore

PAPER_PROMPT = (
    "You will review a research paper. Read the article content and the candidate abstract, then decide whether you like it.\n"
    "Respond with a single word: 'like' if you like it, or 'dislike' if you do not.\n\n"
    "<ARTICLE>\n{article}\n</ARTICLE>\n\n"
    "<CANDIDATE_ABSTRACT>\n{abstract}\n</CANDIDATE_ABSTRACT>"
)
TRANSLATION_PROMPT = (
    "You will review a Chinese translation of an English article. Read the English source and the candidate translation, then decide whether you like it.\n"
    "Respond with only 'like' or 'dislike'.\n\n"
    "<SOURCE_EN>\n{source}\n</SOURCE_EN>\n\n"
    "<CANDIDATE_ZH>\n{translation}\n</CANDIDATE_ZH>"
)
FALLBACK_SUFFIX = (
    " Please reply with exactly one word: like or dislike."
)


def list_json_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.glob("*.json")):
        if path.is_file():
            yield path


def sanitize(value: str) -> str:
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
    abstract = payload.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return abstract
    target = payload.get("target")
    if isinstance(target, str) and target.strip():
        return target
    return None


def extract_article(payload: Dict) -> Optional[str]:
    return payload.get("article")


def extract_source_text(payload: Dict) -> Optional[str]:
    for key in ("source", "source_text", "source_text_en", "article"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_prompt(dataset: str, article: Optional[str], candidate: str, reinforce: bool = False) -> str:
    if dataset == "paper":
        if not article:
            raise ValueError("Article text required for paper dataset")
        body = PAPER_PROMPT.format(article=article.strip(), abstract=candidate.strip())
    else:
        if not article:
            raise ValueError("Source text required for translation dataset")
        body = TRANSLATION_PROMPT.format(source=article.strip(), translation=candidate.strip())
    if reinforce:
        body = f"{body}{FALLBACK_SUFFIX}"
    return body


def parse_preference(text: str) -> Optional[str]:
    lowered = text.strip().lower()
    if not lowered:
        return None
    if re.search(r"\bdislike\b", lowered):
        return "dislike"
    if re.search(r"\blike\b", lowered):
        return "like"
    return None


def load_gpt_client(module_path: Path):
    spec = importlib.util.spec_from_file_location("prefer_gpt", module_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import GPT client from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "GPT4oClient"):
        raise SystemExit("GPT client module missing GPT4oClient class")
    return module.GPT4oClient()


def request_vllm(
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


def request_gpt(
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
    parser.add_argument("--dataset", default="paper")
    parser.add_argument("--evaluator-models", nargs="+", required=True)
    parser.add_argument("--generator-models", nargs="+", required=True)
    parser.add_argument("--human-dir", type=Path)
    parser.add_argument("--generator-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--prefix", type=Path, default=Path("/mnt/blob_output/v-junyichen"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--overwrite-existing", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpt-client-path", type=Path, default=Path("/home/v-junyichen/LLaMA-Factory/fairness/GPT.py"))
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
    args.generator_root = resolve_path("llm", args.generator_root)
    args.output_root = resolve_path("prefer_individual", args.output_root)
    return args


def gather_human_files(human_dir: Path) -> List[Path]:
    files = list(list_json_files(human_dir))
    if not files:
        raise SystemExit(f"No human files found under {human_dir}")
    return files


def process_sample(
    evaluator: str,
    evaluator_dir: Path,
    candidate_source: str,
    candidate_path: Path,
    args: argparse.Namespace,
    client: Optional[OpenAI],
    gpt_client,
    max_tokens: Optional[int],
) -> None:
    output_dir = evaluator_dir / sanitize(candidate_source)
    ensure_dir(output_dir)
    output_path = output_dir / candidate_path.name
    if args.skip_existing and output_path.exists():
        print(f"[skip-existing] {evaluator} {candidate_source} {output_path}")
        return

    payload = load_json(candidate_path)
    candidate_text = extract_description(payload)
    if not candidate_text:
        print(f"[skip-empty] {candidate_path} has no description")
        return
    if args.dataset == "paper":
        article = extract_article(payload)
    else:
        article = extract_source_text(payload)
    if not article:
        print(f"[skip-context] {candidate_path} missing context")
        return

    prompt = build_prompt(args.dataset, article, candidate_text)
    use_gpt = is_gpt_model(evaluator)
    responder = request_gpt if use_gpt else request_vllm
    selected_client = gpt_client if use_gpt else client
    if selected_client is None and not args.dry_run:
        raise SystemExit("Client not initialized")

    if args.dry_run:
        print(f"[dry-run] {evaluator} would score {candidate_path}")
        return

    response = responder(selected_client, evaluator, prompt, args.temperature, max_tokens)
    preference = parse_preference(response)
    if preference is None:
        fallback_prompt = build_prompt(args.dataset, article, candidate_text, reinforce=True)
        response = responder(selected_client, evaluator, fallback_prompt, args.temperature, max_tokens)
        preference = parse_preference(response)

    result = {
        "file": candidate_path.name,
        "evaluator": evaluator,
        "source": candidate_source,
        "preference": preference,
        "response": response,
    }
    with output_path.open("w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] {evaluator} {candidate_source} {output_path}")


def process_evaluator(
    evaluator: str,
    human_files: List[Path],
    args: argparse.Namespace,
    client: Optional[OpenAI],
    gpt_client,
    max_tokens: Optional[int],
) -> None:
    base_output = args.output_root / sanitize(evaluator)
    evaluator_dir = base_output if is_gpt_model(evaluator) else apply_prefix(args.prefix, base_output)
    ensure_dir(evaluator_dir)

    # Determine generator directories for this evaluator
    generator_dirs: Dict[str, Path] = {}
    use_local_generators = is_gpt_model(evaluator)
    for model in args.generator_models:
        base_path = args.generator_root / sanitize(model)
        if use_local_generators or is_gpt_model(model):
            generator_dirs[model] = base_path
        else:
            generator_dirs[model] = apply_prefix(args.prefix, base_path)

    for human_path in human_files:
        process_sample(
            evaluator,
            evaluator_dir,
            "human",
            human_path,
            args,
            client,
            gpt_client,
            max_tokens,
        )
        for model, folder in generator_dirs.items():
            candidate_path = folder / human_path.name
            if not candidate_path.exists():
                print(f"[skip-missing] {evaluator} missing {candidate_path}")
                continue
            process_sample(
                evaluator,
                evaluator_dir,
                model,
                candidate_path,
                args,
                client,
                gpt_client,
                max_tokens,
            )


def main() -> None:
    args = parse_args()
    human_files = gather_human_files(args.human_dir)

    evaluator_has_gpt = any(is_gpt_model(m) for m in args.evaluator_models)
    evaluator_has_vllm = any(not is_gpt_model(m) for m in args.evaluator_models)

    client: Optional[OpenAI] = None
    gpt_client = None
    if not args.dry_run and evaluator_has_vllm:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    if not args.dry_run and evaluator_has_gpt:
        gpt_client = load_gpt_client(args.gpt_client_path)
    max_tokens = args.max_tokens if args.max_tokens > 0 else None

    for evaluator in args.evaluator_models:
        process_evaluator(evaluator, human_files, args, client, gpt_client, max_tokens)


if __name__ == "__main__":
    main()
