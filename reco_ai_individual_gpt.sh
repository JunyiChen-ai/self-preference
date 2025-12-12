#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
DATASETS=(paper trans_seg)
GPT_EVALUATORS=(
  "gpt-4.1-nano_2025-04-14"
  "gpt-4o_2024-08-06"
  "gpt-5-chat_2025-08-07"
)
GENERATOR_MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
  "gpt-4.1-nano_2025-04-14"
  "gpt-4o_2024-08-06"
  "gpt-5-chat_2025-08-07"
)

for dataset in "${DATASETS[@]}"; do
  for evaluator in "${GPT_EVALUATORS[@]}"; do
    echo "[reco-ai] dataset=$dataset evaluator=$evaluator"
    python "$SCRIPT_DIR/reco_ai_individual.py" \
      --dataset "$dataset" \
      --evaluator-models "$evaluator" \
      --generator-models "${GENERATOR_MODELS[@]}"
  done
done
