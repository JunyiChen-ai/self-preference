#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PREFIX="/mnt/blob_output/v-junyichen"
DATASETS=(paper trans_seg)
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
GPT_RECOGNIZERS=(
  "gpt-4.1-nano_2025-04-14"
  "gpt-5-chat_2025-08-07"
)
MAX_TOKENS=32
TEMP=0

for dataset in "${DATASETS[@]}"; do
  if [[ "$dataset" == "paper" ]]; then
    DATASET_SUBDIR="paper"
  elif [[ "$dataset" == "trans_seg" ]]; then
    DATASET_SUBDIR="news_segment"
  else
    DATASET_SUBDIR="$dataset"
  fi

  if [[ -z "$DATASET_SUBDIR" ]]; then
    GEN_ROOT="$SCRIPT_DIR/data/llm"
    OUT_ROOT="$SCRIPT_DIR/data/recognition_individual"
    HUMAN_DIR="$SCRIPT_DIR/data/human"
  else
    GEN_ROOT="$SCRIPT_DIR/data/$DATASET_SUBDIR/llm"
    OUT_ROOT="$SCRIPT_DIR/data/$DATASET_SUBDIR/recognition_individual"
    HUMAN_DIR="$SCRIPT_DIR/data/$DATASET_SUBDIR/human"
  fi

  for recognizer in "${GPT_RECOGNIZERS[@]}"; do
    echo "[run] dataset=$dataset recognizer=$recognizer"
    python "$SCRIPT_DIR/recognition_individual.py" \
      --dataset "$dataset" \
      --recognizer-model "$recognizer" \
      --generator-models "${GENERATOR_MODELS[@]}" \
      --human-dir "$HUMAN_DIR" \
      --generator-root "$GEN_ROOT" \
      --output-root "$OUT_ROOT" \
      --prefix "$PREFIX" \
      --temperature "$TEMP" \
      --max-tokens "$MAX_TOKENS"
  done
done
