#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PREFIX="/mnt/blob_output/v-junyichen"
DATASET=${DATASET:-paper}
if [[ "$DATASET" == "paper" ]]; then
  DATASET_SUBDIR="paper"
elif [[ "$DATASET" == "trans_seg" ]]; then
  DATASET_SUBDIR="news_segment"
else
  DATASET_SUBDIR="$DATASET"
fi

if [[ -z "$DATASET_SUBDIR" ]]; then
  GEN_ROOT="$SCRIPT_DIR/data/llm"
  OUT_ROOT_BASE="$SCRIPT_DIR/data/prefer_individual"
  HUMAN_DIR="$SCRIPT_DIR/data/human"
else
  GEN_ROOT="$SCRIPT_DIR/data/$DATASET_SUBDIR/llm"
  OUT_ROOT_BASE="$SCRIPT_DIR/data/$DATASET_SUBDIR/prefer_individual"
  HUMAN_DIR="$SCRIPT_DIR/data/$DATASET_SUBDIR/human"
fi
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_TOKENS=128
TEMP=0

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
EVALUATOR_MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
)

check_ready() {
  local max_wait=600
  local interval=10
  local elapsed=0
  until curl -sf "$BASE_URL/models" >/dev/null; do
    sleep $interval
    elapsed=$((elapsed + interval))
    if (( elapsed >= max_wait )); then
      echo "vLLM server did not become ready within ${max_wait}s" >&2
      return 1
    fi
  done
}

sanitize() {
  local name="$1"
  name=${name//\//_}
  name=${name// /_}
  printf '%s' "$name"
}

start_vllm() {
  local model="$1"
  local gpus="$2"
  local log_name=$(sanitize "$model")
  local log_file="$PREFIX/vllm_${log_name}.log"
  mkdir -p "$(dirname "$log_file")"
  echo "[info] launching vLLM for $model, logging to $log_file"
  vllm serve "$model" \
    --trust-remote-code \
    --tensor-parallel-size "$gpus" \
    --host 0.0.0.0 \
    --port 8000 >"$log_file" 2>&1 &
  VLLM_PID=$!
}

stop_vllm() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    unset VLLM_PID
  fi
}

for evaluator in "${EVALUATOR_MODELS[@]}"; do
  echo "[run] evaluator=$evaluator"
  start_vllm "$evaluator" 4
  if ! check_ready; then
    echo "[error] evaluator $evaluator failed to start"
    stop_vllm
    continue
  fi

  python "$SCRIPT_DIR/prefer_individual.py" \
    --evaluator-models "$evaluator" \
    --generator-models "${GENERATOR_MODELS[@]}" \
    --human-dir "$HUMAN_DIR" \
    --generator-root "$GEN_ROOT" \
    --output-root "$OUT_ROOT_BASE" \
    --prefix "$PREFIX" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --temperature "$TEMP" \
    --max-tokens "$MAX_TOKENS" \
    --dataset "$DATASET"

  stop_vllm
done
