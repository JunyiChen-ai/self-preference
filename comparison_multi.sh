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
  OUT_ROOT_BASE="$SCRIPT_DIR/data/comparison_multi"
  HUMAN_DIR="$SCRIPT_DIR/data/human"
else
  GEN_ROOT="$SCRIPT_DIR/data/$DATASET_SUBDIR/llm"
  OUT_ROOT_BASE="$SCRIPT_DIR/data/$DATASET_SUBDIR/comparison_multi"
  HUMAN_DIR="$SCRIPT_DIR/data/$DATASET_SUBDIR/human"
fi
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_TOKENS=256
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
COMPARISON_MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
)
LAST_COMPARISON="google/gemma-3-27b-it"
GENERATOR_COUNT=${#GENERATOR_MODELS[@]}
VARIANT_SUFFIX="comparison_${GENERATOR_COUNT}"
OUT_ROOT="$OUT_ROOT_BASE/$VARIANT_SUFFIX"

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

real_prefixed_path() {
  local rel="$1"
  rel=${rel#/}
  printf '%s/%s' "$PREFIX" "$rel"
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

have_all_comparisons_multi() {
  local target_dir="$1"
  [[ -d "$target_dir" ]] || return 1
  while IFS= read -r -d '' human_file; do
    local base
    base=$(basename "$human_file")
    if [[ ! -f "$target_dir/$base" ]]; then
      return 1
    fi
  done < <(find "$HUMAN_DIR" -maxdepth 1 -type f -name '*.json' -print0)
  return 0
}

priority_models=()
late_models=()
for eval in "${COMPARISON_MODELS[@]}"; do
  if [[ "$eval" == "$LAST_COMPARISON" ]]; then
    late_models+=("$eval")
  else
    priority_models+=("$eval")
  fi
done
if [[ ${#priority_models[@]} -gt 0 && ${#late_models[@]} -gt 0 ]]; then
  ordered_models=("${priority_models[@]}" "${late_models[@]}")
elif [[ ${#priority_models[@]} -gt 0 ]]; then
  ordered_models=("${priority_models[@]}")
else
  ordered_models=("${late_models[@]}")
fi

for comparison_model in "${ordered_models[@]}"; do
  folder=$(sanitize "$comparison_model")
  OUT_LOCAL="$OUT_ROOT/$folder"
  OUT_STORED=$(real_prefixed_path "$OUT_LOCAL")

  if have_all_comparisons_multi "$OUT_STORED"; then
    echo "[skip] comparison already exists for $comparison_model"
    continue
  fi

  echo "[run] comparison_model=$comparison_model"
  start_vllm "$comparison_model" 4
  if ! check_ready; then
    echo "[error] comparison_model $comparison_model failed to start"
    stop_vllm
    continue
  fi

  python "$SCRIPT_DIR/comparison_multi.py" \
    --comparison-models "$comparison_model" \
    --generator-models "${GENERATOR_MODELS[@]}" \
    --human-dir "$HUMAN_DIR" \
    --generator-root "$GEN_ROOT" \
    --output-root "$OUT_ROOT_BASE" \
    --variant-suffix "$VARIANT_SUFFIX" \
    --prefix "$PREFIX" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --temperature "$TEMP" \
    --max-tokens "$MAX_TOKENS" \
    --dataset "$DATASET"

  stop_vllm
done
