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
  OUT_ROOT="$SCRIPT_DIR/data/comparison"
  HUMAN_DIR="$SCRIPT_DIR/data/human"
else
  GEN_ROOT="$SCRIPT_DIR/data/$DATASET_SUBDIR/llm"
  OUT_ROOT="$SCRIPT_DIR/data/$DATASET_SUBDIR/comparison"
  HUMAN_DIR="$SCRIPT_DIR/data/$DATASET_SUBDIR/human"
fi
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_TOKENS=256
TEMP=0

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

MODELS=(
  # "Qwen/Qwen3-30B-A3B-Instruct-2507"
  # "Qwen/Qwen3-4B-Instruct-2507"
  # "Qwen/Qwen3-Next-80B-A3B-Instruct"
  "google/gemma-3-4b-it"
  "google/gemma-3-12b-it"
  "google/gemma-3-27b-it"
)
# LAST_EVAL="Qwen/Qwen3-Next-80B-A3B-Instruct"
LAST_EVAL="google/gemma-3-27b-it"

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

have_all_comparisons() {
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

priority_pairs=()
late_pairs=()
for gen in "${MODELS[@]}"; do
  for eval in "${MODELS[@]}"; do
    if [[ "$eval" == "$LAST_EVAL" ]]; then
      late_pairs+=("$gen::$eval")
    else
      priority_pairs+=("$gen::$eval")
    fi
  done
done
if [[ ${#priority_pairs[@]} -gt 0 && ${#late_pairs[@]} -gt 0 ]]; then
  pairs=("${priority_pairs[@]}" "${late_pairs[@]}")
elif [[ ${#priority_pairs[@]} -gt 0 ]]; then
  pairs=("${priority_pairs[@]}")
else
  pairs=("${late_pairs[@]}")
fi

for pair in "${pairs[@]}"; do
  GEN_MODEL=${pair%%::*}
  EVAL_MODEL=${pair##*::}
  GEN_FOLDER=$(sanitize "$GEN_MODEL")
  GEN_LOCAL="$GEN_ROOT/$GEN_FOLDER"
  GEN_STORED=$(real_prefixed_path "$GEN_LOCAL")

  if [[ ! -d "$GEN_STORED" ]]; then
    echo "[skip] missing generator outputs for $GEN_MODEL ($GEN_STORED)"
    continue
  fi

  EVAL_FOLDER=$(sanitize "$EVAL_MODEL")
  OUT_LOCAL="$OUT_ROOT/$EVAL_FOLDER/$GEN_FOLDER"
  OUT_STORED=$(real_prefixed_path "$OUT_LOCAL")
  if have_all_comparisons "$OUT_STORED"; then
    echo "[skip] comparison already exists for $GEN_MODEL vs $EVAL_MODEL"
    continue
  fi

  echo "[run] generator=$GEN_MODEL evaluator=$EVAL_MODEL"
  start_vllm "$EVAL_MODEL" 4
  if ! check_ready; then
    echo "[error] evaluator $EVAL_MODEL failed to start"
    stop_vllm
    continue
  fi
  python "$SCRIPT_DIR/comparison.py" \
    --human-dir "$HUMAN_DIR" \
    --generator-root "$GEN_ROOT" \
    --generator-model "$GEN_FOLDER" \
    --prefix "$PREFIX" \
    --output-root "$OUT_ROOT" \
    --evaluator-model "$EVAL_MODEL" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --temperature "$TEMP" \
    --max-tokens "$MAX_TOKENS" \
    --dataset "$DATASET"
  stop_vllm
done
