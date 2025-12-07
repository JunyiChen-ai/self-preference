#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PREFIX="/mnt/blob_output/v-junyichen"
GEN_ROOT="$SCRIPT_DIR/data/llm"
OUT_ROOT="$SCRIPT_DIR/data/recognition_triple"
HUMAN_DIR="$SCRIPT_DIR/data/human"
BASE_URL="http://127.0.0.1:8000/v1"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MAX_TOKENS=64
TEMP=0

MODELS=(
  "Qwen/Qwen3-30B-A3B-Instruct-2507"
  "Qwen/Qwen3-4B-Instruct-2507"
  "Qwen/Qwen3-Next-80B-A3B-Instruct"
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

have_all_recognition_triple() {
  local target_dir="$1"
  [[ -d "$target_dir" ]] || return 1
  while IFS= read -r -d '' sample_file; do
    local base
    base=$(basename "$sample_file")
    if [[ ! -f "$target_dir/$base" ]]; then
      return 1
    fi
  done < <(find "$HUMAN_DIR" -maxdepth 1 -type f -name '*.json' -print0)
  return 0
}

for recognizer in "${MODELS[@]}"; do
  RECOGNIZER_FOLDER=$(sanitize "$recognizer")
  OUT_LOCAL="$OUT_ROOT/$RECOGNIZER_FOLDER"
  OUT_STORED=$(real_prefixed_path "$OUT_LOCAL")

  if have_all_recognition_triple "$OUT_STORED"; then
    echo "[skip] recognition triple already exists for $recognizer"
    continue
  fi

  echo "[run] recognizer=$recognizer"
  start_vllm "$recognizer" 8
  if ! check_ready; then
    echo "[error] recognizer $recognizer failed to start"
    stop_vllm
    continue
  fi

  python "$SCRIPT_DIR/recognition_triple.py" \
    --recognizer-model "$recognizer" \
    --generator-models "${MODELS[@]}" \
    --generator-root "$GEN_ROOT" \
    --output-root "$OUT_ROOT" \
    --prefix "$PREFIX" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --temperature "$TEMP" \
    --max-tokens "$MAX_TOKENS"

  stop_vllm
done
