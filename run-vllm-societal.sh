set -x

PREFIX="/mnt/blob_output/v-junyichen"

sanitize() {
  local name="$1"
  name=${name//\//_}
  name=${name// /_}
  printf '%s' "$name"
}

# model=Qwen/Qwen3-Next-80B-A3B-Instruct
# model=meta-llama/Llama-3.3-70B-Instruct
# model=openai/gpt-oss-120b
model=$1
gpu_count=$2
shift 2

if [ -z "$model" ]; then
  echo "Usage: bash run-vllm-societal.sh <model_name> [gpu_count] [generation.py args...]"
  exit 1
fi

if [ "$gpu_count" == "" ]; then
  gpu_count=8
fi

######### start vllm server #########
log_name=$(sanitize "$model")
log_file="$PREFIX/vllm_${log_name}.log"
mkdir -p "$(dirname "$log_file")"
echo "[info] launching vLLM for $model, logging to $log_file"
nohup vllm serve $model \
  --trust-remote-code \
  --tensor-parallel-size $gpu_count \
  --host 0.0.0.0 \
  --port 8000 >"$log_file" 2>&1 &

echo "*********** Waiting for vllm server to start ***********"
max_wait_secs=600
interval=10
elapsed=0
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
  sleep $interval
  elapsed=$((elapsed + interval))
  if [ $elapsed -ge $max_wait_secs ]; then
    echo "vLLM server did not become ready within ${max_wait_secs}s" >&2
    exit 1
  fi
done
echo "*********** Done waiting ***********"

echo "*********** Running generation.py ***********"
python "$(dirname "$0")/generation.py" --models "$model" "$@"
echo "*********** generation.py complete ***********"

# python src/$application/societal.py $model "Gender Identity" &
# sleep 5
# python src/$application/societal.py $model "Gender Identity" &
# sleep 5
# python src/$application/societal.py $model "Sexual Orientation" &
# sleep 5
# python src/$application/societal.py $model "Sexual Orientation"
# # python src/$application/societal.py $model "Race" &
# sleep 5
# python src/$application/societal.py $model "Race" &
# sleep 5
# python src/$application/societal.py $model "Religious Affiliation" &
# sleep 5
# python src/$application/societal.py $model "Religious Affiliation"
