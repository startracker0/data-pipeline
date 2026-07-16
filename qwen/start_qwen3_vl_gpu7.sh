#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/mnt/xxr/Qwen3-VL-30B-A3B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-7}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-30B-A3B-Instruct}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
if [[ -z "${LIMIT_MM_PER_PROMPT}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image":4,"video":1}'
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在：${MODEL_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
DEFAULT_TENSOR_PARALLEL_SIZE="${#GPU_ID_LIST[@]}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${DEFAULT_TENSOR_PARALLEL_SIZE}}"

if ! [[ "${TENSOR_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] || [[ "${TENSOR_PARALLEL_SIZE}" -lt 1 ]]; then
  echo "TENSOR_PARALLEL_SIZE 必须是正整数，当前值：${TENSOR_PARALLEL_SIZE}" >&2
  exit 1
fi

if [[ "${TENSOR_PARALLEL_SIZE}" -gt "${DEFAULT_TENSOR_PARALLEL_SIZE}" ]]; then
  echo "TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE} 大于可见 GPU 数量 ${DEFAULT_TENSOR_PARALLEL_SIZE}，请调整 GPU_IDS 或 TENSOR_PARALLEL_SIZE。" >&2
  exit 1
fi

if [[ "${MODEL_PATH}" == *"235B"* && "${TENSOR_PARALLEL_SIZE}" -lt 4 ]]; then
  echo "检测到 235B 模型但 tensor parallel size=${TENSOR_PARALLEL_SIZE}。235B FP8 不适合单卡启动；请改用 30B 模型，或设置 GPU_IDS=0,1,2,3... 并相应调大 TENSOR_PARALLEL_SIZE。" >&2
  exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec('vllm') else 1)
PY
then
  echo "当前 Python 环境未检测到 vLLM，请在运行机上先安装匹配 CUDA 的 vLLM 环境。" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS}"
fi

echo "使用物理 GPU ${GPU_IDS} 启动 ${MODEL_PATH}"
echo "tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "OpenAI 兼容接口：http://${HOST}:${PORT}/v1"

exec python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --trust-remote-code \
  --dtype "${DTYPE}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --enable-prefix-caching \
  "${EXTRA_ARGS[@]}"
