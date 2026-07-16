#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/models/Qwen3-VL-235B-A22B-Instruct-FP8}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-235B-A22B-Instruct-FP8}"
DTYPE="${DTYPE:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
VLLM_USE_V1="${VLLM_USE_V1:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-1}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/qwen3_vl_235b_h20_4gpu_$(date +%Y%m%d_%H%M%S).log}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
if [[ -z "${LIMIT_MM_PER_PROMPT}" ]]; then
  LIMIT_MM_PER_PROMPT='{"image":4,"video":1}'
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "模型目录不存在：${MODEL_PATH}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
VISIBLE_GPU_COUNT="${#GPU_ID_LIST[@]}"

if ! [[ "${TENSOR_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] || [[ "${TENSOR_PARALLEL_SIZE}" -lt 1 ]]; then
  echo "TENSOR_PARALLEL_SIZE 必须是正整数，当前值：${TENSOR_PARALLEL_SIZE}" >&2
  exit 1
fi

if ! [[ "${PIPELINE_PARALLEL_SIZE}" =~ ^[0-9]+$ ]] || [[ "${PIPELINE_PARALLEL_SIZE}" -lt 1 ]]; then
  echo "PIPELINE_PARALLEL_SIZE 必须是正整数，当前值：${PIPELINE_PARALLEL_SIZE}" >&2
  exit 1
fi

WORLD_SIZE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
if [[ "${WORLD_SIZE}" -ne "${VISIBLE_GPU_COUNT}" ]]; then
  echo "可见 GPU 数量=${VISIBLE_GPU_COUNT}，但 TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE=${WORLD_SIZE}。请让 GPU_IDS 数量与并行规模一致。" >&2
  exit 1
fi

if [[ "${VISIBLE_GPU_COUNT}" -lt 4 ]]; then
  echo "235B FP8 默认需要至少 4 张 H20 级别 GPU；当前 GPU_IDS=${GPU_IDS}" >&2
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
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_V1="${VLLM_USE_V1}"

SERVER_ARGS=(
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}"
  --distributed-executor-backend "${DISTRIBUTED_EXECUTOR_BACKEND}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}"
)

if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  SERVER_ARGS+=(--enable-prefix-caching)
fi

if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
  SERVER_ARGS+=(--enable-expert-parallel)
fi

if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  SERVER_ARGS+=(--disable-custom-all-reduce)
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  SERVER_ARGS+=(--enforce-eager)
fi

EXTRA_ARGS=()
if [[ -n "${VLLM_EXTRA_ARGS:-}" ]]; then
  read -r -a EXTRA_ARGS <<< "${VLLM_EXTRA_ARGS}"
fi

mkdir -p "${LOG_DIR}"

echo "使用物理 GPU ${GPU_IDS} 启动 ${MODEL_PATH}"
echo "served model name: ${SERVED_MODEL_NAME}"
echo "tensor parallel size: ${TENSOR_PARALLEL_SIZE}"
echo "pipeline parallel size: ${PIPELINE_PARALLEL_SIZE}"
echo "distributed executor backend: ${DISTRIBUTED_EXECUTOR_BACKEND}"
echo "vllm use v1: ${VLLM_USE_V1}"
echo "max model len: ${MAX_MODEL_LEN}"
echo "max num seqs: ${MAX_NUM_SEQS}"
echo "max num batched tokens: ${MAX_NUM_BATCHED_TOKENS}"
echo "gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"
echo "enable expert parallel: ${ENABLE_EXPERT_PARALLEL}"
echo "disable custom all reduce: ${DISABLE_CUSTOM_ALL_REDUCE}"
echo "enable prefix caching: ${ENABLE_PREFIX_CACHING}"
echo "log file: ${LOG_FILE}"
echo "OpenAI 兼容接口：http://${HOST}:${PORT}/v1"
echo "启动命令：python -m vllm.entrypoints.openai.api_server ${SERVER_ARGS[*]} ${EXTRA_ARGS[*]}"

set +e
python -m vllm.entrypoints.openai.api_server "${SERVER_ARGS[@]}" "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "vLLM 启动失败，完整日志在：${LOG_FILE}" >&2
  echo "建议先查看真正 root cause：grep -nEi 'error|exception|traceback|out of memory|cuda|nccl|failed' ${LOG_FILE} | tail -n 80" >&2
fi

exit "${EXIT_CODE}"
