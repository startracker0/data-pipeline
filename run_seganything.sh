#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_PYTHON=${ENV_PYTHON:-python}
GPU=${GPU:-0}
# DEVICE 是 CUDA_VISIBLE_DEVICES 生效后的逻辑编号；例如 GPU=4 时，PyTorch 中应使用 cuda:0。
DEVICE=${DEVICE:-0}
DTYPE=${DTYPE:-float32}

MODEL_DIR=${MODEL_DIR:-/mnt/xxr/SAM2}
VIDEO=${VIDEO:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0/videos/chunk-000/observation.images.cam_high/episode_000500.mp4}
VIDEO_STEM=""
if [[ -n "${VIDEO}" ]]; then
    VIDEO_STEM="$(basename "${VIDEO}")"
    VIDEO_STEM="${VIDEO_STEM%.*}"
    OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/seg_anything_outputs/${VIDEO_STEM}}
else
    OUT_DIR=${OUT_DIR:-${SCRIPT_DIR}/seg_anything_outputs}
fi
FIRST_FRAME_DIR=${FIRST_FRAME_DIR:-${OUT_DIR}/first_frames}
IMAGE=${IMAGE:-/mnt/xxr/code/sam31_seg_anything_image/video_outputs/episode_003000/stream_tmp/current_frame.png}
FFMPEG="${FFMPEG:-/mnt/xxr/bin/ffmpeg}"

if [[ -n "${VIDEO}" ]]; then
    mkdir -p "${FIRST_FRAME_DIR}"
    IMAGE="${FIRST_FRAME_DIR}/${VIDEO_STEM}_first_frame.png"
    "${FFMPEG}" -y -hide_banner -loglevel error -i "${VIDEO}" -frames:v 1 "${IMAGE}"
fi

POINTS_PER_SIDE=${POINTS_PER_SIDE:-32}
POINTS_PER_BATCH=${POINTS_PER_BATCH:-64}
PRED_IOU_THRESH=${PRED_IOU_THRESH:-0.88}
STABILITY_SCORE_THRESH=${STABILITY_SCORE_THRESH:-0.95}
CROP_N_LAYERS=${CROP_N_LAYERS:-0}
CROP_N_POINTS_DOWNSCALE_FACTOR=${CROP_N_POINTS_DOWNSCALE_FACTOR:-1}
MIN_MASK_REGION_AREA=${MIN_MASK_REGION_AREA:-100}
MAX_NUM_MASKS=${MAX_NUM_MASKS:-200}
ALPHA=${ALPHA:-0.55}
NO_MASK=${NO_MASK:-0}
DRAW_INDEX=${DRAW_INDEX:-0}
if [[ -n "${VIDEO}" ]]; then
    SAVE_PER_MASK=${SAVE_PER_MASK:-1}
    RUN_QWEN_SELECT=${RUN_QWEN_SELECT:-1}
else
    SAVE_PER_MASK=${SAVE_PER_MASK:-0}
    RUN_QWEN_SELECT=${RUN_QWEN_SELECT:-0}
fi
if [[ "${RUN_QWEN_SELECT}" == "1" ]]; then
    SAVE_PER_MASK=1
fi

EPISODES_JSONL=${EPISODES_JSONL:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0/meta/episodes.jsonl}
EPISODE=${EPISODE:-${VIDEO_STEM:-episode_007000}}
QWEN_API_BASE=${QWEN_API_BASE:-http://127.0.0.1:8000/v1}
QWEN_MAX_MASKS=${QWEN_MAX_MASKS:-0}
QWEN_MIN_MASK_AREA=${QWEN_MIN_MASK_AREA:-0}
QWEN_ALLOW_FAIL=${QWEN_ALLOW_FAIL:-0}
QWEN_INCLUDE_FULL_FRAME=${QWEN_INCLUDE_FULL_FRAME:-0}
if [[ -n "${VIDEO}" ]]; then
    RUN_SAM31_TRACK=${RUN_SAM31_TRACK:-1}
else
    RUN_SAM31_TRACK=${RUN_SAM31_TRACK:-0}
fi
SAM31_MODEL_DIR=${SAM31_MODEL_DIR:-/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/models/sam3}
SAM31_CHECKPOINT=${SAM31_CHECKPOINT:-${SAM31_MODEL_DIR}/sam3.pt}
SAM31_REPO=${SAM31_REPO:-${SCRIPT_DIR}/sam3}
SAM31_OUT_DIR=${SAM31_OUT_DIR:-${OUT_DIR}/sam31_point_tracking}
SAM31_INPUT_MODE=${SAM31_INPUT_MODE:-frames}
SAM31_FRAMES_DIR=${SAM31_FRAMES_DIR:-${SAM31_OUT_DIR}/${VIDEO_STEM}_sam31_input_frames}
SAM31_FORCE_EXTRACT_FRAMES=${SAM31_FORCE_EXTRACT_FRAMES:-0}
SAM31_MAX_INPUT_FRAMES=${SAM31_MAX_INPUT_FRAMES:-0}
SAM31_PROMPT_FRAME=${SAM31_PROMPT_FRAME:-0}
SAM31_PROMPT_TYPE=${SAM31_PROMPT_TYPE:-box}
SAM31_TRACK_SESSION_MODE=${SAM31_TRACK_SESSION_MODE:-multi}
SAM31_INFER_PROB_THRESH=${SAM31_INFER_PROB_THRESH:-0.05}
SAM31_PROPAGATE_MODE=${SAM31_PROPAGATE_MODE:-official}
SAM31_PROPAGATE_PROB_THRESH=${SAM31_PROPAGATE_PROB_THRESH:-}
SAM31_RENDER_PROB_THRESH=${SAM31_RENDER_PROB_THRESH:-0.0}
SAM31_MAX_NUM_OBJECTS=${SAM31_MAX_NUM_OBJECTS:-16}
SAM31_MULTIPLEX_COUNT=${SAM31_MULTIPLEX_COUNT:-16}
SAM31_USE_FA3=${SAM31_USE_FA3:-0}
SAM31_SKIP_TAIL_FRAMES=${SAM31_SKIP_TAIL_FRAMES:-0}
SAM31_FPS=${SAM31_FPS:-0}
SAM31_ALPHA=${SAM31_ALPHA:-0.45}
SAM31_NO_MASK=${SAM31_NO_MASK:-0}
SAM31_BBOX_ONLY=${SAM31_BBOX_ONLY:-1}
SAM31_DEBUG_SAVE_FRAMES=${SAM31_DEBUG_SAVE_FRAMES:-0}

echo "[INFO] using python command: ${ENV_PYTHON}"
echo "[INFO] using SAM2 model dir: ${MODEL_DIR}"
echo "[INFO] using dtype: ${DTYPE}"
export CUDA_VISIBLE_DEVICES="${GPU}"
if [[ "${DEVICE}" != "-1" && "${GPU}" != *","* && "${DEVICE}" != "0" ]]; then
    echo "[WARN] CUDA_VISIBLE_DEVICES=${GPU} 只暴露一张卡，PyTorch 逻辑设备应为 cuda:0；自动将 DEVICE=${DEVICE} 改为 0" >&2
    DEVICE=0
fi

EXTRA_ARGS=()
if [[ "${NO_MASK}" == "1" ]]; then
    EXTRA_ARGS+=(--no-mask)
fi
if [[ "${DRAW_INDEX}" == "1" ]]; then
    EXTRA_ARGS+=(--draw-index)
fi
if [[ "${SAVE_PER_MASK}" == "1" ]]; then
    EXTRA_ARGS+=(--save-per-mask)
fi

"${ENV_PYTHON}" "${SCRIPT_DIR}/seganything.py" \
    --image "${IMAGE}" \
    --model-dir "${MODEL_DIR}" \
    --out-dir "${OUT_DIR}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --points-per-side "${POINTS_PER_SIDE}" \
    --points-per-batch "${POINTS_PER_BATCH}" \
    --pred-iou-thresh "${PRED_IOU_THRESH}" \
    --stability-score-thresh "${STABILITY_SCORE_THRESH}" \
    --crop-n-layers "${CROP_N_LAYERS}" \
    --crop-n-points-downscale-factor "${CROP_N_POINTS_DOWNSCALE_FACTOR}" \
    --min-mask-region-area "${MIN_MASK_REGION_AREA}" \
    --max-num-masks "${MAX_NUM_MASKS}" \
    --alpha "${ALPHA}" \
    "${EXTRA_ARGS[@]}"

if [[ "${RUN_QWEN_SELECT}" == "1" ]]; then
    SEG_STEM="$(basename "${IMAGE}")"
    SEG_STEM="${SEG_STEM%.*}"
    SEG_RESULTS="${OUT_DIR}/${SEG_STEM}_seganything_results.json"
    MASKS_DIR="${OUT_DIR}/masks"
    QWEN_SELECTION_JSON="${OUT_DIR}/${SEG_STEM}_qwen_mask_selection.json"
    QWEN_EXTRA_ARGS=()
    if [[ "${QWEN_ALLOW_FAIL}" == "1" ]]; then
        QWEN_EXTRA_ARGS+=(--allow-fail)
    fi
    if [[ "${QWEN_INCLUDE_FULL_FRAME}" == "1" ]]; then
        QWEN_EXTRA_ARGS+=(--include-full-frame)
    fi

    echo "[INFO] running Qwen mask selection"
    echo "[INFO] episode metadata: ${EPISODES_JSONL}"
    echo "[INFO] episode: ${EPISODE}"
    echo "[INFO] Qwen API base: ${QWEN_API_BASE}"
    "${ENV_PYTHON}" "${SCRIPT_DIR}/select_instruction_masks_with_qwen.py" \
        --episodes-jsonl "${EPISODES_JSONL}" \
        --episode "${EPISODE}" \
        --seg-results "${SEG_RESULTS}" \
        --masks-dir "${MASKS_DIR}" \
        --api-base "${QWEN_API_BASE}" \
        --max-masks "${QWEN_MAX_MASKS}" \
        --min-mask-area "${QWEN_MIN_MASK_AREA}" \
        --output "${QWEN_SELECTION_JSON}" \
        "${QWEN_EXTRA_ARGS[@]}"

    if [[ "${RUN_SAM31_TRACK}" == "1" ]]; then
        if [[ -z "${VIDEO}" ]]; then
            echo "[WARN] RUN_SAM31_TRACK=1 但 VIDEO 为空，跳过 SAM3.1 点追踪" >&2
        else
            SAM31_EXTRA_ARGS=()
            if [[ "${SAM31_NO_MASK}" == "1" ]]; then
                SAM31_EXTRA_ARGS+=(--no-mask)
            fi
            if [[ "${SAM31_BBOX_ONLY}" == "1" ]]; then
                SAM31_EXTRA_ARGS+=(--bbox-only)
            fi
            if [[ "${SAM31_FORCE_EXTRACT_FRAMES}" == "1" ]]; then
                SAM31_EXTRA_ARGS+=(--force-extract-frames)
            fi
            if [[ -n "${SAM31_PROPAGATE_PROB_THRESH}" ]]; then
                SAM31_EXTRA_ARGS+=(--propagate-prob-thresh "${SAM31_PROPAGATE_PROB_THRESH}")
            fi
            HAS_SAM31_TARGETS="$(${ENV_PYTHON} - "${QWEN_SELECTION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

selection_json = Path(sys.argv[1])
data = json.loads(selection_json.read_text(encoding="utf-8"))
for answer in data.get("final_answers") or []:
    if not answer.get("found"):
        continue
    bbox = answer.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        continue
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    if x2 > x1 and y2 > y1:
        print("1")
        break
else:
    print("0")
PY
)"
            if [[ "${HAS_SAM31_TARGETS}" != "1" ]]; then
                echo "[WARN] Qwen selection 中没有 found=true 且 bbox_xyxy 有效的目标，跳过 SAM3.1 tracking: ${QWEN_SELECTION_JSON}" >&2
            else
                echo "[INFO] running SAM3 official SAM2-style point tracking"
                echo "[INFO] selection json: ${QWEN_SELECTION_JSON}"
                echo "[INFO] SAM3.1 repo: ${SAM31_REPO}"
                echo "[INFO] SAM3.1 input mode: ${SAM31_INPUT_MODE}"
                echo "[INFO] SAM3.1 prompt type: ${SAM31_PROMPT_TYPE}"
                echo "[INFO] SAM3.1 track session mode: ${SAM31_TRACK_SESSION_MODE}"
                echo "[INFO] SAM3 tracker mode: official_sam2_style"
                echo "[INFO] SAM3 propagate mode: ${SAM31_PROPAGATE_MODE}"
                "${ENV_PYTHON}" "${SCRIPT_DIR}/track_final_answers_with_sam31.py" \
                    --video "${VIDEO}" \
                    --selection-json "${QWEN_SELECTION_JSON}" \
                    --out-dir "${SAM31_OUT_DIR}" \
                    --model-dir "${SAM31_MODEL_DIR}" \
                    --checkpoint "${SAM31_CHECKPOINT}" \
                    --sam3-repo "${SAM31_REPO}" \
                    --sam3-input-mode "${SAM31_INPUT_MODE}" \
                    --frames-dir "${SAM31_FRAMES_DIR}" \
                    --max-input-frames "${SAM31_MAX_INPUT_FRAMES}" \
                    --prompt-frame "${SAM31_PROMPT_FRAME}" \
                    --prompt-type "${SAM31_PROMPT_TYPE}" \
                    --track-session-mode "${SAM31_TRACK_SESSION_MODE}" \
                    --infer-prob-thresh "${SAM31_INFER_PROB_THRESH}" \
                    --propagate-mode "${SAM31_PROPAGATE_MODE}" \
                    --render-prob-thresh "${SAM31_RENDER_PROB_THRESH}" \
                    --max-num-objects "${SAM31_MAX_NUM_OBJECTS}" \
                    --multiplex-count "${SAM31_MULTIPLEX_COUNT}" \
                    --use-fa3 "${SAM31_USE_FA3}" \
                    --skip-tail-frames "${SAM31_SKIP_TAIL_FRAMES}" \
                    --fps "${SAM31_FPS}" \
                    --alpha "${SAM31_ALPHA}" \
                    --debug-save-frames "${SAM31_DEBUG_SAVE_FRAMES}" \
                    --ffmpeg "${FFMPEG}" \
                    "${SAM31_EXTRA_ARGS[@]}"
            fi
        fi
    fi
fi
