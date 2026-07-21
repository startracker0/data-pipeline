#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Qwen final_answers 的 bbox 中心点作为 SAM3.1 视频点提示进行目标追踪。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 final_answers[*].bbox_xyxy，取中心点送入 SAM3.1 做视频目标追踪。"
    )
    parser.add_argument("--video", type=Path, required=True, help="输入视频路径")
    parser.add_argument("--selection-json", type=Path, required=True, help="Qwen mask selection JSON，需包含 final_answers")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--model-dir", type=Path, default=Path("/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/models/sam3"), help="SAM3 模型目录")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint 路径，默认 <model-dir>/sam3.pt")
    parser.add_argument("--sam3-repo", type=Path, default=Path("/mnt/xxr/code/pipeline/sam3"), help="本地 sam3 源码目录")
    parser.add_argument("--sam3-input-mode", choices=("frames", "video"), default="frames", help="SAM3.1 输入方式；frames 会先用 ffmpeg 抽帧目录，绕过 SAM3 内部 mp4 解码")
    parser.add_argument("--frames-dir", type=Path, default=None, help="抽帧目录；默认 <out-dir>/<video_stem>_sam31_input_frames")
    parser.add_argument("--force-extract-frames", action="store_true", help="强制重新抽帧，覆盖 frames-dir 中已有图片")
    parser.add_argument("--max-input-frames", type=int, default=0, help="只抽取/追踪前 N 帧，0 表示全视频；用于单独调试")
    parser.add_argument("--prompt-frame", type=int, default=0, help="点提示所在帧，默认首帧 0")
    parser.add_argument("--prompt-type", choices=("point", "box"), default="point", help="送入SAM3的首帧提示类型；point 使用bbox中心点做实例追踪；box 使用bbox做实例追踪")
    parser.add_argument("--track-session-mode", choices=("multi",), default="multi", help="保留兼容参数；官方SAM2-style tracker使用单inference_state多obj_id流程")
    parser.add_argument("--infer-prob-thresh", type=float, default=0.05, help="保留兼容参数；SAM2-style tracker不使用此阈值")
    parser.add_argument("--propagate-mode", choices=("official", "bounded"), default="official", help="official 从第0帧正向传播全视频；bounded 从prompt帧双向传播")
    parser.add_argument("--propagate-prob-thresh", type=float, default=None, help="保留兼容参数；SAM2-style tracker不使用此阈值")
    parser.add_argument("--render-prob-thresh", type=float, default=0.0, help="可视化和JSON输出时保留的概率阈值")
    parser.add_argument("--max-num-objects", type=int, default=16)
    parser.add_argument("--multiplex-count", type=int, default=16)
    parser.add_argument("--use-fa3", type=int, default=0, help="是否启用 FlashAttention3；环境缺少时保持 0")
    parser.add_argument("--skip-tail-frames", type=int, default=0, help="末尾跳过传播的帧数，用于规避坏尾帧")
    parser.add_argument("--fps", type=float, default=0.0, help="输出视频 fps，0 表示使用源视频 fps")
    parser.add_argument("--alpha", type=float, default=0.45, help="mask 叠加透明度")
    parser.add_argument("--no-mask", action="store_true", help="只画 bbox/点，不叠加 mask")
    parser.add_argument("--debug-save-frames", type=int, default=0, help="额外保存前 N 帧可视化 png")
    parser.add_argument("--ffmpeg", type=str, default="/mnt/xxr/bin/ffmpeg")
    return parser.parse_args()


def resolve_executable(value: str, name: str) -> str | None:
    value = str(value or "").strip()
    candidates = []
    if value:
        candidates.append(value)
    found = shutil.which(name)
    if found:
        candidates.append(found)
    candidates.extend([f"/usr/local/bin/{name}", f"/usr/bin/{name}", f"/mnt/xxr/bin/{name}"])
    for item in candidates:
        path = Path(item)
        try:
            if path.is_file() and os.access(path, os.X_OK):
                return str(path.resolve())
        except OSError:
            continue
    return None


def to_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.float()
        return tensor.cpu().numpy()
    return np.asarray(value)


def stable_color(index: int) -> np.ndarray:
    rng = np.random.default_rng(20260716 + int(index) * 9973)
    return rng.integers(40, 256, size=3, dtype=np.uint8)


COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "cyan",
    "gold",
    "gray",
    "grey",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "teal",
    "white",
    "yellow",
}
RELATION_WORDS = {
    "above",
    "behind",
    "below",
    "beside",
    "between",
    "front",
    "inside",
    "left",
    "near",
    "next",
    "of",
    "on",
    "right",
    "side",
    "under",
}
RELATION_PHRASE_PREFIXES = (
    "left of",
    "left side",
    "right of",
    "right side",
    "on the left",
    "on the right",
    "next to",
    "beside",
    "near",
)
NON_OBJECT_BASES = COLOR_WORDS | RELATION_WORDS | {"the", "a", "an", "with", "side", "position"}


def normalize_label_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("_", " ").replace("-", " ").split())


def is_non_object_answer(answer: dict[str, Any]) -> bool:
    prompt = normalize_label_text(answer.get("prompt"))
    base_object = normalize_label_text(answer.get("base_object"))
    if base_object in NON_OBJECT_BASES:
        return True
    prompt_words = prompt.split()
    if len(prompt_words) == 1 and prompt in NON_OBJECT_BASES:
        return True
    if any(prompt.startswith(prefix) for prefix in RELATION_PHRASE_PREFIXES):
        return True
    if prompt_words and all(word in NON_OBJECT_BASES for word in prompt_words):
        return True
    return False


def display_label(info: dict[str, Any]) -> str:
    base_object = str(info.get("base_object") or "").strip()
    if base_object:
        return base_object
    return str(info.get("prompt") or "object").strip() or "object"


def load_final_answer_points(selection_json: Path) -> list[dict[str, Any]]:
    data = json.loads(selection_json.read_text(encoding="utf-8"))
    answers = data.get("final_answers")
    if not isinstance(answers, list):
        raise ValueError(f"selection JSON 缺少 final_answers: {selection_json}")

    prompts: list[dict[str, Any]] = []
    for answer in answers:
        if not answer or not answer.get("found"):
            continue
        if is_non_object_answer(answer):
            continue
        bbox = answer.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        if x2 <= x1 or y2 <= y1:
            continue
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        prompts.append(
            {
                "prompt": str(answer.get("prompt", "object")),
                "base_object": str(answer.get("base_object") or "").strip(),
                "mask_index": answer.get("mask_index"),
                "bbox_xyxy": [x1, y1, x2, y2],
                "point_xy": [cx, cy],
                "confidence": answer.get("confidence"),
            }
        )
    if not prompts:
        raise ValueError(f"final_answers 中没有 found=true 且 bbox_xyxy 有效的目标: {selection_json}")
    return prompts


def video_info(video: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"无法读取视频宽高: {video}")
    if fps <= 0:
        fps = 30.0
    return width, height, fps, frames


def frame_sort_key(path: Path) -> tuple[int, Any]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.name


def list_frame_paths(frames_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    return sorted(
        [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts],
        key=frame_sort_key,
    )


def extract_video_frames(
    video: Path,
    frames_dir: Path,
    ffmpeg: str,
    force: bool,
    max_input_frames: int,
) -> list[Path]:
    existing = list_frame_paths(frames_dir) if frames_dir.is_dir() else []
    if existing and not force:
        expected_count = int(max_input_frames) if max_input_frames and max_input_frames > 0 else 0
        if expected_count <= 0 or len(existing) == expected_count:
            print(f"[FRAMES] reuse existing frames_dir={frames_dir} count={len(existing)}", flush=True)
            return existing
        print(
            f"[FRAMES] existing count={len(existing)} != max_input_frames={expected_count}, re-extracting",
            flush=True,
        )
        force = True

    if frames_dir.exists() and force:
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = frames_dir / "%06d.jpg"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video),
    ]
    if max_input_frames and max_input_frames > 0:
        cmd += ["-frames:v", str(int(max_input_frames))]
    cmd += ["-q:v", "2", "-start_number", "0", str(output_pattern)]
    print(f"[FRAMES] extracting with ffmpeg to {frames_dir}", flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg 抽帧失败: {exc.stderr or exc.stdout}") from exc

    frames = list_frame_paths(frames_dir)
    if not frames:
        raise RuntimeError(f"ffmpeg 抽帧后没有得到任何图片: {frames_dir}")
    print(f"[FRAMES] extracted count={len(frames)} first={frames[0].name} last={frames[-1].name}", flush=True)
    return frames


def frames_dir_info(frames_dir: Path) -> tuple[int, int, int]:
    frames = list_frame_paths(frames_dir)
    if not frames:
        raise RuntimeError(f"帧目录为空: {frames_dir}")
    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"无法读取首帧: {frames[0]}")
    height, width = first.shape[:2]
    return width, height, len(frames)


def init_tracker_state(predictor, resource_path: Path) -> tuple[dict[str, Any], int]:
    inference_state = predictor.init_state(
        video_path=str(resource_path),
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    )
    num_frames = int(inference_state.get("num_frames", 0))
    if num_frames <= 0:
        raise RuntimeError("SAM3 tracker init_state 后 num_frames<=0，视频/帧目录可能不可解码")
    print(f"[TRACKER_STATE] initialized resource={resource_path} num_frames={num_frames}", flush=True)
    return inference_state, num_frames


def _mask_to_box_xywh_rel(mask: np.ndarray, width: int, height: int) -> list[float]:
    mask_bool = np.squeeze(mask > 0.0)
    if mask_bool.ndim != 2 or not mask_bool.any():
        return [0.0, 0.0, 0.0, 0.0]
    ys, xs = np.where(mask_bool)
    x1 = float(xs.min()) / float(width)
    y1 = float(ys.min()) / float(height)
    x2 = float(xs.max() + 1) / float(width)
    y2 = float(ys.max() + 1) / float(height)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def tracker_output_to_frame_out(
    obj_ids: Any,
    video_res_masks: Any,
    obj_scores: Any,
    width: int,
    height: int,
) -> dict[str, np.ndarray]:
    obj_ids_np = to_numpy(obj_ids)
    masks_np = to_numpy(video_res_masks)
    scores_np = to_numpy(obj_scores)
    if obj_ids_np is None:
        obj_ids_np = np.zeros((0,), dtype=np.int64)
    obj_ids_np = obj_ids_np.reshape(-1).astype(np.int64)
    if masks_np is None:
        masks_np = np.zeros((0, height, width), dtype=np.float32)
    masks_np = np.asarray(masks_np)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    masks_np = masks_np.astype(np.float32)
    binary_masks_np = (masks_np > 0.0).astype(np.float32)
    if scores_np is None:
        probs_np = np.ones((len(obj_ids_np),), dtype=np.float32)
    else:
        scores_np = np.asarray(scores_np).reshape(-1).astype(np.float32)
        probs_np = 1.0 / (1.0 + np.exp(-scores_np))
        if len(probs_np) < len(obj_ids_np):
            probs_np = np.pad(probs_np, (0, len(obj_ids_np) - len(probs_np)), constant_values=1.0)
    boxes = np.asarray([_mask_to_box_xywh_rel(binary_masks_np[k], width, height) for k in range(min(len(obj_ids_np), len(binary_masks_np)))], dtype=np.float32)
    if len(boxes) < len(obj_ids_np):
        pad = np.zeros((len(obj_ids_np) - len(boxes), 4), dtype=np.float32)
        boxes = np.concatenate([boxes, pad], axis=0) if len(boxes) else pad
    return {
        "out_obj_ids": obj_ids_np,
        "out_probs": probs_np[: len(obj_ids_np)],
        "out_tracker_probs": probs_np[: len(obj_ids_np)],
        "out_boxes_xywh": boxes[: len(obj_ids_np)],
        "out_binary_masks": binary_masks_np[: len(obj_ids_np)],
    }


def add_point_prompts(
    predictor,
    inference_state: dict[str, Any],
    point_prompts: list[dict[str, Any]],
    width: int,
    height: int,
    frame_index: int,
    prompt_type: str,
) -> dict[int, dict[str, Any]]:
    obj_id_to_info: dict[int, dict[str, Any]] = {}
    for obj_idx, item in enumerate(point_prompts, start=1):
        x, y = item["point_xy"]
        x1, y1, x2, y2 = [float(v) for v in item["bbox_xyxy"]]
        rel_point = [[float(x) / float(width), float(y) / float(height)]]
        points = torch.tensor(rel_point, dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int32)
        if prompt_type == "point":
            _, out_obj_ids, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_index,
                obj_id=obj_idx,
                points=points,
                labels=labels,
                clear_old_points=True,
                rel_coordinates=True,
            )
        elif prompt_type == "box":
            rel_box_xyxy = [x1 / float(width), y1 / float(height), x2 / float(width), y2 / float(height)]
            _, out_obj_ids, _, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_index,
                obj_id=obj_idx,
                box=torch.tensor(rel_box_xyxy, dtype=torch.float32),
                clear_old_points=True,
                rel_coordinates=True,
            )
        else:
            raise ValueError(f"不支持的 prompt_type: {prompt_type}")
        ids_np = to_numpy(out_obj_ids)
        ids_list = [obj_idx] if ids_np is None or len(ids_np) == 0 else [int(v) for v in ids_np.reshape(-1).tolist()]
        for oid in ids_list:
            obj_id_to_info.setdefault(
                int(oid),
                {
                    "prompt": item["prompt"],
                    "base_object": item.get("base_object", ""),
                    "mask_index": item.get("mask_index"),
                    "init_bbox_xyxy": item["bbox_xyxy"],
                    "init_point_xy": item["point_xy"],
                    "confidence": item.get("confidence"),
                },
            )
        print(f"[PROMPT] obj_id={obj_idx} type={prompt_type} prompt={item['prompt']} base_object={item.get('base_object', '')} point=({x:.1f},{y:.1f}) bbox={item['bbox_xyxy']} returned_obj_ids={ids_list}", flush=True)
    return obj_id_to_info


def propagate_video(
    predictor,
    inference_state: dict[str, Any],
    prompt_frame: int,
    num_frames: int,
    propagate_mode: str,
    skip_tail_frames: int,
    width: int,
    height: int,
) -> dict[int, dict]:
    safe_last_frame = max(prompt_frame, num_frames - 1 - max(0, int(skip_tail_frames)))
    requests: list[dict[str, Any]] = []
    if propagate_mode == "official":
        requests.append(dict(start_frame_idx=0, max_frame_num_to_track=safe_last_frame, reverse=False, propagate_preflight=True))
    else:
        requests.append(dict(start_frame_idx=prompt_frame, max_frame_num_to_track=max(0, safe_last_frame - prompt_frame), reverse=False, propagate_preflight=True))
        if prompt_frame > 0:
            requests.append(dict(start_frame_idx=prompt_frame, max_frame_num_to_track=prompt_frame, reverse=True, propagate_preflight=False))

    outputs_per_frame: dict[int, dict] = {}
    for req in requests:
        print(f"[PROPAGATE] tracker_args={req}", flush=True)
        for frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            tqdm_disable=False,
            **req,
        ):
            frame_idx = int(frame_idx)
            outputs_per_frame[frame_idx] = tracker_output_to_frame_out(
                obj_ids=obj_ids,
                video_res_masks=video_res_masks,
                obj_scores=obj_scores,
                width=width,
                height=height,
            )
            if frame_idx % 10 == 0:
                n_obj = len(outputs_per_frame[frame_idx].get("out_obj_ids", []))
                print(f"[FRAME] {frame_idx} n_obj={n_obj}", flush=True)
    return outputs_per_frame


def merge_single_object_outputs(
    merged: dict[int, dict],
    single_outputs: dict[int, dict],
    source_obj_id: int,
    target_obj_id: int,
) -> None:
    for frame_idx, frame_out in single_outputs.items():
        obj_ids_np = to_numpy(frame_out.get("out_obj_ids", []))
        if obj_ids_np is None or len(obj_ids_np) == 0:
            continue
        obj_ids_flat = obj_ids_np.reshape(-1).astype(np.int64)
        matches = np.where(obj_ids_flat == int(source_obj_id))[0]
        if len(matches) == 0:
            matches = np.arange(len(obj_ids_flat), dtype=np.int64)
        if len(matches) == 0:
            continue

        dst = merged.setdefault(frame_idx, {})
        for key, value in frame_out.items():
            if key == "out_obj_ids":
                selected = np.full((len(matches),), int(target_obj_id), dtype=np.int64)
            elif key in {"out_probs", "out_tracker_probs", "out_boxes_xywh", "out_binary_masks"}:
                arr = to_numpy(value)
                if arr is None or len(arr) == 0:
                    selected = arr
                else:
                    selected = arr[matches]
            else:
                continue

            if selected is None:
                continue
            existing = dst.get(key)
            if existing is None:
                dst[key] = selected
            else:
                existing_np = to_numpy(existing)
                if existing_np is None or len(existing_np) == 0:
                    dst[key] = selected
                else:
                    dst[key] = np.concatenate([existing_np, selected], axis=0)


def dump_results_jsonl(
    outputs_per_frame: dict[int, dict],
    obj_id_to_info: dict[int, dict[str, Any]],
    out_path: Path,
    n_frames: int,
    prob_thresh: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fw:
        for frame_idx in range(n_frames):
            frame_out = outputs_per_frame.get(frame_idx, {})
            boxes = to_numpy(frame_out.get("out_boxes_xywh", []))
            probs = to_numpy(frame_out.get("out_probs", []))
            obj_ids = to_numpy(frame_out.get("out_obj_ids", []))
            masks = to_numpy(frame_out.get("out_binary_masks", None))
            detections = []
            if obj_ids is not None and len(obj_ids) > 0:
                for k in range(len(obj_ids)):
                    prob = float(probs[k]) if probs is not None and k < len(probs) else 1.0
                    if prob < prob_thresh:
                        continue
                    if masks is not None and k < len(masks) and not (masks[k] > 0.5).any():
                        continue
                    box = boxes[k].tolist() if boxes is not None and k < len(boxes) else [0.0, 0.0, 0.0, 0.0]
                    oid = int(obj_ids[k])
                    info = obj_id_to_info.get(oid, {})
                    detections.append(
                        {
                            "obj_id": oid,
                            "prompt": info.get("prompt", ""),
                            "base_object": info.get("base_object", ""),
                            "display_label": display_label(info),
                            "mask_index": info.get("mask_index"),
                            "bbox_xywh_rel": [float(v) for v in box[:4]],
                            "prob": prob,
                        }
                    )
            fw.write(json.dumps({"frame_idx": frame_idx, "detections": detections}, ensure_ascii=False) + "\n")


def draw_frame(
    image_rgb: np.ndarray,
    frame_out: dict,
    obj_id_to_info: dict[int, dict[str, Any]],
    alpha: float,
    draw_mask: bool,
    render_prob_thresh: float,
) -> np.ndarray:
    canvas = image_rgb.copy()
    height, width = canvas.shape[:2]
    boxes = to_numpy(frame_out.get("out_boxes_xywh", []))
    probs = to_numpy(frame_out.get("out_probs", []))
    obj_ids = to_numpy(frame_out.get("out_obj_ids", []))
    masks = to_numpy(frame_out.get("out_binary_masks", None))
    if obj_ids is None or len(obj_ids) == 0:
        return canvas

    for k in range(len(obj_ids)):
        oid = int(obj_ids[k])
        prob = float(probs[k]) if probs is not None and k < len(probs) else 1.0
        if prob < render_prob_thresh:
            continue
        color = stable_color(oid)
        if draw_mask and masks is not None and k < len(masks):
            mask = np.squeeze(masks[k] > 0.5)
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            colored = canvas.copy()
            colored[mask] = color
            canvas = np.where(mask[..., None], (alpha * colored + (1.0 - alpha) * canvas).astype(np.uint8), canvas)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color.tolist(), 2)

        if boxes is not None and k < len(boxes):
            x, y, w, h = [float(v) for v in boxes[k].tolist()[:4]]
            x1 = int(round(x * width))
            y1 = int(round(y * height))
            x2 = int(round((x + w) * width))
            y2 = int(round((y + h) * height))
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width - 1, x2))
            y2 = max(0, min(height - 1, y2))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color.tolist(), 2)
            info = obj_id_to_info.get(oid, {})
            label = f"{oid}:{display_label(info)} {prob:.2f}".strip()
            cv2.putText(canvas, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color.tolist(), 1, cv2.LINE_AA)

        info = obj_id_to_info.get(oid, {})
        point = info.get("init_point_xy")
        if point and len(point) >= 2:
            px = int(round(float(point[0])))
            py = int(round(float(point[1])))
            cv2.circle(canvas, (px, py), 4, color.tolist(), -1)
    return canvas


def render_video(
    video: Path,
    outputs_per_frame: dict[int, dict],
    obj_id_to_info: dict[int, dict[str, Any]],
    out_video: Path,
    fps: float,
    alpha: float,
    draw_mask: bool,
    render_prob_thresh: float,
    debug_save_frames: int,
    ffmpeg: str | None,
) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video}")
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"视频为空: {video}")
    height, width = first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video = out_video.with_name(out_video.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"cv2.VideoWriter 打开失败: {raw_video}")

    debug_dir = out_video.parent / "debug_frames"
    if debug_save_frames > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = draw_frame(
            image_rgb=image_rgb,
            frame_out=outputs_per_frame.get(frame_idx, {}),
            obj_id_to_info=obj_id_to_info,
            alpha=alpha,
            draw_mask=draw_mask,
            render_prob_thresh=render_prob_thresh,
        )
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        writer.write(overlay_bgr)
        if debug_save_frames > 0 and frame_idx < debug_save_frames:
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}.png"), overlay_bgr)
        frame_idx += 1

    cap.release()
    writer.release()

    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-an",
            str(out_video),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            raw_video.unlink(missing_ok=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] ffmpeg 转码失败，保留 raw mp4v: {exc.stderr}", file=sys.stderr)
            raw_video.replace(out_video)
    else:
        raw_video.replace(out_video)
    return frame_idx


def render_frames_dir_video(
    frames_dir: Path,
    outputs_per_frame: dict[int, dict],
    obj_id_to_info: dict[int, dict[str, Any]],
    out_video: Path,
    fps: float,
    alpha: float,
    draw_mask: bool,
    render_prob_thresh: float,
    debug_save_frames: int,
    ffmpeg: str | None,
) -> int:
    frames = list_frame_paths(frames_dir)
    if not frames:
        raise RuntimeError(f"帧目录为空，无法渲染视频: {frames_dir}")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"无法读取首帧: {frames[0]}")
    height, width = first.shape[:2]

    out_video.parent.mkdir(parents=True, exist_ok=True)
    raw_video = out_video.with_name(out_video.stem + "_raw.mp4")
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter 打开失败: {raw_video}")

    debug_dir = out_video.parent / "debug_frames"
    if debug_save_frames > 0:
        debug_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    for frame_idx, frame_path in enumerate(frames):
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"无法读取帧: {frame_path}")
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb = draw_frame(
            image_rgb=image_rgb,
            frame_out=outputs_per_frame.get(frame_idx, {}),
            obj_id_to_info=obj_id_to_info,
            alpha=alpha,
            draw_mask=draw_mask,
            render_prob_thresh=render_prob_thresh,
        )
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        writer.write(overlay_bgr)
        if debug_save_frames > 0 and frame_idx < debug_save_frames:
            cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}.png"), overlay_bgr)
        rendered += 1

    writer.release()

    if ffmpeg:
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw_video),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            "-an",
            str(out_video),
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            raw_video.unlink(missing_ok=True)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] ffmpeg 转码失败，保留 raw mp4v: {exc.stderr}", file=sys.stderr)
            raw_video.replace(out_video)
    else:
        raw_video.replace(out_video)
    return rendered


def main() -> None:
    args = parse_args()
    video = args.video.expanduser().resolve()
    selection_json = args.selection_json.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint = (args.checkpoint or (model_dir / "sam3.pt")).expanduser().resolve()
    sam3_repo = args.sam3_repo.expanduser().resolve()

    if not video.is_file():
        raise FileNotFoundError(f"视频不存在: {video}")
    if not selection_json.is_file():
        raise FileNotFoundError(f"selection JSON 不存在: {selection_json}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3 checkpoint 不存在: {checkpoint}")
    if not sam3_repo.is_dir():
        raise FileNotFoundError(f"SAM3 源码目录不存在: {sam3_repo}")
    if not torch.cuda.is_available():
        raise RuntimeError("SAM3 需要 CUDA GPU；当前 torch.cuda.is_available() 为 False")

    sys.path.insert(0, str(sam3_repo))
    from sam3.model_builder import build_sam3_video_model

    out_dir.mkdir(parents=True, exist_ok=True)
    point_prompts = load_final_answer_points(selection_json)
    ffmpeg = resolve_executable(args.ffmpeg, "ffmpeg")
    if args.sam3_input_mode == "frames" and not ffmpeg:
        raise RuntimeError("--sam3-input-mode frames 需要可用的 ffmpeg")

    try:
        width, height, source_fps, cv2_frames = video_info(video)
    except Exception as exc:
        if args.sam3_input_mode == "video":
            raise
        print(f"[WARN] OpenCV 读取视频元数据失败，frames 模式下继续用 ffmpeg 抽帧: {exc}", file=sys.stderr)
        width, height, source_fps, cv2_frames = 0, 0, 30.0, 0
    output_fps = float(args.fps) if args.fps and args.fps > 0 else source_fps
    prompt_frame = max(0, int(args.prompt_frame))

    frames_dir = (args.frames_dir or (out_dir / f"{video.stem}_sam31_input_frames")).expanduser().resolve()
    sam3_resource_path = video
    sam3_input_mode = args.sam3_input_mode
    input_frame_count = cv2_frames
    if sam3_input_mode == "frames":
        extracted_frames = extract_video_frames(
            video=video,
            frames_dir=frames_dir,
            ffmpeg=str(ffmpeg),
            force=bool(args.force_extract_frames),
            max_input_frames=int(args.max_input_frames),
        )
        width, height, input_frame_count = frames_dir_info(frames_dir)
        sam3_resource_path = frames_dir
    elif args.max_input_frames and args.max_input_frames > 0:
        print("[WARN] --max-input-frames 只在 --sam3-input-mode frames 下生效", file=sys.stderr)

    print("============================================================")
    print(f"[VIDEO]          {video}")
    print(f"[SAM3_INPUT]     mode={sam3_input_mode} resource={sam3_resource_path}")
    if sam3_input_mode == "frames":
        print(f"[FRAMES_DIR]     {frames_dir}")
    print(f"[SELECTION_JSON] {selection_json}")
    print(f"[SAM3_REPO]      {sam3_repo}")
    print(f"[CHECKPOINT]     {checkpoint}")
    print(f"[SIZE]           {width}x{height}")
    print(f"[FPS]            source={source_fps:.6f}, output={output_fps:.6f}")
    print(f"[INPUT_FRAMES]   {input_frame_count}")
    print(f"[PROMPT_TYPE]    {args.prompt_type}")
    print(f"[TRACKER_MODE]   official_sam2_style")
    print(f"[PROPAGATE_MODE] {args.propagate_mode}")
    print(f"[POINT_PROMPTS]  {point_prompts}")
    print("============================================================")

    sam3_model = build_sam3_video_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        strict_state_dict_loading=True,
        device="cuda",
    )
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone

    inference_state, num_frames = init_tracker_state(predictor, sam3_resource_path)
    actual_prompt_frame = min(prompt_frame, num_frames - 1)
    obj_id_to_info = add_point_prompts(
        predictor=predictor,
        inference_state=inference_state,
        point_prompts=point_prompts,
        width=width,
        height=height,
        frame_index=actual_prompt_frame,
        prompt_type=args.prompt_type,
    )
    outputs_per_frame = propagate_video(
        predictor=predictor,
        inference_state=inference_state,
        prompt_frame=actual_prompt_frame,
        num_frames=num_frames,
        propagate_mode=args.propagate_mode,
        skip_tail_frames=args.skip_tail_frames,
        width=width,
        height=height,
    )

    non_empty_frames = sum(
        1 for frame_out in outputs_per_frame.values()
        if len(to_numpy(frame_out.get("out_obj_ids", []))) > 0
    )
    print(f"[TRACKER] non_empty_frames={non_empty_frames}/{num_frames}", flush=True)

    results_jsonl = out_dir / f"{video.stem}_sam31_point_tracking_results.jsonl"
    metadata_json = out_dir / f"{video.stem}_sam31_point_tracking_metadata.json"
    output_video = out_dir / f"{video.stem}_sam31_point_tracking_overlay.mp4"
    dump_results_jsonl(outputs_per_frame, obj_id_to_info, results_jsonl, num_frames, args.render_prob_thresh)
    if sam3_input_mode == "frames":
        rendered_frames = render_frames_dir_video(
            frames_dir=frames_dir,
            outputs_per_frame=outputs_per_frame,
            obj_id_to_info=obj_id_to_info,
            out_video=output_video,
            fps=output_fps,
            alpha=float(args.alpha),
            draw_mask=not args.no_mask,
            render_prob_thresh=float(args.render_prob_thresh),
            debug_save_frames=int(args.debug_save_frames),
            ffmpeg=ffmpeg,
        )
    else:
        rendered_frames = render_video(
            video=video,
            outputs_per_frame=outputs_per_frame,
            obj_id_to_info=obj_id_to_info,
            out_video=output_video,
            fps=output_fps,
            alpha=float(args.alpha),
            draw_mask=not args.no_mask,
            render_prob_thresh=float(args.render_prob_thresh),
            debug_save_frames=int(args.debug_save_frames),
            ffmpeg=ffmpeg,
        )

    metadata = {
        "video": str(video),
        "sam3_input_mode": sam3_input_mode,
        "sam3_resource_path": str(sam3_resource_path),
        "frames_dir": str(frames_dir) if sam3_input_mode == "frames" else "",
        "selection_json": str(selection_json),
        "checkpoint": str(checkpoint),
        "sam3_repo": str(sam3_repo),
        "width": width,
        "height": height,
        "source_fps": source_fps,
        "output_fps": output_fps,
        "cv2_reported_frames": cv2_frames,
        "input_frame_count": input_frame_count,
        "sam3_num_frames": num_frames,
        "rendered_frames": rendered_frames,
        "prompt_frame": actual_prompt_frame,
        "prompt_type": args.prompt_type,
        "track_session_mode": "official_sam2_style",
        "infer_prob_thresh": args.infer_prob_thresh,
        "propagate_mode": args.propagate_mode,
        "propagate_prob_thresh": args.propagate_prob_thresh,
        "render_prob_thresh": args.render_prob_thresh,
        "point_prompts": point_prompts,
        "obj_id_to_info": obj_id_to_info,
        "results_jsonl": str(results_jsonl),
        "output_video": str(output_video),
    }
    metadata_json.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] results_jsonl = {results_jsonl}")
    print(f"[DONE] metadata      = {metadata_json}")
    print(f"[DONE] output_video  = {output_video}")


if __name__ == "__main__":
    main()
