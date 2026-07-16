#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于 Hugging Face transformers 的 SAM2 自动全图 mask 生成。"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import pipeline

try:
    from transformers import AutoModelForMaskGeneration
except ImportError:  # 兼容较旧 transformers；后面会自动退回 pipeline 直连加载。
    AutoModelForMaskGeneration = None

try:
    from transformers import AutoImageProcessor
except ImportError:
    AutoImageProcessor = None

try:
    from transformers import Sam2ImageProcessor
except ImportError:
    Sam2ImageProcessor = None


DEFAULT_IMAGE = Path(
    "/mnt/xxr/code/sam31_seg_anything_image/video_outputs/episode_003000/stream_tmp/current_frame.png"
)
DEFAULT_MODEL_DIR = Path("/mnt/xxr/SAM2")
DEFAULT_OUTPUT_DIR = Path("./seg_anything_outputs")


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用本地 SAM2 权重做 Segment Anything 自动全图 mask 生成。"
    )
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="输入图片路径")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="SAM2 Hugging Face 模型目录")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--device", type=int, default=0, help="transformers pipeline 使用的设备，GPU 通常为 0，CPU 为 -1")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32", help="模型加载精度，默认 float32 以避免 SAM2 后处理 NMS 的 dtype 不一致问题")

    parser.add_argument("--points-per-side", type=int, default=32, help="每边采样点数；传给 transformers 的 points_per_crop")
    parser.add_argument("--points-per-batch", type=int, default=64, help="每批处理多少个采样点")
    parser.add_argument("--pred-iou-thresh", type=float, default=0.88, help="SAM 自动 mask 生成的 predicted IoU 阈值")
    parser.add_argument("--stability-score-thresh", type=float, default=0.95, help="SAM 自动 mask 生成的 stability score 阈值")
    parser.add_argument("--crop-n-layers", type=int, default=0, help="自动 mask 生成的 crop 层数，较大可提升小物体召回但更慢")
    parser.add_argument("--crop-n-points-downscale-factor", type=int, default=1, help="crop 层采样点下采样系数")
    parser.add_argument("--min-mask-region-area", type=int, default=100, help="过滤 mask 面积（像素）的最小值")
    parser.add_argument("--max-num-masks", type=int, default=200, help="最终最多保留的 mask 数量")

    parser.add_argument("--alpha", type=float, default=0.55, help="mask 叠加透明度")
    parser.add_argument("--no-mask", action="store_true", help="只画 bbox 不叠加 mask")
    parser.add_argument("--draw-index", action="store_true", help="在 bbox 上标注实例序号")
    parser.add_argument("--save-per-mask", action="store_true", help="保存每个实例单独的二值 mask PNG")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def ensure_inputs(args: argparse.Namespace) -> Tuple[Path, Path]:
    image_path = args.image.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在: {image_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"SAM2 模型目录不存在: {model_dir}")

    return image_path, model_dir


def stable_color(index: int) -> np.ndarray:
    rng = np.random.default_rng(20260716 + int(index) * 9973)
    return rng.integers(40, 256, size=3, dtype=np.uint8)


def to_numpy_mask(mask: Any, height: int, width: int) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask)
    elif torch.is_tensor(mask):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            arr = arr[..., 0]

    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    return arr.astype(bool)


def mask_to_bbox_xyxy(mask: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def normalize_box_xyxy(box: Any, mask: np.ndarray) -> List[int]:
    if box is None:
        return list(mask_to_bbox_xyxy(mask))

    values = np.asarray(box).astype(float).reshape(-1).tolist()
    if len(values) < 4:
        return list(mask_to_bbox_xyxy(mask))

    x1, y1, v3, v4 = values[:4]

    # transformers 的 mask-generation 可能返回 xyxy，也可能返回 xywh；优先用 mask 兜底校正。
    if v3 <= x1 or v4 <= y1:
        x2 = x1 + max(0.0, v3)
        y2 = y1 + max(0.0, v4)
    else:
        x2 = v3
        y2 = v4

    bx1, by1, bx2, by2 = [int(round(v)) for v in (x1, y1, x2, y2)]
    if bx2 <= bx1 or by2 <= by1:
        return list(mask_to_bbox_xyxy(mask))
    return [bx1, by1, bx2, by2]


def extract_list(outputs: dict, *keys: str) -> list:
    for key in keys:
        value = outputs.get(key)
        if value is not None:
            return list(value)
    return []

def patch_nms_dtype_mismatch() -> None:
    """修复 SAM2 后处理中 torchvision NMS 要求 boxes/scores dtype 完全一致的问题。"""
    try:
        import torchvision.ops as torchvision_ops
        import torchvision.ops.boxes as torchvision_boxes
    except Exception as exc:
        print(f"[WARN] skip torchvision NMS dtype patch: {exc}", flush=True)
        return

    original_batched_nms = torchvision_boxes.batched_nms
    if getattr(original_batched_nms, "_seganything_dtype_safe", False):
        return

    def dtype_safe_batched_nms(boxes, scores, idxs, iou_threshold):
        if torch.is_tensor(boxes) and torch.is_tensor(scores) and boxes.dtype != scores.dtype:
            boxes = boxes.to(dtype=torch.float32)
            scores = scores.to(dtype=torch.float32)
        return original_batched_nms(boxes, scores, idxs, iou_threshold)

    dtype_safe_batched_nms._seganything_dtype_safe = True
    torchvision_boxes.batched_nms = dtype_safe_batched_nms
    torchvision_ops.batched_nms = dtype_safe_batched_nms

    try:
        import transformers.models.sam2.image_processing_sam2 as sam2_image_processing
        sam2_image_processing.batched_nms = dtype_safe_batched_nms
    except Exception:
        pass

def call_mask_generator(generator, pil_image: Image.Image, generate_kwargs: dict) -> dict:
    """兼容不同 transformers 版本的 mask-generation 参数命名。"""
    attempts = [generate_kwargs]

    alt_kwargs = dict(generate_kwargs)
    if "crops_n_layers" in alt_kwargs:
        alt_kwargs["crop_n_layers"] = alt_kwargs.pop("crops_n_layers")
    attempts.append(alt_kwargs)

    alt_kwargs = dict(generate_kwargs)
    if "crop_n_points_downscale_factor" in alt_kwargs:
        alt_kwargs["crops_n_points_downscale_factor"] = alt_kwargs.pop("crop_n_points_downscale_factor")
    attempts.append(alt_kwargs)

    minimal_kwargs = {
        "points_per_batch": generate_kwargs["points_per_batch"],
        "points_per_crop": generate_kwargs["points_per_crop"],
        "pred_iou_thresh": generate_kwargs["pred_iou_thresh"],
        "stability_score_thresh": generate_kwargs["stability_score_thresh"],
    }
    attempts.append(minimal_kwargs)

    last_error = None
    for kwargs in attempts:
        try:
            return generator(pil_image, **kwargs)
        except TypeError as exc:
            last_error = exc
            message = str(exc)
            if "unexpected keyword" not in message and "got an unexpected" not in message:
                raise

    raise TypeError(f"当前 transformers mask-generation 不接受这些参数: {generate_kwargs}") from last_error


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------


def draw_overlay(
    image_rgb: np.ndarray,
    instances: List[dict],
    out_path: Path,
    alpha: float,
    draw_mask: bool,
    draw_index: bool,
) -> None:
    canvas = image_rgb.copy()
    height, width = canvas.shape[:2]

    order = sorted(range(len(instances)), key=lambda i: instances[i]["area"], reverse=True)

    for inst_idx in order:
        inst = instances[inst_idx]
        color = stable_color(inst_idx)
        mask = inst["mask"].astype(bool)
        if mask.shape[:2] != canvas.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)

        if draw_mask and mask.any():
            colored = np.zeros_like(canvas)
            colored[:] = color
            canvas[mask] = (alpha * colored[mask] + (1.0 - alpha) * canvas[mask]).astype(np.uint8)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color.tolist(), 1)

        x1, y1, x2, y2 = inst["bbox_xyxy"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color.tolist(), 1)
        if draw_index:
            label = f"{inst_idx}:{inst['score']:.2f}"
            text_y = max(12, y1 - 4)
            cv2.putText(canvas, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color.tolist(), 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# 主流程：SAM2 自动 mask generation
# ---------------------------------------------------------------------------


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_image_processor(model_path: str):
    errors = []
    processor = None

    if AutoImageProcessor is not None:
        try:
            processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        except Exception as exc:
            errors.append(f"AutoImageProcessor: {exc}")

    if processor is None and Sam2ImageProcessor is not None:
        try:
            processor = Sam2ImageProcessor.from_pretrained(model_path, local_files_only=True)
        except Exception as exc:
            errors.append(f"Sam2ImageProcessor: {exc}")

    if processor is None:
        detail = " | ".join(errors) if errors else "当前 transformers 不支持 AutoImageProcessor/Sam2ImageProcessor"
        raise RuntimeError(f"无法加载 SAM2 图像 processor: {detail}")

    if not hasattr(processor, "size"):
        raise TypeError(
            f"加载到的 processor={type(processor).__name__} 不适用于 mask-generation 图像 pipeline，缺少 size 属性"
        )

    return processor


def build_mask_generator(model_dir: Path, device: int, dtype_name: str):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    torch_dtype = resolve_torch_dtype(dtype_name)
    model_path = str(model_dir)

    if AutoModelForMaskGeneration is not None:
        try:
            print(f"[2/4] loading image processor from local files...", flush=True)
            processor = load_image_processor(model_path)
            print(f"[2/4] image processor = {type(processor).__name__}, size={processor.size}", flush=True)

            print(f"[2/4] loading model from local files, dtype={dtype_name}...", flush=True)
            t0 = time.time()
            model = AutoModelForMaskGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            print(f"[2/4] model loaded on CPU ({time.time() - t0:.1f}s)", flush=True)

            if int(device) >= 0:
                print(f"[2/4] moving model to cuda:{device}...", flush=True)
                t0 = time.time()
                model = model.to(f"cuda:{device}")
                print(f"[2/4] model moved to cuda:{device} ({time.time() - t0:.1f}s)", flush=True)
            else:
                print("[2/4] using CPU device", flush=True)

            model.eval()
            print("[2/4] building mask-generation pipeline from loaded model...", flush=True)
            pipeline_device = -1 if int(device) >= 0 else int(device)
            return pipeline("mask-generation", model=model, image_processor=processor, device=pipeline_device)
        except Exception as exc:
            print(f"[2/4][WARN] explicit AutoModel loading failed: {exc}", flush=True)
            print("[2/4][WARN] fallback to direct pipeline loading...", flush=True)

    print("[2/4] loading image processor for fallback pipeline...", flush=True)
    processor = load_image_processor(model_path)
    print(f"[2/4] fallback image processor = {type(processor).__name__}, size={processor.size}", flush=True)
    model_kwargs = {"torch_dtype": torch_dtype, "low_cpu_mem_usage": True, "local_files_only": True}
    return pipeline(
        "mask-generation",
        model=model_path,
        image_processor=processor,
        device=int(device),
        model_kwargs=model_kwargs,
    )


def run_segment_anything(args: argparse.Namespace, image_path: Path, model_dir: Path) -> List[dict]:
    print(f"[1/4] image    = {image_path}", flush=True)
    print(f"[1/4] model    = {model_dir}", flush=True)
    print(f"[1/4] device   = {args.device}", flush=True)
    print(f"[1/4] dtype    = {args.dtype}", flush=True)

    t0 = time.time()
    generator = build_mask_generator(model_dir, int(args.device), args.dtype)
    print(f"[2/4] pipeline built ({time.time() - t0:.1f}s)", flush=True)

    pil_image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_image)
    height, width = image_rgb.shape[:2]

    points_per_crop = max(1, int(args.points_per_side))
    generate_kwargs = {
        "points_per_batch": int(args.points_per_batch),
        "points_per_crop": points_per_crop,
        "pred_iou_thresh": float(args.pred_iou_thresh),
        "stability_score_thresh": float(args.stability_score_thresh),
        "crops_n_layers": int(args.crop_n_layers),
        "crop_n_points_downscale_factor": int(args.crop_n_points_downscale_factor),
    }

    print(f"[3/4] generating masks, size=({height},{width}), kwargs={generate_kwargs}", flush=True)
    patch_nms_dtype_mismatch()
    t0 = time.time()
    with torch.inference_mode():
        outputs = call_mask_generator(generator, pil_image, generate_kwargs)
    print(f"[3/4] mask generation done ({time.time() - t0:.1f}s)", flush=True)

    masks = extract_list(outputs, "masks")
    scores = extract_list(outputs, "scores", "iou_scores", "predicted_iou")
    boxes = extract_list(outputs, "boxes", "bboxes", "bounding_boxes")

    instances: List[dict] = []
    for idx, mask_obj in enumerate(masks):
        mask = to_numpy_mask(mask_obj, height, width)
        area = int(mask.sum())
        if area < int(args.min_mask_region_area):
            continue

        score = float(scores[idx]) if idx < len(scores) else 1.0
        box = boxes[idx] if idx < len(boxes) else None
        bbox_xyxy = normalize_box_xyxy(box, mask)
        x1, y1, x2, y2 = bbox_xyxy
        if x2 <= x1 or y2 <= y1:
            continue

        instances.append({
            "mask": mask,
            "score": score,
            "area": area,
            "bbox_xyxy": bbox_xyxy,
            "source": "transformers_sam2_mask_generation",
        })

    instances.sort(key=lambda x: x["score"], reverse=True)
    if len(instances) > int(args.max_num_masks):
        instances = instances[: int(args.max_num_masks)]

    print(f"[4/4] masks kept = {len(instances)} / raw = {len(masks)}")
    return instances


# ---------------------------------------------------------------------------
# 保存
# ---------------------------------------------------------------------------


def save_per_mask(instances: List[dict], out_dir: Path) -> None:
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for idx, inst in enumerate(instances):
        cv2.imwrite(
            str(masks_dir / f"{idx:03d}_score_{inst['score']:.3f}.png"),
            inst["mask"].astype(np.uint8) * 255,
        )


def dump_json(instances: List[dict], args: argparse.Namespace, image_path: Path, model_dir: Path, out_path: Path) -> None:
    serializable = []
    for idx, inst in enumerate(instances):
        serializable.append({
            "index": idx,
            "score": inst["score"],
            "area": inst["area"],
            "bbox_xyxy": inst["bbox_xyxy"],
            "source": inst.get("source", "transformers_sam2_mask_generation"),
        })

    payload = {
        "image": str(image_path),
        "model": str(model_dir),
        "device": args.device,
        "points_per_side": args.points_per_side,
        "points_per_crop": max(1, int(args.points_per_side)),
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
        "max_num_masks": args.max_num_masks,
        "num_instances": len(serializable),
        "instances": serializable,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    image_path, model_dir = ensure_inputs(args)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = run_segment_anything(args, image_path, model_dir)
    image_rgb = np.array(Image.open(image_path).convert("RGB"))

    stem = image_path.stem
    overlay_path = out_dir / f"{stem}_seganything_overlay.png"
    json_path = out_dir / f"{stem}_seganything_results.json"

    draw_overlay(
        image_rgb=image_rgb,
        instances=instances,
        out_path=overlay_path,
        alpha=args.alpha,
        draw_mask=not args.no_mask,
        draw_index=args.draw_index,
    )
    if args.save_per_mask:
        save_per_mask(instances, out_dir)
    dump_json(instances, args, image_path, model_dir, json_path)

    print(f"[DONE] num_instances = {len(instances)}")
    print(f"[DONE] overlay       = {overlay_path}")
    print(f"[DONE] json          = {json_path}")
    if args.save_per_mask:
        print(f"[DONE] per-mask dir  = {out_dir / 'masks'}")


if __name__ == "__main__":
    main()
