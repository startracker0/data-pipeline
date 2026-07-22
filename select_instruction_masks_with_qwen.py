#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用本地 Qwen-VL 判断 SAM2 自动分割出的 mask 中哪些是 instruction 描述的物体。"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib import error, request

import numpy as np
from PIL import Image, ImageDraw

DEFAULT_EPISODES_JSONL = Path(
    "/apdcephfs_gy7/share_305004851/hunyuan/yinanliang/wam/fastwam/data/robotwin2.0/meta/episodes.jsonl"
)
DEFAULT_API_BASE = "http://127.0.0.1:8000/v1"

FOCUS_SYSTEM_PROMPT = """你是机器人视觉数据集标注助手。你只根据用户提供的英文 instructions，归纳该 episode 中需要视觉定位/检测/分割的关键物体。
必须只输出严格 JSON object，不要输出 Markdown，不要输出解释。
JSON 必须使用英文双引号，不允许尾逗号，不允许注释。
"""

FOCUS_USER_PROMPT_TEMPLATE = """请阅读下面同一个 episode 的英文 instructions，找出这个 episode instructions 中所有需要在图像中识别/分割的任务相关实体物体。

要求：
1. 合并同一个物体的不同描述，不要重复输出同一个物体；final_answers 的数量必须等于 instruction 语义上出现的真实物理实体物体数量。下面的多条 instructions 是同一个 episode 的不同 paraphrase/改写，不是多个独立场景。
2. 先按动作角色和空间关系归并真实物体：同一个被抓取/移动/摆放的物体只能输出一次；同一个被放在旁边/里面/左边/右边的参照物也只能输出一次。
3. 如果多条 instructions 只是同一个任务的不同说法，例如 teal microphone、white microphone、compact teal and white microphone 都指同一个 microphone，则只能输出一个 object。
4. 不要因为同一真实物体在不同 instructions 中出现了不同外观短语就拆成多个 object。例如同一批 instructions 都是在把 can 放到 kitchenpot 边上，那么 red sauce can、red can with gold patterns、red and gold sauce can、sauce can with white details、smooth sauce can 都应视为同一个 sauce can，除非同一条 instruction 明确同时要求区分两个不同的 can。
5. 必须输出 instructions 明确提到的所有任务相关实体物体，包括被操作物体、被放置目标、容器、支撑面/垫子、关键参照物；不要只输出被操作物体。但参照物也必须是可见实体物体本身。
6. 不要输出动作、方位、关系短语、颜色词、形容词、机械臂、桌子、背景、gripper、left/right arm、robot。严禁把 yellow、red、right、left、right side、right of the red、right of the box、right side of the small pack 这类颜色/方位/关系短语当作 object。
7. 如果 instruction 写的是 red playingcards box、yellow and red playingcards box、right of the red playingcards box，object 必须是 playingcards box / box，不能输出 red、yellow、right of the red。
8. prompt 使用简短英文名词短语，优先保留稳定、最高频、最适合视觉定位的颜色/形状 + 类别，例如 green bottle、purple mat。禁止把跨 instruction 的属性做并集拼接，不要输出 red and white sauce can、grey and red and black kitchenpot 这类 prompt；应输出 red sauce can、sauce can、kitchenpot 或 white cylindrical pot 这种稳定短语。
9. 如果同一 episode 中存在多个同类物体，必须保留 instructions 中反复出现、能唯一定位实例的稳定属性，例如 brown cap、brown top、lidded、label、plastic lid、round top；不要只输出过宽泛的 white sauce can。
10. 对 kitchenpot / pot 这类在仿真画面里常呈现为白色或浅灰色圆柱锅体的物体，不要强行保留 instructions 中互相冲突的 grey、black base、dark top、red indicator light 等局部描述；优先使用 kitchenpot 或 white cylindrical pot。
11. 如果 instruction 同时描述 white fan 和 purple mat，必须输出两个 objects：white fan 与 purple mat。
12. 严禁输出 aliases、attributes、reason、explanation 等长字段；每个 object 只允许 prompt、base_object、role 三个字段。

输出 JSON schema：
{
  "objects": [
    {
      "prompt": "green bottle",
      "base_object": "bottle",
      "role": "manipulated"
    },
    {
      "prompt": "purple mat",
      "base_object": "mat",
      "role": "placement"
    }
  ],
  "prompts": ["green bottle", "purple mat"]
}

只输出 JSON 本身，最后一个字符必须是 }。不要输出超过 4 个 objects，除非 instructions 明确同时出现超过 4 个不同物理实体物体。

episode: {episode}
instructions:
{instructions}
"""

JUDGE_SYSTEM_PROMPT = """你是机器人视觉分割 mask 审核助手。用户会给你：
1. episode instructions 与目标物体列表；
2. 一张候选 mask 图，图中只有 mask 内区域保留原图真实颜色，mask 外区域被置为黑色，黄色框表示当前候选 mask 的 bbox。

你的任务是判断保留真实颜色的可见区域是否是 instruction 描述的任意关键物体之一。Instruction target objects JSON 中的 manipulated 物体和 reference 参照物都需要匹配；role 只是说明物体在任务里的作用，不能作为排除理由。
必须只输出严格 JSON object，不要输出 Markdown，不要解释。
"""

JUDGE_USER_TEXT_TEMPLATE = """Episode instructions:
{instructions}

Instruction target objects JSON:
{focus_json}

Candidate mask metadata:
{mask_json}

图像说明：当前候选图是原图乘以二值 mask 的结果：mask 内区域保留原图真实颜色，mask 外区域为黑色，黄色框是该 mask 的 bbox。请只根据 mask 内可见物体的真实颜色、形状和上下文判断它是否对应目标物体之一；不要把黄色框当成物体颜色。

输出 JSON schema：
{
  "mask_index": 0,
  "is_target": true,
  "matched_prompt": "green bottle",
  "object_name": "bottle",
  "confidence": 0.0,
  "reason": "short reason"
}

规则：
- 如果可见 mask 区域是背景、桌面、墙、机械臂、夹爪、阴影、画面边界，is_target=false。
- Instruction target objects JSON 中 listed 的每个 prompt 都是要找的物体；即使 role 是 reference，只要 mask 是该参照物，也必须 is_target=true，并填写对应 matched_prompt。
- 不要因为某个物体“不是 sauce can / 不是 manipulated target”而判 false；如果它匹配另一个 prompt，例如 teal kitchenpot，就应该 matched_prompt="teal kitchenpot"。
- 如果可见 mask 区域只覆盖目标物体的一小部分但足以确认目标，is_target 可以为 true，但 confidence 应降低。
- 对仿真画面中的颜色要允许轻微偏差：white、light grey、grey、off-white 对同一个 pot/kitchenpot 类目标应视为兼容；判断 kitchenpot/pot 时更看重圆柱锅体、盖子、把手、容器形状，不要仅因为 prompt 写 grey 但画面偏 white 就判 false。
- 如果画面中有多个同类候选物体，必须使用 episode instructions 中反复出现的稳定实例属性消歧，例如 brown cap、brown top、brown lid、lidded、label、plastic lid、round top、handle；只有候选 mask 内确实可见这些属性时，才给高 confidence。
- 对 sauce can / can / bottle / box 等容易出现多个实例的类别，不能只因为类别大致匹配就判高 confidence；如果缺少 prompt 或 instructions 中的关键实例属性，应降低 confidence，必要时判 false。
- matched_prompt 必须来自 Instruction target objects JSON 的 prompts；如果不匹配任何目标，matched_prompt 使用空字符串。
- confidence 范围是 0 到 1。
- 只输出 JSON 本身。
"""

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

BASE_OBJECT_ALIASES = {
    "kitchen pot": "kitchenpot",
    "saucecan": "sauce can",
}

CONTEXTUAL_BASE_OBJECT_ALIASES = {
    "can": "sauce can",
    "pot": "kitchenpot",
}

LOW_CONFIDENCE_DESCRIPTOR_PHRASES = {
    "black base",
    "black button",
    "black rectangular panel",
    "dark circle inside",
    "dark top",
    "flat bottom",
    "flat top",
    "gold patterns",
    "red and gold",
    "red and white",
    "red indicator light",
    "white details",
}

INSTANCE_DESCRIPTOR_PHRASES = {
    "brown cap",
    "brown lid",
    "brown lidded",
    "brown round top",
    "brown top",
    "label",
    "labeled",
    "labelled",
    "plastic lid",
    "round top",
    "white label",
}

STRONG_INSTANCE_DESCRIPTOR_PHRASES = {
    "brown cap",
    "brown lid",
    "brown lidded",
    "brown round top",
    "brown top",
    "plastic lid",
}

DESCRIPTOR_PHRASE_WEIGHTS = {phrase: 1 for phrase in INSTANCE_DESCRIPTOR_PHRASES}
DESCRIPTOR_PHRASE_WEIGHTS.update({phrase: 5 for phrase in STRONG_INSTANCE_DESCRIPTOR_PHRASES})

REFERENCE_ROLES = {"placement", "reference", "support", "container", "target", "destination"}
MANIPULATED_ROLES = {"", "manipulated", "object", "item"}
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


def contains_word(text: str, word: str) -> bool:
    text = normalize_text(text)
    word = normalize_text(word)
    if not text or not word:
        return False
    pattern = re.escape(word).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<![a-z]){pattern}(?![a-z])", text))


def normalize_base_object(base_object: Any, prompt: str = "") -> str:
    base = normalize_text(base_object)
    prompt = normalize_text(prompt)
    if base in BASE_OBJECT_ALIASES:
        return BASE_OBJECT_ALIASES[base]
    if base == "" and contains_word(prompt, "sauce can"):
        return "sauce can"
    if base.endswith(" can") and contains_word(prompt, "sauce can"):
        return "sauce can"
    return base


def contextual_base_object(base_object: str, prompt: str, available_bases: set[str]) -> str:
    base = normalize_base_object(base_object, prompt)
    contextual = CONTEXTUAL_BASE_OBJECT_ALIASES.get(base)
    if contextual and contextual in available_bases:
        return contextual
    return base


def base_object_of(obj: dict[str, Any]) -> str:
    prompt = normalize_text(obj.get("prompt") or "")
    base = normalize_base_object(obj.get("base_object") or "", prompt)
    if base:
        return base
    words = prompt.split()
    return words[-1] if words else ""


def is_valid_focus_object(obj: dict[str, Any]) -> bool:
    prompt = normalize_text(obj.get("prompt") or "")
    base_object = base_object_of(obj)
    if not prompt or not base_object:
        return False
    if base_object in NON_OBJECT_BASES:
        return False
    prompt_words = prompt.split()
    if len(prompt_words) == 1 and prompt in NON_OBJECT_BASES:
        return False
    if any(prompt.startswith(prefix) for prefix in RELATION_PHRASE_PREFIXES):
        return False
    if prompt_words and all(word in NON_OBJECT_BASES for word in prompt_words):
        return False
    return True


def role_family_of(obj: dict[str, Any]) -> str:
    role = normalize_text(obj.get("role") or "")
    if role in REFERENCE_ROLES:
        return "reference"
    if role in MANIPULATED_ROLES:
        return "manipulated"
    return role or "manipulated"

def object_mention_terms(obj: dict[str, Any]) -> list[str]:
    terms = [normalize_text(obj.get("prompt") or "")]
    aliases = obj.get("aliases") if isinstance(obj.get("aliases"), list) else []
    terms.extend(normalize_text(item) for item in aliases)
    return sorted(dedupe_keep_order([term for term in terms if term]), key=len, reverse=True)

def instruction_mentions_object(instruction: str, obj: dict[str, Any]) -> bool:
    text = normalize_text(instruction)
    return any(term and term in text for term in object_mention_terms(obj))

def instruction_mentions_multiple_same_base_objects(instruction: str, objects: list[dict[str, Any]], base_object: str) -> bool:
    text = normalize_text(instruction)
    mentioned = [obj for obj in objects if instruction_mentions_object(text, obj)]
    if len(mentioned) < 2:
        return False
    return len(re.findall(rf"\b{re.escape(base_object)}s?\b", text)) >= 2

def should_merge_same_base_objects(objects: list[dict[str, Any]], instructions: list[str], base_object: str) -> bool:
    if len(objects) <= 1:
        return False
    for instruction in instructions:
        if instruction_mentions_multiple_same_base_objects(instruction, objects, base_object):
            return False
    return True

def prompt_low_confidence_hits(prompt: str) -> int:
    return sum(1 for phrase in LOW_CONFIDENCE_DESCRIPTOR_PHRASES if contains_word(prompt, phrase))


def stable_prompt_score(prompt: str, base_object: str) -> tuple[int, int, int]:
    prompt = normalize_text(prompt)
    base_object = normalize_base_object(base_object, prompt)
    if not prompt:
        return (-100, 0, 0)
    if base_object and not contains_word(prompt, base_object):
        if not (base_object == "sauce can" and contains_word(prompt, "can")):
            return (-50, 0, 0)

    words = prompt.split()
    color_hits = sum(1 for color in COLOR_WORDS if contains_word(prompt, color))
    low_hits = prompt_low_confidence_hits(prompt)
    score = 0
    score += 30 if base_object and contains_word(prompt, base_object) else 0
    score += 10 if base_object == "sauce can" and prompt == "red sauce can" else 0
    score += 6 if color_hits == 1 else 0
    score += 6 if 2 <= len(words) <= 3 else 0
    score -= 10 if "and" in words else 0
    score -= 8 * max(0, color_hits - 1)
    score -= 12 * low_hits
    score -= max(0, len(words) - 5) * 2
    return (score, -abs(len(words) - 3), -len(prompt))


def select_stable_prompt(candidates: list[str], base_object: str) -> str:
    candidates = dedupe_keep_order([normalize_text(item) for item in candidates if normalize_text(item)])
    base_object = normalize_base_object(base_object, candidates[0] if candidates else "")
    if not candidates:
        return base_object
    if base_object == "sauce can" and "red sauce can" in candidates:
        return "red sauce can"
    if base_object == "kitchenpot":
        for stable_prompt in ("white cylindrical pot", "white pot", "kitchenpot"):
            if stable_prompt in candidates:
                return stable_prompt
        return "white cylindrical pot"
    selected = max(candidates, key=lambda item: stable_prompt_score(item, base_object))
    if base_object == "sauce can" and contains_word(selected, "red") and prompt_low_confidence_hits(selected):
        return "red sauce can"
    if base_object == "kitchenpot" and ("and" in selected.split() or prompt_low_confidence_hits(selected)):
        return base_object
    return selected


def merge_object_group(objects: list[dict[str, Any]], base_object: str) -> dict[str, Any]:
    prompts = [normalize_text(obj.get("prompt") or "") for obj in objects]
    attrs: list[str] = []
    aliases: list[str] = []
    roles: list[str] = []
    for obj in objects:
        attrs.extend(str(item) for item in (obj.get("attributes") if isinstance(obj.get("attributes"), list) else []))
        aliases.extend(str(item) for item in (obj.get("aliases") if isinstance(obj.get("aliases"), list) else []))
        aliases.append(str(obj.get("prompt") or ""))
        roles.append(normalize_text(obj.get("role") or ""))
    attrs = dedupe_keep_order(attrs)
    aliases = dedupe_keep_order(aliases)
    canonical_prompt = select_stable_prompt(prompts + aliases, base_object)
    role = next((role for role in roles if role), "manipulated")
    return {
        "prompt": canonical_prompt,
        "base_object": base_object,
        "attributes": attrs[:8],
        "aliases": [alias for alias in aliases if normalize_text(alias) != canonical_prompt][:6],
        "role": role,
    }

def merge_same_entity_focus_objects(focus: dict[str, Any], instructions: list[str]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    input_objects = [obj for obj in (focus.get("objects") or []) if isinstance(obj, dict) and is_valid_focus_object(obj)]
    available_bases = {base_object_of(obj) for obj in input_objects}
    for obj in input_objects:
        base_object = contextual_base_object(obj.get("base_object") or "", normalize_text(obj.get("prompt") or ""), available_bases)
        if not base_object:
            passthrough.append(obj)
            continue
        grouped.setdefault((role_family_of(obj), base_object), []).append(obj)

    merged_objects: list[dict[str, Any]] = []
    for (_role_family, base_object), group in grouped.items():
        if len(group) == 1:
            merged_objects.append(merge_object_group(group, base_object))
        elif should_merge_same_base_objects(group, instructions, base_object):
            merged_objects.append(merge_object_group(group, base_object))
        else:
            merged_objects.extend(group)
    merged_objects.extend(passthrough)

    prompt_seen: set[str] = set()
    prompts: list[str] = []
    deduped_objects: list[dict[str, Any]] = []
    for obj in merged_objects:
        if not is_valid_focus_object(obj):
            continue
        prompt = normalize_text(obj.get("prompt") or "")
        if not prompt or prompt in prompt_seen:
            continue
        prompt_seen.add(prompt)
        prompts.append(prompt)
        deduped_objects.append(obj)
    return {"objects": deduped_objects, "prompts": prompts}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="让本地 Qwen-VL 判断 seganything 每个 mask 是否是 episode instruction 目标物体。")
    parser.add_argument("--episodes-jsonl", type=Path, default=Path(os.environ.get("EPISODES_JSONL", DEFAULT_EPISODES_JSONL)))
    parser.add_argument("--episode", required=True, help="episode id，例如 7000、007000 或 episode_007000")
    parser.add_argument("--seg-results", type=Path, required=True, help="seganything 输出的 *_seganything_results.json")
    parser.add_argument("--masks-dir", type=Path, required=True, help="seganything 输出的 per-mask PNG 目录")
    parser.add_argument("--output", type=Path, default=None, help="输出 Qwen 判断结果 JSON；默认写到 seg-results 同目录")
    parser.add_argument("--candidates-dir", type=Path, default=None, help="保存发给 Qwen 的候选高亮图目录")
    parser.add_argument("--api-base", default=os.environ.get("QWEN_API_BASE", DEFAULT_API_BASE))
    parser.add_argument("--model", default=os.environ.get("QWEN_MODEL", ""))
    parser.add_argument("--temperature", type=float, default=float(os.environ.get("QWEN_TEMPERATURE", "0")))
    parser.add_argument("--max-tokens", type=int, default=int(os.environ.get("QWEN_MAX_TOKENS", "512")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("QWEN_TIMEOUT", "120")))
    parser.add_argument("--max-instructions", type=int, default=200)
    parser.add_argument("--max-masks", type=int, default=0, help="最多判断多少个 mask；0 表示全部")
    parser.add_argument("--min-mask-area", type=int, default=0, help="跳过面积小于该阈值的 mask；0 表示不额外过滤")
    parser.add_argument("--crop-padding-ratio", type=float, default=0.18, help="候选图按 bbox 裁剪时的上下文 padding 比例")
    parser.add_argument("--max-image-side", type=int, default=768, help="发给 Qwen 的候选图最长边")
    parser.add_argument("--include-full-frame", action="store_true", help="候选图使用整帧而不是 bbox crop")
    parser.add_argument("--allow-fail", action="store_true", help="单个 mask 判断失败时记录错误并继续")
    return parser.parse_args()


def normalize_api_base(api_base: str) -> str:
    api_base = api_base.strip().rstrip("/")
    if api_base.endswith("/models"):
        api_base = api_base[: -len("/models")]
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    return api_base


def request_json(url: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is None:
        req = request.Request(url, headers=headers, method="GET")
    else:
        req = request.Request(url, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model(api_base: str, explicit_model: str, timeout: float) -> str:
    if explicit_model:
        return explicit_model
    data = request_json(f"{api_base}/models", None, timeout)
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"{api_base}/models 没有返回可用模型")
    model_id = models[0].get("id")
    if not model_id:
        raise RuntimeError(f"无法从 /models 返回中读取模型 id: {data}")
    return str(model_id)


def call_chat_completion(
    api_base: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
    use_json_response_format: bool = True,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_json_response_format:
        payload["response_format"] = {"type": "json_object"}
    try:
        data = request_json(f"{api_base}/chat/completions", payload, timeout)
    except error.HTTPError:
        if not use_json_response_format:
            raise
        payload.pop("response_format", None)
        data = request_json(f"{api_base}/chat/completions", payload, timeout)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"chat/completions 没有 choices: {data}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"chat/completions 返回空 content: {data}")
    return content.strip()


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def iter_json_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    in_string = False
    escape = False
    depth = 0
    start: int | None = None
    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : idx + 1])
                    start = None
    return candidates


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fence(text)
    candidates = [cleaned] + iter_json_object_candidates(cleaned)
    errors: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        for maybe_fixed in (candidate, remove_trailing_commas(candidate)):
            try:
                data = json.loads(maybe_fixed)
            except json.JSONDecodeError as exc:
                errors.append(str(exc))
                continue
            if not isinstance(data, dict):
                raise ValueError("模型输出 JSON 顶层必须是 object")
            return data
    snippet = cleaned[:800].replace("\n", "\\n")
    raise ValueError(f"无法从模型输出解析 JSON；errors={errors[:3]}；raw_prefix={snippet}")


def extract_json_string_field(text: str, field: str) -> str:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"]*)"', text, flags=re.DOTALL)
    return normalize_text(match.group(1)) if match else ""


def extract_json_string_array_field(text: str, field: str, limit: int) -> list[str]:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    if not match:
        return []
    values = re.findall(r'"([^"]+)"', match.group(1))
    return dedupe_keep_order(values)[:limit]


def extract_focus_objects_from_loose_text(text: str) -> dict[str, Any]:
    cleaned = strip_markdown_fence(text)
    objects: list[dict[str, Any]] = []
    object_chunks = re.split(r'\n\s*\{\s*\n\s*"prompt"\s*:', cleaned)
    for idx, chunk in enumerate(object_chunks):
        if idx == 0:
            if not re.search(r'"prompt"\s*:', chunk):
                continue
            object_text = chunk
        else:
            object_text = '"prompt":' + chunk
        prompt = extract_json_string_field(object_text, "prompt")
        if not prompt:
            continue
        base_object = extract_json_string_field(object_text, "base_object")
        if not base_object:
            words = prompt.split()
            base_object = words[-1] if words else ""
        role = extract_json_string_field(object_text, "role") or "manipulated"
        attrs = extract_json_string_array_field(object_text, "attributes", 8)
        aliases = extract_json_string_array_field(object_text, "aliases", 6)
        objects.append(
            {
                "prompt": prompt,
                "base_object": base_object,
                "attributes": attrs,
                "aliases": aliases,
                "role": role,
            }
        )

    if not objects:
        prompts_match = re.search(r'"prompts"\s*:\s*\[(.*?)\]', cleaned, flags=re.DOTALL)
        if prompts_match:
            for prompt in dedupe_keep_order(re.findall(r'"([^"]+)"', prompts_match.group(1))):
                words = prompt.split()
                objects.append(
                    {
                        "prompt": prompt,
                        "base_object": words[-1] if words else "",
                        "attributes": [],
                        "aliases": [],
                        "role": "manipulated",
                    }
                )

    prompts = dedupe_keep_order([str(obj.get("prompt") or "") for obj in objects])
    if not prompts:
        raise ValueError("无法从坏 JSON 文本中兜底提取 focus objects")
    return {"objects": objects, "prompts": prompts}


def parse_episode_number(value: str) -> int | None:
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    match = re.search(r"episode[_-]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def item_matches_episode(item: dict[str, Any], episode_text: str) -> bool:
    wanted_number = parse_episode_number(episode_text)
    wanted_text = str(episode_text).strip().lower()

    for key in ("episode_index", "episode_idx", "index", "episode"):
        if key in item:
            value = item.get(key)
            try:
                if wanted_number is not None and int(value) == wanted_number:
                    return True
            except (TypeError, ValueError):
                pass
            if str(value).strip().lower() == wanted_text:
                return True

    if wanted_number is not None:
        canonical = f"episode_{wanted_number:06d}"
        for value in item.values():
            if isinstance(value, str) and canonical in value.lower():
                return True
    return False


def load_episode_item(episodes_jsonl: Path, episode: str) -> dict[str, Any]:
    with episodes_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item_matches_episode(item, episode):
                return item
    raise FileNotFoundError(f"没有在 {episodes_jsonl} 中找到 episode={episode}")


def extract_instructions(item: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("tasks", "instructions", "instruction", "language_instruction", "natural_language_instruction", "task"):
        if key in item:
            value = item.get(key)
            if isinstance(value, list):
                values.extend(value)
            else:
                values.append(value)

    instructions: list[str] = []
    for value in values:
        if isinstance(value, str):
            text = value.strip()
            if text:
                instructions.append(text)
        elif isinstance(value, dict):
            for key in ("instruction", "language_instruction", "task", "description"):
                text = str(value.get(key, "")).strip()
                if text:
                    instructions.append(text)
                    break
    if not instructions:
        raise ValueError(f"episode 记录中没有找到 instruction/tasks 字段: keys={sorted(item.keys())}")
    return instructions


def normalize_text(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.;:\"'`[]{}()")


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result

EXCLUDED_INSTRUCTION_OBJECTS = {
    "robot",
    "the robot",
    "left arm",
    "right arm",
    "arm",
    "arms",
    "hand",
    "hands",
    "left hand",
    "right hand",
    "other hand",
    "another hand",
    "opposite hand",
    "side",
    "sides",
    "other side",
    "another side",
    "opposite side",
    "someone",
    "someone else",
    "somebody",
    "person",
    "else",
    "gripper",
    "grippers",
    "table",
    "surface",
    "desk",
    "background",
    "wall",
    "floor",
}

NON_OBJECT_LEADING_WORDS = {
    "face",
    "faces",
    "facing",
    "point",
    "points",
    "pointing",
    "ensure",
    "ensuring",
    "make",
    "making",
    "verify",
    "confirm",
    "pick",
    "grab",
    "lift",
    "place",
    "put",
    "set",
    "drop",
    "position",
    "align",
    "turn",
    "use",
    "using",
}

PREPOSITIONAL_OBJECT_RE = re.compile(
    r"\b(?:next\s+to|on|onto|in|inside|into|within|near|beside|under|over|above|below)\s+(?:the\s+|a\s+|an\s+)?([^,.;]+)",
    flags=re.IGNORECASE,
)

OBJECT_CLAUSE_SPLIT_RE = re.compile(
    r"\b(?:and|then|so|while|after|before|making|ensuring|ensure|facing|faces|face|toward|towards|oriented|pointing|points|confirm|verify)\b",
    flags=re.IGNORECASE,
)

def canonicalize_instruction_object_phrase(value: str) -> str:
    text = normalize_text(value)
    text = OBJECT_CLAUSE_SPLIT_RE.split(text, maxsplit=1)[0]
    text = re.sub(r"^(?:the|a|an|it|its|this|that)\s+", "", text).strip()
    text = re.sub(r"\bcolou?red\b", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip(" ,.;:-")
    if not text:
        return ""
    first_word = text.split()[0]
    if first_word in NON_OBJECT_LEADING_WORDS:
        return ""
    if text in EXCLUDED_INSTRUCTION_OBJECTS:
        return ""
    if any(text == excluded or text.endswith(f" {excluded}") for excluded in EXCLUDED_INSTRUCTION_OBJECTS):
        return ""
    return text

def extract_instruction_reference_objects(instructions: list[str]) -> list[str]:
    candidates: list[str] = []
    for instruction in instructions:
        for match in PREPOSITIONAL_OBJECT_RE.finditer(instruction):
            prompt = canonicalize_instruction_object_phrase(match.group(1))
            if prompt:
                candidates.append(prompt)
    return dedupe_keep_order(candidates)

def prompt_already_covered(prompt: str, focus: dict[str, Any]) -> bool:
    prompt = normalize_text(prompt)
    if not prompt:
        return True
    for obj in focus.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        for term in focus_object_terms(obj):
            if prompt == term or prompt in term or term in prompt:
                return True
    for existing in focus.get("prompts") or []:
        term = normalize_text(existing)
        if prompt == term or prompt in term or term in prompt:
            return True
    return False

def merge_instruction_reference_objects(focus: dict[str, Any], instructions: list[str]) -> dict[str, Any]:
    objects = list(focus.get("objects") or [])
    prompts = list(focus.get("prompts") or [])
    merged = {"objects": objects, "prompts": prompts}
    for prompt in extract_instruction_reference_objects(instructions):
        if prompt_already_covered(prompt, merged):
            continue
        words = prompt.split()
        base_object = words[-1] if words else ""
        attributes = words[:-1]
        objects.append(
            {
                "prompt": prompt,
                "base_object": base_object,
                "attributes": dedupe_keep_order(attributes)[:6],
                "aliases": [],
                "role": "placement",
            }
        )
        prompts.append(prompt)
    merged["objects"] = objects
    merged["prompts"] = dedupe_keep_order(prompts)
    return merged


def normalize_focus_output(data: dict[str, Any]) -> dict[str, Any]:
    raw_objects = data.get("objects") if isinstance(data.get("objects"), list) else []
    objects: list[dict[str, Any]] = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        prompt = normalize_text(item.get("prompt") or item.get("description") or item.get("name") or "")
        if not prompt:
            continue
        attrs = item.get("attributes") if isinstance(item.get("attributes"), list) else []
        aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
        objects.append(
            {
                "prompt": prompt,
                "base_object": normalize_base_object(item.get("base_object") or "", prompt),
                "attributes": dedupe_keep_order([str(x) for x in attrs])[:6],
                "aliases": dedupe_keep_order([str(x) for x in aliases])[:3],
                "role": normalize_text(item.get("role") or ""),
            }
        )

    prompt_seen: set[str] = set()
    prompts: list[str] = []
    for obj in objects:
        prompt = normalize_text(obj.get("prompt") or "")
        if prompt and prompt not in prompt_seen:
            prompt_seen.add(prompt)
            prompts.append(prompt)
    raw_prompts = data.get("prompts") if isinstance(data.get("prompts"), list) else []
    for item in raw_prompts:
        prompt = normalize_text(item)
        if prompt and prompt not in prompt_seen:
            prompt_seen.add(prompt)
            prompts.append(prompt)
            objects.append({"prompt": prompt, "base_object": "", "attributes": [], "aliases": [], "role": ""})
    if not prompts:
        raise ValueError(f"Qwen 没有输出可用目标物体: {data}")
    return {"objects": objects, "prompts": prompts}


def focus_object_terms(obj: dict[str, Any]) -> list[str]:
    prompt = normalize_text(obj.get("prompt") or "")
    base_object = normalize_text(obj.get("base_object") or "")
    terms = [prompt, base_object]
    if base_object == "kitchenpot" or prompt in {"white cylindrical pot", "white pot", "kitchenpot"}:
        terms.extend(["pot", "white pot", "white cylindrical pot", "cylindrical pot", "kitchenpot"])
    aliases = obj.get("aliases") if isinstance(obj.get("aliases"), list) else []
    terms.extend(normalize_text(item) for item in aliases)
    return dedupe_keep_order([term for term in terms if term])


def infer_prompt_from_object_name(object_name: str, focus: dict[str, Any]) -> str:
    object_text = normalize_text(object_name)
    if not object_text:
        return ""
    for obj in focus.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        prompt = normalize_text(obj.get("prompt") or "")
        if not prompt:
            continue
        for term in focus_object_terms(obj):
            if object_text == term or object_text in term or term in object_text:
                return prompt
        base_object = normalize_text(obj.get("base_object") or "")
        if base_object == "kitchenpot" and (
            "pot" in object_text
            or ("cylindrical" in object_text and ("white" in object_text or "grey" in object_text or "gray" in object_text))
        ):
            return prompt
    return ""


def infer_kitchenpot_prompt_from_text(text: str, focus: dict[str, Any]) -> str:
    text = normalize_text(text)
    if not text:
        return ""
    negative_cues = (
        "not matching",
        "not identifiable",
        "not sufficient",
        "not any target",
        "not matching any target",
        "not matching white",
        "not the target",
        "not target",
        "background",
        "shadow",
        "too ambiguous",
    )
    if any(cue in text for cue in negative_cues):
        return ""
    looks_like_light_pot = (
        ("white" in text or "grey" in text or "gray" in text or "light" in text)
        and ("cylindrical" in text or "round" in text or "container" in text or "pot" in text)
    )
    if not looks_like_light_pot:
        return ""
    for obj in focus.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if normalize_text(obj.get("base_object") or "") == "kitchenpot":
            return normalize_text(obj.get("prompt") or "")
    return ""


def instruction_descriptor_hits(instructions: list[str], prompt: str, base_object: str) -> int:
    instruction_text = normalize_text("\n".join(instructions))
    prompt_text = normalize_text(prompt)
    base_text = normalize_text(base_object)
    score = 0
    for phrase in INSTANCE_DESCRIPTOR_PHRASES:
        if phrase in instruction_text and (phrase in prompt_text or phrase in base_text):
            score += 2
    return score


def instruction_descriptor_presence(instructions: list[str]) -> int:
    instruction_text = normalize_text("\n".join(instructions))
    return sum(1 for phrase in INSTANCE_DESCRIPTOR_PHRASES if phrase in instruction_text)


def remove_quoted_text(text: str) -> str:
    return re.sub(r"['\"`][^'\"`]*['\"`]", " ", text)


def judgment_descriptor_hits(item: dict[str, Any], instructions: list[str]) -> int:
    reason = remove_quoted_text(str(item.get("reason") or ""))
    text = normalize_text(" ".join([
        reason,
        str(item.get("object_name") or ""),
    ]))
    instruction_text = normalize_text("\n".join(instructions))
    score = 0
    for phrase in INSTANCE_DESCRIPTOR_PHRASES:
        if phrase in instruction_text and phrase in text:
            score += DESCRIPTOR_PHRASE_WEIGHTS.get(phrase, 1)
    speculative_cues = (
        "consistent with brown top under lighting",
        "appears light colored but consistent",
        "appears light-colored but consistent",
        "despite top not fully visible",
    )
    negative_top_cues = (
        "top not fully visible",
        "top not visible",
        "lid not fully visible",
        "lid not visible",
        "cap not fully visible",
        "cap not visible",
    )
    if any(cue in text for cue in speculative_cues):
        score = max(0, score - 5)
    if any(cue in text for cue in negative_top_cues):
        score = max(0, score - 5)
    return score


def prompt_strong_descriptor_presence(instructions: list[str], prompt: str) -> bool:
    instruction_text = normalize_text("\n".join(instructions))
    prompt_text = normalize_text(prompt)
    return any(phrase in instruction_text and phrase in prompt_text for phrase in STRONG_INSTANCE_DESCRIPTOR_PHRASES)


def focus_base_object_by_prompt(focus: dict[str, Any], prompt: str) -> str:
    normalized_prompt = normalize_text(prompt)
    for obj in focus.get("objects") or []:
        if not isinstance(obj, dict):
            continue
        if normalize_text(obj.get("prompt") or "") == normalized_prompt:
            return base_object_of(obj)
    return ""


def bbox_area(item: dict[str, Any]) -> float:
    bbox = item.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) < 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return max(0.0, (x2 - x1 + 1.0) * (y2 - y1 + 1.0))


def bbox_fill_ratio(item: dict[str, Any]) -> float:
    box_area = bbox_area(item)
    mask_area = float(item.get("area", 0.0) or 0.0)
    if box_area <= 0.0 or mask_area <= 0.0:
        return 0.0
    return max(0.0, min(1.0, mask_area / box_area))


def candidate_sort_key(item: dict[str, Any], prompt: str, focus: dict[str, Any], instructions: list[str]) -> tuple[float, ...]:
    base_object = focus_base_object_by_prompt(focus, prompt)
    ambiguous_base = base_object in {"sauce can", "can", "bottle", "box"}
    compact_base = base_object in {"kitchenpot", "pot", "pan", "bowl"}
    descriptor_score = judgment_descriptor_hits(item, instructions) if ambiguous_base else 0
    confidence = float(item.get("confidence", 0.0) or 0.0)
    sam_score = float(item.get("sam_score", 0.0) or 0.0)
    area = int(item.get("area", 0) or 0)
    fill_ratio = bbox_fill_ratio(item)
    prompt_descriptor_bonus = instruction_descriptor_hits(instructions, prompt, base_object)
    has_strong_prompt_descriptor = prompt_strong_descriptor_presence(instructions, prompt)
    if ambiguous_base:
        confidence += min(0.3, descriptor_score * 0.06)
    if ambiguous_base and (prompt_descriptor_bonus > 0 or instruction_descriptor_presence(instructions) > 0) and descriptor_score == 0:
        confidence -= 0.25 if has_strong_prompt_descriptor else 0.15
    if ambiguous_base and has_strong_prompt_descriptor:
        return (
            float(descriptor_score),
            confidence,
            fill_ratio,
            sam_score,
            float(area),
        )
    if compact_base:
        return (
            confidence,
            fill_ratio,
            sam_score,
            -bbox_area(item),
            float(area),
        )
    return (
        confidence,
        descriptor_score,
        fill_ratio,
        sam_score,
        float(area),
    )


def build_focus_prompt(episode: str, instructions: list[str], max_instructions: int) -> str:
    kept = instructions[:max_instructions]
    numbered = "\n".join(f"{idx + 1}. {instruction}" for idx, instruction in enumerate(kept))
    return FOCUS_USER_PROMPT_TEMPLATE.replace("{episode}", str(episode)).replace("{instructions}", numbered)


def extract_focus_objects(api_base: str, model: str, episode: str, instructions: list[str], args: argparse.Namespace) -> dict[str, Any]:
    user_prompt = build_focus_prompt(episode, instructions, args.max_instructions)
    raw_text = call_chat_completion(
        api_base,
        model,
        [
            {"role": "system", "content": FOCUS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=args.temperature,
        max_tokens=max(args.max_tokens, 1024),
        timeout=args.timeout,
        use_json_response_format=True,
    )
    try:
        parsed = extract_json_object(raw_text)
    except ValueError:
        parsed = extract_focus_objects_from_loose_text(raw_text)
    focus = normalize_focus_output(parsed)
    focus = merge_same_entity_focus_objects(focus, instructions)
    focus = merge_instruction_reference_objects(focus, instructions)
    focus = merge_same_entity_focus_objects(focus, instructions)
    focus["raw_model_text"] = raw_text
    return focus


def load_seg_results(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data.get("instances"), list):
        raise ValueError(f"seg-results 缺少 instances: {path}")
    return data


def find_mask_path(masks_dir: Path, index: int) -> Path | None:
    prefix = f"{index:03d}_"
    matches = sorted(path for path in masks_dir.glob("*.png") if path.name.startswith(prefix))
    if matches:
        return matches[0]
    direct = masks_dir / f"{index:03d}.png"
    if direct.is_file():
        return direct
    return None


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def clamp_bbox_with_padding(bbox: list[int], width: int, height: int, padding_ratio: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height
    pad = int(max(x2 - x1 + 1, y2 - y1 + 1) * max(0.0, padding_ratio))
    return max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + 1 + pad), min(height, y2 + 1 + pad)


def resize_long_side(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    scale = min(1.0, float(max_side) / float(max(width, height)))
    if scale >= 1.0:
        return image
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def build_candidate_image(
    image_path: Path,
    mask_path: Path,
    instance: dict[str, Any],
    out_path: Path,
    *,
    crop_padding_ratio: float,
    max_image_side: int,
    include_full_frame: bool,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != image.size:
        mask_img = mask_img.resize(image.size, Image.Resampling.NEAREST)

    image_arr = np.array(image).astype(np.float32)
    mask = np.array(mask_img) > 0
    if not mask.any():
        raise ValueError(f"空 mask: {mask_path}")

    canvas = np.zeros_like(image_arr, dtype=np.uint8)
    canvas[mask] = image_arr[mask].astype(np.uint8)
    overlay = Image.fromarray(canvas, mode="RGB")

    bbox = instance.get("bbox_xyxy") or bbox_from_mask(mask)
    draw = ImageDraw.Draw(overlay)
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=3)
    draw.text((max(0, x1), max(0, y1 - 16)), f"mask {int(instance.get('index', -1)):03d}", fill=(255, 255, 0))

    if not include_full_frame:
        crop_box = clamp_bbox_with_padding(bbox, image.width, image.height, crop_padding_ratio)
        overlay = overlay.crop(crop_box)
    overlay = resize_long_side(overlay, max_image_side)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(out_path)
    return out_path


def image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else "png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{data}"


def judge_one_mask(
    api_base: str,
    model: str,
    instructions: list[str],
    focus: dict[str, Any],
    instance: dict[str, Any],
    candidate_image: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    mask_json = {
        "index": instance.get("index"),
        "sam_score": instance.get("score"),
        "area": instance.get("area"),
        "bbox_xyxy": instance.get("bbox_xyxy"),
    }
    user_text = (
        JUDGE_USER_TEXT_TEMPLATE
        .replace("{instructions}", "\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(instructions[: args.max_instructions])))
        .replace("{focus_json}", json.dumps({"objects": focus["objects"], "prompts": focus["prompts"]}, ensure_ascii=False, indent=2))
        .replace("{mask_json}", json.dumps(mask_json, ensure_ascii=False, indent=2))
    )
    content = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": image_to_data_url(candidate_image)}},
    ]
    raw_text = call_chat_completion(
        api_base,
        model,
        [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        use_json_response_format=True,
    )
    parsed = extract_json_object(raw_text)
    result = {
        "mask_index": int(instance.get("index", parsed.get("mask_index", -1))),
        "is_target": bool(parsed.get("is_target", False)),
        "matched_prompt": normalize_text(parsed.get("matched_prompt") or ""),
        "object_name": normalize_text(parsed.get("object_name") or ""),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0),
        "reason": str(parsed.get("reason", "")).strip(),
        "candidate_image": str(candidate_image),
        "sam_score": instance.get("score"),
        "area": instance.get("area"),
        "bbox_xyxy": instance.get("bbox_xyxy"),
        "raw_model_text": raw_text,
    }
    valid_prompts = set(focus["prompts"])
    if result["matched_prompt"] not in valid_prompts:
        inferred_prompt = ""
        fallback_reason = "object_name fallback"
        if result["is_target"] or result["confidence"] > 0.0:
            inferred_prompt = infer_prompt_from_object_name(result["object_name"], focus)
        if inferred_prompt not in valid_prompts and result["is_target"]:
            inferred_prompt = infer_kitchenpot_prompt_from_text(" ".join([result["reason"], raw_text]), focus)
            fallback_reason = "kitchenpot visual fallback"
        if inferred_prompt in valid_prompts:
            result["matched_prompt"] = inferred_prompt
            result["is_target"] = True
            if result["confidence"] <= 0.0:
                result["confidence"] = 0.80
            result["reason"] = (result["reason"] + f" | matched by {fallback_reason}").strip()
        else:
            result["matched_prompt"] = ""
    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
    return result


def select_best_matches(judgments: list[dict[str, Any]], prompts: list[str], focus: dict[str, Any], instructions: list[str]) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for prompt in prompts:
        candidates = [item for item in judgments if item.get("is_target") and item.get("matched_prompt") == prompt]
        if not candidates:
            best[prompt] = None
            continue
        best[prompt] = max(
            candidates,
            key=lambda item: candidate_sort_key(item, prompt, focus, instructions),
        )
    return best

def build_final_answers(best_matches: dict[str, Any], prompts: list[str], focus_objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_to_base_object: dict[str, str] = {}
    for obj in focus_objects:
        if not isinstance(obj, dict):
            continue
        prompt = normalize_text(obj.get("prompt") or "")
        if not prompt:
            continue
        base_object = base_object_of(obj)
        if base_object:
            prompt_to_base_object[prompt] = base_object

    answers: list[dict[str, Any]] = []
    for prompt in prompts:
        base_object = prompt_to_base_object.get(normalize_text(prompt), "")
        match = best_matches.get(prompt)
        if not match:
            answers.append({"prompt": prompt, "base_object": base_object, "found": False, "mask_index": None})
            continue
        answers.append(
            {
                "prompt": prompt,
                "base_object": base_object,
                "found": True,
                "mask_index": match.get("mask_index"),
                "confidence": match.get("confidence"),
                "candidate_image": match.get("candidate_image"),
                "sam_score": match.get("sam_score"),
                "area": match.get("area"),
                "bbox_xyxy": match.get("bbox_xyxy"),
                "reason": match.get("reason"),
            }
        )
    return answers

def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main() -> None:
    args = parse_args()
    episodes_jsonl = args.episodes_jsonl.expanduser().resolve()
    seg_results_path = args.seg_results.expanduser().resolve()
    masks_dir = args.masks_dir.expanduser().resolve()
    output_path = args.output or seg_results_path.with_name(seg_results_path.stem.replace("_seganything_results", "") + "_qwen_mask_selection.json")
    output_path = output_path.expanduser().resolve()
    candidates_dir = args.candidates_dir or output_path.with_name(output_path.stem + "_candidates")
    candidates_dir = candidates_dir.expanduser().resolve()

    if not episodes_jsonl.is_file():
        raise FileNotFoundError(f"episodes-jsonl 不存在: {episodes_jsonl}")
    if not seg_results_path.is_file():
        raise FileNotFoundError(f"seg-results 不存在: {seg_results_path}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks-dir 不存在: {masks_dir}")

    episode_item = load_episode_item(episodes_jsonl, args.episode)
    instructions = extract_instructions(episode_item)
    seg_results = load_seg_results(seg_results_path)
    image_path = Path(str(seg_results.get("image", ""))).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"seg-results 中的 image 不存在: {image_path}")

    api_base = normalize_api_base(args.api_base)
    model = resolve_model(api_base, args.model, args.timeout)
    print(f"[INFO] api_base={api_base}")
    print(f"[INFO] model={model}")
    print(f"[INFO] episode={args.episode} instruction_count={len(instructions)}")

    focus = extract_focus_objects(api_base, model, args.episode, instructions, args)
    print(f"[INFO] instruction objects={', '.join(focus['prompts'])}")

    instances = list(seg_results.get("instances") or [])
    instances.sort(key=lambda item: int(item.get("index", 0)))
    if args.min_mask_area > 0:
        instances = [item for item in instances if int(item.get("area", 0) or 0) >= args.min_mask_area]
    if args.max_masks > 0:
        instances = instances[: args.max_masks]

    judgments: list[dict[str, Any]] = []
    for position, instance in enumerate(instances, 1):
        mask_index = int(instance.get("index", -1))
        mask_path = find_mask_path(masks_dir, mask_index)
        if mask_path is None:
            message = f"找不到 mask_index={mask_index} 对应 PNG"
            if not args.allow_fail:
                raise FileNotFoundError(message)
            judgments.append({"mask_index": mask_index, "is_target": False, "error": message})
            continue

        candidate_image = candidates_dir / f"mask_{mask_index:03d}_candidate.png"
        try:
            build_candidate_image(
                image_path,
                mask_path,
                instance,
                candidate_image,
                crop_padding_ratio=args.crop_padding_ratio,
                max_image_side=args.max_image_side,
                include_full_frame=args.include_full_frame,
            )
            judgment = judge_one_mask(api_base, model, instructions, focus, instance, candidate_image, args)
            judgments.append(judgment)
            status = "TARGET" if judgment.get("is_target") else "no"
            print(
                f"[MASK] {position}/{len(instances)} index={mask_index:03d} {status} "
                f"prompt={judgment.get('matched_prompt', '')} conf={judgment.get('confidence', 0):.2f}"
            )
        except Exception as exc:
            if not args.allow_fail:
                raise
            judgments.append(
                {
                    "mask_index": mask_index,
                    "is_target": False,
                    "candidate_image": str(candidate_image),
                    "sam_score": instance.get("score"),
                    "area": instance.get("area"),
                    "bbox_xyxy": instance.get("bbox_xyxy"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[MASK][FAIL] {position}/{len(instances)} index={mask_index:03d} {type(exc).__name__}: {exc}")

    best_matches = select_best_matches(judgments, focus["prompts"], focus, instructions)
    final_answers = build_final_answers(best_matches, focus["prompts"], focus["objects"])
    payload = {
        "schema_version": 2,
        "episode": args.episode,
        "episode_index": episode_item.get("episode_index"),
        "episodes_jsonl": str(episodes_jsonl),
        "seg_results": str(seg_results_path),
        "image": str(image_path),
        "masks_dir": str(masks_dir),
        "candidates_dir": str(candidates_dir),
        "api_base": api_base,
        "model": model,
        "instructions": instructions,
        "focus_objects": {"objects": focus["objects"], "prompts": focus["prompts"]},
        "judgments": judgments,
        "best_matches": best_matches,
        "final_answers": final_answers,
    }
    atomic_write_json(output_path, payload)
    for answer in final_answers:
        mask_text = "None" if answer.get("mask_index") is None else f"{int(answer['mask_index']):03d}"
        print(f"[FINAL] prompt={answer['prompt']} mask_index={mask_text} found={answer['found']}")
    print(f"[DONE] wrote {output_path}")
    print(f"[DONE] candidates_dir={candidates_dir}")


if __name__ == "__main__":
    main()
