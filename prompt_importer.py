import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aiohttp import web
from PIL import Image

import folder_paths
from server import PromptServer


_PROMPT_WORDS = ("prompt", "text", "caption", "description", "instruction", "提示词", "提示語", "提示语")
_NEGATIVE_WORDS = ("negative", "neg_prompt", "negative_prompt", "反向", "负面", "負面", "负向")
_POSITIVE_WORDS = ("positive", "pos_prompt", "positive_prompt", "正向", "正面")
_STRING_NODES = ("primitivestring", "stringmultiline", "cliptextencode", "textencode", "prompt")
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
_IMAGE_EXTENSIONS = {".png", ".webp", ".jpg", ".jpeg", ".gif", ".avif"}
_FILE_SUFFIXES = (
    ".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx",
    ".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".mkv", ".webm",
)


@dataclass
class _Candidate:
    text: str
    score: int
    label: str
    kind: str
    source: str
    node_id: str = ""

    def as_dict(self):
        return {
            "text": self.text,
            "score": self.score,
            "length": len(self.text),
            "label": self.label,
            "kind": self.kind,
            "source": self.source,
            "node_id": self.node_id,
        }


def _contains(text: Any, words) -> bool:
    lowered = str(text or "").lower()
    return any(word in lowered for word in words)


def _decode(value: Any, max_depth: int = 4):
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    for _ in range(max_depth):
        if not isinstance(value, str):
            break
        stripped = value.strip().strip("\x00")
        if not stripped:
            return ""
        try:
            decoded = json.loads(stripped)
        except (json.JSONDecodeError, TypeError):
            return stripped
        if decoded == value:
            break
        value = decoded
    return value


def _clean_text(value: Any) -> str:
    return value.replace("\x00", "").strip() if isinstance(value, str) else ""


def _usable_text(value: Any) -> bool:
    text = _clean_text(value)
    if not text or len(text) > 250_000:
        return False
    lowered = text.lower()
    if lowered in {"none", "null", "true", "false", "default", "auto"}:
        return False
    if "\n" not in text and len(text) < 512:
        if lowered.endswith(_FILE_SUFFIXES):
            return False
        if ("\\" in text or "/" in text) and re.search(r"\.[a-z0-9]{2,6}$", lowered):
            return False
    return True


def _kind(*parts: str) -> str:
    joined = " ".join(str(part or "") for part in parts)
    if _contains(joined, _NEGATIVE_WORDS):
        return "negative"
    if _contains(joined, _POSITIVE_WORDS):
        return "positive"
    return "prompt"


def _node_score(class_type: str, title: str, field: str, index=None) -> int:
    score = 0
    if _contains(title, ("prompt", "提示词", "提示語", "提示语")):
        score += 130
    if _contains(title, _POSITIVE_WORDS):
        score += 70
    if _contains(title, _NEGATIVE_WORDS):
        score -= 180
    if _contains(class_type, _STRING_NODES):
        score += 100
    lowered = field.lower()
    if "prompt" in lowered:
        score += 95
    elif "text" in lowered:
        score += 80
    elif lowered in {"value", "string"}:
        score += 45
    elif _contains(lowered, _PROMPT_WORDS):
        score += 60
    if _contains(lowered, _NEGATIVE_WORDS):
        score -= 200
    if _contains(lowered, _POSITIVE_WORDS):
        score += 50
    if isinstance(index, int) and index > 0:
        score -= min(index * 3, 30)
    return score


def _consumer_score(consumers) -> int:
    best = 0
    for input_name, target_type, target_title in consumers:
        joined = f"{input_name} {target_type} {target_title}"
        score = 0
        if _contains(input_name, ("prompt", "提示词", "提示語", "提示语")):
            score += 130
        elif "text" in input_name.lower():
            score += 90
        if _contains(target_type, ("conditioning", "textencode", "prompt")):
            score += 55
        if _contains(joined, _NEGATIVE_WORDS):
            score -= 220
        if _contains(joined, _POSITIVE_WORDS):
            score += 60
        best = max(best, score)
    return best


def _api_consumers(prompt):
    result = {}
    for target_id, node in prompt.items():
        if not isinstance(node, dict):
            continue
        for input_name, value in node.get("inputs", {}).items():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                result.setdefault(str(value[0]), []).append((
                    str(input_name), str(node.get("class_type", "")),
                    str(node.get("_meta", {}).get("title", target_id)),
                ))
    return result


def _extract_api(prompt, source):
    candidates = []
    consumers = _api_consumers(prompt)
    for node_id, node in prompt.items():
        if not isinstance(node, dict) or "class_type" not in node:
            continue
        class_type = str(node.get("class_type", ""))
        title = str(node.get("_meta", {}).get("title", ""))
        downstream = _consumer_score(consumers.get(str(node_id), []))
        relevant_node = _contains(class_type, _STRING_NODES) or _contains(title, _PROMPT_WORDS) or downstream > 0
        for field, value in node.get("inputs", {}).items():
            if not isinstance(value, str) or not _usable_text(value):
                continue
            if not relevant_node and not _contains(field, _PROMPT_WORDS):
                continue
            score = _node_score(class_type, title, str(field)) + downstream
            if score < 50:
                continue
            candidates.append(_Candidate(
                _clean_text(value), score, title or class_type or f"Node {node_id}",
                _kind(title, str(field)), source, str(node_id),
            ))
    return candidates


def _workflow_consumers(workflow):
    nodes = {str(node.get("id")): node for node in workflow.get("nodes", []) if isinstance(node, dict)}
    result = {}
    for link in workflow.get("links", []):
        if isinstance(link, dict):
            source_id, target_id, target_slot = str(link.get("origin_id")), str(link.get("target_id")), link.get("target_slot")
        elif isinstance(link, list) and len(link) >= 5:
            source_id, target_id, target_slot = str(link[1]), str(link[3]), link[4]
        else:
            continue
        target = nodes.get(target_id, {})
        inputs = target.get("inputs", [])
        input_name = ""
        if isinstance(target_slot, int) and isinstance(inputs, list) and 0 <= target_slot < len(inputs):
            input_name = str(inputs[target_slot].get("name", ""))
        result.setdefault(source_id, []).append((input_name, str(target.get("type", "")), str(target.get("title", ""))))
    return result


def _extract_workflow(workflow, source):
    candidates = []
    consumers = _workflow_consumers(workflow)
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        class_type = str(node.get("type", ""))
        title = str(node.get("title", ""))
        downstream = _consumer_score(consumers.get(node_id, []))
        relevant_node = _contains(class_type, _STRING_NODES) or _contains(title, _PROMPT_WORDS) or downstream > 0
        values = node.get("widgets_values", [])
        if isinstance(values, dict):
            fields = [(str(key), value, None) for key, value in values.items()]
        elif isinstance(values, list):
            fields = [("value", value, index) for index, value in enumerate(values)]
        elif isinstance(values, str):
            fields = [("value", values, None)]
        else:
            fields = []
        for field, value, index in fields:
            if not isinstance(value, str) or not _usable_text(value):
                continue
            if not relevant_node and not _contains(field, _PROMPT_WORDS):
                continue
            score = _node_score(class_type, title, field, index) + downstream
            if score < 50:
                continue
            candidates.append(_Candidate(
                _clean_text(value), score, title or class_type or f"Node {node_id}",
                _kind(title, field), source, node_id,
            ))
    return candidates


def _a1111(parameters, source):
    text = _clean_text(parameters)
    if not text:
        return []
    positive = re.split(r"\nNegative prompt\s*:", text, maxsplit=1, flags=re.IGNORECASE)[0]
    positive = re.split(r"\nSteps\s*:", positive, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    return [_Candidate(positive, 220, "Parameters prompt", "positive", source)] if _usable_text(positive) else []


def _collect_metadata(metadata, source):
    candidates = []
    for raw_key, raw_value in metadata.items():
        key, value = str(raw_key).lower(), raw_value
        if isinstance(value, str) and ":" in value:
            prefix, remainder = value.split(":", 1)
            if prefix.lower() in {"prompt", "workflow", "parameters"}:
                key, value = prefix.lower(), remainder
        decoded = _decode(value)
        if key == "prompt":
            if isinstance(decoded, dict):
                candidates.extend(_extract_api(decoded, f"{source}:prompt"))
            elif _usable_text(decoded):
                candidates.append(_Candidate(_clean_text(decoded), 250, "Embedded prompt", "prompt", f"{source}:prompt"))
        elif key == "workflow" and isinstance(decoded, dict):
            candidates.extend(_extract_workflow(decoded, f"{source}:workflow"))
        elif key in {"parameters", "comment", "comments", "description"}:
            if isinstance(decoded, dict):
                candidates.extend(_collect_metadata(decoded, f"{source}:{key}"))
            elif key == "parameters" and isinstance(decoded, str):
                candidates.extend(_a1111(decoded, f"{source}:parameters"))
        elif isinstance(decoded, dict) and ("prompt" in decoded or "workflow" in decoded):
            candidates.extend(_collect_metadata(decoded, f"{source}:{key}"))
    return candidates


def _rank(candidates):
    unique = {}
    for candidate in candidates:
        candidate.text = _clean_text(candidate.text)
        if not candidate.text:
            continue
        previous = unique.get(candidate.text)
        if previous is None or candidate.score > previous.score:
            unique[candidate.text] = candidate

    # Negative prompts never become the automatic choice. Among viable prompts,
    # text volume is deliberately the main signal because hidden helper widgets
    # usually contain only a short label while the real H3 prompt is substantial.
    return sorted(
        unique.values(),
        key=lambda item: (item.kind != "negative", len(item.text), item.score),
        reverse=True,
    )


def _extract_json(data, source):
    data = _decode(data)
    candidates = []
    if isinstance(data, dict):
        if isinstance(data.get("nodes"), list):
            candidates.extend(_extract_workflow(data, source))
        if "prompt" in data or "workflow" in data:
            candidates.extend(_collect_metadata(data, source))
        if data and all(isinstance(value, dict) and "class_type" in value for value in data.values()):
            candidates.extend(_extract_api(data, source))
    return _rank(candidates)


def _extract_file(file_object, filename):
    extension = Path(filename).suffix.lower()
    source = os.path.basename(filename)
    file_object.seek(0)
    if extension == ".json":
        return _extract_json(json.load(file_object), source), source
    if extension in _IMAGE_EXTENSIONS:
        with Image.open(file_object) as image:
            metadata = dict(image.info)
            try:
                for key, value in image.getexif().items():
                    metadata[f"exif_{key}"] = value
            except (AttributeError, TypeError, ValueError):
                pass
        return _rank(_collect_metadata(metadata, source)), source
    if extension in _VIDEO_EXTENSIONS:
        import av
        with av.open(file_object, mode="r") as container:
            metadata = dict(container.metadata)
        return _rank(_collect_metadata(metadata, source)), source
    raise ValueError(f"不支持的文件类型：{extension or 'unknown'}")


def _find_companion(filename):
    stem = Path(os.path.basename(filename)).stem
    stems = [stem, re.sub(r"(?:[-_ ]audio)$", "", stem, flags=re.IGNORECASE)]
    names = list(dict.fromkeys(name for item in stems for name in (f"{item}.png", f"{item}.webp")))
    for root in (folder_paths.get_output_directory(), folder_paths.get_input_directory(), folder_paths.get_temp_directory()):
        if not root or not os.path.isdir(root):
            continue
        for current_root, _, files in os.walk(root):
            match = next((name for name in names if name in files), None)
            if match:
                return os.path.join(current_root, match)
    return None


@PromptServer.instance.routes.post("/minimax-h3-chunk-star7/prompt-import")
async def _prompt_import_route(request):
    try:
        post = await request.post()
        upload = post.get("file")
        if upload is None or not getattr(upload, "file", None):
            return web.json_response({"success": False, "message": "没有收到文件，原提示词未改变。"}, status=400)
        filename = os.path.basename(upload.filename or "")
        if not filename:
            return web.json_response({"success": False, "message": "文件名无效，原提示词未改变。"}, status=400)
        candidates, source = _extract_file(upload.file, filename)
        if not candidates and Path(filename).suffix.lower() in _VIDEO_EXTENSIONS:
            companion = _find_companion(filename)
            if companion:
                with open(companion, "rb") as handle:
                    candidates, _ = _extract_file(handle, companion)
                source = f"{filename} → {os.path.basename(companion)}"
        if not candidates:
            return web.json_response({
                "success": False, "message": "文件中没有找到可用提示词，原内容未改变。",
                "candidates": [], "source": source,
            })
        visible = candidates[:20]
        automatic = next((candidate for candidate in visible if candidate.kind != "negative"), None)
        if automatic is None:
            return web.json_response({
                "success": False,
                "message": "只识别到负面提示词，未自动覆盖；可以从候选词中手动选择。",
                "candidates": [candidate.as_dict() for candidate in visible],
                "source": source,
            })
        return web.json_response({
            "success": True,
            "prompt": automatic.text,
            "selected_index": visible.index(automatic),
            "candidates": [candidate.as_dict() for candidate in visible],
            "source": source,
            "message": f"已识别 {len(candidates)} 个候选提示词。",
        })
    except (json.JSONDecodeError, OSError, ValueError) as error:
        return web.json_response({
            "success": False, "message": f"无法读取该文件：{error}。原内容未改变。", "candidates": [],
        })
    except Exception as error:
        print(f"[MiniMaxH3Chunk-Star7 Prompt Import] failed: {error}")
        return web.json_response({
            "success": False, "message": "提示词解析失败，原内容未改变。", "candidates": [],
        }, status=500)


class MiniMaxH3PromptImportStar7:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "", "multiline": True, "dynamicPrompts": True,
                    "tooltip": "Edit directly, or drop a workflow image, video, or JSON file onto the node to import its prompt.",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "get_prompt"
    CATEGORY = "Star7/MiniMax H3"
    DESCRIPTION = "Imports prompts from images, videos, or workflow JSON and prioritizes the strongest long-text candidate. — Star7"

    @classmethod
    def IS_CHANGED(cls, prompt):
        return prompt

    def get_prompt(self, prompt):
        return (str(prompt),)


NODE_CLASS_MAPPINGS = {"MiniMaxH3PromptImportStar7": MiniMaxH3PromptImportStar7}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3PromptImportStar7": "Prompt Load - Star7"}
