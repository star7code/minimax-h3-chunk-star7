import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths

from .nodes import _star7_ffmpeg_path


_LOG = logging.getLogger("MiniMaxH3ActivationChunkStar7")
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_EMBED_MODE = "视频内置工作流"
_VIDEO_ONLY_MODE = "仅视频"
_JSON_MODE = "视频 + 工作流 JSON"
_PNG_MODE = "视频 + 工作流 PNG"


def _allowed_output_path(path):
    resolved = os.path.realpath(os.path.abspath(os.fspath(path)))
    roots = (
        os.path.realpath(folder_paths.get_output_directory()),
        os.path.realpath(folder_paths.get_temp_directory()),
    )
    if not any(os.path.commonpath((root, resolved)) == root for root in roots):
        raise ValueError(f"视频必须位于 ComfyUI 的 output 或 temp 目录：{resolved}")
    return resolved


def _collect_paths(value):
    paths = []
    if isinstance(value, (str, os.PathLike)):
        paths.append(os.fspath(value))
    elif isinstance(value, dict):
        fullpath = value.get("fullpath") or value.get("path")
        if fullpath:
            paths.extend(_collect_paths(fullpath))
        elif value.get("filename"):
            root = folder_paths.get_temp_directory() if value.get("type") == "temp" else folder_paths.get_output_directory()
            paths.append(os.path.join(root, value.get("subfolder", ""), value["filename"]))
        for key in ("files", "filenames"):
            if key in value:
                paths.extend(_collect_paths(value[key]))
    elif isinstance(value, (tuple, list)):
        for item in value:
            paths.extend(_collect_paths(item))
    return paths


def _resolve_video_and_vhs_files(value):
    candidates = []
    for path in _collect_paths(value):
        if not isinstance(path, str) or not path.strip():
            continue
        if not os.path.isabs(path):
            output_path = os.path.join(folder_paths.get_output_directory(), path)
            temp_path = os.path.join(folder_paths.get_temp_directory(), path)
            path = output_path if os.path.exists(output_path) else temp_path
        try:
            path = _allowed_output_path(path)
        except (ValueError, OSError):
            continue
        if os.path.isfile(path):
            candidates.append(path)

    video_path = next(
        (path for path in reversed(candidates) if Path(path).suffix.lower() in _VIDEO_EXTENSIONS),
        None,
    )
    if video_path is None:
        raise ValueError("没有从输入中找到有效的视频文件。可直接连接 VHS 的 filenames 输出或视频文件路径输出。")

    vhs_files = []
    if (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and isinstance(value[1], (tuple, list))
    ):
        for path in value[1]:
            if not isinstance(path, (str, os.PathLike)):
                continue
            try:
                checked = _allowed_output_path(path)
            except (ValueError, OSError):
                continue
            if os.path.isfile(checked):
                vhs_files.append(checked)
    return video_path, vhs_files


def _ffmetadata_value(value):
    value = value.replace("\\", "\\\\")
    value = value.replace(";", "\\;")
    value = value.replace("#", "\\#")
    value = value.replace("=", "\\=")
    return value.replace("\n", "\\\n")


def _replace_video_with_retry(source_path, video_path, attempts=600, delay=0.2):
    """Replace a freshly produced video after transient Windows readers release it."""
    last_error = None
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(source_path, video_path)
            return
        except PermissionError as error:
            last_error = error
            if attempt + 1 >= attempts:
                break
            time.sleep(max(0.0, float(delay)))
    raise PermissionError(
        f"无法写回视频，文件仍被预览器、播放器或安全软件占用：{video_path}"
    ) from last_error


def _remux_metadata(video_path, workflow, prompt, include_workflow):
    ffmpeg = _star7_ffmpeg_path()
    suffix = Path(video_path).suffix
    directory = os.path.dirname(video_path)
    handle, output_path = tempfile.mkstemp(prefix=".star7-workflow-", suffix=suffix, dir=directory)
    os.close(handle)
    os.remove(output_path)
    metadata_path = None
    try:
        args = [ffmpeg, "-v", "error", "-y", "-i", video_path]
        if include_workflow:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".ffmetadata", delete=False,
                dir=folder_paths.get_temp_directory(), newline="\n",
            ) as metadata_file:
                metadata_path = metadata_file.name
                metadata_file.write(";FFMETADATA1\n")
                metadata_file.write(
                    "workflow=" + _ffmetadata_value(json.dumps(workflow, ensure_ascii=False, separators=(",", ":"))) + "\n"
                )
                if prompt is not None:
                    metadata_file.write(
                        "prompt=" + _ffmetadata_value(json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))) + "\n"
                    )
            args.extend(["-f", "ffmetadata", "-i", metadata_path, "-map", "0", "-map_metadata", "1"])
        else:
            args.extend(["-map", "0", "-map_metadata", "-1"])
        args.extend(["-c", "copy"])
        if suffix.lower() in {".mp4", ".mov"}:
            args.extend(["-movflags", "use_metadata_tags+faststart"])
        args.append(output_path)
        try:
            subprocess.run(args, capture_output=True, check=True)
        except subprocess.CalledProcessError as error:
            message = error.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg 写入工作流失败：{message}") from error
        _replace_video_with_retry(output_path, video_path)
    finally:
        if metadata_path and os.path.exists(metadata_path):
            os.remove(metadata_path)
        if os.path.exists(output_path):
            os.remove(output_path)


def _save_workflow_json(video_path, workflow):
    path = str(Path(video_path).with_suffix(".workflow.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, ensure_ascii=False, indent=2)
    return path


def _write_workflow_png(source_path, output_path, workflow, prompt, extra_pnginfo):
    metadata = PngInfo()
    for key, value in (extra_pnginfo or {}).items():
        metadata.add_text(key, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    if "workflow" not in (extra_pnginfo or {}):
        metadata.add_text("workflow", json.dumps(workflow, ensure_ascii=False, separators=(",", ":")))
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt, ensure_ascii=False, separators=(",", ":")))
    with Image.open(source_path) as image:
        image.load()
        output_image = image.copy()
    output_image.save(output_path, pnginfo=metadata, compress_level=4)


def _save_workflow_png(video_path, vhs_files, workflow, prompt, extra_pnginfo):
    source_path = next(
        (path for path in vhs_files if Path(path).suffix.lower() == ".png"),
        None,
    )
    output_path = source_path or str(Path(video_path).with_suffix(".workflow.png"))
    extracted_path = None
    if source_path is None:
        ffmpeg = _star7_ffmpeg_path()
        handle, extracted_path = tempfile.mkstemp(suffix=".png", dir=folder_paths.get_temp_directory())
        os.close(handle)
        os.remove(extracted_path)
        try:
            subprocess.run(
                [ffmpeg, "-v", "error", "-y", "-i", video_path, "-frames:v", "1", extracted_path],
                capture_output=True, check=True,
            )
        except subprocess.CalledProcessError as error:
            message = error.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg 提取工作流图片失败：{message}") from error
        source_path = extracted_path
    try:
        _write_workflow_png(source_path, output_path, workflow, prompt, extra_pnginfo)
    finally:
        if extracted_path and os.path.exists(extracted_path):
            os.remove(extracted_path)
    return output_path


def _prune_vhs_files(vhs_files, video_path, keep_png):
    keep = {os.path.realpath(video_path)}
    if keep_png:
        keep.update(
            os.path.realpath(path)
            for path in vhs_files
            if Path(path).suffix.lower() == ".png"
        )
    for path in vhs_files:
        if os.path.realpath(path) not in keep and os.path.isfile(path):
            os.remove(path)


class Star7VideoWorkflowExport:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "视频文件": ("*",),
                "导出方式": (
                    [_EMBED_MODE, _VIDEO_ONLY_MODE, _JSON_MODE, _PNG_MODE],
                    {"default": _EMBED_MODE},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "export"
    CATEGORY = "Star7/output"

    def export(self, 视频文件, 导出方式, prompt=None, extra_pnginfo=None):
        workflow = (extra_pnginfo or {}).get("workflow")
        if workflow is None:
            raise ValueError("当前任务没有可写入的 ComfyUI 工作流。")
        video_path, vhs_files = _resolve_video_and_vhs_files(视频文件)

        if 导出方式 == _EMBED_MODE:
            _remux_metadata(video_path, workflow, prompt, True)
        elif 导出方式 == _VIDEO_ONLY_MODE:
            _remux_metadata(video_path, workflow, prompt, False)
        elif 导出方式 == _JSON_MODE:
            _save_workflow_json(video_path, workflow)
        elif 导出方式 == _PNG_MODE:
            _save_workflow_png(video_path, vhs_files, workflow, prompt, extra_pnginfo)
        else:
            raise ValueError(f"未知导出方式：{导出方式}")

        _prune_vhs_files(vhs_files, video_path, 导出方式 == _PNG_MODE)
        _LOG.info("[Star7 Workflow Export] %s | %s", 导出方式, video_path)
        # Ensure ComfyUI emits this output node's executed event. The frontend
        # then refreshes VHS after the MP4 is atomically replaced with the
        # metadata-bearing copy.
        return {"ui": {"star7_export": [os.path.basename(video_path)]}, "result": ()}


NODE_CLASS_MAPPINGS = {"Star7VideoWorkflowExport": Star7VideoWorkflowExport}
NODE_DISPLAY_NAME_MAPPINGS = {"Star7VideoWorkflowExport": "Video and Workflow Export - Star7"}
