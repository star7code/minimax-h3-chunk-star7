import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAMES = new Set([
    "MiniMaxH3ActivationChunkStar7",
    "MiniMaxH3RoPEChunkPatch",
]);
const REFERENCE_LOAD_NODE_NAME = "MiniMaxH3ReferenceVideoLoadStar7";
const IMAGE_LOAD_SCALE_NODE_NAME = "MiniMaxH3LoadImageScaleStar7";
const WORKFLOW_EXPORT_NODE_NAME = "Star7VideoWorkflowExport";
const VHS_VIDEO_COMBINE_NODE_NAME = "VHS_VideoCombine";
const REAL_WIDGET_DEFAULTS = {
    chunk_tokens: 8192,
    auto_halve_on_oom: true,
    verbose: true,
    // Migration values only: old saved workflows that predate these fields
    // keep their historical behavior. Fresh node defaults come from Python.
    mlp_chunk_tokens: 4096,
    qkv_chunk_tokens: 4096,
    out_proj_chunk_tokens: 4096,
    disable_dynamic_prefetch: "auto",
    reuse_mlp_weights: true,
    attention_backend: "comfy_kitchen_int8",
};
const REAL_WIDGET_NAMES = Object.keys(REAL_WIDGET_DEFAULTS);
const LEGACY_REAL_WIDGET_NAMES = REAL_WIDGET_NAMES.filter(
    (name) => name !== "out_proj_chunk_tokens",
);

const TEXT = {
    en: {
        title: "MiniMax H3 VRAM Chunk Acceleration - Star7",
        legacyTitle: "MiniMax H3 VRAM Chunk Acceleration (Legacy Workflow)",
        labels: {
            chunk_tokens: "RoPE chunk size",
            auto_halve_on_oom: "Auto-reduce after VRAM error",
            verbose: "Show runtime details",
            mlp_chunk_tokens: "MLP chunk size (main VRAM control)",
            qkv_chunk_tokens: "QKV chunk size",
            out_proj_chunk_tokens: "out_proj fallback chunk size",
            disable_dynamic_prefetch: "Output VRAM guard (may slow down)",
            reuse_mlp_weights: "Reuse chunked weights (faster)",
            attention_backend: "Attention acceleration method",
        },
        tooltips: {
            chunk_tokens: "0 tries full-sequence RoPE first. With automatic reduction enabled, only a RoPE VRAM error makes it retry with a smaller chunk. Usually keep 8192.",
            auto_halve_on_oom: "Retries only the failing RoPE, MLP, or QKV stage with a smaller chunk. Attention output protection is controlled independently. It cannot reduce model weights, attention buffers, or other fixed allocations.",
            verbose: "Shows actual chunk sizes, automatic reductions, and active QKV/MLP weight modes in the console.",
            mlp_chunk_tokens: "0 tries full-sequence MLP first. With automatic reduction enabled, an MLP VRAM error is retried with a smaller chunk. Otherwise this is the main VRAM control.",
            qkv_chunk_tokens: "Controls the temporary QKV projection workspace. 0 tries the full sequence first; with automatic reduction enabled, only a QKV projection VRAM error reduces this stage.",
            out_proj_chunk_tokens: "Fallback workspace control for H3 out_proj. Eligible SM75 TensorWise INT8 uses the full fused CUTLASS contraction without a full INT32 buffer; this value is used only when fused execution is unavailable or the upstream full projection runs out of VRAM.",
            disable_dynamic_prefetch: "Auto keeps normal workloads on full out_proj and uses an internal bounded tile only when a long sequence lacks safe headroom. Off restores the incoming out_proj behavior.",
            reuse_mlp_weights: "Reuses isolated QKV/MLP weight snapshots across token chunks. Falls back safely if a snapshot runs out of VRAM.",
            attention_backend: "Select the incoming backend, CK INT8, or the matching SM75/SM80+ SLA, Sol, or Hybrid path. Architecture-specific sparse modes stop on failure instead of substituting another backend. All-INT8 prioritizes throughput and should be quality-tested for the target workflow.",
        },
        protectionAuto: "Auto",
        protectionOff: "Off",
        current: (label, value) => `${label} in use: ${value} (configured)`,
        limited: (label, value, configured) => `${label} in use: ${value} (set ${configured}, limited by video size)`,
        reduced: (label, value, configured) => `${label} auto-reduced to: ${value} (set ${configured})`,
        full: (label) => `${label}: full sequence (no fixed chunk)`,
        reducedFromFull: (label, value) => `${label} auto-reduced from full sequence to: ${value}`,
        rope: "RoPE",
        mlp: "MLP",
    },
    zh: {
        title: "MiniMax H3 显存分块加速 - Star7",
        legacyTitle: "MiniMax H3 显存分块加速（旧工作流兼容）",
        labels: {
            chunk_tokens: "RoPE 分块大小",
            auto_halve_on_oom: "显存不足时自动降档",
            verbose: "显示运行详情",
            mlp_chunk_tokens: "MLP 分块大小（主要显存调节）",
            qkv_chunk_tokens: "QKV 分块大小（参考显存调节）",
            out_proj_chunk_tokens: "out_proj 兜底分块（参考显存调节）",
            disable_dynamic_prefetch: "输出显存保护（可能降速）",
            reuse_mlp_weights: "复用分块权重（提速）",
            attention_backend: "注意力加速方式",
        },
        tooltips: {
            chunk_tokens: "设为 0 会先尝试整段 RoPE；开启自动降档后，只有 RoPE 显存不足才缩小重试。通常保持 8192。",
            auto_halve_on_oom: "只缩小发生显存不足的 RoPE、MLP 或 QKV 阶段并重试；注意力输出显存保护由独立选项控制。模型权重、注意力固定缓冲等显存不会被误降。",
            verbose: "在控制台显示实际分块、是否自动降档以及 QKV/MLP 权重加速方式。",
            mlp_chunk_tokens: "设为 0 会先尝试整段 MLP；开启自动降档后，只有 MLP 显存不足才缩小重试。它仍是主要显存调节项。",
            qkv_chunk_tokens: "调节 QKV 投影的临时显存。设为 0 会先尝试整段计算；开启自动降档后，只有 QKV 投影显存不足才降低这一阶段。",
            out_proj_chunk_tokens: "用于 out_proj 的显存兜底。符合条件的 SM75 TensorWise INT8 会优先整段调用 fused CUTLASS，不生成完整 INT32 缓冲；仅在 fused 不可用或上游整段投影显存不足时使用此分块值。",
            disable_dynamic_prefetch: "自动：普通任务保持整段 out_proj，长序列显存余量不足时才启用内部安全分块；关闭：恢复传入模型原始的 out_proj 行为。",
            reuse_mlp_weights: "在 token 块之间复用独立 QKV/MLP 权重快照；快照显存不足时自动切换安全流式路径。",
            attention_backend: "可选择传入模型已有后端、CK INT8，或与显卡架构对应的 SM75/SM80+ SLA、Sol、Hybrid 路径。架构专用稀疏模式失败时终止，不替换为其他后端；All-INT8 侧重吞吐，建议按目标工作流验证质量。",
        },
        protectionAuto: "自动",
        protectionOff: "关闭",
        current: (label, value) => `${label} 实际使用：${value}（设定值）`,
        limited: (label, value, configured) => `${label} 实际使用：${value}（设定 ${configured}，视频规模只需要这么多）`,
        reduced: (label, value, configured) => `${label} 已自动降为：${value}（原设定 ${configured}）`,
        full: (label) => `${label}：整段计算（未固定分块）`,
        reducedFromFull: (label, value) => `${label} 已从整段自动降为：${value}`,
        rope: "RoPE",
        mlp: "MLP",
    },
};

const REFERENCE_LOAD_TEXT = {
    en: {
        title: "Reference Video Load - Star7",
        labels: {
            video: "Reference video",
            max_long_edge: "Maximum long edge",
            allow_upscale: "Allow small video upscale",
            trim_enabled: "Trim reference range",
            trim_start_seconds: "Start time (seconds)",
            trim_end_seconds: "End time (seconds)",
        },
        outputs: ["reference video", "reference audio", "frame count", "report"],
    },
    zh: {
        title: "参考视频载入 - Star7",
        labels: {
            video: "参考视频",
            max_long_edge: "最长边限制",
            allow_upscale: "允许小视频放大",
            trim_enabled: "裁切视频范围",
            trim_start_seconds: "开始时间（秒）",
            trim_end_seconds: "结束时间（秒）",
        },
        outputs: ["参考视频画面", "参考视频音频", "帧数", "报告"],
    },
};

const IMAGE_LOAD_SCALE_TEXT = {
    en: {
        title: "Reference Image Load - Star7",
        labels: {
            image: "Image",
            "最长边": "Maximum long edge",
            "允许小图放大": "Allow small image upscale",
            "调整比例": "Crop to aspect ratio",
            "目标比例": "Target aspect ratio",
        },
        landscapeSuffix: " (Landscape)",
        outputs: ["image", "mask"],
    },
    zh: {
        title: "参考图像载入 - Star7",
        labels: {
            image: "图片",
            "最长边": "最长边限制",
            "允许小图放大": "允许小图放大",
            "调整比例": "调整比例",
            "目标比例": "目标比例",
        },
        landscapeSuffix: "（横版）",
        outputs: ["图片", "遮罩"],
    },
};

const WORKFLOW_EXPORT_TEXT = {
    en: {
        title: "Video and Workflow Export - Star7",
        labels: { "视频文件": "Video file", "导出方式": "Export mode" },
        modes: {
            "视频内置工作流": "Video with embedded workflow",
            "仅视频": "Video only",
            "视频 + 工作流 JSON": "Video + workflow JSON",
            "视频 + 工作流 PNG": "Video + workflow PNG",
        },
    },
    zh: {
        title: "视频与工作流导出 - Star7",
        labels: { "视频文件": "视频文件", "导出方式": "导出方式" },
        modes: {
            "视频内置工作流": "视频内置工作流",
            "仅视频": "仅导出视频",
            "视频 + 工作流 JSON": "视频与工作流 JSON",
            "视频 + 工作流 PNG": "视频与内置工作流图片",
        },
    },
};

function language() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language
        ?? "en";
    return String(locale).toLowerCase().startsWith("zh") ? "zh" : "en";
}

function strings() {
    return TEXT[language()];
}

function normalizeProtectionValue(value) {
    if (typeof value === "boolean") return "auto";
    const normalized = String(value ?? "").trim().toLowerCase();
    if (["off", "disabled", "false", "0", "关闭", "关"].includes(normalized)) {
        return "off";
    }
    return "auto";
}

function localizeReferenceLoadNode(node) {
    const text = REFERENCE_LOAD_TEXT[language()];
    node.title = text.title;
    for (const [name, label] of Object.entries(text.labels)) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (widget) {
            widget.label = label;
            widget.localized_name = label;
        }
        const input = node.inputs?.find((item) => item.name === name);
        if (input) input.localized_name = label;
    }
    node.outputs?.forEach((output, index) => {
        if (text.outputs[index]) output.localized_name = text.outputs[index];
    });
}

function setCompactWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!("__star7OriginalType" in widget)) {
        widget.__star7OriginalType = widget.type;
        widget.__star7OriginalComputeSize = widget.computeSize;
        widget.__star7OriginalComputedHeight = widget.computedHeight;
    }
    if (visible) {
        widget.type = widget.__star7OriginalType;
        if (widget.__star7OriginalComputeSize === undefined) {
            delete widget.computeSize;
        } else {
            widget.computeSize = widget.__star7OriginalComputeSize;
        }
        widget.computedHeight = widget.__star7OriginalComputedHeight;
    } else {
        // Keep the widget in node.widgets so prompt/workflow serialization
        // remains positional; collapse only its canvas layout and drawing.
        widget.type = "converted-widget:star7-trim";
        widget.computeSize = () => [0, -4];
        widget.computedHeight = 0;
    }
    for (const key of ["element", "inputEl"]) {
        if (widget[key]?.style) widget[key].style.display = visible ? "" : "none";
    }
}

function refreshReferenceTrimControls(node) {
    const toggle = node.widgets?.find((widget) => widget.name === "trim_enabled");
    const start = node.widgets?.find((widget) => widget.name === "trim_start_seconds");
    const end = node.widgets?.find((widget) => widget.name === "trim_end_seconds");
    if (!toggle || !start || !end) return;
    const visible = toggle.value === true || toggle.value === 1 || toggle.value === "true";
    setCompactWidgetVisible(start, visible);
    setCompactWidgetVisible(end, visible);
    node.setDirtyCanvas?.(true, true);
}

const REFERENCE_PREVIEW_MIN_WIDTH = 320;
const REFERENCE_PREVIEW_MIN_HEIGHT = 120;
const REFERENCE_PREVIEW_WIDGET_NAMES = new Set([
    "video-preview",
    "$$comfy_animation_preview",
    "videopreview",
]);
const REFERENCE_PREVIEW_PASS_THROUGH_CLASS = "star7-reference-preview-pass-through";

function referencePreviewElements(preview) {
    const previewElement = preview?.element ?? preview?.parentEl;
    const previewRoot = preview?.element?.closest?.(".dom-widget")
        ?? preview?.parentEl?.parentElement
        ?? previewElement?.parentElement;
    const media = [
        preview?.videoEl,
        preview?.imgEl,
        ...Array.from(preview?.element?.querySelectorAll?.("video, img") ?? []),
    ].filter(Boolean);
    return {
        previewElement,
        previewRoot,
        media: [...new Set(media)],
    };
}

function installReferencePreviewPassThroughStyle() {
    if (
        typeof document === "undefined"
        || !document.head?.appendChild
        || !document.createElement
        || document.getElementById?.("star7-reference-preview-pass-through-style")
    ) return;
    const style = document.createElement("style");
    style.id = "star7-reference-preview-pass-through-style";
    style.textContent = `
        .${REFERENCE_PREVIEW_PASS_THROUGH_CLASS} {
            pointer-events: none !important;
            user-select: none !important;
        }
        .${REFERENCE_PREVIEW_PASS_THROUGH_CLASS} video {
            pointer-events: auto !important;
        }
    `;
    document.head.appendChild(style);
}

function withReferencePreviewCollapsed(node, preview, callback) {
    const computeSize = preview.computeSize;
    const computedHeight = preview.computedHeight;
    // Keep the widget row's normal layout gap while measuring all non-preview
    // chrome. The returned preview height can then fill the exact remainder.
    preview.computeSize = (width) => [width, 0];
    preview.computedHeight = 0;
    try {
        return callback();
    } finally {
        preview.computeSize = computeSize;
        preview.computedHeight = computedHeight;
    }
}

function withReferenceTrimExpanded(node, callback) {
    const widgets = ["trim_start_seconds", "trim_end_seconds"]
        .map((name) => node.widgets?.find((widget) => widget.name === name))
        .filter(Boolean);
    const state = widgets.map((widget) => ({
        widget,
        type: widget.type,
        computeSize: widget.computeSize,
        computedHeight: widget.computedHeight,
    }));
    for (const widget of widgets) {
        widget.type = widget.__star7OriginalType ?? widget.type;
        if (widget.__star7OriginalComputeSize === undefined) {
            delete widget.computeSize;
        } else {
            widget.computeSize = widget.__star7OriginalComputeSize;
        }
        widget.computedHeight = widget.__star7OriginalComputedHeight;
    }
    try {
        return callback();
    } finally {
        for (const item of state) {
            item.widget.type = item.type;
            item.widget.computeSize = item.computeSize;
            item.widget.computedHeight = item.computedHeight;
        }
    }
}

function referencePreviewChromeHeight(node, preview, width, expandedTrim = false) {
    const calculate = () => withReferencePreviewCollapsed(
        node,
        preview,
        // LiteGraph treats the second value passed to computeSize as a
        // requested minimum.  Passing the restored node height here makes a
        // saved instance permanently unable to shrink after reopening.
        () => Number(node.computeSize?.([width, 0])?.[1]) || 0,
    );
    return expandedTrim ? withReferenceTrimExpanded(node, calculate) : calculate();
}

function referencePreviewMinimumSize(node, preview) {
    const width = Math.max(REFERENCE_PREVIEW_MIN_WIDTH, Number(node.size?.[0]) || 0);
    const chromeHeight = referencePreviewChromeHeight(node, preview, width, true);
    return [REFERENCE_PREVIEW_MIN_WIDTH, chromeHeight + REFERENCE_PREVIEW_MIN_HEIGHT];
}

function clampReferencePreviewSize(node, preview) {
    if (!Array.isArray(node.size)) return;
    const minimum = referencePreviewMinimumSize(node, preview);
    const width = Math.max(minimum[0], Number(node.size[0]) || minimum[0]);
    const height = Math.max(minimum[1], Number(node.size[1]) || minimum[1]);
    if (width !== node.size[0] || height !== node.size[1]) {
        node.setSize?.([width, height]);
    }
}

function rememberReferenceFrameSize(node) {
    if (!Array.isArray(node.size)) return;
    node.__star7ReferenceFrameSize = [
        Number(node.size[0]) || REFERENCE_PREVIEW_MIN_WIDTH,
        Number(node.size[1]) || REFERENCE_PREVIEW_MIN_HEIGHT,
    ];
}

function installReferenceMediaSizeGuard(node, preview) {
    const preserveFrame = () => {
        if (!Array.isArray(node.__star7ReferenceFrameSize)) {
            rememberReferenceFrameSize(node);
        }
        node.__star7ReferenceMediaSizing = true;
        Promise.resolve().then(() => {
            const size = node.__star7ReferenceFrameSize;
            if (Array.isArray(size)) node.setSize?.([...size]);
            node.__star7ReferenceMediaSizing = false;
            node.setDirtyCanvas?.(true, true);
        });
    };
    for (const element of referencePreviewElements(preview).media) {
        if (element.__star7MediaSizeGuardInstalled) continue;
        element.__star7MediaSizeGuardInstalled = true;
        const isVideo = String(element.tagName ?? "").toLowerCase() === "video"
            || element === preview.videoEl;
        element.addEventListener?.(isVideo ? "loadedmetadata" : "load", preserveFrame, true);
    }
}

function referencePreviewWidget(node) {
    return node.widgets?.find((widget) => REFERENCE_PREVIEW_WIDGET_NAMES.has(widget.name));
}

function installReferencePreviewLayout(node) {
    const preview = referencePreviewWidget(node);
    if (!preview) return false;
    if (!Array.isArray(node.__star7ReferenceFrameSize)) {
        rememberReferenceFrameSize(node);
    }
    installReferenceMediaSizeGuard(node, preview);
    if (!preview.__star7FixedFrameLayout) {
        preview.__star7FixedFrameLayout = true;
        preview.__star7OriginalComputeSize = preview.computeSize;
        preview.__star7OriginalComputeLayoutSize = preview.computeLayoutSize;
        // A computeSize tied to node.size becomes LiteGraph's current minimum
        // on every resize, so a taller node can never shrink again. Use the
        // flexible layout API: 120px is the minimum, while the preview receives
        // all remaining body height during normal layout.
        preview.computeSize = undefined;
        preview.computeLayoutSize = () => ({
            minHeight: REFERENCE_PREVIEW_MIN_HEIGHT,
            maxHeight: 1_000_000,
            minWidth: 0,
            maxWidth: 1_000_000,
        });
    }

    // Keep the DOM wrapper transparent to LiteGraph, while allowing the video
    // itself to receive native playback-control events. A small bottom/right
    // gutter leaves the canvas resize handle reachable after media is loaded.
    installReferencePreviewPassThroughStyle();
    const { previewElement, previewRoot, media } = referencePreviewElements(preview);
    for (const element of [previewRoot, previewElement, ...media]) {
        if (!element?.style) continue;
        element.classList?.add?.(REFERENCE_PREVIEW_PASS_THROUGH_CLASS);
        element.style.pointerEvents = "none";
        element.style.userSelect = "none";
    }

    if (previewElement?.style) {
        previewElement.style.width = "100%";
        previewElement.style.height = "100%";
        previewElement.style.boxSizing = "border-box";
        previewElement.style.paddingRight = "10px";
        previewElement.style.paddingBottom = "10px";
        previewElement.style.overflow = "hidden";
        previewElement.style.display = "flex";
        previewElement.style.alignItems = "center";
        previewElement.style.justifyContent = "center";
    }
    for (const element of media) {
        if (!element?.style) continue;
        const isVideo = String(element.tagName ?? "").toLowerCase() === "video"
            || element === preview.videoEl;
        element.style.pointerEvents = isVideo ? "auto" : "none";
        element.style.width = "100%";
        element.style.height = "100%";
        element.style.maxWidth = "100%";
        element.style.maxHeight = "100%";
        element.style.objectFit = "contain";
    }

    if (!node.__star7ReferenceResizeInstalled) {
        node.__star7ReferenceResizeInstalled = true;
        const originalResize = node.onResize;
        node.onResize = function (size) {
            const requested = Array.isArray(size)
                ? [Number(size[0]), Number(size[1])]
                : [Number(this.size?.[0]), Number(this.size?.[1])];
            const target = this.__star7ReferenceMediaSizing
                ? this.__star7ReferenceFrameSize
                : requested;
            const result = originalResize?.apply(this, arguments);
            const minimum = referencePreviewMinimumSize(this, preview);
            const width = Math.max(minimum[0], Number(target?.[0]) || minimum[0]);
            const height = Math.max(minimum[1], Number(target?.[1]) || minimum[1]);
            if (Array.isArray(this.size)) {
                this.size[0] = width;
                this.size[1] = height;
            }
            if (Array.isArray(size)) {
                size[0] = width;
                size[1] = height;
            }
            if (!this.__star7ReferenceMediaSizing) {
                this.__star7ReferenceFrameSize = [width, height];
            }
            this.setDirtyCanvas?.(true, true);
            return result;
        };
    }
    clampReferencePreviewSize(node, preview);
    node.setDirtyCanvas?.(true, true);
    return Boolean(previewRoot);
}

function installReferencePreviewWatcher(node) {
    if (node.__star7ReferencePreviewWatcherInstalled) return;
    node.__star7ReferencePreviewWatcherInstalled = true;
    const originalDrawBackground = node.onDrawBackground;
    function star7ReferencePreviewWatcher() {
        const result = originalDrawBackground?.apply(this, arguments);
        const preview = referencePreviewWidget(this);
        const root = preview ? referencePreviewElements(preview).previewRoot : null;
        if (!root?.classList?.contains?.(REFERENCE_PREVIEW_PASS_THROUGH_CLASS)) {
            if (!installReferencePreviewLayout(this)) return result;
        }
        if (this.onDrawBackground === star7ReferencePreviewWatcher) {
            this.onDrawBackground = originalDrawBackground;
        }
        this.__star7ReferencePreviewWatcherInstalled = false;
        return result;
    }
    node.onDrawBackground = star7ReferencePreviewWatcher;
}

function refreshReferenceVideoLayout(node) {
    refreshReferenceTrimControls(node);
    installReferencePreviewLayout(node);
    node.setDirtyCanvas?.(true, true);
}

function scheduleReferenceVideoLayout(node) {
    installReferencePreviewWatcher(node);
    refreshReferenceVideoLayout(node);
    const schedule = globalThis.requestAnimationFrame;
    if (typeof schedule === "function") {
        schedule(() => refreshReferenceVideoLayout(node));
    }
    // Core creates its media preview after restoring the node and loading the
    // video. It therefore may not exist during configure or the first frame.
    const defer = globalThis.setTimeout;
    if (typeof defer === "function") {
        const generation = (node.__star7ReferenceLayoutGeneration || 0) + 1;
        node.__star7ReferenceLayoutGeneration = generation;
        for (const delay of [50, 250, 1000, 2500]) {
            defer(() => {
                if (node.__star7ReferenceLayoutGeneration === generation) {
                    refreshReferenceVideoLayout(node);
                }
            }, delay);
        }
    }
}

function applyReferenceVideoDuration(node, sourceDuration, resetRange = false) {
    if (!Number.isFinite(sourceDuration) || sourceDuration <= 0) return;
    const durationValue = Math.round(sourceDuration * 1000) / 1000;
    const start = node.widgets?.find((widget) => widget.name === "trim_start_seconds");
    const end = node.widgets?.find((widget) => widget.name === "trim_end_seconds");
    if (!start || !end) return;
    node.__star7ReferenceDuration = durationValue;
    start.options ??= {};
    end.options ??= {};
    start.options.max = durationValue;
    if (resetRange) start.value = 0;
    start.value = Math.min(durationValue, Math.max(0, Number(start.value) || 0));
    end.options.min = start.value;
    end.options.max = durationValue;
    if (resetRange) {
        end.value = durationValue;
    } else {
        const current = Number(end.value) || durationValue;
        end.value = Math.min(durationValue, Math.max(start.value, current));
    }
    node.setDirtyCanvas?.(true, true);
}

function refreshReferenceDurationLimit(node) {
    const sourceDuration = Number(node.__star7ReferenceDuration);
    if (!Number.isFinite(sourceDuration) || sourceDuration <= 0) return;
    applyReferenceVideoDuration(node, sourceDuration, false);
}

function referenceVideoViewUrl(value) {
    if (!value) return null;
    const raw = String(value).replace(/\s*\[(?:input|output|temp)\]\s*$/i, "");
    const normalized = raw.replaceAll("\\", "/");
    const slash = normalized.lastIndexOf("/");
    const filename = slash >= 0 ? normalized.slice(slash + 1) : normalized;
    const subfolder = slash >= 0 ? normalized.slice(0, slash) : "";
    const query = new URLSearchParams({ filename, type: "input", subfolder });
    return api.apiURL(`/view?${query.toString()}`);
}

function probeReferenceVideoDuration(node, resetRange = false) {
    if (typeof document === "undefined") return;
    const videoWidget = node.widgets?.find((widget) => widget.name === "video");
    const url = referenceVideoViewUrl(videoWidget?.value);
    if (!url) return;
    const requestId = (node.__star7DurationRequestId || 0) + 1;
    node.__star7DurationRequestId = requestId;
    const media = document.createElement("video");
    media.preload = "metadata";
    media.onloadedmetadata = () => {
        if (node.__star7DurationRequestId !== requestId) return;
        applyReferenceVideoDuration(node, Number(media.duration), resetRange);
        media.removeAttribute("src");
        media.load?.();
    };
    media.onerror = () => {
        if (node.__star7DurationRequestId === requestId) {
            console.warn("[Star7] Unable to read reference video duration", videoWidget?.value);
        }
    };
    media.src = url;
}

function installReferenceTrimControls(node) {
    const toggle = node.widgets?.find((widget) => widget.name === "trim_enabled");
    const video = node.widgets?.find((widget) => widget.name === "video");
    const start = node.widgets?.find((widget) => widget.name === "trim_start_seconds");
    const end = node.widgets?.find((widget) => widget.name === "trim_end_seconds");
    if (!toggle) return;
    if (!node.__star7ReferenceTrimInstalled) {
        node.__star7ReferenceTrimInstalled = true;
        const originalCallback = toggle.callback;
        toggle.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            refreshReferenceVideoLayout(node);
            return result;
        };
        const originalVideoCallback = video?.callback;
        if (video) video.callback = function () {
            const result = originalVideoCallback?.apply(this, arguments);
            probeReferenceVideoDuration(node, !node.__star7Configuring);
            return result;
        };
        const originalStartCallback = start?.callback;
        if (start) start.callback = function () {
            const result = originalStartCallback?.apply(this, arguments);
            refreshReferenceDurationLimit(node);
            return result;
        };
        const originalEndCallback = end?.callback;
        if (end) end.callback = function () {
            const result = originalEndCallback?.apply(this, arguments);
            refreshReferenceDurationLimit(node);
            return result;
        };
        probeReferenceVideoDuration(node, true);
    }
    refreshReferenceTrimControls(node);
}

function isReferenceMediaFile(file, kind) {
    const name = String(file?.name || "").toLowerCase();
    const mime = String(file?.type || "").toLowerCase();
    if (kind === "video") {
        return mime.startsWith("video/") || /\.(mp4|mov|mkv|webm|avi|m4v|wmv|flv|mpeg|mpg)$/.test(name);
    }
    return mime.startsWith("image/") || /\.(png|jpe?g|webp|bmp|gif|tiff?|avif)$/.test(name);
}

async function uploadReferenceMedia(node, file, kind) {
    if (!isReferenceMediaFile(file, kind)) return false;
    const form = new FormData();
    form.append("image", file, file.name);
    const response = await api.fetchApi("/upload/image", { method: "POST", body: form });
    if (!response.ok) throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
    const uploaded = await response.json();
    const value = uploaded.subfolder
        ? `${uploaded.subfolder.replaceAll("\\", "/")}/${uploaded.name}`
        : uploaded.name;
    const widgetName = kind === "video" ? "video" : "image";
    const widget = node.widgets?.find((item) => item.name === widgetName);
    if (!widget) return false;
    const values = widget.options?.values;
    if (Array.isArray(values) && !values.includes(value)) values.push(value);
    widget.value = value;
    widget.callback?.(value);
    node.graph?.setDirtyCanvas?.(true, true);
    node.setDirtyCanvas?.(true, true);
    return true;
}

function installReferenceMediaDrop(node, kind) {
    if (node.__star7ReferenceMediaDropInstalled) return;
    node.__star7ReferenceMediaDropInstalled = true;
    const previousDragOver = node.onDragOver;
    const previousDragDrop = node.onDragDrop;
    node.onDragOver = function (event) {
        if (Array.from(event?.dataTransfer?.files || []).some((file) => isReferenceMediaFile(file, kind))) return true;
        return previousDragOver?.apply(this, arguments) ?? false;
    };
    node.onDragDrop = function (event) {
        const file = Array.from(event?.dataTransfer?.files || []).find((item) => isReferenceMediaFile(item, kind));
        if (!file) return previousDragDrop?.apply(this, arguments) ?? false;
        event.preventDefault?.();
        event.stopPropagation?.();
        event.stopImmediatePropagation?.();
        void uploadReferenceMedia(this, file, kind).catch((error) => console.error("[Star7]", error));
        return true;
    };
}

function localizeImageLoadScaleNode(node) {
    const text = IMAGE_LOAD_SCALE_TEXT[language()];
    node.title = text.title;
    for (const [name, label] of Object.entries(text.labels)) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (widget) {
            widget.label = label;
            widget.localized_name = label;
        }
    }
    node.outputs?.forEach((output, index) => {
        if (text.outputs[index]) output.localized_name = text.outputs[index];
    });
    const ratio = node.widgets?.find((item) => item.name === "目标比例");
    if (ratio) {
        ratio.options ??= {};
        ratio.options.getOptionLabel = (value) => {
            const [width, height] = String(value).split(":").map(Number);
            return width > height ? `${value}${text.landscapeSuffix}` : String(value);
        };
    }
    refreshImageAspectControls(node);
}

function refreshImageAspectControls(node) {
    const toggle = node.widgets?.find((widget) => widget.name === "调整比例");
    const ratio = node.widgets?.find((widget) => widget.name === "目标比例");
    if (!toggle || !ratio) return;
    const visible = toggle.value === true || toggle.value === 1 || toggle.value === "true";
    setCompactWidgetVisible(ratio, visible);
    node.setDirtyCanvas?.(true, true);
}

function installImageAspectControls(node) {
    const toggle = node.widgets?.find((widget) => widget.name === "调整比例");
    if (!toggle) return;
    if (!node.__star7ImageAspectInstalled) {
        node.__star7ImageAspectInstalled = true;
        const originalCallback = toggle.callback;
        toggle.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            refreshImageAspectControls(node);
            return result;
        };
    }
    refreshImageAspectControls(node);
}

function localizeWorkflowExportNode(node) {
    const text = WORKFLOW_EXPORT_TEXT[language()];
    node.title = text.title;
    for (const [name, label] of Object.entries(text.labels)) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (widget) {
            widget.label = label;
            widget.localized_name = label;
        }
        const input = node.inputs?.find((item) => item.name === name);
        if (input) input.localized_name = label;
    }
    const mode = node.widgets?.find((item) => item.name === "导出方式");
    if (mode) {
        mode.options ??= {};
        mode.options.getOptionLabel = (value) => text.modes[value] ?? String(value);
    }
}

function linkedVHSVideoCombine(node) {
    const linkId = node.inputs?.find((input) => input.name === "视频文件")?.link
        ?? node.inputs?.[0]?.link;
    const link = app.graph?.links?.[linkId];
    if (!link) return null;
    const upstream = app.graph?.getNodeById?.(link.origin_id);
    return upstream?.comfyClass === VHS_VIDEO_COMBINE_NODE_NAME
        || upstream?.type === VHS_VIDEO_COMBINE_NODE_NAME
        ? upstream
        : null;
}

function installVHSPlaybackRecovery(node) {
    const preview = node.widgets?.find((widget) => widget.name === "videopreview");
    const video = preview?.videoEl;
    if (!preview || !video || video.__star7PlaybackRecoveryInstalled) return false;
    video.__star7PlaybackRecoveryInstalled = true;
    const nativePlay = video.play.bind(video);
    video.play = function () {
        let request;
        try {
            request = nativePlay();
        } catch (error) {
            request = Promise.reject(error);
        }
        if (!request?.catch) return request;
        return request.catch(() => {
            // VHS keeps a cached/transcoded URL. The workflow exporter replaces
            // the completed MP4 to add metadata, so that URL can become stale.
            // Rebuild it with VHS' own cache-busting timestamp and retry once.
            if (video.__star7PlaybackRecoveryPending) return undefined;
            video.__star7PlaybackRecoveryPending = true;
            globalThis.setTimeout?.(() => {
                preview.updateSource?.();
                video.addEventListener?.("canplay", () => {
                    video.__star7PlaybackRecoveryPending = false;
                    if (!preview.value?.paused && !preview.value?.hidden) {
                        nativePlay().catch?.(() => {});
                    }
                }, { once: true });
                globalThis.setTimeout?.(() => {
                    video.__star7PlaybackRecoveryPending = false;
                }, 3000);
            }, 80);
            return undefined;
        });
    };
    return true;
}

function scheduleVHSPlaybackRecovery(node) {
    installVHSPlaybackRecovery(node);
    for (const delay of [0, 100, 500, 1500]) {
        globalThis.setTimeout?.(() => installVHSPlaybackRecovery(node), delay);
    }
}

function installLinkedVHSPlaybackRecovery(exportNode) {
    const videoNode = linkedVHSVideoCombine(exportNode);
    if (videoNode) scheduleVHSPlaybackRecovery(videoNode);
}

function refreshLinkedVHSPreview(exportNode) {
    const videoNode = linkedVHSVideoCombine(exportNode);
    if (!videoNode) return;
    scheduleVHSPlaybackRecovery(videoNode);
    for (const delay of [80, 350]) {
        globalThis.setTimeout?.(() => {
            const preview = videoNode.widgets?.find((widget) => widget.name === "videopreview");
            preview?.updateSource?.();
            videoNode.setDirtyCanvas?.(true, true);
        }, delay);
    }
}

function formatValue(label, effective, configured, reason = "active") {
    const text = strings();
    if (configured === 0) {
        const stage = label.toLowerCase();
        if (reason === `${stage}_oom` && effective > 0) {
            return text.reducedFromFull(label, effective);
        }
        return text.full(label);
    }
    if (effective === configured) {
        return text.current(label, effective);
    }
    if (reason === "sequence_limit") {
        return text.limited(label, effective, configured);
    }
    return text.reduced(label, effective, configured);
}

function localizeNode(node) {
    const text = strings();
    const legacy = node.comfyClass === "MiniMaxH3RoPEChunkPatch"
        || node.type === "MiniMaxH3RoPEChunkPatch";
    const previousTitle = node.__star7LocalizedTitle;
    const nextTitle = legacy ? text.legacyTitle : text.title;
    if (!node.title || node.title === previousTitle || Object.values(TEXT).some(
        (item) => node.title === item.title || node.title === item.legacyTitle,
    )) {
        node.title = nextTitle;
        node.__star7LocalizedTitle = nextTitle;
    }
    for (const [name, label] of Object.entries(text.labels)) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget) continue;
        widget.label = label;
        widget.localized_name = label;
        widget.options ??= {};
        widget.options.tooltip = text.tooltips[name];
        if (name === "disable_dynamic_prefetch") {
            const enabled = normalizeProtectionValue(widget.value) !== "off";
            widget.options.values = [text.protectionAuto, text.protectionOff];
            widget.value = enabled ? text.protectionAuto : text.protectionOff;
        }
    }
    const modelInput = node.inputs?.find((input) => input.name === "model");
    if (modelInput) {
        modelInput.localized_name = language() === "zh" ? "模型" : "model";
    }
    const modelOutput = node.outputs?.find((output) => output.name === "model");
    if (modelOutput) {
        modelOutput.localized_name = language() === "zh" ? "模型" : "model";
    }
}

function removeStatusWidgets(node) {
    const statusNames = new Set([
        "star7_runtime_status",
        "star7_rope_runtime_status",
        "star7_mlp_runtime_status",
        "star7_qkv_runtime_status",
        "RoPE 当前使用",
        "MLP 当前使用",
    ]);
    for (let index = (node.widgets?.length ?? 0) - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (
            !statusNames.has(widget.__star7StatusName)
            && !statusNames.has(widget.name)
        ) {
            continue;
        }
        widget.onRemove?.();
        node.widgets.splice(index, 1);
    }
}

function makeStatusWidget(node, name, value) {
    let widget = node.widgets?.find(
        (item) => item.__star7StatusName === name || item.name === name,
    );
    if (!widget) {
        widget = node.addWidget(
            "text",
            value,
            "",
            () => {},
            { serialize: false },
        );
        widget.__star7StatusName = name;
        widget.disabled = true;
        widget.serializeValue = async () => undefined;
    }
    // Some ComfyUI themes hide the value of disabled text widgets. Put the
    // complete status in the always-visible widget label instead.
    widget.name = value;
    widget.value = "";
    return widget;
}

function normalizeConfiguredInputs(node) {
    for (const [name, fallback] of [
        ["chunk_tokens", 8192],
        ["mlp_chunk_tokens", 8192],
        ["qkv_chunk_tokens", 8192],
        ["out_proj_chunk_tokens", 4096],
    ]) {
        const widget = node.widgets?.find((item) => item.name === name);
        const numeric = Number(widget?.value);
        if (!widget || (Number.isInteger(numeric) && (numeric === 0 || numeric >= 256))) {
            continue;
        }
        widget.value = fallback;
        widget.callback?.(fallback);
    }
}

function validSavedValue(name, value) {
    if (name === "chunk_tokens" || name === "mlp_chunk_tokens" || name === "qkv_chunk_tokens" || name === "out_proj_chunk_tokens") {
        const numeric = Number(value);
        return Number.isInteger(numeric) && (numeric === 0 || numeric >= 256);
    }
    if (name === "attention_backend") {
        return value === "existing"
            || value === "comfy_kitchen_int8"
            || value === "sla_sm75_qk_int8_pv_fp16"
            || value === "sla_sm75_all_int8"
            || value === "sla_sm75_all_int8_experimental"
            || value === "sol_sm75_qk_int8_pv_fp16"
            || value === "sol_sm75_all_int8"
            || value === "sol_sm75_all_int8_experimental"
            || value === "hybrid_sm75_ck_sla_all_int8"
            || value === "hybrid_sm75_ck_sol_all_int8"
            || value === "sla_sm80+_qk_int8_pv_fp16"
            || value === "sla_sm80+_qk_int8_pv_bf16"
            || value === "sla_sm80+_all_int8"
            || value === "sla_sm80+_all_int8_experimental"
            || value === "sol_sm80+_bf16_official"
            || value === "sol_sm80+_all_int8"
            || value === "sol_sm80+_all_int8_experimental"
            || value === "hybrid_sm80+_ck_sla_all_int8"
            || value === "hybrid_sm80+_ck_sol_all_int8"
            || value === "hybrid_sm80+_ck_sla_qk_int8_pv_fp16"
            || value === "hybrid_sm80+_ck_sla_qk_int8_pv_bf16"
            || value === "hybrid_sm80+_ck_sol_bf16_official"
            || value === "sla_sm80+_qk_int8_pv_fp16";
    }
    if (name === "disable_dynamic_prefetch") {
        return typeof value === "boolean" || typeof value === "string";
    }
    return typeof value === "boolean";
}

function cleanSavedValues(info) {
    const named = info?.widgets_values_named ?? {};
    const rawPositional = Array.isArray(info?.widgets_values)
        ? info.widgets_values
        : [];
    const positionalNames = rawPositional.length === REAL_WIDGET_NAMES.length
        ? REAL_WIDGET_NAMES
        : rawPositional.length === LEGACY_REAL_WIDGET_NAMES.length
            ? LEGACY_REAL_WIDGET_NAMES
            : [];
    const positional = Object.fromEntries(
        positionalNames.map((name, index) => [name, rawPositional[index]]),
    );
    const values = REAL_WIDGET_NAMES.map((name, index) => {
        const namedValue = named[name];
        if (validSavedValue(name, namedValue)) {
            return name === "disable_dynamic_prefetch"
                ? normalizeProtectionValue(namedValue) : namedValue;
        }
        const positionalValue = positional[name];
        if (validSavedValue(name, positionalValue)) {
            return name === "disable_dynamic_prefetch"
                ? normalizeProtectionValue(positionalValue) : positionalValue;
        }
        return REAL_WIDGET_DEFAULTS[name];
    });
    return {
        ...info,
        widgets_values: values,
        widgets_values_named: Object.fromEntries(
            REAL_WIDGET_NAMES.map((name, index) => [name, values[index]]),
        ),
    };
}

function serializeRealWidgets(node, info) {
    const values = REAL_WIDGET_NAMES.map((name) => {
        const value = node.widgets?.find((widget) => widget.name === name)?.value;
        if (name === "disable_dynamic_prefetch") {
            return normalizeProtectionValue(value);
        }
        return validSavedValue(name, value) ? value : REAL_WIDGET_DEFAULTS[name];
    });
    info.widgets_values = values;
    info.widgets_values_named = Object.fromEntries(
        REAL_WIDGET_NAMES.map((name, index) => [name, values[index]]),
    );
}

function reorderDisplayWidgets(node) {
    const displayOrder = [
        "mlp_chunk_tokens",
        "star7_mlp_runtime_status",
        "qkv_chunk_tokens",
        "star7_qkv_runtime_status",
        "out_proj_chunk_tokens",
        "chunk_tokens",
        "star7_rope_runtime_status",
        "auto_halve_on_oom",
        "verbose",
        "reuse_mlp_weights",
        "disable_dynamic_prefetch",
        "attention_backend",
    ];
    const rank = new Map(displayOrder.map((name, index) => [name, index]));
    node.widgets?.sort((left, right) => {
        const leftName = left.__star7StatusName ?? left.name;
        const rightName = right.__star7StatusName ?? right.name;
        return (rank.get(leftName) ?? displayOrder.length)
            - (rank.get(rightName) ?? displayOrder.length);
    });
}

function hideInternalOutProjWidget(node) {
    const widget = node.widgets?.find((item) => item.name === "out_proj_chunk_tokens");
    if (!widget) return;
    widget.__star7InternalOutProj = true;
    // Legacy LiteGraph uses computeSize, while the newer Vue node renderer
    // observes hidden/type/options. Keep every marker so the internal fallback
    // value remains serialized but is never exposed as a user-facing control.
    widget.hidden = true;
    widget.options ??= {};
    widget.options.hidden = true;
    widget.__star7OriginalType ??= widget.type;
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
}

function restoreSerializedWidgetOrder(node) {
    const canonical = new Map(REAL_WIDGET_NAMES.map((name, index) => [name, index]));
    node.widgets?.sort((left, right) => {
        const leftName = left.__star7StatusName ?? left.name;
        const rightName = right.__star7StatusName ?? right.name;
        return (canonical.get(leftName) ?? REAL_WIDGET_NAMES.length)
            - (canonical.get(rightName) ?? REAL_WIDGET_NAMES.length);
    });
}

function moveAfter(node, widget, anchorName) {
    const widgetIndex = node.widgets.indexOf(widget);
    const anchorIndex = node.widgets.findIndex((item) => item.name === anchorName);
    if (widgetIndex < 0 || anchorIndex < 0 || widgetIndex === anchorIndex + 1) {
        return;
    }
    node.widgets.splice(widgetIndex, 1);
    const nextAnchorIndex = node.widgets.findIndex((item) => item.name === anchorName);
    node.widgets.splice(nextAnchorIndex + 1, 0, widget);
}

function ensureStatusWidgets(node) {
    removeStatusWidgets(node);
    normalizeConfiguredInputs(node);
    hideInternalOutProjWidget(node);
    const configuredRope = Number(
        node.widgets?.find((item) => item.name === "chunk_tokens")?.value ?? 8192,
    );
    const configuredMlp = Number(
        node.widgets?.find((item) => item.name === "mlp_chunk_tokens")?.value ?? 8192,
    );
    const configuredQkv = Number(
        node.widgets?.find((item) => item.name === "qkv_chunk_tokens")?.value ?? 8192,
    );
    const configuredOutProj = Number(
        node.widgets?.find((item) => item.name === "out_proj_chunk_tokens")?.value ?? 4096,
    );
    localizeNode(node);
    const text = strings();
    const runtime = node.__star7RuntimeDetail;
    const runtimeMatchesInputs = runtime
        && Number(runtime.configured_rope) === configuredRope
        && Number(runtime.configured_mlp) === configuredMlp
        && Number(runtime.configured_qkv) === configuredQkv
        && Number(runtime.configured_out_proj) === configuredOutProj;
    const effectiveRope = runtimeMatchesInputs
        ? Number(runtime.effective_rope) : configuredRope;
    const effectiveMlp = runtimeMatchesInputs
        ? Number(runtime.effective_mlp) : configuredMlp;
    const effectiveQkv = runtimeMatchesInputs
        ? Number(runtime.effective_qkv) : configuredQkv;
    const reason = runtimeMatchesInputs ? runtime.reason : "active";
    const rope = makeStatusWidget(
        node,
        "star7_rope_runtime_status",
        formatValue(text.rope, effectiveRope, configuredRope, reason),
    );
    const mlp = makeStatusWidget(
        node,
        "star7_mlp_runtime_status",
        formatValue(text.mlp, effectiveMlp, configuredMlp, reason),
    );
    const qkv = makeStatusWidget(
        node,
        "star7_qkv_runtime_status",
        formatValue("QKV", effectiveQkv, configuredQkv, reason),
    );
    moveAfter(node, rope, "chunk_tokens");
    moveAfter(node, mlp, "mlp_chunk_tokens");
    moveAfter(node, qkv, "qkv_chunk_tokens");
    reorderDisplayWidgets(node);
    return { rope, mlp, qkv };
}

api.addEventListener("star7-h3-chunk-status", ({ detail }) => {
    const rawId = detail?.node_id;
    if (rawId == null) {
        return;
    }
    const node = app.graph?.getNodeById(rawId)
        ?? app.graph?.getNodeById(Number(rawId));
    if (
        !node
        || (!NODE_NAMES.has(node.comfyClass) && !NODE_NAMES.has(node.type))
    ) {
        return;
    }
    node.__star7RuntimeDetail = { ...detail };
    ensureStatusWidgets(node);
    node.setDirtyCanvas?.(true, true);
});

app.registerExtension({
    name: "Star7.MiniMaxH3Chunk.RuntimeStatus",
    nodeCreated(node) {
        if (node.comfyClass === WORKFLOW_EXPORT_NODE_NAME || node.type === WORKFLOW_EXPORT_NODE_NAME) {
            localizeWorkflowExportNode(node);
            installLinkedVHSPlaybackRecovery(node);
            return;
        }
        if (node.comfyClass === IMAGE_LOAD_SCALE_NODE_NAME || node.type === IMAGE_LOAD_SCALE_NODE_NAME) {
            localizeImageLoadScaleNode(node);
            installImageAspectControls(node);
            installReferenceMediaDrop(node, "image");
            return;
        }
        if (node.comfyClass === REFERENCE_LOAD_NODE_NAME || node.type === REFERENCE_LOAD_NODE_NAME) {
            localizeReferenceLoadNode(node);
            installReferenceTrimControls(node);
            scheduleReferenceVideoLayout(node);
            installReferenceMediaDrop(node, "video");
            return;
        }
        if (NODE_NAMES.has(node.comfyClass) || NODE_NAMES.has(node.type)) {
            ensureStatusWidgets(node);
        }
    },
    loadedGraphNode(node) {
        if (node.comfyClass === WORKFLOW_EXPORT_NODE_NAME || node.type === WORKFLOW_EXPORT_NODE_NAME) {
            localizeWorkflowExportNode(node);
            installLinkedVHSPlaybackRecovery(node);
            return;
        }
        if (node.comfyClass === IMAGE_LOAD_SCALE_NODE_NAME || node.type === IMAGE_LOAD_SCALE_NODE_NAME) {
            localizeImageLoadScaleNode(node);
            installImageAspectControls(node);
            installReferenceMediaDrop(node, "image");
            return;
        }
        if (node.comfyClass === REFERENCE_LOAD_NODE_NAME || node.type === REFERENCE_LOAD_NODE_NAME) {
            localizeReferenceLoadNode(node);
            installReferenceTrimControls(node);
            scheduleReferenceVideoLayout(node);
            installReferenceMediaDrop(node, "video");
            probeReferenceVideoDuration(node, false);
            return;
        }
        if (NODE_NAMES.has(node.comfyClass) || NODE_NAMES.has(node.type)) {
            ensureStatusWidgets(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === WORKFLOW_EXPORT_NODE_NAME) {
            const text = WORKFLOW_EXPORT_TEXT[language()];
            nodeData.display_name = text.title;
            for (const [name, spec] of Object.entries(nodeData.input?.required ?? {})) {
                if (!text.labels[name] || !Array.isArray(spec)) continue;
                spec[1] ??= {};
                spec[1].display_name = text.labels[name];
                if (name === "导出方式") {
                    spec[1].getOptionLabel = (value) => text.modes[value] ?? String(value);
                }
            }
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                original?.apply(this, arguments);
                localizeWorkflowExportNode(this);
                installLinkedVHSPlaybackRecovery(this);
            };
            const originalExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function () {
                const result = originalExecuted?.apply(this, arguments);
                refreshLinkedVHSPreview(this);
                return result;
            };
            return;
        }
        if (nodeData.name === IMAGE_LOAD_SCALE_NODE_NAME) {
            const text = IMAGE_LOAD_SCALE_TEXT[language()];
            nodeData.display_name = text.title;
            for (const [name, spec] of Object.entries(nodeData.input?.required ?? {})) {
                if (!text.labels[name] || !Array.isArray(spec)) continue;
                spec[1] ??= {};
                spec[1].display_name = text.labels[name];
                if (name === "调整比例") {
                    spec[1].label_on = language() === "zh" ? "开启" : "On";
                    spec[1].label_off = language() === "zh" ? "关闭" : "Off";
                }
                if (name === "目标比例") {
                    spec[1].getOptionLabel = (value) => {
                        const [width, height] = String(value).split(":").map(Number);
                        return width > height
                            ? `${value}${text.landscapeSuffix}` : String(value);
                    };
                }
            }
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                original?.apply(this, arguments);
                localizeImageLoadScaleNode(this);
                installImageAspectControls(this);
                installReferenceMediaDrop(this, "image");
            };
            const originalConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function () {
                const result = originalConfigure?.apply(this, arguments);
                localizeImageLoadScaleNode(this);
                installImageAspectControls(this);
                installReferenceMediaDrop(this, "image");
                return result;
            };
            return;
        }
        if (nodeData.name === REFERENCE_LOAD_NODE_NAME) {
            const text = REFERENCE_LOAD_TEXT[language()];
            nodeData.display_name = text.title;
            for (const [name, spec] of Object.entries(nodeData.input?.required ?? {})) {
                if (!text.labels[name] || !Array.isArray(spec)) continue;
                spec[1] ??= {};
                spec[1].display_name = text.labels[name];
                if (spec[0] === "BOOLEAN") {
                    const isTrimToggle = name === "trim_enabled";
                    spec[1].label_on = language() === "zh"
                        ? (isTrimToggle ? "开启" : "允许")
                        : (isTrimToggle ? "On" : "Allow");
                    spec[1].label_off = language() === "zh"
                        ? (isTrimToggle ? "关闭" : "禁止")
                        : (isTrimToggle ? "Off" : "Disallow");
                }
            }
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                original?.apply(this, arguments);
                localizeReferenceLoadNode(this);
                installReferenceTrimControls(this);
                scheduleReferenceVideoLayout(this);
                installReferenceMediaDrop(this, "video");
            };

            const originalConfigure = nodeType.prototype.configure;
            nodeType.prototype.configure = function () {
                this.__star7Configuring = true;
                let result;
                try {
                    result = originalConfigure?.apply(this, arguments);
                } finally {
                    this.__star7Configuring = false;
                }
                localizeReferenceLoadNode(this);
                rememberReferenceFrameSize(this);
                installReferenceTrimControls(this);
                scheduleReferenceVideoLayout(this);
                installReferenceMediaDrop(this, "video");
                probeReferenceVideoDuration(this, false);
                return result;
            };
            return;
        }
        if (!NODE_NAMES.has(nodeData.name)) {
            return;
        }
        const text = strings();
        nodeData.display_name = nodeData.name === "MiniMaxH3RoPEChunkPatch"
            ? text.legacyTitle : text.title;
        for (const [name, spec] of Object.entries(nodeData.input?.required ?? {})) {
            if (!text.labels[name] || !Array.isArray(spec)) continue;
            spec[1] ??= {};
            spec[1].display_name = text.labels[name];
            spec[1].tooltip = text.tooltips[name];
            if (spec[0] === "BOOLEAN") {
                spec[1].label_on = language() === "zh" ? "开启" : "On";
                spec[1].label_off = language() === "zh" ? "关闭" : "Off";
            }
        }
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            ensureStatusWidgets(this);
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            // Legacy ComfyUI restores widget values by array position. Keep
            // display-only rows out of that array until real inputs are loaded.
            removeStatusWidgets(this);
            restoreSerializedWidgetOrder(this);
            const args = [...arguments];
            args[0] = cleanSavedValues(args[0]);
            const result = originalConfigure?.apply(this, args);
            ensureStatusWidgets(this);
            return result;
        };

        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            originalSerialize?.apply(this, arguments);
            serializeRealWidgets(this, info);
        };
    },
    setup() {
        app.ui?.settings?.addEventListener?.("Comfy.Locale.change", () => {
            for (const node of app.graph?._nodes ?? []) {
                if (node.comfyClass === WORKFLOW_EXPORT_NODE_NAME || node.type === WORKFLOW_EXPORT_NODE_NAME) {
                    localizeWorkflowExportNode(node);
                    node.setDirtyCanvas?.(true, true);
                    continue;
                }
                if (node.comfyClass === IMAGE_LOAD_SCALE_NODE_NAME || node.type === IMAGE_LOAD_SCALE_NODE_NAME) {
                    localizeImageLoadScaleNode(node);
                    installImageAspectControls(node);
                    node.setDirtyCanvas?.(true, true);
                    continue;
                }
                if (node.comfyClass === REFERENCE_LOAD_NODE_NAME || node.type === REFERENCE_LOAD_NODE_NAME) {
                    localizeReferenceLoadNode(node);
                    installReferenceTrimControls(node);
                    scheduleReferenceVideoLayout(node);
                    node.setDirtyCanvas?.(true, true);
                    continue;
                }
                if (NODE_NAMES.has(node.comfyClass) || NODE_NAMES.has(node.type)) {
                    ensureStatusWidgets(node);
                    node.setDirtyCanvas?.(true, true);
                }
            }
        });
    },
});
