import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FACE_NODE = "MiniMaxH3FaceRefineStar7";
const MATERIAL_NODE = "MiniMaxH3MaterialPromptStar7";
const PRESETS = {
    "自动平衡": { refine_steps: 4, custom_strength: 0.30, custom_canvas: "自动", custom_crop_context: 2.6, custom_blend: 0.90, custom_feather: 20 },
    "真人保真": { refine_steps: 4, custom_strength: 0.25, custom_canvas: "512", custom_crop_context: 2.8, custom_blend: 0.82, custom_feather: 24 },
    "远景小脸": { refine_steps: 4, custom_strength: 0.48, custom_canvas: "768", custom_crop_context: 2.4, custom_blend: 0.95, custom_feather: 18 },
    "动漫角色": { refine_steps: 4, custom_strength: 0.32, custom_canvas: "512", custom_crop_context: 2.7, custom_blend: 0.88, custom_feather: 20 },
};
const BALANCED = { ...PRESETS["自动平衡"] };
const PARAM_NAMES = Object.keys(BALANCED);
const TEXT = {
    zh: {
        materialTitle: "MiniMax H3 多合一条件载入 - Star7", faceTitle: "MiniMax H3 一键人脸修复 - Star7", reset: "重置参数",
        resolutionPending: "修复后分辨率：运行后显示",
        resolutionDisabled: "未开启修复",
        labels: {
            model: "模型", clip: "文本编码器", video_vae: "视频 VAE", audio_vae: "音频 VAE", prompt: "提示词",
            width: "宽度", height: "高度", length: "帧数", task_type: "任务类型", audio_mode: "音频模式",
            audio_denoise_strength: "重混强度（仅重混模式）", reference_quality: "参考素材尺寸", drive_audio: "驱动音频",
            final_audio: "最终输出音频", first_frame: "首帧", last_frame: "尾帧", sampled_av_latent: "采样结果",
            refine_context: "采样上下文", enable_refine: "启用修复", face_count: "修复人脸数量", preset: "修复预设", target_face: "目标人脸优先", refine_steps: "修复步数",
            custom_strength: "修复强度", custom_canvas: "修复尺寸", custom_crop_context: "人脸取景范围",
            custom_blend: "融合强度", custom_feather: "边缘羽化", seed: "修复种子", preserve_repair_detail: "保持修复清晰度",
            positive: "正面条件", av_latent: "音视频潜空间",
            mux_audio: "输出音频", report: "运行报告", refined_images: "图像",
        },
    },
    en: {
        materialTitle: "MiniMax H3 All-in-one Conditioning - Star7", faceTitle: "MiniMax H3 One-click Face Repair - Star7", reset: "Reset parameters",
        resolutionPending: "Repaired resolution: shown after run",
        resolutionDisabled: "Face repair disabled",
        labels: {
            model: "Model", clip: "Text encoder", video_vae: "Video VAE", audio_vae: "Audio VAE", prompt: "Prompt",
            width: "Width", height: "Height", length: "Frames", task_type: "Task type", audio_mode: "Audio mode",
            audio_denoise_strength: "Remix strength (Remix only)", reference_quality: "Reference size", drive_audio: "Driving audio",
            final_audio: "Final output audio", first_frame: "First frame", last_frame: "Last frame", sampled_av_latent: "Sampled result",
            refine_context: "Sampling context", enable_refine: "Enable repair", face_count: "Faces to repair", preset: "Repair preset", target_face: "Face priority", refine_steps: "Repair steps",
            custom_strength: "Repair strength", custom_canvas: "Repair size", custom_crop_context: "Face crop context",
            custom_blend: "Blend strength", custom_feather: "Edge feather", seed: "Repair seed", preserve_repair_detail: "Preserve repair detail",
            positive: "Positive", av_latent: "AV latent",
            mux_audio: "Output audio", report: "Run report", refined_images: "Images",
        },
    },
};

function widget(node, name) { return node.widgets?.find((item) => item.name === name); }
function language() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale") ?? globalThis.navigator?.language ?? "en";
    return String(locale).toLowerCase().startsWith("zh") ? "zh" : "en";
}
function dynamicLabel(name, lang) {
    const patterns = [
        [/^ref_image_(\d+)$/, "参考图", "Reference image"],
        [/^ref_video_(\d+)$/, "参考视频", "Reference video"],
        [/^ref_video_audio_(\d+)$/, "参考视频音频", "Reference-video audio"],
        [/^ref_audio_(\d+)$/, "参考音频", "Reference audio"],
    ];
    for (const [pattern, zh, en] of patterns) {
        const match = name.match(pattern);
        if (match) return `${lang === "zh" ? zh : en} ${Number(match[1]) + 1}`;
    }
    return null;
}
function labelFor(name, lang) { return TEXT[lang].labels[name] ?? dynamicLabel(name, lang); }
function inputBaseName(input) { return String(input?.name || "").split(".").at(-1); }
function referenceImageSlot(input) {
    const match = inputBaseName(input).match(/^ref_image_(\d+)$/);
    return match ? Number(match[1]) : null;
}
function compactMaterialReferenceImages(node) {
    if (node.__star7CompactingReferenceImages || !Array.isArray(node.inputs)) return;
    const references = node.inputs.map((input, index) => ({ input, index, slot: referenceImageSlot(input) }))
        .filter(({ slot }) => slot != null);
    if (!references.length) return;

    const connected = references.filter(({ input }) => input.link != null);
    const nextSlot = connected.length ? Math.max(...connected.map(({ slot }) => slot)) + 1 : 0;
    const spare = references.find(({ input, slot }) => input.link == null && slot === nextSlot)
        ?? references.find(({ input }) => input.link == null);
    const retained = new Set(connected.map(({ input }) => input));
    if (spare) retained.add(spare.input);
    const stale = references.filter(({ input }) => !retained.has(input));
    if (!stale.length) return;

    node.__star7CompactingReferenceImages = true;
    try {
        for (const { index } of stale.sort((a, b) => b.index - a.index)) {
            if (typeof node.removeInput === "function") node.removeInput(index);
            else node.inputs.splice(index, 1);
        }
    } finally {
        node.__star7CompactingReferenceImages = false;
    }
}
function placeMaterialReferenceImages(node) {
    const inputs = node.inputs;
    if (!Array.isArray(inputs)) return;
    const references = inputs.filter((input) => referenceImageSlot(input) != null)
        .sort((a, b) => referenceImageSlot(a) - referenceImageSlot(b));
    if (!references.length) return;
    const remaining = inputs.filter((input) => !references.includes(input));
    const lastFrame = remaining.findIndex((input) => inputBaseName(input) === "last_frame");
    if (lastFrame < 0) return;
    inputs.splice(0, inputs.length,
        ...remaining.slice(0, lastFrame + 1),
        ...references,
        ...remaining.slice(lastFrame + 1),
    );
}
function connectedMediaInputs(node, pattern) {
    return (node.inputs ?? []).map((input) => {
        const match = inputBaseName(input).match(pattern);
        return match ? { input, slot: Number(match[1]) } : null;
    }).filter((item) => item && item.input.link != null).sort((a, b) => a.slot - b.slot);
}
function updateMaterialMediaLabels(node) {
    const lang = language();
    const suffix = (base, tag) => `${base} - <${tag}>`;
    const base = (zh, en, slot) => `${lang === "zh" ? zh : en} ${slot + 1}`;

    const drive = (node.inputs ?? []).find((input) => inputBaseName(input) === "drive_audio");
    if (drive?.link != null) {
        drive.label = drive.localized_name = suffix(
            lang === "zh" ? "驱动音频" : "Driving audio", "Audio D"
        );
    }

    const images = connectedMediaInputs(node, /^ref_image_(\d+)$/);
    images.forEach(({ input }, index) => {
        input.label = input.localized_name = suffix(base("参考图", "Reference image", Number(inputBaseName(input).match(/(\d+)$/)[1])), `Picture ${index + 1}`);
    });
    const videos = connectedMediaInputs(node, /^ref_video_(\d+)$/);
    videos.forEach(({ input }, index) => {
        input.label = input.localized_name = suffix(base("参考视频", "Reference video", Number(inputBaseName(input).match(/(\d+)$/)[1])), `Video ${index + 1}`);
    });

    const connectedVideoSlots = new Set(videos.map(({ slot }) => slot));
    const soundtracks = connectedMediaInputs(node, /^ref_video_audio_(\d+)$/)
        .filter(({ slot }) => connectedVideoSlots.has(slot));
    let audioOrdinal = 1;
    soundtracks.forEach(({ input, slot }) => {
        input.label = input.localized_name = suffix(base("参考视频音频", "Reference-video audio", slot), `Audio ${audioOrdinal++}`);
    });
    const audios = connectedMediaInputs(node, /^ref_audio_(\d+)$/);
    audios.forEach(({ input, slot }) => {
        input.label = input.localized_name = suffix(base("参考音频", "Reference audio", slot), `Audio ${audioOrdinal++}`);
    });
}
function localizeNode(node, isFace) {
    const lang = language();
    const text = TEXT[lang];
    node.title = isFace ? text.faceTitle : text.materialTitle;
    for (const item of [...(node.inputs ?? []), ...(node.outputs ?? []), ...(node.widgets ?? [])]) {
        const label = labelFor(inputBaseName(item), lang);
        if (label) item.label = item.localized_name = label;
    }
}
function migrateMaterialValues(node) {
    const maps = {
        task_type: {
            auto: "自动判断 / Auto", T2VA: "文生视频 / T2VA", I2VA: "首帧生视频 / I2VA",
            FL2VA: "首尾帧生视频 / FL2VA", L2VA: "尾帧生视频 / L2VA",
            Ref2VA: "参考素材生视频 / Ref2VA", Hybrid: "混合条件 / Hybrid",
        },
        audio_mode: {
            lock_source: "锁定原音 / Lock Source", remix_source: "重混原音 / Remix Source",
            reference_only: "仅作音频参考 / Reference Only", native: "模型原生生成 / Native",
        },
        reference_quality: { match: "匹配生成画布 / Match", max: "保留高分辨率参考 / Max（显存较高）" },
    };
    for (const [name, values] of Object.entries(maps)) {
        const item = widget(node, name);
        if (item && values[item.value]) item.value = values[item.value];
    }
}
function setHidden(item, hidden) {
    if (!item) return;
    item.options ??= {};
    if (!item.__star7VisibilityCaptured) {
        item.__star7VisibilityCaptured = true;
        item.__star7OriginalType = item.type;
        item.__star7OriginalComputeSize = item.computeSize;
    }
    item.hidden = item.options.hidden = hidden;
    if (hidden) {
        item.type = "hidden";
        item.computeSize = () => [0, -4];
    } else {
        item.type = item.__star7OriginalType;
        if (item.__star7OriginalComputeSize !== undefined) item.computeSize = item.__star7OriginalComputeSize;
        else delete item.computeSize;
    }
}
function refreshMaterialControls(node) {
    migrateMaterialValues(node);
    // Keep saved controls visible when modes are switched. The label makes clear
    // that this value is consumed only by Remix Source mode.
    setHidden(widget(node, "audio_denoise_strength"), false);
    const size = node.computeSize?.();
    if (size) node.setSize?.([Math.max(node.size?.[0] || 0, 390), size[1]]);
    node.setDirtyCanvas?.(true, true);
}
function installMaterialControls(node) {
    const mode = widget(node, "audio_mode");
    if (mode && !mode.__star7MaterialWrapped) {
        mode.__star7MaterialWrapped = true;
        const original = mode.callback;
        mode.callback = (...args) => {
            original?.apply(mode, args);
            refreshMaterialControls(node);
        };
    }
    refreshMaterialControls(node);
}
function readParams(node) {
    return Object.fromEntries(PARAM_NAMES.map((name) => [name, widget(node, name)?.value]));
}
function writeParams(node, values) {
    node.__star7ApplyingPreset = true;
    try {
        for (const [name, value] of Object.entries(values)) {
            const item = widget(node, name);
            if (item) item.value = value;
        }
    } finally { node.__star7ApplyingPreset = false; }
    node.setDirtyCanvas?.(true, true);
}
function ensureResolutionStatus(node, text) {
    let item = node.widgets?.find((candidate) => candidate.__star7FaceResolutionStatus);
    if (!item) {
        item = node.addWidget("text", text.resolutionPending, "", () => {}, { serialize: false });
        item.__star7FaceResolutionStatus = true;
        item.disabled = true;
        item.serialize = false;
        item.serializeValue = async () => undefined;
    }
    return item;
}
function refreshResolutionStatus(node) {
    const lang = language();
    const text = TEXT[lang];
    const item = ensureResolutionStatus(node, text);
    const enabled = widget(node, "enable_refine")?.value !== false;
    const detail = node.__star7FaceResolution;
    if (!enabled) item.name = text.resolutionDisabled;
    else if (!detail) item.name = text.resolutionPending;
    else item.name = lang === "zh"
        ? `修复后分辨率：${detail.width}×${detail.height} · ${detail.megapixels.toFixed(2)} MP`
        : `Repaired resolution: ${detail.width}×${detail.height} · ${detail.megapixels.toFixed(2)} MP`;
    item.value = "";
    node.setDirtyCanvas?.(true, true);
    return item;
}
function updateResolutionStatus(node, detail) {
    const width = Number(detail?.width);
    const height = Number(detail?.height);
    const megapixels = Number(detail?.megapixels);
    if (!Number.isFinite(width) || !Number.isFinite(height) || !Number.isFinite(megapixels)) return;
    node.__star7FaceResolution = { width, height, megapixels };
    refreshResolutionStatus(node);
}
function placeResolutionBeforeReset(node) {
    const status = node.widgets?.find((item) => item.__star7FaceResolutionStatus);
    const reset = node.__star7ResetButton;
    if (!status || !reset) return;
    const statusIndex = node.widgets.indexOf(status);
    const resetIndex = node.widgets.indexOf(reset);
    if (statusIndex < 0 || resetIndex < 0 || statusIndex === resetIndex - 1) return;
    node.widgets.splice(statusIndex, 1);
    const nextResetIndex = node.widgets.indexOf(reset);
    node.widgets.splice(nextResetIndex, 0, status);
}
function migrateLegacyFaceValues(node, values) {
    // The unreleased switch-based layout was:
    // enable, preset, target, parameters..., seed, multi_face, max_faces.
    // The new direct count sits on row two. Remap once so local test workflows do
    // not shift every following widget into the wrong control.
    if (!Array.isArray(values) || typeof values[1] !== "string") return;
    const names = ["enable_refine", "preset", "target_face", "refine_steps", "custom_strength",
        "custom_canvas", "custom_crop_context", "custom_blend", "custom_feather", "seed"];
    names.forEach((name, index) => {
        const item = widget(node, name);
        if (item && values[index] !== undefined) item.value = values[index];
    });
    const count = values[10] === true ? Number(values[11] ?? 3) : 1;
    const countWidget = widget(node, "face_count");
    if (countWidget) countWidget.value = Math.max(1, Math.min(4, Number.isFinite(count) ? count : 1));
}
function installFaceControls(node, text) {
    node.properties ??= {};
    node.properties.star7CustomFaceParams ??= { ...BALANCED };
    const preset = widget(node, "preset");
    const enabled = widget(node, "enable_refine");
    if (enabled && !enabled.__star7ResolutionWrapped) {
        enabled.__star7ResolutionWrapped = true;
        const original = enabled.callback;
        enabled.callback = (...args) => {
            original?.apply(enabled, args);
            node.__star7FaceResolution = null;
            refreshResolutionStatus(node);
        };
    }
    node.__star7LastPreset ??= String(preset?.value || "自动平衡");
    if (preset && !preset.__star7Wrapped) {
        preset.__star7Wrapped = true;
        const original = preset.callback;
        preset.callback = (...args) => {
            original?.apply(preset, args);
            const previous = node.__star7LastPreset;
            const current = String(preset.value || "自动平衡");
            if (previous === "自定义") node.properties.star7CustomFaceParams = readParams(node);
            writeParams(node, current === "自定义" ? node.properties.star7CustomFaceParams : (PRESETS[current] ?? BALANCED));
            node.__star7LastPreset = current;
        };
    }
    for (const name of PARAM_NAMES) {
        const item = widget(node, name);
        if (!item || item.__star7Wrapped) continue;
        item.__star7Wrapped = true;
        const original = item.callback;
        item.callback = (...args) => {
            original?.apply(item, args);
            if (node.__star7ApplyingPreset) return;
            if (preset && preset.value !== "自定义") {
                preset.value = "自定义";
                node.__star7LastPreset = "自定义";
            }
            node.properties.star7CustomFaceParams = readParams(node);
            node.setDirtyCanvas?.(true, true);
        };
    }
    if (!node.__star7ResetButton) {
        const reset = node.addWidget("button", text.reset, null, () => {
            const current = String(preset?.value || "自动平衡");
            const values = current === "自定义" ? BALANCED : (PRESETS[current] ?? BALANCED);
            if (current === "自定义") node.properties.star7CustomFaceParams = { ...BALANCED };
            writeParams(node, values);
        }, { serialize: false });
        reset.serialize = false;
        node.__star7ResetButton = reset;
    }
    refreshResolutionStatus(node);
    placeResolutionBeforeReset(node);
    if (preset?.value === "自定义") writeParams(node, node.properties.star7CustomFaceParams);
    else writeParams(node, PRESETS[preset?.value] ?? BALANCED);
    // Selection still matters when N > 1: it decides which N faces win when
    // more faces are visible than requested. Keep it visible for every count.
    setHidden(widget(node, "target_face"), false);
    const size = node.computeSize?.();
    if (size) node.setSize?.([
        Math.max(node.size?.[0] || 0, 390),
        Math.max(node.size?.[1] || 0, size[1]),
    ]);
}

api.addEventListener("star7-h3-face-repair-resolution", ({ detail }) => {
    const rawId = detail?.node_id;
    if (rawId == null) return;
    const node = app.graph?.getNodeById?.(rawId) ?? app.graph?.getNodeById?.(Number(rawId));
    if (!node || (node.comfyClass !== FACE_NODE && node.type !== FACE_NODE)) return;
    updateResolutionStatus(node, detail);
});

app.registerExtension({
    name: "star7.h3.face-refine",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== FACE_NODE && nodeData.name !== MATERIAL_NODE) return;
        const isFace = nodeData.name === FACE_NODE;
        const lang = language();
        const text = TEXT[lang];
        nodeData.display_name = isFace ? text.faceTitle : text.materialTitle;
        for (const specs of [nodeData.input?.required, nodeData.input?.optional]) {
            for (const [name, spec] of Object.entries(specs ?? {})) {
                const label = labelFor(name, lang);
                if (!label || !Array.isArray(spec)) continue;
                spec[1] ??= {};
                spec[1].display_name = label;
                if (name === "preserve_repair_detail") {
                    spec[1].label_on = lang === "zh" ? "开启" : "On";
                    spec[1].label_off = lang === "zh" ? "关闭" : "Off";
                    spec[1].tooltip = lang === "zh"
                        ? "低于约 1.0MP 时按原比例自动放大，减少二采细节回贴后的压缩；只放大，绝不缩小。"
                        : "Upscales outputs below about 1.0 MP to retain more second-pass detail; never downscales.";
                }
            }
        }
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            localizeNode(this, isFace);
            if (isFace) requestAnimationFrame(() => installFaceControls(this, text));
            else requestAnimationFrame(() => {
                compactMaterialReferenceImages(this);
                placeMaterialReferenceImages(this);
                localizeNode(this, false);
                installMaterialControls(this);
                updateMaterialMediaLabels(this);
            });
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            if (isFace) migrateLegacyFaceValues(this, arguments[0]?.widgets_values);
            requestAnimationFrame(() => {
                localizeNode(this, isFace);
                if (isFace) installFaceControls(this, text);
                else {
                    compactMaterialReferenceImages(this);
                    placeMaterialReferenceImages(this);
                    installMaterialControls(this);
                    updateMaterialMediaLabels(this);
                }
            });
            return result;
        };
        if (!isFace) {
            const connectionsChanged = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function () {
                const result = connectionsChanged?.apply(this, arguments);
                requestAnimationFrame(() => {
                    compactMaterialReferenceImages(this);
                    placeMaterialReferenceImages(this);
                    localizeNode(this, false);
                    updateMaterialMediaLabels(this);
                    this.setDirtyCanvas?.(true, true);
                });
                return result;
            };
        }
    },
});
