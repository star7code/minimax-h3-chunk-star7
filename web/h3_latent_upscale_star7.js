import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE = "MiniMaxH3OneClickHDStar7";
const PRESETS = {
    "平衡高清": { target_megapixels: 1.0, refine_steps: 2, refine_strength: 0.25 },
    "高质量": { target_megapixels: 1.0, refine_steps: 3, refine_strength: 0.30 },
    "远景小脸": { target_megapixels: 1.0, refine_steps: 3, refine_strength: 0.36 },
    "高速运动": { target_megapixels: 1.0, refine_steps: 2, refine_strength: 0.18 },
};
const BASE_PRESETS = {
    "平衡高清": { target_megapixels: 1.0, refine_steps: 4, refine_strength: 0.20 },
    "高质量": { target_megapixels: 1.0, refine_steps: 6, refine_strength: 0.25 },
    "远景小脸": { target_megapixels: 1.0, refine_steps: 5, refine_strength: 0.30 },
    "高速运动": { target_megapixels: 1.0, refine_steps: 4, refine_strength: 0.15 },
};
const BALANCED = { ...PRESETS["平衡高清"] };
const PARAMS = Object.keys(BALANCED);
const HD_CONTROLS = ["upscale_model", "preset", "target_megapixels", "refine_steps", "refine_strength", "seed", "second_pass_attention"];
const TILE_CONTROLS = ["tile_count", "tile_overlap"];
const SAVED_DEFAULTS = {
    enable_hd: true, upscale_model: "minimax_h3_latent_upscaler_3d_fp16.safetensors",
    preset: "平衡高清", target_megapixels: 1.0,
    refine_steps: 2, refine_strength: 0.25, seed: 0,
    enable_tiling: false, tile_count: 2, tile_overlap: 128,
    second_pass_attention: "继承一采",
};

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function compactModelName(value) {
    const name = String(value ?? "");
    return name.length <= 25 ? name : `${name.slice(0, 8)}…${name.slice(-19)}`;
}

function installCompactModelPainter(item) {
    if (!item || item.__star7HDCompactPainter) return;
    item.__star7HDCompactPainter = true;
    // Draw only the collapsed row ourselves. The combo's native click handler
    // remains intact and receives the untouched values list, so its popup shows
    // the complete checkpoint filename.
    item.draw = function (ctx, node, width, y, height) {
        const margin = 15;
        const disabled = Boolean(this.computedDisabled ?? this.disabled);
        ctx.save();
        ctx.globalAlpha *= disabled ? 0.5 : 1;
        ctx.fillStyle = "#222";
        ctx.strokeStyle = globalThis.LiteGraph?.WIDGET_OUTLINE_COLOR ?? "#666";
        ctx.beginPath();
        ctx.roundRect(margin, y, width - margin * 2, height, height * 0.5);
        ctx.fill();
        if (!disabled) ctx.stroke();

        ctx.fillStyle = globalThis.LiteGraph?.WIDGET_SECONDARY_TEXT_COLOR ?? "#AAA";
        ctx.textAlign = "left";
        ctx.fillText(this.label || this.name, margin * 2 + 5, y + height * 0.7);

        ctx.fillStyle = globalThis.LiteGraph?.WIDGET_TEXT_COLOR ?? "#DDD";
        ctx.textAlign = "right";
        const right = width - margin * 2 - 15;
        ctx.fillText(compactModelName(this.value), right, y + height * 0.7);

        ctx.beginPath();
        ctx.moveTo(margin + 16, y + 5);
        ctx.lineTo(margin + 6, y + height * 0.5);
        ctx.lineTo(margin + 16, y + height - 5);
        ctx.fill();
        ctx.beginPath();
        ctx.moveTo(width - margin - 16, y + 5);
        ctx.lineTo(width - margin - 6, y + height * 0.5);
        ctx.lineTo(width - margin - 16, y + height - 5);
        ctx.fill();
        ctx.restore();
    };
}

function language() {
    const locale = app.ui?.settings?.getSettingValue?.("Comfy.Locale")
        ?? globalThis.navigator?.language ?? "en";
    return String(locale).toLowerCase().startsWith("zh") ? "zh" : "en";
}

function readParams(node) {
    return Object.fromEntries(PARAMS.map((name) => {
        const value = widget(node, name)?.value;
        return [name, typeof value === "number" && Number.isFinite(value) ? value : BALANCED[name]];
    }));
}

function validSavedValue(name, value) {
    if (name === "enable_hd" || name === "enable_tiling") return typeof value === "boolean";
    if (name === "preset") return Object.hasOwn(PRESETS, value) || value === "自定义";
    if (name === "second_pass_attention" || name === "upscale_model") {
        return typeof value === "string" && value.length > 0;
    }
    return typeof value === "number" && Number.isFinite(value);
}

function snapshotInputs(node) {
    const result = {};
    for (const [name, fallback] of Object.entries(SAVED_DEFAULTS)) {
        const value = widget(node, name)?.value;
        result[name] = validSavedValue(name, value) ? value : fallback;
    }
    return result;
}

function repairInvalidInputs(node) {
    const saved = node.properties?.star7HDSavedValues ?? {};
    for (const [name, fallback] of Object.entries(SAVED_DEFAULTS)) {
        const item = widget(node, name);
        if (!item) continue;
        if (validSavedValue(name, item.value)) continue;
        const savedValue = saved[name];
        item.value = validSavedValue(name, savedValue) ? savedValue : fallback;
    }
    const custom = node.properties?.star7CustomHDParams ?? {};
    node.properties.star7CustomHDParams = Object.fromEntries(PARAMS.map((name) => [
        name,
        typeof custom[name] === "number" && Number.isFinite(custom[name])
            ? custom[name] : BALANCED[name],
    ]));
}

function presetValues(node, name) {
    const table = String(node.__star7HDProfile ?? "").startsWith("Base")
        ? BASE_PRESETS : PRESETS;
    return table[name] ?? BALANCED;
}

function writeParams(node, values) {
    node.__star7ApplyingHDPreset = true;
    try {
        for (const [name, value] of Object.entries(values)) {
            const item = widget(node, name);
            if (item) item.value = value;
        }
    } finally {
        node.__star7ApplyingHDPreset = false;
    }
    node.setDirtyCanvas?.(true, true);
}

function ensureSigmaStatus(node, lang) {
    let item = node.widgets?.find((candidate) => candidate.__star7HDSigmaStatus);
    if (!item) {
        item = node.addWidget(
            "text",
            lang === "zh" ? "实际 Sigma：等待运行" : "Actual sigmas: run to calculate",
            "",
            () => {},
            { serialize: false },
        );
        item.__star7HDSigmaStatus = true;
        item.disabled = true;
        item.serialize = false;
        item.serializeValue = async () => undefined;
    }
    return item;
}

function invalidateSigmaStatus(node) {
    const lang = language();
    const item = ensureSigmaStatus(node, lang);
    item.name = lang === "zh" ? "实际 Sigma：等待运行" : "Actual sigmas: run to calculate";
    item.value = "";
    node.setDirtyCanvas?.(true, true);
}

function setWidgetDisabled(item, disabled) {
    if (!item) return;
    item.disabled = Boolean(disabled);
    item.options ??= {};
    item.options.disabled = Boolean(disabled);
}

function refreshEnabledState(node) {
    const enabled = Boolean(widget(node, "enable_hd")?.value);
    const tiling = enabled && Boolean(widget(node, "enable_tiling")?.value);
    for (const name of HD_CONTROLS) setWidgetDisabled(widget(node, name), !enabled);
    setWidgetDisabled(widget(node, "enable_tiling"), !enabled);
    for (const name of TILE_CONTROLS) setWidgetDisabled(widget(node, name), !tiling);
    if (!enabled) {
        const lang = language();
        const status = ensureSigmaStatus(node, lang);
        status.name = lang === "zh" ? "实际 Sigma：未启用二采" : "Actual sigmas: refine disabled";
        status.value = "";
    }
    node.setDirtyCanvas?.(true, true);
}

function updateSigmaStatus(node, detail) {
    const lang = language();
    const item = ensureSigmaStatus(node, lang);
    const raw = String(detail?.sigmas ?? "off");
    const sequence = raw === "off" ? (lang === "zh" ? "未启用二采" : "refine disabled")
        : raw.split(",").join(" → ");
    const shift = Number(detail?.shift);
    const suffix = Number.isFinite(shift) ? ` · Shift ${shift}` : "";
    const profile = String(detail?.profile ?? "");
    node.__star7HDProfile = profile;
    const profileSuffix = profile ? ` · ${profile}` : "";
    const actualSteps = Number(detail?.steps);
    const actualStrength = Number(detail?.strength);
    if (Number.isFinite(actualSteps) && Number.isFinite(actualStrength)) {
        node.__star7ApplyingHDPreset = true;
        try {
            const steps = widget(node, "refine_steps");
            const strength = widget(node, "refine_strength");
            if (steps) steps.value = actualSteps;
            if (strength) strength.value = actualStrength;
        } finally {
            node.__star7ApplyingHDPreset = false;
        }
    }
    item.name = lang === "zh"
        ? `实际 Sigma：${sequence}${suffix}${profileSuffix}`
        : `Actual sigmas: ${sequence}${suffix}${profileSuffix}`;
    item.value = "";
    node.setDirtyCanvas?.(true, true);
}

function placeDynamicWidgets(node) {
    const status = node.widgets?.find((item) => item.__star7HDSigmaStatus);
    const reset = node.__star7HDReset;
    if (!status || !reset) return;
    const statusIndex = node.widgets.indexOf(status);
    if (statusIndex >= 0) node.widgets.splice(statusIndex, 1);
    // Keep every non-serializing display/button after all native inputs. Putting
    // one between native widgets can shift positional workflow values to null.
    const insertAt = node.widgets.indexOf(reset);
    node.widgets.splice(insertAt, 0, status);
}

function install(node) {
    const lang = language();
    node.title = lang === "zh"
        ? "MiniMax H3 一键高清放大 - Star7"
        : "MiniMax H3 One-click HD Upscale - Star7";
    const labels = lang === "zh" ? {
        enable_hd: "启用高清二采",
        sampled_av_latent: "采样结果", h3_context: "采样上下文",
        upscale_model: "高清放大模型", preset: "高清预设",
        target_megapixels: "目标像素（MP）", refine_steps: "高清修复步数",
        refine_strength: "高清修复强度", seed: "高清种子",
        enable_tiling: "启用分格", tile_count: "目标分格数量（智能）", tile_overlap: "分格重叠（像素）",
        second_pass_attention: "二采注意力",
        hd_av_latent: "采样结果", report: "运行报告",
    } : {
        enable_hd: "Enable HD second pass",
        sampled_av_latent: "Sampled result", h3_context: "Sampling context",
        upscale_model: "Latent upscaler model", preset: "HD preset",
        target_megapixels: "Target megapixels", refine_steps: "HD refine steps",
        refine_strength: "HD refine strength", seed: "HD seed",
        enable_tiling: "Enable tiling", tile_count: "Target tiles (smart)", tile_overlap: "Tile overlap (pixels)",
        second_pass_attention: "Second-pass attention",
        hd_av_latent: "Sampled result", report: "Run report",
    };
    for (const item of [...(node.inputs ?? []), ...(node.outputs ?? []), ...(node.widgets ?? [])]) {
        if (labels[item.name]) item.label = item.localized_name = labels[item.name];
    }
    // Keep the real checkpoint filename everywhere; only the closed row paints
    // a shortened copy so the localized title remains fully visible.
    const modelWidget = widget(node, "upscale_model");
    installCompactModelPainter(modelWidget);

    node.properties ??= {};
    repairInvalidInputs(node);
    node.properties.star7HDSavedValues ??= snapshotInputs(node);
    if (!node.__star7HDPersistenceWrapped) {
        node.__star7HDPersistenceWrapped = true;
        const originalConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = originalConfigure?.apply(this, args);
            repairInvalidInputs(this);
            this.__star7LastHDPreset = String(widget(this, "preset")?.value || "平衡高清");
            refreshEnabledState(this);
            return result;
        };
        const originalSerialize = node.onSerialize;
        node.onSerialize = function (...args) {
            repairInvalidInputs(this);
            this.properties.star7HDSavedValues = snapshotInputs(this);
            if (String(widget(this, "preset")?.value) === "自定义") {
                this.properties.star7CustomHDParams = readParams(this);
            }
            return originalSerialize?.apply(this, args);
        };
    }
    const preset = widget(node, "preset");
    for (const toggleName of ["enable_hd", "enable_tiling"]) {
        const toggle = widget(node, toggleName);
        if (!toggle || toggle.__star7HDWrapped) continue;
        toggle.__star7HDWrapped = true;
        const original = toggle.callback;
        toggle.callback = (...args) => {
            original?.apply(toggle, args);
            refreshEnabledState(node);
            invalidateSigmaStatus(node);
            refreshEnabledState(node);
            node.properties.star7HDSavedValues = snapshotInputs(node);
        };
    }
    node.__star7LastHDPreset ??= String(preset?.value || "平衡高清");
    if (preset && !preset.__star7HDWrapped) {
        preset.__star7HDWrapped = true;
        const original = preset.callback;
        preset.callback = (...args) => {
            original?.apply(preset, args);
            const previous = node.__star7LastHDPreset;
            const current = String(preset.value || "平衡高清");
            if (previous === "自定义") node.properties.star7CustomHDParams = readParams(node);
            writeParams(node, current === "自定义"
                ? node.properties.star7CustomHDParams
                : presetValues(node, current));
            node.__star7LastHDPreset = current;
            node.properties.star7HDSavedValues = snapshotInputs(node);
            invalidateSigmaStatus(node);
        };
    }
    for (const name of PARAMS) {
        const item = widget(node, name);
        if (!item || item.__star7HDWrapped) continue;
        item.__star7HDWrapped = true;
        const original = item.callback;
        item.callback = (...args) => {
            original?.apply(item, args);
            if (node.__star7ApplyingHDPreset) return;
            const steps = widget(node, "refine_steps");
            const strength = widget(node, "refine_strength");
            if (name === "refine_steps") {
                if (Number(item.value) > 0 && Number(strength?.value) <= 0) strength.value = 0.18;
                if (Number(item.value) <= 0 && strength) strength.value = 0.0;
            } else if (name === "refine_strength") {
                if (Number(item.value) > 0 && Number(steps?.value) <= 0) steps.value = 1;
                if (Number(item.value) <= 0 && steps) steps.value = 0;
            }
            if (preset && preset.value !== "自定义") {
                preset.value = "自定义";
                node.__star7LastHDPreset = "自定义";
            }
            node.properties.star7CustomHDParams = readParams(node);
            node.properties.star7HDSavedValues = snapshotInputs(node);
            invalidateSigmaStatus(node);
            node.setDirtyCanvas?.(true, true);
        };
    }
    if (!node.__star7HDReset) {
        const reset = node.addWidget(
            "button", lang === "zh" ? "重置参数" : "Reset parameters", null,
            () => {
                const current = String(preset?.value || "平衡高清");
                const values = current === "自定义" ? presetValues(node, "平衡高清")
                    : presetValues(node, current);
                if (current === "自定义") node.properties.star7CustomHDParams = { ...values };
                writeParams(node, values);
            },
            { serialize: false },
        );
        reset.serialize = false;
        node.__star7HDReset = reset;
    }
    ensureSigmaStatus(node, lang);
    placeDynamicWidgets(node);
    refreshEnabledState(node);
    node.setSize?.([Math.max(node.size?.[0] || 0, 390), node.computeSize?.()[1] || node.size?.[1]]);
}

api.addEventListener("star7-h3-hd-sigmas", ({ detail }) => {
    const rawId = detail?.node_id;
    if (rawId == null) return;
    const node = app.graph?.getNodeById?.(rawId) ?? app.graph?.getNodeById?.(Number(rawId));
    if (!node || (node.comfyClass !== NODE && node.type !== NODE)) return;
    updateSigmaStatus(node, detail);
});

app.registerExtension({
    name: "Star7.H3OneClickHD",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE) return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = original?.apply(this, args);
            install(this);
            return result;
        };
    },
});
