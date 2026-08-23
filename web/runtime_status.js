import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAMES = new Set([
    "MiniMaxH3ActivationChunkStar7",
    "MiniMaxH3RoPEChunkPatch",
]);
const REAL_WIDGET_DEFAULTS = {
    chunk_tokens: 8192,
    auto_halve_on_oom: true,
    verbose: true,
    mlp_chunk_tokens: 4096,
    disable_dynamic_prefetch: "实验功能已移除",
    reuse_mlp_weights: true,
    attention_backend: "comfy_kitchen_int8",
};
const REAL_WIDGET_NAMES = Object.keys(REAL_WIDGET_DEFAULTS);

const TEXT = {
    en: {
        title: "MiniMax H3 VRAM Chunk Acceleration - Star7",
        legacyTitle: "MiniMax H3 VRAM Chunk Acceleration (Legacy Workflow)",
        labels: {
            chunk_tokens: "RoPE chunk size",
            auto_halve_on_oom: "Auto-reduce after VRAM error",
            verbose: "Show runtime details",
            mlp_chunk_tokens: "MLP chunk size (main VRAM control)",
            disable_dynamic_prefetch: "Preload next block (experimental feature removed)",
            reuse_mlp_weights: "Reuse MLP weights (faster)",
            attention_backend: "Attention method",
        },
        tooltips: {
            chunk_tokens: "Usually keep 8192. Lower it only when the log specifically reports a RoPE VRAM error.",
            auto_halve_on_oom: "Automatically halves the failing RoPE or MLP chunk and retries instead of stopping the task.",
            verbose: "Shows actual chunk sizes, automatic reductions, and the active MLP weight mode in the console.",
            mlp_chunk_tokens: "The main VRAM control. Smaller values save more VRAM but may be slower. Lower this before RoPE.",
            disable_dynamic_prefetch: "Legacy workflow field only. This experimental feature has been removed and is always disabled.",
            reuse_mlp_weights: "Uses isolated MLP weight snapshots to avoid repeated preparation. Falls back safely if snapshots fail or run out of VRAM.",
            attention_backend: "Strict SLA never falls back. The SM75 All-INT8 option is experimental and may reduce quality; the FP16-PV option remains recommended.",
        },
        current: (label, value) => `${label} in use: ${value} (configured)`,
        limited: (label, value, configured) => `${label} in use: ${value} (set ${configured}, limited by video size)`,
        reduced: (label, value, configured) => `${label} auto-reduced to: ${value} (set ${configured})`,
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
            disable_dynamic_prefetch: "提前加载下一层（实验功能已移除）",
            reuse_mlp_weights: "复用 MLP 权重（提速）",
            attention_backend: "注意力计算方式",
        },
        tooltips: {
            chunk_tokens: "通常保持 8192。它对显存影响相对较小，只有日志明确提示 RoPE 显存不足时才降低。",
            auto_halve_on_oom: "RoPE 或 MLP 分块显存不足时自动减半重试，避免任务直接停止。",
            verbose: "在控制台显示实际分块、是否自动降档以及 MLP 权重加速方式。",
            mlp_chunk_tokens: "主要显存调节项。数值越小越省显存，但可能更慢；显存不足时优先降低它。",
            disable_dynamic_prefetch: "仅为兼容旧工作流保留，不再参与计算，功能始终关闭。",
            reuse_mlp_weights: "使用独立 MLP 权重快照减少重复准备；快照失败或显存不足时会自动切换安全模式。",
            attention_backend: "严格 SLA 绝不回退。SM75 All-INT8 是可能降低质量的实验模式，仍推荐使用 FP16-PV 模式。",
        },
        current: (label, value) => `${label} 实际使用：${value}（设定值）`,
        limited: (label, value, configured) => `${label} 实际使用：${value}（设定 ${configured}，视频规模只需要这么多）`,
        reduced: (label, value, configured) => `${label} 已自动降为：${value}（原设定 ${configured}）`,
        rope: "RoPE",
        mlp: "MLP",
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

function formatValue(label, effective, configured, reason = "active") {
    const text = strings();
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
        ["mlp_chunk_tokens", 4096],
    ]) {
        const widget = node.widgets?.find((item) => item.name === name);
        if (!widget || Number.isFinite(Number(widget.value)) && Number(widget.value) >= 256) {
            continue;
        }
        widget.value = fallback;
        widget.callback?.(fallback);
    }
}

function validSavedValue(name, value) {
    if (name === "chunk_tokens" || name === "mlp_chunk_tokens") {
        return Number.isInteger(Number(value)) && Number(value) >= 256;
    }
    if (name === "attention_backend") {
        return value === "existing"
            || value === "comfy_kitchen_int8"
            || value === "sla_sm75_qk_int8_pv_fp16"
            || value === "sla_sm75_all_int8_experimental"
            || value === "sla_sm80+_qk_int8_pv_fp16";
    }
    if (name === "disable_dynamic_prefetch") {
        // Old files contain a boolean here; it is now a non-functional label.
        return typeof value === "boolean" || typeof value === "string";
    }
    return typeof value === "boolean";
}

function cleanSavedValues(info) {
    const named = info?.widgets_values_named ?? {};
    const positional = Array.isArray(info?.widgets_values)
        && info.widgets_values.length === REAL_WIDGET_NAMES.length
        ? info.widgets_values
        : [];
    const values = REAL_WIDGET_NAMES.map((name, index) => {
        if (name === "disable_dynamic_prefetch") {
            return REAL_WIDGET_DEFAULTS[name];
        }
        const namedValue = named[name];
        if (validSavedValue(name, namedValue)) {
            return namedValue;
        }
        const positionalValue = positional[index];
        if (validSavedValue(name, positionalValue)) {
            return positionalValue;
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
        if (name === "disable_dynamic_prefetch") {
            return REAL_WIDGET_DEFAULTS[name];
        }
        const value = node.widgets?.find((widget) => widget.name === name)?.value;
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
    const configuredRope = Number(
        node.widgets?.find((item) => item.name === "chunk_tokens")?.value ?? 4096,
    );
    const configuredMlp = Number(
        node.widgets?.find((item) => item.name === "mlp_chunk_tokens")?.value ?? 4096,
    );
    localizeNode(node);
    const text = strings();
    const runtime = node.__star7RuntimeDetail;
    const runtimeMatchesInputs = runtime
        && Number(runtime.configured_rope) === configuredRope
        && Number(runtime.configured_mlp) === configuredMlp;
    const effectiveRope = runtimeMatchesInputs
        ? Number(runtime.effective_rope) : configuredRope;
    const effectiveMlp = runtimeMatchesInputs
        ? Number(runtime.effective_mlp) : configuredMlp;
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
    moveAfter(node, rope, "chunk_tokens");
    moveAfter(node, mlp, "mlp_chunk_tokens");
    reorderDisplayWidgets(node);
    return { rope, mlp };
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
        if (NODE_NAMES.has(node.comfyClass) || NODE_NAMES.has(node.type)) {
            ensureStatusWidgets(node);
        }
    },
    loadedGraphNode(node) {
        if (NODE_NAMES.has(node.comfyClass) || NODE_NAMES.has(node.type)) {
            ensureStatusWidgets(node);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
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
                if (NODE_NAMES.has(node.comfyClass) || NODE_NAMES.has(node.type)) {
                    ensureStatusWidgets(node);
                    node.setDirtyCanvas?.(true, true);
                }
            }
        });
    },
});
