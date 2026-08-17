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
    disable_dynamic_prefetch: true,
    reuse_mlp_weights: true,
    attention_backend: "comfy_kitchen_int8",
};
const REAL_WIDGET_NAMES = Object.keys(REAL_WIDGET_DEFAULTS);

function formatValue(label, effective, configured) {
    return effective === configured
        ? `${label} 当前使用：${effective}（设定值）`
        : `${label} 已降级：${effective}（设定 ${configured}）`;
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
        return value === "existing" || value === "comfy_kitchen_int8";
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
        const value = node.widgets?.find((widget) => widget.name === name)?.value;
        return validSavedValue(name, value) ? value : REAL_WIDGET_DEFAULTS[name];
    });
    info.widgets_values = values;
    info.widgets_values_named = Object.fromEntries(
        REAL_WIDGET_NAMES.map((name, index) => [name, values[index]]),
    );
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
    const rope = makeStatusWidget(
        node,
        "star7_rope_runtime_status",
        formatValue("RoPE", configuredRope, configuredRope),
    );
    const mlp = makeStatusWidget(
        node,
        "star7_mlp_runtime_status",
        formatValue("MLP", configuredMlp, configuredMlp),
    );
    moveAfter(node, rope, "chunk_tokens");
    moveAfter(node, mlp, "mlp_chunk_tokens");
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
    const widgets = ensureStatusWidgets(node);
    widgets.rope.name = formatValue(
        "RoPE", detail.effective_rope, detail.configured_rope,
    );
    widgets.mlp.name = formatValue(
        "MLP", detail.effective_mlp, detail.configured_mlp,
    );
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
});
