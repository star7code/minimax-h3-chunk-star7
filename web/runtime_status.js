import { api } from "/scripts/api.js";
import { app } from "/scripts/app.js";

const NODE_NAMES = new Set([
    "MiniMaxH3ActivationChunkStar7",
    "MiniMaxH3RoPEChunkPatch",
]);

function formatValue(label, effective, configured) {
    return effective === configured
        ? `${label}：当前使用 ${effective}（设定值）`
        : `${label}：已降级为 ${effective}（设定 ${configured}）`;
}

function removeLegacyStatusWidget(node) {
    const index = node.widgets?.findIndex(
        (item) => item.__star7StatusName === "star7_runtime_status"
            || item.name === "star7_runtime_status",
    );
    if (index == null || index < 0) {
        return;
    }
    node.widgets[index]?.onRemove?.();
    node.widgets.splice(index, 1);
}

function makeStatusWidget(node, name, label, value) {
    let widget = node.widgets?.find(
        (item) => item.__star7StatusName === name || item.name === name,
    );
    if (!widget) {
        widget = node.addWidget(
            "text",
            label,
            value,
            () => {},
            { serialize: false },
        );
        widget.__star7StatusName = name;
        widget.disabled = true;
        widget.serializeValue = async () => undefined;
    }
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
    removeLegacyStatusWidget(node);
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
        "RoPE 当前使用",
        formatValue("RoPE", configuredRope, configuredRope),
    );
    const mlp = makeStatusWidget(
        node,
        "star7_mlp_runtime_status",
        "MLP 当前使用",
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
    widgets.rope.value = formatValue(
        "RoPE", detail.effective_rope, detail.configured_rope,
    );
    widgets.mlp.value = formatValue(
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
    },
});
