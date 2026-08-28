import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
let statusHandler;
let localeChangeHandler;
let locale = "zh-CN";
const source = fs.readFileSync(new URL("./web/runtime_status.js", import.meta.url), "utf8")
    .replace(/^import .*?;\r?\n/gm, "");
const promptImporterSource = fs.readFileSync(
    new URL("./web/prompt_importer.js", import.meta.url), "utf8",
);
assert.doesNotMatch(source, /document\.addEventListener\(["'](?:drop|dragover)["']/);
assert.match(promptImporterSource, /promptNodeAtDrop\(event\)/);

const context = {
    api: {
        addEventListener(name, handler) {
            if (name === "star7-h3-chunk-status") {
                statusHandler = handler;
            }
        },
    },
    app: {
        ui: {
            settings: {
                getSettingValue(name) {
                    assert.equal(name, "Comfy.Locale");
                    return locale;
                },
                addEventListener(name, handler) {
                    assert.equal(name, "Comfy.Locale.change");
                    localeChangeHandler = handler;
                },
            },
        },
        registerExtension(value) {
            extension = value;
        },
    },
    Set,
    Number,
};
vm.runInNewContext(source, context);

class MockNode {
    constructor() {
        this.type = "MiniMaxH3ActivationChunkStar7";
        this.comfyClass = this.type;
        this.inputs = [{ name: "model" }];
        this.outputs = [{ name: "model" }];
        this.widgets = [
            { name: "chunk_tokens", value: 8192 },
            { name: "auto_halve_on_oom", value: true },
            { name: "verbose", value: true },
            { name: "mlp_chunk_tokens", value: 4096 },
            { name: "qkv_chunk_tokens", value: 4096 },
            { name: "disable_dynamic_prefetch", value: true },
            { name: "reuse_mlp_weights", value: true },
            { name: "attention_backend", value: "existing" },
        ];
    }

    addWidget(type, name, value, callback, options) {
        const widget = { type, name, value, callback, options };
        this.widgets.push(widget);
        return widget;
    }

    onNodeCreated() {}

    configure(info) {
        // Simulate legacy ComfyUI restoring every widget by array position.
        info.widgets_values.forEach((value, index) => {
            if (this.widgets[index]) {
                this.widgets[index].value = value;
            }
        });
    }
}

const nodeData = {
    name: "MiniMaxH3ActivationChunkStar7",
    input: {
        required: Object.fromEntries(Object.entries({
            chunk_tokens: "INT",
            auto_halve_on_oom: "BOOLEAN",
            verbose: "BOOLEAN",
            mlp_chunk_tokens: "INT",
            qkv_chunk_tokens: "INT",
            disable_dynamic_prefetch: "BOOLEAN",
            reuse_mlp_weights: "BOOLEAN",
            attention_backend: ["existing", "comfy_kitchen_int8"],
        }).map(([name, type]) => [name, [type, {}]])),
    },
};
await extension.beforeRegisterNodeDef(MockNode, nodeData);
extension.setup();

assert.equal(nodeData.display_name, "MiniMax H3 显存分块加速 - Star7");
assert.equal(
    nodeData.input.required.mlp_chunk_tokens[1].display_name,
    "MLP 分块大小（主要显存调节）",
);
assert.equal(
    nodeData.input.required.auto_halve_on_oom[1].label_on,
    "开启",
);
const node = new MockNode();
node.onNodeCreated();
node.configure({
    widgets_values: [
        8192,
        true,
        true,
        4096,
        4096,
        true,
        true,
        "comfy_kitchen_int8",
    ],
});

assert.equal(
    node.widgets.find((widget) => widget.name === "attention_backend")?.value,
    "comfy_kitchen_int8",
);
assert.equal(
    node.widgets.find((widget) => widget.name === "disable_dynamic_prefetch")?.value,
    "实验功能已移除",
);
assert.equal(node.title, "MiniMax H3 显存分块加速 - Star7");
assert.equal(
    node.widgets.find((widget) => widget.name === "mlp_chunk_tokens")?.label,
    "MLP 分块大小（主要显存调节）",
);
assert.equal(
    node.widgets.find((widget) => widget.name === "disable_dynamic_prefetch")?.label,
    "提前加载下一层（实验功能已移除）",
);
assert.deepEqual(
    node.widgets.filter((widget) => !widget.__star7StatusName).map((widget) => widget.name),
    [
        "mlp_chunk_tokens",
        "qkv_chunk_tokens",
        "chunk_tokens",
        "auto_halve_on_oom",
        "verbose",
        "reuse_mlp_weights",
        "disable_dynamic_prefetch",
        "attention_backend",
    ],
);
assert.equal(
    node.widgets.filter((widget) => widget.__star7StatusName).length,
    3,
);
assert.match(
    node.widgets.find((widget) => widget.__star7StatusName === "star7_rope_runtime_status").name,
    /8192/,
);

const corrupted = new MockNode();
corrupted.onNodeCreated();
corrupted.configure({
    widgets_values: [
        8192,
        "RoPE status",
        true,
        true,
        4096,
        "MLP status",
        4096,
        true,
        "comfy_kitchen_int8",
    ],
    widgets_values_named: {
        chunk_tokens: 8192,
        "RoPE 当前使用": "RoPE status",
        auto_halve_on_oom: "RoPE status",
        verbose: "RoPE status",
        mlp_chunk_tokens: 4096,
        "MLP 当前使用": "MLP status",
        disable_dynamic_prefetch: 4096,
        reuse_mlp_weights: "MLP status",
        attention_backend: "comfy_kitchen_int8",
    },
});

const restored = Object.fromEntries(
    corrupted.widgets
        .filter((widget) => !widget.__star7StatusName)
        .map((widget) => [widget.name, widget.value]),
);
assert.deepEqual(restored, {
    chunk_tokens: 8192,
    auto_halve_on_oom: true,
    verbose: true,
    mlp_chunk_tokens: 4096,
    qkv_chunk_tokens: 4096,
    disable_dynamic_prefetch: "实验功能已移除",
    reuse_mlp_weights: true,
    attention_backend: "comfy_kitchen_int8",
});

const serialized = {};
corrupted.onSerialize(serialized);
assert.equal(
    JSON.stringify(serialized.widgets_values),
    JSON.stringify([8192, true, true, 4096, 4096, "Experimental feature removed", true, "comfy_kitchen_int8"]),
);
assert.equal(
    JSON.stringify(serialized.widgets_values_named),
    JSON.stringify({
        chunk_tokens: 8192,
        auto_halve_on_oom: true,
        verbose: true,
        mlp_chunk_tokens: 4096,
        qkv_chunk_tokens: 4096,
        disable_dynamic_prefetch: "Experimental feature removed",
        reuse_mlp_weights: true,
        attention_backend: "comfy_kitchen_int8",
    }),
);

const customTitleNode = new MockNode();
customTitleNode.title = "我的自定义标题";
customTitleNode.onNodeCreated();
assert.equal(customTitleNode.title, "我的自定义标题");

context.app.graph = {
    _nodes: [corrupted, customTitleNode],
    getNodeById: () => corrupted,
};
statusHandler({
    detail: {
        node_id: "299",
        configured_rope: 8192,
        effective_rope: 4096,
        configured_mlp: 4096,
        effective_mlp: 2048,
        configured_qkv: 4096,
        effective_qkv: 4096,
    },
});
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_rope_runtime_status",
    ).name,
    /已自动降为：4096（原设定 8192）/,
);
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_mlp_runtime_status",
    ).name,
    /已自动降为：2048（原设定 4096）/,
);

corrupted.widgets.find((widget) => widget.name === "chunk_tokens").value = 0;
corrupted.widgets.find((widget) => widget.name === "qkv_chunk_tokens").value = 0;
statusHandler({
    detail: {
        node_id: "299",
        configured_rope: 0,
        effective_rope: 103546,
        configured_mlp: 4096,
        effective_mlp: 4096,
        configured_qkv: 0,
        effective_qkv: 51773,
        reason: "qkv_oom",
    },
});
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_rope_runtime_status",
    ).name,
    /RoPE：整段计算（未固定分块）/,
);
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_qkv_runtime_status",
    ).name,
    /QKV 已从整段自动降为：51773/,
);
corrupted.widgets.find((widget) => widget.name === "chunk_tokens").value = 8192;
corrupted.widgets.find((widget) => widget.name === "qkv_chunk_tokens").value = 4096;
statusHandler({
    detail: {
        node_id: "299",
        configured_rope: 8192,
        effective_rope: 4096,
        configured_mlp: 4096,
        effective_mlp: 2048,
        configured_qkv: 4096,
        effective_qkv: 4096,
    },
});

locale = "en-US";
localeChangeHandler();
assert.equal(corrupted.title, "MiniMax H3 VRAM Chunk Acceleration - Star7");
assert.equal(customTitleNode.title, "我的自定义标题");
assert.equal(
    corrupted.widgets.find((widget) => widget.name === "mlp_chunk_tokens")?.label,
    "MLP chunk size (main VRAM control)",
);
assert.equal(
    corrupted.widgets.find((widget) => widget.name === "disable_dynamic_prefetch")?.value,
    "Experimental feature removed",
);
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_rope_runtime_status",
    ).name,
    /auto-reduced to: 4096 \(set 8192\)/,
);

locale = "zh-CN";
localeChangeHandler();

statusHandler({
    detail: {
        node_id: "299",
        configured_rope: 8192,
        effective_rope: 2048,
        configured_mlp: 4096,
        effective_mlp: 2048,
        configured_qkv: 4096,
        effective_qkv: 4096,
        reason: "sequence_limit",
    },
});
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_rope_runtime_status",
    ).name,
    /实际使用：2048（设定 8192，视频规模只需要这么多）/,
);
assert.equal(
    corrupted.widgets.find((widget) => widget.name === "attention_backend")?.value,
    "comfy_kitchen_int8",
);

class MockReferenceNode {
    constructor() {
        this.type = "MiniMaxH3ReferenceVideoLoadStar7";
        this.comfyClass = this.type;
        this.size = [420, 500];
        const videoEl = {
            tagName: "VIDEO",
            style: {},
            listeners: {},
            addEventListener(name, callback) { this.listeners[name] = callback; },
        };
        const imgEl = {
            tagName: "IMG",
            style: {},
            listeners: {},
            addEventListener(name, callback) { this.listeners[name] = callback; },
        };
        const previewClasses = new Set();
        const previewSurface = {
            style: {},
            classList: {
                add(name) { previewClasses.add(name); },
                contains(name) { return previewClasses.has(name); },
            },
        };
        const previewElement = {
            hidden: false,
            style: {},
            parentElement: previewSurface,
            classList: {
                add(name) { previewClasses.add(name); },
                contains(name) { return previewClasses.has(name); },
            },
            closest(selector) { return selector === ".dom-widget" ? previewSurface : null; },
            querySelectorAll(selector) { return selector === "video, img" ? [videoEl, imgEl] : []; },
        };
        this.previewClasses = previewClasses;
        this.previewSurface = previewSurface;
        this.previewElement = previewElement;
        this.videoEl = videoEl;
        this.imgEl = imgEl;
        this.widgets = [
            { type: "combo", name: "video", value: "reference.mp4" },
            { type: "number", name: "max_long_edge", value: 720 },
            { type: "toggle", name: "allow_upscale", value: false },
            { type: "toggle", name: "trim_enabled", value: false },
            { type: "number", name: "trim_start_seconds", value: 0.0 },
            { type: "number", name: "trim_end_seconds", value: 0.0 },
            {
                type: "preview",
                name: "video-preview",
                element: previewElement,
                computeSize(width) { return [width, 260]; },
            },
        ];
    }

    onNodeCreated() {}

    configure(info) {
        if (Array.isArray(info.size)) this.size = [...info.size];
        info.widgets_values?.forEach((value, index) => {
            if (this.widgets[index]) this.widgets[index].value = value;
        });
    }

    computeSize(minimum = [0, 0]) {
        const widgetHeight = this.widgets.reduce((total, widget) => {
            const computed = widget.computeSize?.();
            const layout = !computed ? widget.computeLayoutSize?.(this) : null;
            return total + (computed ? computed[1] : (layout?.minHeight ?? 24)) + 4;
        }, 0);
        return [
            Math.max(Number(minimum?.[0]) || 0, 320),
            Math.max(Number(minimum?.[1]) || 0, 80 + widgetHeight),
        ];
    }

    setSize(size) {
        this.size = [...size];
        this.onResize?.(this.size);
    }

    resizeFromCanvas(size) {
        const minimum = this.computeSize();
        this.setSize([
            Math.max(size[0], minimum[0]),
            Math.max(size[1], minimum[1]),
        ]);
    }
}

const referenceNodeData = {
    name: "MiniMaxH3ReferenceVideoLoadStar7",
    input: {
        required: {
            video: [["reference.mp4"], {}],
            max_long_edge: ["INT", {}],
            allow_upscale: ["BOOLEAN", {}],
            trim_enabled: ["BOOLEAN", {}],
            trim_start_seconds: ["FLOAT", {}],
            trim_end_seconds: ["FLOAT", {}],
        },
    },
};
await extension.beforeRegisterNodeDef(MockReferenceNode, referenceNodeData);
const lateReferenceNode = new MockReferenceNode();
const latePreviewWidget = lateReferenceNode.widgets.pop();
lateReferenceNode.onNodeCreated();
assert.equal(lateReferenceNode.previewSurface.style.pointerEvents, undefined);
lateReferenceNode.widgets.push(latePreviewWidget);
lateReferenceNode.onDrawBackground();
assert.equal(lateReferenceNode.previewSurface.style.pointerEvents, "none");
const referenceNode = new MockReferenceNode();
referenceNode.onNodeCreated();
assert.equal(referenceNode.title, "参考视频载入 - Star7");
assert.equal(referenceNodeData.input.required.trim_enabled[1].label_on, "开启");
assert.equal(referenceNodeData.input.required.trim_enabled[1].label_off, "关闭");
assert.match(
    referenceNode.widgets.find((widget) => widget.name === "trim_start_seconds").type,
    /^converted-widget/,
);
const collapsedHeight = referenceNode.size[1];
const previewWidget = referenceNode.widgets.find(
    (widget) => widget.name === "video-preview",
);
assert.equal(previewWidget.computeSize, undefined);
assert.equal(previewWidget.computeLayoutSize(referenceNode).minHeight, 120);
const collapsedMinimumHeight = referenceNode.computeSize()[1];
assert.ok(collapsedMinimumHeight < referenceNode.size[1]);
const trimToggle = referenceNode.widgets.find((widget) => widget.name === "trim_enabled");
trimToggle.value = true;
trimToggle.callback(true);
assert.equal(
    referenceNode.widgets.find((widget) => widget.name === "trim_start_seconds").type,
    "number",
);
assert.equal(referenceNode.size[1], collapsedHeight);
assert.ok(referenceNode.computeSize()[1] > collapsedMinimumHeight);
assert.ok(referenceNode.computeSize()[1] < referenceNode.size[1]);
assert.equal(referenceNode.videoEl.style.objectFit, "contain");
assert.equal(referenceNode.previewSurface.style.pointerEvents, "none");
assert.equal(referenceNode.videoEl.style.pointerEvents, "auto");
assert.equal(referenceNode.imgEl.style.pointerEvents, "none");
assert.equal(referenceNode.previewElement.style.paddingRight, "10px");
assert.equal(referenceNode.previewElement.style.paddingBottom, "10px");
assert.ok(referenceNode.previewClasses.has("star7-reference-preview-pass-through"));
context.applyReferenceVideoDuration(referenceNode, 20.2, true);
assert.equal(referenceNode.widgets.find((widget) => widget.name === "trim_start_seconds").value, 0);
assert.equal(referenceNode.widgets.find((widget) => widget.name === "trim_end_seconds").value, 20.2);
referenceNode.widgets.find((widget) => widget.name === "trim_start_seconds").value = 12.0;
referenceNode.widgets.find((widget) => widget.name === "trim_start_seconds").callback?.(12.0);
assert.equal(referenceNode.widgets.find((widget) => widget.name === "trim_end_seconds").value, 20.2);
referenceNode.configure({
    size: [560, 640],
    widgets_values: ["reference.mp4", 1344, false, false, 2.0, 9.0],
});
assert.deepEqual(referenceNode.size, [560, 640]);
assert.equal(referenceNode.widgets.find((widget) => widget.name === "max_long_edge").value, 1344);
assert.equal(referenceNode.widgets.find((widget) => widget.name === "trim_end_seconds").value, 9.0);
assert.match(
    referenceNode.widgets.find((widget) => widget.name === "trim_end_seconds").type,
    /^converted-widget/,
);
referenceNode.configure({
    size: [560, 640],
    widgets_values: ["reference.mp4", 1344, false, true, 2.0, 9.0],
});
assert.equal(
    referenceNode.widgets.find((widget) => widget.name === "trim_start_seconds").type,
    "number",
);
assert.deepEqual(referenceNode.size, [560, 640]);
referenceNode.resizeFromCanvas([400, 420]);
assert.deepEqual(referenceNode.size, [400, 420]);
referenceNode.resizeFromCanvas([700, 1100]);
assert.deepEqual(referenceNode.size, [700, 1100]);
referenceNode.resizeFromCanvas([700, 720]);
assert.deepEqual(referenceNode.size, [700, 720]);
assert.deepEqual([...referenceNode.__star7ReferenceFrameSize], [700, 720]);
referenceNode.videoEl.listeners.loadedmetadata();
referenceNode.resizeFromCanvas([700, 1100]);
await Promise.resolve();
assert.deepEqual(referenceNode.size, [700, 720]);
assert.deepEqual([...referenceNode.__star7ReferenceFrameSize], [700, 720]);
console.log("Runtime status localization and workflow restore tests passed");
