import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
let statusHandler;
let localeChangeHandler;
let locale = "zh-CN";
const source = fs.readFileSync(new URL("./web/runtime_status.js", import.meta.url), "utf8")
    .replace(/^import .*?;\r?\n/gm, "");

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
    JSON.stringify([8192, true, true, 4096, 4096, "实验功能已移除", true, "comfy_kitchen_int8"]),
);
for (const [name, value] of Object.entries(restored)) {
    assert.equal(serialized.widgets_values_named[name], value);
}

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

locale = "en-US";
localeChangeHandler();
assert.equal(corrupted.title, "MiniMax H3 VRAM Chunk Acceleration - Star7");
assert.equal(customTitleNode.title, "我的自定义标题");
assert.equal(
    corrupted.widgets.find((widget) => widget.name === "mlp_chunk_tokens")?.label,
    "MLP chunk size (main VRAM control)",
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
console.log("Runtime status localization and workflow restore tests passed");
