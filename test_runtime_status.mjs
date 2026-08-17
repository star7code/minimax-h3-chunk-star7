import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
let statusHandler;
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
        this.type = "MiniMaxH3RoPEChunkPatch";
        this.comfyClass = this.type;
        this.widgets = [
            { name: "chunk_tokens", value: 8192 },
            { name: "auto_halve_on_oom", value: true },
            { name: "verbose", value: true },
            { name: "mlp_chunk_tokens", value: 4096 },
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

await extension.beforeRegisterNodeDef(MockNode, { name: "MiniMaxH3RoPEChunkPatch" });
const node = new MockNode();
node.onNodeCreated();
node.configure({
    widgets_values: [
        8192,
        true,
        true,
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
    node.widgets.filter((widget) => widget.__star7StatusName).length,
    2,
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
    disable_dynamic_prefetch: true,
    reuse_mlp_weights: true,
    attention_backend: "comfy_kitchen_int8",
});

const serialized = {};
corrupted.onSerialize(serialized);
assert.equal(
    JSON.stringify(serialized.widgets_values),
    JSON.stringify([8192, true, true, 4096, true, true, "comfy_kitchen_int8"]),
);
assert.equal(JSON.stringify(serialized.widgets_values_named), JSON.stringify(restored));

context.app.graph = { getNodeById: () => corrupted };
statusHandler({
    detail: {
        node_id: "299",
        configured_rope: 8192,
        effective_rope: 4096,
        configured_mlp: 4096,
        effective_mlp: 2048,
    },
});
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_rope_runtime_status",
    ).name,
    /已降级：4096（设定 8192）/,
);
assert.match(
    corrupted.widgets.find(
        (widget) => widget.__star7StatusName === "star7_mlp_runtime_status",
    ).name,
    /已降级：2048（设定 4096）/,
);
console.log("Runtime status workflow restore test passed");
