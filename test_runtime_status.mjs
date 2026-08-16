import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
const source = fs.readFileSync(new URL("./web/runtime_status.js", import.meta.url), "utf8")
    .replace(/^import .*?;\r?\n/gm, "");

const context = {
    api: { addEventListener() {} },
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
console.log("Runtime status workflow restore test passed");
