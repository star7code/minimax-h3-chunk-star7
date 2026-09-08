import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
const app = { registerExtension(value) { extension = value; } };
const path = new URL("./web/dlss_neural_enhance.js", import.meta.url);
const source = fs.readFileSync(path, "utf8").replace(
    'import { app } from "/scripts/app.js";',
    "const app = globalThis.__star7TestApp;",
);
vm.runInNewContext(source, { __star7TestApp: app });
assert.ok(extension);

class NodeType {}
await extension.beforeRegisterNodeDef(NodeType, { name: "Star7DLSSNeuralEnhance" });

const defaults = {
    "风格预设": "真实风格",
    "NR 强度": 0.90,
    "局部结构": 0.70,
    "局部色调": 0.75,
    "皮肤结构": -1.0,
    "时序稳定": 0.55,
    "自动蒙版": true,
};
const node = Object.create(NodeType.prototype);
node.widgets = Object.entries(defaults).map(([name, value]) => ({ name, value }));
node.properties = {};
node.graph = { change() {}, setDirtyCanvas() {} };
node.setDirtyCanvas = () => {};
node.addWidget = (type, name, value, callback, options) => {
    const widget = { type, name, value, callback, options };
    node.widgets.push(widget);
    return widget;
};
node.onNodeCreated();

const widgets = Object.fromEntries(node.widgets.map((widget) => [widget.name, widget]));
widgets["NR 强度"].value = 0.2;
widgets["NR 强度"].callback(0.2);
assert.equal(widgets["风格预设"].value, "自定义参数");
widgets["重置参数"].callback();
assert.equal(widgets["NR 强度"].value, 0.90);
assert.equal(widgets["时序稳定"].value, 0.55);
assert.equal(node.properties.star7_dlss_custom["NR 强度"], 0.90);

widgets["风格预设"].value = "真实人像优化";
widgets["风格预设"].callback("真实人像优化");
widgets["NR 强度"].value = 0.4;
widgets["重置参数"].callback();
assert.equal(widgets["NR 强度"].value, 0.85);
assert.equal(widgets["时序稳定"].value, 0.60);

widgets["风格预设"].value = "真实风格加强";
node.onConfigure({});
assert.equal(widgets["风格预设"].value, "真实风格");
assert.equal(widgets["NR 强度"].value, 0.90);

console.log("DLSS UI reset tests: PASS");
