import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let extension;
const app = { registerExtension(value) { extension = value; } };
const api = {
    listeners: {},
    addEventListener(name, callback) { this.listeners[name] = callback; },
};
const path = new URL("./web/h3_face_refine_star7.js", import.meta.url);
const source = fs.readFileSync(path, "utf8")
    .replace(
        'import { app } from "../../scripts/app.js";',
        "const app = globalThis.__star7TestApp;",
    )
    .replace(
        'import { api } from "../../scripts/api.js";',
        "const api = globalThis.__star7TestApi;",
    );
vm.runInNewContext(source, {
    __star7TestApp: app,
    __star7TestApi: api,
    navigator: { language: "zh-CN" },
    requestAnimationFrame(callback) { callback(); },
});
assert.ok(extension);

class NodeType {}
await extension.beforeRegisterNodeDef(NodeType, { name: "MiniMaxH3FaceRefineStar7" });

const values = {
    enable_refine: true,
    face_count: 1,
    preset: "自动平衡",
    target_face: "主人物",
    refine_steps: 4,
    custom_strength: 0.30,
    custom_canvas: "自动",
    custom_crop_context: 2.6,
    custom_blend: 0.90,
    custom_feather: 20,
    seed: 0,
    preserve_repair_detail: true,
};
const node = Object.create(NodeType.prototype);
node.widgets = Object.entries(values).map(([name, value]) => ({ name, value, type: "number", options: {} }));
node.inputs = [{ name: "sampled_av_latent" }, { name: "refine_context" }];
node.outputs = [{ name: "refined_images" }];
node.properties = {};
node.size = [420, 700];
node.computeSize = () => [390, 500];
node.setSize = (size) => { node.size = size; };
node.setDirtyCanvas = () => {};
node.addWidget = (type, name, value, callback, options) => {
    const item = { type, name, value, callback, options };
    node.widgets.push(item);
    return item;
};
node.onNodeCreated();

const widgets = Object.fromEntries(node.widgets.map((item) => [item.name, item]));
assert.equal(node.title, "MiniMax H3 一键人脸修复 - Star7");
assert.equal(node.inputs[0].label, "采样结果");
assert.equal(node.inputs[1].label, "采样上下文");
assert.equal(node.outputs[0].label, "图像");
assert.equal(widgets.target_face.options.hidden, false);
assert.equal(node.size[0], 420);
assert.equal(node.size[1], 700);
assert.equal(widgets.preserve_repair_detail.value, true);
assert.ok(node.widgets.some((item) => item.__star7FaceResolutionStatus));
const resolutionIndex = node.widgets.findIndex((item) => item.__star7FaceResolutionStatus);
const resetIndex = node.widgets.indexOf(node.__star7ResetButton);
assert.equal(resolutionIndex, resetIndex - 1);
node.comfyClass = "MiniMaxH3FaceRefineStar7";
app.graph = { getNodeById(id) { return String(id) === "124" ? node : null; } };
api.listeners["star7-h3-face-repair-resolution"]({
    detail: { node_id: "124", width: 768, height: 1376, megapixels: 1.01 },
});
assert.equal(
    node.widgets.find((item) => item.__star7FaceResolutionStatus).name,
    "修复后分辨率：768×1376 · 1.01 MP",
);
widgets.enable_refine.value = false;
widgets.enable_refine.callback();
assert.equal(
    node.widgets.find((item) => item.__star7FaceResolutionStatus).name,
    "未开启修复",
);
api.listeners["star7-h3-face-repair-resolution"]({
    detail: { node_id: "124", width: 608, height: 1056, megapixels: 0.61 },
});
assert.equal(
    node.widgets.find((item) => item.__star7FaceResolutionStatus).name,
    "未开启修复",
);
widgets.enable_refine.value = true;
widgets.enable_refine.callback();
assert.equal(
    node.widgets.find((item) => item.__star7FaceResolutionStatus).name,
    "修复后分辨率：运行后显示",
);
widgets.face_count.value = 3;
node.onConfigure({ widgets_values: [
    true, 3, "自动平衡", "画面中央", 4, 0.30, "自动", 2.6, 0.90, 20, 0,
] });
assert.equal(widgets.face_count.value, 3);
assert.equal(widgets.target_face.options.hidden, false);
assert.equal(widgets.target_face.type, "number");
widgets.face_count.value = 1;
node.onConfigure({ widgets_values: [
    true, 1, "自动平衡", "主人物", 4, 0.30, "自动", 2.6, 0.90, 20, 0,
] });
assert.equal(widgets.target_face.options.hidden, false);
assert.equal(widgets.target_face.type, "number");
assert.equal(widgets.target_face.computeSize, undefined);

// Preserve local workflows made with the short-lived switch + maximum layout.
node.onConfigure({ widgets_values: [
    true, "真人保真", "参考图匹配", 6, 0.25, "512", 2.8, 0.82, 24, 123,
    true, 4,
] });
assert.equal(widgets.face_count.value, 4);
assert.equal(widgets.preset.value, "真人保真");
assert.equal(widgets.target_face.value, "参考图匹配");
assert.equal(widgets.target_face.options.hidden, false);
assert.equal(widgets.refine_steps.value, 4); // preset applies its own saved values
assert.equal(node.size[0], 420);
assert.equal(node.size[1], 700);

console.log("H3 face repair UI tests: PASS");

class MaterialNodeType {}
await extension.beforeRegisterNodeDef(MaterialNodeType, { name: "MiniMaxH3MaterialPromptStar7" });
const material = Object.create(MaterialNodeType.prototype);
material.inputs = [
    { name: "drive_audio", link: null },
    { name: "ref_images.ref_image_0", link: null },
];
material.outputs = [{ name: "refine_context" }];
material.widgets = [];
material.size = [420, 500];
material.computeSize = () => [390, 500];
material.setSize = (size) => { material.size = size; };
material.setDirtyCanvas = () => {};
material.onNodeCreated();
assert.equal(material.inputs[0].label, "驱动音频");
assert.equal(material.inputs[1].label, "参考图 1");
assert.equal(material.outputs[0].label, "采样上下文");
material.inputs[0].link = 12;
material.onConnectionsChange();
assert.equal(material.inputs[0].label, "驱动音频 - <Audio D>");
material.inputs[0].link = null;
material.onConnectionsChange();
assert.equal(material.inputs[0].label, "驱动音频");
material.inputs[1].link = 13;
material.inputs.push({ name: "ref_images.ref_image_1", link: null });
material.onConnectionsChange();
assert.equal(material.inputs[1].label, "参考图 1 - <Picture 1>");
assert.equal(material.inputs[2].label, "参考图 2");
material.inputs[2].link = 14;
material.inputs.push({ name: "ref_images.ref_image_2", link: null });
material.onConnectionsChange();
assert.equal(material.inputs[2].label, "参考图 2 - <Picture 2>");
assert.equal(material.inputs[3].label, "参考图 3");

console.log("H3 conditioning dynamic audio label tests: PASS");
