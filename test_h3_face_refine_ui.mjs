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
const faceNodeData = {
    name: "MiniMaxH3FaceRefineLatentStar7",
    input: { required: {
        sampled_av_latent: ["LATENT", {}], refine_context: ["STAR7_H3_REFINE_CONTEXT", {}],
        face_lora: [["继承一采"], {}], face_lora_strength: ["FLOAT", {}],
    } },
    output_name: ["refined_av_latent", "report"],
};
await extension.beforeRegisterNodeDef(NodeType, faceNodeData);
assert.equal(faceNodeData.display_name, "MiniMax H3 一键人脸修复 - Star7");
assert.equal(faceNodeData.input.required.face_lora[1].display_name, "修脸 LoRA");
assert.equal(faceNodeData.input.required.face_lora_strength[1].display_name, "修脸 LoRA 强度");
assert.deepEqual(faceNodeData.output_name, ["采样结果", "运行报告"]);

const values = {
    enable_refine: true,
    face_detector: "face_yolov8m.pt",
    face_lora: "继承一采",
    face_lora_strength: 1.0,
    face_attention: "继承一采",
    face_count: 1,
    preset: "自动平衡",
    target_face: "主人物",
    refine_steps: 4,
    custom_strength: 0.30,
    custom_canvas: "自动",
    custom_crop_context: 3.0,
    custom_blend: 0.90,
    custom_feather: 20,
    seed: 0,
    preserve_repair_detail: true,
};
const node = Object.create(NodeType.prototype);
node.widgets = Object.entries(values).map(([name, value]) => ({ name, value, type: "number", options: {} }));
node.inputs = [{ name: "sampled_av_latent" }, { name: "refine_context" }];
node.outputs = [{ name: "refined_av_latent" }, { name: "report" }];
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
assert.equal(node.outputs[0].label, "采样结果");
assert.equal(node.outputs[1].label, "运行报告");
assert.equal(widgets.target_face.options.hidden, false);
assert.equal(node.size[0], 420);
assert.equal(node.size[1], 700);
assert.equal(widgets.preserve_repair_detail.value, true);
assert.equal(widgets.face_detector.label, "人脸检测模型");
assert.equal(widgets.face_lora.label, "修脸 LoRA");
assert.equal(widgets.face_lora_strength.label, "修脸 LoRA 强度");
assert.equal(widgets.face_attention.label, "修脸注意力");
widgets.preset.value = "远景小脸";
widgets.preset.callback();
assert.equal(widgets.custom_crop_context.value, 2.4);
assert.equal(widgets.custom_strength.value, 0.48);
assert.equal(widgets.custom_blend.value, 0.95);
widgets.preset.value = "自动平衡";
widgets.preset.callback();
assert.equal(widgets.custom_crop_context.value, 2.6);
assert.equal(widgets.custom_strength.value, 0.30);
assert.ok(node.widgets.some((item) => item.__star7FaceResolutionStatus));
const resolutionIndex = node.widgets.findIndex((item) => item.__star7FaceResolutionStatus);
const resetIndex = node.widgets.indexOf(node.__star7ResetButton);
assert.equal(resolutionIndex, resetIndex - 1);
node.comfyClass = "MiniMaxH3FaceRefineLatentStar7";
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
assert.equal(widgets.face_lora.value, "继承一采");
assert.equal(widgets.face_attention.value, "继承一采");
assert.equal(widgets.face_lora_strength.value, 1.0);
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

// The previous layout already had detector, LoRA and attention rows, but no
// strength row. Restore values by meaning so face count and presets do not shift.
node.onConfigure({ widgets_values: [
    true, "face_yolov8n.pt", "portrait.safetensors", "vsa_sm75", 2,
    "自定义", "画面中央", 5, 0.38, "768", 2.4, 0.91, 18, 456,
    "randomize", false,
] });
assert.equal(widgets.face_detector.value, "face_yolov8n.pt");
assert.equal(widgets.face_lora.value, "portrait.safetensors");
assert.equal(widgets.face_lora_strength.value, 1.0);
assert.equal(widgets.face_attention.value, "vsa_sm75");
assert.equal(widgets.face_count.value, 2);
assert.equal(widgets.preset.value, "自定义");
assert.equal(widgets.target_face.value, "画面中央");
assert.equal(widgets.preserve_repair_detail.value, false);

class LegacyFaceNodeType {}
await extension.beforeRegisterNodeDef(LegacyFaceNodeType, { name: "MiniMaxH3FaceRefineStar7" });
const legacyFace = Object.create(LegacyFaceNodeType.prototype);
legacyFace.widgets = Object.entries(values).map(([name, value]) => ({ name, value, type: "number", options: {} }));
legacyFace.inputs = [{ name: "sampled_av_latent" }, { name: "refine_context" }];
legacyFace.outputs = [{ name: "refined_images" }];
legacyFace.properties = {};
legacyFace.size = [420, 700];
legacyFace.computeSize = () => [390, 500];
legacyFace.setSize = (size) => { legacyFace.size = size; };
legacyFace.setDirtyCanvas = () => {};
legacyFace.addWidget = (type, name, value, callback, options) => {
    const item = { type, name, value, callback, options };
    legacyFace.widgets.push(item);
    return item;
};
legacyFace.onNodeCreated();
assert.equal(
    legacyFace.title,
    "MiniMax H3 一键人脸修复（旧工作流兼容）- Star7",
);
assert.equal(legacyFace.outputs[0].label, "图像");

console.log("H3 face repair UI tests: PASS");

class MaterialNodeType {}
await extension.beforeRegisterNodeDef(MaterialNodeType, { name: "MiniMaxH3MaterialPromptStar7" });
const material = Object.create(MaterialNodeType.prototype);
material.inputs = [
    { name: "drive_audio", link: null },
    { name: "last_frame", link: null },
    { name: "ref_video_0", link: 30 },
    { name: "ref_video_audio_0", link: null },
    { name: "ref_audio_0", link: null },
    ...Array.from({ length: 16 }, (_, index) => ({
        name: `ref_images.ref_image_${index}`,
        link: index < 3 ? index + 13 : null,
    })),
];
material.outputs = [{ name: "refine_context" }];
material.widgets = [];
material.size = [420, 500];
material.computeSize = () => [390, 500];
material.setSize = (size) => { material.size = size; };
material.setDirtyCanvas = () => {};
material.removeInput = (index) => { material.inputs.splice(index, 1); };
material.addInput = (name, type) => {
    const input = { name, type, link: null };
    material.inputs.push(input);
    return input;
};
material.id = 275;
app.graph.links = {
    13: { target_id: 275, target_slot: 5 },
    14: { target_id: 275, target_slot: 6 },
    15: { target_id: 275, target_slot: 7 },
    30: { target_id: 275, target_slot: 2 },
};
material.onNodeCreated();
assert.equal(material.inputs[0].label, "驱动音频");
assert.deepEqual(
    material.inputs.map((input) => input.name),
    [
        "drive_audio", "last_frame",
        "ref_images.ref_image_0", "ref_images.ref_image_1",
        "ref_images.ref_image_2", "ref_images.ref_image_3",
        "ref_video_0", "ref_video_audio_0", "ref_audio_0",
    ],
);
assert.equal(material.inputs[2].label, "参考图 1 - <Picture 1>");
assert.equal(material.inputs[5].label, "参考图 4");
assert.equal(material.outputs[0].label, "采样上下文");
assert.equal(app.graph.links[13].target_slot, 2);
assert.equal(app.graph.links[14].target_slot, 3);
assert.equal(app.graph.links[15].target_slot, 4);
assert.equal(app.graph.links[30].target_slot, 6);
material.inputs[0].link = 12;
material.onConnectionsChange();
assert.equal(material.inputs[0].label, "驱动音频 - <Audio D>");
material.inputs[0].link = null;
material.onConnectionsChange();
assert.equal(material.inputs[0].label, "驱动音频");
material.inputs[5].link = 16;
app.graph.links[16] = { target_id: 275, target_slot: 5 };
material.onConnectionsChange();
assert.deepEqual(
    material.inputs.slice(2, 7).map((input) => input.name),
    [
        "ref_images.ref_image_0", "ref_images.ref_image_1",
        "ref_images.ref_image_2", "ref_images.ref_image_3",
        "ref_images.ref_image_4",
    ],
);
assert.equal(material.inputs[5].label, "参考图 4 - <Picture 4>");
assert.equal(material.inputs[6].label, "参考图 5");
assert.equal(app.graph.links[16].target_slot, 5);
assert.equal(app.graph.links[30].target_slot, 7);

for (let slot = 4; slot < 9; slot += 1) {
    const input = material.inputs.find((item) => item.name === `ref_images.ref_image_${slot}`);
    assert.ok(input, `reference image ${slot + 1} must be available`);
    input.link = slot + 13;
    app.graph.links[slot + 13] = { target_id: 275, target_slot: material.inputs.indexOf(input) };
    material.onConnectionsChange();
}
assert.deepEqual(
    material.inputs.filter((input) => input.name.startsWith("ref_images.")).map((input) => input.name),
    Array.from({ length: 9 }, (_, slot) => `ref_images.ref_image_${slot}`),
);
assert.equal(app.graph.links[30].target_slot, 11);

const materialWithHole = Object.create(MaterialNodeType.prototype);
materialWithHole.id = 276;
materialWithHole.inputs = [
    { name: "last_frame", link: null },
    { name: "ref_images.ref_image_0", type: "IMAGE", link: 40 },
    { name: "ref_images.ref_image_1", type: "IMAGE", link: 41 },
    { name: "ref_images.ref_image_2", type: "IMAGE", link: 42 },
    { name: "ref_images.ref_image_3", type: "IMAGE", link: null },
    { name: "ref_images.ref_image_4", type: "IMAGE", link: 44 },
];
materialWithHole.outputs = [];
materialWithHole.widgets = [];
materialWithHole.size = [420, 500];
materialWithHole.computeSize = () => [390, 500];
materialWithHole.setSize = (size) => { materialWithHole.size = size; };
materialWithHole.setDirtyCanvas = () => {};
materialWithHole.removeInput = (index) => { materialWithHole.inputs.splice(index, 1); };
materialWithHole.addInput = (name, type) => {
    const input = { name, type, link: null };
    materialWithHole.inputs.push(input);
    return input;
};
Object.assign(app.graph.links, {
    40: { target_id: 276, target_slot: 1 },
    41: { target_id: 276, target_slot: 2 },
    42: { target_id: 276, target_slot: 3 },
    44: { target_id: 276, target_slot: 5 },
});
materialWithHole.onNodeCreated();
assert.deepEqual(
    materialWithHole.inputs.map((input) => [input.name, input.link]),
    [
        ["last_frame", null],
        ["ref_images.ref_image_0", 40],
        ["ref_images.ref_image_1", 41],
        ["ref_images.ref_image_2", 42],
        ["ref_images.ref_image_3", 44],
        ["ref_images.ref_image_4", null],
    ],
);
assert.equal(app.graph.links[44].target_slot, 4);

console.log("H3 conditioning dynamic audio label tests: PASS");
